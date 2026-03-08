"""
Per-class encoder for TravelAPI.

Two-stage routing (pipedream pattern):
  Stage 1: extract_api_class() detects TravelAPI (in main intent.py)
  Stage 2: This encoder + per-class exemplars route within the 22 travel functions

Exports:
  ENCODER_CONFIG    — EncoderConfig with travel-tuned lexicons
  encode_query(q)   — NL → Concept dict for HDC similarity
  entry_to_record(e) — exemplar entry → build record
  extract_action(q)  — extract travel-domain action verb
  extract_target(q)  — extract travel-domain target noun
"""

import re
from pathlib import Path

from glyphh.core.config import EncoderConfig, Layer, Role, Segment

# Import per-class intent maps
_INTENT_DIR = Path(__file__).parent
import sys
if str(_INTENT_DIR) not in sys.path:
    sys.path.insert(0, str(_INTENT_DIR))

from intent import (
    ACTION_SYNONYMS,
    TARGET_OVERRIDES,
    FUNCTION_INTENTS,
    CLASS_DOMAIN,
)

# ---------------------------------------------------------------------------
# ENCODER_CONFIG — Travel-tuned two-layer HDC
# ---------------------------------------------------------------------------

ENCODER_CONFIG = EncoderConfig(
    dimension=10000,
    seed=42,
    apply_weights_during_encoding=False,
    include_temporal=False,
    layers=[
        Layer(
            name="signature",
            similarity_weight=0.55,
            segments=[
                Segment(
                    name="identity",
                    roles=[
                        Role(
                            name="action",
                            similarity_weight=1.0,
                            lexicons=[
                                "get", "set", "create", "delete", "list",
                                "calculate", "send", "check", "start",
                                "none",
                            ],
                        ),
                        Role(
                            name="target",
                            similarity_weight=0.7,
                            lexicons=[
                                "flight", "hotel", "route", "currency",
                                "balance", "order", "data", "message",
                                "none",
                            ],
                        ),
                        Role(
                            name="function_name",
                            similarity_weight=0.9,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="semantics",
            similarity_weight=0.45,
            segments=[
                Segment(
                    name="context",
                    roles=[
                        Role(
                            name="description",
                            similarity_weight=0.8,
                            text_encoding="bag_of_words",
                        ),
                        Role(
                            name="parameters",
                            similarity_weight=0.6,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
    ],
)

# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "the", "a", "an", "to", "for", "on", "in", "is", "it", "i",
    "do", "can", "please", "now", "up", "my", "our", "me", "we",
    "and", "or", "of", "with", "from", "this", "that", "about",
    "how", "what", "when", "where", "which", "who", "also",
    "then", "but", "just", "them", "their", "its", "be", "been",
    "have", "has", "had", "not", "dont", "want", "think",
    "given", "using", "by", "if", "so", "as", "at", "into",
    "are", "was", "were", "will", "would", "could", "should",
    "may", "might", "shall", "need", "does", "did", "am",
    "you", "your", "he", "she", "they", "us", "his", "her",
}

_HOW_WORDS = {
    "tell", "give", "show", "help", "let", "know", "want", "need",
    "please", "can", "could", "would", "should", "like", "try",
    "using", "via", "through", "use",
}

_TOOL_DESC_RE = re.compile(
    r"^.*?Tool description:\s*",
    re.IGNORECASE | re.DOTALL,
)

_FILLER_PREFIX_RE = re.compile(
    r"^(this function|a function that|use this (function )?to|this method|"
    r"this api( call)?|returns? (the|a|an) |the function|helper function|"
    r"utility function|this (is a|provides?)|function to)[,\s]*",
    re.IGNORECASE,
)


def _split_camel_snake(name: str) -> str:
    name = name.replace("_", " ").replace("-", " ").replace(".", " ")
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return name.lower().strip()


def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return [w for w in cleaned.split() if w not in _STOP_WORDS and len(w) > 1]


def _bow_value(words: list[str]) -> str:
    unique = list(dict.fromkeys(words))
    return " ".join(unique) if unique else "none"


def _clean_description(text: str) -> str:
    if "Tool description:" in text:
        text = _TOOL_DESC_RE.sub("", text)
    return _FILLER_PREFIX_RE.sub("", text).strip()


def _extract_param_tokens(func_def: dict) -> list[str]:
    params = func_def.get("parameters", {})
    tokens = []
    if not isinstance(params, dict):
        return tokens
    props = params.get("properties", {})
    for pname, pdef in props.items():
        tokens.extend(_split_camel_snake(pname).split())
        if "enum" in pdef:
            for v in pdef["enum"][:8]:
                tokens.extend(_tokenize(str(v)))
        if "description" in pdef:
            tokens.extend(_tokenize(pdef["description"]))
        ptype = pdef.get("type", "")
        if ptype and ptype not in ("object", "dict", "string", "str"):
            tokens.append(ptype)
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# Intent extraction — travel-domain tuned
# ---------------------------------------------------------------------------

# Build reverse lookup: sorted by length (longest match first)
_ACTION_KEYS = sorted(ACTION_SYNONYMS.keys(), key=len, reverse=True)
_TARGET_KEYS = sorted(TARGET_OVERRIDES.keys(), key=len, reverse=True)


def extract_action(text: str) -> str:
    """Extract travel-domain action from NL text."""
    text_lower = text.lower()

    # Multi-word phrase match first (longest match wins)
    for phrase in _ACTION_KEYS:
        if " " in phrase and phrase in text_lower:
            func = ACTION_SYNONYMS[phrase]
            return FUNCTION_INTENTS.get(func, ("none",))[0]

    # Single-word match
    tokens = text_lower.split()
    for tok in tokens:
        if tok in ACTION_SYNONYMS:
            func = ACTION_SYNONYMS[tok]
            return FUNCTION_INTENTS.get(func, ("none",))[0]

    # Fallback: check function name tokens
    for tok in tokens:
        for func_name, (action, _) in FUNCTION_INTENTS.items():
            if tok in _split_camel_snake(func_name).split():
                return action

    return "none"


def extract_target(text: str) -> str:
    """Extract travel-domain target from NL text."""
    text_lower = text.lower()

    # Multi-word phrase match first
    for phrase in _TARGET_KEYS:
        if " " in phrase and phrase in text_lower:
            return TARGET_OVERRIDES[phrase]

    # Single-word match
    tokens = text_lower.split()
    for tok in tokens:
        if tok in TARGET_OVERRIDES:
            return TARGET_OVERRIDES[tok]

    return "none"


def extract_keywords(text: str) -> str:
    """Extract meaningful keywords from query text."""
    tokens = _tokenize(text)
    return _bow_value(tokens)


# ---------------------------------------------------------------------------
# encode_query — NL text → Concept dict
# ---------------------------------------------------------------------------

def encode_query(query: str) -> dict:
    """Convert a user query into a Concept dict for HDC similarity.

    Uses travel-domain intent extraction for action/target.
    Returns dict compatible with ENCODER_CONFIG roles.
    """
    action = extract_action(query)
    target = extract_target(query)
    all_tokens = _tokenize(query)
    what_tokens = [t for t in all_tokens if t not in _HOW_WORDS]

    # Inject function name fragments from ACTION_SYNONYMS matches
    query_lower = query.lower()
    fn_tokens = list(all_tokens)
    for phrase in _ACTION_KEYS:
        if phrase in query_lower:
            func_name = ACTION_SYNONYMS[phrase]
            fn_tokens.extend(_split_camel_snake(func_name).split())
            break

    param_tokens = []
    param_tokens.extend(re.findall(r'"([^"]*)"', query))
    param_tokens.extend(re.findall(r"'([^']*)'", query))
    param_tokens.extend(re.findall(r"\b\d+\.?\d*\b", query))
    param_tokens.extend(_tokenize(query))
    param_tokens = list(dict.fromkeys(param_tokens))

    return {
        "name": "query",
        "attributes": {
            "action": action,
            "target": target,
            "function_name": _bow_value(fn_tokens),
            "description": _bow_value(what_tokens) if what_tokens else _bow_value(all_tokens),
            "parameters": _bow_value(param_tokens) if param_tokens else _bow_value(all_tokens),
        },
    }


# ---------------------------------------------------------------------------
# encode_function — function def → Concept dict
# ---------------------------------------------------------------------------

def encode_function(func_def: dict) -> dict:
    """Convert a BFCL function definition into a Concept dict."""
    name = func_def.get("name", "unknown")
    # Strip class prefix if present
    bare_name = name.split(".")[-1] if "." in name else name
    description = _clean_description(func_def.get("description", ""))

    # Use per-function intent mapping
    intent = FUNCTION_INTENTS.get(bare_name, ("none", "none"))
    action, target = intent

    name_tokens = _tokenize(_split_camel_snake(bare_name))
    desc_tokens = _tokenize(description)
    param_tokens = _extract_param_tokens(func_def)

    return {
        "name": f"func_{name}",
        "attributes": {
            "action": action,
            "target": target,
            "function_name": _bow_value(name_tokens),
            "description": _bow_value(desc_tokens),
            "parameters": _bow_value(param_tokens),
        },
    }


# ---------------------------------------------------------------------------
# entry_to_record — exemplar entry → build record (pipedream pattern)
# ---------------------------------------------------------------------------

def entry_to_record(entry: dict) -> dict:
    """Convert an exemplars.jsonl entry to a Glyphh build record."""
    return {
        "concept_text": entry.get("function_name", "unknown"),
        "attributes": {
            "action": entry.get("action", "none"),
            "target": entry.get("target", "none"),
            "function_name": entry.get("function_name_bow", "none"),
            "description": entry.get("description", "none"),
            "parameters": entry.get("parameters_bow", "none"),
        },
        "metadata": {
            "class_name": entry.get("class_name", ""),
            "function_name": entry.get("function_name", ""),
            "raw_name": entry.get("raw_name", ""),
            "variant": entry.get("variant", 1),
        },
    }


def encode_exemplar(entry: dict) -> dict:
    """Convert an exemplars.jsonl entry to a Concept dict for GlyphSpace."""
    return {
        "name": f"func_{entry['function_name']}",
        "attributes": {
            "action": entry.get("action", "none"),
            "target": entry.get("target", "none"),
            "function_name": entry.get("function_name_bow", "none"),
            "description": entry.get("description", "none"),
            "parameters": entry.get("parameters_bow", "none"),
        },
    }
