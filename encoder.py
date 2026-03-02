"""
Encoder for the BFCL Function Caller model.

Two-layer HDC encoder: signature (action + target + function_name) and
semantics (description + parameters). Functions and queries are both encoded
into the same space so GQL similarity finds the correct function.

Exports:
  ENCODER_CONFIG       — EncoderConfig for the BFCL model
  encode_function(func_def) — JSON Schema function → Concept dict
  encode_query(query)       — NL query → Concept dict
"""

import re

from glyphh.core.config import EncoderConfig, Layer, Role, Segment
from intent import extract_action, extract_target, extract_intent

# ---------------------------------------------------------------------------
# ENCODER_CONFIG
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
                                "get", "set", "create", "update", "delete",
                                "find", "search", "list", "calculate", "convert",
                                "check", "validate", "send", "add", "remove",
                                "fetch", "compute", "sort", "filter", "count",
                                "parse", "format", "generate", "build", "make",
                                "open", "close", "start", "stop", "run",
                                "read", "write", "load", "save", "download",
                                "upload", "move", "copy", "rename", "merge",
                                "split", "join", "compare", "match", "replace",
                                "extract", "transform", "encode", "decode",
                                "encrypt", "decrypt", "compress", "decompress",
                                "none",
                            ],
                        ),
                        Role(
                            name="target",
                            similarity_weight=0.7,
                            lexicons=[
                                # Math — shapes
                                "triangle", "circle", "rectangle", "square",
                                "polygon", "shape",
                                # Math — measurements
                                "area", "circumference", "perimeter",
                                "hypotenuse", "angle", "distance",
                                # Math — algebra / number theory
                                "equation", "roots", "prime", "factorial",
                                "sum", "number",
                                # Math — sequences
                                "fibonacci", "sequence",
                                # Math — calculus / linear algebra / stats
                                "derivative", "integral", "matrix", "vector",
                                "mean", "probability", "standard", "grade",
                                # Physics
                                "velocity", "force", "energy", "temperature",
                                # Physics — particles
                                "particle", "charge", "diameter", "mass",
                                # Health / fitness
                                "hydration", "calorie", "steps",
                                # Filesystem
                                "file", "directory", "path", "content", "disk",
                                "permission",
                                # Finance
                                "stock", "price", "trade", "order", "portfolio",
                                "balance", "option", "growth", "ratio", "revenue",
                                "market", "currency", "depreciation", "mortgage",
                                # Travel / vehicle
                                "flight", "hotel", "trip", "route", "vehicle",
                                "fuel", "gear",
                                # Communication
                                "message", "channel", "notification", "post",
                                "user",
                                # Geography
                                "capital", "city", "country", "population",
                                # Places / Nature
                                "restaurant", "park", "mountain", "tourist",
                                # Weather / misc
                                "weather", "ticket",
                                # Events / History
                                "event",
                                # Geology
                                "era",
                                # Science / Biology
                                "genotype", "discoverer",
                                # Statistics
                                "statistics",
                                # Safety / Crime
                                "crime",
                                # Arts / museum (sculpture maps to artwork)
                                "artwork", "landmark", "museum",
                                # Entertainment / games
                                "game", "book",
                                # Recipes / Cooking
                                "recipe",
                                # Players / sports
                                "player", "team", "sport", "goal",
                                # Legal
                                "case",
                                # Data structures
                                "string", "list", "record", "data", "key",
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
# BoW encoding helpers
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

# Description filler prefixes — stripped to keep signal clean
_FILLER_PREFIX_RE = re.compile(
    r"^(this function|a function that|use this (function )?to|this method|"
    r"this api( call)?|returns? (the|a|an) |the function|helper function|"
    r"utility function|this (is a|provides?)|function to)[,\s]*",
    re.IGNORECASE,
)

# HOW-words filtered from query description role (keep WHAT, not HOW)
_HOW_WORDS = {
    "tell", "give", "show", "help", "let", "know", "want", "need",
    "please", "can", "could", "would", "should", "like", "try",
    "using", "via", "through", "use",
}


def _split_camel_snake(name: str) -> str:
    """Split camelCase and snake_case into space-separated words."""
    name = name.replace("_", " ").replace("-", " ").replace(".", " ")
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return name.lower().strip()


def _tokenize(text: str) -> list[str]:
    """Tokenize into meaningful lowercase words, stripping stop words."""
    cleaned = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return [w for w in cleaned.split() if w not in _STOP_WORDS and len(w) > 1]


def _bow_value(words: list[str]) -> str:
    """Build a deduplicated space-separated BoW string for a role attribute."""
    unique = list(dict.fromkeys(words))  # dedupe, preserve order
    return " ".join(unique) if unique else "none"


def _clean_description(text: str) -> str:
    """Strip boilerplate filler phrases from function descriptions."""
    return _FILLER_PREFIX_RE.sub("", text).strip()


def _extract_param_tokens(func_def: dict) -> list[str]:
    """Extract parameter-level tokens from a function definition schema."""
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


def _extract_query_param_tokens(query: str) -> list[str]:
    """Extract parameter-relevant tokens from a query.

    Prioritizes: quoted strings, numbers, specific nouns.
    Falls back to all tokenized words.
    """
    tokens: list[str] = []
    tokens.extend(re.findall(r'"([^"]*)"', query))
    tokens.extend(re.findall(r"'([^']*)'", query))
    tokens.extend(re.findall(r"\b\d+\.?\d*\b", query))
    tokens.extend(_tokenize(query))
    return list(dict.fromkeys(tokens))


# ---------------------------------------------------------------------------
# encode_function — BFCL function definition → Concept dict
# ---------------------------------------------------------------------------

def encode_function(func_def: dict) -> dict:
    """Convert a BFCL function definition into a Concept-compatible dict.

    action + target come from intent extraction on the function name +
    first 100 chars of description. function_name, description, and
    parameters use bag-of-words encoding.
    """
    name = func_def.get("name", "unknown")
    description = _clean_description(func_def.get("description", ""))
    name_words = _split_camel_snake(name)

    # For dotted names like "games.update.find" or "market_performance.get_data":
    #   - Action: last segment has highest priority (primary verb, e.g. "find")
    #   - Target: exclude last segment to avoid generic words like "data" winning
    #     (e.g. "market_performance.get_data" → target from "market performance")
    parts = name.split(".")
    last_seg = _split_camel_snake(parts[-1]) if len(parts) > 1 else ""

    action_text = (last_seg + " " + name_words + " " + description[:100]).strip()
    action = extract_action(action_text)

    non_action_name = (
        _split_camel_snake(".".join(parts[:-1])) if len(parts) > 1 else name_words
    )
    target_text = (non_action_name + " " + description[:100]).strip()
    target = extract_target(target_text)

    keywords = extract_intent(name_words + " " + description[:200])["keywords"]
    intent = {"action": action, "target": target, "keywords": keywords}

    name_tokens  = _tokenize(name_words)
    desc_tokens  = _tokenize(description)
    param_tokens = _extract_param_tokens(func_def)

    return {
        "name": f"func_{name}",
        "attributes": {
            "action":        intent["action"],
            "target":        intent["target"],
            "function_name": _bow_value(name_tokens),
            "description":   _bow_value(desc_tokens),
            "parameters":    _bow_value(param_tokens),
        },
    }


# ---------------------------------------------------------------------------
# encode_query — NL query → Concept dict
# ---------------------------------------------------------------------------

def encode_query(query: str) -> dict:
    """Convert a user query into a Concept-compatible dict.

    - action + target: from intent extraction
    - function_name:   ALL query tokens — wide net to match function name words
    - description:     WHAT tokens only (HOW/framing words removed)
    - parameters:      numbers, quoted values, specific nouns
    """
    intent       = extract_intent(query)
    all_tokens   = _tokenize(query)
    what_tokens  = [t for t in all_tokens if t not in _HOW_WORDS]
    param_tokens = _extract_query_param_tokens(query)

    return {
        "name": "query",
        "attributes": {
            "action":        intent["action"],
            "target":        intent["target"],
            "function_name": _bow_value(all_tokens),
            "description":   _bow_value(what_tokens) if what_tokens else _bow_value(all_tokens),
            "parameters":    _bow_value(param_tokens) if param_tokens else _bow_value(all_tokens),
        },
    }
