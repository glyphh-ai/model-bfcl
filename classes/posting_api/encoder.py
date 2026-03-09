"""Glyphh Ada model for PostingAPI.

Full HDC encoder with domain-tuned structure for social/posting API.
Two layers: intent (action + target lexicons) + semantics (description + parameters BoW).
PostingAPI shares the same function signatures as TwitterAPI.

Exports:
    ENCODER_CONFIG      -- EncoderConfig with posting-tuned lexicons
    encode_query(q)     -- NL -> Concept dict
    encode_function(fd) -- function def -> Concept dict
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from glyphh.core.config import EncoderConfig, Layer, Role, Segment

# Import per-class intent
_DIR = Path(__file__).parent
import sys
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from intent import extract_intent, FUNC_TO_ACTION, FUNC_TO_TARGET

# -- ENCODER_CONFIG ---------------------------------------------------------
#
# Lexicons come from the posting pack canonical actions/targets.
# Only actions that map to actual PostingAPI functions are included.
#
# Two layers:
#   intent (0.55) -- categorical action + target matching (dominant for routing)
#   semantics (0.45) -- fuzzy BoW matching on description + parameters

ENCODER_CONFIG = EncoderConfig(
    dimension=2000,
    seed=42,
    apply_weights_during_encoding=False,
    include_temporal=False,
    layers=[
        Layer(
            name="intent",
            similarity_weight=0.55,
            segments=[
                Segment(
                    name="action",
                    roles=[
                        Role(
                            name="action",
                            similarity_weight=1.0,
                            # Pack canonical actions for PostingAPI functions
                            lexicons=[
                                "authenticate", "post_tweet", "comment",
                                "retweet", "mention", "follow", "unfollow",
                                "get_tweet", "get_tweet_comments",
                                "get_user_tweets", "get_user_stats",
                                "list_following", "search_tweets",
                                "get_login_status",
                                "none",
                            ],
                        ),
                        Role(
                            name="target",
                            similarity_weight=0.3,
                            # Pack canonical targets relevant to posting
                            lexicons=[
                                "tweet", "user", "account",
                                "none",
                            ],
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
                    name="text",
                    roles=[
                        Role(
                            name="description",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                        Role(
                            name="parameters",
                            similarity_weight=0.7,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# -- Text processing helpers ------------------------------------------------

_STOP_WORDS = frozenset({
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
    "let", "like", "any", "all", "some", "each", "every",
    "there", "here", "via", "per", "after", "before", "over",
})

_HOW_WORDS = frozenset({
    "tell", "give", "show", "help", "let", "know", "want", "need",
    "please", "can", "could", "would", "should", "like", "try",
    "using", "via", "through", "use",
})

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


def _bow(words: list[str]) -> str:
    unique = list(dict.fromkeys(words))
    return " ".join(unique) if unique else "none"


def _clean_desc(text: str) -> str:
    if "Tool description:" in text:
        text = re.sub(r"^.*?Tool description:\s*", "", text, flags=re.IGNORECASE | re.DOTALL)
    return _FILLER_PREFIX_RE.sub("", text).strip()


def _param_tokens(func_def: dict) -> list[str]:
    params = func_def.get("parameters", {})
    tokens: list[str] = []
    if not isinstance(params, dict):
        return tokens
    for pname, pdef in params.get("properties", {}).items():
        tokens.extend(_split_camel_snake(pname).split())
        if "description" in pdef:
            tokens.extend(_tokenize(pdef["description"]))
        if "enum" in pdef:
            for v in pdef["enum"][:8]:
                tokens.extend(_tokenize(str(v)))
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


# -- encode_query -----------------------------------------------------------

def encode_query(query: str) -> dict[str, Any]:
    """Encode a user query into Concept dict for HDC similarity.

    Uses posting-domain intent extraction for action/target (lexicon roles).
    Uses BoW for description/parameters (fuzzy matching roles).
    """
    intent = extract_intent(query)
    action = intent["action"]
    target = intent["target"]

    # Description: meaningful query tokens (minus how-words)
    all_tokens = _tokenize(query)
    desc_tokens = [t for t in all_tokens if t not in _HOW_WORDS]

    # Parameters: quoted strings + numbers + keywords
    param_tokens = list(all_tokens)
    param_tokens.extend(re.findall(r'"([^"]*)"', query))
    param_tokens.extend(re.findall(r"'([^']*)'", query))
    param_tokens.extend(re.findall(r"\b\d+\.?\d*\b", query))
    param_tokens = list(dict.fromkeys(param_tokens))

    return {
        "name": "query",
        "attributes": {
            "action": action,
            "target": target,
            "description": _bow(desc_tokens) if desc_tokens else _bow(all_tokens),
            "parameters": _bow(param_tokens) if param_tokens else _bow(all_tokens),
        },
    }


# -- encode_function --------------------------------------------------------

def encode_function(func_def: dict[str, Any]) -> dict[str, Any]:
    """Encode a function definition into Concept dict for HDC similarity.

    Maps function name to canonical action/target from the posting pack.
    """
    name = func_def.get("name", "unknown")
    bare = name.split(".")[-1] if "." in name else name
    desc = _clean_desc(func_def.get("description", ""))

    action = FUNC_TO_ACTION.get(bare, "none")
    target = FUNC_TO_TARGET.get(bare, "tweet")

    name_tokens = _tokenize(_split_camel_snake(bare))
    desc_tokens = _tokenize(desc)
    p_tokens = _param_tokens(func_def)

    return {
        "name": f"func_{name}",
        "attributes": {
            "action": action,
            "target": target,
            "description": _bow(desc_tokens + name_tokens),
            "parameters": _bow(p_tokens) if p_tokens else _bow(desc_tokens),
        },
    }
