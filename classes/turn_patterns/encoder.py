"""Glyphh Ada model for multi-turn patterns.

HDC encoder for matching NL queries to known turn patterns (ordered function
sequences). Three layers: intent (action + target + domain) + semantics
(description + parameters BoW) + structure (function sequence BoW).

Unlike single-function class encoders, the "functions" here are patterns like
"lockDoors|pressBrakePedal|startEngine". The encoder produces Concept dicts
that can be compared via HDC cosine similarity across all 9 API classes.

Exports:
    ENCODER_CONFIG      — EncoderConfig with cross-class lexicons
    encode_query(q)     → NL → Concept dict
    encode_function(fd) → exemplar/pattern def → Concept dict
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from glyphh.core.config import EncoderConfig, Layer, Role, Segment

# Import local intent module explicitly to avoid sys.path contamination
_DIR = Path(__file__).parent
import importlib.util as _ilu

_intent_spec = _ilu.spec_from_file_location("_turn_patterns_intent", _DIR / "intent.py")
_intent_mod = _ilu.module_from_spec(_intent_spec)
_intent_spec.loader.exec_module(_intent_mod)
extract_intent = _intent_mod.extract_intent

# ── ENCODER_CONFIG ────────────────────────────────────────────────────────
#
# Three layers:
#   intent (0.45) — categorical action + target + domain matching
#   semantics (0.35) — fuzzy BoW matching on description text
#   structure (0.20) — BoW of function names in the pattern (discriminates
#                       between patterns that share intent but differ in
#                       function composition)

# All unique actions from intent.py _PHRASE_MAP + _WORD_MAP
_ACTION_LEXICONS = [
    "start_vehicle", "fill_fuel", "check_tires", "find_tire_shop",
    "navigate", "estimate_distance", "convert_units",
    "lock_doors", "parking_brake", "cruise_control", "headlights",
    "climate_control", "vehicle_status", "check_weather",
    "book_flight", "check_flight_cost", "cancel_booking",
    "purchase_insurance", "contact_support",
    "check_stock", "place_order", "add_watchlist", "check_watchlist",
    "check_account", "convert_currency", "set_budget",
    "cancel_order", "check_order", "retrieve_invoice", "check_history",
    "send_message", "view_messages", "search_messages",
    "post_tweet", "auth_post", "retweet", "comment", "get_user",
    "create_ticket", "ticket_login", "resolve_ticket",
    "edit_ticket", "get_ticket",
    "list_files", "create_dir", "change_dir", "move_file", "copy_file",
    "delete_file", "read_file", "search_file", "count_file",
    "sort_file", "compare_files", "disk_usage", "head_file",
    "tail_file", "write_file",
    "compute", "convert_temp",
    "none",
]

_TARGET_LEXICONS = [
    "engine", "vehicle", "doors", "fuel", "tire", "brake", "speed",
    "lights", "temperature", "route", "distance",
    "flight", "booking", "trip", "insurance", "support",
    "stock", "order", "account", "watchlist", "invoice", "budget",
    "portfolio",
    "message", "tweet", "comment", "user",
    "file", "directory", "content",
    "ticket",
    "number", "result",
    "none",
]

_DOMAIN_LEXICONS = [
    "vehicle", "travel", "trading", "filesystem",
    "messaging", "posting", "ticket", "math",
    "none",
]

ENCODER_CONFIG = EncoderConfig(
    dimension=2000,
    seed=42,
    apply_weights_during_encoding=False,
    include_temporal=False,
    layers=[
        Layer(
            name="intent",
            similarity_weight=0.45,
            segments=[
                Segment(
                    name="action",
                    roles=[
                        Role(
                            name="action",
                            similarity_weight=1.0,
                            lexicons=_ACTION_LEXICONS,
                        ),
                        Role(
                            name="target",
                            similarity_weight=0.3,
                            lexicons=_TARGET_LEXICONS,
                        ),
                        Role(
                            name="domain",
                            similarity_weight=0.5,
                            lexicons=_DOMAIN_LEXICONS,
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="semantics",
            similarity_weight=0.35,
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
                            similarity_weight=0.5,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="structure",
            similarity_weight=0.20,
            segments=[
                Segment(
                    name="functions",
                    roles=[
                        Role(
                            name="function_names",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ── Text processing helpers ──────────────────────────────────────────────

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


# ── encode_query ─────────────────────────────────────────────────────────

def encode_query(query: str) -> dict[str, Any]:
    """Encode a user query into Concept dict for HDC pattern matching.

    Uses cross-class intent extraction for action/target/domain (lexicon roles).
    Uses BoW for description (fuzzy matching).
    Function names role is empty for queries (only populated for patterns).
    """
    intent = extract_intent(query)
    action = intent["action"]
    target = intent["target"]
    domain = intent["domain"]

    # Description: meaningful query tokens
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
            "domain": domain,
            "description": _bow(desc_tokens) if desc_tokens else _bow(all_tokens),
            "parameters": _bow(param_tokens) if param_tokens else "none",
            "function_names": "none",  # queries don't have function names
        },
    }


# ── encode_function ──────────────────────────────────────────────────────

def encode_function(exemplar: dict[str, Any]) -> dict[str, Any]:
    """Encode a pattern exemplar into Concept dict for HDC similarity.

    Exemplar schema (from exemplars.jsonl):
        {
            "function_name": "lockDoors|pressBrakePedal|startEngine",
            "action": "start_vehicle",
            "target": "engine",
            "domain": "vehicle",
            "description": "start car engine lock doors ...",
            "parameters_bow": "...",
            "function_name_bow": "lock doors press brake pedal start engine",
            "variant": 1
        }
    """
    action = exemplar.get("action", "none")
    target = exemplar.get("target", "none")
    domain = exemplar.get("domain", "none")
    description = exemplar.get("description", "none")
    parameters = exemplar.get("parameters_bow", "none")
    func_bow = exemplar.get("function_name_bow", "none")
    name = exemplar.get("function_name", "unknown")

    return {
        "name": f"pattern_{name}",
        "attributes": {
            "action": action,
            "target": target,
            "domain": domain,
            "description": description,
            "parameters": parameters,
            "function_names": func_bow,
        },
    }
