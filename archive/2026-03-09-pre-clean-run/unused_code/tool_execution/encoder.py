"""Glyphh Ada — Tool Execution encoder.

3-layer HDC encoder for multi-turn tool execution:
  1. Intent (0.45)    — action + target lexicons (which function?)
  2. Semantics (0.30)  — description + keyword BoW (fuzzy matching)
  3. Arguments (0.25)  — parameter names + types + entity extraction (what args?)

This encoder works across ALL API classes simultaneously.
Functions from any class are encoded into the same GlyphSpace.

Exports:
    ENCODER_CONFIG       — EncoderConfig with 3-layer structure
    encode_query(q)      → NL → Concept dict
    encode_function(fd)  → function def → Concept dict
"""

from __future__ import annotations

import importlib.util
import re
from pathlib import Path
from typing import Any

from glyphh.core.config import EncoderConfig, Layer, Role, Segment

# Explicit import to avoid bfcl/intent.py shadowing
_DIR = Path(__file__).parent
_intent_spec = importlib.util.spec_from_file_location("_te_intent", _DIR / "intent.py")
_intent_mod = importlib.util.module_from_spec(_intent_spec)
_intent_spec.loader.exec_module(_intent_mod)
extract_intent = _intent_mod.extract_intent
extract_entities = _intent_mod.extract_entities

# ── Unified action lexicon (all 9 classes) ──────────────────────────────

_ALL_ACTIONS = [
    # gorilla_file_system
    "ls", "cd", "mkdir", "rm", "rmdir", "cp", "mv", "cat", "grep",
    "touch", "wc", "pwd", "find_files", "tail", "head", "echo", "diff", "sort", "du",
    # message_api
    "send_message", "delete_message", "add_contact", "view_messages_sent",
    "search_messages", "get_user_id", "list_users", "get_message_stats",
    "message_login", "message_get_login_status",
    # ticket_api
    "create_ticket", "close_ticket", "resolve_ticket", "edit_ticket",
    "get_ticket", "get_user_tickets", "ticket_login", "ticket_get_login_status", "logout",
    # twitter_api / posting_api
    "post_tweet", "retweet", "comment", "mention", "follow_user", "unfollow_user",
    "search_tweets", "get_tweet", "get_tweet_comments", "get_user_tweets",
    "get_user_stats", "list_all_following", "authenticate_twitter", "posting_get_login_status",
    # math_api
    "logarithm", "mean", "standard_deviation", "square_root",
    "add", "subtract", "multiply", "divide", "power", "percentage",
    "absolute_value", "round_number", "max_value", "min_value", "sum_values",
    "imperial_si_conversion", "si_unit_conversion",
    # trading_bot
    "add_to_watchlist", "cancel_order", "place_order", "get_stock_info",
    "get_account_info", "fund_account", "withdraw_funds",
    "remove_stock_from_watchlist", "get_order_history",
    "trading_login", "trading_logout", "trading_get_login_status",
    "get_available_stocks", "filter_stocks_by_price", "get_order_details",
    "get_symbol_by_name", "get_transaction_history", "get_watchlist",
    "make_transaction", "notify_price_change", "update_stock_price",
    # travel_booking
    "book_flight", "cancel_booking", "compute_exchange_rate",
    "contact_customer_support", "get_flight_cost", "get_nearest_airport_by_city",
    "list_all_airports", "get_credit_card_balance", "register_credit_card",
    "retrieve_invoice", "purchase_insurance", "authenticate_travel",
    "get_all_credit_cards", "get_booking_history", "get_budget_fiscal_year",
    "set_budget_limit", "travel_get_login_status", "verify_traveler_information",
    # vehicle_control
    "activateParkingBrake", "adjustClimateControl", "check_tire_pressure",
    "displayCarStatus", "fillFuelTank", "find_nearest_tire_shop",
    "gallon_to_liter", "get_current_speed", "lockDoors", "startEngine",
    "setCruiseControl", "setHeadlights", "set_navigation",
    "pressBrakePedal", "releaseBrakePedal", "liter_to_gallon",
    "estimate_distance", "estimate_drive_feasibility_by_mileage",
    "get_outside_temperature_from_google", "get_outside_temperature_from_weather_com",
    # general
    "none",
]

_ALL_TARGETS = [
    # filesystem
    "directory", "file", "content", "stdout",
    # messaging
    "message", "contact", "user",
    # ticketing
    "ticket",
    # social
    "tweet", "comment_obj", "follower",
    # math
    "number", "value", "unit",
    # trading
    "stock", "order", "account", "watchlist", "transaction",
    # travel
    "flight", "booking", "card", "airport", "budget", "invoice", "insurance", "currency", "traveler", "support",
    # vehicle
    "engine", "brake", "door", "climate", "fuel", "tire", "speed", "headlight", "navigation", "status",
    # general
    "session", "none",
]


# ── ENCODER_CONFIG ──────────────────────────────────────────────────────

ENCODER_CONFIG = EncoderConfig(
    dimension=2000,
    seed=42,
    apply_weights_during_encoding=False,
    include_temporal=False,
    layers=[
        # Layer 1: Intent — which function to call
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
                            lexicons=_ALL_ACTIONS,
                        ),
                        Role(
                            name="target",
                            similarity_weight=0.3,
                            lexicons=_ALL_TARGETS,
                        ),
                    ],
                ),
            ],
        ),
        # Layer 2: Semantics — fuzzy description/keyword matching
        Layer(
            name="semantics",
            similarity_weight=0.30,
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
                            name="keywords",
                            similarity_weight=0.5,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
        # Layer 3: Arguments — parameter structure matching
        Layer(
            name="arguments",
            similarity_weight=0.25,
            segments=[
                Segment(
                    name="params",
                    roles=[
                        Role(
                            name="param_names",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                        Role(
                            name="param_types",
                            similarity_weight=0.3,
                            text_encoding="bag_of_words",
                        ),
                        Role(
                            name="entities",
                            similarity_weight=0.8,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ── Text processing ─────────────────────────────────────────────────────

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
    r"utility function|this (is a|provides?)|function to|this tool belongs to .*?\. )[,\s]*",
    re.IGNORECASE,
)

_TOOL_DESC_RE = re.compile(r"Tool description:\s*", re.IGNORECASE)


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
        text = _TOOL_DESC_RE.split(text)[-1]
    return _FILLER_PREFIX_RE.sub("", text).strip()


# ── encode_query ─────────────────────────────────────────────────────────

def encode_query(query: str) -> dict[str, Any]:
    """Encode a user query into Concept dict for the 3-layer model."""
    intent = extract_intent(query)
    entities = intent["entities"]
    keywords = intent["keywords"]

    desc_tokens = [t for t in keywords if t not in _HOW_WORDS]

    return {
        "name": "query",
        "attributes": {
            # Intent layer
            "action": intent["action"],
            "target": intent["target"],
            # Semantics layer
            "description": _bow(desc_tokens) if desc_tokens else "none",
            "keywords": _bow(keywords) if keywords else "none",
            # Arguments layer
            "param_names": "none",
            "param_types": "none",
            "entities": _bow(entities) if entities else "none",
        },
    }


# ── encode_function ──────────────────────────────────────────────────────

def encode_function(func_def: dict[str, Any]) -> dict[str, Any]:
    """Encode a function definition into Concept dict for the 3-layer model."""
    name = func_def.get("name", "unknown")
    bare = name.split(".")[-1] if "." in name else name
    desc = _clean_desc(func_def.get("description", ""))

    # Intent layer
    action = bare  # function name IS the action
    name_tokens = _split_camel_snake(bare).split()
    target = "none"
    desc_lower = desc.lower()
    for t in ["file", "directory", "message", "ticket", "tweet", "stock", "order",
              "flight", "booking", "account", "engine", "door", "brake", "tire"]:
        if t in desc_lower or t in bare.lower():
            target = t
            break

    # Semantics layer
    desc_tokens = _tokenize(desc)
    keyword_tokens = _tokenize(_split_camel_snake(bare)) + desc_tokens[:10]

    # Arguments layer
    params = func_def.get("parameters", {})
    param_names = []
    param_types = []
    param_entities = []
    if isinstance(params, dict):
        for pname, pdef in params.get("properties", {}).items():
            param_names.extend(_split_camel_snake(pname).split())
            ptype = pdef.get("type", "string")
            param_types.append(ptype)
            if "description" in pdef:
                param_entities.extend(_tokenize(pdef["description"])[:5])
            if "enum" in pdef:
                for v in pdef["enum"][:8]:
                    param_entities.append(str(v).lower())
            if "default" in pdef and pdef["default"] is not None:
                param_entities.append(str(pdef["default"]).lower())

    return {
        "name": f"func_{name}",
        "attributes": {
            # Intent layer
            "action": action,
            "target": target,
            # Semantics layer
            "description": _bow(desc_tokens + name_tokens),
            "keywords": _bow(keyword_tokens),
            # Arguments layer
            "param_names": _bow(param_names) if param_names else "none",
            "param_types": _bow(param_types) if param_types else "none",
            "entities": _bow(param_entities) if param_entities else "none",
        },
    }
