#!/usr/bin/env python3
"""
Build-time exemplar + test generation for BFCL multi-turn classes.

Reads func_doc JSONs + multi-turn ground truth data. For each API class:
  - Generates 3 weighted BoW exemplar variants per function
  - Mines real NL queries from multi-turn ground truth → per-function tests
  - Writes classes/{class}/exemplars.jsonl and tests.jsonl

Usage:
    python discover.py --all          # Generate all
    python discover.py --stats        # Show stats only
    python discover.py --class GorillaFileSystem  # Single class
"""

import argparse
import json
import re
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "bfcl"
FUNC_DOC_DIR = DATA_DIR / "multi_turn_func_doc"
CLASSES_DIR = ROOT / "classes"

# Class name → func_doc filename
CLASS_TO_FILE = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "TwitterAPI":        "posting_api.json",   # TwitterAPI uses posting_api func_doc
    "MessageAPI":        "message_api.json",
    "PostingAPI":        "posting_api.json",
    "TicketAPI":         "ticket_api.json",
    "MathAPI":           "math_api.json",
    "TradingBot":        "trading_bot.json",
    "TravelAPI":         "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
}

# Class name → folder name under classes/
CLASS_TO_FOLDER = {
    "GorillaFileSystem": "gorilla_file_system",
    "TwitterAPI":        "twitter_api",
    "MessageAPI":        "message_api",
    "PostingAPI":        "posting_api",
    "TicketAPI":         "ticket_api",
    "MathAPI":           "math_api",
    "TradingBot":        "trading_bot",
    "TravelAPI":         "travel_booking",
    "VehicleControlAPI": "vehicle_control",
}

# Multi-turn data files
MULTI_TURN_CATS = [
    "BFCL_v4_multi_turn_base.json",
    "BFCL_v4_multi_turn_miss_func.json",
    "BFCL_v4_multi_turn_miss_param.json",
    "BFCL_v4_multi_turn_long_context.json",
]

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

# Description boilerplate prefix (stripped before tokenizing)
_TOOL_DESC_RE = re.compile(
    r"^.*?Tool description:\s*",
    re.IGNORECASE | re.DOTALL,
)


def _clean_description(text: str) -> str:
    """Strip boilerplate from multi-turn func_doc descriptions."""
    if "Tool description:" in text:
        text = _TOOL_DESC_RE.sub("", text)
    return text.strip()


def _tokenize(text: str) -> list[str]:
    """Tokenize into meaningful lowercase words."""
    cleaned = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return [w for w in cleaned.split() if w not in _STOP_WORDS and len(w) > 1]


def _split_camel_snake(name: str) -> str:
    """Split camelCase and snake_case into space-separated words."""
    name = name.replace("_", " ").replace("-", " ").replace(".", " ")
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return name.lower().strip()


def _bow_value(words: list[str]) -> str:
    """Deduplicated space-separated BoW string."""
    unique = list(dict.fromkeys(words))
    return " ".join(unique) if unique else "none"


def _extract_param_tokens(func_def: dict) -> list[str]:
    """Extract parameter-level tokens from a function definition."""
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
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


# ---------------------------------------------------------------------------
# Action/target extraction (simplified from intent.py)
# ---------------------------------------------------------------------------

_ACTION_VERBS = {
    # Filesystem
    "cat": "read", "cd": "navigate", "cp": "copy", "diff": "compare",
    "du": "check", "echo": "write", "find": "find", "grep": "search",
    "ls": "list", "mkdir": "create", "mv": "move", "pwd": "check",
    "rm": "delete", "rmdir": "delete", "sort": "sort", "tail": "read",
    "touch": "create", "wc": "count",
    # Twitter/Posting
    "post_tweet": "send", "comment": "send", "retweet": "send",
    "mention": "send", "post": "send", "share": "send",
    # Message
    "send_message": "send", "view_messages_received": "get",
    "view_messages_sent": "get", "delete_message": "delete",
    "search_messages": "search", "get_message_stats": "get",
    "add_contact": "add", "get_user_id": "get",
    # Ticket
    "create_ticket": "create", "resolve_ticket": "update",
    "get_ticket": "get", "close_ticket": "update",
    "edit_ticket": "update",
    # Math
    "add": "calculate", "subtract": "calculate", "multiply": "calculate",
    "divide": "calculate", "square_root": "calculate", "power": "calculate",
    "absolute_value": "calculate", "logarithm": "calculate",
    "mean": "calculate", "percentage": "calculate", "min_value": "find",
    "max_value": "find", "standard_deviation": "calculate",
    "imperial_si_conversion": "convert", "si_unit_conversion": "convert",
    "round_number": "calculate", "gcd": "calculate", "lcm": "calculate",
    # Trading
    "add_to_watchlist": "add", "cancel_order": "delete",
    "filter_stocks_by_price": "filter", "fund_account": "update",
    "get_account_info": "get", "get_account_balance": "get",
    "get_available_stocks": "list", "get_current_time": "get",
    "get_order_details": "get", "get_order_history": "get",
    "get_stock_info": "get", "get_symbol_by_name": "get",
    "get_transaction_history": "get", "get_watchlist": "get",
    "make_transaction": "send", "notify_price_change": "send",
    "place_order": "create", "remove_stock_from_watchlist": "remove",
    "trading_get_login_status": "check", "trading_login": "start",
    "trading_logout": "stop",
    # Travel
    "authenticate_travel": "start", "book_flight": "create",
    "cancel_booking": "delete", "contact_customer_support": "send",
    "get_flight_cost": "get", "get_nearest_airport_by_city": "get",
    "list_all_airports": "list", "get_airline_info": "get",
    "retrieve_invoice": "get", "set_budget_limit": "set",
    "estimate_distance": "calculate", "get_budget_fiscal_year": "get",
    "compute_exchange_rate": "calculate",
    "verify_traveler_information": "check",
    "get_zipcode_based_on_city": "get",
    "register_credit_card": "create",
    "purchase_insurance": "create",
    "book_hotel": "create",
    # Vehicle
    "activateParkingBrake": "start", "adjustClimateControl": "set",
    "check_tire_pressure": "check", "displayCarStatus": "get",
    "display_log": "get", "estimate_drive_feasibility_by_mileage": "check",
    "estimate_headlight_lifetime": "calculate",
    "fillFuelTank": "update", "find_nearest_tire_shop": "find",
    "gallon_to_liter": "convert", "get_current_speed": "get",
    "getCurbWeight": "get", "getDashboardReading": "get",
    "getOutsideTemperatureFromDABCWeatherService": "get",
    "get_outside_temperature_from_google": "get",
    "liter_to_gallon": "convert", "lockDoors": "set",
    "pressBrakePedal": "start", "releaseBrakePedal": "stop",
    "setCruiseControl": "set", "setHeadlights": "set",
    "set_navigation": "set", "startEngine": "start",
    "unlockDoors": "set",
}

_TARGET_DEFAULTS = {
    # Filesystem
    "cat": "content", "cd": "directory", "cp": "file", "diff": "file",
    "du": "disk", "echo": "content", "find": "file", "grep": "content",
    "ls": "file", "mkdir": "directory", "mv": "file", "pwd": "directory",
    "rm": "file", "rmdir": "directory", "sort": "content", "tail": "content",
    "touch": "file", "wc": "content",
    # Twitter/Posting
    "post_tweet": "post", "comment": "post", "retweet": "post",
    "mention": "user", "post": "post", "share": "post",
    # Message
    "send_message": "message", "view_messages_received": "message",
    "view_messages_sent": "message", "delete_message": "message",
    "search_messages": "message", "get_message_stats": "message",
    "add_contact": "user", "get_user_id": "user",
    # Ticket
    "create_ticket": "ticket", "resolve_ticket": "ticket",
    "get_ticket": "ticket", "close_ticket": "ticket",
    "edit_ticket": "ticket",
    # Math
    "add": "number", "subtract": "number", "multiply": "number",
    "divide": "number", "square_root": "number", "power": "number",
    "absolute_value": "number", "logarithm": "number",
    "mean": "number", "percentage": "number", "min_value": "number",
    "max_value": "number", "standard_deviation": "number",
    "imperial_si_conversion": "number", "si_unit_conversion": "number",
    "round_number": "number", "gcd": "number", "lcm": "number",
    # Trading
    "add_to_watchlist": "stock", "cancel_order": "order",
    "filter_stocks_by_price": "stock", "fund_account": "balance",
    "get_account_info": "balance", "get_account_balance": "balance",
    "get_available_stocks": "stock", "get_current_time": "data",
    "get_order_details": "order", "get_order_history": "order",
    "get_stock_info": "stock", "get_symbol_by_name": "stock",
    "get_transaction_history": "order", "get_watchlist": "stock",
    "make_transaction": "order", "notify_price_change": "stock",
    "place_order": "order", "remove_stock_from_watchlist": "stock",
    "trading_get_login_status": "data", "trading_login": "data",
    "trading_logout": "data",
    # Travel
    "authenticate_travel": "data", "book_flight": "flight",
    "cancel_booking": "flight", "contact_customer_support": "message",
    "get_flight_cost": "flight", "get_nearest_airport_by_city": "flight",
    "list_all_airports": "flight", "get_airline_info": "flight",
    "retrieve_invoice": "order", "set_budget_limit": "balance",
    "estimate_distance": "route", "get_budget_fiscal_year": "balance",
    "compute_exchange_rate": "currency",
    "verify_traveler_information": "data",
    "get_zipcode_based_on_city": "data",
    "register_credit_card": "data",
    "purchase_insurance": "data",
    "book_hotel": "hotel",
    # Vehicle
    "activateParkingBrake": "vehicle", "adjustClimateControl": "vehicle",
    "check_tire_pressure": "vehicle", "displayCarStatus": "vehicle",
    "display_log": "vehicle", "estimate_drive_feasibility_by_mileage": "vehicle",
    "estimate_headlight_lifetime": "vehicle",
    "fillFuelTank": "fuel", "find_nearest_tire_shop": "vehicle",
    "gallon_to_liter": "fuel", "get_current_speed": "vehicle",
    "getCurbWeight": "vehicle", "getDashboardReading": "vehicle",
    "getOutsideTemperatureFromDABCWeatherService": "data",
    "get_outside_temperature_from_google": "data",
    "liter_to_gallon": "fuel", "lockDoors": "vehicle",
    "pressBrakePedal": "vehicle", "releaseBrakePedal": "vehicle",
    "setCruiseControl": "vehicle", "setHeadlights": "vehicle",
    "set_navigation": "route", "startEngine": "vehicle",
    "unlockDoors": "vehicle",
}


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------

def load_jsonl(path: Path) -> list[dict]:
    """Load a JSONL file into a list of dicts."""
    entries = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def load_func_defs(cls: str) -> list[dict]:
    """Load function definitions for a class from multi_turn_func_doc/."""
    fname = CLASS_TO_FILE.get(cls)
    if not fname:
        return []
    fpath = FUNC_DOC_DIR / fname
    if not fpath.exists():
        return []
    return load_jsonl(fpath)


# ---------------------------------------------------------------------------
# Mine queries from multi-turn ground truth
# ---------------------------------------------------------------------------

def mine_queries() -> dict[str, list[str]]:
    """Mine real NL queries mapped to function names from multi-turn data.

    Returns: {func_name: [query1, query2, ...]}
    where func_name is unprefixed (e.g. "cd", not "GorillaFileSystem.cd")
    """
    func_queries: dict[str, list[str]] = {}

    for cat_file in MULTI_TURN_CATS:
        data_path = DATA_DIR / cat_file
        pa_path = DATA_DIR / "possible_answer" / cat_file
        if not data_path.exists() or not pa_path.exists():
            continue

        with open(data_path) as df, open(pa_path) as pf:
            for data_line, pa_line in zip(df, pf):
                entry = json.loads(data_line)
                pa = json.loads(pa_line)

                turns = entry.get("question", [])
                gt_turns = pa.get("ground_truth", [])

                for turn_msgs, gt_calls in zip(turns, gt_turns):
                    if not turn_msgs or not gt_calls:
                        continue

                    # Extract user query
                    query = ""
                    for msg in reversed(turn_msgs):
                        if isinstance(msg, dict) and msg.get("role") == "user":
                            query = msg.get("content", "")
                            break
                    if not query:
                        continue

                    # Extract function names from ground truth
                    for call_str in gt_calls:
                        func_name = call_str.split("(")[0].strip()
                        func_queries.setdefault(func_name, []).append(query)

    return func_queries


# ---------------------------------------------------------------------------
# Generate exemplars
# ---------------------------------------------------------------------------

def generate_exemplars(cls: str, func_defs: list[dict],
                       mined_queries: dict[str, list[str]]) -> list[dict]:
    """Generate 3 weighted BoW exemplar variants per function.

    Follows pipedream pattern: each variant emphasizes different words
    so HDC cosine search matches diverse query phrasings.
    """
    folder = CLASS_TO_FOLDER.get(cls, cls.lower())

    # Compute shared class boilerplate tokens (appear in ALL descriptions)
    all_desc_tokens = []
    for fd in func_defs:
        desc = _clean_description(fd.get("description", ""))
        all_desc_tokens.append(set(_tokenize(desc)))

    # Shared tokens = intersection of all descriptions
    if all_desc_tokens:
        shared_tokens = set.intersection(*all_desc_tokens) if len(all_desc_tokens) > 1 else set()
    else:
        shared_tokens = set()

    exemplars = []

    for fd in func_defs:
        func_name = fd["name"]
        prefixed_name = f"{cls}.{func_name}"
        desc = _clean_description(fd.get("description", ""))
        desc_tokens = _tokenize(desc)
        param_tokens = _extract_param_tokens(fd)
        name_tokens = _tokenize(_split_camel_snake(func_name))

        # Action and target
        action = _ACTION_VERBS.get(func_name, "none")
        target = _TARGET_DEFAULTS.get(func_name, "none")

        # Differentiating words: desc tokens NOT in shared boilerplate
        diff_words = [t for t in desc_tokens if t not in shared_tokens]
        if not diff_words:
            diff_words = desc_tokens[:5]  # fallback

        # Also add mined query tokens as additional diff signal
        mined = mined_queries.get(func_name, [])
        mined_tokens = []
        for q in mined[:10]:  # sample up to 10 queries
            mined_tokens.extend(_tokenize(q))
        # Most frequent mined tokens (that aren't stop words or shared)
        from collections import Counter
        mined_freq = Counter(mined_tokens)
        top_mined = [w for w, _ in mined_freq.most_common(15)
                     if w not in shared_tokens and w not in _STOP_WORDS]

        # Parameter BoW
        param_bow = _bow_value(param_tokens)

        # Function name BoW (method name only, no class prefix)
        fn_bow = _bow_value(name_tokens)

        # Base dict
        base = {
            "class_name": cls,
            "function_name": prefixed_name,
            "raw_name": func_name,
            "action": action,
            "target": target,
            "function_name_bow": fn_bow,
            "parameters_bow": param_bow,
        }

        diff_str = " ".join(diff_words)
        mined_str = " ".join(top_mined[:8])
        class_clean = folder.replace("_", " ")

        # Exemplar 1: diff words emphasized (3x) + class name
        desc1 = f"{action} {diff_str} {diff_str} {diff_str} {class_clean}"
        exemplars.append({**base, "description": desc1, "variant": 1})

        # Exemplar 2: name tokens + all desc tokens + class name
        all_specific = " ".join(list(dict.fromkeys(name_tokens + diff_words)))
        desc2 = f"{action} {all_specific} {all_specific} {class_clean}"
        exemplars.append({**base, "description": desc2, "variant": 2})

        # Exemplar 3: mined query tokens (real NL phrasings) + diff words
        if top_mined:
            desc3 = f"{action} {mined_str} {mined_str} {diff_str} {class_clean}"
        else:
            # No mined data: use param tokens as backup diversity
            param_str = " ".join(param_tokens[:8])
            desc3 = f"{action} {param_str} {diff_str} {diff_str} {class_clean}"
        exemplars.append({**base, "description": desc3, "variant": 3})

    return exemplars


# ---------------------------------------------------------------------------
# Generate test queries
# ---------------------------------------------------------------------------

def generate_tests(cls: str, func_defs: list[dict],
                   mined_queries: dict[str, list[str]]) -> list[dict]:
    """Generate test queries per function from mined real NL data.

    Each test entry: {query, expected_function, class_name, test_type}
    """
    tests = []

    for fd in func_defs:
        func_name = fd["name"]
        prefixed_name = f"{cls}.{func_name}"
        mined = mined_queries.get(func_name, [])

        # Deduplicate
        seen = set()
        unique_queries = []
        for q in mined:
            q_norm = q.strip().lower()
            if q_norm not in seen:
                seen.add(q_norm)
                unique_queries.append(q.strip())

        # Keep up to 20 unique queries
        for q in unique_queries[:20]:
            tests.append({
                "query": q,
                "expected_function": prefixed_name,
                "class_name": cls,
                "test_type": "mined",
            })

    return tests


# ---------------------------------------------------------------------------
# Write outputs
# ---------------------------------------------------------------------------

def write_jsonl(path: Path, entries: list[dict]) -> None:
    """Write entries to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def generate_class(cls: str, mined_queries: dict[str, list[str]],
                   verbose: bool = False) -> dict:
    """Generate exemplars + tests for a single class."""
    folder = CLASS_TO_FOLDER.get(cls)
    if not folder:
        print(f"  Unknown class: {cls}")
        return {}

    func_defs = load_func_defs(cls)
    if not func_defs:
        print(f"  No func_defs for {cls}")
        return {}

    exemplars = generate_exemplars(cls, func_defs, mined_queries)
    tests = generate_tests(cls, func_defs, mined_queries)

    out_dir = CLASSES_DIR / folder
    write_jsonl(out_dir / "exemplars.jsonl", exemplars)
    write_jsonl(out_dir / "tests.jsonl", tests)

    stats = {
        "class": cls,
        "functions": len(func_defs),
        "exemplars": len(exemplars),
        "tests": len(tests),
    }

    if verbose:
        print(f"  {cls}: {len(func_defs)} funcs, {len(exemplars)} exemplars, "
              f"{len(tests)} tests")

    return stats


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate BFCL per-class exemplars")
    parser.add_argument("--all", action="store_true", help="Generate for all classes")
    parser.add_argument("--class", dest="cls", help="Generate for a specific class")
    parser.add_argument("--stats", action="store_true", help="Show stats only")
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    print("Mining queries from multi-turn data...")
    mined = mine_queries()
    total_mined = sum(len(v) for v in mined.values())
    print(f"  Mined {total_mined} query-function pairs across {len(mined)} functions")

    if args.stats:
        for func, queries in sorted(mined.items(), key=lambda x: -len(x[1]))[:20]:
            unique = len(set(q.strip().lower() for q in queries))
            print(f"  {func}: {len(queries)} total, {unique} unique")
        return

    classes = list(CLASS_TO_FILE.keys())
    if args.cls:
        classes = [args.cls]
    elif not args.all:
        parser.print_help()
        return

    print(f"\nGenerating exemplars for {len(classes)} classes...")
    all_stats = []
    for cls in classes:
        stats = generate_class(cls, mined, verbose=True)
        if stats:
            all_stats.append(stats)

    print(f"\nDone. Generated:")
    total_ex = sum(s["exemplars"] for s in all_stats)
    total_tests = sum(s["tests"] for s in all_stats)
    print(f"  {total_ex} exemplars across {len(all_stats)} classes")
    print(f"  {total_tests} test queries")


if __name__ == "__main__":
    main()
