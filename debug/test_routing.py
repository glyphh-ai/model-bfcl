"""Glyphh Ada 1.0 — Unified routing tests for all 9 API classes.

Uses BFCLModelScorer loading pre-encoded glyphs from pgvector.
Requires: docker compose up -d && python load_db.py

Run:
    cd glyphh-models
    PYTHONPATH=../glyphh-runtime python -m pytest bfcl/test_routing.py -v
"""

from __future__ import annotations

from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent))

from scorer import BFCLModelScorer, _load_class_module, _CLASS_DIR_MAP, _DIR_TO_CLASS

_BFCL_DIR = Path(__file__).parent

# ── Per-class scorer fixtures (load from DB = fast) ──────────────────

_scorers: dict[str, BFCLModelScorer] = {}


def _get_scorer(class_dir: str) -> BFCLModelScorer:
    """Get or create a cached scorer for a class, loading glyphs from pgvector."""
    if class_dir not in _scorers:
        scorer = BFCLModelScorer()
        scorer.configure_from_db(class_dir)
        _scorers[class_dir] = scorer
    return _scorers[class_dir]


# ── Build test cases from per-class tests.jsonl ─────────────────────

def _collect_routing_tests() -> list[tuple[str, str, str]]:
    """Collect (class_dir, query, expected_func) from each class's tests.jsonl."""
    cases = []
    for class_dir in _CLASS_DIR_MAP.values():
        tests_path = _BFCL_DIR / "classes" / class_dir / "tests.jsonl"
        if not tests_path.exists():
            continue
        with open(tests_path) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                query = entry.get("query", "")
                expected = entry.get("expected_function", "")
                if query and expected:
                    cases.append((class_dir, query, expected))
    return cases


def _collect_intent_tests() -> list[tuple[str, str, str]]:
    """Collect intent extraction test cases from per-class intent.py."""
    cases = []
    for class_dir in _CLASS_DIR_MAP.values():
        intent_mod = _load_class_module(class_dir, "intent")
        action_to_func = getattr(intent_mod, "ACTION_TO_FUNC", {})
        # Test that every function has a reverse mapping
        func_to_action = getattr(intent_mod, "FUNC_TO_ACTION", {})
        for bare_func, action in func_to_action.items():
            if action in action_to_func:
                cases.append((class_dir, bare_func, action))
    return cases


# ── Routing tests ───────────────────────────────────────────────────

# Representative queries per class (one per function, hand-picked)
_ROUTING_QUERIES: list[tuple[str, str, str]] = [
    # gorilla_file_system
    ("gorilla_file_system", "List all files in the current directory", "ls"),
    ("gorilla_file_system", "Change to the Documents directory", "cd"),
    ("gorilla_file_system", "Create a new directory called projects", "mkdir"),
    ("gorilla_file_system", "Delete the temp file", "rm"),
    ("gorilla_file_system", "Copy report.txt to backup folder", "cp"),
    ("gorilla_file_system", "Move config.yml to the etc directory", "mv"),
    ("gorilla_file_system", "Show the contents of readme.txt", "cat"),
    ("gorilla_file_system", "Search for 'error' in the log file", "grep"),
    ("gorilla_file_system", "Create a new file called notes.txt", "touch"),
    ("gorilla_file_system", "Count the lines in data.csv", "wc"),
    ("gorilla_file_system", "What is the current directory?", "pwd"),
    ("gorilla_file_system", "Find all files named config", "find"),
    ("gorilla_file_system", "Show the last 20 lines of server.log", "tail"),
    ("gorilla_file_system", "Write 'hello world' to output.txt", "echo"),
    ("gorilla_file_system", "Compare file1.txt and file2.txt", "diff"),
    ("gorilla_file_system", "Sort the contents of names.txt", "sort"),
    ("gorilla_file_system", "Check disk usage of the home directory", "du"),
    # message_api
    ("message_api", "Send a message to John saying hello", "send_message"),
    ("message_api", "Add Kelly as a new contact", "add_contact"),
    ("message_api", "Delete message number 5", "delete_message"),
    ("message_api", "Show me all messages I have sent", "view_messages_sent"),
    ("message_api", "Search for messages containing budget", "search_messages"),
    ("message_api", "Get the user ID for john", "get_user_id"),
    ("message_api", "List all users in the system", "list_users"),
    ("message_api", "Get message statistics", "get_message_stats"),
    ("message_api", "Log in with username john and password abc", "message_login"),
    ("message_api", "Check my login status", "message_get_login_status"),
    # ticket_api
    ("ticket_api", "Create a new support ticket titled Server Down", "create_ticket"),
    ("ticket_api", "Close the ticket as the issue is now resolved", "close_ticket"),
    ("ticket_api", "Resolve the ticket with a note saying issue fixed", "resolve_ticket"),
    ("ticket_api", "Edit the ticket to update the priority to level 5", "edit_ticket"),
    ("ticket_api", "Retrieve the details of ticket number 987654", "get_ticket"),
    ("ticket_api", "Show me all tickets I have created", "get_user_tickets"),
    ("ticket_api", "Login to the ticket system with username msmith", "ticket_login"),
    ("ticket_api", "Check my login status on the ticketing system", "ticket_get_login_status"),
    ("ticket_api", "Log out of the ticketing system", "logout"),
    # twitter_api
    ("twitter_api", "Post a tweet saying 'Hello world!'", "post_tweet"),
    ("twitter_api", "Add a comment saying 'Great work!'", "comment"),
    ("twitter_api", "Retweet the tweet I just posted", "retweet"),
    ("twitter_api", "Follow user @newconnection", "follow_user"),
    ("twitter_api", "Unfollow user @spammer", "unfollow_user"),
    ("twitter_api", "Search for tweets about machine learning", "search_tweets"),
    ("twitter_api", "Log in with username john and password abc", "authenticate_twitter"),
    # posting_api
    ("posting_api", "Post a tweet saying 'Hello world!'", "post_tweet"),
    ("posting_api", "Retweet the tweet I just posted", "retweet"),
    ("posting_api", "Search for tweets about climate change", "search_tweets"),
    # math_api
    ("math_api", "Calculate the logarithm base 10 of 100", "logarithm"),
    ("math_api", "Compute the mean of these numbers", "mean"),
    ("math_api", "Calculate the standard deviation", "standard_deviation"),
    ("math_api", "Find the square root of 144", "square_root"),
    ("math_api", "Add 3 and 7 together", "add"),
    ("math_api", "Subtract 5 from 10", "subtract"),
    ("math_api", "Multiply 12 by 8", "multiply"),
    ("math_api", "Divide 100 by 5", "divide"),
    ("math_api", "Raise 2 to the power of 8", "power"),
    ("math_api", "What percentage of 200 is 50?", "percentage"),
    ("math_api", "Find the absolute value of -42", "absolute_value"),
    ("math_api", "Round the number to 2 decimal places", "round_number"),
    ("math_api", "Find the maximum value in the list", "max_value"),
    ("math_api", "Find the minimum value in the dataset", "min_value"),
    ("math_api", "Calculate the sum of all these numbers", "sum_values"),
    ("math_api", "Convert miles to kilometers", "imperial_si_conversion"),
    ("math_api", "Convert meters to centimeters", "si_unit_conversion"),
    # trading_bot
    ("trading_bot", "Add Omega Industries stock to my watchlist", "add_to_watchlist"),
    ("trading_bot", "Cancel my pending order", "cancel_order"),
    ("trading_bot", "Place a buy order for 100 shares of Tesla", "place_order"),
    ("trading_bot", "What is the current stock price of AAPL?", "get_stock_info"),
    ("trading_bot", "Show me my account balance", "get_account_info"),
    ("trading_bot", "Fund my trading account with $10000", "fund_account"),
    ("trading_bot", "Withdraw $500 from my trading account", "withdraw_funds"),
    ("trading_bot", "Remove TSLA from my watchlist", "remove_stock_from_watchlist"),
    ("trading_bot", "Show me my order history", "get_order_history"),
    ("trading_bot", "Log in to my trading account", "trading_login"),
    ("trading_bot", "Log me out of the trading platform", "trading_logout"),
    # travel_booking
    ("travel_booking", "Book a flight from LAX to JFK", "book_flight"),
    ("travel_booking", "Cancel my booking", "cancel_booking"),
    ("travel_booking", "What is the exchange rate from USD to EUR?", "compute_exchange_rate"),
    ("travel_booking", "Contact customer support about my issue", "contact_customer_support"),
    ("travel_booking", "How much does the flight cost?", "get_flight_cost"),
    ("travel_booking", "What is the nearest airport to San Francisco?", "get_nearest_airport_by_city"),
    ("travel_booking", "List all available airports", "list_all_airports"),
    ("travel_booking", "Check my credit card balance", "get_credit_card_balance"),
    ("travel_booking", "Register a new credit card", "register_credit_card"),
    ("travel_booking", "Retrieve the invoice for my booking", "retrieve_invoice"),
    ("travel_booking", "Purchase travel insurance for my flight", "purchase_insurance"),
    ("travel_booking", "Authenticate with my travel account", "authenticate_travel"),
    # vehicle_control
    ("vehicle_control", "Engage the parking brake", "activateParkingBrake"),
    ("vehicle_control", "Adjust the climate control to 72 degrees", "adjustClimateControl"),
    ("vehicle_control", "Check the tire pressure", "check_tire_pressure"),
    ("vehicle_control", "Display the car status", "displayCarStatus"),
    ("vehicle_control", "Fill the fuel tank with 30 gallons", "fillFuelTank"),
    ("vehicle_control", "Find the nearest tire shop", "find_nearest_tire_shop"),
    ("vehicle_control", "Convert 10 gallons to liters", "gallon_to_liter"),
    ("vehicle_control", "What is the current speed?", "get_current_speed"),
    ("vehicle_control", "Lock all doors", "lockDoors"),
    ("vehicle_control", "Start the engine", "startEngine"),
    ("vehicle_control", "Set cruise control to 65 mph", "setCruiseControl"),
    ("vehicle_control", "Turn on the headlights", "setHeadlights"),
    ("vehicle_control", "Navigate to 123 Main Street", "set_navigation"),
]


@pytest.mark.parametrize("class_dir, query, expected_func", _ROUTING_QUERIES,
                         ids=[f"{c}:{f}" for c, _, f in _ROUTING_QUERIES])
def test_function_routing(class_dir: str, query: str, expected_func: str):
    """Each query should route to the correct function."""
    scorer = _get_scorer(class_dir)
    result = scorer.score(query)
    assert result.functions, f"No functions returned for: {query}"
    # Strip class prefix for comparison
    top = result.functions[0]
    bare_top = top.split(".")[-1] if "." in top else top
    assert bare_top == expected_func, (
        f"Query: '{query}'\n"
        f"Expected: {expected_func}\n"
        f"Got: {bare_top} (score={result.confidence:.4f})\n"
        f"Top-3: {[(s['function'].split('.')[-1], round(s['score'], 4)) for s in result.all_scores[:3]]}"
    )


# ── Intent extraction tests ─────────────────────────────────────────

_INTENT_CASES: list[tuple[str, str, str]] = [
    # gorilla_file_system
    ("gorilla_file_system", "List all files in the directory", "ls"),
    ("gorilla_file_system", "Navigate to the Documents directory", "cd"),
    ("gorilla_file_system", "Create a new directory called projects", "mkdir"),
    ("gorilla_file_system", "Search for error in log.txt", "grep"),
    # message_api
    ("message_api", "Send a message to John", "send"),
    ("message_api", "Delete that message", "delete_msg"),
    # ticket_api
    ("ticket_api", "Create a new support ticket", "create"),
    ("ticket_api", "Close the ticket", "close"),
    # trading_bot
    ("trading_bot", "Add stock to my watchlist", "add_watchlist"),
    ("trading_bot", "Cancel my order", "cancel_order"),
    ("trading_bot", "Place a buy order for shares", "place_order"),
    # travel_booking
    ("travel_booking", "Book a flight to London", "book_flight"),
    ("travel_booking", "Cancel my booking", "cancel_booking"),
    # vehicle_control
    ("vehicle_control", "Start the engine", "start_engine"),
    ("vehicle_control", "Lock all doors", "lock_doors"),
    ("vehicle_control", "Check tire pressure", "check_tires"),
]


@pytest.mark.parametrize("class_dir, query, expected_action", _INTENT_CASES,
                         ids=[f"{c}:{a}" for c, _, a in _INTENT_CASES])
def test_intent_extraction(class_dir: str, query: str, expected_action: str):
    """Intent extraction should return the correct canonical action."""
    intent_mod = _load_class_module(class_dir, "intent")
    result = intent_mod.extract_intent(query)
    assert result["action"] == expected_action, (
        f"Query: '{query}'\n"
        f"Expected action: {expected_action}\n"
        f"Got: {result['action']}"
    )


# ── Coverage test: all functions have at least one routing query ────

def test_all_classes_have_routing_coverage():
    """Every class directory should have at least one routing test."""
    tested_classes = {c for c, _, _ in _ROUTING_QUERIES}
    for class_dir in _CLASS_DIR_MAP.values():
        assert class_dir in tested_classes, f"No routing tests for {class_dir}"
