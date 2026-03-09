"""Glyphh Ada — Tool Execution intent extraction.

Unified intent extraction across all 9 BFCL API classes.
Extracts action, target, entities, and keywords from natural language queries.

Exports:
    extract_intent(query) → {action, target, entities, keywords}
    extract_entities(text) → list of entity strings
"""

from __future__ import annotations

import re

# ── Verb → action mapping ───────────────────────────────────────────────
# Ordered longest-first within each domain to avoid partial matches.

_VERB_MAP = {
    # filesystem
    "list all": "ls", "list files": "ls", "list": "ls",
    "show files": "ls", "display files": "ls",
    "change directory": "cd", "change to": "cd",
    "navigate to": "cd", "navigate": "cd",
    "go into": "cd", "go to": "cd",
    "create a new directory": "mkdir", "create directory": "mkdir",
    "make directory": "mkdir", "make a directory": "mkdir",
    "create a new file": "touch", "create file": "touch",
    "create a document": "touch", "draft up a create a document": "touch",
    "delete": "rm", "remove file": "rm",
    "copy": "cp", "duplicate": "cp",
    "move": "mv", "rename": "mv",
    "show the contents": "cat", "show contents": "cat",
    "display the contents": "cat", "read": "cat",
    "search for": "grep", "search using grep": "grep",
    "find text": "grep", "look for": "grep", "identify sections": "grep",
    "count the lines": "wc", "count lines": "wc", "count": "wc",
    "current directory": "pwd", "where am i": "pwd",
    "find all files": "find_files", "find": "find_files",
    "last lines": "tail", "show the last": "tail", "show last": "tail",
    "jot down": "echo", "write": "echo",
    "juxtapose": "diff", "compare": "diff",
    "sort the": "sort", "sort": "sort",
    "disk usage": "du", "disk space": "du",
    # messaging
    "send a message": "send_message", "send message": "send_message",
    "add contact": "add_contact", "add as a new contact": "add_contact",
    "delete message": "delete_message",
    "view messages": "view_messages_sent", "show me all messages": "view_messages_sent",
    "search messages": "search_messages", "search for messages": "search_messages",
    "get the user id": "get_user_id", "get user id": "get_user_id",
    "list all users": "list_users", "list users": "list_users",
    "message statistics": "get_message_stats", "get message stats": "get_message_stats",
    # ticketing
    "create a new support ticket": "create_ticket", "create a ticket": "create_ticket",
    "create ticket": "create_ticket",
    "close the ticket": "close_ticket", "close ticket": "close_ticket",
    "resolve the ticket": "resolve_ticket", "resolve ticket": "resolve_ticket",
    "edit the ticket": "edit_ticket", "edit ticket": "edit_ticket",
    "retrieve the details": "get_ticket", "get ticket": "get_ticket",
    "show me all tickets": "get_user_tickets", "get user tickets": "get_user_tickets",
    "log out": "logout", "logout": "logout",
    # social
    "post a tweet": "post_tweet", "post tweet": "post_tweet",
    "retweet": "retweet",
    "add a comment": "comment", "comment": "comment",
    "follow user": "follow_user", "follow": "follow_user",
    "unfollow user": "unfollow_user", "unfollow": "unfollow_user",
    "search for tweets": "search_tweets", "search tweets": "search_tweets",
    # math
    "square root": "square_root",
    "standard deviation": "standard_deviation",
    "absolute value": "absolute_value",
    "logarithm": "logarithm",
    "percentage": "percentage", "percent": "percentage",
    "round the number": "round_number", "round": "round_number",
    "maximum value": "max_value", "maximum": "max_value",
    "minimum value": "min_value", "minimum": "min_value",
    "sum of": "sum_values", "sum": "sum_values",
    "mean of": "mean", "mean": "mean", "average": "mean",
    "convert miles": "imperial_si_conversion", "convert gallons": "imperial_si_conversion",
    "convert meters": "si_unit_conversion", "convert kilometers": "si_unit_conversion",
    "convert": "imperial_si_conversion",
    "add": "add", "subtract": "subtract",
    "multiply": "multiply", "divide": "divide",
    "raise": "power", "power": "power",
    # trading
    "add to watchlist": "add_to_watchlist", "add to my watchlist": "add_to_watchlist",
    "cancel my pending order": "cancel_order", "cancel order": "cancel_order",
    "cancel my order": "cancel_order",
    "place a buy order": "place_order", "place order": "place_order",
    "place a sell order": "place_order", "place an order": "place_order",
    "stock price": "get_stock_info", "current price": "get_stock_info",
    "current stock price": "get_stock_info",
    "account balance": "get_account_info", "show me my account": "get_account_info",
    "fund my": "fund_account", "fund account": "fund_account", "deposit": "fund_account",
    "withdraw": "withdraw_funds",
    "remove from watchlist": "remove_stock_from_watchlist",
    "remove from my watchlist": "remove_stock_from_watchlist",
    "order history": "get_order_history", "show me my order": "get_order_history",
    # travel
    "book a flight": "book_flight", "book flight": "book_flight",
    "cancel my booking": "cancel_booking", "cancel booking": "cancel_booking",
    "exchange rate": "compute_exchange_rate",
    "contact customer support": "contact_customer_support",
    "contact support": "contact_customer_support",
    "flight cost": "get_flight_cost", "how much does the flight": "get_flight_cost",
    "nearest airport": "get_nearest_airport_by_city",
    "list all available airports": "list_all_airports", "list airports": "list_all_airports",
    "credit card balance": "get_credit_card_balance", "card balance": "get_credit_card_balance",
    "register a new credit card": "register_credit_card", "register credit card": "register_credit_card",
    "retrieve the invoice": "retrieve_invoice", "retrieve invoice": "retrieve_invoice",
    "purchase insurance": "purchase_insurance", "purchase travel insurance": "purchase_insurance",
    # vehicle
    "start the engine": "startEngine", "start engine": "startEngine",
    "parking brake": "activateParkingBrake", "engage the parking brake": "activateParkingBrake",
    "engage brake": "activateParkingBrake",
    "climate control": "adjustClimateControl", "adjust the climate": "adjustClimateControl",
    "check the tire pressure": "check_tire_pressure", "tire pressure": "check_tire_pressure",
    "display the car status": "displayCarStatus", "car status": "displayCarStatus",
    "fill the fuel tank": "fillFuelTank", "fill fuel": "fillFuelTank", "fill tank": "fillFuelTank",
    "find the nearest tire shop": "find_nearest_tire_shop",
    "gallon to liter": "gallon_to_liter", "gallons to liters": "gallon_to_liter",
    "current speed": "get_current_speed",
    "lock all doors": "lockDoors", "lock doors": "lockDoors", "lock the doors": "lockDoors",
    "set cruise control": "setCruiseControl", "cruise control": "setCruiseControl",
    "turn on the headlights": "setHeadlights", "headlights": "setHeadlights",
    "set navigation": "set_navigation",
    "press the brake": "pressBrakePedal", "press brake": "pressBrakePedal",
    "release the brake": "releaseBrakePedal", "release brake": "releaseBrakePedal",
    # auth (generic — matched last)
    "log in": "login", "login": "login", "authenticate": "login",
    "login status": "login_status", "check my login": "login_status",
}

# ── Target mapping ──────────────────────────────────────────────────────

_TARGET_MAP = {
    # filesystem
    "directory": "directory", "folder": "directory", "dir": "directory",
    "file": "file", "document": "file",
    # messaging
    "message": "message", "msg": "message",
    "contact": "contact",
    # ticketing
    "ticket": "ticket",
    # social
    "tweet": "tweet", "post": "tweet",
    # math
    "number": "number", "numbers": "number",
    # trading
    "stock": "stock", "share": "stock", "shares": "stock",
    "order": "order",
    "account": "account", "balance": "account",
    "watchlist": "watchlist",
    # travel
    "flight": "flight",
    "booking": "booking", "reservation": "booking",
    "airport": "airport",
    "credit card": "card", "card": "card",
    "invoice": "invoice",
    "insurance": "insurance",
    # vehicle
    "engine": "engine",
    "brake": "brake",
    "door": "door", "doors": "door",
    "tire": "tire", "tires": "tire",
    "fuel": "fuel", "gas": "fuel", "tank": "fuel",
    "speed": "speed",
    "headlight": "headlight", "light": "headlight",
}

# ── Stop words and helpers ──────────────────────────────────────────────

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


def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return [w for w in cleaned.split() if w not in _STOP_WORDS and len(w) > 1]


# ── Entity extraction ──────────────────────────────────────────────────

def extract_entities(text: str) -> list[str]:
    """Extract entities from text: quoted strings, filenames, numbers, @mentions."""
    entities = []
    # Quoted strings (preserve original case)
    entities.extend(re.findall(r"'([^']*)'", text))
    entities.extend(re.findall(r'"([^"]*)"', text))
    # Filenames (word.ext pattern)
    entities.extend(re.findall(r'\b[\w-]+\.[\w]+\b', text))
    # Numbers
    entities.extend(re.findall(r'\b\d+\.?\d*\b', text))
    # @mentions
    entities.extend(re.findall(r'@(\w+)', text))
    return list(dict.fromkeys(entities))


# ── Main extract_intent ─────────────────────────────────────────────────

def extract_intent(query: str) -> dict:
    """Extract intent from a natural language query.

    Returns:
        {action, target, entities, keywords}
    """
    q = query.lower().strip()

    # Action: longest-match from verb map
    action = "none"
    best_len = 0
    for phrase, act in _VERB_MAP.items():
        if phrase in q and len(phrase) > best_len:
            action = act
            best_len = len(phrase)

    # Target: first match from target map
    target = "none"
    for phrase, tgt in _TARGET_MAP.items():
        if phrase in q:
            target = tgt
            break

    # Entities
    entities = extract_entities(query)

    # Keywords
    keywords = _tokenize(query)

    return {
        "action": action,
        "target": target,
        "entities": entities,
        "keywords": keywords,
    }
