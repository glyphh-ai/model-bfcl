"""Per-class intent extraction for multi-turn patterns.

Cross-class intent extraction covering all 9 BFCL API classes.
Maps NL queries to canonical actions/targets that discriminate between
turn patterns (ordered function sequences).

Unlike single-class intent extractors, this operates across all domains:
vehicle, travel, trading, filesystem, messaging, posting, ticket, math.

Exports:
    extract_intent(query) -> {action, target, domain, keywords}
    PATTERN_TO_ACTION      -- pattern key -> canonical action
    PATTERN_TO_TARGET      -- pattern key -> canonical target
"""

from __future__ import annotations

import re

# ── Cross-domain canonical actions ──────────────────────────────────────
# These are meta-actions that map to multi-turn patterns, not individual
# functions. A pattern like lockDoors|pressBrakePedal|startEngine maps to
# the "start_vehicle" action.

# ── NL phrase → canonical action (longest/most-specific first) ─────────

_PHRASE_MAP: list[tuple[str, str]] = [
    # -- Vehicle startup sequence --
    ("start the engine", "start_vehicle"),
    ("start the car", "start_vehicle"),
    ("start the vehicle", "start_vehicle"),
    ("start up the car", "start_vehicle"),
    ("fire up the engine", "start_vehicle"),
    ("ignite the engine", "start_vehicle"),
    ("crank the engine", "start_vehicle"),
    ("turn on the engine", "start_vehicle"),
    ("power up the engine", "start_vehicle"),
    ("start up the engine", "start_vehicle"),
    ("get the car started", "start_vehicle"),
    ("get the engine running", "start_vehicle"),
    ("ready to drive", "start_vehicle"),
    ("prepare the car", "start_vehicle"),
    ("initiate the engine", "start_vehicle"),
    # -- Vehicle fuel --
    ("fill the fuel", "fill_fuel"),
    ("fill fuel", "fill_fuel"),
    ("fill the tank", "fill_fuel"),
    ("fill up the tank", "fill_fuel"),
    ("refuel", "fill_fuel"),
    ("fuel up", "fill_fuel"),
    ("gas up", "fill_fuel"),
    ("top off the tank", "fill_fuel"),
    ("add fuel", "fill_fuel"),
    ("gallons of fuel", "fill_fuel"),
    ("liters of fuel", "fill_fuel"),
    ("replenish the fuel", "fill_fuel"),
    # -- Vehicle tires --
    ("tire pressure", "check_tires"),
    ("tyre pressure", "check_tires"),
    ("check the tires", "check_tires"),
    ("inspect the tires", "check_tires"),
    ("condition of the tires", "check_tires"),
    ("tire shop", "find_tire_shop"),
    ("nearest tire", "find_tire_shop"),
    # -- Vehicle navigation --
    ("navigate to", "navigate"),
    ("set navigation", "navigate"),
    ("set the destination", "navigate"),
    ("directions to", "navigate"),
    ("route to", "navigate"),
    ("take me to", "navigate"),
    ("drive to", "navigate"),
    # -- Vehicle distance/conversion --
    ("estimate the distance", "estimate_distance"),
    ("distance between", "estimate_distance"),
    ("how far is", "estimate_distance"),
    ("how far between", "estimate_distance"),
    ("gallon to liter", "convert_units"),
    ("liter to gallon", "convert_units"),
    ("convert gallons", "convert_units"),
    ("convert liters", "convert_units"),
    # -- Vehicle controls --
    ("lock all doors", "lock_doors"),
    ("lock the doors", "lock_doors"),
    ("unlock the doors", "lock_doors"),
    ("secure the doors", "lock_doors"),
    ("parking brake", "parking_brake"),
    ("cruise control", "cruise_control"),
    ("headlights", "headlights"),
    ("climate control", "climate_control"),
    ("set the temperature", "climate_control"),
    ("air conditioning", "climate_control"),
    ("car status", "vehicle_status"),
    ("vehicle status", "vehicle_status"),
    ("display the status", "vehicle_status"),
    ("current speed", "vehicle_status"),
    ("outside temperature", "check_weather"),
    # -- Travel --
    ("book a flight", "book_flight"),
    ("book the flight", "book_flight"),
    ("book flight", "book_flight"),
    ("book a trip", "book_flight"),
    ("flight cost", "check_flight_cost"),
    ("cost of a flight", "check_flight_cost"),
    ("cost of the flight", "check_flight_cost"),
    ("how much does a flight", "check_flight_cost"),
    ("how much is the flight", "check_flight_cost"),
    ("how much would it cost to fly", "check_flight_cost"),
    ("price of a flight", "check_flight_cost"),
    ("cancel the booking", "cancel_booking"),
    ("cancel my booking", "cancel_booking"),
    ("cancel booking", "cancel_booking"),
    ("cancel the reservation", "cancel_booking"),
    ("cancel my flight", "cancel_booking"),
    ("purchase insurance", "purchase_insurance"),
    ("buy insurance", "purchase_insurance"),
    ("travel insurance", "purchase_insurance"),
    ("insurance for the trip", "purchase_insurance"),
    ("contact customer support", "contact_support"),
    ("customer support", "contact_support"),
    ("contact support", "contact_support"),
    ("reach out to support", "contact_support"),
    ("reach support", "contact_support"),
    # -- Trading --
    ("stock info", "check_stock"),
    ("stock information", "check_stock"),
    ("stock price", "check_stock"),
    ("current price of", "check_stock"),
    ("check the stock", "check_stock"),
    ("get stock", "check_stock"),
    ("look up the stock", "check_stock"),
    ("place an order", "place_order"),
    ("place order", "place_order"),
    ("buy shares", "place_order"),
    ("sell shares", "place_order"),
    ("purchase shares", "place_order"),
    ("buy stock", "place_order"),
    ("sell stock", "place_order"),
    ("add to watchlist", "add_watchlist"),
    ("add to my watchlist", "add_watchlist"),
    ("watchlist", "check_watchlist"),
    ("watch list", "check_watchlist"),
    ("account info", "check_account"),
    ("account information", "check_account"),
    ("account balance", "check_account"),
    ("account details", "check_account"),
    ("my account", "check_account"),
    ("exchange rate", "convert_currency"),
    ("set budget", "set_budget"),
    ("budget limit", "set_budget"),
    ("cancel order", "cancel_order"),
    ("cancel the order", "cancel_order"),
    ("cancel my order", "cancel_order"),
    ("order details", "check_order"),
    ("order status", "check_order"),
    ("order information", "check_order"),
    ("get the order", "check_order"),
    ("check the order", "check_order"),
    ("retrieve the order", "check_order"),
    ("retrieve invoice", "retrieve_invoice"),
    ("get the invoice", "retrieve_invoice"),
    ("get invoice", "retrieve_invoice"),
    ("view the invoice", "retrieve_invoice"),
    ("check the invoice", "retrieve_invoice"),
    ("transaction history", "check_history"),
    ("trading history", "check_history"),
    ("trade history", "check_history"),
    ("order history", "check_history"),
    # -- Messaging --
    ("send a message", "send_message"),
    ("send message", "send_message"),
    ("send the message", "send_message"),
    ("view messages", "view_messages"),
    ("view my messages", "view_messages"),
    ("view sent messages", "view_messages"),
    ("view my sent messages", "view_messages"),
    ("messages sent", "view_messages"),
    ("check messages", "view_messages"),
    ("my sent messages", "view_messages"),
    ("search messages", "search_messages"),
    ("search for messages", "search_messages"),
    ("find messages", "search_messages"),
    # -- Posting (Twitter) --
    ("post a tweet", "post_tweet"),
    ("post tweet", "post_tweet"),
    ("send a tweet", "post_tweet"),
    ("tweet about", "post_tweet"),
    ("authenticate twitter", "auth_post"),
    ("log in to twitter", "auth_post"),
    ("login to twitter", "auth_post"),
    ("retweet", "retweet"),
    ("share the tweet", "retweet"),
    ("comment on", "comment"),
    ("add a comment", "comment"),
    ("post a comment", "comment"),
    ("reply to", "comment"),
    ("get user", "get_user"),
    ("get the user", "get_user"),
    # -- Ticket --
    ("create a ticket", "create_ticket"),
    ("create ticket", "create_ticket"),
    ("open a ticket", "create_ticket"),
    ("file a ticket", "create_ticket"),
    ("submit a ticket", "create_ticket"),
    ("log in to ticket", "ticket_login"),
    ("login to ticket", "ticket_login"),
    ("ticket login", "ticket_login"),
    ("resolve the ticket", "resolve_ticket"),
    ("close the ticket", "resolve_ticket"),
    ("edit the ticket", "edit_ticket"),
    ("update the ticket", "edit_ticket"),
    ("get ticket", "get_ticket"),
    ("ticket details", "get_ticket"),
    ("ticket status", "get_ticket"),
    # -- Filesystem --
    ("list of files", "list_files"),
    ("list all files", "list_files"),
    ("list the files", "list_files"),
    ("available files", "list_files"),
    ("new directory", "create_dir"),
    ("create directory", "create_dir"),
    ("make directory", "create_dir"),
    ("new folder", "create_dir"),
    ("create folder", "create_dir"),
    ("change directory", "change_dir"),
    ("navigate to", "change_dir"),
    ("go to", "change_dir"),
    ("move the file", "move_file"),
    ("rename the file", "move_file"),
    ("copy the file", "copy_file"),
    ("duplicate the file", "copy_file"),
    ("delete the file", "delete_file"),
    ("remove the file", "delete_file"),
    ("read the content", "read_file"),
    ("display contents", "read_file"),
    ("show contents", "read_file"),
    ("view its contents", "read_file"),
    ("search in file", "search_file"),
    ("search for text", "search_file"),
    ("search the file", "search_file"),
    ("occurrence of", "search_file"),
    ("word count", "count_file"),
    ("line count", "count_file"),
    ("how many lines", "count_file"),
    ("how many words", "count_file"),
    ("sort the file", "sort_file"),
    ("sort the output", "sort_file"),
    ("compare the", "compare_files"),
    ("differences between", "compare_files"),
    ("disk usage", "disk_usage"),
    ("how much space", "disk_usage"),
    ("first lines", "head_file"),
    ("last lines", "tail_file"),
    ("write into", "write_file"),
    ("jot down", "write_file"),
    # -- Math --
    ("calculate", "compute"),
    ("compute", "compute"),
    ("add up", "compute"),
    ("multiply", "compute"),
    ("logarithm", "compute"),
    ("mean of", "compute"),
    ("standard deviation", "compute"),
    ("percentage of", "compute"),
    ("greatest common", "compute"),
    ("least common", "compute"),
    ("square root", "compute"),
    ("convert celsius", "convert_temp"),
    ("convert fahrenheit", "convert_temp"),
    ("celsius to", "convert_temp"),
    ("fahrenheit to", "convert_temp"),
]

_WORD_MAP: dict[str, str] = {
    # Vehicle
    "lock": "lock_doors",
    "unlock": "lock_doors",
    "refuel": "fill_fuel",
    "cruise": "cruise_control",
    "brake": "parking_brake",
    # Travel
    "book": "book_flight",
    "cancel": "cancel_order",
    "insurance": "purchase_insurance",
    # Trading
    "stock": "check_stock",
    "shares": "place_order",
    "watchlist": "check_watchlist",
    "invoice": "retrieve_invoice",
    # Messaging
    "message": "send_message",
    "tweet": "post_tweet",
    # Filesystem
    "list": "list_files",
    "copy": "copy_file",
    "move": "move_file",
    "rename": "move_file",
    "delete": "delete_file",
    "remove": "delete_file",
    "search": "search_file",
    "compare": "compare_files",
    "sort": "sort_file",
    "read": "read_file",
    "display": "read_file",
    # Ticket
    "ticket": "create_ticket",
    # Math
    "calculate": "compute",
    "compute": "compute",
    "logarithm": "compute",
}

# ── Target vocabulary ───────────────────────────────────────────────────

_TARGET_MAP: dict[str, str] = {
    # Vehicle
    "engine": "engine",
    "car": "vehicle",
    "vehicle": "vehicle",
    "door": "doors",
    "doors": "doors",
    "fuel": "fuel",
    "gas": "fuel",
    "tank": "fuel",
    "tire": "tire",
    "tires": "tire",
    "brake": "brake",
    "speed": "speed",
    "headlights": "lights",
    "lights": "lights",
    "temperature": "temperature",
    "navigation": "route",
    "destination": "route",
    "distance": "distance",
    "mileage": "distance",
    # Travel
    "flight": "flight",
    "booking": "booking",
    "reservation": "booking",
    "trip": "trip",
    "insurance": "insurance",
    "support": "support",
    # Trading
    "stock": "stock",
    "shares": "stock",
    "order": "order",
    "account": "account",
    "watchlist": "watchlist",
    "invoice": "invoice",
    "budget": "budget",
    "portfolio": "portfolio",
    # Messaging
    "message": "message",
    "messages": "message",
    "tweet": "tweet",
    "comment": "comment",
    "user": "user",
    # Filesystem
    "file": "file",
    "files": "file",
    "document": "file",
    "directory": "directory",
    "folder": "directory",
    "content": "content",
    "contents": "content",
    # Ticket
    "ticket": "ticket",
    "issue": "ticket",
    # Math
    "number": "number",
    "result": "result",
}

# ── Domain detection ────────────────────────────────────────────────────

_DOMAIN_SIGNALS: dict[str, list[str]] = {
    "vehicle": [
        "engine", "car", "vehicle", "door", "doors", "fuel", "gas", "tank",
        "tire", "tires", "brake", "speed", "headlights", "cruise", "ignition",
        "gallon", "liter", "mileage", "odometer", "drive", "lock", "unlock",
        "parking", "navigation", "zipcode",
    ],
    "travel": [
        "flight", "booking", "reservation", "trip", "travel", "airport",
        "airline", "passenger", "insurance", "itinerary", "boarding",
    ],
    "trading": [
        "stock", "shares", "order", "account", "watchlist", "invoice",
        "portfolio", "trading", "market", "exchange", "budget", "trade",
    ],
    "filesystem": [
        "file", "files", "directory", "folder", "document", "contents",
        "path", "disk", "copy", "move", "rename", "search", "grep",
    ],
    "messaging": [
        "message", "messages", "inbox", "sent", "receiver", "sender",
    ],
    "posting": [
        "tweet", "twitter", "retweet", "post", "comment", "hashtag",
    ],
    "ticket": [
        "ticket", "issue", "priority", "resolve", "assigned",
    ],
    "math": [
        "calculate", "compute", "logarithm", "mean", "deviation",
        "percentage", "multiply", "divide", "factorial", "gcd", "lcm",
    ],
}

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


def _detect_domain(query_lower: str) -> str:
    """Detect the dominant domain from query keywords."""
    scores: dict[str, int] = {}
    words = set(re.sub(r"[^a-z0-9\s]", " ", query_lower).split())
    for domain, signals in _DOMAIN_SIGNALS.items():
        score = sum(1 for s in signals if s in words)
        if score > 0:
            scores[domain] = score
    if not scores:
        return "none"
    return max(scores, key=scores.get)


def extract_intent(query: str) -> dict[str, str]:
    """Extract multi-turn intent from NL query.

    Returns:
        {action: str, target: str, domain: str, keywords: str}
    """
    q_lower = query.lower()

    # 1. Phrase match (longest first)
    action = "none"
    for phrase, act in _PHRASE_MAP:
        if phrase in q_lower:
            action = act
            break

    # 2. Word match fallback
    if action == "none":
        words = re.sub(r"[^a-z0-9\s]", " ", q_lower).split()
        for w in words:
            if w in _WORD_MAP:
                action = _WORD_MAP[w]
                break

    # 3. Target extraction
    target = "none"
    words = re.sub(r"[^a-z0-9\s]", " ", q_lower).split()
    for w in words:
        if w in _TARGET_MAP:
            target = _TARGET_MAP[w]
            break

    # 4. Domain detection
    domain = _detect_domain(q_lower)

    # 5. Keywords (stop words removed)
    kw_tokens = [w for w in words if w not in _STOP_WORDS and len(w) > 1]
    keywords = " ".join(dict.fromkeys(kw_tokens))

    return {"action": action, "target": target, "domain": domain, "keywords": keywords}
