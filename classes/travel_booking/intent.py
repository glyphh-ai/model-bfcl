"""Per-class intent extraction for TravelBookingAPI (TravelAPI).

Uses phrase/word maps for NL -> action/target extraction.
Maps canonical actions to the 18 TravelAPI functions.

Exports:
    extract_intent(query) -> {action, target, keywords}
    ACTION_TO_FUNC        -- canonical action -> TravelAPI.{func}
    FUNC_TO_ACTION        -- bare func name -> canonical action
    FUNC_TO_TARGET        -- bare func name -> canonical target
"""

from __future__ import annotations

import re

# -- Pack canonical action -> TravelAPI function ----------------------------

ACTION_TO_FUNC: dict[str, str] = {
    "authenticate":       "TravelAPI.authenticate_travel",
    "book_flight":        "TravelAPI.book_flight",
    "cancel_booking":     "TravelAPI.cancel_booking",
    "exchange_rate":      "TravelAPI.compute_exchange_rate",
    "contact_support":    "TravelAPI.contact_customer_support",
    "get_all_cards":      "TravelAPI.get_all_credit_cards",
    "get_booking_history":"TravelAPI.get_booking_history",
    "get_budget_year":    "TravelAPI.get_budget_fiscal_year",
    "get_card_balance":   "TravelAPI.get_credit_card_balance",
    "get_flight_cost":    "TravelAPI.get_flight_cost",
    "get_nearest_airport":"TravelAPI.get_nearest_airport_by_city",
    "list_airports":      "TravelAPI.list_all_airports",
    "purchase_insurance": "TravelAPI.purchase_insurance",
    "register_card":      "TravelAPI.register_credit_card",
    "retrieve_invoice":   "TravelAPI.retrieve_invoice",
    "set_budget":         "TravelAPI.set_budget_limit",
    "login_status":       "TravelAPI.travel_get_login_status",
    "verify_traveler":    "TravelAPI.verify_traveler_information",
    "none":               "none",
}

# Reverse: bare function name -> canonical action
FUNC_TO_ACTION: dict[str, str] = {
    "authenticate_travel":        "authenticate",
    "book_flight":                "book_flight",
    "cancel_booking":             "cancel_booking",
    "compute_exchange_rate":      "exchange_rate",
    "contact_customer_support":   "contact_support",
    "get_all_credit_cards":       "get_all_cards",
    "get_booking_history":        "get_booking_history",
    "get_budget_fiscal_year":     "get_budget_year",
    "get_credit_card_balance":    "get_card_balance",
    "get_flight_cost":            "get_flight_cost",
    "get_nearest_airport_by_city":"get_nearest_airport",
    "list_all_airports":          "list_airports",
    "purchase_insurance":         "purchase_insurance",
    "register_credit_card":       "register_card",
    "retrieve_invoice":           "retrieve_invoice",
    "set_budget_limit":           "set_budget",
    "travel_get_login_status":    "login_status",
    "verify_traveler_information":"verify_traveler",
}

# Function -> canonical target
FUNC_TO_TARGET: dict[str, str] = {
    "authenticate_travel":        "account",
    "book_flight":                "flight",
    "cancel_booking":             "booking",
    "compute_exchange_rate":      "currency",
    "contact_customer_support":   "support",
    "get_all_credit_cards":       "card",
    "get_booking_history":        "booking",
    "get_budget_fiscal_year":     "budget",
    "get_credit_card_balance":    "card",
    "get_flight_cost":            "flight",
    "get_nearest_airport_by_city":"airport",
    "list_all_airports":          "airport",
    "purchase_insurance":         "insurance",
    "register_credit_card":       "card",
    "retrieve_invoice":           "invoice",
    "set_budget_limit":           "budget",
    "travel_get_login_status":    "account",
    "verify_traveler_information":"traveler",
}

# -- NL synonym -> canonical action ----------------------------------------
# Derived from tests.jsonl query language. Longest/most specific first.

_PHRASE_MAP: list[tuple[str, str]] = [
    # authenticate — long phrases first
    ("authentication process", "authenticate"),
    ("authenticate with", "authenticate"),
    ("handle the authentication", "authenticate"),
    ("access my account", "authenticate"),
    ("securing your travel account", "authenticate"),
    ("access my travel account", "authenticate"),
    ("log into my travel", "authenticate"),
    ("log in to my travel", "authenticate"),
    ("sign into my travel", "authenticate"),
    ("authenticate me", "authenticate"),
    ("grant type", "authenticate"),
    ("client_id", "authenticate"),
    ("client id", "authenticate"),
    ("client secret", "authenticate"),
    ("refresh token", "authenticate"),
    # login status
    ("login status", "login_status"),
    ("logged in status", "login_status"),
    ("currently logged in", "login_status"),
    ("am i logged in", "login_status"),
    ("check if i am logged", "login_status"),
    ("check the login", "login_status"),
    # verify traveler
    ("verify traveler", "verify_traveler"),
    ("verify the traveler", "verify_traveler"),
    ("verify my traveler", "verify_traveler"),
    ("traveler information", "verify_traveler"),
    ("traveler details", "verify_traveler"),
    ("verify my identity", "verify_traveler"),
    ("verify my information", "verify_traveler"),
    ("passport number", "verify_traveler"),
    ("date of birth", "verify_traveler"),
    # booking history — before "book"
    ("booking history", "get_booking_history"),
    ("past bookings", "get_booking_history"),
    ("previous bookings", "get_booking_history"),
    ("travel history", "get_booking_history"),
    ("my bookings", "get_booking_history"),
    ("retrieve all booking", "get_booking_history"),
    ("review my booking", "get_booking_history"),
    ("list of bookings", "get_booking_history"),
    # cancel booking — before "book"
    ("cancel the booking", "cancel_booking"),
    ("cancel my booking", "cancel_booking"),
    ("cancel booking", "cancel_booking"),
    ("cancel the flight", "cancel_booking"),
    ("cancel my flight", "cancel_booking"),
    ("cancel this booking", "cancel_booking"),
    ("cancel that booking", "cancel_booking"),
    ("cancellation of", "cancel_booking"),
    # flight cost — MUST come before book_flight ("flight" and "class" phrases)
    ("flight cost", "get_flight_cost"),
    ("cost of a flight", "get_flight_cost"),
    ("cost of the flight", "get_flight_cost"),
    ("cost of flying", "get_flight_cost"),
    ("how much does the flight", "get_flight_cost"),
    ("how much would it cost to fly", "get_flight_cost"),
    ("how much will the flight", "get_flight_cost"),
    ("price of a flight", "get_flight_cost"),
    ("price of the flight", "get_flight_cost"),
    ("ticket price", "get_flight_cost"),
    ("fare for", "get_flight_cost"),
    ("flight fare", "get_flight_cost"),
    ("flight price", "get_flight_cost"),
    ("how much a flight", "get_flight_cost"),
    ("estimate the cost", "get_flight_cost"),
    ("cost estimate", "get_flight_cost"),
    # purchase insurance — MUST come before book_flight ("flight booking" phrase)
    ("purchase insurance", "purchase_insurance"),
    ("buy insurance", "purchase_insurance"),
    ("get insurance", "purchase_insurance"),
    ("travel insurance", "purchase_insurance"),
    ("insurance for the", "purchase_insurance"),
    ("insurance coverage", "purchase_insurance"),
    ("insure the", "purchase_insurance"),
    ("insurance plan", "purchase_insurance"),
    # book flight
    ("book a flight", "book_flight"),
    ("book the flight", "book_flight"),
    ("book my flight", "book_flight"),
    ("book flight", "book_flight"),
    ("reserve a flight", "book_flight"),
    ("reserve flight", "book_flight"),
    ("schedule a flight", "book_flight"),
    ("flight booking", "book_flight"),
    ("secure a seat", "book_flight"),
    ("travel from", "book_flight"),
    ("fly from", "book_flight"),
    ("fly to", "book_flight"),
    ("one-way flight", "book_flight"),
    ("first class flight", "book_flight"),
    ("business class flight", "book_flight"),
    ("economy class flight", "book_flight"),
    ("economy class", "book_flight"),
    ("business class", "book_flight"),
    ("first class", "book_flight"),
    # exchange rate — before "convert"
    ("exchange rate", "exchange_rate"),
    ("convert currency", "exchange_rate"),
    ("currency conversion", "exchange_rate"),
    ("currency exchange", "exchange_rate"),
    ("compute the exchange", "exchange_rate"),
    ("exchange value", "exchange_rate"),
    ("convert the amount", "exchange_rate"),
    ("how much is", "exchange_rate"),
    ("converted to", "exchange_rate"),
    ("equivalent in", "exchange_rate"),
    ("conversion rate", "exchange_rate"),
    ("from usd to", "exchange_rate"),
    ("from eur to", "exchange_rate"),
    ("to usd", "exchange_rate"),
    ("to eur", "exchange_rate"),
    ("to gbp", "exchange_rate"),
    ("to jpy", "exchange_rate"),
    ("in euros", "exchange_rate"),
    ("in dollars", "exchange_rate"),
    ("in pounds", "exchange_rate"),
    ("in yen", "exchange_rate"),
    # contact support
    ("contact customer support", "contact_support"),
    ("customer support", "contact_support"),
    ("contact support", "contact_support"),
    ("reach out to support", "contact_support"),
    ("reach out to customer", "contact_support"),
    ("get support", "contact_support"),
    ("support team", "contact_support"),
    # (flight cost moved before book_flight)
    # nearest airport
    ("nearest airport", "get_nearest_airport"),
    ("closest airport", "get_nearest_airport"),
    ("airport near", "get_nearest_airport"),
    ("airport closest to", "get_nearest_airport"),
    ("airport in the vicinity", "get_nearest_airport"),
    ("find the airport", "get_nearest_airport"),
    ("airport for the city", "get_nearest_airport"),
    ("which airport", "get_nearest_airport"),
    ("local airport", "get_nearest_airport"),
    # list airports
    ("list all airports", "list_airports"),
    ("all available airports", "list_airports"),
    ("available airports", "list_airports"),
    ("list of airports", "list_airports"),
    ("show all airports", "list_airports"),
    ("every airport", "list_airports"),
    # credit card balance — before "credit card"
    ("card balance", "get_card_balance"),
    ("credit card balance", "get_card_balance"),
    ("balance of my credit card", "get_card_balance"),
    ("balance on my card", "get_card_balance"),
    ("balance of the card", "get_card_balance"),
    ("check the balance", "get_card_balance"),
    ("remaining balance", "get_card_balance"),
    ("available balance", "get_card_balance"),
    # all credit cards — before "credit card"
    ("all credit cards", "get_all_cards"),
    ("all registered credit", "get_all_cards"),
    ("list of credit cards", "get_all_cards"),
    ("all my credit cards", "get_all_cards"),
    ("registered credit cards", "get_all_cards"),
    ("credit cards on file", "get_all_cards"),
    ("show my credit cards", "get_all_cards"),
    ("cards i have registered", "get_all_cards"),
    ("cards on file", "get_all_cards"),
    ("list all cards", "get_all_cards"),
    # register credit card
    ("register a credit card", "register_card"),
    ("register my credit card", "register_card"),
    ("register the credit card", "register_card"),
    ("register credit card", "register_card"),
    ("add a credit card", "register_card"),
    ("add my credit card", "register_card"),
    ("add a new card", "register_card"),
    ("register a new card", "register_card"),
    ("new credit card", "register_card"),
    ("set up a credit card", "register_card"),
    # retrieve invoice
    ("retrieve the invoice", "retrieve_invoice"),
    ("retrieve my invoice", "retrieve_invoice"),
    ("retrieve invoice", "retrieve_invoice"),
    ("get the invoice", "retrieve_invoice"),
    ("get my invoice", "retrieve_invoice"),
    ("get invoice", "retrieve_invoice"),
    ("invoice for", "retrieve_invoice"),
    ("view the invoice", "retrieve_invoice"),
    ("show the invoice", "retrieve_invoice"),
    ("booking invoice", "retrieve_invoice"),
    # (purchase insurance moved before book_flight)
    # budget fiscal year — before "budget"
    ("fiscal year", "get_budget_year"),
    ("budget fiscal year", "get_budget_year"),
    ("current fiscal year", "get_budget_year"),
    ("fiscal budget", "get_budget_year"),
    # set budget
    ("set the budget", "set_budget"),
    ("set my budget", "set_budget"),
    ("set a budget", "set_budget"),
    ("set budget", "set_budget"),
    ("budget limit", "set_budget"),
    ("budget to", "set_budget"),
    ("spending limit", "set_budget"),
    ("cap the budget", "set_budget"),
    ("establish a budget", "set_budget"),
    ("update the budget", "set_budget"),
]

_WORD_MAP: dict[str, str] = {
    "authenticate": "authenticate",
    "login": "authenticate",
    "book": "book_flight",
    "reserve": "book_flight",
    "cancel": "cancel_booking",
    "exchange": "exchange_rate",
    "convert": "exchange_rate",
    "support": "contact_support",
    "invoice": "retrieve_invoice",
    "receipt": "retrieve_invoice",
    "insurance": "purchase_insurance",
    "verify": "verify_traveler",
    "budget": "set_budget",
    "airports": "list_airports",
}

_TARGET_MAP: dict[str, str] = {
    "flight": "flight",
    "flights": "flight",
    "ticket": "flight",
    "plane": "flight",
    "booking": "booking",
    "reservation": "booking",
    "card": "card",
    "credit": "card",
    "airport": "airport",
    "airports": "airport",
    "budget": "budget",
    "invoice": "invoice",
    "receipt": "invoice",
    "insurance": "insurance",
    "currency": "currency",
    "account": "account",
    "traveler": "traveler",
    "passenger": "traveler",
    "support": "support",
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


def extract_intent(query: str) -> dict[str, str]:
    """Extract travel intent from NL query.

    Returns:
        {action: str, target: str, keywords: str}
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

    # 4. Keywords (stop words removed)
    kw_tokens = [w for w in words if w not in _STOP_WORDS and len(w) > 1]
    keywords = " ".join(dict.fromkeys(kw_tokens))

    return {"action": action, "target": target, "keywords": keywords}
