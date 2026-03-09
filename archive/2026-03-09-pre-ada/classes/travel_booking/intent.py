"""Per-class intent overrides for TravelAPI.

Maps NL verbs/targets to the 22 travel booking functions.
Used by the per-class encoder for Stage 2 routing.
"""

CLASS_ALIASES = ["TravelAPI", "TravelBookingAPI", "travel", "booking", "flight"]
CLASS_DOMAIN = "travel"

# NL verb → canonical function name (bare, without class prefix)
ACTION_SYNONYMS = {
    # Booking / flights
    "book": "book_flight", "reserve": "book_flight", "fly": "book_flight",
    "book a flight": "book_flight", "reserve flight": "book_flight",
    "schedule flight": "book_flight",
    "book hotel": "book_hotel", "reserve hotel": "book_hotel",
    "reserve room": "book_hotel", "book room": "book_hotel",
    # Cancellation
    "cancel": "cancel_booking", "abort": "cancel_booking",
    "cancel booking": "cancel_booking", "cancel flight": "cancel_booking",
    "cancel reservation": "cancel_booking",
    # Currency / exchange
    "exchange rate": "compute_exchange_rate", "convert currency": "compute_exchange_rate",
    "currency conversion": "compute_exchange_rate", "exchange": "compute_exchange_rate",
    "convert": "compute_exchange_rate",
    # Cost / price
    "cost": "get_flight_cost", "price": "get_flight_cost",
    "how much": "get_flight_cost", "fare": "get_flight_cost",
    "flight cost": "get_flight_cost", "ticket price": "get_flight_cost",
    # Airport
    "airport": "get_nearest_airport_by_city",
    "nearest airport": "get_nearest_airport_by_city",
    "closest airport": "get_nearest_airport_by_city",
    "find airport": "get_nearest_airport_by_city",
    # Authentication / login
    "login": "authenticate_travel", "sign in": "authenticate_travel",
    "log in": "authenticate_travel", "authenticate": "authenticate_travel",
    # Budget
    "budget": "set_budget_limit", "set budget": "set_budget_limit",
    "budget limit": "set_budget_limit",
    # Invoice / receipt
    "invoice": "retrieve_invoice", "receipt": "retrieve_invoice",
    "retrieve invoice": "retrieve_invoice", "get invoice": "retrieve_invoice",
    "get receipt": "retrieve_invoice",
    # Insurance
    "insurance": "purchase_insurance", "buy insurance": "purchase_insurance",
    "purchase insurance": "purchase_insurance", "travel insurance": "purchase_insurance",
    # Credit card
    "credit card": "register_credit_card", "register card": "register_credit_card",
    "add card": "register_credit_card", "new card": "register_credit_card",
    # Support
    "support": "contact_customer_support", "help": "contact_customer_support",
    "customer support": "contact_customer_support",
    "contact support": "contact_customer_support",
    "inquire": "contact_customer_support",
    # Zipcode
    "zipcode": "get_zipcode_based_on_city", "zip code": "get_zipcode_based_on_city",
    "postal code": "get_zipcode_based_on_city",
    # Distance
    "distance": "estimate_distance", "how far": "estimate_distance",
    "estimate distance": "estimate_distance", "distance between": "estimate_distance",
    # Airline info
    "airline": "get_airline_info", "airline info": "get_airline_info",
    "airline information": "get_airline_info",
    # Booking history
    "booking history": "get_booking_history", "past bookings": "get_booking_history",
    "my bookings": "get_booking_history", "travel history": "get_booking_history",
    # Credit card balance
    "card balance": "get_credit_card_balance", "credit balance": "get_credit_card_balance",
    # Budget fiscal year
    "fiscal year": "get_budget_fiscal_year", "fiscal budget": "get_budget_fiscal_year",
    # All credit cards
    "all cards": "get_all_credit_cards", "list cards": "get_all_credit_cards",
    "my cards": "get_all_credit_cards",
    # List airports
    "list airports": "list_all_airports", "all airports": "list_all_airports",
    "airports": "list_all_airports",
    # Login status
    "login status": "travel_get_login_status", "logged in": "travel_get_login_status",
    "am i logged in": "travel_get_login_status",
    # Verify traveler
    "verify": "verify_traveler_information",
    "verify traveler": "verify_traveler_information",
    "traveler info": "verify_traveler_information",
    "verify information": "verify_traveler_information",
}

# NL noun → canonical target for encoder lexicon matching
TARGET_OVERRIDES = {
    # Flight-related
    "trip": "flight", "journey": "flight", "vacation": "flight",
    "ticket": "flight", "reservation": "flight", "booking": "flight",
    "plane": "flight", "airplane": "flight", "air": "flight",
    "airport": "flight", "airline": "flight", "departure": "flight",
    "arrival": "flight", "itinerary": "flight",
    # Hotel
    "room": "hotel", "accommodation": "hotel", "stay": "hotel",
    "lodging": "hotel", "hostel": "hotel",
    # Order / invoice
    "bill": "order", "receipt": "order", "invoice": "order",
    "payment": "order", "charge": "order",
    # Balance / budget / money
    "money": "balance", "budget": "balance", "funds": "balance",
    "credit": "balance", "card": "balance", "wallet": "balance",
    "fiscal": "balance", "spending": "balance",
    # Route / distance
    "distance": "route", "miles": "route", "kilometers": "route",
    "location": "route", "directions": "route",
    # Currency
    "exchange": "currency", "rate": "currency", "conversion": "currency",
    "dollar": "currency", "euro": "currency", "pound": "currency",
    # Message / support
    "support": "message", "complaint": "message", "inquiry": "message",
    "question": "message", "assistance": "message",
    # Data
    "zipcode": "data", "zip": "data", "postal": "data",
    "login": "data", "authentication": "data", "credentials": "data",
    "insurance": "data", "traveler": "data", "passenger": "data",
}

# Function name → (action, target) for encoder
FUNCTION_INTENTS = {
    "authenticate_travel":       ("start", "data"),
    "book_flight":               ("create", "flight"),
    "book_hotel":                ("create", "hotel"),
    "cancel_booking":            ("delete", "flight"),
    "compute_exchange_rate":     ("calculate", "currency"),
    "contact_customer_support":  ("send", "message"),
    "estimate_distance":         ("calculate", "route"),
    "get_airline_info":          ("get", "flight"),
    "get_all_credit_cards":      ("get", "data"),
    "get_booking_history":       ("get", "flight"),
    "get_budget_fiscal_year":    ("get", "balance"),
    "get_credit_card_balance":   ("get", "balance"),
    "get_flight_cost":           ("get", "flight"),
    "get_nearest_airport_by_city": ("get", "flight"),
    "get_zipcode_based_on_city": ("get", "data"),
    "list_all_airports":         ("list", "flight"),
    "purchase_insurance":        ("create", "data"),
    "register_credit_card":      ("create", "data"),
    "retrieve_invoice":          ("get", "order"),
    "set_budget_limit":          ("set", "balance"),
    "travel_get_login_status":   ("check", "data"),
    "verify_traveler_information": ("check", "data"),
}
