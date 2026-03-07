"""Per-class intent overrides for TravelAPI."""

CLASS_ALIASES = ["TravelAPI", "TravelBookingAPI", "travel", "booking", "flight"]
CLASS_DOMAIN = "travel"

ACTION_SYNONYMS = {
    "reserve": "book_flight", "book": "book_flight",
    "cancel": "cancel_booking", "abort": "cancel_booking",
    "inquire": "contact_customer_support",
}

TARGET_OVERRIDES = {
    "trip": "flight", "journey": "flight", "vacation": "flight",
    "ticket": "flight", "reservation": "flight",
    "bill": "invoice", "receipt": "invoice",
}
