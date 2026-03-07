"""Per-class intent overrides for TicketAPI."""

CLASS_ALIASES = ["TicketAPI", "ticket", "support ticket"]
CLASS_DOMAIN = "support"

ACTION_SYNONYMS = {
    "open": "create_ticket", "file": "create_ticket",
    "resolve": "resolve_ticket", "mark": "resolve_ticket",
    "close": "close_ticket", "shut": "close_ticket",
    "modify": "edit_ticket", "update": "edit_ticket",
}

TARGET_OVERRIDES = {
    "issue": "ticket", "case": "ticket", "request": "ticket",
}
