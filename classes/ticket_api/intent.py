"""Per-class intent overrides for TicketAPI.

Maps NL verbs/targets to the 9 ticket support functions.
Used by the per-class encoder for Stage 2 routing.
"""

CLASS_ALIASES = ["TicketAPI", "ticket", "support ticket", "helpdesk"]
CLASS_DOMAIN = "support"

# NL verb → canonical function name (bare, without class prefix)
ACTION_SYNONYMS = {
    # Create ticket
    "open": "create_ticket", "file": "create_ticket", "submit": "create_ticket",
    "raise": "create_ticket",
    # Resolve ticket
    "resolve": "resolve_ticket", "fix": "resolve_ticket",
    "mark resolved": "resolve_ticket",
    # Close ticket
    "close": "close_ticket", "shut": "close_ticket",
    # Edit ticket
    "modify": "edit_ticket", "update": "edit_ticket", "change": "edit_ticket",
    # View ticket
    "view": "get_ticket", "check ticket": "get_ticket",
    # Login / logout
    "login": "ticket_login", "sign in": "ticket_login",
    "logout": "logout",
    # List tickets
    "my tickets": "get_user_tickets", "list tickets": "get_user_tickets",
}

# NL noun → canonical target for encoder lexicon matching
TARGET_OVERRIDES = {
    "issue": "ticket", "case": "ticket", "request": "ticket",
    "bug": "ticket", "problem": "ticket", "report": "ticket",
}

# Function name → (action, target) for encoder
FUNCTION_INTENTS = {
    "close_ticket":             ("update", "ticket"),
    "create_ticket":            ("create", "ticket"),
    "edit_ticket":              ("update", "ticket"),
    "get_ticket":               ("get", "ticket"),
    "get_user_tickets":         ("list", "ticket"),
    "logout":                   ("stop", "data"),
    "resolve_ticket":           ("update", "ticket"),
    "ticket_get_login_status":  ("check", "data"),
    "ticket_login":             ("start", "data"),
}
