"""Per-class intent overrides for MessageAPI.

Maps NL verbs/targets to the 11 message API functions.
Used by the per-class encoder for Stage 2 routing.
"""

CLASS_ALIASES = ["MessageAPI", "message", "messaging", "dm", "chat"]
CLASS_DOMAIN = "messaging"

# NL verb → canonical function name (bare, without class prefix)
ACTION_SYNONYMS = {
    # Send / compose
    "send": "send_message", "text": "send_message", "dm": "send_message",
    "write": "send_message",
    # Read / inbox
    "read": "view_messages_received", "inbox": "view_messages_received",
    "received": "view_messages_received",
    # Sent / outbox
    "sent": "view_messages_sent", "outbox": "view_messages_sent",
    # Delete
    "delete": "delete_message", "remove": "delete_message",
    "erase": "delete_message",
    # Search
    "search": "search_messages", "find": "search_messages",
    "look up": "search_messages",
    # Stats
    "stats": "get_message_stats", "statistics": "get_message_stats",
    "count": "get_message_stats",
    # Contact
    "contact": "add_contact", "add contact": "add_contact",
    # User lookup
    "user id": "get_user_id", "lookup user": "get_user_id",
    # Login
    "login": "message_login", "sign in": "message_login",
    "log in": "message_login",
    # List users
    "list users": "list_users", "users": "list_users",
}

# NL noun → canonical target for encoder lexicon matching
TARGET_OVERRIDES = {
    "text": "message", "dm": "message", "chat": "message", "note": "message",
    "contact": "user", "person": "user", "friend": "user", "recipient": "user",
}

# Function name → (action, target) for encoder
FUNCTION_INTENTS = {
    "add_contact":              ("add", "user"),
    "delete_message":           ("delete", "message"),
    "get_message_stats":        ("get", "message"),
    "get_user_id":              ("get", "user"),
    "list_users":               ("list", "user"),
    "message_get_login_status": ("check", "data"),
    "message_login":            ("start", "data"),
    "search_messages":          ("search", "message"),
    "send_message":             ("send", "message"),
    "view_messages_received":   ("get", "message"),
    "view_messages_sent":       ("get", "message"),
}
