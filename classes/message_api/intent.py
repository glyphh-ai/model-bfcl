"""Per-class intent overrides for MessageAPI."""

CLASS_ALIASES = ["MessageAPI", "message", "messaging"]
CLASS_DOMAIN = "messaging"

ACTION_SYNONYMS = {
    "dispatch": "send_message", "notify": "send_message",
    "check": "view_messages_received", "inbox": "view_messages_received",
}

TARGET_OVERRIDES = {
    "msg": "message", "text": "message", "dm": "message",
}
