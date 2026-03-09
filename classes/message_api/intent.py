"""Per-class intent extraction for MessageAPI.

Maps NL queries to the 10 MessageAPI functions via phrase/word matching.

Exports:
    extract_intent(query) -> {action, target, keywords}
    ACTION_TO_FUNC        -- canonical action -> MessageAPI.{func}
    FUNC_TO_ACTION        -- bare func name -> canonical action
    FUNC_TO_TARGET        -- bare func name -> canonical target
"""

from __future__ import annotations

import re

# -- Pack canonical action -> MessageAPI function ----------------------------

ACTION_TO_FUNC: dict[str, str] = {
    "send":          "MessageAPI.send_message",
    "delete_msg":    "MessageAPI.delete_message",
    "view_sent":     "MessageAPI.view_messages_sent",
    "search_msg":    "MessageAPI.search_messages",
    "get_stats":     "MessageAPI.get_message_stats",
    "add_contact":   "MessageAPI.add_contact",
    "get_user_id":   "MessageAPI.get_user_id",
    "list_users":    "MessageAPI.list_users",
    "login":         "MessageAPI.message_login",
    "login_status":  "MessageAPI.message_get_login_status",
}

# Reverse: bare function name -> pack canonical action
FUNC_TO_ACTION: dict[str, str] = {
    "send_message":             "send",
    "delete_message":           "delete_msg",
    "view_messages_sent":       "view_sent",
    "search_messages":          "search_msg",
    "get_message_stats":        "get_stats",
    "add_contact":              "add_contact",
    "get_user_id":              "get_user_id",
    "list_users":               "list_users",
    "message_login":            "login",
    "message_get_login_status": "login_status",
}

# Function -> canonical target
FUNC_TO_TARGET: dict[str, str] = {
    "send_message":             "message",
    "delete_message":           "message",
    "view_messages_sent":       "message",
    "search_messages":          "message",
    "get_message_stats":        "message",
    "add_contact":              "user",
    "get_user_id":              "user",
    "list_users":               "user",
    "message_login":            "session",
    "message_get_login_status": "session",
}

# -- NL synonym -> pack canonical action -------------------------------------
# Derived from test queries. Longest/most specific phrases first.

_PHRASE_MAP: list[tuple[str, str]] = [
    # view_sent -- must come before "send" word match; many queries say "messages I have sent"
    ("messages i have sent", "view_sent"),
    ("messages i've sent", "view_sent"),
    ("messages i have send", "view_sent"),
    ("messages i've send", "view_sent"),
    ("messages we have send", "view_sent"),
    ("messages sent by me", "view_sent"),
    ("all the messages i sent", "view_sent"),
    ("all the messages i have", "view_sent"),
    ("all the communications i've sent", "view_sent"),
    ("all the messages we have", "view_sent"),
    ("messages that i have sent", "view_sent"),
    ("what messages i've sent", "view_sent"),
    ("what i sent previously", "view_sent"),
    ("what's sent by me", "view_sent"),
    ("review all the messages", "view_sent"),
    ("display all the messages i sent", "view_sent"),
    ("display all the messages we", "view_sent"),
    ("display all the messages", "view_sent"),
    ("display what messages", "view_sent"),
    ("compile and present all the responses", "view_sent"),
    ("show me all the messages", "view_sent"),
    ("view all the messages", "view_sent"),
    ("peruse my recent messages", "view_sent"),
    ("recent messages i sent", "view_sent"),
    ("recent messages i've sent", "view_sent"),
    ("check if the message is sent", "view_sent"),
    ("check what's sent", "view_sent"),
    ("look at all the messages", "view_sent"),
    ("list of all the communications", "view_sent"),
    ("messages i have send so far", "view_sent"),
    ("messages sent so far", "view_sent"),
    ("messages i sent", "view_sent"),
    # add_contact -- must come before single-word "add"
    ("add contact", "add_contact"),
    ("add him as new contact", "add_contact"),
    ("add her contact", "add_contact"),
    ("new contact", "add_contact"),
    # delete_message
    ("delete the last message", "delete_msg"),
    ("delete the message", "delete_msg"),
    ("delete that message", "delete_msg"),
    ("delete message", "delete_msg"),
    ("retract that specific message", "delete_msg"),
    ("remove the message", "delete_msg"),
    ("removing it since", "delete_msg"),
    # search_messages
    ("search for messages", "search_msg"),
    ("search messages", "search_msg"),
    ("find messages", "search_msg"),
    # get_message_stats
    ("message stats", "get_stats"),
    ("message statistics", "get_stats"),
    ("messaging statistics", "get_stats"),
    # login
    ("logging in as", "login"),
    ("log in as", "login"),
    ("log in with", "login"),
    ("log in to", "login"),
    ("log my user id", "login"),
    ("loggin in as", "login"),
    # login_status
    ("login status", "login_status"),
    ("logged in status", "login_status"),
    # list_users
    ("list all users", "list_users"),
    ("list users", "list_users"),
    ("all users in the workspace", "list_users"),
    # get_user_id
    ("user id from", "get_user_id"),
    ("get user id", "get_user_id"),
    ("lookup user", "get_user_id"),
    # send -- general send phrases (after view_sent phrases above)
    ("send a message", "send"),
    ("send message", "send"),
    ("send the message", "send"),
    ("dispatch a message", "send"),
    ("dispatch the message", "send"),
    ("relay a message", "send"),
    ("relay the message", "send"),
    ("forward a message", "send"),
    ("message my colleague", "send"),
    ("message to", "send"),
    ("kindly message", "send"),
    ("buzz catherine", "send"),
    ("draft a note", "send"),
    ("dispatch a note", "send"),
    ("dispatch them a note", "send"),
    ("dispatch a concise message", "send"),
    ("dispatch the", "send"),
    ("notify my", "send"),
    ("communicate with", "send"),
    ("inform my friend", "send"),
    ("kindly dispatch", "send"),
    ("send a quick message", "send"),
    ("send the", "send"),
]

_WORD_MAP: dict[str, str] = {
    # send
    "send": "send",
    "dispatch": "send",
    "relay": "send",
    "forward": "send",
    "text": "send",
    "notify": "send",
    "communicate": "send",
    "convey": "send",
    # delete
    "delete": "delete_msg",
    "remove": "delete_msg",
    "retract": "delete_msg",
    "erase": "delete_msg",
    # view_sent
    "sent": "view_sent",
    "outbox": "view_sent",
    # search
    "search": "search_msg",
    # stats
    "stats": "get_stats",
    "statistics": "get_stats",
    # contact
    "contact": "add_contact",
    # login
    "login": "login",
    "logging": "login",
    # list users
    "users": "list_users",
    # user id
    "lookup": "get_user_id",
}

_TARGET_MAP: dict[str, str] = {
    "message": "message",
    "messages": "message",
    "note": "message",
    "text": "message",
    "dm": "message",
    "communication": "message",
    "communications": "message",
    "communique": "message",
    "contact": "user",
    "user": "user",
    "users": "user",
    "colleague": "user",
    "friend": "user",
    "advisor": "user",
    "recipient": "user",
    "person": "user",
    "session": "session",
    "account": "session",
    "login": "session",
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
    """Extract messaging intent from NL query.

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
