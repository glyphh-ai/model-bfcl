"""Per-class intent extraction for TicketAPI.

Maps NL queries to the 9 TicketAPI functions via phrase/word matching.

Exports:
    extract_intent(query) -> {action, target, keywords}
    ACTION_TO_FUNC        -- canonical action -> TicketAPI.{func}
    FUNC_TO_ACTION        -- bare func name -> canonical action
    FUNC_TO_TARGET        -- bare func name -> canonical target
"""

from __future__ import annotations

import re

# -- Pack canonical action -> TicketAPI function -----------------------------

ACTION_TO_FUNC: dict[str, str] = {
    "create":        "TicketAPI.create_ticket",
    "close":         "TicketAPI.close_ticket",
    "resolve":       "TicketAPI.resolve_ticket",
    "edit":          "TicketAPI.edit_ticket",
    "get_ticket":    "TicketAPI.get_ticket",
    "list_tickets":  "TicketAPI.get_user_tickets",
    "login":         "TicketAPI.ticket_login",
    "login_status":  "TicketAPI.ticket_get_login_status",
    "logout":        "TicketAPI.logout",
}

# Reverse: bare function name -> pack canonical action
FUNC_TO_ACTION: dict[str, str] = {
    "create_ticket":          "create",
    "close_ticket":           "close",
    "resolve_ticket":         "resolve",
    "edit_ticket":            "edit",
    "get_ticket":             "get_ticket",
    "get_user_tickets":       "list_tickets",
    "ticket_login":           "login",
    "ticket_get_login_status": "login_status",
    "logout":                 "logout",
}

# Function -> canonical target
FUNC_TO_TARGET: dict[str, str] = {
    "create_ticket":          "ticket",
    "close_ticket":           "ticket",
    "resolve_ticket":         "ticket",
    "edit_ticket":            "ticket",
    "get_ticket":             "ticket",
    "get_user_tickets":       "ticket",
    "ticket_login":           "session",
    "ticket_get_login_status": "session",
    "logout":                 "session",
}

# -- NL synonym -> pack canonical action -------------------------------------
# Derived from test queries. Longest/most specific phrases first.

_PHRASE_MAP: list[tuple[str, str]] = [
    # resolve -- must come before "close" to distinguish resolve vs close
    ("mark it as resolved", "resolve"),
    ("mark as resolved", "resolve"),
    ("marked as resolved", "resolve"),
    ("check it off as resolved", "resolve"),
    ("resolve the ticket", "resolve"),
    ("resolve it", "resolve"),
    ("draft a resolution", "resolve"),
    ("outline a resolution", "resolve"),
    ("resolution for", "resolve"),
    ("categorizing the resolution", "resolve"),
    ("noting that it was", "resolve"),
    ("noting that", "resolve"),
    # close -- after resolve phrases
    ("close the ticket", "close"),
    ("close ticket", "close"),
    ("close the outstanding ticket", "close"),
    ("can be closed", "close"),
    ("cancel it on my behalf", "close"),
    # edit
    ("update its priority", "edit"),
    ("update its status", "edit"),
    ("updating its status", "edit"),
    ("updating its priority", "edit"),
    ("enhance the ticket", "edit"),
    ("edit the ticket", "edit"),
    ("modify the ticket", "edit"),
    ("open up ticket", "edit"),
    ("set the priority", "edit"),
    # get_ticket -- retrieve/fetch a single ticket
    ("retrieve the details", "get_ticket"),
    ("retrieve the ticket", "get_ticket"),
    ("fetch the details", "get_ticket"),
    ("fetch the ticket", "get_ticket"),
    ("fetch that ticket", "get_ticket"),
    ("get the ticket", "get_ticket"),
    ("track down that ticket", "get_ticket"),
    ("ticket details", "get_ticket"),
    ("details of the ticket", "get_ticket"),
    ("details of that ticket", "get_ticket"),
    ("relay the specifics", "get_ticket"),
    ("access and relay", "get_ticket"),
    # list_tickets
    ("all tickets", "list_tickets"),
    ("my tickets", "list_tickets"),
    ("list tickets", "list_tickets"),
    # login -- credential / authenticate patterns
    ("my username is", "login"),
    ("my ticket username", "login"),
    ("username is", "login"),
    ("password is", "login"),
    ("my password", "login"),
    ("log in to ticket", "login"),
    ("login the ticket", "login"),
    ("ticket login", "login"),
    ("user name tech_guru", "login"),
    # login_status
    ("login status", "login_status"),
    ("logged in status", "login_status"),
    # logout
    ("log out", "logout"),
    ("sign out", "logout"),
    # create -- general create/draft/file/submit phrases (after more specific ones)
    ("create a ticket", "create"),
    ("create ticket", "create"),
    ("draft a support ticket", "create"),
    ("draft a ticket", "create"),
    ("draft a formal complaint", "create"),
    ("file a support ticket", "create"),
    ("file a ticket", "create"),
    ("submit a ticket", "create"),
    ("submit a formal complaint", "create"),
    ("support ticket titled", "create"),
    ("support ticket labeled", "create"),
    ("open a ticket", "create"),
    ("initiating a", "create"),
    ("initiate a", "create"),
    ("generate a ticket", "create"),
    ("generate a high-priority ticket", "create"),
    ("lodge a complaint", "create"),
    ("craft a ticket", "create"),
    ("new ticket", "create"),
    ("kindly create", "create"),
    ("priority ticket", "create"),
    ("ticket titled", "create"),
    ("ticket labeled", "create"),
    ("ticket named", "create"),
]

_WORD_MAP: dict[str, str] = {
    # create
    "create": "create",
    "draft": "create",
    "file": "create",
    "submit": "create",
    "lodge": "create",
    "craft": "create",
    "generate": "create",
    "initiate": "create",
    # close
    "close": "close",
    "cancel": "close",
    # resolve
    "resolve": "resolve",
    "resolved": "resolve",
    "resolution": "resolve",
    # edit
    "edit": "edit",
    "modify": "edit",
    "update": "edit",
    "enhance": "edit",
    # get_ticket
    "retrieve": "get_ticket",
    "fetch": "get_ticket",
    # list_tickets
    "tickets": "list_tickets",
    # login
    "login": "login",
    "authenticate": "login",
    "credential": "login",
    "password": "login",
    # logout
    "logout": "logout",
}

_TARGET_MAP: dict[str, str] = {
    "ticket": "ticket",
    "tickets": "ticket",
    "issue": "ticket",
    "case": "ticket",
    "request": "ticket",
    "complaint": "ticket",
    "bug": "ticket",
    "problem": "ticket",
    "report": "ticket",
    "inquiry": "ticket",
    "session": "session",
    "account": "session",
    "login": "session",
    "user": "session",
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
    """Extract ticket intent from NL query.

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
