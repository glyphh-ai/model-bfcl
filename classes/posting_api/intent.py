"""Per-class intent overrides for PostingAPI."""

CLASS_ALIASES = ["PostingAPI", "posting", "post"]
CLASS_DOMAIN = "social"

ACTION_SYNONYMS = {
    "publish": "post", "create": "post",
    "reply": "comment", "respond": "comment",
    "repost": "share", "forward": "share",
}

TARGET_OVERRIDES = {
    "article": "post", "update": "post",
}
