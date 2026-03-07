"""Per-class intent overrides for TwitterAPI."""

CLASS_ALIASES = ["TwitterAPI", "twitter", "tweet"]
CLASS_DOMAIN = "social"

ACTION_SYNONYMS = {
    "broadcast": "post_tweet", "share": "retweet", "amplify": "retweet",
    "reply": "comment", "tag": "mention",
}

TARGET_OVERRIDES = {
    "status": "tweet", "update": "tweet",
}
