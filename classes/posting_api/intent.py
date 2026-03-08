"""Per-class intent overrides for PostingAPI.

Maps NL verbs/targets to the 14 posting/social functions.
Used by the per-class encoder for Stage 2 routing.
"""

CLASS_ALIASES = ["PostingAPI", "posting", "post", "social"]
CLASS_DOMAIN = "social"

# NL verb → canonical function name (bare, without class prefix)
ACTION_SYNONYMS = {
    # Posting
    "tweet": "post", "publish": "post", "compose": "post",
    # Replying / commenting
    "reply": "comment", "respond": "comment", "comment": "comment",
    # Sharing
    "repost": "share", "retweet": "share", "share": "share",
    # Following
    "follow": "follow_user", "subscribe": "follow_user",
    # Unfollowing
    "unfollow": "unfollow_user", "unsubscribe": "unfollow_user",
    # Mentioning
    "mention": "mention", "tag": "mention", "@": "mention",
    # Searching
    "search": "search_tweets", "find tweet": "search_tweets",
    "look up": "search_tweets",
    # Authentication
    "login": "authenticate_twitter", "sign in": "authenticate_twitter",
    "authenticate": "authenticate_twitter",
    # Stats / analytics
    "stats": "get_user_stats", "analytics": "get_user_stats",
    "followers": "get_user_stats",
}

# NL noun → canonical target for encoder lexicon matching
TARGET_OVERRIDES = {
    "tweet": "post", "tweets": "post", "status": "post", "update": "post",
    "follower": "user", "following": "user", "account": "user", "profile": "user",
    "comments": "post", "replies": "post", "thread": "post",
}

# Function name → (action, target) for encoder
FUNCTION_INTENTS = {
    "authenticate_twitter":     ("start", "data"),
    "comment":                  ("send", "post"),
    "follow_user":              ("add", "user"),
    "get_tweet":                ("get", "post"),
    "get_tweet_comments":       ("get", "post"),
    "get_user_stats":           ("get", "user"),
    "get_user_tweets":          ("get", "post"),
    "list_all_following":       ("list", "user"),
    "mention":                  ("send", "user"),
    "post":                     ("send", "post"),
    "posting_get_login_status": ("check", "data"),
    "share":                    ("send", "post"),
    "search_tweets":            ("search", "post"),
    "unfollow_user":            ("remove", "user"),
}
