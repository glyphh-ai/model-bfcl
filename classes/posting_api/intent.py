"""Per-class intent extraction for PostingAPI.

Uses phrase/word maps for NL -> action/target extraction.
Maps canonical actions to the 14 PostingAPI functions.
PostingAPI shares the same function signatures as TwitterAPI.

Exports:
    extract_intent(query) -> {action, target, keywords}
    ACTION_TO_FUNC        -- canonical action -> PostingAPI.{func}
    FUNC_TO_ACTION        -- bare func name -> canonical action
    FUNC_TO_TARGET        -- bare func name -> canonical target
"""

from __future__ import annotations

import re

# -- Pack canonical action -> PostingAPI function ---------------------------

ACTION_TO_FUNC: dict[str, str] = {
    "authenticate":       "PostingAPI.authenticate_twitter",
    "post_tweet":         "PostingAPI.post_tweet",
    "comment":            "PostingAPI.comment",
    "retweet":            "PostingAPI.retweet",
    "mention":            "PostingAPI.mention",
    "follow":             "PostingAPI.follow_user",
    "unfollow":           "PostingAPI.unfollow_user",
    "get_tweet":          "PostingAPI.get_tweet",
    "get_tweet_comments": "PostingAPI.get_tweet_comments",
    "get_user_tweets":    "PostingAPI.get_user_tweets",
    "get_user_stats":     "PostingAPI.get_user_stats",
    "list_following":     "PostingAPI.list_all_following",
    "search_tweets":      "PostingAPI.search_tweets",
    "get_login_status":   "PostingAPI.posting_get_login_status",
}

# Reverse: bare function name -> canonical action
FUNC_TO_ACTION: dict[str, str] = {
    "authenticate_twitter":     "authenticate",
    "post_tweet":               "post_tweet",
    "comment":                  "comment",
    "retweet":                  "retweet",
    "mention":                  "mention",
    "follow_user":              "follow",
    "unfollow_user":            "unfollow",
    "get_tweet":                "get_tweet",
    "get_tweet_comments":       "get_tweet_comments",
    "get_user_tweets":          "get_user_tweets",
    "get_user_stats":           "get_user_stats",
    "list_all_following":       "list_following",
    "search_tweets":            "search_tweets",
    "posting_get_login_status": "get_login_status",
}

# Function -> canonical target
FUNC_TO_TARGET: dict[str, str] = {
    "authenticate_twitter":     "account",
    "post_tweet":               "tweet",
    "comment":                  "tweet",
    "retweet":                  "tweet",
    "mention":                  "user",
    "follow_user":              "user",
    "unfollow_user":            "user",
    "get_tweet":                "tweet",
    "get_tweet_comments":       "tweet",
    "get_user_tweets":          "tweet",
    "get_user_stats":           "user",
    "list_all_following":       "user",
    "search_tweets":            "tweet",
    "posting_get_login_status": "account",
}

# -- NL synonym -> canonical action ----------------------------------------
# Ordered: longest/most specific phrases first for greedy matching

_PHRASE_MAP: list[tuple[str, str]] = [
    # authenticate -- credentials / login patterns (must be before post/tweet)
    ("authenticate using", "authenticate"),
    ("log in to twitter", "authenticate"),
    ("sign in to twitter", "authenticate"),
    ("login twitter", "authenticate"),
    ("username and password", "authenticate"),
    ("my username", "authenticate"),
    ("my user name", "authenticate"),
    ("my password", "authenticate"),
    ("using username", "authenticate"),
    ("using my username", "authenticate"),
    ("need to log in", "authenticate"),
    ("if you need to log in", "authenticate"),
    # get_tweet_comments -- must come before comment
    ("retrieve all comments", "get_tweet_comments"),
    ("get the comments", "get_tweet_comments"),
    ("fetch comments", "get_tweet_comments"),
    ("view comments", "get_tweet_comments"),
    ("see the comments", "get_tweet_comments"),
    # comment -- reply patterns
    ("commenting underneath", "comment"),
    ("comment underneath", "comment"),
    ("supportive comment", "comment"),
    ("add a comment", "comment"),
    ("drop a comment", "comment"),
    ("append a thoughtful comment", "comment"),
    ("append a comment", "comment"),
    ("add a supportive", "comment"),
    ("comment on that tweet", "comment"),
    ("comment saying", "comment"),
    ("comment content", "comment"),
    ("comment that tweet", "comment"),
    # retweet -- sharing/amplifying patterns
    ("retweeting it", "retweet"),
    ("retweet it", "retweet"),
    ("retweeting that", "retweet"),
    ("retweet the tweet", "retweet"),
    ("retweeting my", "retweet"),
    ("amplify its reach", "retweet"),
    ("give it a boost", "retweet"),
    ("widen its reach", "retweet"),
    ("widen the circle", "retweet"),
    ("boost by retweeting", "retweet"),
    ("assist in retweeting", "retweet"),
    ("share it with my followers", "retweet"),
    ("maximize visibility", "retweet"),
    ("reach a larger audience", "retweet"),
    # mention -- tagging patterns (must be before post_tweet)
    ("add a mention", "mention"),
    ("tagging @", "mention"),
    ("by tagging", "mention"),
    ("gains more momentum by tagging", "mention"),
    # get_user_stats
    ("get statistics", "get_user_stats"),
    ("user statistics", "get_user_stats"),
    ("user stats", "get_user_stats"),
    ("account stats", "get_user_stats"),
    ("how many followers", "get_user_stats"),
    # get_user_tweets
    ("retrieve all tweets from", "get_user_tweets"),
    ("get all tweets from", "get_user_tweets"),
    ("tweets from user", "get_user_tweets"),
    ("user's tweets", "get_user_tweets"),
    # get_tweet
    ("retrieve a specific tweet", "get_tweet"),
    ("get the tweet", "get_tweet"),
    ("fetch the tweet", "get_tweet"),
    ("retrieve tweet", "get_tweet"),
    # list_following
    ("list all following", "list_following"),
    ("who am i following", "list_following"),
    ("users i follow", "list_following"),
    ("following list", "list_following"),
    ("list of following", "list_following"),
    # search_tweets
    ("search for tweets", "search_tweets"),
    ("search tweets", "search_tweets"),
    ("find tweets", "search_tweets"),
    ("tweets containing", "search_tweets"),
    # get_login_status
    ("login status", "get_login_status"),
    ("logged in", "get_login_status"),
    ("check login", "get_login_status"),
    # post_tweet -- posting/tweeting patterns (broad, goes last)
    ("draft a tweet", "post_tweet"),
    ("craft a tweet", "post_tweet"),
    ("crafting a tweet", "post_tweet"),
    ("compose a tweet", "post_tweet"),
    ("post a tweet", "post_tweet"),
    ("post on twitter", "post_tweet"),
    ("post the tweet", "post_tweet"),
    ("tweet about", "post_tweet"),
    ("share on twitter", "post_tweet"),
    ("share on social", "post_tweet"),
    ("on social media", "post_tweet"),
    ("broadcast on", "post_tweet"),
    ("publish a tweet", "post_tweet"),
    ("posting a summary", "post_tweet"),
    ("send a tweet", "post_tweet"),
    ("toss a tweet", "post_tweet"),
    ("share a tweet", "post_tweet"),
    ("pen a tweet", "post_tweet"),
    ("draft and publish", "post_tweet"),
    ("tweet saying", "post_tweet"),
    ("tweet expressing", "post_tweet"),
    ("tweet that says", "post_tweet"),
    ("tweet announcing", "post_tweet"),
    ("help me tweet", "post_tweet"),
    ("social media update", "post_tweet"),
    ("social media presence", "post_tweet"),
    ("share the result", "post_tweet"),
    ("share the verbatim", "post_tweet"),
    ("body of the post", "post_tweet"),
    ("post an update", "post_tweet"),
    ("share a quick update", "post_tweet"),
    ("share my travel", "post_tweet"),
    ("share the updates", "post_tweet"),
    ("share the tire", "post_tweet"),
    ("share our findings", "post_tweet"),
    ("electrifying tweet", "post_tweet"),
    ("swift update", "post_tweet"),
    ("thank you tweet", "post_tweet"),
    # follow / unfollow
    ("unfollow user", "unfollow"),
    ("stop following", "unfollow"),
    ("follow user", "follow"),
    ("start following", "follow"),
]

_WORD_MAP: dict[str, str] = {
    "authenticate": "authenticate",
    "login": "authenticate",
    "tweet": "post_tweet",
    "post": "post_tweet",
    "publish": "post_tweet",
    "broadcast": "post_tweet",
    "comment": "comment",
    "reply": "comment",
    "respond": "comment",
    "retweet": "retweet",
    "repost": "retweet",
    "amplify": "retweet",
    "mention": "mention",
    "tag": "mention",
    "follow": "follow",
    "subscribe": "follow",
    "unfollow": "unfollow",
    "unsubscribe": "unfollow",
    "search": "search_tweets",
    "stats": "get_user_stats",
    "analytics": "get_user_stats",
    "followers": "get_user_stats",
}

_TARGET_MAP: dict[str, str] = {
    "tweet": "tweet",
    "tweets": "tweet",
    "post": "tweet",
    "posts": "tweet",
    "status": "tweet",
    "message": "tweet",
    "user": "user",
    "users": "user",
    "follower": "user",
    "followers": "user",
    "following": "user",
    "account": "account",
    "profile": "account",
    "login": "account",
    "credential": "account",
    "credentials": "account",
    "password": "account",
    "username": "account",
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
    """Extract posting/social intent from NL query.

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
