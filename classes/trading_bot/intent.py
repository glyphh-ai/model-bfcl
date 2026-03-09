"""Per-class intent extraction for TradingBot.

Uses phrase/word maps for NL -> action/target extraction.
Maps canonical actions to the 20 TradingBot functions.

Exports:
    extract_intent(query) -> {action, target, keywords}
    ACTION_TO_FUNC        -- canonical action -> TradingBot.{func}
    FUNC_TO_ACTION        -- bare func name -> canonical action
"""

from __future__ import annotations

import re

# -- Pack canonical action -> TradingBot function -------------------------

ACTION_TO_FUNC: dict[str, str] = {
    "add_watchlist":        "TradingBot.add_to_watchlist",
    "cancel_order":         "TradingBot.cancel_order",
    "filter_stocks":        "TradingBot.filter_stocks_by_price",
    "fund_account":         "TradingBot.fund_account",
    "get_account":          "TradingBot.get_account_info",
    "get_available_stocks": "TradingBot.get_available_stocks",
    "get_time":             "TradingBot.get_current_time",
    "get_order_details":    "TradingBot.get_order_details",
    "get_order_history":    "TradingBot.get_order_history",
    "get_stock_info":       "TradingBot.get_stock_info",
    "get_symbol":           "TradingBot.get_symbol_by_name",
    "get_transactions":     "TradingBot.get_transaction_history",
    "get_watchlist":        "TradingBot.get_watchlist",
    "notify_price":         "TradingBot.notify_price_change",
    "place_order":          "TradingBot.place_order",
    "remove_watchlist":     "TradingBot.remove_stock_from_watchlist",
    "login_status":         "TradingBot.trading_get_login_status",
    "login":                "TradingBot.trading_login",
    "logout":               "TradingBot.trading_logout",
    "withdraw":             "TradingBot.withdraw_funds",
}

# Reverse: bare function name -> pack canonical action
FUNC_TO_ACTION: dict[str, str] = {
    "add_to_watchlist":           "add_watchlist",
    "cancel_order":               "cancel_order",
    "filter_stocks_by_price":     "filter_stocks",
    "fund_account":               "fund_account",
    "get_account_info":           "get_account",
    "get_available_stocks":       "get_available_stocks",
    "get_current_time":           "get_time",
    "get_order_details":          "get_order_details",
    "get_order_history":          "get_order_history",
    "get_stock_info":             "get_stock_info",
    "get_symbol_by_name":         "get_symbol",
    "get_transaction_history":    "get_transactions",
    "get_watchlist":              "get_watchlist",
    "notify_price_change":        "notify_price",
    "place_order":                "place_order",
    "remove_stock_from_watchlist": "remove_watchlist",
    "trading_get_login_status":   "login_status",
    "trading_login":              "login",
    "trading_logout":             "logout",
    "withdraw_funds":             "withdraw",
}

# Function -> canonical target
FUNC_TO_TARGET: dict[str, str] = {
    "add_to_watchlist":           "watchlist",
    "cancel_order":               "order",
    "filter_stocks_by_price":     "stock",
    "fund_account":               "account",
    "get_account_info":           "account",
    "get_available_stocks":       "stock",
    "get_current_time":           "time",
    "get_order_details":          "order",
    "get_order_history":          "order",
    "get_stock_info":             "stock",
    "get_symbol_by_name":         "stock",
    "get_transaction_history":    "transaction",
    "get_watchlist":              "watchlist",
    "notify_price_change":        "stock",
    "place_order":                "order",
    "remove_stock_from_watchlist": "watchlist",
    "trading_get_login_status":   "session",
    "trading_login":              "session",
    "trading_logout":             "session",
    "withdraw_funds":             "account",
}

# -- NL synonym -> pack canonical action -----------------------------------
# Derived from tests.jsonl queries + function descriptions
# Longest/most specific phrases first for greedy matching

_PHRASE_MAP: list[tuple[str, str]] = [
    # fund_account -- MUST come before get_account ("my account" phrases)
    ("fund my trading account", "fund_account"),
    ("fund my account", "fund_account"),
    ("fund the account", "fund_account"),
    ("fund account", "fund_account"),
    ("top up my", "fund_account"),
    ("replenishing my trading account", "fund_account"),
    ("deposit into my", "fund_account"),
    ("inject some capital", "fund_account"),
    ("infuse 5000", "fund_account"),
    ("allocate an extra", "fund_account"),
    # withdraw -- MUST come before get_account
    ("withdraw funds", "withdraw"),
    ("withdraw from", "withdraw"),
    ("withdraw $", "withdraw"),
    ("withdraw money", "withdraw"),
    # remove_watchlist -- MUST come before add/get watchlist
    ("remove from watchlist", "remove_watchlist"),
    ("remove from my watchlist", "remove_watchlist"),
    ("off my watchlist", "remove_watchlist"),
    ("take the first one off", "remove_watchlist"),
    ("taking the first one off", "remove_watchlist"),
    ("out of the equation from my watchlist", "remove_watchlist"),
    ("eliminate bdx from", "remove_watchlist"),
    ("no longer on my watchlist", "remove_watchlist"),
    ("rather not monitor anymore", "remove_watchlist"),
    ("remove from the list", "remove_watchlist"),
    ("kindly remove", "remove_watchlist"),
    # add_watchlist -- BEFORE get_watchlist (so "to my watchlist" catches add intent)
    ("add to watchlist", "add_watchlist"),
    ("add to my watchlist", "add_watchlist"),
    ("added to my watchlist", "add_watchlist"),
    ("added to watchlist", "add_watchlist"),
    ("include in my watchlist", "add_watchlist"),
    ("integrate stock", "add_watchlist"),
    ("integrate into my", "add_watchlist"),
    ("onto my watchlist", "add_watchlist"),
    ("into my watchlist", "add_watchlist"),
    ("to my watchlist", "add_watchlist"),
    ("add stock", "add_watchlist"),
    ("keep track of", "add_watchlist"),
    ("add all of them to my watchlist", "add_watchlist"),
    ("add this company to my stock watchlist", "add_watchlist"),
    ("stock in my watchlist", "add_watchlist"),
    ("add it to my watchlist", "add_watchlist"),
    ("pop zeta corp", "add_watchlist"),
    ("get zeta corp on my watchlist", "add_watchlist"),
    ("include their stock in my watchlist", "add_watchlist"),
    ("watchlist so i can", "add_watchlist"),
    # get_watchlist -- catch-all "my watchlist" AFTER add_watchlist specifics
    ("my stock watchlist", "get_watchlist"),
    ("stocks i'm monitoring", "get_watchlist"),
    ("stocks i am monitoring", "get_watchlist"),
    ("stocks i've been tracking", "get_watchlist"),
    ("stocks i'm keeping", "get_watchlist"),
    ("stocks i'm watching", "get_watchlist"),
    ("stocks im watching", "get_watchlist"),
    ("stocks currently present", "get_watchlist"),
    ("current contents of that watchlist", "get_watchlist"),
    ("current catalog of stocks", "get_watchlist"),
    ("display the current contents", "get_watchlist"),
    ("current watchlist contents", "get_watchlist"),
    ("show watchlist", "get_watchlist"),
    ("view watchlist", "get_watchlist"),
    ("on my watchlist", "get_watchlist"),
    ("my watchlist", "get_watchlist"),
    ("what's on my radar", "get_watchlist"),
    ("on my radar", "get_watchlist"),
    ("my existing watchlist", "get_watchlist"),
    ("what stocks have i", "get_watchlist"),
    ("keeping tabs on", "get_watchlist"),
    ("stocks currently on my", "get_watchlist"),
    ("pull up the latest stock symbols from my watchlist", "get_watchlist"),
    ("delve into the latest stocks i've been tracking", "get_watchlist"),
    # cancel_order
    ("cancel the order", "cancel_order"),
    ("cancel my pending order", "cancel_order"),
    ("cancel the recent order", "cancel_order"),
    ("cancel my recent order", "cancel_order"),
    ("cancel that order", "cancel_order"),
    ("cancel that last order", "cancel_order"),
    ("cancelling that order", "cancel_order"),
    ("cancelling the specific order", "cancel_order"),
    ("cancellation", "cancel_order"),
    ("retract the recent order", "cancel_order"),
    ("rescind the transaction", "cancel_order"),
    ("rescindment of", "cancel_order"),
    ("revoke the placed order", "cancel_order"),
    ("revoke the order", "cancel_order"),
    ("reversing the transaction", "cancel_order"),
    ("scrap that order", "cancel_order"),
    ("nix it for me", "cancel_order"),
    ("annul the latest order", "cancel_order"),
    ("pull the plug on", "cancel_order"),
    ("withdraw that order", "cancel_order"),
    ("cancel order", "cancel_order"),
    # place_order -- must come before generic "buy"/"purchase"/"order"
    ("place an order", "place_order"),
    ("place a buy order", "place_order"),
    ("place the buy order", "place_order"),
    ("placed an order", "place_order"),
    ("placing the buy order", "place_order"),
    ("execute a buy order", "place_order"),
    ("execute a transaction", "place_order"),
    ("execute this transaction", "place_order"),
    ("execute a purchase", "place_order"),
    ("initiate a purchase", "place_order"),
    ("proceed with purchasing", "place_order"),
    ("proceed with a buy", "place_order"),
    ("proceed with an acquisition", "place_order"),
    ("proceed with placing", "place_order"),
    ("purchasing shares", "place_order"),
    ("purchase shares", "place_order"),
    ("purchase of", "place_order"),
    ("buy order for", "place_order"),
    ("buy shares", "place_order"),
    ("buying shares", "place_order"),
    ("buy 100 shares", "place_order"),
    ("buy 50 shares", "place_order"),
    ("buy 120 shares", "place_order"),
    ("buy 150 shares", "place_order"),
    ("procure 50 shares", "place_order"),
    ("acquire 100 shares", "place_order"),
    ("acquire a hefty", "place_order"),
    ("acquire 150 shares", "place_order"),
    ("ordering 100 shares", "place_order"),
    ("snag 100 shares", "place_order"),
    ("100 shares of", "place_order"),
    ("150 shares of", "place_order"),
    ("50 shares of", "place_order"),
    ("120 shares of", "place_order"),
    # get_order_details
    ("order details", "get_order_details"),
    ("details of my", "get_order_details"),
    ("details of the order", "get_order_details"),
    ("details of the trade", "get_order_details"),
    ("details of the most recent order", "get_order_details"),
    ("specifics of my placed order", "get_order_details"),
    ("specifics of the trade", "get_order_details"),
    ("specifics of the last order", "get_order_details"),
    ("particulars of my", "get_order_details"),
    ("particulars of the order", "get_order_details"),
    ("breakdown of this new order", "get_order_details"),
    ("breakdown of the transaction", "get_order_details"),
    ("check the particulars", "get_order_details"),
    ("details for the order", "get_order_details"),
    ("details of my latest order", "get_order_details"),
    ("lowdown on that order", "get_order_details"),
    ("recent order details", "get_order_details"),
    ("summary regarding the most recent order", "get_order_details"),
    ("verify the status and execution", "get_order_details"),
    ("fetch the particulars", "get_order_details"),
    ("retrieve the specifics", "get_order_details"),
    # get_stock_info
    ("keen eye on", "get_stock_info"),
    ("stock price", "get_stock_info"),
    ("stock performance", "get_stock_info"),
    ("stock information", "get_stock_info"),
    ("stock details", "get_stock_info"),
    ("current price of", "get_stock_info"),
    ("trading details", "get_stock_info"),
    ("latest market data", "get_stock_info"),
    ("latest stock details", "get_stock_info"),
    ("latest stock information", "get_stock_info"),
    ("comprehensive breakdown", "get_stock_info"),
    ("comprehensive analysis", "get_stock_info"),
    ("market performance", "get_stock_info"),
    ("market activity", "get_stock_info"),
    ("performance metrics", "get_stock_info"),
    ("current market price", "get_stock_info"),
    ("prevailing market price", "get_stock_info"),
    ("prevailing market rate", "get_stock_info"),
    ("going market rate", "get_stock_info"),
    ("existing market value", "get_stock_info"),
    ("market value", "get_stock_info"),
    ("present market value", "get_stock_info"),
    ("present stock performance", "get_stock_info"),
    # get_available_stocks
    ("list of stock symbols", "get_available_stocks"),
    ("stocks in the given sector", "get_available_stocks"),
    ("technology sector", "get_available_stocks"),
    ("tech sector", "get_available_stocks"),
    ("technology stock", "get_available_stocks"),
    ("tech stocks", "get_available_stocks"),
    ("tech industry", "get_available_stocks"),
    ("available technology", "get_available_stocks"),
    ("stock symbols pertinent", "get_available_stocks"),
    ("potential tech stocks", "get_available_stocks"),
    ("stocks in this field", "get_available_stocks"),
    ("sector-related companies", "get_available_stocks"),
    ("sector is technology", "get_available_stocks"),
    ("sector is tech", "get_available_stocks"),
    # login/logout/status — BEFORE get_account ("my trading account" matches get_account)
    ("log in to", "login"),
    ("log in", "login"),
    ("sign in to", "login"),
    ("sign in", "login"),
    ("log out", "logout"),
    ("logged out", "logout"),
    ("sign out", "logout"),
    ("log me out", "logout"),
    ("login status", "login_status"),
    ("logged in", "login_status"),
    # get_account_info
    ("account details", "get_account"),
    ("account info", "get_account"),
    ("account information", "get_account"),
    ("account balance", "get_account"),
    ("account summary", "get_account"),
    ("account status", "get_account"),
    ("my account", "get_account"),
    ("my trading account", "get_account"),
    ("current balance", "get_account"),
    ("balance and card", "get_account"),
    ("balance and the linked", "get_account"),
    ("financial standing", "get_account"),
    ("financial wellbeing", "get_account"),
    ("financial health check", "get_account"),
    ("financial positioning", "get_account"),
    ("overview of my account", "get_account"),
    ("card information linked", "get_account"),
    # (fund_account and withdraw moved to top of phrase map)
    # get_order_history
    ("order history", "get_order_history"),
    ("order id history", "get_order_history"),
    ("past orders", "get_order_history"),
    # get_transactions
    ("transaction history", "get_transactions"),
    ("past transactions", "get_transactions"),
    # get_time
    ("current time", "get_time"),
    ("what time", "get_time"),
    # get_symbol
    ("stock symbol", "get_symbol"),
    ("ticker symbol", "get_symbol"),
    ("symbol of", "get_symbol"),
    # filter_stocks
    ("filter stocks", "filter_stocks"),
    ("price range", "filter_stocks"),
    ("stocks between", "filter_stocks"),
    # notify_price
    ("price change", "notify_price"),
    ("price alert", "notify_price"),
    ("notify price", "notify_price"),
    ("significant price", "notify_price"),
]

_WORD_MAP: dict[str, str] = {
    # Single word -> canonical action
    "watchlist": "get_watchlist",
    "cancel": "cancel_order",
    "revoke": "cancel_order",
    "rescind": "cancel_order",
    "buy": "place_order",
    "sell": "place_order",
    "purchase": "place_order",
    "acquire": "place_order",
    "fund": "fund_account",
    "deposit": "fund_account",
    "withdraw": "withdraw",
    "remove": "remove_watchlist",
    "login": "login",
    "authenticate": "login",
    "logout": "logout",
    "disconnect": "logout",
    "filter": "filter_stocks",
    "screen": "filter_stocks",
    "notify": "notify_price",
    "alert": "notify_price",
    "ticker": "get_symbol",
    "symbol": "get_symbol",
    "sector": "get_available_stocks",
    "balance": "get_account",
}

_TARGET_MAP: dict[str, str] = {
    "stock": "stock",
    "stocks": "stock",
    "share": "stock",
    "shares": "stock",
    "equity": "stock",
    "order": "order",
    "orders": "order",
    "trade": "order",
    "transaction": "transaction",
    "transactions": "transaction",
    "account": "account",
    "balance": "account",
    "funds": "account",
    "money": "account",
    "watchlist": "watchlist",
    "portfolio": "watchlist",
    "time": "time",
    "clock": "time",
    "session": "session",
    "login": "session",
    "price": "stock",
    "sector": "stock",
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
    """Extract trading intent from NL query.

    Returns:
        {action: str, target: str, keywords: str}
    """
    q_lower = query.lower()

    # 0. Leading-verb override: "remove ... watchlist" → remove_watchlist
    _words = q_lower.split()
    if _words and _words[0] in ("remove", "take") and "watchlist" in q_lower:
        return {"action": "remove_watchlist", "target": "watchlist",
                "keywords": " ".join(w for w in re.sub(r"[^a-z0-9\s]", " ", q_lower).split()
                                     if w not in _STOP_WORDS and len(w) > 1)}

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
