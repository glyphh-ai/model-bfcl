"""Per-class intent overrides for TradingBot.

Maps NL verbs/targets to the 22 trading bot functions.
Used by the per-class encoder for Stage 2 routing.
"""

CLASS_ALIASES = ["TradingBot", "trading", "stock trading", "trade"]
CLASS_DOMAIN = "trading"

# NL verb → canonical function name (bare, without class prefix)
ACTION_SYNONYMS = {
    # Placing orders (buy/sell)
    "buy": "place_order", "sell": "place_order", "purchase": "place_order",
    "execute order": "place_order", "submit order": "place_order",
    "make order": "place_order", "create order": "place_order",
    "long": "place_order", "short": "place_order",
    # Watchlist add
    "watch": "add_to_watchlist", "track": "add_to_watchlist",
    "monitor": "add_to_watchlist", "follow": "add_to_watchlist",
    "add to watchlist": "add_to_watchlist", "add stock": "add_to_watchlist",
    # Watchlist remove
    "unwatch": "remove_stock_from_watchlist", "untrack": "remove_stock_from_watchlist",
    "remove from watchlist": "remove_stock_from_watchlist",
    "stop watching": "remove_stock_from_watchlist",
    "stop tracking": "remove_stock_from_watchlist",
    # Cancel order
    "cancel": "cancel_order", "revoke": "cancel_order",
    "liquidate": "cancel_order", "abort": "cancel_order",
    "cancel order": "cancel_order", "void": "cancel_order",
    # Fund account
    "deposit": "fund_account", "fund": "fund_account",
    "add money": "fund_account", "replenish": "fund_account",
    "top up": "fund_account", "add funds": "fund_account",
    "load": "fund_account", "transfer funds": "fund_account",
    # Login / logout
    "login": "trading_login", "log in": "trading_login",
    "sign in": "trading_login", "authenticate": "trading_login",
    "logout": "trading_logout", "log out": "trading_logout",
    "sign out": "trading_logout", "disconnect": "trading_logout",
    # Login status
    "login status": "trading_get_login_status",
    "am i logged in": "trading_get_login_status",
    "session status": "trading_get_login_status",
    # Stock info / quote
    "quote": "get_stock_info", "stock info": "get_stock_info",
    "stock price": "get_stock_info", "lookup stock": "get_stock_info",
    "stock details": "get_stock_info", "price of": "get_stock_info",
    # Symbol lookup
    "symbol": "get_symbol_by_name", "ticker": "get_symbol_by_name",
    "find symbol": "get_symbol_by_name", "symbol for": "get_symbol_by_name",
    "ticker symbol": "get_symbol_by_name", "stock symbol": "get_symbol_by_name",
    # Available stocks
    "list stocks": "get_available_stocks", "available stocks": "get_available_stocks",
    "show stocks": "get_available_stocks", "all stocks": "get_available_stocks",
    "browse stocks": "get_available_stocks",
    # Filter stocks
    "filter": "filter_stocks_by_price", "filter stocks": "filter_stocks_by_price",
    "screen": "filter_stocks_by_price", "scan": "filter_stocks_by_price",
    "stocks under": "filter_stocks_by_price", "stocks above": "filter_stocks_by_price",
    "price range": "filter_stocks_by_price", "stocks between": "filter_stocks_by_price",
    # Order details / history
    "order details": "get_order_details", "order status": "get_order_details",
    "check order": "get_order_details", "order info": "get_order_details",
    "order history": "get_order_history", "past orders": "get_order_history",
    "previous orders": "get_order_history", "my orders": "get_order_history",
    # Transaction history
    "transaction history": "get_transaction_history",
    "transactions": "get_transaction_history",
    "past transactions": "get_transaction_history",
    # Account info / balance
    "account info": "get_account_info", "my account": "get_account_info",
    "account details": "get_account_info", "account summary": "get_account_info",
    "balance": "get_account_info", "check balance": "get_account_info",
    "account balance": "get_account_info",
    # Watchlist view
    "my watchlist": "get_watchlist", "show watchlist": "get_watchlist",
    "view watchlist": "get_watchlist", "watchlist": "get_watchlist",
    # Make transaction
    "transact": "make_transaction", "execute trade": "make_transaction",
    "make transaction": "make_transaction", "process trade": "make_transaction",
    # Notify price change
    "alert": "notify_price_change", "notify": "notify_price_change",
    "price alert": "notify_price_change", "price change": "notify_price_change",
    "price notification": "notify_price_change",
    # Update stock price
    "update price": "update_stock_price", "set price": "update_stock_price",
    "change price": "update_stock_price", "modify price": "update_stock_price",
    # Current time
    "current time": "get_current_time", "what time": "get_current_time",
    "time now": "get_current_time",
}

# NL noun → canonical target for encoder lexicon matching
TARGET_OVERRIDES = {
    # Stock synonyms
    "share": "stock", "shares": "stock", "equity": "stock", "equities": "stock",
    "ticker": "stock", "tickers": "stock", "asset": "stock", "assets": "stock",
    "security": "stock", "securities": "stock", "instrument": "stock",
    "portfolio": "stock", "holdings": "stock", "position": "stock",
    "positions": "stock", "watchlist": "stock",
    # Order synonyms
    "trade": "order", "trades": "order", "transaction": "order",
    "transactions": "order", "purchase": "order", "purchases": "order",
    "execution": "order", "bid": "order", "offer": "order",
    # Balance synonyms
    "account": "balance", "funds": "balance", "money": "balance",
    "cash": "balance", "deposit": "balance", "capital": "balance",
    "wallet": "balance",
    # Price synonyms
    "quote": "price", "quotes": "price", "valuation": "price",
    "cost": "price", "rate": "price", "value": "price",
    # Symbol synonyms
    "name": "symbol", "identifier": "symbol", "code": "symbol",
    # Data synonyms
    "time": "data", "clock": "data", "session": "data",
    "status": "data", "login": "data", "credentials": "data",
}

# Function name → (action, target) for encoder
FUNCTION_INTENTS = {
    "add_to_watchlist":              ("add", "stock"),
    "cancel_order":                  ("delete", "order"),
    "filter_stocks_by_price":        ("filter", "stock"),
    "fund_account":                  ("update", "balance"),
    "get_account_info":              ("get", "balance"),
    "get_account_balance":           ("get", "balance"),
    "get_available_stocks":          ("list", "stock"),
    "get_current_time":              ("get", "data"),
    "get_order_details":             ("get", "order"),
    "get_order_history":             ("get", "order"),
    "get_stock_info":                ("get", "stock"),
    "get_symbol_by_name":            ("get", "stock"),
    "get_transaction_history":       ("get", "order"),
    "get_watchlist":                 ("get", "stock"),
    "make_transaction":              ("send", "order"),
    "notify_price_change":           ("send", "stock"),
    "place_order":                   ("create", "order"),
    "remove_stock_from_watchlist":   ("remove", "stock"),
    "trading_get_login_status":      ("check", "data"),
    "trading_login":                 ("start", "data"),
    "trading_logout":                ("stop", "data"),
    "update_stock_price":            ("update", "stock"),
}
