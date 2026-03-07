"""Per-class intent overrides for TradingBot."""

CLASS_ALIASES = ["TradingBot", "trading", "stock trading", "trade"]
CLASS_DOMAIN = "trading"

ACTION_SYNONYMS = {
    "purchase": "place_order", "buy": "place_order",
    "sell": "place_order", "liquidate": "cancel_order",
    "deposit": "fund_account", "replenish": "fund_account",
    "track": "add_to_watchlist", "watch": "add_to_watchlist",
}

TARGET_OVERRIDES = {
    "share": "stock", "equity": "stock", "asset": "stock",
    "position": "order", "trade": "order",
}
