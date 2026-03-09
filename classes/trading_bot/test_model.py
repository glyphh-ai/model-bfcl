"""Tests for the TradingBot Glyphh Ada model.

Verifies:
  1. Intent extraction: NL queries -> correct action/target
  2. HDC encoding: queries and functions encode into compatible Glyphs
  3. Routing accuracy: each function is the top match for its representative queries
"""

import json
import sys
from pathlib import Path

import pytest

# Setup path -- local dir MUST come first to shadow top-level intent.py
_DIR = Path(__file__).parent
_BFCL_DIR = _DIR.parent.parent
sys.path.insert(0, str(_DIR))

from intent import extract_intent, ACTION_TO_FUNC, FUNC_TO_ACTION, FUNC_TO_TARGET
from encoder import ENCODER_CONFIG, encode_query, encode_function

from glyphh import Encoder
from glyphh.core.types import Concept, Glyph
from glyphh.core.ops import cosine_similarity


def _layer_weighted_score(q: Glyph, f: Glyph) -> float:
    """Layer-weighted similarity using ENCODER_CONFIG weights.

    Computes per-layer scores (weighted-average of role similarities within
    each layer), then combines layers using their similarity_weight.
    This lets the intent layer's action role dominate over noisy BoW roles.
    """
    config = ENCODER_CONFIG
    layer_configs = {lc.name: lc for lc in config.layers}

    total_score = 0.0
    total_weight = 0.0

    for lname in q.layers:
        if lname not in f.layers:
            continue
        lc = layer_configs.get(lname)
        if not lc:
            continue
        lw = lc.similarity_weight

        # Weighted average of role similarities within this layer
        role_score = 0.0
        role_weight = 0.0
        for seg_cfg in lc.segments:
            sname = seg_cfg.name
            if sname not in q.layers[lname].segments or sname not in f.layers[lname].segments:
                continue
            for role_cfg in seg_cfg.roles:
                rname = role_cfg.name
                qs = q.layers[lname].segments[sname]
                fs = f.layers[lname].segments[sname]
                if rname in qs.roles and rname in fs.roles:
                    sim = float(cosine_similarity(qs.roles[rname].data, fs.roles[rname].data))
                    rw = role_cfg.similarity_weight
                    role_score += sim * rw
                    role_weight += rw

        if role_weight > 0:
            layer_sim = role_score / role_weight
            total_score += layer_sim * lw
            total_weight += lw

    return total_score / total_weight if total_weight > 0 else 0.0


# -- Intent extraction tests -----------------------------------------------

class TestIntentExtraction:
    """Verify NL queries extract correct trading actions."""

    @pytest.mark.parametrize("query, expected_action", [
        # add_watchlist
        ("Integrate stock for Omega Industries into my watchlist", "add_watchlist"),
        ("Add Zeta Corp's stock to my watchlist", "add_watchlist"),
        ("Include their stock in my watchlist so I can monitor it", "add_watchlist"),
        # remove_watchlist
        ("Let's take Zeta Corp out of the equation from my watchlist", "remove_watchlist"),
        ("Would you mind taking the first one off my watchlist?", "remove_watchlist"),
        ("I'd rather not monitor anymore, namely TSLA", "remove_watchlist"),
        # get_watchlist
        ("Which stocks I've been tracking lately?", "get_watchlist"),
        ("Display the current contents of that watchlist", "get_watchlist"),
        ("What stocks have I been keeping tabs on?", "get_watchlist"),
        # cancel_order
        ("Cancel the recent order for me", "cancel_order"),
        ("Let's scrap that order", "cancel_order"),
        ("Revoke the placed order", "cancel_order"),
        # place_order
        ("Execute a buy order for 100 shares of AAPL", "place_order"),
        ("Purchase 150 shares of TSLA at $700", "place_order"),
        ("Procure 50 shares of this stock", "place_order"),
        # get_order_details
        ("Provide the details of my most recent order", "get_order_details"),
        ("Can you fetch the particulars of the last order?", "get_order_details"),
        # get_stock_info
        ("Get the current stock price of AAPL", "get_stock_info"),
        ("Provide the latest market data for MSFT", "get_stock_info"),
        # get_available_stocks
        ("List all technology sector stocks", "get_available_stocks"),
        ("Get me potential tech stocks to invest in", "get_available_stocks"),
        # get_account
        ("Show me my account details and balance", "get_account"),
        ("Overview of my account including balance", "get_account"),
        # fund_account
        ("Top up my trading account with $5000", "fund_account"),
        ("Fund my account with 10000", "fund_account"),
        # withdraw
        ("Withdraw $500 from my account", "withdraw"),
        # get_time
        ("What is the current time?", "get_time"),
        # logout
        ("Log me out of the trading platform", "logout"),
        # login
        ("Log in to the trading system", "login"),
    ])
    def test_action_extraction(self, query: str, expected_action: str):
        result = extract_intent(query)
        assert result["action"] == expected_action, (
            f"Query: '{query}'\n"
            f"Expected action: {expected_action}\n"
            f"Got: {result['action']}"
        )

    def test_all_actions_have_functions(self):
        """Every action in the lexicon maps to a real function."""
        for action, func in ACTION_TO_FUNC.items():
            assert func.startswith("TradingBot."), f"{action} -> {func}"

    def test_all_functions_have_actions(self):
        """Every TradingBot function has a reverse action mapping."""
        expected_funcs = {
            "add_to_watchlist", "cancel_order", "filter_stocks_by_price",
            "fund_account", "get_account_info", "get_available_stocks",
            "get_current_time", "get_order_details", "get_order_history",
            "get_stock_info", "get_symbol_by_name", "get_transaction_history",
            "get_watchlist", "notify_price_change", "place_order",
            "remove_stock_from_watchlist", "trading_get_login_status",
            "trading_login", "trading_logout", "withdraw_funds",
        }
        for func in expected_funcs:
            assert func in FUNC_TO_ACTION, f"Missing FUNC_TO_ACTION for {func}"
            assert func in FUNC_TO_TARGET, f"Missing FUNC_TO_TARGET for {func}"


# -- HDC encoding tests -----------------------------------------------------

class TestEncoding:
    """Verify Glyphs encode correctly and score as expected."""

    @pytest.fixture
    def encoder(self):
        return Encoder(ENCODER_CONFIG)

    @pytest.fixture
    def func_defs(self):
        """Load actual function definitions from func_doc."""
        path = _BFCL_DIR / "data" / "bfcl" / "multi_turn_func_doc" / "trading_bot.json"
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    @pytest.fixture
    def func_glyphs(self, encoder, func_defs):
        """Encode all function defs into Glyphs."""
        glyphs = {}
        for fd in func_defs:
            cd = encode_function(fd)
            glyph = encoder.encode(Concept(name=cd["name"], attributes=cd["attributes"]))
            glyphs[fd["name"]] = glyph
        return glyphs

    def _score(self, encoder, func_glyphs, query: str) -> list[tuple[str, float]]:
        """Score a query against all function Glyphs using hierarchical scoring."""
        qd = encode_query(query)
        q_glyph = encoder.encode(Concept(name=qd["name"], attributes=qd["attributes"]))

        scores = []
        for fname, fg in func_glyphs.items():
            sim = _layer_weighted_score(q_glyph, fg)
            scores.append((fname, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def test_all_functions_encoded(self, func_glyphs):
        """All 20 functions should be encoded."""
        assert len(func_glyphs) == 20

    @pytest.mark.parametrize("query, expected_func", [
        # Each function must be top-1 for at least one representative query
        ("Add Omega Industries stock to my watchlist", "add_to_watchlist"),
        ("Cancel my pending order immediately", "cancel_order"),
        ("Filter stocks by price range between 100 and 500", "filter_stocks_by_price"),
        ("Fund my trading account with $10000", "fund_account"),
        ("Show me my account details and current balance", "get_account_info"),
        ("List all available stocks in the technology sector", "get_available_stocks"),
        ("What is the current time right now?", "get_current_time"),
        ("Get the details of my most recent order", "get_order_details"),
        ("Show me my complete order history", "get_order_history"),
        ("Get the current stock information for AAPL", "get_stock_info"),
        ("Find the stock symbol for Quasar Ltd", "get_symbol_by_name"),
        ("Show my transaction history for the past month", "get_transaction_history"),
        ("Display the current contents of my watchlist", "get_watchlist"),
        ("Notify me if there is a significant price change", "notify_price_change"),
        ("Place a buy order for 100 shares of TSLA at $700", "place_order"),
        ("Remove TSLA from my watchlist", "remove_stock_from_watchlist"),
        ("Check if I am currently logged in", "trading_get_login_status"),
        ("Log in to the trading platform with my credentials", "trading_login"),
        ("Log me out of the trading system securely", "trading_logout"),
        ("Withdraw $500 from my trading account", "withdraw_funds"),
    ])
    def test_function_routing(self, encoder, func_glyphs, query: str, expected_func: str):
        """Each function should be the top match for its representative query."""
        scores = self._score(encoder, func_glyphs, query)
        top_func = scores[0][0]
        top_score = scores[0][1]

        assert top_func == expected_func, (
            f"Query: '{query}'\n"
            f"Expected: {expected_func}\n"
            f"Got: {top_func} (score={top_score:.4f})\n"
            f"Top-3: {[(f, round(s, 4)) for f, s in scores[:3]]}"
        )

    @pytest.mark.parametrize("query, expected_func", [
        # Multi-turn context queries (from actual BFCL tests.jsonl entries)
        ("Integrate stock for Omega Industries into my watchlist effectively", "add_to_watchlist"),
        ("Upon further contemplation, I need to reassess my approach. Kindly proceed with canceling that order", "cancel_order"),
        ("I feel it's time to inject some capital into my portfolio", "fund_account"),
        ("Finally, I'm contemplating some shifts in my investments, review my account status", "get_account_info"),
        ("I heard that the technology sector is booming. Get me a list of potential tech stocks", "get_available_stocks"),
        ("Post-execution, it's imperative for me to review the details of the most recent order", "get_order_details"),
        ("I've been keeping a keen eye on the stock under the symbol XTC", "get_stock_info"),
        ("Hey there, could you help me by identifying the stocks currently present on my watchlist?", "get_watchlist"),
        ("Having assessed the market status, I've resolved to initiate a purchase of 100 shares of Tesla", "place_order"),
        ("I'm inclined to shake things up. Let's take Zeta Corp out of the equation from my watchlist", "remove_stock_from_watchlist"),
        ("For the moment, log me out of the trading platform", "trading_logout"),
    ])
    def test_multi_turn_queries(self, encoder, func_glyphs, query: str, expected_func: str):
        """Queries from actual multi-turn entries should route correctly."""
        scores = self._score(encoder, func_glyphs, query)
        top_func = scores[0][0]
        assert top_func == expected_func, (
            f"Query: '{query}'\n"
            f"Expected: {expected_func}\n"
            f"Got: {top_func}\n"
            f"Top-3: {[(f, round(s, 4)) for f, s in scores[:3]]}"
        )

    def test_separation(self, encoder, func_glyphs):
        """Top match should have meaningful separation from second match."""
        test_queries = [
            "Cancel my recent stock order",
            "Add this stock to my watchlist",
            "Show me my account balance",
            "Place a buy order for 50 shares",
            "Log me out of the trading system",
        ]
        for query in test_queries:
            scores = self._score(encoder, func_glyphs, query)
            top = scores[0][1]
            second = scores[1][1]
            gap = top - second
            assert gap > 0.01, (
                f"Query: '{query}' -- insufficient separation\n"
                f"Top: {scores[0][0]}={top:.4f}, Second: {scores[1][0]}={second:.4f}, Gap={gap:.4f}"
            )
