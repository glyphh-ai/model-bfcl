"""Tests for the TravelBookingAPI (TravelAPI) Glyphh Ada model.

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


# -- Intent extraction tests ------------------------------------------------

class TestIntentExtraction:
    """Verify NL queries extract correct travel actions."""

    @pytest.mark.parametrize("query, expected_action", [
        # authenticate
        ("Log into my travel account with client ID and secret", "authenticate"),
        ("Assistance with the authentication process would be appreciated", "authenticate"),
        ("I need to access my account with my credentials", "authenticate"),
        # book_flight
        ("Book a flight from New York to London", "book_flight"),
        ("Reserve a first class flight to Paris", "book_flight"),
        ("I want to fly from LAX to JFK", "book_flight"),
        # cancel_booking
        ("Cancel the booking for my upcoming trip", "cancel_booking"),
        ("Cancel my flight reservation", "cancel_booking"),
        # exchange_rate
        ("What is the exchange rate from USD to EUR?", "exchange_rate"),
        ("Convert 100 dollars to euros", "exchange_rate"),
        ("How much is 500 USD in GBP?", "exchange_rate"),
        # contact_support
        ("Contact customer support about my issue", "contact_support"),
        ("I need to reach out to customer support", "contact_support"),
        # get_flight_cost
        ("How much does the flight from LAX to JFK cost?", "get_flight_cost"),
        ("What is the price of a flight to London?", "get_flight_cost"),
        ("Get the flight fare for business class", "get_flight_cost"),
        # get_nearest_airport
        ("What is the nearest airport to San Francisco?", "get_nearest_airport"),
        ("Find the airport closest to London", "get_nearest_airport"),
        # list_airports
        ("List all airports available", "list_airports"),
        ("Show all available airports", "list_airports"),
        # get_card_balance
        ("Check the balance of my credit card", "get_card_balance"),
        ("What is my credit card balance?", "get_card_balance"),
        # get_all_cards
        ("Show all registered credit cards", "get_all_cards"),
        ("List all my credit cards", "get_all_cards"),
        # register_card
        ("Register a new credit card", "register_card"),
        ("Add a credit card to my account", "register_card"),
        # retrieve_invoice
        ("Retrieve the invoice for my booking", "retrieve_invoice"),
        ("Get my invoice for the trip", "retrieve_invoice"),
        # purchase_insurance
        ("Purchase travel insurance for my flight", "purchase_insurance"),
        ("Buy insurance for the booking", "purchase_insurance"),
        # get_budget_year
        ("What is the current fiscal year?", "get_budget_year"),
        ("Get the budget fiscal year", "get_budget_year"),
        # set_budget
        ("Set the budget limit to 5000 dollars", "set_budget"),
        ("Establish a budget of 3000 for the trip", "set_budget"),
        # login_status
        ("Check if I am logged in", "login_status"),
        ("What is my login status?", "login_status"),
        # verify_traveler
        ("Verify my traveler information", "verify_traveler"),
        ("Verify the traveler details with passport number", "verify_traveler"),
        # get_booking_history
        ("Show my booking history", "get_booking_history"),
        ("Retrieve all past bookings", "get_booking_history"),
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
            if action == "none":
                continue
            assert func.startswith("TravelAPI."), f"{action} -> {func}"

    def test_all_functions_have_actions(self):
        """Every TravelAPI function has a reverse action mapping."""
        expected_funcs = {
            "authenticate_travel", "book_flight", "cancel_booking",
            "compute_exchange_rate", "contact_customer_support",
            "get_all_credit_cards", "get_booking_history",
            "get_budget_fiscal_year", "get_credit_card_balance",
            "get_flight_cost", "get_nearest_airport_by_city",
            "list_all_airports", "purchase_insurance",
            "register_credit_card", "retrieve_invoice",
            "set_budget_limit", "travel_get_login_status",
            "verify_traveler_information",
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
        path = _BFCL_DIR / "data" / "bfcl" / "multi_turn_func_doc" / "travel_booking.json"
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
        """All 18 functions should be encoded."""
        assert len(func_glyphs) == 18

    @pytest.mark.parametrize("query, expected_func", [
        # Each function must be top-1 for at least one representative query
        ("Log into my travel account using client ID and secret", "authenticate_travel"),
        ("Book a flight from New York to London in business class", "book_flight"),
        ("Cancel the booking for my upcoming trip", "cancel_booking"),
        ("Compute the exchange rate from USD to EUR", "compute_exchange_rate"),
        ("Contact customer support about my booking issue", "contact_customer_support"),
        ("Show all registered credit cards on file", "get_all_credit_cards"),
        ("Retrieve all my past booking history", "get_booking_history"),
        ("Get the budget fiscal year information", "get_budget_fiscal_year"),
        ("Check the balance of my credit card", "get_credit_card_balance"),
        ("How much does a flight from LAX to JFK cost?", "get_flight_cost"),
        ("What is the nearest airport to San Francisco?", "get_nearest_airport_by_city"),
        ("List all available airports in the system", "list_all_airports"),
        ("Purchase travel insurance for my flight booking", "purchase_insurance"),
        ("Register a new credit card with my account", "register_credit_card"),
        ("Retrieve the invoice for my recent booking", "retrieve_invoice"),
        ("Set the budget limit to 5000 dollars", "set_budget_limit"),
        ("Check if I am currently logged in", "travel_get_login_status"),
        ("Verify the traveler information with passport number", "verify_traveler_information"),
    ])
    def test_function_routing(self, encoder, func_glyphs, query: str, expected_func: str):
        """Each function should be the top match for its representative query."""
        scores = self._score(encoder, func_glyphs, query)
        top_func = scores[0][0]
        top_score = scores[0][1]
        second_score = scores[1][1] if len(scores) > 1 else 0.0

        assert top_func == expected_func, (
            f"Query: '{query}'\n"
            f"Expected: {expected_func}\n"
            f"Got: {top_func} (score={top_score:.4f})\n"
            f"Top-3: {[(f, round(s, 4)) for f, s in scores[:3]]}"
        )

    @pytest.mark.parametrize("query, expected_func", [
        # Multi-turn context queries (from actual BFCL entries)
        ("Assistance with the authentication process would be immensely appreciated", "authenticate_travel"),
        ("I want to fly from Rivermist to Stonebrook on 2024-10-12 in business class", "book_flight"),
        ("Cancel that booking right away", "cancel_booking"),
        ("Convert 1000 USD to EUR for me", "compute_exchange_rate"),
        ("I need to reach out to customer support about a problem with my booking", "contact_customer_support"),
        ("Show me all the credit cards I have on file", "get_all_credit_cards"),
        ("Can you pull up my complete booking history?", "get_booking_history"),
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
            "Book a flight to London",
            "Cancel my booking",
            "What is the exchange rate for USD to EUR",
            "Register a new credit card",
            "Get the invoice for my trip",
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
