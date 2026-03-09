"""Tests for the MessageAPI Glyphh Ada model.

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


# -- Intent extraction tests -------------------------------------------------

class TestIntentExtraction:
    """Verify NL queries extract correct messaging actions."""

    @pytest.mark.parametrize("query, expected_action", [
        # send
        ("Send a message to USR002 saying hello", "send"),
        ("Please dispatch a note to my colleague", "send"),
        ("Relay a message to my advisor", "send"),
        ("Could you kindly dispatch the message to USR003", "send"),
        ("Notify my financial adviser about this trade", "send"),
        # delete_msg
        ("Delete the last message to Alice", "delete_msg"),
        ("Please remove the message I sent", "delete_msg"),
        ("Retract that specific message from their inbox", "delete_msg"),
        # view_sent
        ("Show me all the messages I have sent so far", "view_sent"),
        ("Could you display all the messages I sent?", "view_sent"),
        ("Review all the messages I have send so far", "view_sent"),
        ("Peruse my recent messages to ensure the communique was transmitted", "view_sent"),
        ("Check if the message is sent", "view_sent"),
        # search_msg
        ("Search for messages containing the keyword 'budget'", "search_msg"),
        # get_stats
        ("Show me the message statistics", "get_stats"),
        # add_contact
        ("Add him as new contact in the workspace", "add_contact"),
        ("I need to add her contact Kelly", "add_contact"),
        # login
        ("Logging in as USR001 to the messaging system", "login"),
        ("Log my user id USR001 in", "login"),
        # list_users
        ("List all users in the workspace", "list_users"),
        # get_user_id
        ("Get user id for Bob", "get_user_id"),
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
            assert func.startswith("MessageAPI."), f"{action} -> {func}"

    def test_all_functions_have_actions(self):
        """Every MessageAPI function has a reverse action mapping."""
        expected_funcs = {
            "send_message", "delete_message", "view_messages_sent",
            "search_messages", "get_message_stats", "add_contact",
            "get_user_id", "list_users", "message_login",
            "message_get_login_status",
        }
        for func in expected_funcs:
            assert func in FUNC_TO_ACTION, f"Missing FUNC_TO_ACTION for {func}"
            assert func in FUNC_TO_TARGET, f"Missing FUNC_TO_TARGET for {func}"


# -- HDC encoding tests ------------------------------------------------------

class TestEncoding:
    """Verify Glyphs encode correctly and score as expected."""

    @pytest.fixture
    def encoder(self):
        return Encoder(ENCODER_CONFIG)

    @pytest.fixture
    def func_defs(self):
        """Load actual function definitions from func_doc."""
        path = _BFCL_DIR / "data" / "bfcl" / "multi_turn_func_doc" / "message_api.json"
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
        """All 10 functions should be encoded."""
        assert len(func_glyphs) == 10

    @pytest.mark.parametrize("query, expected_func", [
        # Each function must be top-1 for at least one representative query
        ("Send a message to my colleague saying hello", "send_message"),
        ("Delete the last message I sent to Alice", "delete_message"),
        ("Show me all the messages I have sent so far", "view_messages_sent"),
        ("Search for messages containing the keyword budget", "search_messages"),
        ("Get the message statistics for my account", "get_message_stats"),
        ("Add a new contact named Kelly to the workspace", "add_contact"),
        ("Get the user id for Bob", "get_user_id"),
        ("List all users in the workspace", "list_users"),
        ("Logging in as USR001 to the messaging application", "message_login"),
        ("Check my login status on the messaging system", "message_get_login_status"),
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
        # Multi-turn context queries (from actual BFCL entries)
        ("Attempt to relay a message to the individual with ID USR002 updating them on the finalization of the report", "send_message"),
        ("Could you double check if the message is sent?", "view_messages_sent"),
        ("I've sent incorrect information to my advisor. Please delete the last message.", "delete_message"),
        ("Please dispatch of the report to Kelly in the format of Kelly Total Score", "send_message"),
        ("Also at your earliest convenience can you show me all the messages I have send so far?", "view_messages_sent"),
        ("Logging in as USR001 to notify my financial advisor", "message_login"),
        ("I need to add her contact Kelly", "add_contact"),
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
            "Send a message to Bob",
            "Delete the last message",
            "Show all messages I have sent",
            "Search messages for keyword error",
            "Add a new contact to the workspace",
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
