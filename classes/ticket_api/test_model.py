"""Tests for the TicketAPI Glyphh Ada model.

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
    """Verify NL queries extract correct ticket actions."""

    @pytest.mark.parametrize("query, expected_action", [
        # create
        ("Create a ticket titled 'Server Down' with priority 3", "create"),
        ("Draft a support ticket for the billing issue", "create"),
        ("File a support ticket about the account error", "create"),
        ("I need to submit a formal complaint about the cancellation", "create"),
        ("Lodge a complaint titled 'Platform Error'", "create"),
        # close
        ("Close the ticket for me as the issue is resolved", "close"),
        ("Close the outstanding ticket with id 3", "close"),
        ("I noticed there's a ticket I no longer find necessary. Can you cancel it on my behalf?", "close"),
        # resolve
        ("Resolve it for me since I've addressed the problem", "resolve"),
        ("Let's resolve the tire pressure ticket with an update", "resolve"),
        ("Mark it as resolved with the resolution details", "resolve"),
        ("Check it off as resolved", "resolve"),
        # edit
        ("Enhance the ticket's details by updating its status to Urgent", "edit"),
        ("Open up ticket 654321 and set the priority to 3", "edit"),
        # get_ticket
        ("Retrieve the details of that ticket for me", "get_ticket"),
        ("Fetch that ticket for me so I can review the status", "get_ticket"),
        ("Could you please retrieve the details of the last service ticket", "get_ticket"),
        # list_tickets
        ("Show me all my tickets", "list_tickets"),
        # login
        ("My ticket username is msmith, password is SecurePass123", "login"),
        ("Login the ticket with my username mthompson and password securePass123", "login"),
        ("My username is mzhang and password is SecurePass123", "login"),
        # logout
        ("Log out of the ticketing system", "logout"),
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
            assert func.startswith("TicketAPI."), f"{action} -> {func}"

    def test_all_functions_have_actions(self):
        """Every TicketAPI function has a reverse action mapping."""
        expected_funcs = {
            "create_ticket", "close_ticket", "resolve_ticket",
            "edit_ticket", "get_ticket", "get_user_tickets",
            "ticket_login", "ticket_get_login_status", "logout",
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
        path = _BFCL_DIR / "data" / "bfcl" / "multi_turn_func_doc" / "ticket_api.json"
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
        """All 9 functions should be encoded."""
        assert len(func_glyphs) == 9

    @pytest.mark.parametrize("query, expected_func", [
        # Each function must be top-1 for at least one representative query
        ("Create a new support ticket titled Server Down", "create_ticket"),
        ("Close the ticket as the issue is now resolved", "close_ticket"),
        ("Resolve the ticket with a note saying issue fixed", "resolve_ticket"),
        ("Edit the ticket to update the priority to level 5", "edit_ticket"),
        ("Retrieve the details of ticket number 987654", "get_ticket"),
        ("Show me all tickets I have created", "get_user_tickets"),
        ("Login to the ticket system with username msmith and password SecurePass123", "ticket_login"),
        ("Check my login status on the ticketing system", "ticket_get_login_status"),
        ("Log out of the ticketing system", "logout"),
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
        ("I've come across an urgent conundrum: servers are down. Draft a support ticket with description highlighting this dilemma and assign it a priority level of 3", "create_ticket"),
        ("Great now that everything is sorted kindly proceed to close the ticket for me as the tire issue is resolved", "close_ticket"),
        ("With the information from the ticket at hand resolve it for me since I've already addressed the core problem. Summarize the resolution as Fixed through manual troubleshooting.", "resolve_ticket"),
        ("Could you enhance the ticket's details by updating its status to Urgent and its priority to the highest level", "edit_ticket"),
        ("I recently documented a support ticket and I recall the ticket number was 987654. Retrieve the details of that ticket for me.", "get_ticket"),
        ("My ticket username is msmith password is SecurePass123", "ticket_login"),
        ("There's a minor snag in our ticketing system. Ticket 7423 is still unresolved but with our recent brainstorming feedback just go ahead and check it off as resolved.", "resolve_ticket"),
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
            "Create a new support ticket",
            "Close this ticket",
            "Resolve the ticket with this resolution",
            "Retrieve the ticket details",
            "Log out of the system",
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
