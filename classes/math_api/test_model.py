"""Tests for the MathAPI Glyphh Ada model.

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
    """Verify NL queries extract correct math actions."""

    @pytest.mark.parametrize("query, expected_action", [
        # logarithm
        ("Determine the logarithm of the character count to the base 6", "logarithm"),
        ("Compute the base 10 logarithm for the number", "logarithm"),
        ("Calculate the logarithm of the distance considering base 10", "logarithm"),
        # mean
        ("Compute the average of the three numerical values", "mean"),
        ("Get the mean of character number of all files", "mean"),
        ("What's the mean of the quarterly revenue?", "mean"),
        ("What's the average tire pressure?", "mean"),
        # standard_deviation
        ("What about the standard deviation?", "standard_deviation"),
        ("The metric is standard deviation", "standard_deviation"),
        # add
        ("Add two numbers together", "add"),
        # subtract
        ("Subtract one number from another", "subtract"),
        # multiply
        ("Multiply two values", "multiply"),
        ("Calculate the product of these numbers", "multiply"),
        # divide
        ("Divide 100 by 5", "divide"),
        # square_root
        ("Calculate the square root of 144", "square_root"),
        # power
        ("Raise 2 to the power of 8", "power"),
        # percentage
        ("What percentage of 200 is 50?", "percentage"),
        # absolute_value
        ("Find the absolute value of -42", "absolute_value"),
        # round_number
        ("Round the number to 2 decimal places", "round_number"),
        # max_value
        ("Find the maximum value in the list", "max_value"),
        # min_value
        ("Find the minimum value in the dataset", "min_value"),
        # sum_values
        ("Calculate the sum of all these numbers", "sum_values"),
        # si_convert
        ("Convert the value from meters to centimeters", "si_convert"),
        # imperial_convert
        ("Convert miles to kilometers", "imperial_convert"),
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
            assert func.startswith("MathAPI."), f"{action} -> {func}"

    def test_all_functions_have_actions(self):
        """Every MathAPI function has a reverse action mapping."""
        expected_funcs = {
            "absolute_value", "add", "divide", "imperial_si_conversion",
            "logarithm", "max_value", "mean", "min_value", "multiply",
            "percentage", "power", "round_number", "si_unit_conversion",
            "square_root", "standard_deviation", "subtract", "sum_values",
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
        path = _BFCL_DIR / "data" / "bfcl" / "multi_turn_func_doc" / "math_api.json"
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
        """All 17 functions should be encoded."""
        assert len(func_glyphs) == 17

    @pytest.mark.parametrize("query, expected_func", [
        # Each function must be top-1 for at least one representative query
        ("Calculate the absolute value of negative five", "absolute_value"),
        ("Add 3 and 7 together", "add"),
        ("Divide 100 by 4", "divide"),
        ("Convert miles to kilometers using imperial conversion", "imperial_si_conversion"),
        ("Compute the logarithm of 1000 base 10 with precision 5", "logarithm"),
        ("Find the maximum value in the list of numbers", "max_value"),
        ("Calculate the mean of these scores", "mean"),
        ("Find the minimum value from the dataset", "min_value"),
        ("Multiply 12 by 8", "multiply"),
        ("What percentage of 500 is 75?", "percentage"),
        ("Raise 3 to the power of 4", "power"),
        ("Round 3.14159 to 2 decimal places", "round_number"),
        ("Convert meters to centimeters in SI units", "si_unit_conversion"),
        ("Calculate the square root of 256 with precision 10", "square_root"),
        ("Compute the standard deviation of the dataset", "standard_deviation"),
        ("Subtract 15 from 42", "subtract"),
        ("Calculate the sum of all values in the list", "sum_values"),
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
        ("While analyzing my project's numerical data, determine the logarithm of the character count to the base 6 with precision up to four decimal places", "logarithm"),
        ("Could you compute the average of the three numerical value obtained?", "mean"),
        ("What's the mean of the quarterly revenue?", "mean"),
        ("What about the standard deviation?", "standard_deviation"),
        ("Look at the student_record.txt and tell me the average score", "mean"),
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
            "Calculate the logarithm of 100 base 10",
            "Compute the average of these numbers",
            "Find the square root of 64",
            "Subtract 5 from 20",
            "What percentage of 200 is 50?",
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
