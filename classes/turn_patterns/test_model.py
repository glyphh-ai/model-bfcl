"""Tests for the TurnPattern Glyphh Ada model.

Verifies:
  1. Intent extraction: NL queries → correct action/target/domain
  2. HDC routing via GlyphSpace + pgvector: queries route to correct patterns
  3. Bulk accuracy: top-1 / top-3 / top-5 accuracy across all test pairs

Uses the full SDK pipeline: BFCLModelScorer.configure_from_db() → GlyphSpace → find_similar().
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import pytest

_DIR = Path(__file__).parent
_BFCL_DIR = _DIR.parent.parent

# Import local intent module explicitly to avoid sys.path contamination
sys.path.insert(0, str(_DIR))
from intent import extract_intent

# Now add bfcl dir for scorer
sys.path.insert(0, str(_BFCL_DIR))
from scorer import BFCLModelScorer


# ── Fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def scorer():
    """Load TurnPattern scorer from pgvector."""
    s = BFCLModelScorer()
    s.configure_from_db("turn_patterns")
    return s


@pytest.fixture(scope="module")
def test_pairs():
    """Load test pairs from tests.jsonl."""
    path = _DIR / "tests.jsonl"
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def _score_query(scorer, query: str) -> list[tuple[str, float]]:
    """Score query via GlyphSpace and return [(pattern_key, best_score)] per pattern.

    Multiple exemplars exist per pattern (keyed as pattern__v1, __v2, ...),
    so we take the max score across variants for each pattern.
    """
    result = scorer.score(query)
    if not result.all_scores:
        return []

    pattern_scores: dict[str, float] = {}
    for entry in result.all_scores:
        full_key = entry["function"]
        pattern_key = full_key.rsplit("__v", 1)[0]
        score = entry["score"]
        if pattern_key not in pattern_scores or score > pattern_scores[pattern_key]:
            pattern_scores[pattern_key] = score

    return sorted(pattern_scores.items(), key=lambda x: -x[1])


# ── Intent extraction tests ─────────────────────────────────────────────

class TestIntentExtraction:

    @pytest.mark.parametrize("query, expected_action", [
        ("Start the car engine after locking doors", "start_vehicle"),
        ("Fill the fuel tank with 50 gallons", "fill_fuel"),
        ("Check the tire pressure on all tires", "check_tires"),
        ("Set navigation to San Francisco", "navigate"),
        ("Lock all doors and secure the vehicle", "lock_doors"),
        ("Book a flight from NYC to LA", "book_flight"),
        ("Cancel the booking I made yesterday", "cancel_booking"),
        ("How much does a flight to Chicago cost?", "check_flight_cost"),
        ("Purchase insurance for the trip", "purchase_insurance"),
        ("Contact customer support about my issue", "contact_support"),
        ("Check the stock price of AAPL", "check_stock"),
        ("Place an order to buy 100 shares", "place_order"),
        ("Add to my watchlist", "add_watchlist"),
        ("Cancel the order I placed", "cancel_order"),
        ("Get the order details for order 12345", "check_order"),
        ("Retrieve the invoice for my purchase", "retrieve_invoice"),
        ("Send a message to John", "send_message"),
        ("View my sent messages", "view_messages"),
        ("Post a tweet about the event", "post_tweet"),
        ("List all files in the directory", "list_files"),
        ("Move the file to the backup folder", "move_file"),
        ("Search for 'error' in the log file", "search_file"),
        ("Sort the file alphabetically", "sort_file"),
    ])
    def test_action_extraction(self, query: str, expected_action: str):
        intent = extract_intent(query)
        assert intent["action"] == expected_action

    @pytest.mark.parametrize("query, expected_domain", [
        ("Start the car engine", "vehicle"),
        ("Book a flight to LA", "travel"),
        ("Check the stock price", "trading"),
        ("List all files", "filesystem"),
        ("Send a message to John", "messaging"),
        ("Post a tweet", "posting"),
        ("Create a ticket for the bug", "ticket"),
        ("Calculate the square root of 144", "math"),
    ])
    def test_domain_detection(self, query: str, expected_domain: str):
        intent = extract_intent(query)
        assert intent["domain"] == expected_domain


# ── GQL routing tests ──────────────────────────────────────────────────

class TestPatternRouting:
    """Verify high-frequency patterns appear in top-5 via GlyphSpace."""

    @pytest.mark.parametrize("query, expected_pattern", [
        ("Cancel the order I placed earlier", "cancel_order"),
        ("Fill up the fuel tank", "fillFuelTank"),
        ("View my sent messages", "view_messages_sent"),
        ("Check what's on my watchlist", "get_watchlist"),
    ])
    def test_top1_routing(self, scorer, query: str, expected_pattern: str):
        ranked = _score_query(scorer, query)
        top_pattern = ranked[0][0] if ranked else "NONE"
        assert top_pattern == expected_pattern, (
            f"Query: '{query}'\n"
            f"Expected: {expected_pattern}\n"
            f"Got: {top_pattern}\n"
            f"Top-5: {[(p, round(s, 4)) for p, s in ranked[:5]]}"
        )

    @pytest.mark.parametrize("query, expected_pattern", [
        ("Start the car after securing it", "lockDoors|pressBrakePedal|startEngine"),
        ("Book a flight from LA to NYC", "book_flight"),
        ("Get the order details", "get_order_details"),
        ("Retrieve the invoice", "retrieve_invoice"),
        ("Cancel my flight booking", "cancel_booking"),
        ("Send a message to the team", "send_message"),
        ("Post a tweet about the launch", "post_tweet"),
        ("Contact customer support", "contact_customer_support"),
        ("Get flight cost and book it", "get_flight_cost|book_flight"),
    ])
    def test_top5_routing(self, scorer, query: str, expected_pattern: str):
        ranked = _score_query(scorer, query)
        top5 = [p for p, _ in ranked[:5]]
        assert expected_pattern in top5, (
            f"Query: '{query}'\n"
            f"Expected in top-5: {expected_pattern}\n"
            f"Top-5: {[(p, round(s, 4)) for p, s in ranked[:5]]}"
        )


# ── Bulk accuracy via GQL ──────────────────────────────────────────────

class TestBulkAccuracy:
    """Measure overall routing accuracy using GlyphSpace.find_similar()."""

    def test_top1_accuracy(self, scorer, test_pairs):
        correct = 0
        total = 0
        failures_by_expected = defaultdict(int)

        for tp in test_pairs:
            ranked = _score_query(scorer, tp["query"])
            top_pattern = ranked[0][0] if ranked else "NONE"
            if top_pattern == tp["expected"]:
                correct += 1
            else:
                failures_by_expected[tp["expected"]] += 1
            total += 1

        accuracy = correct / total if total else 0
        print(f"\nTop-1 accuracy: {correct}/{total} = {accuracy:.1%}")
        worst = sorted(failures_by_expected.items(), key=lambda x: -x[1])[:10]
        print("Worst patterns:")
        for pat, count in worst:
            print(f"  {count:4d} failures  {pat}")

        assert accuracy > 0.30, f"Top-1 accuracy {accuracy:.1%} below 30%"

    def test_top1_excluding_empty(self, scorer, test_pairs):
        """Top-1 accuracy excluding [] (empty/refusal) pattern."""
        correct = 0
        total = 0
        for tp in test_pairs:
            if tp["expected"] == "[]":
                continue
            ranked = _score_query(scorer, tp["query"])
            top_pattern = ranked[0][0] if ranked else "NONE"
            if top_pattern == tp["expected"]:
                correct += 1
            total += 1

        accuracy = correct / total if total else 0
        print(f"\nTop-1 accuracy (excl empty): {correct}/{total} = {accuracy:.1%}")
        assert accuracy > 0.50, f"Top-1 accuracy (excl empty) {accuracy:.1%} below 50%"

    def test_top3_accuracy(self, scorer, test_pairs):
        correct = 0
        total = 0
        for tp in test_pairs:
            ranked = _score_query(scorer, tp["query"])
            top3 = [p for p, _ in ranked[:3]]
            if tp["expected"] in top3:
                correct += 1
            total += 1

        accuracy = correct / total if total else 0
        print(f"\nTop-3 accuracy: {correct}/{total} = {accuracy:.1%}")
        assert accuracy > 0.40, f"Top-3 accuracy {accuracy:.1%} below 40%"

    def test_top5_accuracy(self, scorer, test_pairs):
        correct = 0
        total = 0
        for tp in test_pairs:
            ranked = _score_query(scorer, tp["query"])
            top5 = [p for p, _ in ranked[:5]]
            if tp["expected"] in top5:
                correct += 1
            total += 1

        accuracy = correct / total if total else 0
        print(f"\nTop-5 accuracy: {correct}/{total} = {accuracy:.1%}")
        assert accuracy > 0.50, f"Top-5 accuracy {accuracy:.1%} below 50%"
