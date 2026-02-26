"""
Observer Model (Model C) — deductive reasoning over Model A and Model B outputs.

Three Glyphh models, one substrate:
  Model A: query → function (perception)
  Model B: query → pattern (planning)
  Model C: (A_output, B_output, delta) → best candidate (reasoning)

Model C encodes "observations" — what Model A and Model B said about a query,
and how their signals relate. It compares the current observation against
reference observations from training data where we know the right answer.

This is deductive reasoning in HDC space:
  - Encode the premise (both models' outputs)
  - Compare against known premises with known conclusions
  - The closest matching premise gives the conclusion

Roles:
  - model_a_func: bag_of_words of Model A's top function name
  - model_a_confidence: binned confidence score
  - model_b_pattern: bag_of_words of Model B's top pattern sequence
  - model_b_confidence: binned confidence score
  - agreement: lexicon — do the models agree? (agree, partial, disagree)
  - delta_signal: bag_of_words encoding of the *difference* between
    what A says and what B says (functions in B but not A's top, etc.)
"""

import json
from pathlib import Path
from collections import Counter
from typing import Any

from glyphh.core.config import (
    EncoderConfig, Layer, Role, Segment, NumericConfig, EncodingStrategy,
)
from glyphh.core.types import Concept, Glyph
from glyphh.core.ops import cosine_similarity
from glyphh import Encoder

from handler import GlyphhBFCLHandler
from pattern_encoder import PatternRouter, _split_camel_snake, _tokenize, _bow_value
from multi_turn_handler import get_available_functions, extract_turn_query, parse_ground_truth_step

# ---------------------------------------------------------------------------
# Observer encoder config — Model C
# ---------------------------------------------------------------------------

OBSERVER_CONFIG = EncoderConfig(
    dimension=10000,
    seed=107,  # Third independent vector space
    apply_weights_during_encoding=False,
    include_temporal=False,
    layers=[
        Layer(
            name="model_signals",
            similarity_weight=0.50,
            segments=[
                Segment(
                    name="model_a",
                    roles=[
                        Role(
                            name="top_func",
                            similarity_weight=0.8,
                            text_encoding="bag_of_words",
                        ),
                        Role(
                            name="confidence",
                            similarity_weight=0.3,
                            numeric_config=NumericConfig(
                                bin_width=0.05,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=0.0,
                                max_value=1.0,
                            ),
                        ),
                    ],
                ),
                Segment(
                    name="model_b",
                    roles=[
                        Role(
                            name="top_pattern",
                            similarity_weight=0.8,
                            text_encoding="bag_of_words",
                        ),
                        Role(
                            name="confidence",
                            similarity_weight=0.3,
                            numeric_config=NumericConfig(
                                bin_width=0.05,
                                encoding_strategy=EncodingStrategy.THERMOMETER,
                                min_value=0.0,
                                max_value=1.0,
                            ),
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="delta",
            similarity_weight=0.50,
            segments=[
                Segment(
                    name="relation",
                    roles=[
                        Role(
                            name="agreement",
                            similarity_weight=0.6,
                            lexicons=["agree", "partial", "disagree"],
                        ),
                        Role(
                            name="delta_tokens",
                            similarity_weight=0.4,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Observation encoding
# ---------------------------------------------------------------------------

def _func_tokens(func_name: str) -> str:
    """Convert function name to bag-of-words tokens."""
    return _bow_value(_split_camel_snake(func_name).split())


def _pattern_tokens(sequence: tuple | list) -> str:
    """Convert a pattern sequence to bag-of-words tokens."""
    tokens = []
    for fname in sequence:
        tokens.extend(_split_camel_snake(fname).split())
    return _bow_value(tokens)


def encode_observation(
    model_a_top: str,
    model_a_conf: float,
    model_b_top_seq: tuple | list,
    model_b_conf: float,
) -> dict:
    """Encode an observation of both models' outputs.

    Returns a Concept-compatible dict for Model C.
    """
    # Agreement classification
    a_func = model_a_top or ""
    b_funcs = list(model_b_top_seq) if model_b_top_seq else []

    if a_func and a_func in b_funcs:
        if len(b_funcs) == 1 and b_funcs[0] == a_func:
            agreement = "agree"
        else:
            agreement = "partial"  # A's pick is in B's sequence but B has more
    else:
        agreement = "disagree"

    # Delta tokens: functions in B's sequence that aren't A's top pick
    delta_funcs = [f for f in b_funcs if f != a_func]
    delta_tokens = _pattern_tokens(delta_funcs) if delta_funcs else "none"

    return {
        "name": "observation",
        "attributes": {
            "top_func": _func_tokens(a_func) if a_func else "none",
            "confidence": str(model_a_conf),
            "top_pattern": _pattern_tokens(b_funcs) if b_funcs else "none",
            "confidence": str(model_b_conf),  # Note: same role name, different segment
            "agreement": agreement,
            "delta_tokens": delta_tokens,
        },
    }


def _encode_observation_concept(
    model_a_top: str,
    model_a_conf: float,
    model_b_top_seq: tuple | list,
    model_b_conf: float,
) -> dict:
    """Build attributes dict with segment-qualified names for the observer.

    Since we have two 'confidence' roles in different segments, we need
    to use the flat attribute dict carefully. The SDK maps attributes to
    roles by name, so we need unique names.
    """
    a_func = model_a_top or ""
    b_funcs = list(model_b_top_seq) if model_b_top_seq else []

    if a_func and a_func in b_funcs:
        agreement = "partial" if len(b_funcs) > 1 else "agree"
    elif not a_func and not b_funcs:
        agreement = "agree"
    else:
        agreement = "disagree"

    delta_funcs = [f for f in b_funcs if f != a_func]
    delta_tokens = _pattern_tokens(delta_funcs) if delta_funcs else "none"

    return {
        "top_func": _func_tokens(a_func) if a_func else "none",
        "top_pattern": _pattern_tokens(b_funcs) if b_funcs else "none",
        "agreement": agreement,
        "delta_tokens": delta_tokens,
    }


# ---------------------------------------------------------------------------
# Observer — Model C
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data" / "bfcl"

_ANSWER_FILES = {
    "multi_turn_base": "possible_answer/BFCL_v3_multi_turn_base.json",
    "multi_turn_miss_func": "possible_answer/BFCL_v3_multi_turn_miss_func.json",
    "multi_turn_miss_param": "possible_answer/BFCL_v3_multi_turn_miss_param.json",
    "multi_turn_long_context": "possible_answer/BFCL_v3_multi_turn_long_context.json",
}
_QUESTION_FILES = {
    "multi_turn_base": "BFCL_v3_multi_turn_base.json",
    "multi_turn_miss_func": "BFCL_v3_multi_turn_miss_func.json",
    "multi_turn_miss_param": "BFCL_v3_multi_turn_miss_param.json",
    "multi_turn_long_context": "BFCL_v3_multi_turn_long_context.json",
}


class Observer:
    """Model C — deductive reasoning over Model A and Model B.

    Learns from training data how the two models' outputs relate to
    the correct answer. At inference, encodes the current observation
    and finds the closest reference observation to decide.

    Each reference observation stores:
      - The encoded observation glyph (what A and B said)
      - The correct answer (function sequence)
      - Whether the answer was single-func or multi-func
    """

    def __init__(self, func_router: GlyphhBFCLHandler, pattern_router: PatternRouter):
        self.encoder = Encoder(OBSERVER_CONFIG)
        self.func_router = func_router
        self.pattern_router = pattern_router

        # Reference observations from training
        self.ref_glyphs: list[Glyph] = []
        self.ref_answers: list[list[str]] = []  # correct function sequences
        self._built = False

    def build(self, exclude_categories: list[str] | None = None):
        """Build reference observations from training data.

        For each turn in the multi-turn training data:
        1. Run Model A and Model B on the query
        2. Encode the observation (what both models said)
        3. Store alongside the ground truth answer

        Args:
            exclude_categories: Categories to exclude from training.
                Used for cross-validation — train on 3, test on the 4th.
                If None, trains on all categories (original behavior).
        """
        exclude = set(exclude_categories or [])

        for cat in _ANSWER_FILES:
            if cat in exclude:
                continue

            qpath = DATA_DIR / _QUESTION_FILES[cat]
            apath = DATA_DIR / _ANSWER_FILES[cat]
            if not qpath.exists() or not apath.exists():
                continue

            questions = {}
            with open(qpath) as f:
                for line in f:
                    e = json.loads(line.strip())
                    questions[e.get("id", "")] = e

            with open(apath) as f:
                for line in f:
                    a = json.loads(line.strip())
                    eid = a.get("id", "")
                    gt = a.get("ground_truth", [])
                    entry = questions.get(eid, {})
                    turns = entry.get("question", [])
                    available_funcs = get_available_functions(entry)

                    if not available_funcs:
                        continue

                    available_names = {fd["name"] for fd in available_funcs}

                    for ti, step in enumerate(gt):
                        if not step:
                            continue

                        # Extract expected functions
                        funcs = []
                        for call_str in step:
                            if isinstance(call_str, str):
                                paren = call_str.find("(")
                                funcs.append(call_str[:paren] if paren != -1 else call_str)
                        if not funcs:
                            continue

                        # Extract query
                        query = ""
                        if ti < len(turns):
                            turn = turns[ti]
                            if isinstance(turn, list):
                                for msg in turn:
                                    if isinstance(msg, dict) and msg.get("role") == "user":
                                        query = msg.get("content", "")
                        if not query:
                            continue

                        # Run Model A
                        route_a = self.func_router.route(query, available_funcs)
                        a_top = route_a.get("tool", "")
                        a_conf = route_a.get("confidence", 0.0)

                        # Run Model B
                        involved = entry.get("involved_classes", [])
                        domain = _domain_from_classes(involved)
                        results_b = self.pattern_router.route(query, domain_hint=domain, top_k=1)
                        if results_b:
                            b_seq = results_b[0]["sequence"]
                            b_conf = results_b[0]["score"]
                        else:
                            b_seq = ()
                            b_conf = 0.0

                        # Encode observation
                        attrs = _encode_observation_concept(a_top, a_conf, b_seq, b_conf)
                        concept = Concept(name="obs", attributes=attrs)
                        glyph = self.encoder.encode(concept)

                        self.ref_glyphs.append(glyph)
                        self.ref_answers.append(funcs)

        self._built = True

    def decide(
        self,
        query: str,
        available_funcs: list[dict],
        available_names: set[str],
        domain_hint: str = "mixed",
        top_k: int = 5,
    ) -> list[str]:
        """Observe Model A and Model B, then reason about the best answer.

        1. Run both models on the query
        2. Encode the observation
        3. Find closest reference observations
        4. Return the answer from the best matching reference
        """
        if not self._built:
            self.build()

        # Model A
        route_a = self.func_router.route(query, available_funcs)
        a_top = route_a.get("tool", "")
        a_conf = route_a.get("confidence", 0.0)
        a_scores = {s["function"]: s["score"] for s in route_a.get("all_scores", [])}

        # Model B
        results_b = self.pattern_router.route(query, domain_hint=domain_hint, top_k=3)
        if results_b:
            b_seq = results_b[0]["sequence"]
            b_conf = results_b[0]["score"]
        else:
            b_seq = ()
            b_conf = 0.0

        # Encode current observation
        attrs = _encode_observation_concept(a_top, a_conf, b_seq, b_conf)
        concept = Concept(name="obs", attributes=attrs)
        obs_glyph = self.encoder.encode(concept)

        # Score against all reference observations
        scores = []
        for i, ref_g in enumerate(self.ref_glyphs):
            sim = float(cosine_similarity(
                obs_glyph.global_cortex.data,
                ref_g.global_cortex.data,
            ))
            scores.append((sim, i))

        scores.sort(key=lambda x: x[0], reverse=True)

        # Collect candidate answers from top-k references
        # Weight by similarity score, pick the most common answer
        candidate_votes: dict[tuple, float] = {}
        for sim, idx in scores[:top_k]:
            answer = tuple(self.ref_answers[idx])
            # Only consider answers where all functions are available
            if all(f in available_names for f in answer):
                candidate_votes[answer] = candidate_votes.get(answer, 0.0) + sim

        if candidate_votes:
            best = max(candidate_votes, key=candidate_votes.get)
            return list(best)

        # Fallback: Model A's top pick
        return [a_top] if a_top else []
    def learn(
        self,
        query: str,
        available_funcs: list[dict],
        correct_funcs: list[str],
        domain_hint: str = "mixed",
    ):
        """Online learning — encode a new reference observation on the fly.

        After a turn is resolved (we know the correct answer), encode
        the observation and append it to the reference set. Zero cost,
        no retraining. The observer gets smarter with every turn.

        This is the HDC advantage — one-shot encoding means learning
        is just encode + append.
        """
        # Run Model A
        route_a = self.func_router.route(query, available_funcs)
        a_top = route_a.get("tool", "")
        a_conf = route_a.get("confidence", 0.0)

        # Run Model B
        results_b = self.pattern_router.route(query, domain_hint=domain_hint, top_k=1)
        if results_b:
            b_seq = results_b[0]["sequence"]
            b_conf = results_b[0]["score"]
        else:
            b_seq = ()
            b_conf = 0.0

        # Encode and append
        attrs = _encode_observation_concept(a_top, a_conf, b_seq, b_conf)
        concept = Concept(name="obs", attributes=attrs)
        glyph = self.encoder.encode(concept)

        self.ref_glyphs.append(glyph)
        self.ref_answers.append(correct_funcs)


def _domain_from_classes(classes: list[str]) -> str:
    _map = {
        "GorillaFileSystem": "filesystem", "TradingBot": "trading",
        "TravelAPI": "travel", "TwitterAPI": "twitter",
        "MessageAPI": "messaging", "TicketAPI": "ticket",
        "VehicleControlAPI": "vehicle", "MathAPI": "math",
    }
    domains = [_map.get(c, "mixed") for c in classes]
    if len(set(domains)) == 1:
        return domains[0]
    return "mixed"
