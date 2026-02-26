"""
Dual-model multi-turn handler for BFCL V4.

Two Glyphh models, one substrate:
  Model A (handler.py)         — function router: query → per-function scores
  Model B (pattern_encoder.py) — pattern router:  query → per-sequence scores

Beam search combines both signals:
  - Each beam candidate is a possible function sequence for the current turn
  - Candidates come from BOTH models (Model A proposes singles, Model B proposes sequences)
  - Each candidate is scored by weighted combination of both models
  - Beam prunes to top-k, best candidate wins

This avoids the "pick one model" problem. Both models vote, beam resolves.
"""

import time
from typing import Any

from handler import GlyphhBFCLHandler
from pattern_encoder import PatternRouter
from multi_turn_handler import (
    get_available_functions,
    extract_turn_query,
    parse_ground_truth_step,
)


class DualModelHandler:
    """Two-Glyphh-model handler with beam search for multi-turn BFCL."""

    def __init__(
        self,
        threshold: float = 0.15,
        beam_width: int = 5,
        pattern_min_count: int = 2,
        model_a_weight: float = 0.55,
        model_b_weight: float = 0.45,
    ):
        self.func_router = GlyphhBFCLHandler(threshold=threshold)
        self.pattern_router = PatternRouter()
        self.pattern_router.build(min_count=pattern_min_count)
        self.beam_width = beam_width
        self.w_a = model_a_weight
        self.w_b = model_b_weight

    def eval_entry(
        self,
        entry: dict,
        ground_truth: list,
        llm=None,
    ) -> dict:
        """Evaluate a single multi-turn entry."""
        turns = entry.get("question", [])
        available_funcs = get_available_functions(entry)
        involved_classes = entry.get("involved_classes", [])

        if not available_funcs:
            return _empty_result(entry, turns)

        available_names = {f["name"] for f in available_funcs}
        domain_hint = _domain_from_classes(involved_classes)

        turn_results = []
        correct_turns = 0

        for turn_idx, turn in enumerate(turns):
            query = extract_turn_query(turn)
            if not query:
                turn_results.append({
                    "turn": turn_idx, "correct": False, "reason": "no_query",
                })
                continue

            expected_step = ground_truth[turn_idx] if turn_idx < len(ground_truth) else []
            expected_calls = parse_ground_truth_step(expected_step)

            if not expected_calls:
                turn_results.append({
                    "turn": turn_idx, "correct": True,
                    "expected": [], "predicted": [],
                    "reason": "no_call_expected",
                })
                correct_turns += 1
                continue

            expected_funcs = []
            for call in expected_calls:
                expected_funcs.extend(call.keys())

            # --- BEAM SEARCH ACROSS BOTH MODELS ---
            predicted_funcs = self._beam_route(
                query, available_funcs, available_names, domain_hint,
            )

            func_correct = list(predicted_funcs) == list(expected_funcs)

            turn_results.append({
                "turn": turn_idx,
                "correct": func_correct,
                "expected_funcs": expected_funcs,
                "predicted_funcs": predicted_funcs,
                "mode": "beam_dual",
            })

            if func_correct:
                correct_turns += 1

        total_turns = len(turn_results)
        return {
            "id": entry.get("id", "?"),
            "turns": len(turns),
            "correct_turns": correct_turns,
            "total_turns": total_turns,
            "correct": correct_turns == total_turns,
            "details": turn_results,
        }

    def _beam_route(
        self,
        query: str,
        available_funcs: list[dict],
        available_names: set[str],
        domain_hint: str,
    ) -> list[str]:
        """Beam search combining Model A and Model B signals.

        Both models produce scores on different scales. We normalize:
          - Model A singles: score as-is (range ~0.2-0.5)
          - Model B multi-step: w_a * max_a + w_b * pattern_score
            This must be on the SAME scale as Model A singles.

        To normalize: Model A singles also get the weighted treatment:
          single score = w_a * model_a(f) + w_b * best_single_pattern(f)
        This way both single and multi candidates use the same formula.
        """
        # ── Model A: per-function scores (always runs) ──
        func_route = self.func_router.route(query, available_funcs)
        model_a_top = func_route.get("tool")
        model_a_scores = {
            s["function"]: s["score"]
            for s in func_route.get("all_scores", [])
        }

        if not model_a_top:
            return []

        # ── Model B: pattern scores ──
        pattern_results = self.pattern_router.route(
            query, domain_hint=domain_hint, top_k=self.beam_width * 3,
        )

        # ── Build candidate beam ──
        # ALL candidates use same scoring formula:
        #   score = w_a * model_a_signal + w_b * model_b_signal
        candidates: list[tuple[list[str], float]] = []

        # Build a lookup: func_name → best Model B score for single-func pattern
        single_b_scores: dict[str, float] = {}
        for pr in pattern_results:
            if len(pr["sequence"]) == 1:
                fname = pr["sequence"][0]
                single_b_scores[fname] = max(
                    single_b_scores.get(fname, 0.0), pr["score"]
                )

        # Model A's top-5 as single-func candidates
        top_a = sorted(model_a_scores.items(), key=lambda x: x[1], reverse=True)
        for fname, a_score in top_a[:5]:
            b_score = single_b_scores.get(fname, 0.0)
            combined = self.w_a * a_score + self.w_b * b_score
            candidates.append(([fname], combined))

        # Model B single-func patterns (may include functions not in Model A's top-5)
        for pr in pattern_results:
            if len(pr["sequence"]) != 1:
                continue
            fname = pr["sequence"][0]
            if fname not in available_names:
                continue
            a_score = model_a_scores.get(fname, 0.0)
            combined = self.w_a * a_score + self.w_b * pr["score"]
            candidates.append(([fname], combined))

        # Model B multi-step candidates
        for pr in pattern_results:
            filtered = [f for f in pr["sequence"] if f in available_names]
            if len(filtered) < 2:
                continue
            max_a = max(model_a_scores.get(f, 0.0) for f in filtered)
            combined = self.w_a * max_a + self.w_b * pr["score"]
            candidates.append((filtered, combined))

        # ── Beam: sort, deduplicate, top-k ──
        candidates.sort(key=lambda x: x[1], reverse=True)

        seen = set()
        beam = []
        for seq, score in candidates:
            key = tuple(seq)
            if key not in seen:
                seen.add(key)
                beam.append((seq, score))
            if len(beam) >= self.beam_width:
                break

        return beam[0][0] if beam else [model_a_top]




def _empty_result(entry: dict, turns: list) -> dict:
    return {
        "id": entry.get("id", "?"),
        "turns": len(turns),
        "correct_turns": 0,
        "total_turns": len(turns),
        "correct": False,
        "details": [],
    }


_CLASS_DOMAIN_MAP = {
    "GorillaFileSystem": "filesystem",
    "TradingBot": "trading",
    "TravelAPI": "travel",
    "TwitterAPI": "twitter",
    "MessageAPI": "messaging",
    "TicketAPI": "ticket",
    "VehicleControlAPI": "vehicle",
    "MathAPI": "math",
}


def _domain_from_classes(classes: list[str]) -> str:
    domains = [_CLASS_DOMAIN_MAP.get(c, "mixed") for c in classes]
    if len(set(domains)) == 1:
        return domains[0]
    return "mixed"
