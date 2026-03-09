"""Glyphh Ada 1.0 — Multi-turn tests from BFCL gorilla benchmark.

Tests scenarios end-to-end: setup handler once per scenario, process turns
sequentially so CognitiveLoop tracks state across turns.

Ground truth format per turn: ["cd(folder='document')", "mkdir(dir_name='temp')"]
Model must produce identical call strings — correct functions AND correct arguments.

Architecture:
  HDC (BFCLModelScorer)  → routes which functions (top-N candidates)
  CognitiveLoop          → tracks state (CWD), injects prerequisites
  LLM (Claude Haiku)     → picks from candidates + extracts arg values

Run:
    cd glyphh-models
    PYTHONPATH=../glyphh-runtime python -m pytest bfcl/test_multi_turn.py -v
    PYTHONPATH=../glyphh-runtime python -m pytest bfcl/test_multi_turn.py -v -k "multi_turn_base_0"
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

import sys
sys.path.insert(0, str(Path(__file__).parent))

from multi_turn_handler import MultiTurnHandler, load_func_defs

_BFCL_DIR = Path(__file__).parent


# ── Load BFCL data ───────────────────────────────────────────────────

_BFCL_FILES = {
    "multi_turn_base": "BFCL_v4_multi_turn_base.json",
    "multi_turn_miss_func": "BFCL_v4_multi_turn_miss_func.json",
    "multi_turn_miss_param": "BFCL_v4_multi_turn_miss_param.json",
    "multi_turn_long_context": "BFCL_v4_multi_turn_long_context.json",
}


def _load_scenarios() -> dict[str, dict]:
    """Load full BFCL entries keyed by scenario ID.

    Returns {scenario_id: full_bfcl_entry} with question list, ground_truth, etc.
    """
    scenarios = {}
    for cat, fname in _BFCL_FILES.items():
        path = _BFCL_DIR / "data" / "bfcl" / fname
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                scenarios[entry["id"]] = entry
    return scenarios


_SCENARIOS = _load_scenarios()


def _load_answers() -> dict[str, list]:
    """Load ground truth answers keyed by scenario ID."""
    answers = {}
    for cat, fname in _BFCL_FILES.items():
        path = _BFCL_DIR / "data" / "bfcl" / "possible_answer" / fname
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                entry = json.loads(line)
                answers[entry["id"]] = entry.get("ground_truth", [])
    return answers


_ANSWERS = _load_answers()


# ── Build test cases (one per scenario, not per turn) ────────────────

def _build_test_cases() -> list[tuple[str, dict, list]]:
    """Build (scenario_id, entry, ground_truth_list) tuples."""
    cases = []
    for sid, entry in _SCENARIOS.items():
        gt = _ANSWERS.get(sid, [])
        if gt:
            cases.append((sid, entry, gt))
    return cases


_TEST_CASES = _build_test_cases()


# ── Extract turn queries from BFCL entry format ─────────────────────

def _extract_turn_queries(entry: dict) -> list[str]:
    """Extract the user query from each turn in a BFCL entry.

    BFCL format: entry["question"] = [[{"role": "user", "content": "..."}], ...]
    """
    queries = []
    for turn_messages in entry.get("question", []):
        query = ""
        for msg in turn_messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                query = msg.get("content", "")
                break
        queries.append(query)
    return queries


# ── Tests ─────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "scenario_id, entry, ground_truth_list",
    _TEST_CASES,
    ids=[c[0] for c in _TEST_CASES],
)
def test_scenario(
    scenario_id: str,
    entry: dict,
    ground_truth_list: list,
):
    """Run all turns of a scenario sequentially, comparing each turn's output.

    The handler maintains state (CWD) across turns via CognitiveLoop.
    """
    involved_classes = entry.get("involved_classes", [])
    excluded = entry.get("excluded_function", [])
    func_defs = load_func_defs(involved_classes, excluded)

    handler = MultiTurnHandler()
    handler.setup(entry, func_defs)

    queries = _extract_turn_queries(entry)

    try:
        for turn_idx, query in enumerate(queries):
            if not query:
                continue

            gt_turn = ground_truth_list[turn_idx] if turn_idx < len(ground_truth_list) else []

            predicted_calls = handler.process_turn(query)

            # Compare
            parsed_pred = [_parse_call(c) for c in predicted_calls]
            parsed_gt = [_parse_call(c) for c in gt_turn]

            assert _calls_match(parsed_pred, parsed_gt), (
                f"\nScenario: {scenario_id}, Turn {turn_idx}\n"
                f"Query: '{query[:100]}'\n"
                f"Expected: {gt_turn}\n"
                f"Got: {predicted_calls}"
            )
    finally:
        handler.reset()


# ── Call string parsing ───────────────────────────────────────────────

def _parse_call(call_str: str) -> tuple[str, dict]:
    """Parse 'func(arg1='val1', arg2=val2)' into (func_name, {arg: value})."""
    m = re.match(r"(\w+)\((.*)\)$", call_str, re.DOTALL)
    if not m:
        return (call_str, {})
    func_name = m.group(1)
    args_str = m.group(2).strip()
    if not args_str:
        return (func_name, {})
    try:
        args = eval(f"dict({args_str})")
        return (func_name, args)
    except Exception:
        pass
    # Positional args
    try:
        vals = eval(f"({args_str},)")
        args = {f"_pos_{i}": v for i, v in enumerate(vals)}
        return (func_name, args)
    except Exception:
        return (func_name, {"_raw": args_str})


def _calls_match(pred: list[tuple[str, dict]], gt: list[tuple[str, dict]]) -> bool:
    """Compare predicted and ground truth calls."""
    if len(pred) != len(gt):
        return False
    for (pf, pa), (gf, ga) in zip(pred, gt):
        if pf != gf:
            return False
        p_has_pos = any(k.startswith("_pos_") for k in pa)
        g_has_pos = any(k.startswith("_pos_") for k in ga)
        if p_has_pos or g_has_pos:
            p_vals = sorted(str(v) for v in pa.values())
            g_vals = sorted(str(v) for v in ga.values())
            if p_vals != g_vals:
                return False
        else:
            if pa != ga:
                return False
    return True
