#!/usr/bin/env python3
"""Run Glyphh Ada against BFCL multi-turn entries and validate with gorilla eval checker.

Architecture:
  HDC + CognitiveLoop  → routes which functions to call (deterministic)
  LLM (Claude Haiku)   → extracts argument values only (scoped task)
  CognitiveLoop        → tracks state (CWD), injects prerequisites (cd before grep)

Usage:
    cd glyphh-models/bfcl
    PYTHONPATH=../../glyphh-runtime python3 eval_gorilla.py                    # entry 0
    PYTHONPATH=../../glyphh-runtime python3 eval_gorilla.py --entry 5          # entry 5
    PYTHONPATH=../../glyphh-runtime python3 eval_gorilla.py --category multi_turn_base --max-entries 10
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

# ── Path setup ─────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).parent))

GORILLA_ROOT = Path(__file__).resolve().parent.parent.parent / "gorilla" / "berkeley-function-call-leaderboard"
sys.path.insert(0, str(GORILLA_ROOT))

from handler import BFCLHandler
from run_bfcl import load_multi_turn_func_defs, load_jsonl, _strip_class_prefix, _inject_prerequisites
from tool_execution.llm_extractor import LLMArgumentExtractor

from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
    multi_turn_checker,
    multi_turn_irrelevance_checker,
)

# ── Data loading ───────────────────────────────────────────────────────

DATA_DIR = Path(__file__).parent / "data" / "bfcl"
ANSWER_DIR = DATA_DIR / "possible_answer"

CATEGORIES = {
    "multi_turn_base":         "BFCL_v4_multi_turn_base.json",
    "multi_turn_miss_func":    "BFCL_v4_multi_turn_miss_func.json",
    "multi_turn_miss_param":   "BFCL_v4_multi_turn_miss_param.json",
    "multi_turn_long_context": "BFCL_v4_multi_turn_long_context.json",
}


def load_entry_pairs(category: str) -> list[tuple[dict, dict]]:
    """Load (entry, answer) pairs for a category."""
    data_path = DATA_DIR / CATEGORIES[category]
    answer_path = ANSWER_DIR / CATEGORIES[category]
    entries = load_jsonl(data_path)
    answers = {a["id"]: a for a in load_jsonl(answer_path)}
    return [(e, answers[e["id"]]) for e in entries if e["id"] in answers]


# ── Shared LLM extractor ──────────────────────────────────────────────

_llm: LLMArgumentExtractor | None = None

def get_llm() -> LLMArgumentExtractor:
    global _llm
    if _llm is None:
        _llm = LLMArgumentExtractor()
    return _llm


# ── Format call string ────────────────────────────────────────────────

def _format_call(func_name: str, args: dict) -> str:
    """Format as func_name(arg1='val1', arg2='val2')."""
    if not isinstance(args, dict) or not args:
        return f"{func_name}()"
    parts = []
    for k, v in args.items():
        if isinstance(v, str):
            parts.append(f"{k}='{v}'")
        elif isinstance(v, bool):
            parts.append(f"{k}={v}")
        else:
            parts.append(f"{k}={v}")
    return f"{func_name}({','.join(parts)})"


# ── Eval one entry ─────────────────────────────────────────────────────

def eval_entry(
    bfcl_handler: BFCLHandler,
    entry: dict,
    answer: dict,
    func_defs: list[dict],
    verbose: bool = False,
) -> dict:
    """Run HDC routing + LLM extraction on all turns, validate with gorilla checker."""
    entry_id = entry["id"]
    involved_classes = entry.get("involved_classes", [])
    initial_config = entry.get("initial_config", {})
    turns = entry.get("question", [])
    ground_truth_list = answer.get("ground_truth", [])

    llm = get_llm()

    # Setup CognitiveLoops per class (fresh per entry)
    ctx = bfcl_handler.setup_multi_turn(entry, func_defs)
    available_bare = {_strip_class_prefix(f["name"]) for f in func_defs}

    model_result_decoded: list[list[list[str]]] = []
    already_called: set[str] = set()
    conversation_history: list[str] = []

    for turn_idx, turn_messages in enumerate(turns):
        query = bfcl_handler._extract_turn_query(turn_messages)
        if not query:
            model_result_decoded.append([])
            conversation_history.append("")
            continue

        # ── STEP 1: HDC routes which functions ──
        funcs = bfcl_handler.route_turn(query, ctx)

        # Inject prerequisites (auth, cd, etc.)
        for cls in involved_classes:
            funcs = _inject_prerequisites(funcs, cls, available_bare, already_called)

        # Get current state from CognitiveLoop
        loops = ctx.get("loops", {})
        current_state = None
        if funcs:
            first_cls = funcs[0].split(".")[0]
            loop = loops.get(first_cls)
            if loop:
                current_state = dict(loop._state)

        if verbose:
            cwd = current_state.get("primary", "") if current_state else ""
            print(f"\n  Turn {turn_idx} [CWD: {cwd}]: {query[:80]}")
            bare_funcs = [_strip_class_prefix(f) for f in funcs]
            print(f"    HDC routed: {bare_funcs}")

        # ── STEP 2: LLM extracts argument values only ──
        turn_fdefs = []
        for fname in funcs:
            fdef = next((f for f in func_defs if f["name"] == fname), None)
            if fdef:
                turn_fdefs.append(fdef)

        if turn_fdefs and query:
            calls = llm.extract_multi_turn(
                query, turn_fdefs, conversation_history,
                initial_config=initial_config,
                current_state=current_state,
            )
        else:
            calls = [{_strip_class_prefix(f): {}} for f in funcs]

        # Track what was called
        for call in calls:
            for fname in call:
                already_called.add(fname)

        # ── STEP 3: Update CognitiveLoop state ──
        bfcl_handler.update_turn_state(ctx, calls)

        # Format as call strings
        call_strings = []
        for call in calls:
            for fn, args in call.items():
                call_strings.append(_format_call(fn, args if isinstance(args, dict) else {}))

        if verbose:
            gt = ground_truth_list[turn_idx] if turn_idx < len(ground_truth_list) else []
            print(f"    GT:   {gt}")
            print(f"    Pred: {call_strings}")
            # Show CWD change
            if funcs:
                first_cls = funcs[0].split(".")[0]
                loop = loops.get(first_cls)
                if loop and current_state:
                    new_cwd = loop._state.get("primary", "")
                    old_cwd = current_state.get("primary", "")
                    if new_cwd != old_cwd:
                        print(f"    CWD: {old_cwd} → {new_cwd}")

        # Wrap for gorilla checker
        if call_strings:
            model_result_decoded.append([call_strings])
        else:
            model_result_decoded.append([])

        conversation_history.append(query)

    # Run gorilla checker
    result = multi_turn_checker(
        multi_turn_model_result_list_decoded=model_result_decoded,
        multi_turn_ground_truth_list=ground_truth_list,
        test_entry=entry,
        test_category=entry_id.rsplit("_", 1)[0],
        model_name="glyphh_ada",
    )

    irr_result = multi_turn_irrelevance_checker(
        multi_turn_model_result_list_decoded=model_result_decoded,
        multi_turn_ground_truth_list=ground_truth_list,
    )

    valid = result.get("valid", False) and irr_result.get("valid", True)
    return {
        "id": entry_id,
        "valid": valid,
        "checker_result": result,
        "irrelevance_result": irr_result,
        "model_output": model_result_decoded,
    }


# ── Main ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Gorilla eval for Glyphh Ada")
    parser.add_argument("--category", default="multi_turn_base", choices=list(CATEGORIES.keys()))
    parser.add_argument("--entry", type=int, default=None, help="Specific entry index (0-based)")
    parser.add_argument("--max-entries", type=int, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    pairs = load_entry_pairs(args.category)
    if args.entry is not None:
        pairs = [pairs[args.entry]]
    elif args.max_entries:
        pairs = pairs[:args.max_entries]

    print(f"Category: {args.category}")
    print(f"Entries: {len(pairs)}")
    print()

    # Create BFCLHandler with LLM extractor for arg extraction
    llm = get_llm()
    bfcl_handler = BFCLHandler(extractor=llm)

    passed = 0
    failed = 0
    for entry, answer in pairs:
        entry_id = entry["id"]
        involved = entry.get("involved_classes", [])
        excluded = entry.get("excluded_function", [])
        func_defs = load_multi_turn_func_defs(involved, excluded)

        print(f"--- {entry_id} ---")
        result = eval_entry(bfcl_handler, entry, answer, func_defs, verbose=args.verbose)

        if result["valid"]:
            print(f"  PASS")
            passed += 1
        else:
            print(f"  FAIL: {result['checker_result'].get('error_message', 'unknown')}")
            if args.verbose and "details" in result["checker_result"]:
                details = result["checker_result"]["details"]
                if "differences" in details:
                    try:
                        print(f"    Differences: {json.dumps(details['differences'], indent=4, default=str)[:500]}")
                    except Exception:
                        print(f"    Differences: {str(details['differences'])[:500]}")
            failed += 1

    total = passed + failed
    print(f"\n{'='*50}")
    print(f"Results: {passed}/{total} passed ({100*passed/total:.1f}%)" if total else "No entries")


if __name__ == "__main__":
    main()
