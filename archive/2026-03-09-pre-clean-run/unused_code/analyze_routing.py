#!/usr/bin/env python3
"""
Analyze routing accuracy for multi-turn entries.

Compares HDC routing (function names only) against ground truth per turn.
No LLM needed — pure routing analysis.

Usage:
    PYTHONPATH=../../glyphh-runtime python3 analyze_routing.py
    PYTHONPATH=../../glyphh-runtime python3 analyze_routing.py --max-entries 20
    PYTHONPATH=../../glyphh-runtime python3 analyze_routing.py --verbose
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from handler import BFCLHandler
from run_bfcl import (
    DATA_DIR, BFCL_FILES, BFCL_ANSWER_FILES,
    load_multi_turn_func_defs, load_jsonl,
)


def _strip_class(name: str) -> str:
    """Remove class prefix: GorillaFileSystem.cd -> cd"""
    return name.split(".")[-1]


def _gt_func_names(turn_gt: list[str]) -> list[str]:
    """Extract function names from ground truth strings like cd(folder='x')."""
    names = []
    for call_str in turn_gt:
        match = re.match(r"(\w+)\(", call_str)
        if match:
            names.append(match.group(1))
    return names


def analyze(max_entries=None, verbose=False, category="multi_turn_base"):
    handler = BFCLHandler(confidence_threshold=0.22)

    filepath = DATA_DIR / BFCL_FILES[category]
    entries = load_jsonl(filepath)

    answer_path = DATA_DIR / BFCL_ANSWER_FILES[category]
    gt_by_id = {}
    if answer_path.exists():
        for ae in load_jsonl(answer_path):
            gt_by_id[ae.get("id", "")] = ae.get("ground_truth", [])

    if max_entries:
        entries = entries[:max_entries]

    # Aggregate stats
    total_turns = 0
    exact_match_turns = 0
    total_entries = 0
    exact_match_entries = 0

    missing_counter = Counter()  # function names we missed
    extra_counter = Counter()    # function names we added incorrectly
    wrong_counter = Counter()    # (predicted, expected) pairs

    # Per-class stats
    class_stats = defaultdict(lambda: {"turns": 0, "exact": 0})

    # Error details for verbose
    errors = []

    for i, entry in enumerate(entries):
        entry_id = entry.get("id", f"{category}_{i}")
        involved = entry.get("involved_classes", [])
        excluded = entry.get("excluded_function", [])
        path = entry.get("path", [])

        if not path or not involved:
            continue

        func_defs = load_multi_turn_func_defs(involved, excluded)
        gt_turns = gt_by_id.get(entry_id, [])

        if not gt_turns:
            continue

        total_entries += 1
        ctx = handler.setup_multi_turn(entry, func_defs)
        turns = entry.get("question", [])

        entry_perfect = True

        for t_idx, turn_messages in enumerate(turns):
            if t_idx >= len(gt_turns):
                break

            gt_funcs = _gt_func_names(gt_turns[t_idx])
            if not gt_funcs and not turn_messages:
                continue

            total_turns += 1

            # Get query
            query = handler._extract_turn_query(turn_messages) if turn_messages else ""

            # Route
            routed = handler.route_turn(query, ctx) if query else []
            routed_bare = [_strip_class(f) for f in routed]

            # Compare: exact set match (order-independent)
            gt_set = sorted(gt_funcs)
            pred_set = sorted(routed_bare)

            # Track class
            cls = involved[0] if involved else "unknown"
            class_stats[cls]["turns"] += 1

            if gt_set == pred_set:
                exact_match_turns += 1
                class_stats[cls]["exact"] += 1
            else:
                entry_perfect = False

                # Missing = in GT but not in predicted
                gt_remaining = list(gt_funcs)
                pred_remaining = list(routed_bare)
                matched = []
                for f in list(pred_remaining):
                    if f in gt_remaining:
                        gt_remaining.remove(f)
                        pred_remaining.remove(f)
                        matched.append(f)

                for f in gt_remaining:
                    missing_counter[f] += 1
                for f in pred_remaining:
                    extra_counter[f] += 1

                if verbose:
                    errors.append({
                        "entry": entry_id,
                        "turn": t_idx,
                        "query": query[:120],
                        "expected": gt_funcs,
                        "predicted": routed_bare,
                        "missing": gt_remaining,
                        "extra": pred_remaining,
                    })

            # Override CognitiveLoop state with GT args so subsequent turns
            # route correctly. step() already updated state with SDK-extracted args,
            # but those are often wrong. GT args give correct state for next turn.
            gt_calls = []
            for call_str in gt_turns[t_idx]:
                match = re.match(r"(\w+)\((.*)\)", call_str, re.DOTALL)
                if match:
                    fname = match.group(1)
                    args_str = match.group(2)
                    args = {}
                    for kv in re.findall(r"(\w+)='([^']*)'", args_str):
                        args[kv[0]] = kv[1]
                    for kv in re.findall(r"(\w+)=(\d+)", args_str):
                        args[kv[0]] = int(kv[1])
                    gt_calls.append({fname: args})
            if gt_calls:
                handler.update_turn_state(ctx, gt_calls)

        if entry_perfect:
            exact_match_entries += 1

    # Report
    print(f"\n{'='*60}")
    print(f"ROUTING ANALYSIS — {category}")
    print(f"{'='*60}")
    print(f"Entries: {exact_match_entries}/{total_entries} perfect ({100*exact_match_entries/total_entries:.1f}%)")
    print(f"Turns:   {exact_match_turns}/{total_turns} exact match ({100*exact_match_turns/total_turns:.1f}%)")

    print(f"\n── Missing functions (in GT, not predicted) ──")
    for func, count in missing_counter.most_common(20):
        print(f"  {func}: {count}")

    print(f"\n── Extra functions (predicted, not in GT) ──")
    for func, count in extra_counter.most_common(20):
        print(f"  {func}: {count}")

    print(f"\n── Per-class accuracy ──")
    for cls, stats in sorted(class_stats.items()):
        pct = 100 * stats["exact"] / stats["turns"] if stats["turns"] else 0
        print(f"  {cls}: {stats['exact']}/{stats['turns']} ({pct:.0f}%)")

    if verbose and errors:
        print(f"\n── Error details (first 30) ──")
        for e in errors[:30]:
            print(f"\n  [{e['entry']}] turn {e['turn']}")
            print(f"    Q: {e['query']}")
            print(f"    Expected:  {e['expected']}")
            print(f"    Predicted: {e['predicted']}")
            if e['missing']:
                print(f"    Missing:   {e['missing']}")
            if e['extra']:
                print(f"    Extra:     {e['extra']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-entries", type=int, default=None)
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--category", default="multi_turn_base")
    args = parser.parse_args()
    analyze(max_entries=args.max_entries, verbose=args.verbose, category=args.category)
