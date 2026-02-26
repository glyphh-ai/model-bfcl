#!/usr/bin/env python3
"""
Cross-validated Observer evaluation on all 4 multi-turn categories.

Leave-one-out: train on 3 categories, test on the held-out 4th.
This gives honest generalization numbers — no memorizing the answer key.

Also runs the "cheating" mode (train on all, test on all) for comparison.

Usage:
    .venv/bin/python3 glyphh-models/bfcl/eval_observer_cv.py
"""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from handler import GlyphhBFCLHandler
from pattern_encoder import PatternRouter
from observer import Observer, _domain_from_classes
from multi_turn_handler import (
    get_available_functions,
    extract_turn_query,
    parse_ground_truth_step,
)

DATA_DIR = Path(__file__).parent / "data" / "bfcl"

CATEGORIES = [
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
]

QUESTION_FILES = {
    "multi_turn_base": "BFCL_v3_multi_turn_base.json",
    "multi_turn_miss_func": "BFCL_v3_multi_turn_miss_func.json",
    "multi_turn_miss_param": "BFCL_v3_multi_turn_miss_param.json",
    "multi_turn_long_context": "BFCL_v3_multi_turn_long_context.json",
}
ANSWER_FILES = {
    "multi_turn_base": "possible_answer/BFCL_v3_multi_turn_base.json",
    "multi_turn_miss_func": "possible_answer/BFCL_v3_multi_turn_miss_func.json",
    "multi_turn_miss_param": "possible_answer/BFCL_v3_multi_turn_miss_param.json",
    "multi_turn_long_context": "possible_answer/BFCL_v3_multi_turn_long_context.json",
}


def load_jsonl(path: Path) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def eval_category(observer: Observer, category: str, top_k: int = 3) -> dict:
    """Evaluate observer on a single category. Returns detailed results."""
    qpath = DATA_DIR / QUESTION_FILES[category]
    apath = DATA_DIR / ANSWER_FILES[category]

    if not qpath.exists() or not apath.exists():
        print(f"  Data not found for {category}")
        return {}

    questions = {}
    for e in load_jsonl(qpath):
        questions[e.get("id", "")] = e

    answers = {}
    for a in load_jsonl(apath):
        answers[a.get("id", "")] = a

    total_entries = 0
    correct_entries = 0
    total_turns = 0
    correct_turns = 0
    single_correct = 0
    single_total = 0
    multi_correct = 0
    multi_total = 0

    for eid, answer in answers.items():
        entry = questions.get(eid, {})
        gt = answer.get("ground_truth", [])
        turns = entry.get("question", [])
        available_funcs = get_available_functions(entry)

        if not available_funcs or not gt:
            continue

        total_entries += 1
        available_names = {f["name"] for f in available_funcs}
        involved = entry.get("involved_classes", [])
        domain = _domain_from_classes(involved)

        entry_all_correct = True

        for ti, step in enumerate(gt):
            if not step:
                continue

            query = extract_turn_query(turns[ti] if ti < len(turns) else "")
            if not query:
                continue

            expected_calls = parse_ground_truth_step(step)
            if not expected_calls:
                continue

            expected_funcs = []
            for call in expected_calls:
                expected_funcs.extend(call.keys())

            total_turns += 1
            is_multi = len(expected_funcs) > 1

            if is_multi:
                multi_total += 1
            else:
                single_total += 1

            predicted = observer.decide(
                query, available_funcs, available_names,
                domain_hint=domain, top_k=top_k,
            )

            if list(predicted) == list(expected_funcs):
                correct_turns += 1
                if is_multi:
                    multi_correct += 1
                else:
                    single_correct += 1
            else:
                entry_all_correct = False

        if entry_all_correct:
            correct_entries += 1

    return {
        "category": category,
        "entries": total_entries,
        "entry_acc": correct_entries / total_entries if total_entries else 0,
        "correct_entries": correct_entries,
        "turns": total_turns,
        "turn_acc": correct_turns / total_turns if total_turns else 0,
        "correct_turns": correct_turns,
        "single_total": single_total,
        "single_acc": single_correct / single_total if single_total else 0,
        "multi_total": multi_total,
        "multi_acc": multi_correct / multi_total if multi_total else 0,
    }
def eval_category_online(observer: Observer, category: str, top_k: int = 3) -> dict:
    """Evaluate with online learning — observer learns from each turn after predicting.

    Predict first, then learn. Honest because the prediction happens before
    the ground truth is seen. But the observer accumulates experience from
    the test set as it goes.
    """
    qpath = DATA_DIR / QUESTION_FILES[category]
    apath = DATA_DIR / ANSWER_FILES[category]

    if not qpath.exists() or not apath.exists():
        print(f"  Data not found for {category}")
        return {}

    questions = {}
    for e in load_jsonl(qpath):
        questions[e.get("id", "")] = e

    answers = {}
    for a in load_jsonl(apath):
        answers[a.get("id", "")] = a

    total_entries = 0
    correct_entries = 0
    total_turns = 0
    correct_turns = 0
    single_correct = 0
    single_total = 0
    multi_correct = 0
    multi_total = 0
    refs_before = len(observer.ref_glyphs)

    for eid, answer in answers.items():
        entry = questions.get(eid, {})
        gt = answer.get("ground_truth", [])
        turns = entry.get("question", [])
        available_funcs = get_available_functions(entry)

        if not available_funcs or not gt:
            continue

        total_entries += 1
        available_names = {f["name"] for f in available_funcs}
        involved = entry.get("involved_classes", [])
        domain = _domain_from_classes(involved)

        entry_all_correct = True

        for ti, step in enumerate(gt):
            if not step:
                continue

            query = extract_turn_query(turns[ti] if ti < len(turns) else "")
            if not query:
                continue

            expected_calls = parse_ground_truth_step(step)
            if not expected_calls:
                continue

            expected_funcs = []
            for call in expected_calls:
                expected_funcs.extend(call.keys())

            total_turns += 1
            is_multi = len(expected_funcs) > 1

            if is_multi:
                multi_total += 1
            else:
                single_total += 1

            # PREDICT FIRST (before learning)
            predicted = observer.decide(
                query, available_funcs, available_names,
                domain_hint=domain, top_k=top_k,
            )

            hit = list(predicted) == list(expected_funcs)
            if hit:
                correct_turns += 1
                if is_multi:
                    multi_correct += 1
                else:
                    single_correct += 1
            else:
                entry_all_correct = False

            # LEARN AFTER (encode ground truth as new reference)
            observer.learn(
                query, available_funcs, expected_funcs,
                domain_hint=domain,
            )

        if entry_all_correct:
            correct_entries += 1

    refs_after = len(observer.ref_glyphs)

    return {
        "category": category,
        "entries": total_entries,
        "entry_acc": correct_entries / total_entries if total_entries else 0,
        "correct_entries": correct_entries,
        "turns": total_turns,
        "turn_acc": correct_turns / total_turns if total_turns else 0,
        "correct_turns": correct_turns,
        "single_total": single_total,
        "single_acc": single_correct / single_total if single_total else 0,
        "multi_total": multi_total,
        "multi_acc": multi_correct / multi_total if multi_total else 0,
        "refs_before": refs_before,
        "refs_after": refs_after,
    }


def main():
    print("=" * 70)
    print("  OBSERVER (MODEL C) — CROSS-VALIDATED MULTI-TURN EVALUATION")
    print("=" * 70)

    # Shared Model A and Model B
    print("\nInitializing Model A (function router)...")
    func_router = GlyphhBFCLHandler(threshold=0.15)
    print("Initializing Model B (pattern router)...")
    pattern_router = PatternRouter()
    pattern_router.build(min_count=2)
    print(f"  {len(pattern_router.patterns)} patterns loaded")

    # ── CROSS-VALIDATION: leave-one-out ──
    print("\n" + "─" * 70)
    print("  CROSS-VALIDATION (train on 3, test on held-out 1)")
    print("─" * 70)

    cv_results = []
    total_cv_time = 0

    for held_out in CATEGORIES:
        train_cats = [c for c in CATEGORIES if c != held_out]
        print(f"\n  Held out: {held_out}")
        print(f"  Training: {', '.join(train_cats)}")

        t0 = time.perf_counter()

        observer = Observer(func_router, pattern_router)
        observer.build(exclude_categories=[held_out])
        n_refs = len(observer.ref_glyphs)
        build_time = time.perf_counter() - t0

        print(f"  References: {n_refs} (built in {build_time:.1f}s)")

        t1 = time.perf_counter()
        result = eval_category(observer, held_out, top_k=3)
        eval_time = time.perf_counter() - t1
        total_cv_time += (time.perf_counter() - t0)

        result["build_time"] = build_time
        result["eval_time"] = eval_time
        result["n_refs"] = n_refs
        cv_results.append(result)

        print(f"  Entry acc: {result['entry_acc']:.1%} ({result['correct_entries']}/{result['entries']})")
        print(f"  Turn acc:  {result['turn_acc']:.1%} ({result['correct_turns']}/{result['turns']})")
        print(f"  Single:    {result['single_acc']:.1%} ({result['single_total']})")
        print(f"  Multi:     {result['multi_acc']:.1%} ({result['multi_total']})")
        print(f"  Eval time: {eval_time:.1f}s")

    # CV summary
    total_turns_cv = sum(r["turns"] for r in cv_results)
    correct_turns_cv = sum(r["correct_turns"] for r in cv_results)
    total_entries_cv = sum(r["entries"] for r in cv_results)
    correct_entries_cv = sum(r["correct_entries"] for r in cv_results)

    print(f"\n  {'─' * 50}")
    print(f"  CV TOTALS:")
    print(f"    Entry accuracy: {correct_entries_cv}/{total_entries_cv} = {correct_entries_cv/total_entries_cv:.1%}")
    print(f"    Turn accuracy:  {correct_turns_cv}/{total_turns_cv} = {correct_turns_cv/total_turns_cv:.1%}")
    print(f"    Total time:     {total_cv_time:.1f}s")

    # ── FULL (train on all, test on all — for comparison) ──
    print("\n" + "─" * 70)
    print("  FULL (train on all, test on all — memorization baseline)")
    print("─" * 70)

    t0 = time.perf_counter()
    observer_full = Observer(func_router, pattern_router)
    observer_full.build()
    build_time = time.perf_counter() - t0
    print(f"  References: {len(observer_full.ref_glyphs)} (built in {build_time:.1f}s)")

    full_results = []
    for cat in CATEGORIES:
        t1 = time.perf_counter()
        result = eval_category(observer_full, cat, top_k=3)
        eval_time = time.perf_counter() - t1
        result["eval_time"] = eval_time
        full_results.append(result)
        print(f"  {cat:<30} entry={result['entry_acc']:.1%}  turn={result['turn_acc']:.1%}  ({eval_time:.1f}s)")

    total_turns_full = sum(r["turns"] for r in full_results)
    correct_turns_full = sum(r["correct_turns"] for r in full_results)
    total_entries_full = sum(r["entries"] for r in full_results)
    correct_entries_full = sum(r["correct_entries"] for r in full_results)

    print(f"\n  FULL TOTALS:")
    print(f"    Entry accuracy: {correct_entries_full}/{total_entries_full} = {correct_entries_full/total_entries_full:.1%}")
    print(f"    Turn accuracy:  {correct_turns_full}/{total_turns_full} = {correct_turns_full/total_turns_full:.1%}")

    # ── ONLINE LEARNING: CV + learn from each turn after predicting ──
    print("\n" + "─" * 70)
    print("  ONLINE LEARNING (CV base + learn from test turns after predicting)")
    print("─" * 70)

    online_results = []
    total_online_time = 0

    for held_out in CATEGORIES:
        train_cats = [c for c in CATEGORIES if c != held_out]
        print(f"\n  Held out: {held_out}")

        t0 = time.perf_counter()

        observer_online = Observer(func_router, pattern_router)
        observer_online.build(exclude_categories=[held_out])
        n_refs = len(observer_online.ref_glyphs)

        t1 = time.perf_counter()
        result = eval_category_online(observer_online, held_out, top_k=3)
        eval_time = time.perf_counter() - t1
        total_online_time += (time.perf_counter() - t0)

        result["eval_time"] = eval_time
        result["n_refs_start"] = n_refs
        online_results.append(result)

        print(f"  Refs: {result['refs_before']} → {result['refs_after']} (+{result['refs_after'] - result['refs_before']})")
        print(f"  Entry acc: {result['entry_acc']:.1%} ({result['correct_entries']}/{result['entries']})")
        print(f"  Turn acc:  {result['turn_acc']:.1%} ({result['correct_turns']}/{result['turns']})")
        print(f"  Single:    {result['single_acc']:.1%} ({result['single_total']})")
        print(f"  Multi:     {result['multi_acc']:.1%} ({result['multi_total']})")

    total_turns_online = sum(r["turns"] for r in online_results)
    correct_turns_online = sum(r["correct_turns"] for r in online_results)
    total_entries_online = sum(r["entries"] for r in online_results)
    correct_entries_online = sum(r["correct_entries"] for r in online_results)

    print(f"\n  {'─' * 50}")
    print(f"  ONLINE TOTALS:")
    print(f"    Entry accuracy: {correct_entries_online}/{total_entries_online} = {correct_entries_online/total_entries_online:.1%}")
    print(f"    Turn accuracy:  {correct_turns_online}/{total_turns_online} = {correct_turns_online/total_turns_online:.1%}")
    print(f"    Total time:     {total_online_time:.1f}s")

    # ── COMPARISON ──
    print("\n" + "=" * 70)
    print("  COMPARISON")
    print("=" * 70)
    print(f"  {'Mode':<25} {'Entry Acc':>12} {'Turn Acc':>12}")
    print(f"  {'─' * 49}")
    print(f"  {'Cross-validated':<25} {correct_entries_cv/total_entries_cv:>11.1%} {correct_turns_cv/total_turns_cv:>11.1%}")
    print(f"  {'CV + Online Learning':<25} {correct_entries_online/total_entries_online:>11.1%} {correct_turns_online/total_turns_online:>11.1%}")
    print(f"  {'Full (memorized)':<25} {correct_entries_full/total_entries_full:>11.1%} {correct_turns_full/total_turns_full:>11.1%}")
    gap_cv_online = (correct_turns_online/total_turns_online - correct_turns_cv/total_turns_cv) * 100
    gap_online_full = (correct_turns_full/total_turns_full - correct_turns_online/total_turns_online) * 100
    print(f"  {'─' * 49}")
    print(f"  {'CV → Online boost':<25} {'':>12} {gap_cv_online:>+10.1f}pp")
    print(f"  {'Online → Full gap':<25} {'':>12} {gap_online_full:>+10.1f}pp")
    print("=" * 70)


if __name__ == "__main__":
    main()
