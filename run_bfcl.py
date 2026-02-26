#!/usr/bin/env python3
"""
Run Glyphh HDC against the full BFCL V4 benchmark.

BFCL V4 Overall = (Agentic × 40%) + (Multi-Turn × 30%) + (Live × 10%) + (Non-Live × 10%) + (Hallucination × 10%)

Glyphh handles routing (Non-Live, Live, Hallucination = 30% of overall).
A cheap LLM handles argument extraction + multi-turn + agentic (70% of overall).

Usage:
    # Glyphh-only routing (no API key needed)
    python run_bfcl.py --routing-only

    # Full V4 eval with cheap LLM (needs OPENAI_API_KEY or GOOGLE_API_KEY)
    python run_bfcl.py --full-v4 --llm-provider openai --llm-model gpt-4.1-nano

    # Specific categories
    python run_bfcl.py --categories simple java javascript live_simple

    # Download data only
    python run_bfcl.py --download-only
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from handler import GlyphhBFCLHandler

# Use V4 data from the bfcl_eval package (installed via pip)
import bfcl_eval
DATA_DIR = Path(bfcl_eval.__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"

HF_BASE = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main"

# ── All BFCL V4 data files (using actual V4 data) ──

BFCL_FILES = {
    # Non-Live AST (Glyphh routing) — V4 splits simple by language
    "simple": "BFCL_v4_simple_python.json",
    "multiple": "BFCL_v4_multiple.json",
    "parallel": "BFCL_v4_parallel.json",
    "parallel_multiple": "BFCL_v4_parallel_multiple.json",
    "java": "BFCL_v4_simple_java.json",
    "javascript": "BFCL_v4_simple_javascript.json",
    # Irrelevance / Hallucination
    "irrelevance": "BFCL_v4_irrelevance.json",
    "live_irrelevance": "BFCL_v4_live_irrelevance.json",
    # Live AST
    "live_simple": "BFCL_v4_live_simple.json",
    "live_multiple": "BFCL_v4_live_multiple.json",
    "live_parallel": "BFCL_v4_live_parallel.json",
    "live_parallel_multiple": "BFCL_v4_live_parallel_multiple.json",
    # Live relevance (non-scoring)
    "live_relevance": "BFCL_v4_live_relevance.json",
    # Multi-turn
    "multi_turn_base": "BFCL_v4_multi_turn_base.json",
    "multi_turn_miss_func": "BFCL_v4_multi_turn_miss_func.json",
    "multi_turn_miss_param": "BFCL_v4_multi_turn_miss_param.json",
    "multi_turn_long_context": "BFCL_v4_multi_turn_long_context.json",
}

BFCL_ANSWER_FILES = {
    "simple": "possible_answer/BFCL_v4_simple_python.json",
    "multiple": "possible_answer/BFCL_v4_multiple.json",
    "parallel": "possible_answer/BFCL_v4_parallel.json",
    "parallel_multiple": "possible_answer/BFCL_v4_parallel_multiple.json",
    "java": "possible_answer/BFCL_v4_simple_java.json",
    "javascript": "possible_answer/BFCL_v4_simple_javascript.json",
    "live_simple": "possible_answer/BFCL_v4_live_simple.json",
    "live_multiple": "possible_answer/BFCL_v4_live_multiple.json",
    "live_parallel": "possible_answer/BFCL_v4_live_parallel.json",
    "live_parallel_multiple": "possible_answer/BFCL_v4_live_parallel_multiple.json",
    "multi_turn_base": "possible_answer/BFCL_v4_multi_turn_base.json",
    "multi_turn_miss_func": "possible_answer/BFCL_v4_multi_turn_miss_func.json",
    "multi_turn_miss_param": "possible_answer/BFCL_v4_multi_turn_miss_param.json",
    "multi_turn_long_context": "possible_answer/BFCL_v4_multi_turn_long_context.json",
}

# ── V4 scoring structure ──

V4_NONLIVE_CATS = ["simple", "java", "javascript", "multiple", "parallel", "parallel_multiple"]
V4_HALLUCINATION_CATS = ["irrelevance", "live_irrelevance"]
V4_LIVE_CATS = ["live_simple", "live_multiple", "live_parallel", "live_parallel_multiple"]
V4_MULTITURN_CATS = ["multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"]

# Glyphh-only categories (no LLM needed)
GLYPHH_ONLY_CATS = V4_NONLIVE_CATS + V4_HALLUCINATION_CATS + V4_LIVE_CATS

# Categories that need multi-function routing
MULTI_ROUTE_CATS = {"parallel_multiple", "live_parallel_multiple"}

# Categories that are irrelevance detection
IRRELEVANCE_CATS = {"irrelevance", "live_irrelevance"}

# Categories that are "simple" style (single function, AST match)
SIMPLE_CATS = {"simple", "java", "javascript", "live_simple"}

# Categories that are "multiple" style (single function from many, AST match)
MULTIPLE_CATS = {"multiple", "live_multiple"}

# Categories that are "parallel" style (multiple calls to same function)
PARALLEL_CATS = {"parallel", "live_parallel"}


# ── Data download ──

def download_data(categories: list[str] | None = None):
    """Check that BFCL V4 data exists in the bfcl_eval package."""
    cats = categories or list(BFCL_FILES.keys())
    for cat in cats:
        if cat in BFCL_FILES:
            filepath = DATA_DIR / BFCL_FILES[cat]
            if filepath.exists():
                print(f"  ✓ {BFCL_FILES[cat]}")
            else:
                print(f"  ✗ {BFCL_FILES[cat]} — MISSING from bfcl_eval package")
        if cat in BFCL_ANSWER_FILES:
            filepath = DATA_DIR / BFCL_ANSWER_FILES[cat]
            if filepath.exists():
                print(f"  ✓ {BFCL_ANSWER_FILES[cat]}")
            else:
                print(f"  ✗ {BFCL_ANSWER_FILES[cat]} — MISSING")




# ── Data loading ──

def load_bfcl_file(filepath: Path) -> list[dict]:
    """Load a BFCL JSONL file."""
    results = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def extract_func_defs(entry: dict) -> list[dict]:
    """Extract function definitions from a BFCL test entry."""
    funcs = entry.get("function", [])
    if isinstance(funcs, str):
        funcs = json.loads(funcs)
    if not isinstance(funcs, list):
        funcs = [funcs]

    extracted = []
    for f in funcs:
        if isinstance(f, dict):
            if "function" in f and "type" in f:
                f = f["function"]
            extracted.append(f)

    return extracted


def extract_query(entry: dict) -> str:
    """Extract the user query from a BFCL test entry."""
    messages = entry.get("question", entry.get("prompt", []))
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list):
        flat = []
        for item in messages:
            if isinstance(item, list):
                flat.extend(item)
            elif isinstance(item, dict):
                flat.append(item)
        for msg in flat:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
        if flat and isinstance(flat[-1], dict):
            return flat[-1].get("content", "")
    return ""


def extract_expected_tools(answer_entry: dict, category: str) -> set[str] | str | None:
    """Extract expected tool name(s) from an answer entry.

    Returns:
        - set of tool names for multi-route categories
        - single tool name string for single-route categories
        - None if no answer available
    """
    if not answer_entry:
        return None

    ground_truth = answer_entry.get("ground_truth", answer_entry.get("result", []))
    if isinstance(ground_truth, str):
        try:
            ground_truth = json.loads(ground_truth)
        except json.JSONDecodeError:
            return ground_truth.split("(")[0].strip()

    if not isinstance(ground_truth, list) or not ground_truth:
        return None

    if category in MULTI_ROUTE_CATS:
        tools = set()
        for call in ground_truth:
            if isinstance(call, dict):
                tools.update(call.keys())
            elif isinstance(call, str):
                tools.add(call.split("(")[0].strip())
        return tools

    # Single function: extract from first call
    first = ground_truth[0]
    if isinstance(first, str):
        return first.split("(")[0].strip()
    elif isinstance(first, dict):
        return list(first.keys())[0] if first else None
    return None


# ── Category evaluation ──

def run_category(
    handler: GlyphhBFCLHandler,
    category: str,
    max_entries: int | None = None,
) -> dict:
    """Run evaluation on a single BFCL category (Glyphh routing only)."""
    filename = BFCL_FILES.get(category)
    if not filename:
        print(f"  Unknown category: {category}")
        return {}

    filepath = DATA_DIR / filename
    if not filepath.exists():
        print(f"  Data not found: {filepath}. Run with --download-only first.")
        return {}

    entries = load_bfcl_file(filepath)
    if max_entries:
        entries = entries[:max_entries]

    # Load answers
    answer_file = BFCL_ANSWER_FILES.get(category)
    answers = {}
    if answer_file:
        answer_path = DATA_DIR / answer_file
        if answer_path.exists():
            for ae in load_bfcl_file(answer_path):
                answers[ae.get("id", "")] = ae

    results = []
    correct = 0
    total = 0
    total_latency = 0.0

    for i, entry in enumerate(entries):
        _progress(i + 1, len(entries), category[:25])

        query = extract_query(entry)
        func_defs = extract_func_defs(entry)
        entry_id = entry.get("id", f"{category}_{i}")

        if not query or not func_defs:
            continue

        total += 1

        if category in IRRELEVANCE_CATS:
            result = _eval_irrelevance(handler, query, func_defs, entry_id)
        elif category in MULTI_ROUTE_CATS:
            expected = extract_expected_tools(answers.get(entry_id), category)
            result = _eval_multi_route(handler, query, func_defs, entry_id, expected)
        else:
            expected = extract_expected_tools(answers.get(entry_id), category)
            result = _eval_single_route(handler, query, func_defs, entry_id, expected)

        total_latency += result["latency_ms"]
        results.append(result)
        if result["correct"]:
            correct += 1

    print()  # newline after progress bar

    accuracy = correct / total if total else 0
    avg_latency = total_latency / total if total else 0

    return {
        "category": category,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "avg_latency_ms": avg_latency,
        "results": results,
    }


def _eval_single_route(
    handler: GlyphhBFCLHandler,
    query: str,
    func_defs: list[dict],
    entry_id: str,
    expected: str | None,
) -> dict:
    """Evaluate single-function routing."""
    route_result = handler.route(query, func_defs)
    is_correct = route_result["tool"] == expected if expected else False

    return {
        "id": entry_id,
        "query": query[:100],
        "expected": expected,
        "predicted": route_result["tool"],
        "confidence": route_result["confidence"],
        "correct": is_correct,
        "latency_ms": route_result["latency_ms"],
        "top_k": route_result["top_k"],
    }


def _eval_multi_route(
    handler: GlyphhBFCLHandler,
    query: str,
    func_defs: list[dict],
    entry_id: str,
    expected: set[str] | None,
) -> dict:
    """Evaluate multi-function routing."""
    route_result = handler.route_multi(query, func_defs)
    predicted_tools = set(route_result["tools"])
    is_correct = predicted_tools == expected if expected else False

    return {
        "id": entry_id,
        "query": query[:100],
        "expected": sorted(expected) if expected else [],
        "predicted": sorted(predicted_tools),
        "confidence": route_result["confidence"],
        "correct": is_correct,
        "latency_ms": route_result["latency_ms"],
        "top_k": route_result["top_k"],
    }


def _eval_irrelevance(
    handler: GlyphhBFCLHandler,
    query: str,
    func_defs: list[dict],
    entry_id: str,
) -> dict:
    """Evaluate irrelevance detection. Correct = query rejected as irrelevant."""
    route_result = handler.route(query, func_defs)
    q_glyph = route_result.get("_query_glyph")
    f_glyph = route_result.get("_best_func_glyph")

    is_irrelevant = (
        q_glyph is not None
        and f_glyph is not None
        and handler.is_irrelevant(q_glyph, f_glyph, route_result["confidence"])
    )

    return {
        "id": entry_id,
        "query": query[:100],
        "expected": None,
        "predicted": None if is_irrelevant else route_result["tool"],
        "confidence": route_result["confidence"],
        "correct": is_irrelevant,
        "latency_ms": route_result["latency_ms"],
        "top_k": route_result["top_k"],
    }


# ── Multi-turn evaluation ──

def run_multi_turn_category(
    handler: GlyphhBFCLHandler,
    category: str,
    llm=None,
    max_entries: int | None = None,
) -> dict:
    """Run evaluation on a multi-turn BFCL category.

    Glyphh routes each turn. LLM extracts args (if provided).
    Scoring is per-entry: all turns must have correct routing.
    """
    from multi_turn_handler import (
        get_available_functions,
        extract_turn_query,
        eval_multi_turn_entry,
    )

    filename = BFCL_FILES.get(category)
    if not filename:
        print(f"  Unknown category: {category}")
        return {}

    filepath = DATA_DIR / filename
    if not filepath.exists():
        print(f"  Data not found: {filepath}. Run with --download-only first.")
        return {}

    entries = load_bfcl_file(filepath)
    if max_entries:
        entries = entries[:max_entries]

    # Load answers
    answer_file = BFCL_ANSWER_FILES.get(category)
    answers = {}
    if answer_file:
        answer_path = DATA_DIR / answer_file
        if answer_path.exists():
            for ae in load_bfcl_file(answer_path):
                answers[ae.get("id", "")] = ae

    results = []
    correct = 0
    total = 0
    total_latency = 0.0

    for i, entry in enumerate(entries):
        _progress(i + 1, len(entries), category[:25])

        entry_id = entry.get("id", f"{category}_{i}")
        answer = answers.get(entry_id, {})
        ground_truth = answer.get("ground_truth", [])

        if not ground_truth:
            continue

        total += 1
        start = time.perf_counter()

        result = eval_multi_turn_entry(handler, llm, entry, ground_truth)
        elapsed_ms = (time.perf_counter() - start) * 1000
        total_latency += elapsed_ms

        result["latency_ms"] = elapsed_ms
        results.append(result)

        if result["correct"]:
            correct += 1

    print()

    accuracy = correct / total if total else 0
    avg_latency = total_latency / total if total else 0

    # Also compute per-turn routing accuracy
    total_turns = sum(r["total_turns"] for r in results)
    correct_turns = sum(r["correct_turns"] for r in results)
    turn_accuracy = correct_turns / total_turns if total_turns else 0

    return {
        "category": category,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "avg_latency_ms": avg_latency,
        "turn_accuracy": turn_accuracy,
        "total_turns": total_turns,
        "correct_turns": correct_turns,
        "results": results,
    }

def run_multi_turn_with_observer(
    observer,
    category: str,
    max_entries: int | None = None,
    top_k: int = 3,
) -> dict:
    """Run multi-turn evaluation using the Observer (Model C).

    The Observer runs Model A + Model B internally and uses deductive
    reasoning to pick the best function sequence per turn.
    """
    from multi_turn_handler import (
        get_available_functions,
        extract_turn_query,
        parse_ground_truth_step,
    )
    from observer import _domain_from_classes

    filename = BFCL_FILES.get(category)
    if not filename:
        print(f"  Unknown category: {category}")
        return {}

    filepath = DATA_DIR / filename
    if not filepath.exists():
        print(f"  Data not found: {filepath}. Run with --download-only first.")
        return {}

    entries = load_bfcl_file(filepath)
    if max_entries:
        entries = entries[:max_entries]

    # Load answers
    answer_file = BFCL_ANSWER_FILES.get(category)
    answers = {}
    if answer_file:
        answer_path = DATA_DIR / answer_file
        if answer_path.exists():
            for ae in load_bfcl_file(answer_path):
                answers[ae.get("id", "")] = ae

    results = []
    correct = 0
    total = 0
    total_latency = 0.0

    for i, entry in enumerate(entries):
        _progress(i + 1, len(entries), category[:25])

        entry_id = entry.get("id", f"{category}_{i}")
        answer = answers.get(entry_id, {})
        ground_truth = answer.get("ground_truth", [])

        if not ground_truth:
            continue

        total += 1
        start = time.perf_counter()

        turns = entry.get("question", [])
        available_funcs = get_available_functions(entry)
        involved_classes = entry.get("involved_classes", [])

        if not available_funcs:
            results.append({
                "id": entry_id, "turns": len(turns),
                "correct_turns": 0, "total_turns": len(turns),
                "correct": False, "details": [],
            })
            continue

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

            # Observer decides using Model A + Model B + deductive reasoning
            predicted_funcs = observer.decide(
                query, available_funcs, available_names,
                domain_hint=domain_hint, top_k=top_k,
            )

            func_correct = list(predicted_funcs) == list(expected_funcs)

            turn_results.append({
                "turn": turn_idx,
                "correct": func_correct,
                "expected_funcs": expected_funcs,
                "predicted_funcs": predicted_funcs,
                "mode": "observer",
            })

            if func_correct:
                correct_turns += 1

        elapsed_ms = (time.perf_counter() - start) * 1000
        total_latency += elapsed_ms

        entry_result = {
            "id": entry_id,
            "turns": len(turns),
            "correct_turns": correct_turns,
            "total_turns": len(turn_results),
            "correct": correct_turns == len(turn_results),
            "details": turn_results,
            "latency_ms": elapsed_ms,
        }
        results.append(entry_result)

        if entry_result["correct"]:
            correct += 1

    print()

    accuracy = correct / total if total else 0
    avg_latency = total_latency / total if total else 0

    total_turns = sum(r["total_turns"] for r in results)
    correct_turns_total = sum(r["correct_turns"] for r in results)
    turn_accuracy = correct_turns_total / total_turns if total_turns else 0

    return {
        "category": category,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "avg_latency_ms": avg_latency,
        "turn_accuracy": turn_accuracy,
        "total_turns": total_turns,
        "correct_turns": correct_turns_total,
        "results": results,
    }



# ── V4 score computation ──

def compute_v4_score(category_results: dict[str, dict]) -> dict:
    """Compute the full BFCL V4 overall score.

    V4 Overall = (Agentic × 40%) + (Multi-Turn × 30%) + (Live × 10%) + (Non-Live × 10%) + (Hallucination × 10%)

    Non-Live: unweighted average of subcategories
    Hallucination: unweighted average of non-live irrelevance + live irrelevance
    Live: weighted average by test case count
    Multi-Turn: unweighted average of base + augmented cases
    Agentic: unweighted average of web search + memory
    """
    scores = {}

    # Non-Live (10%): unweighted average
    nonlive_accs = []
    for cat in V4_NONLIVE_CATS:
        if cat in category_results:
            nonlive_accs.append(category_results[cat]["accuracy"])
    nonlive_score = sum(nonlive_accs) / len(nonlive_accs) if nonlive_accs else None
    scores["non_live"] = nonlive_score

    # Hallucination (10%): unweighted average
    hall_accs = []
    for cat in V4_HALLUCINATION_CATS:
        if cat in category_results:
            hall_accs.append(category_results[cat]["accuracy"])
    hall_score = sum(hall_accs) / len(hall_accs) if hall_accs else None
    scores["hallucination"] = hall_score

    # Live (10%): weighted average by test case count
    live_weighted_sum = 0.0
    live_total_entries = 0
    for cat in V4_LIVE_CATS:
        if cat in category_results:
            cr = category_results[cat]
            live_weighted_sum += cr["accuracy"] * cr["total"]
            live_total_entries += cr["total"]
    live_score = live_weighted_sum / live_total_entries if live_total_entries else None
    scores["live"] = live_score

    # Multi-Turn (30%): unweighted average
    mt_accs = []
    for cat in V4_MULTITURN_CATS:
        if cat in category_results:
            mt_accs.append(category_results[cat]["accuracy"])
    mt_score = sum(mt_accs) / len(mt_accs) if mt_accs else None
    scores["multi_turn"] = mt_score

    # Agentic (40%): not yet implemented
    scores["agentic"] = None

    # Overall
    weights = {
        "agentic": 0.40,
        "multi_turn": 0.30,
        "live": 0.10,
        "non_live": 0.10,
        "hallucination": 0.10,
    }

    overall_num = 0.0
    overall_denom = 0.0
    for key, weight in weights.items():
        if scores[key] is not None:
            overall_num += scores[key] * weight
            overall_denom += weight

    scores["overall"] = overall_num / overall_denom if overall_denom > 0 else None
    scores["overall_weight_covered"] = overall_denom

    return scores


# ── Reporting ──

def _progress(current: int, total: int, label: str = ""):
    pct = current / total if total else 0
    bar_len = 30
    filled = int(bar_len * pct)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  {label:<25} [{bar}] {current}/{total}", end="", flush=True)


def print_report(category_results: list[dict]):
    """Print a summary report with V4 scoring."""
    print("\n" + "=" * 80)
    print("  GLYPHH HDC — BFCL V4 BENCHMARK RESULTS")
    print("=" * 80)

    # Group by V4 section
    sections = [
        ("NON-LIVE (10%)", V4_NONLIVE_CATS),
        ("HALLUCINATION (10%)", V4_HALLUCINATION_CATS),
        ("LIVE (10%)", V4_LIVE_CATS),
        ("MULTI-TURN (30%)", V4_MULTITURN_CATS),
    ]

    cr_map = {cr["category"]: cr for cr in category_results}

    header = f"  {'Category':<30} {'Accuracy':>10} {'Correct':>10} {'Total':>8} {'Lat(ms)':>10}"
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))

    total_correct = 0
    total_entries = 0

    for section_name, section_cats in sections:
        section_results = [cr_map[c] for c in section_cats if c in cr_map]
        if not section_results:
            continue

        print(f"\n  ┌─ {section_name}")
        for cr in section_results:
            extra = ""
            if "turn_accuracy" in cr:
                extra = f"  (turns: {cr['turn_accuracy']:.1%})"
            print(
                f"  │ {cr['category']:<28} "
                f"{cr['accuracy']:>9.1%} "
                f"{cr['correct']:>10} "
                f"{cr['total']:>8} "
                f"{cr['avg_latency_ms']:>9.1f}"
                f"{extra}"
            )
            total_correct += cr["correct"]
            total_entries += cr["total"]

    # Any uncategorized results
    categorized = set()
    for _, cats in sections:
        categorized.update(cats)
    uncategorized = [cr for cr in category_results if cr["category"] not in categorized]
    if uncategorized:
        print(f"\n  ┌─ OTHER")
        for cr in uncategorized:
            print(
                f"  │ {cr['category']:<28} "
                f"{cr['accuracy']:>9.1%} "
                f"{cr['correct']:>10} "
                f"{cr['total']:>8} "
                f"{cr['avg_latency_ms']:>9.1f}"
            )
            total_correct += cr["correct"]
            total_entries += cr["total"]

    # V4 composite scores
    v4 = compute_v4_score(cr_map)

    print("\n  " + "=" * (len(header) - 2))
    print("  V4 COMPOSITE SCORES:")
    print(f"    Non-Live (10%):      {v4['non_live']:>7.1%}" if v4["non_live"] is not None else "    Non-Live (10%):      N/A")
    print(f"    Hallucination (10%): {v4['hallucination']:>7.1%}" if v4["hallucination"] is not None else "    Hallucination (10%): N/A")
    print(f"    Live (10%):          {v4['live']:>7.1%}" if v4["live"] is not None else "    Live (10%):          N/A")
    print(f"    Multi-Turn (30%):    {v4['multi_turn']:>7.1%}" if v4["multi_turn"] is not None else "    Multi-Turn (30%):    N/A")
    print(f"    Agentic (40%):       N/A (not yet implemented)")

    weight_pct = v4["overall_weight_covered"] * 100
    if v4["overall"] is not None:
        print(f"\n    OVERALL ({weight_pct:.0f}% of V4): {v4['overall']:>7.1%}")
    print("  " + "=" * (len(header) - 2))

    # Errors
    for cr in category_results:
        errors = [r for r in cr["results"] if not r["correct"]]
        if errors and len(errors) <= 30:
            print(f"\n  ERRORS in {cr['category']} ({len(errors)}):")
            for e in errors[:10]:
                # Multi-turn entries have different structure
                if "details" in e:
                    print(f"    [{e['id']}] {e['correct_turns']}/{e['total_turns']} turns correct")
                    for d in e.get("details", []):
                        if not d.get("correct"):
                            ef = d.get("expected_func", d.get("expected_funcs", "?"))
                            pf = d.get("predicted_func", d.get("predicted_funcs", "?"))
                            print(f"      turn {d['turn']}: expected={ef} got={pf} "
                                  f"(conf={d.get('confidence', 0):.3f})")
                else:
                    print(f"    [{e['id']}] expected={e.get('expected')} got={e.get('predicted')} "
                          f"(conf={e.get('confidence', 0):.3f})")
                    print(f"      query: {e.get('query', '')}")
                    if e.get("top_k"):
                        top = ", ".join(f"{t['function']}={t['score']:.3f}" for t in e["top_k"][:3])
                        print(f"      top_k: {top}")


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Run Glyphh HDC against BFCL V4 benchmark")
    parser.add_argument(
        "--categories", nargs="+",
        default=None,
        choices=list(BFCL_FILES.keys()),
        help="Specific BFCL categories to evaluate",
    )
    parser.add_argument("--routing-only", action="store_true", default=False,
                        help="Only evaluate Glyphh routing categories (Non-Live + Hallucination + Live)")
    parser.add_argument("--full-v4", action="store_true", default=False,
                        help="Run full V4 evaluation (needs LLM for multi-turn + agentic)")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download BFCL data, don't run eval")
    parser.add_argument("--max-entries", type=int, default=None,
                        help="Max entries per category (for quick testing)")
    parser.add_argument("--threshold", type=float, default=0.15,
                        help="Similarity threshold for routing")
    parser.add_argument("--output", type=str, default=None,
                        help="Directory to save results JSON")
    parser.add_argument("--observer", action="store_true", default=False,
                        help="Use Observer (Model C) for multi-turn — 3-model cognitive architecture")
    # LLM options (for multi-turn + arg extraction)
    parser.add_argument("--llm-provider", type=str, default="openai",
                        choices=["openai", "gemini"],
                        help="LLM provider for argument extraction")
    parser.add_argument("--llm-model", type=str, default="gpt-4.1-nano",
                        help="LLM model name")
    args = parser.parse_args()

    # Determine categories to run
    if args.categories:
        categories = args.categories
    elif args.routing_only:
        categories = GLYPHH_ONLY_CATS
    elif args.full_v4:
        categories = GLYPHH_ONLY_CATS + V4_MULTITURN_CATS
    else:
        # Default: original 5 categories
        categories = ["simple", "multiple", "parallel", "parallel_multiple", "irrelevance"]

    # Download multi-turn func docs if needed
    mt_cats = [c for c in categories if c in set(V4_MULTITURN_CATS)]
    if mt_cats:
        _download_multi_turn_func_docs()

    print("Downloading BFCL data...")
    download_data(categories)

    if args.download_only:
        print("Done.")
        return

    handler = GlyphhBFCLHandler(threshold=args.threshold)

    # Initialize LLM client if multi-turn categories are requested
    llm = None
    if mt_cats and not args.observer:
        try:
            from llm_client import get_client
            llm = get_client(args.llm_provider, args.llm_model)
            print(f"LLM: {args.llm_provider}/{args.llm_model}")
        except Exception as e:
            print(f"LLM init failed ({e}), multi-turn will use routing-only")

    # Initialize Observer (Model C) if requested
    observer = None
    if args.observer and mt_cats:
        from observer import Observer
        from pattern_encoder import PatternRouter
        print("Building Observer (Model C)...")
        pattern_router = PatternRouter()
        pattern_router.build(min_count=2)
        observer = Observer(handler, pattern_router)
        print("  Building reference observations from training data...")
        observer.build()
        print(f"  Observer ready: {len(observer.ref_glyphs)} reference observations")

    print(f"\nCategories: {categories}")
    print(f"Threshold: {args.threshold}")
    if observer:
        print(f"Multi-turn handler: Observer (Model C) — 3-model cognitive architecture")
    print()

    category_results = []
    for cat in categories:
        print(f"── {cat} ──")
        if cat in set(V4_MULTITURN_CATS):
            if observer:
                result = run_multi_turn_with_observer(
                    observer, cat, max_entries=args.max_entries,
                )
            else:
                result = run_multi_turn_category(
                    handler, cat, llm=llm, max_entries=args.max_entries,
                )
        else:
            result = run_category(handler, cat, max_entries=args.max_entries)
        if result:
            category_results.append(result)

    print_report(category_results)

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        for cr in category_results:
            with open(out_dir / f"bfcl_{cr['category']}.json", "w") as f:
                json.dump(cr, f, indent=2, default=str)
        cr_map = {cr["category"]: cr for cr in category_results}
        v4 = compute_v4_score(cr_map)
        with open(out_dir / "bfcl_v4_score.json", "w") as f:
            json.dump(v4, f, indent=2)
        print(f"\nResults saved to {out_dir}/")


def _download_multi_turn_func_docs():
    """Download multi-turn function documentation files."""
    import urllib.request

    doc_dir = DATA_DIR / "multi_turn_func_doc"
    doc_dir.mkdir(parents=True, exist_ok=True)

    files = [
        "gorilla_file_system.json", "math_api.json", "message_api.json",
        "posting_api.json", "ticket_api.json", "trading_bot.json",
        "travel_booking.json", "vehicle_control.json",
    ]

    for fname in files:
        fpath = doc_dir / fname
        if fpath.exists():
            continue
        url = f"{HF_BASE}/multi_turn_func_doc/{fname}"
        print(f"  ↓ multi_turn_func_doc/{fname}...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, fpath)
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")


if __name__ == "__main__":
    main()
