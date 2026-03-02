#!/usr/bin/env python3
"""
Run Glyphh HDC against the full BFCL V4 benchmark.

BFCL V4 Overall = (Agentic × 40%) + (Multi-Turn × 30%) + (Live × 10%) + (Non-Live × 10%) + (Hallucination × 10%)

Glyphh's CognitiveLoop handles all routing via SchemaIntentClassifier
backed by a local LLM (Qwen3-4B via llama-cpp-python).

Usage:
    # Default eval (all routing categories)
    python run_bfcl.py

    # Full V4 eval (routing + multi-turn)
    python run_bfcl.py --full-v4

    # Specific categories
    python run_bfcl.py --categories simple java javascript live_simple

    # Quick test
    python run_bfcl.py --max-entries 5

    # Custom model path
    python run_bfcl.py --model-path /path/to/model.gguf

    # HDC-only mode (no LLM, sub-ms per query)
    python run_bfcl.py --hdc-only --routing-only

    # LLM-only mode (no HDC scorer)
    python run_bfcl.py --no-scorer

    # Legacy keyword fallback (no scorer, no LLM)
    python run_bfcl.py --no-llm --routing-only
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from cognitive_handler import CognitiveBFCLHandler

# Local data cache — downloaded from HuggingFace on first run
DATA_DIR = Path(__file__).parent / "data" / "bfcl"
DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = Path(__file__).parent / "results"

GH_BASE = "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/berkeley-function-call-leaderboard/bfcl_eval/data"

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

# Glyphh-only categories (no multi-turn)
GLYPHH_ONLY_CATS = V4_NONLIVE_CATS + V4_HALLUCINATION_CATS + V4_LIVE_CATS

# Categories that need multi-function routing
MULTI_ROUTE_CATS = {"parallel_multiple", "live_parallel_multiple"}

# Categories that are irrelevance detection
IRRELEVANCE_CATS = {"irrelevance", "live_irrelevance"}


# ── Data download ──

def download_data(categories: list[str] | None = None):
    """Download BFCL V4 data files from GitHub if not already present."""
    import urllib.request

    cats = categories or list(BFCL_FILES.keys())
    for cat in cats:
        if cat in BFCL_FILES:
            filepath = DATA_DIR / BFCL_FILES[cat]
            if filepath.exists():
                print(f"  ✓ {BFCL_FILES[cat]}")
            else:
                url = f"{GH_BASE}/{BFCL_FILES[cat]}"
                print(f"  ↓ {BFCL_FILES[cat]}...", end=" ", flush=True)
                try:
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    urllib.request.urlretrieve(url, filepath)
                    print("done")
                except Exception as e:
                    print(f"FAILED: {e}")
        if cat in BFCL_ANSWER_FILES:
            filepath = DATA_DIR / BFCL_ANSWER_FILES[cat]
            if filepath.exists():
                print(f"  ✓ {BFCL_ANSWER_FILES[cat]}")
            else:
                url = f"{GH_BASE}/{BFCL_ANSWER_FILES[cat]}"
                print(f"  ↓ {BFCL_ANSWER_FILES[cat]}...", end=" ", flush=True)
                try:
                    filepath.parent.mkdir(parents=True, exist_ok=True)
                    urllib.request.urlretrieve(url, filepath)
                    print("done")
                except Exception as e:
                    print(f"FAILED: {e}")


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
    """Extract expected tool name(s) from an answer entry."""
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
    handler: CognitiveBFCLHandler,
    category: str,
    max_entries: int | None = None,
) -> dict:
    """Run evaluation on a single BFCL category."""
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
    handler: CognitiveBFCLHandler,
    query: str,
    func_defs: list[dict],
    entry_id: str,
    expected: str | None,
) -> dict:
    """Evaluate single-function routing via CognitiveLoop."""
    route_result = handler.route(query, func_defs)
    predicted = route_result["tool"]
    conf = route_result["confidence"]

    is_correct = predicted == expected if expected else False

    return {
        "id": entry_id,
        "query": query[:100],
        "expected": expected,
        "predicted": predicted,
        "confidence": conf,
        "correct": is_correct,
        "latency_ms": route_result["latency_ms"],
        "top_k": route_result["top_k"],
    }


def _eval_multi_route(
    handler: CognitiveBFCLHandler,
    query: str,
    func_defs: list[dict],
    entry_id: str,
    expected: set[str] | None,
) -> dict:
    """Evaluate multi-function routing via CognitiveLoop."""
    route_result = handler.route_multi(query, func_defs)
    predicted_tools = set(route_result["tools"])
    conf = route_result["confidence"]

    is_correct = predicted_tools == expected if expected else False

    return {
        "id": entry_id,
        "query": query[:100],
        "expected": sorted(expected) if expected else [],
        "predicted": sorted(predicted_tools),
        "confidence": conf,
        "correct": is_correct,
        "latency_ms": route_result["latency_ms"],
        "top_k": route_result["top_k"],
    }


def _eval_irrelevance(
    handler: CognitiveBFCLHandler,
    query: str,
    func_defs: list[dict],
    entry_id: str,
) -> dict:
    """Evaluate irrelevance detection. Correct = query rejected as irrelevant."""
    is_irrelevant, route_result = handler.is_irrelevant_query(query, func_defs)

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
    category: str,
    engine=None,
    max_entries: int | None = None,
) -> dict:
    """Run evaluation on a multi-turn BFCL category via CognitiveLoop."""
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

        result = eval_multi_turn_entry(entry, ground_truth, engine=engine)
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

# ── V4 score computation ──

def compute_v4_score(category_results: dict[str, dict]) -> dict:
    """Compute the full BFCL V4 overall score.

    V4 Overall = (Agentic × 40%) + (Multi-Turn × 30%) + (Live × 10%) + (Non-Live × 10%) + (Hallucination × 10%)
    """
    scores = {}

    # Non-Live (10%): unweighted average
    nonlive_accs = []
    for cat in V4_NONLIVE_CATS:
        if cat in category_results:
            nonlive_accs.append(category_results[cat]["accuracy"])
    scores["non_live"] = sum(nonlive_accs) / len(nonlive_accs) if nonlive_accs else None

    # Hallucination (10%): unweighted average
    hall_accs = []
    for cat in V4_HALLUCINATION_CATS:
        if cat in category_results:
            hall_accs.append(category_results[cat]["accuracy"])
    scores["hallucination"] = sum(hall_accs) / len(hall_accs) if hall_accs else None

    # Live (10%): weighted average by test case count
    live_weighted_sum = 0.0
    live_total_entries = 0
    for cat in V4_LIVE_CATS:
        if cat in category_results:
            cr = category_results[cat]
            live_weighted_sum += cr["accuracy"] * cr["total"]
            live_total_entries += cr["total"]
    scores["live"] = live_weighted_sum / live_total_entries if live_total_entries else None

    # Multi-Turn (30%): unweighted average
    mt_accs = []
    for cat in V4_MULTITURN_CATS:
        if cat in category_results:
            mt_accs.append(category_results[cat]["accuracy"])
    scores["multi_turn"] = sum(mt_accs) / len(mt_accs) if mt_accs else None

    # Agentic (40%): not yet implemented here (see eval_full_v4.py)
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
    print(f"    Agentic (40%):       N/A (see eval_full_v4.py)")

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
                        help="Only evaluate routing categories (Non-Live + Hallucination + Live)")
    parser.add_argument("--full-v4", action="store_true", default=False,
                        help="Run full V4 evaluation (routing + multi-turn)")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download BFCL data, don't run eval")
    parser.add_argument("--max-entries", type=int, default=None,
                        help="Max entries per category (for quick testing)")
    parser.add_argument("--threshold", type=float, default=0.25,
                        help="Confidence threshold for routing")
    parser.add_argument("--output", type=str, default=None,
                        help="Directory to save results JSON")
    # LLM options
    parser.add_argument("--model-path", type=str, default=None,
                        help="Path to GGUF model file (default: auto-discover)")
    parser.add_argument("--n-ctx", type=int, default=4096,
                        help="LLM context window size (default: 4096)")
    parser.add_argument("--n-gpu-layers", type=int, default=-1,
                        help="Number of GPU layers (-1=all, 0=CPU-only)")
    parser.add_argument("--no-llm", action="store_true",
                        help="Run without LLM (HDC keyword fallback only, no scorer)")
    parser.add_argument("--hdc-only", action="store_true",
                        help="Run HDC scorer only (no LLM, sub-ms per query)")
    parser.add_argument("--no-scorer", action="store_true",
                        help="Disable HDC model scorer (LLM-only mode)")
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

    # Initialize LLMEngine
    engine = None
    skip_llm = args.no_llm or args.hdc_only
    if not skip_llm:
        try:
            from glyphh.llm import LLMEngine
            engine = LLMEngine(
                model_path=args.model_path,
                n_ctx=args.n_ctx,
                n_gpu_layers=args.n_gpu_layers,
            )
            gpu_label = "CPU-only" if args.n_gpu_layers == 0 else f"GPU({args.n_gpu_layers})"
            print(f"LLM: local LLMEngine (n_ctx={args.n_ctx}, {gpu_label}, lazy-load)")
        except Exception as e:
            print(f"LLMEngine init failed ({e}), running without LLM")

    # Determine scorer mode:
    # --hdc-only:   use_scorer=True,  engine=None  (pure HDC, sub-ms)
    # --no-scorer:  use_scorer=False, engine=...   (LLM-only, backward compat)
    # --no-llm:     use_scorer=False, engine=None  (keyword fallback, legacy)
    # default:      use_scorer=True,  engine=...   (full hybrid)
    use_scorer = not args.no_scorer and not args.no_llm
    if args.hdc_only:
        use_scorer = True

    handler = CognitiveBFCLHandler(
        engine=engine,
        confidence_threshold=args.threshold,
        use_scorer=use_scorer,
    )

    mode = "hybrid (HDC + LLM)" if use_scorer and engine else \
           "HDC-only" if use_scorer else \
           "LLM-only" if engine else "keyword fallback"

    print(f"\nCategories: {categories}")
    print(f"Threshold: {args.threshold}")
    print(f"Mode: {mode}")
    print(f"LLM: {'local' if engine else 'disabled'}")
    print(f"HDC Scorer: {'enabled' if use_scorer else 'disabled'}")
    print()

    category_results = []
    for cat in categories:
        print(f"── {cat} ──")
        if cat in set(V4_MULTITURN_CATS):
            result = run_multi_turn_category(
                cat, engine=engine, max_entries=args.max_entries,
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
        url = f"{GH_BASE}/multi_turn_func_doc/{fname}"
        print(f"  ↓ multi_turn_func_doc/{fname}...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, fpath)
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")


if __name__ == "__main__":
    main()
