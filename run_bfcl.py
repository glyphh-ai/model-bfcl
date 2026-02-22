#!/usr/bin/env python3
"""
Run Glyphh HDC against BFCL test data locally.

Downloads BFCL test data from HuggingFace and evaluates Glyphh's
routing accuracy across all single-turn categories.

Usage:
    # Routing only (no API key needed)
    python run_bfcl.py --routing-only

    # Full eval with LLM args (needs OPENAI_API_KEY)
    python run_bfcl.py --categories simple_python multiple parallel

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

DATA_DIR = Path(__file__).parent / "data" / "bfcl"
RESULTS_DIR = Path(__file__).parent / "results"

# BFCL data files on HuggingFace (raw URLs)
HF_BASE = "https://huggingface.co/datasets/gorilla-llm/Berkeley-Function-Calling-Leaderboard/resolve/main"

BFCL_FILES = {
    "simple_python": "BFCL_v3_simple.json",
    "multiple": "BFCL_v3_multiple.json",
    "parallel": "BFCL_v3_parallel.json",
    "parallel_multiple": "BFCL_v3_parallel_multiple.json",
    "irrelevance": "BFCL_v3_irrelevance.json",
}

# Answer files for evaluation
BFCL_ANSWER_FILES = {
    "simple_python": "BFCL_v3_simple_python_answer.json",
    "multiple": "BFCL_v3_multiple_answer.json",
    "parallel": "BFCL_v3_parallel_answer.json",
    "parallel_multiple": "BFCL_v3_parallel_multiple_answer.json",
    "irrelevance": "BFCL_v3_irrelevance_answer.json",
}


def download_data():
    """Download BFCL test data from HuggingFace."""
    import urllib.request

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for category, filename in {**BFCL_FILES, **BFCL_ANSWER_FILES}.items():
        filepath = DATA_DIR / filename
        if filepath.exists():
            print(f"  ✓ {filename} (cached)")
            continue

        url = f"{HF_BASE}/{filename}"
        print(f"  ↓ {filename}...", end=" ", flush=True)
        try:
            urllib.request.urlretrieve(url, filepath)
            print("done")
        except Exception as e:
            print(f"FAILED: {e}")


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
    """Extract function definitions from a BFCL test entry.

    BFCL entries have function defs in various formats:
    - "function" key with a list of function defs
    - Each func def may be wrapped in {"type": "function", "function": {...}}
    """
    funcs = entry.get("function", [])
    if isinstance(funcs, str):
        funcs = json.loads(funcs)
    if not isinstance(funcs, list):
        funcs = [funcs]

    extracted = []
    for f in funcs:
        if isinstance(f, dict):
            # Unwrap OpenAI-style {"type": "function", "function": {...}}
            if "function" in f and "type" in f:
                f = f["function"]
            extracted.append(f)

    return extracted


def extract_query(entry: dict) -> str:
    """Extract the user query from a BFCL test entry."""
    # BFCL entries have a list of message dicts
    messages = entry.get("question", entry.get("prompt", []))
    if isinstance(messages, str):
        return messages
    if isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
        # Fallback: last message
        if messages and isinstance(messages[-1], dict):
            return messages[-1].get("content", "")
    return ""


def run_category(
    handler: GlyphhBFCLHandler,
    category: str,
    routing_only: bool = True,
    llm_model: str = "gpt-4o-mini",
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

    # Load answers if available
    answer_file = BFCL_ANSWER_FILES.get(category)
    answers = {}
    if answer_file:
        answer_path = DATA_DIR / answer_file
        if answer_path.exists():
            answer_entries = load_bfcl_file(answer_path)
            for ae in answer_entries:
                aid = ae.get("id", "")
                answers[aid] = ae

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

        route_result = handler.route(query, func_defs)
        total += 1
        total_latency += route_result["latency_ms"]

        # Check correctness against answer
        expected = answers.get(entry_id, {})
        expected_tool = None
        if expected:
            # BFCL answers are in various formats
            ground_truth = expected.get("ground_truth", expected.get("result", []))
            if isinstance(ground_truth, list) and ground_truth:
                first = ground_truth[0]
                if isinstance(first, str):
                    # "func_name(args)" format — extract func name
                    expected_tool = first.split("(")[0].strip()
                elif isinstance(first, dict):
                    expected_tool = list(first.keys())[0] if first else None

        is_correct = False
        if category == "irrelevance":
            # For irrelevance, correct = no function selected
            is_correct = route_result["tool"] is None
        elif expected_tool:
            is_correct = route_result["tool"] == expected_tool

        if is_correct:
            correct += 1

        results.append({
            "id": entry_id,
            "query": query[:100],
            "expected": expected_tool,
            "predicted": route_result["tool"],
            "confidence": route_result["confidence"],
            "correct": is_correct,
            "latency_ms": route_result["latency_ms"],
            "top_k": route_result["top_k"],
        })

    print()  # newline after progress bar

    accuracy = correct / total if total else 0
    avg_latency = total_latency / total if total else 0

    return {
        "category": category,
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "avg_latency_ms": avg_latency,
        "tokens": 0,
        "results": results,
    }


def _progress(current: int, total: int, label: str = ""):
    pct = current / total if total else 0
    bar_len = 30
    filled = int(bar_len * pct)
    bar = "█" * filled + "░" * (bar_len - filled)
    print(f"\r  {label:<25} [{bar}] {current}/{total}", end="", flush=True)


def print_report(category_results: list[dict]):
    """Print a summary report."""
    print("\n" + "=" * 80)
    print("  GLYPHH HDC — BFCL BENCHMARK RESULTS")
    print("=" * 80)

    header = f"  {'Category':<25} {'Accuracy':>10} {'Correct':>10} {'Total':>8} {'Lat(ms)':>10} {'Tokens':>8}"
    print(f"\n{header}")
    print("  " + "-" * (len(header) - 2))

    total_correct = 0
    total_entries = 0

    for cr in category_results:
        print(
            f"  {cr['category']:<25} "
            f"{cr['accuracy']:>9.1%} "
            f"{cr['correct']:>10} "
            f"{cr['total']:>8} "
            f"{cr['avg_latency_ms']:>9.1f} "
            f"{cr['tokens']:>8}"
        )
        total_correct += cr["correct"]
        total_entries += cr["total"]

    if total_entries:
        overall = total_correct / total_entries
        print("  " + "-" * (len(header) - 2))
        print(f"  {'OVERALL':<25} {overall:>9.1%} {total_correct:>10} {total_entries:>8}")

    print("\n" + "=" * 80)

    # Print errors for debugging
    for cr in category_results:
        errors = [r for r in cr["results"] if not r["correct"]]
        if errors:
            print(f"\n  ERRORS in {cr['category']} ({len(errors)}):")
            for e in errors[:10]:
                print(f"    [{e['id']}] expected={e['expected']} got={e['predicted']} "
                      f"(conf={e['confidence']:.3f})")
                print(f"      query: {e['query']}")
                if e["top_k"]:
                    top = ", ".join(f"{t['function']}={t['score']:.3f}" for t in e["top_k"])
                    print(f"      top_k: {top}")


def main():
    parser = argparse.ArgumentParser(description="Run Glyphh HDC against BFCL benchmark data")
    parser.add_argument(
        "--categories", nargs="+",
        default=["simple_python", "multiple", "irrelevance"],
        choices=list(BFCL_FILES.keys()),
        help="BFCL categories to evaluate",
    )
    parser.add_argument("--routing-only", action="store_true", default=True,
                        help="Only evaluate routing accuracy (no LLM args)")
    parser.add_argument("--with-args", action="store_true",
                        help="Include LLM argument extraction (needs OPENAI_API_KEY)")
    parser.add_argument("--download-only", action="store_true",
                        help="Only download BFCL data, don't run eval")
    parser.add_argument("--max-entries", type=int, default=None,
                        help="Max entries per category (for quick testing)")
    parser.add_argument("--threshold", type=float, default=0.45,
                        help="Similarity threshold for routing")
    parser.add_argument("--output", type=str, default=None,
                        help="Directory to save results JSON")
    args = parser.parse_args()

    print("Downloading BFCL data...")
    download_data()

    if args.download_only:
        print("Done.")
        return

    handler = GlyphhBFCLHandler(
        threshold=args.threshold,
        use_llm_for_args=args.with_args,
    )

    print(f"\nCategories: {args.categories}")
    print(f"Mode: {'routing + args' if args.with_args else 'routing only'}")
    print(f"Threshold: {args.threshold}\n")

    category_results = []
    for cat in args.categories:
        print(f"── {cat} ──")
        result = run_category(
            handler, cat,
            routing_only=not args.with_args,
            max_entries=args.max_entries,
        )
        if result:
            category_results.append(result)

    print_report(category_results)

    if args.output:
        out_dir = Path(args.output)
        out_dir.mkdir(parents=True, exist_ok=True)
        for cr in category_results:
            with open(out_dir / f"bfcl_{cr['category']}.json", "w") as f:
                json.dump(cr, f, indent=2, default=str)
        print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
