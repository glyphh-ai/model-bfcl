#!/usr/bin/env python3
"""
Run Glyphh pure-HDC model against the BFCL V4 benchmark.

BFCL V4 Overall = (Agentic × 40%) + (Multi-Turn × 30%) + (Live × 10%) + (Non-Live × 10%) + (Hallucination × 10%)

Pure HDC. No LLM. No external services.

Usage:
    # Default (simple, multiple, parallel, parallel_multiple, irrelevance)
    python run_bfcl.py

    # Full routing suite (non-live + live + irrelevance)
    python run_bfcl.py --routing-only

    # Quick test — 5 entries per category
    python run_bfcl.py --max-entries 5

    # Specific categories
    python run_bfcl.py --categories simple multiple irrelevance

    # Show all mismatches
    python run_bfcl.py --verbose

    # Download data only
    python run_bfcl.py --download-only
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from handler import BFCLHandler

DATA_DIR    = Path(__file__).parent / "data" / "bfcl"
RESULTS_DIR = Path(__file__).parent / "results"

GH_BASE = (
    "https://raw.githubusercontent.com/ShishirPatil/gorilla/main/"
    "berkeley-function-call-leaderboard/bfcl_eval/data"
)

# ── Dataset catalogue ──────────────────────────────────────────────────────

BFCL_FILES = {
    "simple":             "BFCL_v4_simple_python.json",
    "multiple":           "BFCL_v4_multiple.json",
    "parallel":           "BFCL_v4_parallel.json",
    "parallel_multiple":  "BFCL_v4_parallel_multiple.json",
    "java":               "BFCL_v4_simple_java.json",
    "javascript":         "BFCL_v4_simple_javascript.json",
    "irrelevance":        "BFCL_v4_irrelevance.json",
    "live_irrelevance":   "BFCL_v4_live_irrelevance.json",
    "live_simple":        "BFCL_v4_live_simple.json",
    "live_multiple":      "BFCL_v4_live_multiple.json",
    "live_parallel":      "BFCL_v4_live_parallel.json",
    "live_parallel_multiple": "BFCL_v4_live_parallel_multiple.json",
}

BFCL_ANSWER_FILES = {
    "simple":             "possible_answer/BFCL_v4_simple_python.json",
    "multiple":           "possible_answer/BFCL_v4_multiple.json",
    "parallel":           "possible_answer/BFCL_v4_parallel.json",
    "parallel_multiple":  "possible_answer/BFCL_v4_parallel_multiple.json",
    "java":               "possible_answer/BFCL_v4_simple_java.json",
    "javascript":         "possible_answer/BFCL_v4_simple_javascript.json",
    "live_simple":        "possible_answer/BFCL_v4_live_simple.json",
    "live_multiple":      "possible_answer/BFCL_v4_live_multiple.json",
    "live_parallel":      "possible_answer/BFCL_v4_live_parallel.json",
    "live_parallel_multiple": "possible_answer/BFCL_v4_live_parallel_multiple.json",
}

V4_NONLIVE_CATS       = ["simple", "java", "javascript", "multiple", "parallel", "parallel_multiple"]
V4_HALLUCINATION_CATS = ["irrelevance", "live_irrelevance"]
V4_LIVE_CATS          = ["live_simple", "live_multiple", "live_parallel", "live_parallel_multiple"]
ROUTING_CATS          = V4_NONLIVE_CATS + V4_HALLUCINATION_CATS + V4_LIVE_CATS

MULTI_ROUTE_CATS  = {"parallel_multiple", "live_parallel_multiple"}
IRRELEVANCE_CATS  = {"irrelevance", "live_irrelevance"}
PARALLEL_CATS     = {"parallel", "live_parallel"}
DEFAULT_CATS      = ["simple", "multiple", "parallel", "parallel_multiple", "irrelevance"]


# ── Data download ─────────────────────────────────────────────────────────

def download_data(categories: list[str]) -> None:
    """Download BFCL V4 data files from GitHub if not already present."""
    import urllib.request

    for cat in categories:
        for fmap in (BFCL_FILES, BFCL_ANSWER_FILES):
            if cat not in fmap:
                continue
            fpath = DATA_DIR / fmap[cat]
            if fpath.exists():
                print(f"  ✓ {fmap[cat]}")
                continue
            url = f"{GH_BASE}/{fmap[cat]}"
            print(f"  ↓ {fmap[cat]}...", end=" ", flush=True)
            try:
                fpath.parent.mkdir(parents=True, exist_ok=True)
                urllib.request.urlretrieve(url, fpath)
                print("done")
            except Exception as e:
                print(f"FAILED: {e}")


# ── Data loading ──────────────────────────────────────────────────────────

def load_jsonl(path: Path) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def extract_func_defs(entry: dict) -> list[dict]:
    funcs = entry.get("function", [])
    if isinstance(funcs, str):
        funcs = json.loads(funcs)
    if not isinstance(funcs, list):
        funcs = [funcs]
    out = []
    for f in funcs:
        if isinstance(f, dict):
            if "function" in f and "type" in f:
                f = f["function"]
            out.append(f)
    return out


def extract_query(entry: dict) -> str:
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


def extract_expected_tool(answer: dict, category: str) -> Any:
    """Return expected tool(s) from an answer entry."""
    if not answer:
        return None
    gt = answer.get("ground_truth", answer.get("result", []))
    if isinstance(gt, str):
        try:
            gt = json.loads(gt)
        except json.JSONDecodeError:
            return gt.split("(")[0].strip()

    if not isinstance(gt, list) or not gt:
        return None

    if category in MULTI_ROUTE_CATS | PARALLEL_CATS:
        tools: set[str] = set()
        for call in gt:
            if isinstance(call, dict):
                tools.update(call.keys())
            elif isinstance(call, str):
                tools.add(call.split("(")[0].strip())
        return tools

    first = gt[0]
    if isinstance(first, str):
        return first.split("(")[0].strip()
    if isinstance(first, dict):
        return list(first.keys())[0] if first else None
    return None


# ── Category evaluation ───────────────────────────────────────────────────

def run_category(
    handler: BFCLHandler,
    category: str,
    max_entries: int | None = None,
    verbose: bool = False,
) -> dict:
    filename = BFCL_FILES.get(category)
    if not filename:
        print(f"  Unknown category: {category}")
        return {}

    filepath = DATA_DIR / filename
    if not filepath.exists():
        print(f"  Data not found: {filepath}. Run with --download-only first.")
        return {}

    entries = load_jsonl(filepath)
    if max_entries:
        entries = entries[:max_entries]

    # Load ground-truth answers
    answers: dict[str, dict] = {}
    afile = BFCL_ANSWER_FILES.get(category)
    if afile:
        apath = DATA_DIR / afile
        if apath.exists():
            for ae in load_jsonl(apath):
                answers[ae.get("id", "")] = ae

    results   = []
    correct   = 0
    total     = 0
    total_lat = 0.0

    need_multi = category in MULTI_ROUTE_CATS | PARALLEL_CATS

    for i, entry in enumerate(entries):
        _progress(i + 1, len(entries), category[:25])

        query     = extract_query(entry)
        func_defs = extract_func_defs(entry)
        entry_id  = entry.get("id", f"{category}_{i}")

        if not query or not func_defs:
            continue

        total += 1

        if category in IRRELEVANCE_CATS:
            res = _eval_irrelevance(handler, query, func_defs, entry_id)
        elif need_multi:
            expected = extract_expected_tool(answers.get(entry_id), category)
            res = _eval_multi(handler, query, func_defs, entry_id, expected, force=True)
        else:
            expected = extract_expected_tool(answers.get(entry_id), category)
            res = _eval_single(handler, query, func_defs, entry_id, expected, force=True)

        total_lat += res["latency_ms"]
        results.append(res)
        if res["correct"]:
            correct += 1

    print()  # newline after progress bar

    accuracy    = correct / total if total else 0.0
    avg_latency = total_lat / total if total else 0.0

    if verbose:
        _print_errors(category, results)

    return {
        "category":       category,
        "total":          total,
        "correct":        correct,
        "accuracy":       accuracy,
        "avg_latency_ms": avg_latency,
        "results":        results,
    }


def _eval_single(
    handler: BFCLHandler,
    query: str,
    func_defs: list[dict],
    entry_id: str,
    expected: str | None,
    force: bool = False,
) -> dict:
    r         = handler.route(query, func_defs, force=force)
    predicted = r["tool"]
    correct   = predicted == expected if expected is not None else False
    return {
        "id":         entry_id,
        "query":      query[:100],
        "expected":   expected,
        "predicted":  predicted,
        "confidence": r["confidence"],
        "correct":    correct,
        "latency_ms": r["latency_ms"],
        "top_k":      r["top_k"],
    }


def _eval_multi(
    handler: BFCLHandler,
    query: str,
    func_defs: list[dict],
    entry_id: str,
    expected: set[str] | None,
    force: bool = False,
) -> dict:
    r              = handler.route_multi(query, func_defs, force=force)
    predicted_set  = set(r["tools"])
    correct        = predicted_set == expected if expected is not None else False
    return {
        "id":         entry_id,
        "query":      query[:100],
        "expected":   sorted(expected) if expected else [],
        "predicted":  sorted(predicted_set),
        "confidence": r["confidence"],
        "correct":    correct,
        "latency_ms": r["latency_ms"],
        "top_k":      r["top_k"],
    }


def _eval_irrelevance(
    handler: BFCLHandler,
    query: str,
    func_defs: list[dict],
    entry_id: str,
) -> dict:
    is_irr, r = handler.is_irrelevant(query, func_defs)
    return {
        "id":         entry_id,
        "query":      query[:100],
        "expected":   None,
        "predicted":  None if is_irr else r["tool"],
        "confidence": r["confidence"],
        "correct":    is_irr,
        "latency_ms": r["latency_ms"],
        "top_k":      r["top_k"],
    }


# ── V4 score computation ──────────────────────────────────────────────────

def compute_v4_score(cr_map: dict[str, dict]) -> dict:
    """Compute the BFCL V4 composite score from available category results."""
    def avg(cats):
        accs = [cr_map[c]["accuracy"] for c in cats if c in cr_map]
        return sum(accs) / len(accs) if accs else None

    def wavg(cats):
        num = den = 0.0
        for c in cats:
            if c in cr_map:
                num += cr_map[c]["accuracy"] * cr_map[c]["total"]
                den += cr_map[c]["total"]
        return num / den if den > 0 else None

    scores = {
        "non_live":     avg(V4_NONLIVE_CATS),
        "hallucination": avg(V4_HALLUCINATION_CATS),
        "live":         wavg(V4_LIVE_CATS),
        "agentic":      None,
    }

    weights = {"agentic": 0.40, "non_live": 0.10, "hallucination": 0.10, "live": 0.10}
    num = den = 0.0
    for k, w in weights.items():
        if scores[k] is not None:
            num += scores[k] * w
            den += w

    scores["overall"]               = num / den if den > 0 else None
    scores["overall_weight_covered"] = den
    return scores


# ── Reporting ─────────────────────────────────────────────────────────────

def _progress(current: int, total: int, label: str = "") -> None:
    pct    = current / total if total else 0
    bar_w  = 30
    filled = int(bar_w * pct)
    bar    = "█" * filled + "░" * (bar_w - filled)
    print(f"\r  {label:<25} [{bar}] {current}/{total}", end="", flush=True)


def _print_errors(category: str, results: list[dict]) -> None:
    errors = [r for r in results if not r["correct"]]
    if not errors:
        return
    print(f"\n  Errors in {category} ({len(errors)}):")
    for e in errors[:20]:
        top = ", ".join(f"{t['function']}={t['score']:.3f}" for t in e.get("top_k", [])[:3])
        print(f"    [{e['id']}]")
        print(f"      query:    {e['query']}")
        print(f"      expected: {e.get('expected')}")
        print(f"      got:      {e.get('predicted')} (conf={e.get('confidence', 0):.3f})")
        if top:
            print(f"      top_k:    {top}")


def print_report(category_results: list[dict]) -> None:
    print("\n" + "=" * 80)
    print("  GLYPHH — PURE HDC — BFCL V4 BENCHMARK RESULTS")
    print("=" * 80)

    sections = [
        ("NON-LIVE (10%)",     V4_NONLIVE_CATS),
        ("HALLUCINATION (10%)", V4_HALLUCINATION_CATS),
        ("LIVE (10%)",         V4_LIVE_CATS),
    ]

    cr_map = {cr["category"]: cr for cr in category_results}
    hdr = f"  {'Category':<30} {'Accuracy':>10} {'Correct':>10} {'Total':>8} {'Lat(ms)':>10}"
    print(f"\n{hdr}")
    print("  " + "-" * (len(hdr) - 2))

    for section_name, section_cats in sections:
        section_results = [cr_map[c] for c in section_cats if c in cr_map]
        if not section_results:
            continue
        print(f"\n  ┌─ {section_name}")
        for cr in section_results:
            print(
                f"  │ {cr['category']:<28} "
                f"{cr['accuracy']:>9.1%} "
                f"{cr['correct']:>10} "
                f"{cr['total']:>8} "
                f"{cr['avg_latency_ms']:>9.1f}"
            )

    # Uncategorised
    categorised = {c for _, cats in sections for c in cats}
    others = [cr for cr in category_results if cr["category"] not in categorised]
    if others:
        print(f"\n  ┌─ OTHER")
        for cr in others:
            print(
                f"  │ {cr['category']:<28} "
                f"{cr['accuracy']:>9.1%} "
                f"{cr['correct']:>10} "
                f"{cr['total']:>8} "
                f"{cr['avg_latency_ms']:>9.1f}"
            )

    v4 = compute_v4_score(cr_map)
    pct_w = v4["overall_weight_covered"] * 100

    print("\n  " + "=" * (len(hdr) - 2))
    print("  V4 COMPOSITE SCORES:")
    for k in ("non_live", "hallucination", "live"):
        label = {"non_live": "Non-Live (10%)", "hallucination": "Hallucination (10%)", "live": "Live (10%)"}[k]
        val = v4.get(k)
        s = f"{val:>7.1%}" if val is not None else "    N/A"
        print(f"    {label:<24} {s}")
    print(f"    {'Agentic (40%)':<24}     N/A")
    if v4["overall"] is not None:
        print(f"\n    OVERALL ({pct_w:.0f}% of V4):     {v4['overall']:>7.1%}")
    print("  " + "=" * (len(hdr) - 2))


# ── Main ──────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Glyphh pure-HDC model against BFCL V4"
    )
    parser.add_argument(
        "--categories", nargs="+", default=None,
        choices=list(BFCL_FILES.keys()),
        help="Specific categories to evaluate",
    )
    parser.add_argument(
        "--routing-only", action="store_true",
        help="Run all routing categories (non-live + live + irrelevance)",
    )
    parser.add_argument(
        "--max-entries", type=int, default=None,
        help="Max entries per category (for quick testing)",
    )
    parser.add_argument(
        "--threshold", type=float, default=0.22,
        help="Irrelevance confidence threshold (default: 0.22)",
    )
    parser.add_argument(
        "--download-only", action="store_true",
        help="Only download data, do not run evaluation",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print per-category error details",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Directory to save results JSON",
    )
    args = parser.parse_args()

    if args.categories:
        categories = args.categories
    elif args.routing_only:
        categories = ROUTING_CATS
    else:
        categories = DEFAULT_CATS

    print("Downloading BFCL data...")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    download_data(categories)

    if args.download_only:
        print("Done.")
        return

    handler = BFCLHandler(confidence_threshold=args.threshold)

    print(f"\nCategories : {categories}")
    print(f"Threshold  : {args.threshold}")
    print(f"Mode       : pure HDC (no LLM)")
    print()

    category_results: list[dict] = []
    for cat in categories:
        print(f"── {cat} ──")
        result = run_category(
            handler, cat,
            max_entries=args.max_entries,
            verbose=args.verbose,
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
        cr_map = {cr["category"]: cr for cr in category_results}
        with open(out_dir / "bfcl_v4_score.json", "w") as f:
            json.dump(compute_v4_score(cr_map), f, indent=2)
        print(f"\nResults saved to {out_dir}/")


if __name__ == "__main__":
    main()
