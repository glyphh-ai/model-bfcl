#!/usr/bin/env python3
"""
Full BFCL V4 evaluation — all categories, gpt-4.1, results saved to disk.

V4 Overall = (Agentic × 40%) + (Multi-Turn × 30%) + (Live × 10%) + (Non-Live × 10%) + (Hallucination × 10%)

Agentic = avg(web_search, memory)

Saves per-category JSON + final V4 score to results/full_v4/

Usage:
    .venv/bin/python3 glyphh-models/bfcl/eval_full_v4.py
    .venv/bin/python3 glyphh-models/bfcl/eval_full_v4.py --skip-routing   # skip already-done routing cats
    .venv/bin/python3 glyphh-models/bfcl/eval_full_v4.py --only multi_turn
    .venv/bin/python3 glyphh-models/bfcl/eval_full_v4.py --only agentic
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import bfcl_eval

DATA_DIR = Path(bfcl_eval.__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results" / "full_v4"

# ── Import handlers ──
from handler import GlyphhBFCLHandler
from run_bfcl import (
    BFCL_FILES, BFCL_ANSWER_FILES,
    V4_NONLIVE_CATS, V4_HALLUCINATION_CATS, V4_LIVE_CATS, V4_MULTITURN_CATS,
    run_category, run_multi_turn_category,
    load_bfcl_file, _download_multi_turn_func_docs,
)


LLM_PROVIDER = "openai"
LLM_MODEL = "gpt-4.1"


def save_result(name: str, data: dict):
    """Save a category result to JSON."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    with open(path, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"  → saved {path}")


def load_result(name: str) -> dict | None:
    """Load a previously saved result."""
    path = RESULTS_DIR / f"{name}.json"
    if path.exists():
        with open(path) as f:
            return json.load(f)
    return None


# ── Memory eval (Glyphh-only, no LLM needed for retrieval) ──

def run_memory_eval() -> dict:
    """Run BFCL V4 memory eval. Glyphh IS the memory."""
    from memory_handler import GlyphhMemory, extract_facts_chunked

    SCENARIOS = ["customer", "healthcare", "finance", "student", "notetaker"]
    PREREQ_FILES = {s: f"memory_prereq_conversation/memory_{s}.json" for s in SCENARIOS}

    questions = load_bfcl_file(DATA_DIR / "BFCL_v4_memory.json")
    answers = {a["id"]: a for a in load_bfcl_file(DATA_DIR / "possible_answer" / "BFCL_v4_memory.json")}

    # Build memory per scenario
    memories = {}
    for scenario in SCENARIOS:
        prereq_path = DATA_DIR / PREREQ_FILES[scenario]
        if not prereq_path.exists():
            print(f"    {scenario}: prereq not found")
            continue
        prereq_entries = load_bfcl_file(prereq_path)
        facts = extract_facts_chunked(prereq_entries, scenario, chunk_size=3, overlap=1)
        mem = GlyphhMemory()
        for fact in facts:
            mem.store(fact["text"], topic=fact["topic"], domain=scenario)
        memories[scenario] = mem
        print(f"    {scenario}: {mem.size} facts")

    correct = 0
    total = 0
    results = []

    for q in questions:
        qid = q["id"]
        scenario = q.get("scenario", "general")
        ans = answers.get(qid, {})
        gt = ans.get("ground_truth", [])
        if not gt:
            continue

        query = ""
        turns = q.get("question", [])
        if turns:
            turn = turns[0]
            if isinstance(turn, list):
                for msg in turn:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        query = msg.get("content", "")
            elif isinstance(turn, dict):
                query = turn.get("content", "")
        if not query:
            continue

        total += 1
        mem = memories.get(scenario)
        if not mem:
            results.append({"id": qid, "correct": False, "reason": "no_memory"})
            continue

        retrieved = mem.retrieve(query, top_k=10)
        hit = False
        for r in retrieved:
            std_resp = re.sub(r"[,./\-_*^()]", "", r["text"]).lower().replace("'", '"')
            for g in gt:
                std_gt = re.sub(r"[,./\-_*^()]", "", g).lower().replace("'", '"')
                if re.search(rf"\b{re.escape(std_gt)}\b", std_resp):
                    hit = True
                    break
            if hit:
                break

        if hit:
            correct += 1
        results.append({"id": qid, "correct": hit, "scenario": scenario})

    accuracy = correct / total if total else 0
    return {
        "category": "memory",
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "results": results,
    }


# ── Web search eval (needs LLM) ──

def run_web_search_eval(max_entries: int | None = None) -> dict:
    """Run BFCL V4 web search eval with Glyphh routing + structural verification."""
    from web_search_handler import WebSearchAgent

    questions = load_bfcl_file(DATA_DIR / "BFCL_v4_web_search.json")
    answers = {a["id"]: a for a in load_bfcl_file(
        DATA_DIR / "possible_answer" / "BFCL_v4_web_search.json"
    )}

    if max_entries:
        questions = questions[:max_entries]

    agent = WebSearchAgent(provider=LLM_PROVIDER, model=LLM_MODEL)
    print(f"    LLM: {LLM_PROVIDER}/{LLM_MODEL}")
    print(f"    Questions: {len(questions)}")

    correct = 0
    total = 0
    results = []

    for i, q in enumerate(questions):
        qid = q["id"]
        ans = answers.get(qid, {})
        gt = ans.get("ground_truth", [])

        query = ""
        turns = q.get("question", [])
        if turns:
            turn = turns[0]
            if isinstance(turn, list):
                for msg in turn:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        query = msg.get("content", "")
        if not query or not gt:
            continue

        total += 1
        print(f"    [{i+1}/{len(questions)}] {qid}...", end=" ", flush=True)

        t0 = time.perf_counter()
        try:
            response = agent.answer(query, max_hops=6)
        except Exception as e:
            response = f"error: {e}"
        elapsed = time.perf_counter() - t0

        std_resp = re.sub(r"[,./\-_*^()]", "", response).lower().replace("'", '"')
        hit = False
        for g in gt:
            std_gt = re.sub(r"[,./\-_*^()]", "", g).lower().replace("'", '"')
            if re.search(rf"\b{re.escape(std_gt)}\b", std_resp):
                hit = True
                break

        if hit:
            correct += 1
            print(f"✓ ({elapsed:.1f}s)")
        else:
            print(f"✗ ({elapsed:.1f}s)")

        results.append({
            "id": qid, "correct": hit,
            "response": response[:200], "gt": gt,
            "latency_s": round(elapsed, 1),
        })

        # Save incrementally every 5 questions
        if total % 5 == 0:
            partial = {
                "category": "web_search",
                "total": total, "correct": correct,
                "accuracy": correct / total,
                "results": results,
                "partial": True,
            }
            save_result("web_search_partial", partial)

    accuracy = correct / total if total else 0
    return {
        "category": "web_search",
        "total": total,
        "correct": correct,
        "accuracy": accuracy,
        "results": results,
    }


# ── V4 score computation (includes agentic) ──

def compute_full_v4_score(results: dict[str, dict]) -> dict:
    """Compute the complete BFCL V4 overall score including agentic."""

    def _avg(cats):
        accs = [results[c]["accuracy"] for c in cats if c in results]
        return sum(accs) / len(accs) if accs else None

    def _weighted_avg(cats):
        num, den = 0.0, 0
        for c in cats:
            if c in results:
                num += results[c]["accuracy"] * results[c]["total"]
                den += results[c]["total"]
        return num / den if den else None

    scores = {
        "non_live": _avg(V4_NONLIVE_CATS),
        "hallucination": _avg(V4_HALLUCINATION_CATS),
        "live": _weighted_avg(V4_LIVE_CATS),
        "multi_turn": _avg(V4_MULTITURN_CATS),
    }

    # Agentic = avg(web_search, memory)
    agentic_cats = ["web_search", "memory"]
    agentic_accs = [results[c]["accuracy"] for c in agentic_cats if c in results]
    scores["agentic"] = sum(agentic_accs) / len(agentic_accs) if agentic_accs else None
    scores["agentic_web_search"] = results.get("web_search", {}).get("accuracy")
    scores["agentic_memory"] = results.get("memory", {}).get("accuracy")

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
    scores["timestamp"] = datetime.now().isoformat()
    scores["llm_model"] = LLM_MODEL

    return scores


# ── Main ──

def main():
    parser = argparse.ArgumentParser(description="Full BFCL V4 evaluation")
    parser.add_argument("--skip-routing", action="store_true",
                        help="Skip non-live/live/hallucination (use saved results)")
    parser.add_argument("--only", type=str, default=None,
                        choices=["routing", "multi_turn", "agentic", "memory", "web_search"],
                        help="Run only a specific section")
    parser.add_argument("--max-web", type=int, default=None,
                        help="Max web search questions (for testing)")
    parser.add_argument("--max-entries", type=int, default=None,
                        help="Max entries per category")
    parser.add_argument("--threshold", type=float, default=0.15)
    args = parser.parse_args()

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  GLYPHH HDC — FULL BFCL V4 EVALUATION")
    print(f"  LLM: {LLM_PROVIDER}/{LLM_MODEL}")
    print(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    handler = GlyphhBFCLHandler(threshold=args.threshold)
    all_results = {}

    # ── 1. Routing categories (Non-Live + Hallucination + Live) ──
    routing_cats = V4_NONLIVE_CATS + V4_HALLUCINATION_CATS + V4_LIVE_CATS
    if args.only and args.only not in ("routing",):
        # Load from saved
        for cat in routing_cats:
            saved = load_result(cat)
            if not saved:
                # Try old results dir
                old_path = Path(__file__).parent / "results" / f"bfcl_{cat}.json"
                if old_path.exists():
                    with open(old_path) as f:
                        saved = json.load(f)
            if saved:
                all_results[cat] = saved
                print(f"  [loaded] {cat}: {saved['accuracy']:.1%}")
    elif args.skip_routing:
        for cat in routing_cats:
            saved = load_result(cat)
            if not saved:
                old_path = Path(__file__).parent / "results" / f"bfcl_{cat}.json"
                if old_path.exists():
                    with open(old_path) as f:
                        saved = json.load(f)
            if saved:
                all_results[cat] = saved
                print(f"  [loaded] {cat}: {saved['accuracy']:.1%}")
            else:
                print(f"  [running] {cat}")
                result = run_category(handler, cat, max_entries=args.max_entries)
                if result:
                    all_results[cat] = result
                    save_result(cat, result)
    else:
        if not args.only or args.only == "routing":
            print("\n── ROUTING (Non-Live + Hallucination + Live) ──")
            for cat in routing_cats:
                print(f"\n  {cat}:")
                result = run_category(handler, cat, max_entries=args.max_entries)
                if result:
                    all_results[cat] = result
                    save_result(cat, result)
                    print(f"  {cat}: {result['accuracy']:.1%} ({result['correct']}/{result['total']})")

    # ── 2. Multi-Turn ──
    if not args.only or args.only == "multi_turn":
        print("\n── MULTI-TURN (30% of V4) ──")
        _download_multi_turn_func_docs()

        from llm_client import get_client
        llm = get_client(LLM_PROVIDER, LLM_MODEL)
        print(f"  LLM: {LLM_PROVIDER}/{LLM_MODEL}")

        for cat in V4_MULTITURN_CATS:
            print(f"\n  {cat}:")
            result = run_multi_turn_category(
                handler, cat, llm=llm, max_entries=args.max_entries,
            )
            if result:
                all_results[cat] = result
                save_result(cat, result)
                extra = f" (turns: {result.get('turn_accuracy', 0):.1%})" if "turn_accuracy" in result else ""
                print(f"  {cat}: {result['accuracy']:.1%} ({result['correct']}/{result['total']}){extra}")
    else:
        # Load saved multi-turn
        for cat in V4_MULTITURN_CATS:
            saved = load_result(cat)
            if saved:
                all_results[cat] = saved
                print(f"  [loaded] {cat}: {saved['accuracy']:.1%}")

    # ── 3. Agentic — Memory ──
    if not args.only or args.only in ("agentic", "memory"):
        print("\n── AGENTIC: MEMORY ──")
        result = run_memory_eval()
        all_results["memory"] = result
        save_result("memory", result)
        print(f"  memory: {result['accuracy']:.1%} ({result['correct']}/{result['total']})")
    else:
        saved = load_result("memory")
        if saved:
            all_results["memory"] = saved
            print(f"  [loaded] memory: {saved['accuracy']:.1%}")

    # ── 4. Agentic — Web Search ──
    if not args.only or args.only in ("agentic", "web_search"):
        print("\n── AGENTIC: WEB SEARCH ──")
        result = run_web_search_eval(max_entries=args.max_web)
        all_results["web_search"] = result
        save_result("web_search", result)
        print(f"  web_search: {result['accuracy']:.1%} ({result['correct']}/{result['total']})")
    else:
        saved = load_result("web_search")
        if saved:
            all_results["web_search"] = saved
            print(f"  [loaded] web_search: {saved['accuracy']:.1%}")

    # ── Final V4 Score ──
    print("\n" + "=" * 70)
    print("  FINAL BFCL V4 SCORES")
    print("=" * 70)

    v4 = compute_full_v4_score(all_results)

    sections = [
        ("Non-Live (10%)", "non_live", V4_NONLIVE_CATS),
        ("Hallucination (10%)", "hallucination", V4_HALLUCINATION_CATS),
        ("Live (10%)", "live", V4_LIVE_CATS),
        ("Multi-Turn (30%)", "multi_turn", V4_MULTITURN_CATS),
    ]

    for section_name, score_key, cats in sections:
        score = v4[score_key]
        score_str = f"{score:.1%}" if score is not None else "N/A"
        print(f"\n  {section_name}: {score_str}")
        for cat in cats:
            if cat in all_results:
                r = all_results[cat]
                extra = ""
                if "turn_accuracy" in r:
                    extra = f"  (turns: {r['turn_accuracy']:.1%})"
                print(f"    {cat:<30} {r['accuracy']:.1%}  ({r['correct']}/{r['total']}){extra}")

    # Agentic
    agentic_str = f"{v4['agentic']:.1%}" if v4["agentic"] is not None else "N/A"
    print(f"\n  Agentic (40%): {agentic_str}")
    if v4["agentic_memory"] is not None:
        r = all_results["memory"]
        print(f"    memory                         {v4['agentic_memory']:.1%}  ({r['correct']}/{r['total']})")
    if v4["agentic_web_search"] is not None:
        r = all_results["web_search"]
        print(f"    web_search                     {v4['agentic_web_search']:.1%}  ({r['correct']}/{r['total']})")

    weight_pct = v4["overall_weight_covered"] * 100
    overall_str = f"{v4['overall']:.1%}" if v4["overall"] is not None else "N/A"
    print(f"\n  {'─' * 50}")
    print(f"  OVERALL ({weight_pct:.0f}% of V4): {overall_str}")
    print(f"  {'─' * 50}")

    # Save final score
    save_result("v4_final_score", v4)

    # Also save a summary with all category accuracies
    summary = {
        "v4_score": v4,
        "categories": {
            cat: {"accuracy": r["accuracy"], "correct": r["correct"], "total": r["total"]}
            for cat, r in all_results.items()
        },
        "timestamp": datetime.now().isoformat(),
        "llm_model": LLM_MODEL,
    }
    save_result("summary", summary)

    print(f"\n  All results saved to {RESULTS_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
