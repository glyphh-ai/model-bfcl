#!/usr/bin/env python3
"""
Evaluate web search on BFCL V4 Agentic — Web Search Category.

LLM decomposes multihop questions, searches DuckDuckGo, extracts answers.

Usage:
    .venv/bin/python3 glyphh-models/bfcl/eval_web_search.py
    .venv/bin/python3 glyphh-models/bfcl/eval_web_search.py --max 10  # quick test
"""

import argparse
import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import bfcl_eval
from web_search_handler import WebSearchAgent

DATA_DIR = Path(bfcl_eval.__file__).parent / "data"


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l.strip()) for l in f if l.strip()]


def standardize(s: str) -> str:
    return re.sub(r"[,./\-_*^()]", "", s).lower().replace("'", '"')


def check_answer(response: str, ground_truth: list[str]) -> bool:
    std_resp = standardize(response)
    for gt in ground_truth:
        std_gt = standardize(gt)
        if re.search(rf"\b{re.escape(std_gt)}\b", std_resp):
            return True
    return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max", type=int, default=None)
    parser.add_argument("--provider", default="openai")
    parser.add_argument("--model", default="gpt-4.1")
    args = parser.parse_args()

    print("=" * 60)
    print("  BFCL V4 AGENTIC — WEB SEARCH")
    print("=" * 60)

    questions = load_jsonl(DATA_DIR / "BFCL_v4_web_search.json")
    answers = {a["id"]: a for a in load_jsonl(DATA_DIR / "possible_answer" / "BFCL_v4_web_search.json")}

    if args.max:
        questions = questions[:args.max]

    agent = WebSearchAgent(provider=args.provider, model=args.model)
    print(f"LLM: {args.provider}/{args.model}")
    print(f"Questions: {len(questions)}\n")

    correct = 0
    total = 0
    errors = []

    for i, q in enumerate(questions):
        qid = q["id"]
        ans = answers.get(qid, {})
        gt = ans.get("ground_truth", [])
        num_hops = ans.get("num_hops", "?")

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
        print(f"  [{i+1}/{len(questions)}] {qid} ({num_hops}h)...", end=" ", flush=True)

        t0 = time.perf_counter()
        try:
            response = agent.answer(query, max_hops=6)
        except Exception as e:
            response = f"error: {e}"
        elapsed = time.perf_counter() - t0

        hit = check_answer(response, gt)
        if hit:
            correct += 1
            print(f"✓ ({elapsed:.1f}s)")
        else:
            print(f"✗ ({elapsed:.1f}s)")
            errors.append({
                "id": qid, "hops": num_hops,
                "gt": gt, "got": response[:100],
                "query": query[:100],
            })

    acc = correct / total if total else 0
    print(f"\n{'=' * 60}")
    print(f"  Web Search: {correct}/{total} = {acc:.1%}")
    print(f"{'=' * 60}")

    if errors[:10]:
        print(f"\n  Errors ({len(errors)}):")
        for e in errors[:10]:
            print(f"    [{e['id']}] ({e['hops']}h) expected={e['gt']} got={e['got']}")

    return acc


if __name__ == "__main__":
    main()
