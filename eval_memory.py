#!/usr/bin/env python3
"""
Evaluate Glyphh memory on BFCL V4 Memory category.

Glyphh IS the memory backend:
  1. Load prereq conversations for each scenario
  2. Extract facts → encode as glyphs → store in GlyphhMemory
  3. For each eval question, retrieve top-k facts by cosine similarity
  4. Check if ground truth answer appears in retrieved facts

No LLM needed for this eval — if the right fact is retrieved,
the answer is in the text. We just check string containment.

Usage:
    .venv/bin/python3 glyphh-models/bfcl/eval_memory.py
"""

import json
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import bfcl_eval
from memory_handler import GlyphhMemory, extract_facts, extract_facts_chunked

DATA_DIR = Path(bfcl_eval.__file__).parent / "data"

SCENARIOS = ["customer", "healthcare", "finance", "student", "notetaker"]

PREREQ_FILES = {s: f"memory_prereq_conversation/memory_{s}.json" for s in SCENARIOS}


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(l.strip()) for l in f if l.strip()]


def standardize(s: str) -> str:
    """Match BFCL's standardization for answer checking."""
    return re.sub(r"[,./\-_*^()]", "", s).lower().replace("'", '"')


def check_answer(response_text: str, ground_truth: list[str]) -> bool:
    """Check if any ground truth answer appears in the response."""
    std_response = standardize(response_text)
    for gt in ground_truth:
        std_gt = standardize(gt)
        if re.search(rf"\b{re.escape(std_gt)}\b", std_response):
            return True
    return False


def main():
    print("=" * 60)
    print("  GLYPHH MEMORY — BFCL V4 AGENTIC (MEMORY)")
    print("=" * 60)

    # Load eval questions and answers
    questions = load_jsonl(DATA_DIR / "BFCL_v4_memory.json")
    answers = {a["id"]: a for a in load_jsonl(DATA_DIR / "possible_answer" / "BFCL_v4_memory.json")}
    print(f"Questions: {len(questions)}")

    # Build memory per scenario
    memories: dict[str, GlyphhMemory] = {}
    t0 = time.perf_counter()

    for scenario in SCENARIOS:
        prereq_path = DATA_DIR / PREREQ_FILES[scenario]
        if not prereq_path.exists():
            print(f"  {scenario}: prereq not found")
            continue

        prereq_entries = load_jsonl(prereq_path)
        facts = extract_facts_chunked(prereq_entries, scenario, chunk_size=3, overlap=1)

        mem = GlyphhMemory()
        for fact in facts:
            mem.store(fact["text"], topic=fact["topic"], domain=scenario)

        memories[scenario] = mem
        print(f"  {scenario}: {len(prereq_entries)} convos → {mem.size} facts")

    build_time = time.perf_counter() - t0
    total_facts = sum(m.size for m in memories.values())
    print(f"Total facts: {total_facts} in {build_time:.1f}s")

    # Evaluate
    print(f"\n{'─' * 60}")
    correct = 0
    total = 0
    by_scenario: dict[str, dict] = {s: {"correct": 0, "total": 0} for s in SCENARIOS}
    errors = []

    for q in questions:
        qid = q["id"]
        scenario = q.get("scenario", "general")
        ans = answers.get(qid, {})
        gt = ans.get("ground_truth", [])

        if not gt:
            continue

        # Extract query text
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
        by_scenario[scenario]["total"] += 1

        # Retrieve from Glyphh memory
        mem = memories.get(scenario)
        if not mem:
            errors.append({"id": qid, "query": query[:80], "gt": gt, "reason": "no_memory"})
            continue

        results = mem.retrieve(query, top_k=10)

        # Check if any retrieved fact contains the answer
        hit = False
        matched_text = ""
        for r in results:
            if check_answer(r["text"], gt):
                hit = True
                matched_text = r["text"]
                break

        if hit:
            correct += 1
            by_scenario[scenario]["correct"] += 1
        else:
            errors.append({
                "id": qid,
                "query": query[:80],
                "gt": gt,
                "top_retrieved": results[0]["text"][:100] if results else "none",
                "top_score": results[0]["score"] if results else 0,
            })

    # Results
    acc = correct / total if total else 0
    print(f"\n  Overall: {correct}/{total} = {acc:.1%}")
    print(f"\n  By scenario:")
    for s in SCENARIOS:
        sc = by_scenario[s]
        if sc["total"] > 0:
            sacc = sc["correct"] / sc["total"]
            print(f"    {s:<15} {sc['correct']}/{sc['total']} = {sacc:.1%}")

    print(f"\n  Errors ({len(errors)}):")
    for e in errors[:15]:
        print(f"    [{e['id']}] Q: {e['query']}")
        print(f"      Expected: {e['gt']}")
        if "top_retrieved" in e:
            print(f"      Got: {e['top_retrieved']}... (score={e.get('top_score',0):.3f})")
        print()

    print(f"{'=' * 60}")
    print(f"  Memory accuracy: {acc:.1%}")
    print(f"  This is part of Agentic (40% of V4)")
    print(f"{'=' * 60}")

    return acc


if __name__ == "__main__":
    main()
