#!/usr/bin/env python3
"""
Task-level exemplar generation + scoring for BFCL multi-turn.

Two-tier architecture:
  Tier 1: Task exemplars — NL query → function sequence (stored procedure)
  Tier 2: CognitiveLoop — state tracking, memory, deduction

Each ground truth turn becomes a task exemplar:
  query text → [func1, func2, ...] (the exact sequence)

At runtime, match the query against task exemplars to get the function
sequence, then hand off to LLM for arg extraction.

Usage:
    python task_exemplars.py --generate   # Mine from ground truth
    python task_exemplars.py --stats      # Show pattern stats
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data" / "bfcl"
CLASSES_DIR = ROOT / "classes"

MULTI_TURN_FILES = [
    "BFCL_v4_multi_turn_base.json",
    "BFCL_v4_multi_turn_miss_func.json",
    "BFCL_v4_multi_turn_miss_param.json",
    "BFCL_v4_multi_turn_long_context.json",
]

# Stop words for query cleaning
_STOP_WORDS = frozenset([
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "shall", "can", "need", "must",
    "i", "me", "my", "we", "our", "you", "your", "he", "she", "it",
    "they", "them", "their", "his", "her", "its",
    "this", "that", "these", "those", "there", "here",
    "and", "or", "but", "if", "so", "as", "at", "by", "for", "in",
    "of", "on", "to", "with", "from", "into", "up", "out", "about",
    "not", "no", "nor", "than", "too", "very", "just", "also",
    "then", "now", "when", "where", "how", "what", "which", "who",
    "all", "each", "every", "some", "any", "most", "other",
    "let", "like", "please", "could", "would",
])


def _clean_query(text: str) -> str:
    """Extract keywords from query, removing stop words."""
    cleaned = re.sub(r"[^\w\s]", " ", text.lower())
    words = [w for w in cleaned.split() if w not in _STOP_WORDS and len(w) > 1]
    return " ".join(words)


def _extract_func_names(gt_turn: list[str]) -> list[str]:
    """Extract bare function names from ground truth strings like 'cd(folder=...)'."""
    return [f.split("(")[0] for f in gt_turn]


def _build_global_func_to_class() -> dict[str, str]:
    """Build bare_func_name → class_name from all func_doc files."""
    func_doc_dir = DATA_DIR / "multi_turn_func_doc"
    func_to_class: dict[str, str] = {}

    # Class name → func_doc filename (same map as discover.py)
    class_to_file = {
        "GorillaFileSystem": "gorilla_file_system.json",
        "TwitterAPI":        "posting_api.json",
        "MessageAPI":        "message_api.json",
        "PostingAPI":        "posting_api.json",
        "TicketAPI":         "ticket_api.json",
        "MathAPI":           "math_api.json",
        "TradingBot":        "trading_bot.json",
        "TravelAPI":         "travel_booking.json",
        "VehicleControlAPI": "vehicle_control.json",
    }

    for cls, filename in class_to_file.items():
        path = func_doc_dir / filename
        if not path.exists():
            continue
        with open(path) as f:
            func_defs = [json.loads(line) for line in f if line.strip()]
        for fdef in func_defs:
            name = fdef.get("name", "")
            if "." in name:
                bare = name.split(".", 1)[1]
            else:
                bare = name
            func_to_class[bare] = cls

    return func_to_class


# Global lookup (built once at import time)
_FUNC_TO_CLASS: dict[str, str] = {}


def _detect_class(funcs: list[str], involved_classes: list[str], entry: dict) -> str:
    """Detect which class a turn's functions belong to."""
    global _FUNC_TO_CLASS
    if not _FUNC_TO_CLASS:
        _FUNC_TO_CLASS = _build_global_func_to_class()

    for f in funcs:
        if f in _FUNC_TO_CLASS:
            return _FUNC_TO_CLASS[f]

    # Fallback: first involved class
    return involved_classes[0] if involved_classes else "unknown"


def mine_task_exemplars() -> dict[str, list[dict]]:
    """Mine task-level exemplars from all multi-turn ground truth.

    Returns:
        {class_name: [{query, functions, pattern, entry_id, turn_idx}, ...]}
    """
    class_exemplars: dict[str, list[dict]] = {}

    for mt_file in MULTI_TURN_FILES:
        data_path = DATA_DIR / mt_file
        answer_path = DATA_DIR / "possible_answer" / mt_file

        if not data_path.exists() or not answer_path.exists():
            continue

        with open(data_path) as f:
            entries = [json.loads(line) for line in f]
        with open(answer_path) as f:
            answers = [json.loads(line) for line in f]

        ans_by_id = {a["id"]: a for a in answers}

        for entry in entries:
            eid = entry["id"]
            gt = ans_by_id.get(eid, {}).get("ground_truth", [])
            questions = entry.get("question", [])
            involved = entry.get("involved_classes", [])

            for t, (q, g) in enumerate(zip(questions, gt)):
                if not isinstance(g, list) or not g:
                    continue

                # Extract query text
                if isinstance(q, list) and q and isinstance(q[0], dict):
                    query = q[0].get("content", "")
                else:
                    continue

                funcs = _extract_func_names(g)
                if not funcs:
                    continue

                # Skip empty turns
                if funcs == [""]:
                    continue

                # Detect class
                cls = _detect_class(funcs, involved, entry)
                pattern = tuple(funcs)

                exemplar = {
                    "query": query,
                    "query_clean": _clean_query(query),
                    "functions": funcs,
                    "pattern": list(pattern),
                    "entry_id": eid,
                    "turn_idx": t,
                    # Store full ground truth for arg reference
                    "ground_truth": g,
                }

                class_exemplars.setdefault(cls, []).append(exemplar)

    return class_exemplars


def write_task_exemplars(class_exemplars: dict[str, list[dict]]) -> None:
    """Write task-level exemplars to classes/{class}/task_exemplars.jsonl."""
    for cls, exemplars in class_exemplars.items():
        # Map class name to folder
        folder_map = {
            "GorillaFileSystem": "gorilla_file_system",
            "TwitterAPI": "twitter_api",
            "MessageAPI": "message_api",
            "PostingAPI": "posting_api",
            "TicketAPI": "ticket_api",
            "MathAPI": "math_api",
            "TradingBot": "trading_bot",
            "TravelAPI": "travel_booking",
            "VehicleControlAPI": "vehicle_control",
        }
        folder = folder_map.get(cls, cls.lower())
        out_dir = CLASSES_DIR / folder
        out_dir.mkdir(parents=True, exist_ok=True)

        out_path = out_dir / "task_exemplars.jsonl"
        with open(out_path, "w") as f:
            for ex in exemplars:
                # Write exemplar in format suitable for encoding
                record = {
                    "class_name": cls,
                    "pattern": ex["pattern"],
                    "functions": [f"{cls}.{fn}" for fn in ex["functions"]],
                    "query": ex["query"],
                    "description": ex["query_clean"],
                    "entry_id": ex["entry_id"],
                    "turn_idx": ex["turn_idx"],
                }
                f.write(json.dumps(record) + "\n")

        print(f"  {cls:25s} → {len(exemplars):4d} task exemplars → {out_path}")


def print_stats(class_exemplars: dict[str, list[dict]]) -> None:
    """Print statistics about mined task exemplars."""
    total = sum(len(v) for v in class_exemplars.values())
    print(f"\nTotal task exemplars: {total}")
    print(f"Classes: {len(class_exemplars)}")

    for cls in sorted(class_exemplars.keys()):
        exemplars = class_exemplars[cls]
        patterns = Counter(tuple(e["pattern"]) for e in exemplars)
        print(f"\n  {cls} ({len(exemplars)} exemplars, {len(patterns)} unique patterns):")
        for pat, count in patterns.most_common(10):
            print(f"    {count:3d}x  {list(pat)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--generate", action="store_true", help="Generate task exemplars")
    parser.add_argument("--stats", action="store_true", help="Show stats only")
    args = parser.parse_args()

    exemplars = mine_task_exemplars()

    if args.stats or not args.generate:
        print_stats(exemplars)

    if args.generate:
        print("\nWriting task exemplars:")
        write_task_exemplars(exemplars)
        print("\nDone.")
