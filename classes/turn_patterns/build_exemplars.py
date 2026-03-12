"""Build exemplars.jsonl for the turn_patterns class.

Reads all 800 multi-turn entries (4 categories × 200), extracts per-turn
(query → function sequence pattern) pairs, and generates exemplars.jsonl
in the standard Glyphh class format.

Each pattern gets 3 variants:
  v1: canonical description (from intent extraction)
  v2: canonical + function names in description
  v3: actual NL query BoW (one per unique query, up to a limit)

Output: classes/turn_patterns/exemplars.jsonl
Also:   classes/turn_patterns/tests.jsonl — all (query, pattern) pairs for testing

Usage:
    cd glyphh-models/bfcl
    PYTHONPATH=../../glyphh-runtime python classes/turn_patterns/build_exemplars.py
"""

from __future__ import annotations

import json
import re
import sys
from collections import defaultdict
from pathlib import Path

# Add turn_patterns dir to path for intent import
_DIR = Path(__file__).parent
if str(_DIR) not in sys.path:
    sys.path.insert(0, str(_DIR))

from intent import extract_intent

DATA_DIR = Path(__file__).parents[1].parent / "data" / "bfcl"
GT_DIR = DATA_DIR / "possible_answer"
OUT_DIR = _DIR

CATEGORIES = ["base", "miss_func", "miss_param", "long_context"]

# Max NL query variants per pattern (v3+). High-count patterns get more.
MAX_NL_VARIANTS = 10


def _extract_func_name(call_str: str) -> str:
    m = re.match(r"(\w+)\(", call_str)
    return m.group(1) if m else call_str


def _split_camel_snake(name: str) -> str:
    name = name.replace("_", " ").replace("-", " ").replace(".", " ")
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return name.lower().strip()


_STOP_WORDS = frozenset({
    "the", "a", "an", "to", "for", "on", "in", "is", "it", "i",
    "do", "can", "please", "now", "up", "my", "our", "me", "we",
    "and", "or", "of", "with", "from", "this", "that", "about",
    "how", "what", "when", "where", "which", "who", "also",
    "then", "but", "just", "them", "their", "its", "be", "been",
    "have", "has", "had", "not", "dont", "want", "think",
    "given", "using", "by", "if", "so", "as", "at", "into",
    "are", "was", "were", "will", "would", "could", "should",
    "may", "might", "shall", "need", "does", "did", "am",
    "you", "your", "he", "she", "they", "us", "his", "her",
    "let", "like", "any", "all", "some", "each", "every",
    "there", "here", "via", "per", "after", "before", "over",
})


def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return [w for w in cleaned.split() if w not in _STOP_WORDS and len(w) > 1]


def _bow(words: list[str]) -> str:
    unique = list(dict.fromkeys(words))
    return " ".join(unique) if unique else "none"


def _func_name_bow(pattern: list[str]) -> str:
    """Convert pattern function names to BoW string.

    ['lockDoors', 'pressBrakePedal', 'startEngine']
    → 'lock doors press brake pedal start engine'
    """
    if not pattern:
        return "none"
    tokens = []
    for fn in pattern:
        tokens.extend(_split_camel_snake(fn).split())
    return _bow(tokens)


def _param_bow_from_query(query: str) -> str:
    """Extract parameter-relevant tokens from the query."""
    tokens = _tokenize(query)
    # Add quoted strings and numbers
    tokens.extend(re.findall(r'"([^"]*)"', query))
    tokens.extend(re.findall(r"'([^']*)'", query))
    tokens.extend(re.findall(r"\b\d+\.?\d*\b", query))
    return _bow(list(dict.fromkeys(tokens)))


def load_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build():
    """Extract patterns and generate exemplars.jsonl + tests.jsonl."""

    # Collect all (pattern_key, queries, classes, domains) tuples
    pattern_data: dict[str, dict] = defaultdict(lambda: {
        "queries": [],
        "classes": set(),
        "func_names": [],
    })

    test_pairs: list[dict] = []

    for cat in CATEGORIES:
        q_file = DATA_DIR / f"BFCL_v4_multi_turn_{cat}.json"
        gt_file = GT_DIR / f"BFCL_v4_multi_turn_{cat}.json"

        questions = load_jsonl(q_file)
        gts = load_jsonl(gt_file)

        for q_entry, gt_entry in zip(questions, gts):
            involved = q_entry.get("involved_classes", [])

            for turn_idx, (turn_q, turn_gt) in enumerate(
                zip(q_entry["question"], gt_entry["ground_truth"])
            ):
                query = turn_q[0]["content"] if turn_q else ""
                if not query.strip():
                    continue

                func_names = [_extract_func_name(c) for c in turn_gt]
                pattern_key = "|".join(func_names) if func_names else "[]"

                pat = pattern_data[pattern_key]
                pat["queries"].append(query)
                pat["classes"].update(involved)
                if not pat["func_names"]:
                    pat["func_names"] = func_names

                test_pairs.append({
                    "query": query,
                    "expected": pattern_key,
                    "category": cat,
                    "entry_id": q_entry["id"],
                    "turn_idx": turn_idx,
                })

    # Generate exemplars.jsonl
    exemplars = []

    for pattern_key, pat in sorted(pattern_data.items(), key=lambda x: -len(x[1]["queries"])):
        func_names = pat["func_names"]
        queries = pat["queries"]
        fn_bow = _func_name_bow(func_names)

        # Use first query's intent as canonical
        canonical_intent = extract_intent(queries[0])
        action = canonical_intent["action"]
        target = canonical_intent["target"]
        domain = canonical_intent["domain"]

        # Vote on domain across all queries for this pattern
        domain_votes: dict[str, int] = defaultdict(int)
        for q in queries:
            d = extract_intent(q)["domain"]
            if d != "none":
                domain_votes[d] += 1
        if domain_votes:
            domain = max(domain_votes, key=domain_votes.get)

        # v1: canonical description from intent keywords
        desc_tokens = _tokenize(queries[0])
        v1_desc = _bow(desc_tokens) if desc_tokens else fn_bow

        exemplars.append({
            "class_name": "TurnPattern",
            "function_name": pattern_key,
            "raw_name": pattern_key,
            "action": action,
            "target": target,
            "domain": domain,
            "function_name_bow": fn_bow,
            "parameters_bow": _param_bow_from_query(queries[0]),
            "description": f"{v1_desc} {fn_bow}" if fn_bow != "none" else v1_desc,
            "variant": 1,
        })

        # v2: function-name-heavy description
        v2_desc_parts = []
        if action != "none":
            v2_desc_parts.append(action.replace("_", " "))
        v2_desc_parts.append(fn_bow if fn_bow != "none" else "empty")
        if domain != "none":
            v2_desc_parts.append(domain)
        v2_desc = " ".join(v2_desc_parts)

        exemplars.append({
            "class_name": "TurnPattern",
            "function_name": pattern_key,
            "raw_name": pattern_key,
            "action": action,
            "target": target,
            "domain": domain,
            "function_name_bow": fn_bow,
            "parameters_bow": _param_bow_from_query(queries[0]),
            "description": v2_desc,
            "variant": 2,
        })

        # v3+: NL query variants (deduplicated, up to MAX_NL_VARIANTS)
        seen_queries = set()
        variant = 3
        for q in queries:
            q_norm = q.strip().lower()
            if q_norm in seen_queries:
                continue
            seen_queries.add(q_norm)

            q_intent = extract_intent(q)
            q_tokens = _tokenize(q)
            q_desc = _bow(q_tokens) if q_tokens else "none"

            exemplars.append({
                "class_name": "TurnPattern",
                "function_name": pattern_key,
                "raw_name": pattern_key,
                "action": q_intent["action"],
                "target": q_intent["target"],
                "domain": q_intent["domain"],
                "function_name_bow": fn_bow,
                "parameters_bow": _param_bow_from_query(q),
                "description": f"{q_desc} {fn_bow}" if fn_bow != "none" else q_desc,
                "variant": variant,
            })
            variant += 1
            if variant - 2 >= MAX_NL_VARIANTS:
                break

    # Write exemplars
    exemplar_path = OUT_DIR / "exemplars.jsonl"
    with open(exemplar_path, "w") as f:
        for ex in exemplars:
            f.write(json.dumps(ex) + "\n")

    # Write tests
    test_path = OUT_DIR / "tests.jsonl"
    with open(test_path, "w") as f:
        for tp in test_pairs:
            f.write(json.dumps(tp) + "\n")

    # Stats
    n_patterns = len(pattern_data)
    n_exemplars = len(exemplars)
    n_tests = len(test_pairs)

    print(f"Patterns: {n_patterns}")
    print(f"Exemplars: {n_exemplars} ({n_exemplars / n_patterns:.1f} avg per pattern)")
    print(f"Test pairs: {n_tests}")
    print(f"Output: {exemplar_path}")
    print(f"Tests: {test_path}")
    print()

    # Pattern distribution
    print("Top 20 patterns:")
    for pk, pd in sorted(pattern_data.items(), key=lambda x: -len(x[1]["queries"]))[:20]:
        print(f"  {len(pd['queries']):4d} queries  {pk}")

    # Domain distribution
    domain_counts = defaultdict(int)
    for pk, pd in pattern_data.items():
        intent = extract_intent(pd["queries"][0])
        domain_counts[intent.get("domain", "none")] += 1
    print(f"\nDomain distribution:")
    for d, c in sorted(domain_counts.items(), key=lambda x: -x[1]):
        print(f"  {c:4d}  {d}")


if __name__ == "__main__":
    build()
