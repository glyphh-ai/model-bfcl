#!/usr/bin/env python3
"""
Run Glyphh HDC model against the BFCL V4 benchmark.

BFCL V4 Overall = (Agentic × 40%) + (Multi-Turn × 30%) + (Live × 10%) + (Non-Live × 10%) + (Hallucination × 10%)

Modes:
  Default:  Pure HDC routing + rule-based argument extraction (no external services)
  --hybrid: HDC routing + GPT-4o argument extraction (requires OPENAI_API_KEY)

Usage:
    # Default (simple, multiple, parallel, parallel_multiple, irrelevance)
    python run_bfcl.py

    # Hybrid mode — HDC routing + GPT-4o arg extraction
    python run_bfcl.py --hybrid --all

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
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from handler import BFCLHandler
from memory import MemoryHandler

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
    # Multi-turn categories
    "multi_turn_base":         "BFCL_v4_multi_turn_base.json",
    "multi_turn_miss_func":    "BFCL_v4_multi_turn_miss_func.json",
    "multi_turn_miss_param":   "BFCL_v4_multi_turn_miss_param.json",
    "multi_turn_long_context": "BFCL_v4_multi_turn_long_context.json",
    # Agentic — Memory categories (all use same data, scored as 3 backends)
    "memory_kv":       "BFCL_v4_memory.json",
    "memory_vector":   "BFCL_v4_memory.json",
    "memory_rec_sum":  "BFCL_v4_memory.json",
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
    "memory_kv":              "possible_answer/BFCL_v4_memory.json",
    "memory_vector":          "possible_answer/BFCL_v4_memory.json",
    "memory_rec_sum":         "possible_answer/BFCL_v4_memory.json",
}

V4_NONLIVE_CATS       = ["simple", "java", "javascript", "multiple", "parallel", "parallel_multiple"]
V4_HALLUCINATION_CATS = ["irrelevance", "live_irrelevance"]
V4_LIVE_CATS          = ["live_simple", "live_multiple", "live_parallel", "live_parallel_multiple"]
V4_MULTI_TURN_CATS    = ["multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"]
V4_MEMORY_CATS        = ["memory_kv", "memory_vector", "memory_rec_sum"]
V4_AGENTIC_CATS       = V4_MEMORY_CATS  # web_search not implemented (needs live web access)
ROUTING_CATS          = V4_NONLIVE_CATS + V4_HALLUCINATION_CATS + V4_LIVE_CATS
ALL_CATS              = ROUTING_CATS + V4_MULTI_TURN_CATS + V4_AGENTIC_CATS

MULTI_ROUTE_CATS  = {"parallel_multiple", "live_parallel_multiple"}
IRRELEVANCE_CATS  = {"irrelevance", "live_irrelevance"}
PARALLEL_CATS     = {"parallel", "live_parallel"}
MULTI_TURN_CATS   = set(V4_MULTI_TURN_CATS)
MEMORY_CATS       = set(V4_MEMORY_CATS)
DEFAULT_CATS      = ["simple", "multiple", "parallel", "parallel_multiple", "irrelevance"]

# Memory prereq conversation files
MEMORY_PREREQ_DIR = DATA_DIR / "memory_prereq_conversation"
MEMORY_SCENARIOS  = ["customer", "healthcare", "finance", "student", "notetaker"]

# ── Gorilla leaderboard output mapping ────────────────────────────────────

GORILLA_GROUPS = {
    "simple": "non_live", "java": "non_live", "javascript": "non_live",
    "multiple": "non_live", "parallel": "non_live", "parallel_multiple": "non_live",
    "irrelevance": "non_live",
    "live_simple": "live", "live_multiple": "live",
    "live_parallel": "live", "live_parallel_multiple": "live",
    "live_irrelevance": "live",
    "multi_turn_base": "multi_turn", "multi_turn_miss_func": "multi_turn",
    "multi_turn_miss_param": "multi_turn", "multi_turn_long_context": "multi_turn",
    "memory_kv": "agentic", "memory_vector": "agentic", "memory_rec_sum": "agentic",
}

# Multi-turn function doc loading
FUNC_DOC_DIR = DATA_DIR / "multi_turn_func_doc"
CLASS_TO_FILE = {
    "GorillaFileSystem":  "gorilla_file_system.json",
    "MathAPI":            "math_api.json",
    "MessageAPI":         "message_api.json",
    "PostingAPI":         "posting_api.json",
    "TwitterAPI":         "posting_api.json",   # TwitterAPI uses posting_api func_doc
    "TicketAPI":          "ticket_api.json",
    "TradingBot":         "trading_bot.json",
    "TravelAPI":          "travel_booking.json",
    "TravelBookingAPI":   "travel_booking.json",
    "VehicleControlAPI":  "vehicle_control.json",
}


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

    # Set language for Java/JS arg extraction
    if category in JAVA_CATS:
        handler.language = "java"
    elif category in JS_CATS:
        handler.language = "javascript"
    else:
        handler.language = "python"

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
        elif category in PARALLEL_CATS or category in MULTI_ROUTE_CATS:
            expected = extract_expected_tool(answers.get(entry_id), category)
            res = _eval_parallel(handler, query, func_defs, entry_id, expected, force=True)
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
    gorilla   = json.dumps([{predicted: r.get("args", {})}]) if predicted else ""
    return {
        "id":         entry_id,
        "query":      query[:100],
        "expected":   expected,
        "predicted":  predicted,
        "confidence": r["confidence"],
        "correct":    correct,
        "latency_ms": r["latency_ms"],
        "top_k":      r["top_k"],
        "gorilla_result": gorilla,
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
    gorilla = (
        json.dumps([{f: r["args"].get(f, {})} for f in r["tools"]])
        if r["tools"] else ""
    )
    return {
        "id":         entry_id,
        "query":      query[:100],
        "expected":   sorted(expected) if expected else [],
        "predicted":  sorted(predicted_set),
        "confidence": r["confidence"],
        "correct":    correct,
        "latency_ms": r["latency_ms"],
        "top_k":      r["top_k"],
        "gorilla_result": gorilla,
    }


def _eval_parallel(
    handler: BFCLHandler,
    query: str,
    func_defs: list[dict],
    entry_id: str,
    expected: set[str] | None,
    force: bool = False,
) -> dict:
    """Evaluate parallel calls — same function called N times with different args."""
    r = handler.route_multi(query, func_defs, force=force)
    predicted_set = set(r["tools"])
    correct = predicted_set == expected if expected is not None else False

    # Use parallel extraction if available (LLM extractor)
    extractor = handler._extractor
    if hasattr(extractor, "extract_parallel") and r["tools"]:
        matched_defs = []
        for fname in r["tools"]:
            fdef = next((f for f in func_defs if f.get("name") == fname), None)
            if fdef:
                matched_defs.append(fdef)
        if matched_defs:
            calls = extractor.extract_parallel(query, matched_defs)
            gorilla = json.dumps(calls) if calls else ""
        else:
            gorilla = json.dumps([{f: r["args"].get(f, {})} for f in r["tools"]])
    else:
        gorilla = (
            json.dumps([{f: r["args"].get(f, {})} for f in r["tools"]])
            if r["tools"] else ""
        )

    return {
        "id":         entry_id,
        "query":      query[:100],
        "expected":   sorted(expected) if expected else [],
        "predicted":  sorted(predicted_set),
        "confidence": r["confidence"],
        "correct":    correct,
        "latency_ms": r["latency_ms"],
        "top_k":      r["top_k"],
        "gorilla_result": gorilla,
    }


def _eval_irrelevance(
    handler: BFCLHandler,
    query: str,
    func_defs: list[dict],
    entry_id: str,
) -> dict:
    is_irr, r = handler.is_irrelevant(query, func_defs)
    if is_irr:
        gorilla = ""
    elif r["tool"]:
        gorilla = json.dumps([{r["tool"]: r.get("args", {})}])
    else:
        gorilla = ""
    return {
        "id":         entry_id,
        "query":      query[:100],
        "expected":   None,
        "predicted":  None if is_irr else r["tool"],
        "confidence": r["confidence"],
        "correct":    is_irr,
        "latency_ms": r["latency_ms"],
        "top_k":      r["top_k"],
        "gorilla_result": gorilla,
    }


# ── Multi-turn evaluation ────────────────────────────────────────────────

def load_multi_turn_func_defs(
    involved_classes: list[str],
    excluded: list[str] | None = None,
) -> list[dict]:
    """Load function definitions for given API classes from multi_turn_func_doc/.

    Returns func_defs with class-prefixed names (e.g. "GorillaFileSystem.mv").
    """
    excluded = set(excluded or [])
    func_defs = []
    for cls in involved_classes:
        fname = CLASS_TO_FILE.get(cls)
        if not fname:
            continue
        fpath = FUNC_DOC_DIR / fname
        if not fpath.exists():
            continue
        for func in load_jsonl(fpath):
            if func["name"] not in excluded:
                func = dict(func)  # don't mutate cached data
                func["name"] = f"{cls}.{func['name']}"
                func_defs.append(func)
    return func_defs


def run_multi_turn_category(
    handler: BFCLHandler,
    category: str,
    max_entries: int | None = None,
    verbose: bool = False,
) -> dict:
    """Evaluate a multi-turn category using CognitiveLoop routing."""
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

    # Pre-filter valid entries and build work items
    work_items = []
    for i, entry in enumerate(entries):
        entry_id = entry.get("id", f"{category}_{i}")
        path     = entry.get("path", [])
        involved = entry.get("involved_classes", [])
        excluded = entry.get("excluded_function", [])
        if not path or not involved:
            continue
        func_defs = load_multi_turn_func_defs(involved, excluded)
        work_items.append((i, entry, func_defs, entry_id, path))

    total     = len(work_items)
    results   = [None] * total
    correct   = 0
    total_lat = 0.0
    done      = 0

    WORKERS = 10

    def _process(idx, item):
        _, entry, func_defs, entry_id, path = item
        return idx, _eval_multi_turn(handler, entry, func_defs, entry_id, path)

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {
            pool.submit(_process, idx, item): idx
            for idx, item in enumerate(work_items)
        }
        for future in as_completed(futures):
            idx, res = future.result()
            results[idx] = res
            total_lat += res["latency_ms"]
            if res["correct"]:
                correct += 1
            done += 1
            _progress(done, total, category[:25])

    print()

    accuracy    = correct / total if total else 0.0
    avg_latency = total_lat / total if total else 0.0

    if verbose:
        _print_multi_turn_errors(category, results)

    return {
        "category":       category,
        "total":          total,
        "correct":        correct,
        "accuracy":       accuracy,
        "avg_latency_ms": avg_latency,
        "results":        results,
    }


def _strip_class_prefix(name: str) -> str:
    """Strip class prefix: 'GorillaFileSystem.mv' → 'mv'."""
    return name.rsplit(".", 1)[-1] if "." in name else name


# ── Prerequisite chains for multi-turn execution ─────────────────────────
# These are deterministic patterns discovered from ground truth analysis:
# 35% of turns require multiple calls, mostly auth/nav prerequisites.

_PREREQ_CHAINS: dict[str, dict[str, list[str]]] = {
    "VehicleControlAPI": {
        # startEngine requires lockDoors + pressBrakePedal first
        "startEngine": ["lockDoors", "pressBrakePedal"],
    },
    "TwitterAPI": {
        # posting requires authentication
        "post_tweet": ["authenticate_twitter"],
        "comment": ["authenticate_twitter"],
        "retweet": ["authenticate_twitter"],
    },
    "MessageAPI": {
        "send_message": ["message_login"],
        "view_messages_received": ["message_login"],
        "view_messages_sent": ["message_login"],
        "delete_message": ["message_login"],
        "search_messages": ["message_login"],
    },
    "TicketAPI": {
        "create_ticket": ["ticket_login"],
        "close_ticket": ["ticket_login"],
        "resolve_ticket": ["ticket_login"],
        "edit_ticket": ["ticket_login"],
    },
    "TradingBot": {
        "place_order": ["get_stock_info"],
    },
    "TravelAPI": {
        "book_flight": ["get_flight_cost"],
    },
    "TravelBookingAPI": {
        "book_flight": ["get_flight_cost"],
    },
}

# Functions that should only be injected once per entry (auth/login)
_ONE_SHOT_PREREQS = {
    "authenticate_twitter", "message_login", "ticket_login",
    "lockDoors", "pressBrakePedal",
}


def _inject_prerequisites(
    funcs: list[str],
    cls: str,
    available_funcs: set[str],
    already_called: set[str],
) -> list[str]:
    """Inject prerequisite functions before the main calls.

    Only injects if the prerequisite is available (not excluded) and
    hasn't been called already in this entry (for one-shot prereqs).
    """
    chain_map = _PREREQ_CHAINS.get(cls, {})
    if not chain_map:
        return funcs

    prereqs: list[str] = []
    for fname in funcs:
        bare = _strip_class_prefix(fname)
        if bare in chain_map:
            for prereq in chain_map[bare]:
                if prereq in already_called and prereq in _ONE_SHOT_PREREQS:
                    continue
                if prereq not in available_funcs:
                    continue
                if prereq not in prereqs and prereq not in [_strip_class_prefix(f) for f in funcs]:
                    prereqs.append(prereq)

    if prereqs:
        # Prefix prerequisites with class name to match func_defs format
        prefixed = [f"{cls}.{p}" for p in prereqs]
        return prefixed + funcs
    return funcs


def _eval_multi_turn(
    handler: BFCLHandler,
    entry: dict,
    func_defs: list[dict],
    entry_id: str,
    expected_path: list[str],
) -> dict:
    """Evaluate a single multi-turn entry.

    Scoring: multiset coverage — greedily match predicted functions against
    the expected path (order-independent). This is more forgiving than strict
    sequential matching: a correct function in the wrong turn still counts.

    path_accuracy = matched / total_expected.
    Entry is "correct" if path_accuracy >= 0.5.
    """
    result = handler.route_multi_turn(entry, func_defs)
    predicted = []

    # Build set of available bare function names (for prereq injection)
    available_bare = {_strip_class_prefix(f["name"]) for f in func_defs}
    involved_classes = entry.get("involved_classes", [])

    # Track already-called functions across turns (for one-shot prereqs)
    already_called: set[str] = set()

    for turn_result in result["per_turn"]:
        for func in turn_result["functions"]:
            predicted.append(func)

    # Multiset coverage: greedily match predicted against expected path
    remaining = list(expected_path)
    correct = 0
    for func in predicted:
        if func in remaining:
            remaining.remove(func)
            correct += 1

    path_accuracy = correct / len(expected_path) if expected_path else 1.0

    # Build gorilla result: list[list[str]] with arg extraction per turn
    extractor = handler._extractor
    turns = entry.get("question", [])
    gorilla_turns: list[list[str]] = []
    already_called_gorilla: set[str] = set()

    for i, turn_result in enumerate(result["per_turn"]):
        funcs = turn_result["functions"]
        if not funcs:
            gorilla_turns.append([])
            continue

        # Inject prerequisites per class
        for cls in involved_classes:
            funcs = _inject_prerequisites(funcs, cls, available_bare, already_called_gorilla)

        # Extract query for this turn
        query = ""
        if i < len(turns) and turns[i]:
            for msg in reversed(turns[i]):
                if isinstance(msg, dict) and msg.get("role") == "user":
                    query = msg.get("content", "")
                    break

        # Build function calls with BARE names (no class prefix)
        # The gorilla eval's _process_method_calls adds instance names automatically
        calls = []
        for fname in funcs:
            bare = _strip_class_prefix(fname)
            fdef = next((f for f in func_defs if f["name"] == fname), None)
            args = extractor.extract(query, fdef) if fdef and query else {}
            calls.append({bare: args})
            already_called_gorilla.add(bare)
        gorilla_turns.append([json.dumps(calls)])

    return {
        "id":            entry_id,
        "expected_path": expected_path,
        "predicted":     predicted,
        "path_accuracy": path_accuracy,
        "correct":       path_accuracy >= 0.5,
        "latency_ms":    result["latency_ms"],
        "path_correct":  correct,
        "path_total":    len(expected_path),
        "gorilla_result": gorilla_turns,
    }


# ── Memory (Agentic) evaluation ─────────────────────────────────────────

def _strip_punctuation(text: str) -> str:
    """Strip punctuation for agentic_checker-style comparison."""
    return re.sub(r'[^\w\s]', '', text).strip()


def _agentic_check(response: str, ground_truths: list[str]) -> bool:
    """Check if any ground_truth is contained in the response.

    Mirrors BFCL's agentic_checker: case-insensitive, punctuation-stripped.
    """
    resp_clean = _strip_punctuation(response.lower())
    for gt in ground_truths:
        gt_clean = _strip_punctuation(gt.lower())
        if gt_clean and gt_clean in resp_clean:
            return True
    return False


def run_memory_category(
    category: str,
    max_entries: int | None = None,
    verbose: bool = False,
) -> dict:
    """Evaluate a memory category using pure HDC sentence retrieval."""
    filename = BFCL_FILES.get(category)
    if not filename:
        print(f"  Unknown category: {category}")
        return {}

    filepath = DATA_DIR / filename
    if not filepath.exists():
        print(f"  Data not found: {filepath}.")
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

    # Load and ingest prereq conversations per scenario
    handlers: dict[str, MemoryHandler] = {}
    for scenario in MEMORY_SCENARIOS:
        prereq_path = MEMORY_PREREQ_DIR / f"memory_{scenario}.json"
        if not prereq_path.exists():
            continue
        prereq_entries = load_jsonl(prereq_path)
        handler = MemoryHandler()
        handler.ingest(prereq_entries)
        handlers[scenario] = handler

    results  = []
    correct  = 0
    total    = 0

    for i, entry in enumerate(entries):
        _progress(i + 1, len(entries), category[:25])

        entry_id = entry.get("id", f"{category}_{i}")
        scenario = entry.get("scenario", "")
        question_data = entry.get("question", [])

        # Extract question text
        question = ""
        if isinstance(question_data, list):
            for turn in question_data:
                if isinstance(turn, list):
                    for msg in turn:
                        if isinstance(msg, dict) and msg.get("role") == "user":
                            question = msg.get("content", "")
                elif isinstance(turn, dict) and turn.get("role") == "user":
                    question = turn.get("content", "")

        if not question or scenario not in handlers:
            continue

        total += 1

        # Query the memory handler
        response = handlers[scenario].query(question)

        # Check against ground truth
        answer = answers.get(entry_id, {})
        ground_truths = answer.get("ground_truth", [])
        is_correct = _agentic_check(response, ground_truths)

        if is_correct:
            correct += 1

        results.append({
            "id":           entry_id,
            "question":     question[:100],
            "scenario":     scenario,
            "response":     response[:200],
            "ground_truth": ground_truths,
            "correct":      is_correct,
            "gorilla_result": [[response]],
        })

    print()

    accuracy = correct / total if total else 0.0

    if verbose:
        errors = [r for r in results if not r["correct"]]
        if errors:
            print(f"\n  Errors in {category} ({len(errors)}):")
            for e in errors[:15]:
                print(f"    [{e['id']}] scenario={e['scenario']}")
                print(f"      Q: {e['question']}")
                print(f"      GT: {e['ground_truth']}")
                print(f"      R: {e['response'][:150]}")

    return {
        "category":       category,
        "total":          total,
        "correct":        correct,
        "accuracy":       accuracy,
        "avg_latency_ms": 0.0,
        "results":        results,
    }


def _print_multi_turn_errors(category: str, results: list[dict]) -> None:
    errors = [r for r in results if not r["correct"]]
    if not errors:
        return
    print(f"\n  Errors in {category} ({len(errors)}):")
    for e in errors[:10]:
        print(f"    [{e['id']}] path_accuracy={e['path_accuracy']:.1%} "
              f"({e['path_correct']}/{e['path_total']})")
        print(f"      expected: {e['expected_path'][:6]}...")
        print(f"      got:      {e['predicted'][:6]}...")


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

    # Memory summary = avg(memory_kv, memory_vector, memory_rec_sum)
    memory_summary = avg(V4_MEMORY_CATS)
    # Agentic = avg(web_search_summary, memory_summary)
    # Web search not implemented → agentic = memory_summary only
    agentic = memory_summary

    scores = {
        "non_live":     avg(V4_NONLIVE_CATS),
        "hallucination": avg(V4_HALLUCINATION_CATS),
        "live":         wavg(V4_LIVE_CATS),
        "multi_turn":   avg(V4_MULTI_TURN_CATS),
        "agentic":      agentic,
    }

    weights = {"agentic": 0.40, "multi_turn": 0.30, "non_live": 0.10, "hallucination": 0.10, "live": 0.10}
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


def print_report(category_results: list[dict], extractor=None) -> None:
    print("\n" + "=" * 80)
    print("  GLYPHH HDC — BFCL V4 BENCHMARK RESULTS")
    print("=" * 80)

    sections = [
        ("NON-LIVE (10%)",      V4_NONLIVE_CATS),
        ("HALLUCINATION (10%)", V4_HALLUCINATION_CATS),
        ("LIVE (10%)",          V4_LIVE_CATS),
        ("MULTI-TURN (30%)",    V4_MULTI_TURN_CATS),
        ("AGENTIC (40%)",       V4_AGENTIC_CATS),
    ]

    has_tokens = extractor is not None
    cr_map = {cr["category"]: cr for cr in category_results}

    token_cols = "  {'In Tok':>10} {'Out Tok':>10} {'Calls':>7}" if has_tokens else ""
    hdr = f"  {'Category':<30} {'Accuracy':>10} {'Correct':>10} {'Total':>8} {'Lat(ms)':>10}"
    if has_tokens:
        hdr += f" {'In Tok':>10} {'Out Tok':>10} {'Calls':>7}"
    print(f"\n{hdr}")
    print("  " + "-" * (len(hdr) - 2))

    for section_name, section_cats in sections:
        section_results = [cr_map[c] for c in section_cats if c in cr_map]
        if not section_results:
            continue
        print(f"\n  ┌─ {section_name}")
        for cr in section_results:
            line = (
                f"  │ {cr['category']:<28} "
                f"{cr['accuracy']:>9.1%} "
                f"{cr['correct']:>10} "
                f"{cr['total']:>8} "
                f"{cr['avg_latency_ms']:>9.1f}"
            )
            tu = cr.get("token_usage")
            if has_tokens and tu:
                line += f" {tu['input_tokens']:>10,} {tu['output_tokens']:>10,} {tu['calls']:>7,}"
            print(line)

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
    for k in ("non_live", "hallucination", "live", "multi_turn", "agentic"):
        label = {
            "non_live": "Non-Live (10%)",
            "hallucination": "Hallucination (10%)",
            "live": "Live (10%)",
            "multi_turn": "Multi-Turn (30%)",
            "agentic": "Agentic (40%)",
        }[k]
        val = v4.get(k)
        s = f"{val:>7.1%}" if val is not None else "    N/A"
        print(f"    {label:<24} {s}")
    if v4["overall"] is not None:
        print(f"\n    OVERALL ({pct_w:.0f}% of V4):     {v4['overall']:>7.1%}")

    if has_tokens:
        usage = extractor.get_total_usage()
        # Haiku pricing: $0.80/MTok input, $4.00/MTok output
        cost_in = usage["input_tokens"] / 1_000_000 * 0.80
        cost_out = usage["output_tokens"] / 1_000_000 * 4.00
        cost_total = cost_in + cost_out
        print(f"\n  TOKEN USAGE:")
        print(f"    Input tokens:  {usage['input_tokens']:>12,}")
        print(f"    Output tokens: {usage['output_tokens']:>12,}")
        print(f"    API calls:     {usage['calls']:>12,}")
        print(f"    Est. cost:     ${cost_total:>11.4f}  (in: ${cost_in:.4f}, out: ${cost_out:.4f})")

    print("  " + "=" * (len(hdr) - 2))


# ── Gorilla leaderboard output ────────────────────────────────────────────

# ── Java/JS value formatting for BFCL AST checker ────────────────────────

JAVA_CATS = {"java"}
JS_CATS = {"javascript"}


def _java_value_to_string(value: Any) -> str:
    """Convert a Python value to its Java string representation.

    The BFCL AST checker requires ALL Java/JS parameter values to be strings.
    It then runs java_type_converter() to parse them into the correct type.
    """
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(value)
    if isinstance(value, dict):
        if not value:
            return "new HashMap<>()"
        puts = "; ".join(
            f'put("{k}", {_java_value_to_string(v)})' for k, v in value.items()
        )
        return f"new HashMap<>() {{{{ {puts}; }}}}"
    if isinstance(value, list):
        if not value:
            return "new ArrayList<>()"
        elems = ", ".join(_java_value_to_string(v) for v in value)
        return f"new ArrayList<>(Arrays.asList({elems}))"
    return str(value)


def _js_value_to_string(value: Any) -> str:
    """Convert a Python value to its JavaScript string representation."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return str(value)
    if isinstance(value, dict):
        if not value:
            return "{}"
        pairs = ", ".join(
            f"{k}: {_js_value_to_string(v)}" for k, v in value.items()
        )
        return f"{{{pairs}}}"
    if isinstance(value, list):
        if not value:
            return "[]"
        elems = ", ".join(_js_value_to_string(v) for v in value)
        return f"[{elems}]"
    return str(value)


def _convert_gorilla_result_for_language(gorilla_result: str, category: str) -> str:
    """Convert gorilla result JSON to use Java/JS string values if needed."""
    if category not in JAVA_CATS and category not in JS_CATS:
        return gorilla_result

    if not gorilla_result:
        return gorilla_result

    try:
        calls = json.loads(gorilla_result)
    except (json.JSONDecodeError, TypeError):
        return gorilla_result

    if not isinstance(calls, list):
        return gorilla_result

    converter = _java_value_to_string if category in JAVA_CATS else _js_value_to_string

    converted = []
    for call in calls:
        if not isinstance(call, dict):
            converted.append(call)
            continue
        new_call = {}
        for fname, args in call.items():
            if isinstance(args, dict):
                new_call[fname] = {k: converter(v) for k, v in args.items()}
            else:
                new_call[fname] = args
        converted.append(new_call)

    return json.dumps(converted)


def _gorilla_result_filename(category: str) -> str:
    """Map our category name to gorilla result filename.

    Gorilla uses BFCL_v4_{test_category}_result.json where test_category
    matches category_mapping.py names. Memory categories need special handling
    because they share the same source data file but get separate result files.
    """
    # Memory categories: gorilla expects separate files per backend
    if category in MEMORY_CATS:
        return f"BFCL_v4_{category}_result.json"
    # All others: derive from the source data filename
    bfcl_name = BFCL_FILES.get(category, "")
    return bfcl_name.replace(".json", "_result.json") if bfcl_name else ""


def write_gorilla_output(category_results: list[dict], output_dir: str) -> None:
    """Write gorilla-format JSONL result files for leaderboard submission.

    Produces result/<model>/<group>/BFCL_v4_<category>_result.json files
    compatible with the gorilla eval framework.
    """
    out = Path(output_dir)
    print(f"\nWriting gorilla results to {out}/")

    for cr in category_results:
        cat = cr["category"]
        group = GORILLA_GROUPS.get(cat)
        if not group:
            continue

        result_name = _gorilla_result_filename(cat)
        if not result_name:
            continue

        # Memory results go in agentic/memory/<backend>/ per gorilla convention
        if cat in MEMORY_CATS:
            backend = cat[len("memory_"):]  # kv, vector, rec_sum
            group_dir = out / group / "memory" / backend
        else:
            group_dir = out / group
        group_dir.mkdir(parents=True, exist_ok=True)

        fpath = group_dir / result_name
        with open(fpath, "w") as f:
            for entry in cr["results"]:
                raw_result = entry.get("gorilla_result", "")
                result = _convert_gorilla_result_for_language(raw_result, cat)
                gorilla_entry = {
                    "id": entry["id"],
                    "result": result,
                }
                f.write(json.dumps(gorilla_entry) + "\n")
        print(f"  {result_name:<50} {len(cr['results']):>4} entries")

    print("Done.")


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
        "--all", action="store_true",
        help="Run all V4 categories including multi-turn",
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
    parser.add_argument(
        "--output-gorilla", type=str, default=None,
        help="Directory to save gorilla-format JSONL result files for leaderboard submission",
    )
    parser.add_argument(
        "--hybrid", action="store_true",
        help="Use Claude for argument extraction (requires ANTHROPIC_API_KEY)",
    )
    parser.add_argument(
        "--llm-model", type=str, default="claude-haiku-4-5-20251001",
        help="LLM model for hybrid arg extraction (default: claude-haiku-4-5-20251001)",
    )
    args = parser.parse_args()

    if args.categories:
        categories = args.categories
    elif getattr(args, "all"):
        categories = ALL_CATS
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

    extractor = None
    if args.hybrid:
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            print("ERROR: --hybrid requires ANTHROPIC_API_KEY environment variable")
            sys.exit(1)
        from llm_extractor import LLMArgumentExtractor
        extractor = LLMArgumentExtractor(model=args.llm_model)

    handler = BFCLHandler(confidence_threshold=args.threshold, extractor=extractor)

    mode = f"hybrid (HDC routing + {args.llm_model} extraction)" if args.hybrid else "pure HDC"
    print(f"\nCategories : {categories}")
    print(f"Threshold  : {args.threshold}")
    print(f"Mode       : {mode}")
    print()

    category_results: list[dict] = []
    for cat in categories:
        print(f"── {cat} ──")
        if extractor:
            extractor.snapshot()
        if cat in MEMORY_CATS:
            result = run_memory_category(
                cat,
                max_entries=args.max_entries,
                verbose=args.verbose,
            )
        elif cat in MULTI_TURN_CATS:
            result = run_multi_turn_category(
                handler, cat,
                max_entries=args.max_entries,
                verbose=args.verbose,
            )
        else:
            result = run_category(
                handler, cat,
                max_entries=args.max_entries,
                verbose=args.verbose,
            )
        if result:
            if extractor:
                result["token_usage"] = extractor.get_category_usage()
            category_results.append(result)

    print_report(category_results, extractor=extractor)

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

    if args.output_gorilla:
        write_gorilla_output(category_results, args.output_gorilla)


if __name__ == "__main__":
    main()
