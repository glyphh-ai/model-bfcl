#!/usr/bin/env python3
"""Full front-to-back BFCL V4 evaluation.

Runs ALL categories in order, captures:
  - Per-category accuracy, tokens, latency, cost
  - V4 composite score with proper weights
  - Leaderboard-format summary

Architecture:
  1. Non-Live (HDC routing + hybrid LLM extraction): simple, java, js, multiple, parallel, parallel_multiple
  2. Hallucination (HDC routing): irrelevance, live_irrelevance
  3. Live (HDC routing + hybrid LLM extraction): live_simple, live_multiple, live_parallel, live_parallel_multiple
  4. Multi-Turn (LLM execution loop): all 4 sub-categories
  5. Agentic Memory (pure HDC retrieval): memory_kv, memory_vector, memory_rec_sum
  6. Agentic Web Search (LLM + SerpAPI): web_search_base, web_search_no_snippet

Usage:
  python run_full_eval.py [--workers N] [--multi-turn-workers N] [--web-search-workers N]
  python run_full_eval.py --categories non_live hallucination  # run specific sections
  python run_full_eval.py --skip multi_turn web_search  # skip slow sections
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import os
import re
import statistics
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# Load env
_BFCL_DIR = Path(__file__).parent
load_dotenv(_BFCL_DIR / ".env")

# Add paths
sys.path.insert(0, str(_BFCL_DIR))
GORILLA_ROOT = Path(__file__).parent.parent.parent.parent / "gorilla" / "berkeley-function-call-leaderboard"
sys.path.insert(0, str(GORILLA_ROOT))

# Gorilla eval imports
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
    multi_turn_checker,
)
from bfcl_eval.eval_checker.agentic_eval.agentic_checker import agentic_checker
from bfcl_eval.model_handler.utils import convert_to_function_call as gorilla_convert_to_function_call

# Local imports
from scorer import BFCLModelScorer, _CLASS_DIR_MAP
from sidecar import IrrelevanceSidecar
from memory import MemoryHandler
from multi_turn_handler import MultiTurnHandler

# ── Configuration ────────────────────────────────────────────────────────

MODEL = os.environ.get("BFCL_MODEL", "claude-haiku-4-5-20251001")
DATA_DIR = _BFCL_DIR / "data" / "bfcl"
RESULT_DIR = _BFCL_DIR / "results" / "full-eval"

# Haiku 4.5 pricing
PRICE_INPUT_PER_MTOK = 0.80
PRICE_OUTPUT_PER_MTOK = 4.00

# Thread-safe printing
_print_lock = threading.Lock()
def safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs, flush=True)


# ── Data loading ─────────────────────────────────────────────────────────

def load_jsonl(path: Path, max_entries: int | None = None) -> list[dict]:
    entries = []
    with open(path) as f:
        for i, line in enumerate(f):
            if max_entries and i >= max_entries:
                break
            entries.append(json.loads(line))
    return entries


# ── Category definitions ──────────────────────────────────────────────────

BFCL_FILES = {
    "simple": "BFCL_v4_simple_python.json",
    "multiple": "BFCL_v4_multiple.json",
    "parallel": "BFCL_v4_parallel.json",
    "parallel_multiple": "BFCL_v4_parallel_multiple.json",
    "java": "BFCL_v4_simple_java.json",
    "javascript": "BFCL_v4_simple_javascript.json",
    "irrelevance": "BFCL_v4_irrelevance.json",
    "live_irrelevance": "BFCL_v4_live_irrelevance.json",
    "live_simple": "BFCL_v4_live_simple.json",
    "live_multiple": "BFCL_v4_live_multiple.json",
    "live_parallel": "BFCL_v4_live_parallel.json",
    "live_parallel_multiple": "BFCL_v4_live_parallel_multiple.json",
    "multi_turn_base": "BFCL_v4_multi_turn_base.json",
    "multi_turn_miss_func": "BFCL_v4_multi_turn_miss_func.json",
    "multi_turn_miss_param": "BFCL_v4_multi_turn_miss_param.json",
    "multi_turn_long_context": "BFCL_v4_multi_turn_long_context.json",
    "memory_kv": "BFCL_v4_memory.json",
    "memory_vector": "BFCL_v4_memory.json",
    "memory_rec_sum": "BFCL_v4_memory.json",
    "web_search": "BFCL_v4_web_search.json",
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
    "memory_kv": "possible_answer/BFCL_v4_memory.json",
    "memory_vector": "possible_answer/BFCL_v4_memory.json",
    "memory_rec_sum": "possible_answer/BFCL_v4_memory.json",
    "web_search": "possible_answer/BFCL_v4_web_search.json",
}

V4_NONLIVE_CATS = ["simple", "java", "javascript", "multiple", "parallel", "parallel_multiple"]
V4_HALLUCINATION_CATS = ["irrelevance", "live_irrelevance"]
V4_LIVE_CATS = ["live_simple", "live_multiple", "live_parallel", "live_parallel_multiple"]
V4_MULTI_TURN_CATS = ["multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"]
V4_MEMORY_CATS = ["memory_kv", "memory_vector", "memory_rec_sum"]
V4_WEB_SEARCH_CATS = ["web_search_base", "web_search_no_snippet"]

MEMORY_SCENARIOS = ["customer", "healthcare", "finance", "student", "notetaker"]
MEMORY_PREREQ_DIR = DATA_DIR / "memory_prereq_conversation"

# Function doc mapping
FUNC_DOC_DIR = DATA_DIR / "multi_turn_func_doc"
CLASS_TO_FILE = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
}

# Known class prefixes for dot-name restoration
_CLASS_PREFIXES = [
    "GorillaFileSystem", "TwitterAPI", "PostingAPI", "MessageAPI",
    "TicketAPI", "MathAPI", "TradingBot", "TravelAPI", "VehicleControlAPI",
]

# ── Shared utilities ──────────────────────────────────────────────────────

def load_func_defs(involved_classes: list[str], excluded: list[str] | None = None) -> list[dict]:
    """Load function definitions with class-prefixed names for multi-turn."""
    excluded_set = set(excluded or [])
    func_defs = []
    for cls in involved_classes:
        fname = CLASS_TO_FILE.get(cls)
        if not fname:
            continue
        fpath = FUNC_DOC_DIR / fname
        if not fpath.exists():
            continue
        with open(fpath) as f:
            for line in f:
                func = json.loads(line)
                if func["name"] not in excluded_set:
                    func = dict(func)
                    func["name"] = f"{cls}.{func['name']}"
                    func_defs.append(func)
    return func_defs


def _restore_dot_name(name: str) -> str:
    for prefix in _CLASS_PREFIXES:
        if name.startswith(prefix + "_"):
            return f"{prefix}.{name[len(prefix) + 1:]}"
    return name


def _to_anthropic_tool(fd: dict) -> dict:
    """Convert BFCL func def to Anthropic native tool schema."""
    full_name = fd.get("name", "")
    bare_name = full_name.split(".")[-1] if "." in full_name else full_name
    params = copy.deepcopy(fd.get("parameters", {}))
    if params.get("type") == "dict":
        params["type"] = "object"
    if "properties" not in params:
        params["properties"] = {}
    for prop in params.get("properties", {}).values():
        if isinstance(prop, dict):
            t = prop.get("type", "")
            if t == "dict": prop["type"] = "object"
            elif t == "float": prop["type"] = "number"
            elif t == "array":
                items = prop.get("items", {})
                if isinstance(items, dict):
                    if items.get("type") == "dict": items["type"] = "object"
                    if items.get("type") == "float": items["type"] = "number"
    return {"name": bare_name, "description": fd.get("description", ""), "input_schema": params}


def _format_fs_tree(node: dict, indent: int = 0) -> str:
    lines = []
    prefix = " " * indent
    for name, info in node.items():
        if isinstance(info, dict) and info.get("type") == "directory":
            contents = info.get("contents", {})
            child_names = list(contents.keys()) if contents else []
            lines.append(f"{prefix}{name}/ (contains: {child_names})\n")
            if contents:
                lines.append(_format_fs_tree(contents, indent + 2))
        elif isinstance(info, dict) and info.get("type") == "file":
            content = info.get("content", "")
            if content:
                lines.append(f"{prefix}{name} (file, content: {repr(content[:100])})\n")
            else:
                lines.append(f"{prefix}{name} (file)\n")
        elif isinstance(info, dict):
            lines.append(f"{prefix}{name}/\n")
            lines.append(_format_fs_tree(info, indent + 2))
    return "".join(lines)


# ── CategoryResult data class ────────────────────────────────────────────

class CategoryResult:
    def __init__(self, category: str):
        self.category = category
        self.total = 0
        self.correct = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.api_calls = 0
        self.latencies: list[float] = []  # per-entry latency in seconds
        self.results: list[dict] = []  # per-entry results for export

    @property
    def accuracy(self) -> float:
        return self.correct / self.total if self.total else 0.0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    @property
    def cost(self) -> float:
        return (self.input_tokens / 1_000_000 * PRICE_INPUT_PER_MTOK +
                self.output_tokens / 1_000_000 * PRICE_OUTPUT_PER_MTOK)

    @property
    def latency_mean(self) -> float:
        return statistics.mean(self.latencies) if self.latencies else 0.0

    @property
    def latency_sd(self) -> float:
        return statistics.stdev(self.latencies) if len(self.latencies) > 1 else 0.0

    @property
    def latency_p95(self) -> float:
        if not self.latencies:
            return 0.0
        s = sorted(self.latencies)
        idx = int(math.ceil(0.95 * len(s))) - 1
        return s[max(0, idx)]

    def to_dict(self) -> dict:
        return {
            "category": self.category,
            "total": self.total,
            "correct": self.correct,
            "accuracy": self.accuracy,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "api_calls": self.api_calls,
            "cost": self.cost,
            "latency_mean_s": self.latency_mean,
            "latency_sd_s": self.latency_sd,
            "latency_p95_s": self.latency_p95,
        }


# ══════════════════════════════════════════════════════════════════════════
# SECTION 1: Non-Live + Live + Hallucination (HDC routing + hybrid LLM)
# ══════════════════════════════════════════════════════════════════════════

IRRELEVANCE_CATS = {"irrelevance", "live_irrelevance"}
PARALLEL_CATS = {"parallel", "live_parallel", "parallel_multiple", "live_parallel_multiple"}
MULTI_ROUTE_CATS = {"parallel_multiple", "live_parallel_multiple"}


def _extract_func_defs(entry: dict) -> list[dict]:
    """Extract function definitions from a BFCL entry."""
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


def _extract_query(entry: dict) -> str:
    """Extract user query from a BFCL entry."""
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


GORILLA_TO_OPENAPI = {
    "integer": "integer", "number": "number", "float": "number",
    "string": "string", "boolean": "boolean", "bool": "boolean",
    "array": "array", "list": "array", "dict": "object", "object": "object",
    "tuple": "array", "any": "string",
}


def _cast_types_routing(properties: dict):
    """Cast BFCL types to JSON Schema types in-place."""
    for key, value in properties.items():
        if "type" not in value:
            value["type"] = "string"
        else:
            var_type = value["type"]
            if var_type == "float":
                value["format"] = "float"
                value["description"] = value.get("description", "") + " This is a float type value."
            value["type"] = GORILLA_TO_OPENAPI.get(var_type, "string")
        if value["type"] in ("array", "object"):
            if "properties" in value:
                _cast_types_routing(value["properties"])
            elif "items" in value:
                item_type = value["items"].get("type", "string")
                value["items"]["type"] = GORILLA_TO_OPENAPI.get(item_type, "string")


def _to_anthropic_tool_routing(fd: dict) -> dict:
    """Convert BFCL func def to Anthropic tool format for routing categories."""
    fd = copy.deepcopy(fd)
    name = re.sub(r"\.", "_", fd.get("name", ""))
    fd["parameters"]["type"] = "object"
    _cast_types_routing(fd["parameters"].get("properties", {}))
    tool = {
        "name": name,
        "description": fd.get("description", ""),
        "input_schema": fd["parameters"],
    }
    if "response" in fd:
        tool["description"] += f" Response schema: {json.dumps(fd['response'])}"
    return tool


def _run_routing_entry(entry: dict, gt_entry: dict | None, category: str,
                       extractor) -> dict:
    """Run one routing entry using HDC routing + LLM argument extraction.

    HDC (GlyphSpace) selects which function(s) match the query.
    LLM (Anthropic tool_use) extracts argument values for the matched functions.
    For irrelevance: HDC top score below threshold = correctly irrelevant.
    """
    entry_id = entry.get("id", "")
    func_defs = _extract_func_defs(entry)
    query = _extract_query(entry)
    is_irrelevance = category in IRRELEVANCE_CATS
    is_parallel = category in PARALLEL_CATS

    if not query:
        return {"id": entry_id, "correct": is_irrelevance, "latency": 0.0,
                "tokens": {"input_tokens": 0, "output_tokens": 0, "api_calls": 0}}

    t0 = time.time()
    token_usage = {"input_tokens": 0, "output_tokens": 0, "api_calls": 0}

    # ── HDC routing: encode functions + query, find matches ──
    scorer = BFCLModelScorer()
    scorer.configure_generic(func_defs)
    result = scorer.score(query)

    # For irrelevance: use sidecar to validate the main encoder match
    if is_irrelevance:
        top_score = result.all_scores[0]["score"] if result.all_scores else 0.0
        sidecar = IrrelevanceSidecar()
        sidecar.configure(func_defs)
        is_relevant = sidecar.is_relevant(query, top_score)
        is_irrelevant = not is_relevant
        latency = time.time() - t0
        gorilla_calls = [] if is_irrelevant else [{result.all_scores[0]["function"]: "{}"}]
        return {"id": entry_id, "correct": is_irrelevant, "latency": latency,
                "tokens": token_usage, "result": gorilla_calls}

    # Get HDC-ranked functions — top candidates above minimum threshold
    if not result.all_scores:
        latency = time.time() - t0
        return {"id": entry_id, "correct": False, "latency": latency,
                "tokens": token_usage, "result": ""}

    # For parallel categories: pass all available functions — the query asks for
    # multiple operations and HDC naturally scores one higher, but both are needed.
    # For simple/multiple: HDC narrows to top-3 candidates.
    if is_parallel:
        matched_func_defs = func_defs
    else:
        matched_names = {s["function"] for s in result.all_scores[:3]
                         if s["score"] > 0.05}
        matched_func_defs = [fd for fd in func_defs if fd["name"] in matched_names]
        if not matched_func_defs:
            matched_func_defs = func_defs[:3]  # fallback

    # ── LLM arg extraction ──
    # Detect language for Java/JS-specific prompting
    if "java" in category and "javascript" not in category:
        language = "java"
    elif "javascript" in category:
        language = "javascript"
    else:
        language = "python"

    if language in ("java", "javascript") and extractor:
        # Java/JS: use LLMArgumentExtractor with language-specific prompting
        # (native tool_use returns JSON types; Java/JS need string representations)
        if is_parallel:
            gorilla_calls = extractor.extract_parallel(query, matched_func_defs)
        else:
            # Simple/multiple: HDC top-1 function, extract args
            top_fd = matched_func_defs[0]
            args = extractor.extract(query, top_fd, language=language)
            gorilla_calls = [{top_fd["name"]: args}]

        token_usage["input_tokens"] = extractor.total_input_tokens - token_usage.get("_snap_in", 0)
        token_usage["output_tokens"] = extractor.total_output_tokens - token_usage.get("_snap_out", 0)
        token_usage["api_calls"] += 1

        # Convert to gorilla format: args as JSON strings
        tool_calls = gorilla_calls
        gorilla_calls = [{fn: json.dumps(args) for fn, args in call.items()}
                         for call in gorilla_calls]
    else:
        # Python: use native Anthropic tool_use (returns correct JSON types)
        tool_name_map = {}
        for fd in matched_func_defs:
            orig = fd.get("name", "")
            flat = re.sub(r"\.", "_", orig)
            tool_name_map[flat] = orig
        tools = [_to_anthropic_tool_routing(fd) for fd in matched_func_defs]

        system = (
            "You are an expert in composing functions. Based on the question, make one or more "
            "function/tool calls to achieve the purpose. If none of the functions can be used, "
            "point it out and refuse to make tool calls."
        )

        client = extractor._client if extractor else anthropic.Anthropic()

        try:
            response = client.messages.create(
                model=MODEL, temperature=0.0, max_tokens=1024,
                system=system, tools=tools if tools else None,
                messages=[{"role": "user", "content": query}],
            )
            if response.usage:
                token_usage["input_tokens"] += response.usage.input_tokens
                token_usage["output_tokens"] += response.usage.output_tokens
            token_usage["api_calls"] += 1
        except Exception as e:
            return {"id": entry_id, "correct": False, "latency": time.time() - t0,
                    "tokens": token_usage}

        # Extract tool calls — restore original dotted names
        tool_calls = []
        gorilla_calls = []
        for block in response.content:
            if block.type == "tool_use":
                orig_name = tool_name_map.get(block.name, block.name)
                tool_calls.append({orig_name: block.input})
                gorilla_calls.append({orig_name: json.dumps(block.input)})

    latency = time.time() - t0

    # Correctness check using gorilla's ast_checker
    correct = False
    if gt_entry and tool_calls:
        try:
            from bfcl_eval.eval_checker.ast_eval.ast_checker import ast_checker, Language

            possible_answer = gt_entry.get("ground_truth", gt_entry.get("possible_answer", []))

            if "java" in category and "javascript" not in category:
                lang = Language.JAVA
            elif "javascript" in category:
                lang = Language.JAVASCRIPT
            else:
                lang = Language.PYTHON

            check = ast_checker(
                func_description=func_defs,
                model_output=tool_calls,
                possible_answer=possible_answer,
                language=lang,
                test_category=category,
                model_name="glyphh-ada-1.1",
            )
            correct = check.get("valid", False)
        except Exception:
            pass

    return {
        "id": entry_id,
        "correct": correct,
        "latency": latency,
        "result": gorilla_calls,
        "tokens": token_usage,
    }


def run_routing_categories(categories: list[str], hybrid: bool = True,
                           workers: int = 5) -> list[CategoryResult]:
    """Run non-live, live, and hallucination categories.

    Architecture: HDC routes (which function) + LLM extracts (arg values).
    No gaming: no hint injection, no keyword lists, no prerequisite chains.
    """
    results = []

    extractor = None
    if hybrid:
        from llm_extractor import LLMArgumentExtractor
        extractor = LLMArgumentExtractor(model=MODEL)

    for cat in categories:
        print(f"\n── {cat} (workers={workers}) ──")
        cr = CategoryResult(cat)
        t0 = time.time()

        filepath = DATA_DIR / BFCL_FILES.get(cat, "")
        if not filepath.exists():
            print(f"  Data not found: {filepath}")
            continue

        entries = load_jsonl(filepath)

        # Load ground truth
        gt_by_id = {}
        gt_file = BFCL_ANSWER_FILES.get(cat)
        if gt_file:
            gt_path = DATA_DIR / gt_file
            if gt_path.exists():
                for ae in load_jsonl(gt_path):
                    gt_by_id[ae.get("id", "")] = ae

        done_count = 0
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_routing_entry, entry,
                                gt_by_id.get(entry.get("id", f"{cat}_{i}")),
                                cat, extractor): i
                for i, entry in enumerate(entries)
            }
            for future in as_completed(futures):
                r = future.result()
                cr.total += 1
                if r["correct"]:
                    cr.correct += 1
                cr.latencies.append(r["latency"])
                cr.results.append(r)

                # Aggregate token usage from per-entry results
                tokens = r.get("tokens", {})
                cr.input_tokens += tokens.get("input_tokens", 0)
                cr.output_tokens += tokens.get("output_tokens", 0)
                cr.api_calls += tokens.get("api_calls", 0)

                done_count += 1
                if done_count % 50 == 0 or done_count == len(entries):
                    elapsed = time.time() - t0
                    safe_print(f"  {cat}: {done_count}/{len(entries)} done, "
                               f"{cr.correct} correct, {elapsed:.0f}s")

        elapsed = time.time() - t0
        print(f"  {cat}: {cr.correct}/{cr.total} ({cr.accuracy:.1%}) in {elapsed:.0f}s")
        results.append(cr)

    return results


# ══════════════════════════════════════════════════════════════════════════
# SECTION 2: Multi-Turn (LLM execution loop with threading)
# ══════════════════════════════════════════════════════════════════════════

# Clean system prompt — general function-calling guidance only, no BFCL-specific gaming
MULTI_TURN_SYSTEM_PROMPT_RULES = ""

MAX_STEPS_PER_TURN = 10


def _build_mt_system_prompt(initial_config: dict) -> str:
    parts = [MULTI_TURN_SYSTEM_PROMPT_RULES]
    if initial_config:
        state_context = "\n\nSystem state (use EXACT names from here):\n"
        for cls_name, config in initial_config.items():
            if "root" in config:
                root = config["root"]
                root_name = list(root.keys())[0] if root else ""
                root_info = root.get(root_name, {})
                contents = root_info.get("contents", root_info) if isinstance(root_info, dict) else {}
                state_context += f"  {cls_name} filesystem (contents of {root_name}/):\n"
                state_context += _format_fs_tree(contents, indent=4)
            else:
                essential = {}
                for k, v in config.items():
                    if k in ("username", "user_id", "authenticated", "token"):
                        essential[k] = v
                if essential:
                    state_context += f"  {cls_name}: {json.dumps(essential)}\n"
        parts.append(state_context)
    return "\n".join(parts)


def _setup_hdc_scorers(involved_classes: list[str]) -> dict[str, BFCLModelScorer]:
    scorers = {}
    for class_name in involved_classes:
        class_dir = _CLASS_DIR_MAP.get(class_name)
        if not class_dir:
            continue
        scorer = BFCLModelScorer()
        scorer.configure_from_db(class_dir)
        scorers[class_name] = scorer
    return scorers


def _hdc_route(query: str, scorers: dict, all_func_defs: list[dict],
               involved_classes: list[str]) -> tuple[list[dict], list[tuple[str, float]]]:
    class_scores: dict[str, list[tuple[str, float]]] = {}
    for class_name in involved_classes:
        scorer = scorers.get(class_name)
        if not scorer:
            continue
        result = scorer.score(query)
        if not result.all_scores:
            continue
        top = [(s["function"], s["score"]) for s in result.all_scores[:8] if s["score"] > 0.08]
        if top:
            class_scores[class_name] = top

    if not class_scores:
        return all_func_defs, []

    class_max = {cn: max(s for _, s in funcs) for cn, funcs in class_scores.items()}
    best_score = max(class_max.values())
    relevant_classes = [cn for cn, ms in class_max.items() if ms >= best_score - 0.20]

    # Collect candidate functions from HDC scores only — no keyword gaming
    candidate_bare_names: set[str] = set()
    for class_name in relevant_classes:
        for fn, sim in class_scores.get(class_name, []):
            bare = fn.split(".")[-1] if "." in fn else fn
            candidate_bare_names.add(bare)

    relevant_class_set = set(relevant_classes)
    filtered = []
    seen = set()
    for fd in all_func_defs:
        full_name = fd.get("name", "")
        bare_name = full_name.split(".")[-1] if "." in full_name else full_name
        class_name = full_name.split(".")[0] if "." in full_name else ""
        if class_name in relevant_class_set and bare_name not in seen:
            filtered.append(fd)
            seen.add(bare_name)

    top_matches = []
    for class_name in relevant_classes:
        for fn, sim in class_scores.get(class_name, [])[:3]:
            bare = fn.split(".")[-1] if "." in fn else fn
            top_matches.append((bare, sim))
    top_matches.sort(key=lambda x: -x[1])

    return filtered if filtered else all_func_defs, top_matches[:5]


def _run_mt_entry(entry: dict, gt_entry: dict, client: anthropic.Anthropic,
                  category: str) -> dict:
    """Run one multi-turn entry — hybrid HDC routing + CognitiveLoop + multi-step LLM loop.

    Architecture:
      1. MultiTurnHandler provides HDC routing (tool filtering) + CognitiveLoop (CWD tracking)
      2. Native tool_use multi-step loop (Claude can course-correct with execution feedback)
      3. CognitiveLoop state updated after each step

    Returns result dict with pass/fail + tokens + latency + gorilla-format result.
    """
    scenario_id = entry["id"]
    ground_truth = gt_entry["ground_truth"]
    initial_config = entry.get("initial_config", {})
    involved_classes = entry["involved_classes"]

    # Exclude both explicitly excluded functions AND held-out (missed) functions
    excluded = list(entry.get("excluded_function", []))
    holdout_raw = entry.get("missed_function", {})
    for turn_funcs in holdout_raw.values():
        excluded.extend(turn_funcs)
    func_defs = load_func_defs(involved_classes, excluded)

    # missed_function: map of turn_idx -> list of function names to add back at that turn
    holdout_function = entry.get("missed_function", {})
    holdout_func_defs = {}  # bare_name -> func_def (with class prefix)
    if holdout_function:
        all_func_defs_full = load_func_defs(involved_classes)  # no exclusions
        excluded_names = {fd["name"] for fd in func_defs}
        for fd in all_func_defs_full:
            if fd["name"] not in excluded_names:
                bare = fd["name"].split(".", 1)[-1] if "." in fd["name"] else fd["name"]
                holdout_func_defs[bare] = fd

    queries = []
    for turn_messages in entry.get("question", []):
        for msg in turn_messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                queries.append(msg.get("content", ""))
                break
        else:
            queries.append("")

    # Setup MultiTurnHandler for HDC routing + CognitiveLoop
    handler = MultiTurnHandler()
    handler.setup(entry, func_defs)

    # Build system prompt with state context
    system_prompt = _build_mt_system_prompt(initial_config)

    messages = []
    all_turn_decoded = []
    all_turn_gorilla = []
    token_usage = {"input_tokens": 0, "output_tokens": 0, "api_calls": 0}
    t0 = time.time()

    try:
        for turn_idx, query in enumerate(queries):
            # Check if this is a holdout turn — add back held-out functions
            if str(turn_idx) in holdout_function:
                added_fds = []
                for hf_name in holdout_function[str(turn_idx)]:
                    if hf_name in holdout_func_defs:
                        fd = holdout_func_defs[hf_name]
                        func_defs.append(fd)
                        added_fds.append(fd)
                if added_fds:
                    handler.add_functions(added_fds)
                query = "I have updated some more functions you can choose from. What about now?"

            if not query:
                all_turn_decoded.append([])
                all_turn_gorilla.append([])
                handler.record_turn("", [])
                continue

            # HDC route to get filtered tools + CWD from CognitiveLoop
            filtered_defs, current_cwd = handler.get_filtered_tools_and_cwd(query)
            filtered_tools = []
            filtered_names = set()
            for fd in filtered_defs:
                tool = _to_anthropic_tool(fd)
                if tool["name"] not in filtered_names:
                    filtered_tools.append(tool)
                    filtered_names.add(tool["name"])

            # Fallback to all tools if filtering produced nothing
            if not filtered_tools:
                for fd in func_defs:
                    tool = _to_anthropic_tool(fd)
                    if tool["name"] not in filtered_names:
                        filtered_tools.append(tool)
                        filtered_names.add(tool["name"])

            # Inject CWD into system prompt for this turn
            turn_system = system_prompt
            if current_cwd and current_cwd != "/":
                turn_system += f"\n\nCURRENT WORKING DIRECTORY: {current_cwd}\nYou are ALREADY inside '{current_cwd}'. Do NOT cd into '{current_cwd}'."

            messages.append({"role": "user", "content": [{"type": "text", "text": query}]})

            turn_steps = []
            turn_gorilla_steps = []
            turn_raw_calls = []
            step = 0

            while step < MAX_STEPS_PER_TURN:
                try:
                    response = client.messages.create(
                        model=MODEL, temperature=0.0, max_tokens=4096,
                        system=turn_system, tools=filtered_tools, messages=messages,
                    )
                except Exception as e:
                    safe_print(f"    API error ({scenario_id}): {e}")
                    break

                if hasattr(response, "usage") and response.usage:
                    token_usage["input_tokens"] += response.usage.input_tokens
                    token_usage["output_tokens"] += response.usage.output_tokens
                token_usage["api_calls"] += 1

                tool_uses = [b for b in response.content if b.type == "tool_use"]

                assistant_content = []
                for block in response.content:
                    if block.type == "text":
                        assistant_content.append({"type": "text", "text": block.text})
                    elif block.type == "tool_use":
                        assistant_content.append({
                            "type": "tool_use", "id": block.id,
                            "name": block.name, "input": block.input,
                        })
                messages.append({"role": "assistant", "content": assistant_content})

                if not tool_uses:
                    break

                call_strings = []
                gorilla_step = []
                step_raw_calls = []
                for tu in tool_uses:
                    arg_parts = [f"{k}={repr(v)}" for k, v in tu.input.items()]
                    call_strings.append(f"{tu.name}({', '.join(arg_parts)})")
                    gorilla_step.append({tu.name: json.dumps(tu.input)})
                    step_raw_calls.append({tu.name: tu.input})

                turn_steps.append(call_strings)
                turn_gorilla_steps.append(gorilla_step)
                turn_raw_calls.extend(step_raw_calls)

                # Update CognitiveLoop state from this step's calls
                handler.update_state(step_raw_calls)

                # Execute and feed results back to Claude
                execution_results, _ = execute_multi_turn_func_call(
                    call_strings, initial_config, involved_classes,
                    f"full_eval_{scenario_id}", scenario_id,
                )

                tool_results_content = []
                for tu, result in zip(tool_uses, execution_results):
                    result_str = str(result)
                    tr = {"type": "tool_result", "tool_use_id": tu.id, "content": result_str}
                    if result_str.startswith("Error during execution:"):
                        tr["is_error"] = True
                    tool_results_content.append(tr)
                messages.append({"role": "user", "content": tool_results_content})

                step += 1
                if response.stop_reason == "end_turn":
                    break

            all_turn_decoded.append(turn_steps)
            all_turn_gorilla.append(turn_gorilla_steps)
            handler.record_turn(query, turn_raw_calls)
    finally:
        handler.reset()

    latency = time.time() - t0

    # Evaluate with gorilla state checker
    while len(all_turn_decoded) < len(ground_truth):
        all_turn_decoded.append([])

    model_name = f"full_eval_{MODEL.replace('-', '_')}"
    check_result = multi_turn_checker(
        multi_turn_model_result_list_decoded=all_turn_decoded,
        multi_turn_ground_truth_list=ground_truth,
        test_entry=entry,
        test_category=category,
        model_name=model_name,
    )

    # Pad gorilla result to match ground truth length
    while len(all_turn_gorilla) < len(ground_truth):
        all_turn_gorilla.append([])

    return {
        "id": scenario_id,
        "passed": check_result["valid"],
        "result": all_turn_gorilla,
        "tokens": token_usage,
        "latency": latency,
    }


def run_multi_turn_categories(categories: list[str], workers: int = 3) -> list[CategoryResult]:
    """Run multi-turn categories with threaded execution."""
    results = []
    client = anthropic.Anthropic()

    for cat in categories:
        print(f"\n── {cat} ──")
        cr = CategoryResult(cat)

        entries = load_jsonl(DATA_DIR / BFCL_FILES[cat])
        gt_entries = load_jsonl(DATA_DIR / BFCL_ANSWER_FILES[cat])

        t0 = time.time()
        done_count = 0

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_mt_entry, entry, gt_entry, client, cat): idx
                for idx, (entry, gt_entry) in enumerate(zip(entries, gt_entries))
            }
            for future in as_completed(futures):
                r = future.result()
                cr.total += 1
                if r["passed"]:
                    cr.correct += 1
                cr.input_tokens += r["tokens"]["input_tokens"]
                cr.output_tokens += r["tokens"]["output_tokens"]
                cr.api_calls += r["tokens"]["api_calls"]
                cr.latencies.append(r["latency"])
                cr.results.append(r)

                done_count += 1
                passed_so_far = cr.correct
                elapsed = time.time() - t0
                status = "PASS" if r["passed"] else "FAIL"
                safe_print(f"  [{done_count}/{len(entries)}] {r['id']}: {status}  "
                          f"({passed_so_far} passed, {elapsed:.0f}s)")

        elapsed = time.time() - t0
        print(f"  {cat}: {cr.correct}/{cr.total} ({cr.accuracy:.1%}) in {elapsed:.0f}s")
        results.append(cr)

    return results


# ══════════════════════════════════════════════════════════════════════════
# SECTION 3: Agentic — Memory (pure HDC, no LLM)
# ══════════════════════════════════════════════════════════════════════════

def run_memory_categories(categories: list[str]) -> list[CategoryResult]:
    """Run memory categories using pure HDC retrieval."""
    results = []

    # Load and ingest prereq conversations
    handlers: dict[str, MemoryHandler] = {}
    for scenario in MEMORY_SCENARIOS:
        prereq_path = MEMORY_PREREQ_DIR / f"memory_{scenario}.json"
        if not prereq_path.exists():
            continue
        prereq_entries = load_jsonl(prereq_path)
        handler = MemoryHandler()
        handler.ingest(prereq_entries)
        handlers[scenario] = handler

    # Load data and answers
    entries = load_jsonl(DATA_DIR / "BFCL_v4_memory.json")
    answers = {}
    for ae in load_jsonl(DATA_DIR / "possible_answer" / "BFCL_v4_memory.json"):
        answers[ae.get("id", "")] = ae

    for cat in categories:
        print(f"\n── {cat} ──")
        cr = CategoryResult(cat)
        t0 = time.time()

        for entry in entries:
            entry_id = entry.get("id", "")
            scenario = entry.get("scenario", "")
            question_data = entry.get("question", [])

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

            cr.total += 1
            entry_t0 = time.time()
            response = handlers[scenario].query(question)
            entry_lat = time.time() - entry_t0
            cr.latencies.append(entry_lat)

            answer = answers.get(entry_id, {})
            ground_truths = answer.get("ground_truth", [])
            check = agentic_checker(response, ground_truths)

            if check["valid"]:
                cr.correct += 1
            cr.results.append({
                "id": entry_id, "passed": check["valid"],
                "result": [[response]],
            })

        elapsed = time.time() - t0
        print(f"  {cat}: {cr.correct}/{cr.total} ({cr.accuracy:.1%}) in {elapsed:.1f}s")
        results.append(cr)

    return results


# ══════════════════════════════════════════════════════════════════════════
# SECTION 4: Agentic — Web Search (LLM + SerpAPI)
# ══════════════════════════════════════════════════════════════════════════

# Map env var
if os.environ.get("SERPAPI_KEY") and not os.environ.get("SERPAPI_API_KEY"):
    os.environ["SERPAPI_API_KEY"] = os.environ["SERPAPI_KEY"]


class WebSearchRunner:
    def __init__(self, show_snippet: bool = True):
        from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.web_search import WebSearchAPI
        self._api = WebSearchAPI()
        self._api.show_snippet = show_snippet

    def execute(self, func_name: str, args: dict) -> str:
        if func_name == "search_engine_query":
            result = self._api.search_engine_query(**args)
        elif func_name == "fetch_url_content":
            result = self._api.fetch_url_content(**args)
        else:
            return f"Error: unknown function {func_name}"
        if isinstance(result, dict) and "error" in result:
            return f"Error: {result['error']}"
        return json.dumps(result, indent=2) if not isinstance(result, str) else result


WS_TOOLS = [
    {
        "name": "search_engine_query",
        "description": "Search the web using DuckDuckGo. Returns results with title, href, and body.",
        "input_schema": {
            "type": "object",
            "properties": {
                "keywords": {"type": "string", "description": "The search keywords."},
                "max_results": {"type": "integer", "description": "Max results (default 10).", "default": 10},
            },
            "required": ["keywords"],
        },
    },
    {
        "name": "fetch_url_content",
        "description": "Fetch and process content from a URL.",
        "input_schema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to fetch (http:// or https://)."},
                "mode": {"type": "string", "description": "Processing mode.", "enum": ["raw", "markdown", "truncate"], "default": "truncate"},
            },
            "required": ["url"],
        },
    },
]

WS_SYSTEM_PROMPT = (
    "You are a research assistant that answers questions by searching the web. "
    "Break complex questions into sub-questions and search for each piece of information step by step. "
    "When you have found the answer, state it clearly in your final response. "
    "Use the 'truncate' mode when fetching URLs to get clean readable text. "
    "Be concise and factual. Always include the final answer explicitly in your last message."
)

MAX_WS_STEPS = 15


def _run_ws_entry(idx: int, entry: dict, gt_entry: dict, client: anthropic.Anthropic,
                  show_snippet: bool) -> dict:
    """Run one web search entry."""
    entry_id = entry["id"]
    ground_truth = gt_entry["ground_truth"]

    question = ""
    for turn in entry.get("question", []):
        for msg in turn:
            if isinstance(msg, dict) and msg.get("role") == "user":
                question = msg.get("content", "")
                break

    runner = WebSearchRunner(show_snippet=show_snippet)
    token_usage = {"input_tokens": 0, "output_tokens": 0, "api_calls": 0}
    messages = [{"role": "user", "content": question}]
    final_text = ""
    t0 = time.time()

    for step in range(MAX_WS_STEPS):
        try:
            response = client.messages.create(
                model=MODEL, temperature=0.0, max_tokens=4096,
                system=WS_SYSTEM_PROMPT, tools=WS_TOOLS, messages=messages,
            )
        except Exception as e:
            safe_print(f"    API error ({entry_id}): {e}")
            break

        if hasattr(response, "usage") and response.usage:
            token_usage["input_tokens"] += response.usage.input_tokens
            token_usage["output_tokens"] += response.usage.output_tokens
        token_usage["api_calls"] += 1

        tool_uses = []
        for block in response.content:
            if block.type == "tool_use":
                tool_uses.append(block)
            elif block.type == "text":
                final_text = block.text

        assistant_content = []
        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use", "id": block.id,
                    "name": block.name, "input": block.input,
                })
        messages.append({"role": "assistant", "content": assistant_content})

        if not tool_uses:
            break

        tool_results = []
        for tu in tool_uses:
            result_str = runner.execute(tu.name, tu.input)
            if len(result_str) > 8000:
                result_str = result_str[:8000] + "\n...[truncated]"
            tool_results.append({
                "type": "tool_result", "tool_use_id": tu.id, "content": result_str,
            })
        messages.append({"role": "user", "content": tool_results})

        if response.stop_reason == "end_turn":
            break

    latency = time.time() - t0
    check = agentic_checker(final_text, ground_truth)
    passed = check["valid"]

    status = "PASS" if passed else "FAIL"
    safe_print(f"  [{idx}] {entry_id}: {status}  (expected: {ground_truth[:3]})")

    return {
        "id": entry_id,
        "passed": passed,
        "result": [[final_text]],
        "tokens": token_usage,
        "latency": latency,
    }


def run_web_search_categories(workers: int = 5) -> list[CategoryResult]:
    """Run web_search_base and web_search_no_snippet."""
    results = []
    client = anthropic.Anthropic()

    entries = load_jsonl(DATA_DIR / "BFCL_v4_web_search.json")
    gt_entries = load_jsonl(DATA_DIR / "possible_answer" / "BFCL_v4_web_search.json")

    for variant, show_snippet in [("web_search_base", True), ("web_search_no_snippet", False)]:
        print(f"\n── {variant} (workers={workers}) ──")
        cr = CategoryResult(variant)
        t0 = time.time()

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(_run_ws_entry, idx, entry, gt_entry, client, show_snippet): idx
                for idx, (entry, gt_entry) in enumerate(zip(entries, gt_entries))
            }
            done_count = 0
            for future in as_completed(futures):
                r = future.result()
                cr.total += 1
                if r["passed"]:
                    cr.correct += 1
                cr.input_tokens += r["tokens"]["input_tokens"]
                cr.output_tokens += r["tokens"]["output_tokens"]
                cr.api_calls += r["tokens"]["api_calls"]
                cr.latencies.append(r["latency"])
                cr.results.append(r)

                done_count += 1
                elapsed = time.time() - t0
                safe_print(f"  -- {done_count}/{len(entries)} done, {cr.correct} passed, {elapsed:.0f}s")

        elapsed = time.time() - t0
        print(f"  {variant}: {cr.correct}/{cr.total} ({cr.accuracy:.1%}) in {elapsed:.0f}s")
        results.append(cr)

    return results


# ══════════════════════════════════════════════════════════════════════════
# V4 Score Computation & Reporting
# ══════════════════════════════════════════════════════════════════════════

def compute_v4_score(cr_map: dict[str, CategoryResult]) -> dict:
    def avg(cats):
        accs = [cr_map[c].accuracy for c in cats if c in cr_map]
        return sum(accs) / len(accs) if accs else None

    memory_summary = avg(V4_MEMORY_CATS)
    web_search_summary = avg(V4_WEB_SEARCH_CATS)

    # Agentic = avg(memory_summary, web_search_summary)
    agentic_parts = [x for x in [memory_summary, web_search_summary] if x is not None]
    agentic = sum(agentic_parts) / len(agentic_parts) if agentic_parts else None

    scores = {
        "non_live": avg(V4_NONLIVE_CATS),
        "hallucination": avg(V4_HALLUCINATION_CATS),
        "live": avg(V4_LIVE_CATS),
        "multi_turn": avg(V4_MULTI_TURN_CATS),
        "memory_summary": memory_summary,
        "web_search_summary": web_search_summary,
        "agentic": agentic,
    }

    weights = {"agentic": 0.40, "multi_turn": 0.30, "non_live": 0.10, "hallucination": 0.10, "live": 0.10}
    num = den = 0.0
    for k, w in weights.items():
        if scores.get(k) is not None:
            num += scores[k] * w
            den += w

    scores["overall"] = num / den if den > 0 else None
    scores["weight_covered"] = den
    return scores


def print_final_report(all_results: list[CategoryResult]):
    cr_map = {cr.category: cr for cr in all_results}

    # Aggregate totals
    total_input = sum(cr.input_tokens for cr in all_results)
    total_output = sum(cr.output_tokens for cr in all_results)
    total_calls = sum(cr.api_calls for cr in all_results)
    total_cost = sum(cr.cost for cr in all_results)
    all_latencies = []
    for cr in all_results:
        all_latencies.extend(cr.latencies)

    print(f"\n{'='*100}")
    print(f"  GLYPHH — BFCL V4 FULL EVALUATION REPORT")
    print(f"  Model: {MODEL}")
    print(f"{'='*100}")

    sections = [
        ("NON-LIVE (10%)", V4_NONLIVE_CATS),
        ("HALLUCINATION (10%)", V4_HALLUCINATION_CATS),
        ("LIVE (10%)", V4_LIVE_CATS),
        ("MULTI-TURN (30%)", V4_MULTI_TURN_CATS),
        ("AGENTIC — MEMORY", V4_MEMORY_CATS),
        ("AGENTIC — WEB SEARCH", V4_WEB_SEARCH_CATS),
    ]

    hdr = (f"  {'Category':<30} {'Acc':>8} {'Correct':>8} {'Total':>6} "
           f"{'InTok':>10} {'OutTok':>10} {'Calls':>6} {'Cost$':>8} "
           f"{'Lat Mean':>9} {'Lat SD':>8} {'Lat P95':>8}")
    print(f"\n{hdr}")
    print("  " + "-" * (len(hdr) - 2))

    for section_name, section_cats in sections:
        section_results = [cr_map[c] for c in section_cats if c in cr_map]
        if not section_results:
            continue
        print(f"\n  {section_name}")
        for cr in section_results:
            line = (
                f"  │ {cr.category:<28} "
                f"{cr.accuracy:>7.1%} "
                f"{cr.correct:>8} "
                f"{cr.total:>6} "
                f"{cr.input_tokens:>10,} "
                f"{cr.output_tokens:>10,} "
                f"{cr.api_calls:>6,} "
                f"{cr.cost:>7.4f} "
                f"{cr.latency_mean:>8.2f}s "
                f"{cr.latency_sd:>7.2f}s "
                f"{cr.latency_p95:>7.2f}s"
            )
            print(line)

    # V4 Composite
    v4 = compute_v4_score(cr_map)
    print(f"\n  {'='*95}")
    print(f"  V4 COMPOSITE SCORES:")
    for k, label, w in [
        ("non_live", "Non-Live", "10%"),
        ("hallucination", "Hallucination", "10%"),
        ("live", "Live", "10%"),
        ("multi_turn", "Multi-Turn", "30%"),
        ("memory_summary", "  Memory Summary", ""),
        ("web_search_summary", "  Web Search Summary", ""),
        ("agentic", "Agentic", "40%"),
    ]:
        val = v4.get(k)
        s = f"{val:>7.1%}" if val is not None else "    N/A"
        wt = f" ({w})" if w else ""
        print(f"    {label:<24} {s}{wt}")

    if v4["overall"] is not None:
        pct_w = v4["weight_covered"] * 100
        print(f"\n    {'OVERALL':.<24} {v4['overall']:>7.1%}  ({pct_w:.0f}% of V4 weight)")

    print(f"\n  TOTAL TOKEN USAGE:")
    print(f"    Input tokens:  {total_input:>12,}")
    print(f"    Output tokens: {total_output:>12,}")
    print(f"    Total tokens:  {total_input + total_output:>12,}")
    print(f"    API calls:     {total_calls:>12,}")
    print(f"    Total cost:    ${total_cost:>11.4f}")

    if all_latencies:
        print(f"\n  OVERALL LATENCY:")
        print(f"    Mean:  {statistics.mean(all_latencies):>8.2f}s")
        if len(all_latencies) > 1:
            print(f"    SD:    {statistics.stdev(all_latencies):>8.2f}s")
        s = sorted(all_latencies)
        p95_idx = int(math.ceil(0.95 * len(s))) - 1
        print(f"    P95:   {s[max(0, p95_idx)]:>8.2f}s")

    print(f"  {'='*95}")


def _gorilla_category_name(cat: str) -> str:
    """Map our category name to gorilla's test category name."""
    return {"simple": "simple_python", "java": "simple_java", "javascript": "simple_javascript"}.get(cat, cat)


def _gorilla_group(cat: str) -> str:
    """Map our category to gorilla's directory grouping."""
    gorilla_cat = _gorilla_category_name(cat)
    if gorilla_cat in ("simple_python", "simple_java", "simple_javascript",
                        "multiple", "parallel", "parallel_multiple", "irrelevance"):
        return "non_live"
    if gorilla_cat.startswith("live_"):
        return "live"
    if gorilla_cat.startswith("multi_turn_"):
        return "multi_turn"
    if gorilla_cat.startswith("memory_"):
        backend = gorilla_cat.replace("memory_", "")
        return os.path.join("agentic", "memory", backend)
    if gorilla_cat.startswith("web_search"):
        return os.path.join("agentic", "web_search")
    return "non_live"


GORILLA_MODEL_NAME = "glyphh-ada-1.1"


def export_gorilla_results(all_results: list[CategoryResult]):
    """Export results in gorilla's expected directory structure for bfcl evaluate."""
    gorilla_result_dir = GORILLA_ROOT / "result" / GORILLA_MODEL_NAME

    for cr in all_results:
        if not cr.results:
            continue
        gorilla_cat = _gorilla_category_name(cr.category)
        group_dir = gorilla_result_dir / _gorilla_group(cr.category)
        group_dir.mkdir(parents=True, exist_ok=True)

        fpath = group_dir / f"BFCL_v4_{gorilla_cat}_result.json"
        with open(fpath, "w") as f:
            for r in cr.results:
                rid = r.get("id", "")
                # Gorilla expects category-prefixed IDs:
                #   memory_0-... → memory_kv_0-...
                #   web_search_4 → web_search_base_4
                if gorilla_cat.startswith("memory_") and gorilla_cat != "memory" and not rid.startswith(gorilla_cat):
                    rid = rid.replace("memory_", gorilla_cat + "_", 1)
                elif gorilla_cat.startswith("web_search_") and not rid.startswith(gorilla_cat):
                    rid = rid.replace("web_search_", gorilla_cat + "_", 1)
                f.write(json.dumps({"id": rid, "result": r.get("result", [])}) + "\n")
        print(f"  Exported: {fpath.relative_to(GORILLA_ROOT)}")

    print(f"\nGorilla results exported to {gorilla_result_dir}/")


def export_all_results(all_results: list[CategoryResult]):
    """Export per-category result files + summary JSON + gorilla format."""
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    for cr in all_results:
        if not cr.results:
            continue
        fpath = RESULT_DIR / f"BFCL_v4_{cr.category}_result.json"
        with open(fpath, "w") as f:
            for r in cr.results:
                f.write(json.dumps({"id": r.get("id", ""), "result": r.get("result", [])}) + "\n")

    # Summary
    summary = {
        "model": MODEL,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "categories": {cr.category: cr.to_dict() for cr in all_results},
        "v4_scores": compute_v4_score({cr.category: cr for cr in all_results}),
    }
    with open(RESULT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults exported to {RESULT_DIR}/")

    # Also export in gorilla format
    export_gorilla_results(all_results)


# ══════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════

SECTION_MAP = {
    "non_live": V4_NONLIVE_CATS,
    "hallucination": V4_HALLUCINATION_CATS,
    "live": V4_LIVE_CATS,
    "multi_turn": V4_MULTI_TURN_CATS,
    "memory": V4_MEMORY_CATS,
    "web_search": V4_WEB_SEARCH_CATS,
}


def main():
    parser = argparse.ArgumentParser(description="BFCL V4 Full Evaluation")
    parser.add_argument("--sections", nargs="+", default=list(SECTION_MAP.keys()),
                       choices=list(SECTION_MAP.keys()),
                       help="Sections to run (default: all)")
    parser.add_argument("--skip", nargs="+", default=[],
                       choices=list(SECTION_MAP.keys()),
                       help="Sections to skip")
    parser.add_argument("--workers", type=int, default=5,
                       help="Routing (non-live/live/hallucination) parallel workers (default 5)")
    parser.add_argument("--mt-workers", type=int, default=3,
                       help="Multi-turn parallel workers (default 3)")
    parser.add_argument("--ws-workers", type=int, default=5,
                       help="Web search parallel workers (default 5)")
    parser.add_argument("--no-hybrid", action="store_true",
                       help="Disable LLM hybrid extraction for routing categories")
    args = parser.parse_args()

    sections = [s for s in args.sections if s not in args.skip]

    print(f"BFCL V4 Full Evaluation")
    print(f"Model: {MODEL}")
    print(f"Sections: {sections}")
    print(f"Routing workers: {args.workers}")
    print(f"Multi-turn workers: {args.mt_workers}")
    print(f"Web search workers: {args.ws_workers}")

    all_results: list[CategoryResult] = []
    t_start = time.time()

    # 1. Non-Live + Hallucination + Live (HDC routing)
    routing_sections = [s for s in ["non_live", "hallucination", "live"] if s in sections]
    if routing_sections:
        routing_cats = []
        for s in routing_sections:
            routing_cats.extend(SECTION_MAP[s])
        results = run_routing_categories(routing_cats, hybrid=not args.no_hybrid,
                                         workers=args.workers)
        all_results.extend(results)

    # 2. Multi-Turn (LLM execution loop)
    if "multi_turn" in sections:
        results = run_multi_turn_categories(V4_MULTI_TURN_CATS, workers=args.mt_workers)
        all_results.extend(results)

    # 3. Memory (pure HDC)
    if "memory" in sections:
        results = run_memory_categories(V4_MEMORY_CATS)
        all_results.extend(results)

    # 4. Web Search (LLM + SerpAPI)
    if "web_search" in sections:
        results = run_web_search_categories(workers=args.ws_workers)
        all_results.extend(results)

    total_time = time.time() - t_start
    print(f"\n\nTotal evaluation time: {total_time:.0f}s ({total_time/60:.1f}m)")

    # Report & export
    print_final_report(all_results)
    export_all_results(all_results)


if __name__ == "__main__":
    main()
