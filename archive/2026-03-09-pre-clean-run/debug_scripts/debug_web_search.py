"""Run BFCL V4 web_search evaluation with Anthropic tool_use + SerpAPI.

Architecture:
  1. Give LLM two tools: search_engine_query() and fetch_url_content()
  2. LLM searches, reads results, chains multi-hop queries
  3. LLM produces final text answer
  4. Check if ground truth answer appears in response (agentic_checker)
  5. Run twice: web_search_base (with snippets) and web_search_no_snippet

Usage:
  python debug_web_search.py [N] [base|no_snippet|both] [--workers W]
  python debug_web_search.py 100 base --workers 10
"""

from __future__ import annotations

import json
import os
import re
import sys
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import anthropic
from dotenv import load_dotenv

# Load env
_BFCL_DIR = Path(__file__).parent
load_dotenv(_BFCL_DIR / ".env")

# Add gorilla path for WebSearchAPI + checker
GORILLA_ROOT = Path(__file__).parent.parent.parent.parent / "gorilla" / "berkeley-function-call-leaderboard"
sys.path.insert(0, str(GORILLA_ROOT))

from bfcl_eval.eval_checker.agentic_eval.agentic_checker import agentic_checker

# Map env var name: our .env uses SERPAPI_KEY, gorilla expects SERPAPI_API_KEY
if os.environ.get("SERPAPI_KEY") and not os.environ.get("SERPAPI_API_KEY"):
    os.environ["SERPAPI_API_KEY"] = os.environ["SERPAPI_KEY"]

# Parse args
MAX_ENTRIES = int(sys.argv[1]) if len(sys.argv) > 1 else 10
VARIANT_FILTER = sys.argv[2] if len(sys.argv) > 2 and not sys.argv[2].startswith("-") else "both"
WORKERS = 5  # default parallel workers
for i, arg in enumerate(sys.argv):
    if arg == "--workers" and i + 1 < len(sys.argv):
        WORKERS = int(sys.argv[i + 1])

MODEL = os.environ.get("BFCL_MODEL", "claude-haiku-4-5-20251001")
MAX_STEPS = 15

# Result export
RESULT_DIR = _BFCL_DIR / "results" / "glyphh-hdc-loop"

# Thread-safe print
_print_lock = threading.Lock()
def safe_print(*args, **kwargs):
    with _print_lock:
        print(*args, **kwargs)


# ── WebSearchAPI wrapper ─────────────────────────────────────────────

class WebSearchRunner:
    """Wraps gorilla's WebSearchAPI for tool execution."""

    def __init__(self, show_snippet: bool = True):
        from bfcl_eval.eval_checker.multi_turn_eval.func_source_code.web_search import WebSearchAPI
        self._api = WebSearchAPI()
        self._api.show_snippet = show_snippet

    def execute(self, func_name: str, args: dict) -> str:
        """Execute a web search function and return result as string."""
        if func_name == "search_engine_query":
            result = self._api.search_engine_query(**args)
        elif func_name == "fetch_url_content":
            result = self._api.fetch_url_content(**args)
        else:
            return f"Error: unknown function {func_name}"

        if isinstance(result, dict) and "error" in result:
            return f"Error: {result['error']}"
        return json.dumps(result, indent=2) if not isinstance(result, str) else result


# ── Anthropic tool schemas ───────────────────────────────────────────

SEARCH_TOOL = {
    "name": "search_engine_query",
    "description": "Search the web using DuckDuckGo. Returns a list of results with title, href, and body (snippet).",
    "input_schema": {
        "type": "object",
        "properties": {
            "keywords": {
                "type": "string",
                "description": "The search keywords.",
            },
            "max_results": {
                "type": "integer",
                "description": "Max results to return (default 10).",
                "default": 10,
            },
        },
        "required": ["keywords"],
    },
}

FETCH_TOOL = {
    "name": "fetch_url_content",
    "description": "Fetch and process content from a URL. Returns the page content.",
    "input_schema": {
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch. Must start with http:// or https://.",
            },
            "mode": {
                "type": "string",
                "description": "Processing mode: 'raw' (HTML), 'markdown' (readable), or 'truncate' (clean text).",
                "enum": ["raw", "markdown", "truncate"],
                "default": "truncate",
            },
        },
        "required": ["url"],
    },
}

TOOLS = [SEARCH_TOOL, FETCH_TOOL]

SYSTEM_PROMPT = (
    "You are a research assistant that answers questions by searching the web. "
    "Break complex questions into sub-questions and search for each piece of information step by step. "
    "When you have found the answer, state it clearly in your final response. "
    "Use the 'truncate' mode when fetching URLs to get clean readable text. "
    "Be concise and factual. Always include the final answer explicitly in your last message."
)


# ── Entry runner ─────────────────────────────────────────────────────

def run_entry(entry: dict, gt_entry: dict, client: anthropic.Anthropic,
              show_snippet: bool = True) -> tuple[str, dict]:
    """Run one web_search entry.

    Returns (final_response_text, token_usage).
    """
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

    for step in range(MAX_STEPS):
        try:
            response = client.messages.create(
                model=MODEL,
                temperature=0.0,
                max_tokens=4096,
                system=SYSTEM_PROMPT,
                tools=TOOLS,
                messages=messages,
            )
        except Exception as e:
            safe_print(f"    API error: {e}")
            break

        # Track tokens
        if hasattr(response, "usage") and response.usage:
            token_usage["input_tokens"] += response.usage.input_tokens
            token_usage["output_tokens"] += response.usage.output_tokens
        token_usage["api_calls"] += 1

        # Collect tool uses and text
        tool_uses = []
        for block in response.content:
            if block.type == "tool_use":
                tool_uses.append(block)
            elif block.type == "text":
                final_text = block.text

        # Add assistant message
        assistant_content = []
        for block in response.content:
            if block.type == "text":
                assistant_content.append({"type": "text", "text": block.text})
            elif block.type == "tool_use":
                assistant_content.append({
                    "type": "tool_use",
                    "id": block.id,
                    "name": block.name,
                    "input": block.input,
                })
        messages.append({"role": "assistant", "content": assistant_content})

        if not tool_uses:
            break

        # Execute tools
        tool_results = []
        for tu in tool_uses:
            result_str = runner.execute(tu.name, tu.input)
            # Truncate very long results to avoid context overflow
            if len(result_str) > 8000:
                result_str = result_str[:8000] + "\n...[truncated]"
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": tu.id,
                "content": result_str,
            })
        messages.append({"role": "user", "content": tool_results})

        if response.stop_reason == "end_turn":
            break

    return final_text, token_usage


def process_one(idx: int, entry: dict, gt_entry: dict, client: anthropic.Anthropic,
                show_snippet: bool) -> dict:
    """Process a single entry — designed for parallel execution."""
    entry_id = entry["id"]
    ground_truth = gt_entry["ground_truth"]

    try:
        response_text, entry_tokens = run_entry(entry, gt_entry, client, show_snippet=show_snippet)
        check = agentic_checker(response_text, ground_truth)
        passed = check["valid"]

        if passed:
            safe_print(f"  [{idx}] {entry_id}: PASS  (expected: {ground_truth})")
        else:
            answer_preview = response_text[:120].replace('\n', ' ') if response_text else "(empty)"
            safe_print(f"  [{idx}] {entry_id}: FAIL  (expected: {ground_truth})")
            safe_print(f"         got: {answer_preview}")

        return {
            "idx": idx,
            "id": entry_id,
            "passed": passed,
            "result": [[response_text]],
            "tokens": entry_tokens,
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        safe_print(f"  [{idx}] {entry_id}: ERROR: {e}")
        return {
            "idx": idx,
            "id": entry_id,
            "passed": False,
            "result": [[""]],
            "tokens": {"input_tokens": 0, "output_tokens": 0, "api_calls": 0},
        }


# ── Main ─────────────────────────────────────────────────────────────

def main():
    data_path = _BFCL_DIR / "data" / "bfcl" / "BFCL_v4_web_search.json"
    gt_path = _BFCL_DIR / "data" / "bfcl" / "possible_answer" / "BFCL_v4_web_search.json"

    entries, gt_entries = [], []
    with open(data_path) as f:
        for i, line in enumerate(f):
            if i >= MAX_ENTRIES:
                break
            entries.append(json.loads(line))
    with open(gt_path) as f:
        for i, line in enumerate(f):
            if i >= MAX_ENTRIES:
                break
            gt_entries.append(json.loads(line))

    client = anthropic.Anthropic()

    # Determine which variants to run
    variants = []
    if VARIANT_FILTER in ("both", "base"):
        variants.append(("web_search_base", True))
    if VARIANT_FILTER in ("both", "no_snippet"):
        variants.append(("web_search_no_snippet", False))

    for variant, show_snippet in variants:
        print(f"\n{'='*60}")
        print(f"  {variant} (show_snippet={show_snippet}, workers={WORKERS})")
        print(f"{'='*60}")

        t0 = time.time()

        # Run entries in parallel
        results_by_idx = {}
        with ThreadPoolExecutor(max_workers=WORKERS) as executor:
            futures = {
                executor.submit(process_one, idx, entry, gt_entry, client, show_snippet): idx
                for idx, (entry, gt_entry) in enumerate(zip(entries, gt_entries))
            }

            done_count = 0
            for future in as_completed(futures):
                result = future.result()
                results_by_idx[result["idx"]] = result
                done_count += 1
                passed_so_far = sum(1 for r in results_by_idx.values() if r["passed"])
                elapsed = time.time() - t0
                safe_print(f"  -- {done_count}/{len(entries)} done, {passed_so_far} passed, {elapsed:.0f}s")

        # Collect results in order
        total_tokens = {"input_tokens": 0, "output_tokens": 0, "api_calls": 0}
        all_results = []
        passed = 0
        for idx in range(len(entries)):
            r = results_by_idx[idx]
            if r["passed"]:
                passed += 1
            total_tokens["input_tokens"] += r["tokens"]["input_tokens"]
            total_tokens["output_tokens"] += r["tokens"]["output_tokens"]
            total_tokens["api_calls"] += r["tokens"]["api_calls"]
            all_results.append({"id": r["id"], "result": r["result"]})

        total = len(entries)
        print(f"\n{'='*60}")
        print(f"{variant} RESULTS")
        print(f"MODEL: {MODEL}")
        print(f"FINAL: {passed}/{total} ({passed/max(total,1)*100:.1f}%)")
        print(f"       {time.time()-t0:.0f}s total")
        print(f"\nTOKEN USAGE:")
        print(f"  Input tokens:  {total_tokens['input_tokens']:,}")
        print(f"  Output tokens: {total_tokens['output_tokens']:,}")
        print(f"  Total tokens:  {total_tokens['input_tokens'] + total_tokens['output_tokens']:,}")
        print(f"  API calls:     {total_tokens['api_calls']:,}")
        print(f"  Avg input/entry:  {total_tokens['input_tokens'] / max(total, 1):,.0f}")
        print(f"  Avg output/entry: {total_tokens['output_tokens'] / max(total, 1):,.0f}")
        print(f"  Avg calls/entry:  {total_tokens['api_calls'] / max(total, 1):.1f}")

        # Export results
        result_path = RESULT_DIR / "web_search" / f"BFCL_v4_{variant}_result.json"
        result_path.parent.mkdir(parents=True, exist_ok=True)
        with open(result_path, "w") as f:
            for r in all_results:
                f.write(json.dumps(r) + "\n")
        print(f"\nExported to {result_path}")


if __name__ == "__main__":
    main()
