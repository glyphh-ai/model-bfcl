"""Run multi_turn_base with execution loop + HDC routing + native tool_use.

Architecture:
  1. HDC routes → filtered candidate functions (per turn)
  2. Anthropic native tool_use with filtered tools
  3. Execute calls against real BFCL Python class instances
  4. Feed results back → model continues until done
  5. Compare final state against ground truth
"""

from __future__ import annotations

import copy
import importlib
import inspect
import json
import os
import re
import sys
import time
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

from multi_turn_handler import load_func_defs
from scorer import BFCLModelScorer, _CLASS_DIR_MAP
from domain_config import CLASS_DOMAIN_CONFIGS

# BFCL execution infrastructure
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
    is_empty_execute_response,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
    multi_turn_checker,
)

MAX_ENTRIES = int(sys.argv[1]) if len(sys.argv) > 1 else 20
MODEL = os.environ.get("BFCL_MODEL", "claude-haiku-4-5-20251001")
MAX_STEPS_PER_TURN = 10
MAXIMUM_STEP_LIMIT = 15

# Result export
RESULT_DIR = _BFCL_DIR / "results" / "glyphh-hdc-loop" / "multi_turn"

# Gorilla decode utility (for roundtrip verification)
from bfcl_eval.model_handler.utils import convert_to_function_call as gorilla_convert_to_function_call


# ── Tool schema conversion ──────────────────────────────────────────

def _build_bare_to_full_map(func_defs: list[dict]) -> dict[str, str]:
    """Build mapping from bare name (cd) → full name (GorillaFileSystem.cd)."""
    m = {}
    for fd in func_defs:
        full = fd.get("name", "")
        bare = full.split(".")[-1] if "." in full else full
        m[bare] = full
    return m


def _to_anthropic_tool(fd: dict) -> dict:
    """Convert BFCL func def to Anthropic native tool schema."""
    full_name = fd.get("name", "")
    bare_name = full_name.split(".")[-1] if "." in full_name else full_name
    params = fd.get("parameters", {})

    input_schema = copy.deepcopy(params)
    # Convert BFCL "dict" type to JSON Schema "object"
    if input_schema.get("type") == "dict":
        input_schema["type"] = "object"
    if "properties" not in input_schema:
        input_schema["properties"] = {}
    # Fix non-standard BFCL types to valid JSON Schema
    for prop in input_schema.get("properties", {}).values():
        if isinstance(prop, dict):
            t = prop.get("type", "")
            if t == "dict":
                prop["type"] = "object"
            elif t == "float":
                prop["type"] = "number"
            elif t == "integer":
                prop["type"] = "integer"
            elif t == "array":
                items = prop.get("items", {})
                if isinstance(items, dict) and items.get("type") == "dict":
                    items["type"] = "object"
                if isinstance(items, dict) and items.get("type") == "float":
                    items["type"] = "number"

    return {
        "name": bare_name,
        "description": fd.get("description", ""),
        "input_schema": input_schema,
    }


def _format_fs_tree(node: dict, indent: int = 0) -> str:
    """Format a filesystem tree for the system prompt."""
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


def build_system_prompt(initial_config: dict) -> str:
    """Build system prompt with state context."""
    parts = [
        "You are an expert in composing functions. You are given a question and a set of possible functions. "
        "Based on the question, you will need to make one or more function/tool calls to achieve the purpose. "
        "If none of the functions can be used, point it out and do not make any tool calls.\n\n"
        "RULES:\n"
        "1. ALL parameters are PLAIN NAMES, never paths. No '/' in any argument.\n"
        "   WRONG: mv(destination='temp/file.pdf') or cp(destination='archives/backup.txt')\n"
        "   RIGHT: mv(destination='temp') or cp(destination='archives')\n"
        "2. cd() moves ONE level and takes a 'folder' parameter. To operate on a file in a subdirectory, cd() there FIRST, then operate.\n"
        "   To rename a file with mv(): the destination is just the new name, do NOT add a file extension unless the user explicitly says to.\n"
        "3. Use EXACT parameter names from the function schema. Do not invent parameter names.\n"
        "   cd(folder='x') NOT cd(dir='x') or cd(path='x')\n"
        "4. When posting tweets:\n"
        "   - Use the 'tags' parameter for hashtags (e.g. tags=['#topic']), do NOT embed # in the content string.\n"
        "   - Use the 'mentions' parameter for @mentions (e.g. mentions=['@user']), do NOT embed @ in the content string.\n"
        "   - For content, use the EXACT text from previous function results verbatim. Do not paraphrase or reformat.\n"
        "   - When using diff output as content, concatenate the lines without adding newlines between them.\n"
        "5. To delete a directory: first cd() into it, rm() the files inside, cd('..') back, then rmdir().\n"
        "6. When a function returns a result you need for the next call, wait for the result before proceeding.\n"
        "7. echo() creates the file if it doesn't exist and writes content. No need to call touch() before echo().\n"
        "8. Only call functions that accomplish the task. Do not add extra ls(), pwd(), find() calls unless the task explicitly requires their output.\n"
        "9. After copying a file to a directory, if the next task says to operate on that file, cd() to that directory first.\n"
        "10. mean() takes a list of ALL numbers at once: mean(numbers=[a,b,c]). Do NOT call mean() separately per number.\n"
        "11. ALWAYS make tool calls when the user asks you to do something. Never respond with just text when a function call is needed."
    ]

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
                # Show only essential API state (auth status, username) — not history
                essential = {}
                for k, v in config.items():
                    if k in ("username", "user_id", "authenticated", "token"):
                        essential[k] = v
                if essential:
                    state_context += f"  {cls_name}: {json.dumps(essential)}\n"
        parts.append(state_context)

    return "\n".join(parts)


# ── HDC routing ──────────────────────────────────────────────────────

def setup_hdc_scorers(involved_classes: list[str]) -> dict[str, BFCLModelScorer]:
    """Set up per-class HDC scorers."""
    scorers = {}
    for class_name in involved_classes:
        class_dir = _CLASS_DIR_MAP.get(class_name)
        if not class_dir:
            continue
        scorer = BFCLModelScorer()
        scorer.configure_from_db(class_dir)
        scorers[class_name] = scorer
    return scorers


def hdc_route(query: str, scorers: dict[str, BFCLModelScorer],
              all_func_defs: list[dict], involved_classes: list[str]) -> tuple[list[dict], list[tuple[str, float]]]:
    """Use HDC to filter candidate functions for this query.

    Returns (filtered func_defs, top_matches) where top_matches is [(func_name, score), ...].
    """
    # Score per class
    class_scores: dict[str, list[tuple[str, float]]] = {}
    for class_name in involved_classes:
        scorer = scorers.get(class_name)
        if not scorer:
            continue
        result = scorer.score(query)
        if not result.all_scores:
            continue
        top = [(s["function"], s["score"]) for s in result.all_scores[:8]
               if s["score"] > 0.08]
        if top:
            class_scores[class_name] = top

    if not class_scores:
        # Fallback: return all functions
        return all_func_defs, []

    # Level 1: pick relevant classes
    class_max = {cn: max(s for _, s in funcs) for cn, funcs in class_scores.items()}
    best_score = max(class_max.values())
    relevant_classes = [cn for cn, ms in class_max.items()
                        if ms >= best_score - 0.20]

    # Level 2: collect candidate function names
    candidate_bare_names: set[str] = set()
    for class_name in relevant_classes:
        # Keyword detection from DomainConfig
        domain_config = CLASS_DOMAIN_CONFIGS.get(class_name)
        if domain_config:
            query_lower = query.lower()
            for func_name, keywords in domain_config.multi_action_keywords.items():
                for kw in keywords:
                    if kw.lower() in query_lower:
                        bare = func_name.split(".")[-1] if "." in func_name else func_name
                        candidate_bare_names.add(bare)
                        break

        # HDC top-N
        for fn, sim in class_scores.get(class_name, []):
            bare = fn.split(".")[-1] if "." in fn else fn
            candidate_bare_names.add(bare)

    # Filter func_defs to candidates only
    # But ALWAYS include all functions from relevant classes (the HDC narrows, but we give the LLM
    # the full class context so it doesn't miss prerequisite calls like auth/login)
    relevant_class_set = set(relevant_classes)
    filtered = []
    seen = set()
    for fd in all_func_defs:
        full_name = fd.get("name", "")
        bare_name = full_name.split(".")[-1] if "." in full_name else full_name
        class_name = full_name.split(".")[0] if "." in full_name else ""

        # Include if: in a relevant class
        if class_name in relevant_class_set and bare_name not in seen:
            filtered.append(fd)
            seen.add(bare_name)

    # Collect top matches for hint
    top_matches = []
    for class_name in relevant_classes:
        for fn, sim in class_scores.get(class_name, [])[:3]:
            bare = fn.split(".")[-1] if "." in fn else fn
            top_matches.append((bare, sim))
    top_matches.sort(key=lambda x: -x[1])

    return filtered if filtered else all_func_defs, top_matches[:5]


# ── Call string parsing for comparison ───────────────────────────────

def _parse_call(call_str):
    m = re.match(r"(\w+)\((.*)\)$", call_str, re.DOTALL)
    if not m:
        return (call_str, {})
    func_name = m.group(1)
    args_str = m.group(2).strip()
    if not args_str:
        return (func_name, {})
    try:
        args = eval(f"dict({args_str})")
        return (func_name, args)
    except:
        pass
    try:
        vals = eval(f"({args_str},)")
        args = {f"_pos_{i}": v for i, v in enumerate(vals)}
        return (func_name, args)
    except:
        return (func_name, {"_raw": args_str})


def _calls_match(pred, gt):
    if len(pred) != len(gt):
        return False
    for (pf, pa), (gf, ga) in zip(pred, gt):
        if pf != gf:
            return False
        p_has_pos = any(k.startswith("_pos_") for k in pa)
        g_has_pos = any(k.startswith("_pos_") for k in ga)
        if p_has_pos or g_has_pos:
            p_vals = sorted(str(v) for v in pa.values())
            g_vals = sorted(str(v) for v in ga.values())
            if p_vals != g_vals:
                return False
        else:
            if pa != ga:
                return False
    return True


def _format_call(func_name: str, args: dict) -> str:
    if not args:
        return f"{func_name}()"
    parts = []
    for k, v in args.items():
        if isinstance(v, str):
            parts.append(f"{k}='{v}'")
        elif isinstance(v, bool):
            parts.append(f"{k}={v}")
        else:
            parts.append(f"{k}={v}")
    return f"{func_name}({', '.join(parts)})"


# ── Main loop ────────────────────────────────────────────────────────

def run_entry(entry: dict, gt_entry: dict, client: anthropic.Anthropic) -> tuple[list, list, dict]:
    """Run one multi-turn entry with execution loop.

    Returns (all_turn_decoded, all_turn_raw, token_usage):
      - all_turn_decoded: per-turn list of per-step call string lists (for gorilla checker)
      - all_turn_raw: per-turn list of per-step raw dicts (for result file export)
      - token_usage: {"input_tokens": int, "output_tokens": int, "api_calls": int}
    """
    scenario_id = entry["id"]
    ground_truth = gt_entry["ground_truth"]
    initial_config = entry.get("initial_config", {})
    involved_classes = entry["involved_classes"]

    # Load func defs
    func_defs = load_func_defs(involved_classes, entry.get("excluded_function", []))

    # Set up HDC scorers
    scorers = setup_hdc_scorers(involved_classes)

    # Build Anthropic tools (all available — we'll filter per-turn)
    all_tools = []
    seen_tool_names = set()
    for fd in func_defs:
        tool = _to_anthropic_tool(fd)
        if tool["name"] not in seen_tool_names:
            all_tools.append(tool)
            seen_tool_names.add(tool["name"])

    # Build system prompt
    system_prompt = build_system_prompt(initial_config)

    # Extract user queries per turn
    queries = []
    for turn_messages in entry.get("question", []):
        for msg in turn_messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                queries.append(msg.get("content", ""))
                break
        else:
            queries.append("")

    messages = []  # Conversation history for Anthropic API
    all_turn_decoded = []  # per-turn list of per-step call string lists
    all_turn_raw = []  # per-turn list of per-step raw dict lists (for export)
    token_usage = {"input_tokens": 0, "output_tokens": 0, "api_calls": 0}

    for turn_idx, query in enumerate(queries):
        if not query:
            all_turn_decoded.append([])
            all_turn_raw.append([])
            continue

        # HDC route → filter tools for this turn
        filtered_defs, top_matches = hdc_route(query, scorers, func_defs, involved_classes)
        filtered_tools = []
        filtered_names = set()
        for fd in filtered_defs:
            tool = _to_anthropic_tool(fd)
            if tool["name"] not in filtered_names:
                filtered_tools.append(tool)
                filtered_names.add(tool["name"])

        # Use all tools if HDC filtered too aggressively
        tools = filtered_tools if filtered_tools else all_tools

        # Add user message with HDC routing hint
        user_text = query
        if top_matches:
            top1 = top_matches[0][0]
            user_text += f"\n\n[System: the primary function for this request is {top1}()]"
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": user_text}],
        })

        turn_steps = []  # list of per-step call string lists
        turn_raw = []  # list of per-step raw dict lists (for export)
        step = 0

        while step < MAX_STEPS_PER_TURN:
            try:
                response = client.messages.create(
                    model=MODEL,
                    temperature=0.0,
                    max_tokens=4096,
                    system=system_prompt,
                    tools=tools,
                    messages=messages,
                )
            except Exception as e:
                print(f"    API error: {e}")
                break

            # Track token usage
            if hasattr(response, "usage") and response.usage:
                token_usage["input_tokens"] += response.usage.input_tokens
                token_usage["output_tokens"] += response.usage.output_tokens
            token_usage["api_calls"] += 1

            # Parse response
            tool_uses = []
            for block in response.content:
                if block.type == "tool_use":
                    tool_uses.append(block)

            # Add assistant message to history
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

            # Build call strings for this step
            call_strings = []
            raw_dicts = []  # for result file export
            for tu in tool_uses:
                arg_parts = [f"{k}={repr(v)}" for k, v in tu.input.items()]
                call_strings.append(f"{tu.name}({', '.join(arg_parts)})")
                # Raw dict with bare name (same as gorilla Claude handler)
                raw_dicts.append({tu.name: json.dumps(tu.input)})

            turn_steps.append(call_strings)
            turn_raw.append(raw_dicts)

            # Execute against BFCL instances
            execution_results, _ = execute_multi_turn_func_call(
                call_strings, initial_config, involved_classes,
                f"glyphh_hdc_{scenario_id}", scenario_id,
            )

            # Feed results back
            tool_results_content = []
            for tu, result in zip(tool_uses, execution_results):
                result_str = str(result)
                is_error = result_str.startswith("Error during execution:")
                tr = {
                    "type": "tool_result",
                    "tool_use_id": tu.id,
                    "content": result_str,
                }
                if is_error:
                    tr["is_error"] = True
                tool_results_content.append(tr)
            messages.append({"role": "user", "content": tool_results_content})

            step += 1
            if response.stop_reason == "end_turn":
                break

        all_turn_decoded.append(turn_steps)
        all_turn_raw.append(turn_raw)

    return all_turn_decoded, all_turn_raw, token_usage


def evaluate_with_gorilla(all_turn_decoded: list, entry: dict, gt_entry: dict) -> tuple[bool, list[str]]:
    """Evaluate using gorilla's official state checker."""
    ground_truth = gt_entry["ground_truth"]

    # Pad with empty turns if needed
    while len(all_turn_decoded) < len(ground_truth):
        all_turn_decoded.append([])

    model_name = f"hdc_llm_{MODEL.replace('-', '_')}"
    check_result = multi_turn_checker(
        multi_turn_model_result_list_decoded=all_turn_decoded,
        multi_turn_ground_truth_list=ground_truth,
        test_entry=entry,
        test_category="multi_turn_base",
        model_name=model_name,
    )

    errors = []
    if not check_result["valid"]:
        errors.append(f"  {check_result.get('error_type', 'unknown')}")
        errors.append(f"  {check_result.get('error_message', '')}")
        for turn_idx, turn_steps in enumerate(all_turn_decoded):
            flat = [c for step in turn_steps for c in step]
            gt_turn = ground_truth[turn_idx] if turn_idx < len(ground_truth) else []
            if flat != gt_turn:
                errors.append(f"  Turn {turn_idx} Got: {flat}")
                errors.append(f"  Turn {turn_idx} Exp: {gt_turn}")

    return check_result["valid"], errors


# ── Export & Roundtrip ────────────────────────────────────────────────

def export_results(all_results: list[dict], output_path: Path):
    """Export results in gorilla JSONL format.

    Each line: {"id": "...", "result": [[step_json_str, ...], ...]}
    where step_json_str is json.dumps([{ClassName.method: params_json}, ...])
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for r in all_results:
            # r["raw"] is list[list[list[dict]]] = turns[steps[{name: params_json}]]
            result_turns = []
            for turn_raw in r["raw"]:
                turn_steps = []
                for step_dicts in turn_raw:
                    # Each step is a list of {ClassName.method: json_params_str} dicts
                    turn_steps.append(json.dumps(step_dicts))
                result_turns.append(turn_steps)
            f.write(json.dumps({"id": r["id"], "result": result_turns}) + "\n")
    print(f"\nExported {len(all_results)} results to {output_path}")


def roundtrip_verify(output_path: Path, entries: list[dict], gt_entries: list[dict]):
    """Read back the result file, decode via gorilla's convert_to_function_call, run checker.

    Prints pass/fail for each entry and verifies it matches the direct check result.
    """
    print(f"\n{'='*60}")
    print(f"ROUNDTRIP VERIFICATION (re-import → decode → check)")

    # Load result file
    results = []
    with open(output_path) as f:
        for line in f:
            results.append(json.loads(line))

    # Build id→entry lookup
    entry_by_id = {e["id"]: e for e in entries}
    gt_by_id = {e["id"]: e for e in gt_entries}

    roundtrip_passed = 0
    roundtrip_total = 0

    for r in results:
        entry_id = r["id"]
        entry = entry_by_id.get(entry_id)
        gt_entry = gt_by_id.get(entry_id)
        if not entry or not gt_entry:
            print(f"  {entry_id}: SKIP (missing entry)")
            continue

        model_result_list = r["result"]  # list[list[str]] = turns[steps_json_str]
        ground_truth = gt_entry["ground_truth"]

        # Decode: same as gorilla's _evaluate_single_multi_turn_entry
        multi_turn_decoded = []
        for single_turn_steps in model_result_list:
            single_turn_decoded = []
            for step_json_str in single_turn_steps:
                try:
                    step_dicts = json.loads(step_json_str)
                    decoded = gorilla_convert_to_function_call(step_dicts)
                    if decoded:
                        single_turn_decoded.append(decoded)
                except Exception:
                    continue
            multi_turn_decoded.append(single_turn_decoded)

        # Pad
        while len(multi_turn_decoded) < len(ground_truth):
            multi_turn_decoded.append([])

        model_name = f"hdc_llm_roundtrip_{MODEL.replace('-', '_')}"
        check_result = multi_turn_checker(
            multi_turn_model_result_list_decoded=multi_turn_decoded,
            multi_turn_ground_truth_list=ground_truth,
            test_entry=entry,
            test_category="multi_turn_base",
            model_name=model_name,
        )

        roundtrip_total += 1
        status = "PASS" if check_result["valid"] else "FAIL"
        if check_result["valid"]:
            roundtrip_passed += 1
        print(f"  {entry_id}: {status}")
        if not check_result["valid"]:
            print(f"    {check_result.get('error_type', '')}: {check_result.get('error_message', '')[:120]}")

    print(f"\nROUNDTRIP: {roundtrip_passed}/{roundtrip_total} scenarios")
    return roundtrip_passed, roundtrip_total


# ── Main ─────────────────────────────────────────────────────────────

def main():
    path = _BFCL_DIR / "data" / "bfcl" / "BFCL_v4_multi_turn_base.json"
    gt_path = _BFCL_DIR / "data" / "bfcl" / "possible_answer" / "BFCL_v4_multi_turn_base.json"

    entries, gt_entries = [], []
    with open(path) as f:
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

    total_scenarios = 0
    passed_scenarios = 0
    all_export_results = []
    direct_results = {}  # id → pass/fail for roundtrip comparison
    total_tokens = {"input_tokens": 0, "output_tokens": 0, "api_calls": 0}
    t0 = time.time()

    for idx, (entry, gt_entry) in enumerate(zip(entries, gt_entries)):
        total_scenarios += 1
        scenario_id = entry["id"]
        print(f"--- {scenario_id} ---")

        try:
            all_turn_decoded, all_turn_raw, entry_tokens = run_entry(entry, gt_entry, client)
            scenario_pass, errors = evaluate_with_gorilla(all_turn_decoded, entry, gt_entry)

            # Accumulate token usage
            total_tokens["input_tokens"] += entry_tokens["input_tokens"]
            total_tokens["output_tokens"] += entry_tokens["output_tokens"]
            total_tokens["api_calls"] += entry_tokens["api_calls"]

            all_export_results.append({
                "id": scenario_id,
                "raw": all_turn_raw,
            })
            direct_results[scenario_id] = scenario_pass

            if scenario_pass:
                passed_scenarios += 1
                print(f"  PASS")
            else:
                print(f"  FAIL")
                for e in errors:
                    print(e)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  ERROR: {e}")
            direct_results[scenario_id] = False

        elapsed = time.time() - t0
        print(f"  [{idx+1}/{MAX_ENTRIES}] {passed_scenarios}/{total_scenarios} scenarios, {elapsed:.0f}s")

    print(f"\n{'='*60}")
    print(f"HDC + LLM (gorilla state checker)")
    print(f"MODEL: {MODEL}")
    print(f"FINAL: {passed_scenarios}/{total_scenarios} scenarios ({passed_scenarios/total_scenarios*100:.1f}%)")
    print(f"       {time.time()-t0:.0f}s total")
    print(f"\nTOKEN USAGE:")
    print(f"  Input tokens:  {total_tokens['input_tokens']:,}")
    print(f"  Output tokens: {total_tokens['output_tokens']:,}")
    print(f"  Total tokens:  {total_tokens['input_tokens'] + total_tokens['output_tokens']:,}")
    print(f"  API calls:     {total_tokens['api_calls']:,}")
    print(f"  Avg input/entry:  {total_tokens['input_tokens'] / max(total_scenarios, 1):,.0f}")
    print(f"  Avg output/entry: {total_tokens['output_tokens'] / max(total_scenarios, 1):,.0f}")
    print(f"  Avg calls/entry:  {total_tokens['api_calls'] / max(total_scenarios, 1):.1f}")

    # Export results
    result_path = RESULT_DIR / "BFCL_v4_multi_turn_base_result.json"
    export_results(all_export_results, result_path)

    # Roundtrip verification
    rt_passed, rt_total = roundtrip_verify(result_path, entries, gt_entries)

    # Compare direct vs roundtrip
    print(f"\nDirect: {passed_scenarios}/{total_scenarios} | Roundtrip: {rt_passed}/{rt_total}")
    if passed_scenarios == rt_passed and total_scenarios == rt_total:
        print("MATCH: Direct and roundtrip results are identical.")
    else:
        print("MISMATCH: Direct and roundtrip results differ!")


if __name__ == "__main__":
    main()
