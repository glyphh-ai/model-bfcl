#!/usr/bin/env python3
"""Debug a single multi-turn entry — compare vanilla Claude vs Glyphh-filtered."""

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, "../../glyphh-runtime")
sys.path.insert(0, str(Path(__file__).parent))

GORILLA_ROOT = Path(__file__).parent.parent.parent.parent / "gorilla" / "gorilla" / "berkeley-function-call-leaderboard"
sys.path.insert(0, str(GORILLA_ROOT))

import anthropic
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import execute_multi_turn_func_call
from multi_turn_handler import MultiTurnHandler

MODEL = "claude-opus-4-5-20251101"
DATA_DIR = Path(__file__).parent / "data" / "bfcl"

BFCL_FILES = {
    "multi_turn_base": "BFCL_v4_multi_turn_base.json",
    "multi_turn_miss_func": "BFCL_v4_multi_turn_miss_func.json",
    "multi_turn_miss_param": "BFCL_v4_multi_turn_miss_param.json",
    "multi_turn_long_context": "BFCL_v4_multi_turn_long_context.json",
}
BFCL_ANSWER_FILES = {k: f"possible_answer/{v}" for k, v in BFCL_FILES.items()}

FUNC_DOC_DIR = DATA_DIR / "multi_turn_func_doc"

def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]

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

def load_func_defs(involved_classes, excluded=None):
    excluded = set(excluded or [])
    defs = []
    for cls in involved_classes:
        fname = CLASS_TO_FILE.get(cls)
        if not fname:
            continue
        path = FUNC_DOC_DIR / fname
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                fd = json.loads(line)
                if fd["name"] not in excluded:
                    fd = dict(fd)
                    fd["name"] = f"{cls}.{fd['name']}"
                    defs.append(fd)
    return defs

def _to_anthropic_tool(fd):
    import copy as _copy
    full_name = fd.get("name", "")
    bare_name = full_name.split(".")[-1] if "." in full_name else full_name
    params = _copy.deepcopy(fd.get("parameters", {}))
    if params.get("type") == "dict":
        params["type"] = "object"
    if "type" not in params:
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

def find_entry(entry_id):
    for cat in BFCL_FILES:
        entries = load_jsonl(DATA_DIR / BFCL_FILES[cat])
        gt_entries = load_jsonl(DATA_DIR / BFCL_ANSWER_FILES[cat])
        for entry, gt_entry in zip(entries, gt_entries):
            if entry["id"] == entry_id:
                return entry, gt_entry, cat
    return None, None, None

def run_entry(entry, gt_entry, category, mode="glyphh"):
    """Run a single entry. mode='vanilla' or 'glyphh'."""
    scenario_id = entry["id"]
    ground_truth = gt_entry["ground_truth"]
    initial_config = entry.get("initial_config", {})
    involved_classes = entry["involved_classes"]

    excluded = list(entry.get("excluded_function", []))
    func_defs = load_func_defs(involved_classes, excluded)

    # Build all tools (deduped by bare name)
    all_tools = []
    all_names = set()
    for fd in func_defs:
        tool = _to_anthropic_tool(fd)
        if tool["name"] not in all_names:
            all_tools.append(tool)
            all_names.add(tool["name"])

    queries = []
    for turn_messages in entry.get("question", []):
        for msg in turn_messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                queries.append(msg.get("content", ""))
                break
        else:
            queries.append("")

    handler = MultiTurnHandler()
    handler.setup(entry, func_defs)

    client = anthropic.Anthropic()
    messages = []
    total_input = 0
    total_output = 0
    total_calls = 0

    # No system prompt — it hurts performance

    label = "VANILLA" if mode == "vanilla" else "GLYPHH"
    print(f"\n{'='*80}")
    print(f"[{label}] SCENARIO: {scenario_id}")
    print(f"Classes: {involved_classes}  |  All tools: {sorted(all_names)}")
    print(f"Turns: {len(queries)}")
    print(f"{'='*80}")

    all_turn_calls = []

    for turn_idx, query in enumerate(queries):
        print(f"\n{'─'*80}")
        print(f"[{label}] TURN {turn_idx}")
        print(f"{'─'*80}")
        print(f"USER: {query}")
        print(f"GT:   {ground_truth[turn_idx]}")

        # Decide which tools to send
        if mode == "vanilla":
            turn_tools = all_tools
            print(f"TOOLS: ALL ({len(all_tools)})")
        else:
            # Glyphh pattern matching to filter tools
            pm_funcs, pm_conf = handler._match_pattern(query)
            print(f"PATTERN: {pm_funcs} (conf={pm_conf:.3f})")

            # Trust the pattern match — if it includes ls/pwd, they're intentional
            # If it doesn't, they shouldn't be there
            if pm_funcs is not None and pm_conf >= handler.PATTERN_CONFIDENCE_THRESHOLD:
                pattern_set = set(pm_funcs)
                filtered = [fd for fd in func_defs if fd["name"].split(".")[-1] in pattern_set]
                filtered_defs = filtered if filtered else func_defs
            else:
                filtered_defs = func_defs

            turn_tools = []
            turn_names = set()
            for fd in filtered_defs:
                tool = _to_anthropic_tool(fd)
                if tool["name"] not in turn_names:
                    turn_tools.append(tool)
                    turn_names.add(tool["name"])
            print(f"TOOLS: {sorted(turn_names)} ({len(turn_tools)})")

        # No system prompt in either mode
        messages.append({"role": "user", "content": [{"type": "text", "text": query}]})

        step = 0
        turn_raw_calls = []
        while step < 10:
            total_calls += 1
            api_kwargs = dict(
                model=MODEL, temperature=0.0, max_tokens=8192,
                tools=turn_tools,
                messages=messages,
            )
            if mode == "glyphh" and pm_funcs is not None:
                # Force tool calls until all pattern tools have been called at least once
                called_so_far = set(k for raw in turn_raw_calls for k in raw.keys())
                if not set(pm_funcs).issubset(called_so_far):
                    api_kwargs["tool_choice"] = {"type": "any"}
            response = client.messages.create(**api_kwargs)
            total_input += response.usage.input_tokens
            total_output += response.usage.output_tokens

            tool_uses = []
            for block in response.content:
                if block.type == "text" and block.text:
                    print(f"  TEXT: {block.text[:300]}")
                elif block.type == "tool_use":
                    tool_uses.append(block)
                    print(f"  CALL: {block.name}({json.dumps(block.input)})")

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
            step_raw_calls = []
            for tu in tool_uses:
                arg_parts = [f"{k}={repr(v)}" for k, v in tu.input.items()]
                call_strings.append(f"{tu.name}({', '.join(arg_parts)})")
                step_raw_calls.append({tu.name: tu.input})

            turn_raw_calls.extend(step_raw_calls)
            handler.update_state(step_raw_calls)

            execution_results, _ = execute_multi_turn_func_call(
                call_strings, initial_config, involved_classes,
                f"{mode}_{scenario_id}", scenario_id,
            )

            tool_results_content = []
            for tu, result in zip(tool_uses, execution_results):
                result_str = str(result)
                print(f"  RESULT({tu.name}): {result_str[:300]}")
                tr = {"type": "tool_result", "tool_use_id": tu.id, "content": result_str}
                if result_str.startswith("Error during execution:"):
                    tr["is_error"] = True
                tool_results_content.append(tr)
            messages.append({"role": "user", "content": tool_results_content})

            step += 1
            if response.stop_reason == "end_turn":
                break

        handler.record_turn(query, turn_raw_calls)
        all_turn_calls.append(turn_raw_calls)

    handler.reset()

    print(f"\n{'─'*80}")
    print(f"[{label}] SUMMARY")
    print(f"API calls: {total_calls}  |  Input: {total_input:,}  |  Output: {total_output:,}")
    cost = total_input / 1_000_000 * 15 + total_output / 1_000_000 * 75
    print(f"Est. cost: ${cost:.4f}")

    # Show call comparison
    print(f"\nCALL COMPARISON:")
    for i, (gt_calls, model_calls) in enumerate(zip(ground_truth, all_turn_calls)):
        model_strs = []
        for raw in model_calls:
            for fname, args in raw.items():
                parts = [f"{k}={repr(v)}" for k, v in args.items()]
                model_strs.append(f"{fname}({', '.join(parts)})")
        match = "MATCH" if str(gt_calls) == str(model_strs) else "DIFF"
        print(f"  Turn {i} [{match}]:")
        print(f"    GT:    {gt_calls}")
        print(f"    Model: {model_strs}")
    print(f"{'='*80}")

    return {
        "mode": mode,
        "entry_id": scenario_id,
        "api_calls": total_calls,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "cost": cost,
        "turns": all_turn_calls,
    }


if __name__ == "__main__":
    entry_id = sys.argv[1] if len(sys.argv) > 1 else "multi_turn_base_4"
    modes = sys.argv[2:] if len(sys.argv) > 2 else ["vanilla", "glyphh"]

    entry, gt_entry, cat = find_entry(entry_id)
    if entry is None:
        print(f"Entry {entry_id} not found")
        sys.exit(1)

    import copy
    results = {}
    for mode in modes:
        # Deep copy to prevent mutation between runs
        results[mode] = run_entry(copy.deepcopy(entry), copy.deepcopy(gt_entry), cat, mode=mode)

    if len(results) == 2 and "vanilla" in results and "glyphh" in results:
        v = results["vanilla"]
        g = results["glyphh"]
        print(f"\n{'='*80}")
        print(f"DELTA: {entry_id}")
        print(f"{'='*80}")
        print(f"  API calls:    vanilla={v['api_calls']}  glyphh={g['api_calls']}  delta={g['api_calls']-v['api_calls']}")
        print(f"  Input tokens: vanilla={v['input_tokens']:,}  glyphh={g['input_tokens']:,}  delta={g['input_tokens']-v['input_tokens']:,}")
        print(f"  Output tokens:vanilla={v['output_tokens']:,}  glyphh={g['output_tokens']:,}  delta={g['output_tokens']-v['output_tokens']:,}")
        print(f"  Cost:         vanilla=${v['cost']:.4f}  glyphh=${g['cost']:.4f}  delta=${g['cost']-v['cost']:.4f}")
