"""Clean LLM-only multi-turn eval. No Glyphh, no post-processing.

Exactly mirrors what the gorilla eval does:
  1. Convert function defs to Anthropic tool format
  2. Send user message + tools → Claude
  3. Claude responds with tool_use blocks
  4. Execute against real Python class instances
  5. Feed tool_result back
  6. Loop until Claude stops making tool calls
  7. Compare state against ground truth
"""

from __future__ import annotations

import copy
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent / ".env")

from anthropic import Anthropic
from anthropic.types import TextBlock, ToolUseBlock

# Gorilla eval imports
GORILLA_ROOT = Path(__file__).parent.parent.parent.parent / "gorilla" / "berkeley-function-call-leaderboard"
sys.path.insert(0, str(GORILLA_ROOT))

from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
)
from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_checker import (
    multi_turn_checker,
)

_BFCL_DIR = Path(__file__).parent
MAX_ENTRIES = int(sys.argv[1]) if len(sys.argv) > 1 else 5
MODEL = os.environ.get("BFCL_MODEL", "claude-haiku-4-5-20251001")
MAX_STEPS_PER_TURN = 20
TEMPERATURE = 0.0

# Function doc file mapping (same as gorilla)
_FUNC_DOC_DIR = _BFCL_DIR / "data" / "bfcl" / "multi_turn_func_doc"
_FUNC_DOC_MAP = {
    "GorillaFileSystem": "gorilla_file_system.json",
    "MathAPI": "math_api.json",
    "MessageAPI": "message_api.json",
    "TwitterAPI": "posting_api.json",
    "TicketAPI": "ticket_api.json",
    "TradingBot": "trading_bot.json",
    "TravelAPI": "travel_booking.json",
    "VehicleControlAPI": "vehicle_control.json",
}


def load_func_defs_for_entry(entry: dict) -> list[dict]:
    """Load function definitions for an entry based on involved_classes."""
    functions = []
    for cls_name in entry["involved_classes"]:
        fname = _FUNC_DOC_MAP.get(cls_name)
        if not fname:
            continue
        with open(_FUNC_DOC_DIR / fname) as f:
            for line in f:
                functions.append(json.loads(line))
    return functions


# Type mappings (same as gorilla)
GORILLA_TO_OPENAPI = {
    "integer": "integer", "number": "number", "float": "number",
    "string": "string", "boolean": "boolean", "bool": "boolean",
    "array": "array", "list": "array", "dict": "object", "object": "object",
    "tuple": "array", "any": "string",
}


def convert_to_anthropic_tools(functions: list[dict]) -> list[dict]:
    """Convert BFCL function defs to Anthropic tool format."""
    tools = []
    for item in copy.deepcopy(functions):
        # Replace dots in names (Anthropic doesn't support them)
        item["name"] = re.sub(r"\.", "_", item["name"])
        item["parameters"]["type"] = "object"
        # Cast types
        _cast_types(item["parameters"].get("properties", {}))
        # Anthropic format: input_schema instead of parameters
        tool = {
            "name": item["name"],
            "description": item.get("description", ""),
            "input_schema": item["parameters"],
        }
        # Handle response field
        if "response" in item:
            tool["description"] += f" The response field has the following schema: {json.dumps(item['response'])}"
        tools.append(tool)
    return tools


def _cast_types(properties: dict):
    """Cast BFCL types to OpenAPI types in-place."""
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
                _cast_types(value["properties"])
            elif "items" in value:
                item_type = value["items"].get("type", "string")
                value["items"]["type"] = GORILLA_TO_OPENAPI.get(item_type, "string")
                if value["items"]["type"] == "array" and "items" in value["items"]:
                    sub_type = value["items"]["items"].get("type", "string")
                    value["items"]["items"]["type"] = GORILLA_TO_OPENAPI.get(sub_type, "string")
                elif value["items"]["type"] == "object" and "properties" in value["items"]:
                    _cast_types(value["items"]["properties"])


def decode_tool_calls(tool_calls: list[dict]) -> list[str]:
    """Convert Anthropic tool_use responses to gorilla function call strings.

    Returns list like: ["GorillaFileSystem.cd(folder='document')", ...]
    """
    result = []
    for tc in tool_calls:
        name = list(tc.keys())[0]
        params = json.loads(tc[name])
        # Convert back from underscore to dot notation
        # e.g., GorillaFileSystem_cd -> GorillaFileSystem.cd
        parts = name.split("_", 1)
        if len(parts) == 2:
            # Try to reconstruct the class.method format
            func_name = name  # keep as-is for now, decode_execute handles it
        else:
            func_name = name

        # Build the function call string
        args_parts = []
        for k, v in params.items():
            args_parts.append(f"{k}={repr(v)}")
        call_str = f"{func_name}({', '.join(args_parts)})"
        result.append({func_name: json.dumps(params)})
    return result


def convert_to_function_call(tool_outputs: list[dict]) -> list[str]:
    """Convert tool outputs to executable function call strings.

    Same as gorilla's convert_to_function_call.
    """
    result = []
    for item in tool_outputs:
        func_name = list(item.keys())[0]
        params = json.loads(item[func_name])
        # Reconstruct dot notation: GorillaFileSystem_cd -> GorillaFileSystem.cd
        # The underscore replacement was done for API compatibility
        # We need to find where the class name ends
        func_name_dotted = _restore_dot_name(func_name)
        args_str = ", ".join(f"{k}={repr(v)}" for k, v in params.items())
        result.append(f"{func_name_dotted}({args_str})")
    return result


def _restore_dot_name(name: str) -> str:
    """Restore GorillaFileSystem_cd to GorillaFileSystem.cd etc."""
    # Known class prefixes
    prefixes = [
        "GorillaFileSystem", "TwitterAPI", "PostingAPI", "MessageAPI",
        "TicketAPI", "MathAPI", "TradingBot", "TravelAPI", "VehicleControlAPI",
    ]
    for prefix in prefixes:
        if name.startswith(prefix + "_"):
            method = name[len(prefix) + 1:]
            return f"{prefix}.{method}"
    return name


def run_entry(client: Anthropic, entry: dict, gt_entry: dict, tools: list[dict]) -> dict:
    """Run a single multi-turn entry. Returns result dict."""
    scenario_id = entry["id"]
    initial_config = entry.get("initial_config", {})
    involved_classes = entry["involved_classes"]
    ground_truth = gt_entry["ground_truth"]

    messages = []
    all_model_responses = []  # per-turn list of step responses
    total_input_tokens = 0
    total_output_tokens = 0

    queries = []
    for turn_messages in entry.get("question", []):
        for msg in turn_messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                queries.append(msg.get("content", ""))
                break
        else:
            queries.append("")

    for turn_idx, query in enumerate(queries):
        if not query:
            continue

        # Add user message
        messages.append({
            "role": "user",
            "content": [{"type": "text", "text": query}],
        })

        turn_responses = []
        step = 0

        while step < MAX_STEPS_PER_TURN:
            # Call Claude
            response = client.messages.create(
                model=MODEL,
                max_tokens=4096,
                temperature=TEMPERATURE,
                tools=tools,
                messages=messages,
            )

            total_input_tokens += response.usage.input_tokens
            total_output_tokens += response.usage.output_tokens

            # Parse response
            tool_calls = []
            tool_call_ids = []
            text_output = []

            for content in response.content:
                if isinstance(content, ToolUseBlock):
                    tool_calls.append({content.name: json.dumps(content.input)})
                    tool_call_ids.append(content.id)
                elif isinstance(content, TextBlock):
                    text_output.append(content.text)

            # Add assistant message to history (convert to dicts for SDK compat)
            assistant_content = []
            for content in response.content:
                if isinstance(content, TextBlock):
                    assistant_content.append({"type": "text", "text": content.text})
                elif isinstance(content, ToolUseBlock):
                    assistant_content.append({
                        "type": "tool_use",
                        "id": content.id,
                        "name": content.name,
                        "input": content.input,
                    })
            messages.append({
                "role": "assistant",
                "content": assistant_content,
            })

            # If no tool calls, turn is done
            if not tool_calls:
                break

            turn_responses.append(tool_calls)

            # Decode and execute
            try:
                decoded = convert_to_function_call(tool_calls)
            except Exception as e:
                print(f"    Decode error: {e}")
                break

            execution_results, _ = execute_multi_turn_func_call(
                decoded,
                initial_config,
                involved_classes,
                f"clean_llm_{MODEL.replace('-', '_')}",
                scenario_id,
                long_context=False,
                is_evaL_run=False,
            )

            # Feed results back as tool_result
            tool_message = {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "content": str(result),
                        "tool_use_id": tid,
                    }
                    for result, tid in zip(execution_results, tool_call_ids)
                ],
            }
            messages.append(tool_message)

            step += 1

        all_model_responses.append(turn_responses)

    return {
        "scenario_id": scenario_id,
        "all_model_responses": all_model_responses,
        "input_tokens": total_input_tokens,
        "output_tokens": total_output_tokens,
        "ground_truth": ground_truth,
        "initial_config": initial_config,
        "involved_classes": involved_classes,
    }


def evaluate_result(result: dict, entry: dict) -> tuple[bool, list[str]]:
    """Evaluate using gorilla's official state checker."""
    all_responses = result["all_model_responses"]
    ground_truth = result["ground_truth"]

    # Build multi_turn_model_result_list_decoded: list[list[list[str]]]
    # Shape: [turn][step][call_string]
    decoded_per_turn = []
    for turn_responses in all_responses:
        decoded_steps = []
        for step_calls in turn_responses:
            decoded = convert_to_function_call(step_calls)
            decoded_steps.append(decoded)
        decoded_per_turn.append(decoded_steps)

    # Pad with empty turns if model produced fewer turns than ground truth
    while len(decoded_per_turn) < len(ground_truth):
        decoded_per_turn.append([])

    # Run gorilla checker
    model_name = f"clean_llm_{MODEL.replace('-', '_')}"
    check_result = multi_turn_checker(
        multi_turn_model_result_list_decoded=decoded_per_turn,
        multi_turn_ground_truth_list=ground_truth,
        test_entry=entry,
        test_category="multi_turn_base",
        model_name=model_name,
    )

    errors = []
    if not check_result["valid"]:
        errors.append(f"  {check_result.get('error_type', 'unknown')}")
        errors.append(f"  {check_result.get('error_message', '')}")
        # Show what we predicted vs expected for debugging
        for turn_idx, turn_responses in enumerate(decoded_per_turn):
            flat = []
            for step in turn_responses:
                flat.extend(step)
            gt_turn = ground_truth[turn_idx] if turn_idx < len(ground_truth) else []
            if flat != gt_turn:
                errors.append(f"  Turn {turn_idx} Got: {flat}")
                errors.append(f"  Turn {turn_idx} Exp: {gt_turn}")

    return check_result["valid"], errors


def main():
    client = Anthropic()

    # Load data
    bfcl_dir = Path(__file__).parent
    path = bfcl_dir / "data" / "bfcl" / "BFCL_v4_multi_turn_base.json"
    gt_path = bfcl_dir / "data" / "bfcl" / "possible_answer" / "BFCL_v4_multi_turn_base.json"

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

    print(f"Model: {MODEL}")
    print(f"Entries: {MAX_ENTRIES}")
    print(f"Temperature: {TEMPERATURE}")
    print()

    total_scenarios = 0
    passed_scenarios = 0
    total_input_tokens = 0
    total_output_tokens = 0
    t0 = time.time()

    for idx, (entry, gt_entry) in enumerate(zip(entries, gt_entries)):
        scenario_id = entry["id"]
        func_defs = load_func_defs_for_entry(entry)
        tools = convert_to_anthropic_tools(func_defs)

        print(f"--- {scenario_id} ---")
        result = run_entry(client, entry, gt_entry, tools)

        total_input_tokens += result["input_tokens"]
        total_output_tokens += result["output_tokens"]

        scenario_pass, errors = evaluate_result(result, entry)
        total_scenarios += 1

        if scenario_pass:
            passed_scenarios += 1
            print(f"  PASS")
        else:
            print(f"  FAIL")
            for e in errors:
                print(e)

        elapsed = time.time() - t0
        print(f"  [{idx+1}/{MAX_ENTRIES}] {passed_scenarios}/{total_scenarios} scenarios, "
              f"{elapsed:.1f}s, {total_input_tokens}in/{total_output_tokens}out tokens")

    print(f"\n{'='*60}")
    print(f"CLEAN LLM (no Glyphh, no post-processing)")
    print(f"Model: {MODEL}")
    print(f"FINAL: {passed_scenarios}/{total_scenarios} scenarios ({passed_scenarios/total_scenarios*100:.1f}%)")
    print(f"       {total_input_tokens} input + {total_output_tokens} output tokens")
    print(f"       {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()
