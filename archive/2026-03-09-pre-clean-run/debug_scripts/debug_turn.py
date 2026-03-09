"""Run multi_turn_base (20 entries) and report accuracy."""

from __future__ import annotations
import json, re, sys, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from multi_turn_handler import MultiTurnHandler, load_func_defs

_BFCL_DIR = Path(__file__).parent
MAX_ENTRIES = int(sys.argv[1]) if len(sys.argv) > 1 else 20


def _parse_call(call_str):
    m = re.match(r"(\w+)\((.*)\)$", call_str, re.DOTALL)
    if not m: return (call_str, {})
    func_name = m.group(1)
    args_str = m.group(2).strip()
    if not args_str: return (func_name, {})
    try:
        args = eval(f"dict({args_str})")
        return (func_name, args)
    except: pass
    try:
        vals = eval(f"({args_str},)")
        args = {f"_pos_{i}": v for i, v in enumerate(vals)}
        return (func_name, args)
    except:
        return (func_name, {"_raw": args_str})


def _calls_match(pred, gt):
    if len(pred) != len(gt): return False
    for (pf, pa), (gf, ga) in zip(pred, gt):
        if pf != gf: return False
        p_has_pos = any(k.startswith("_pos_") for k in pa)
        g_has_pos = any(k.startswith("_pos_") for k in ga)
        if p_has_pos or g_has_pos:
            p_vals = sorted(str(v) for v in pa.values())
            g_vals = sorted(str(v) for v in ga.values())
            if p_vals != g_vals: return False
        else:
            if pa != ga: return False
    return True


# Load entries
path = _BFCL_DIR / "data" / "bfcl" / "BFCL_v4_multi_turn_base.json"
gt_path = _BFCL_DIR / "data" / "bfcl" / "possible_answer" / "BFCL_v4_multi_turn_base.json"

entries, gt_entries = [], []
with open(path) as f:
    for i, line in enumerate(f):
        if i >= MAX_ENTRIES: break
        entries.append(json.loads(line))
with open(gt_path) as f:
    for i, line in enumerate(f):
        if i >= MAX_ENTRIES: break
        gt_entries.append(json.loads(line))

total_scenarios = 0
passed_scenarios = 0
total_turns = 0
passed_turns = 0
t0 = time.time()

for idx, (entry, gt_entry) in enumerate(zip(entries, gt_entries)):
    scenario_id = entry["id"]
    ground_truth = gt_entry["ground_truth"]

    func_defs = load_func_defs(entry["involved_classes"], entry.get("excluded_function", []))
    handler = MultiTurnHandler()
    handler.setup(entry, func_defs)

    queries = []
    for turn_messages in entry.get("question", []):
        for msg in turn_messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                queries.append(msg.get("content", ""))
                break
        else:
            queries.append("")

    scenario_pass = True
    total_scenarios += 1

    try:
        for turn_idx, query in enumerate(queries):
            if not query:
                continue

            gt_turn = ground_truth[turn_idx] if turn_idx < len(ground_truth) else []
            predicted = handler.process_turn(query)

            parsed_pred = [_parse_call(c) for c in predicted]
            parsed_gt = [_parse_call(c) for c in gt_turn]
            match = _calls_match(parsed_pred, parsed_gt)

            total_turns += 1
            if match:
                passed_turns += 1
            else:
                scenario_pass = False
                print(f"  FAIL {scenario_id} turn {turn_idx}: {query[:60]}")
                print(f"    Exp: {gt_turn}")
                print(f"    Got: {predicted}")
    except Exception as e:
        scenario_pass = False
        print(f"  ERROR {scenario_id}: {e}")
    finally:
        handler.reset()

    if scenario_pass:
        passed_scenarios += 1
        print(f"  PASS {scenario_id}")

    # Progress
    elapsed = time.time() - t0
    print(f"  [{idx+1}/{MAX_ENTRIES}] {passed_scenarios}/{total_scenarios} scenarios, "
          f"{passed_turns}/{total_turns} turns, {elapsed:.0f}s")

print(f"\n{'='*60}")
print(f"FINAL: {passed_scenarios}/{total_scenarios} scenarios ({passed_scenarios/total_scenarios*100:.1f}%)")
print(f"       {passed_turns}/{total_turns} turns ({passed_turns/total_turns*100:.1f}%)")
print(f"       {time.time()-t0:.0f}s total")
