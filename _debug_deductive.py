"""Debug: trace what the DeductiveLayer produces for cd-missing turns."""
import sys, json
sys.path.insert(0, ".")

from deductive_layer import DeductiveLayer
from state_tracker import ConversationStateTracker
from multi_turn_handler import get_available_functions, extract_turn_query, parse_ground_truth_step

with open("data/bfcl/BFCL_v4_multi_turn_base.json") as f:
    entries = [json.loads(l) for l in f]
with open("data/bfcl/possible_answer/BFCL_v4_multi_turn_base.json") as f:
    gts = [json.loads(l) for l in f]

# Test entry 0: turn 1 expects [cd, grep] but gets [grep]
entry = entries[0]
gt = gts[0]

state_tracker = ConversationStateTracker()
state_tracker.init_from_config(entry.get("initial_config", {}))

deductive = DeductiveLayer()
# Initial observation
deductive.observe(
    location=state_tracker.get_cwd(),
    entities=state_tracker.get_files_at_cwd()[:5],
    actions=[],
)

print(f"Initial CWD: {state_tracker.get_cwd()}")
print(f"Initial files: {state_tracker.get_files_at_cwd()}")
print(f"Initial dirs: {state_tracker.get_dirs_at_cwd()}")
print()

turns = entry.get("question", [])
ground_truth = gt.get("ground_truth", gt) if isinstance(gt, dict) else gt

for turn_idx in range(min(4, len(turns))):
    query = extract_turn_query(turns[turn_idx])
    expected_step = ground_truth[turn_idx] if turn_idx < len(ground_truth) else []
    expected_calls = parse_ground_truth_step(expected_step)
    expected_funcs = set()
    for call in expected_calls:
        expected_funcs.update(call.keys())

    print(f"=== Turn {turn_idx} ===")
    print(f"Query: {query}")
    print(f"Expected: {sorted(expected_funcs)}")
    print(f"CWD: {state_tracker.get_cwd()}")

    # Get state hint
    state_hint = state_tracker.get_state_hint(query)
    print(f"StateTracker needs_cd: {state_hint['needs_cd_signal']}")
    print(f"  mentions_child: {state_hint.get('query_mentions_child')}")
    print(f"  mentions_other: {state_hint.get('query_mentions_other_dir')}")

    # Deductive layer
    env_ctx = deductive.get_env_context()
    print(f"DeductiveLayer has {len(deductive._observations)} observations")

    deduction = deductive.deduce(query, state_hint)
    print(f"Deduction: {deduction}")

    # Simulate correct prediction (use expected funcs)
    for f in sorted(expected_funcs):
        if f == "cd":
            # Simulate cd updating state
            child = state_hint.get("query_mentions_child")
            if child:
                state_tracker.update_from_prediction({"cd": {"folder": child}})
            else:
                # Check if there's a dir in the expected args
                for call in expected_calls:
                    if "cd" in call:
                        folder = call["cd"].get("folder", "")
                        if folder:
                            state_tracker.update_from_prediction({"cd": {"folder": folder}})
                            break

    # Update deductive layer with correct actions
    deductive.observe(
        location=state_tracker.get_cwd(),
        entities=state_tracker.get_files_at_cwd()[:5],
        actions=list(expected_funcs),
    )

    if "cd" in expected_funcs:
        deductive.confirm("cd" in deduction.get("prerequisites", []), "cd")
        print(f"  Hebbian confirm: {'cd' in deduction.get('prerequisites', [])}")

    print()
