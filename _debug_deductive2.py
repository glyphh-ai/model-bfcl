"""Debug: trace what the DeductiveLayer receives in the eval loop."""
import sys, json
sys.path.insert(0, ".")

from glyphh.state import DeductiveLayer
from state_tracker import ConversationStateTracker
from multi_turn_handler import get_available_functions, extract_turn_query, parse_ground_truth_step

with open("data/bfcl/BFCL_v4_multi_turn_base.json") as f:
    entries = [json.loads(l) for l in f]
with open("data/bfcl/possible_answer/BFCL_v4_multi_turn_base.json") as f:
    gts = [json.loads(l) for l in f]

entry = entries[0]
gt = gts[0]

state_tracker = ConversationStateTracker()
state_tracker.init_from_config(entry.get("initial_config", {}))

deductive = DeductiveLayer(dimension=10000, seed=89)
deductive.add_transition(
    name="location_change",
    directing_actions=["mv", "cp", "mkdir", "touch"],
    operating_actions=["grep", "cat", "sort", "wc", "head", "tail", "diff", "echo", "find", "ls"],
    prerequisite="cd",
)

# Seed initial observation
deductive.observe(state=state_tracker.get_cwd(), actions=[])

print(f"Initial CWD: {state_tracker.get_cwd()}")
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
    print(f"Query: {query[:80]}...")
    print(f"Expected: {sorted(expected_funcs)}")
    print(f"CWD: {state_tracker.get_cwd()}")

    # What would deduce produce?
    result = deductive.deduce(query, current_state=state_tracker.get_cwd())
    print(f"Deduce: {result}")
    print(f"  target_history: {len(deductive._target_history)}, last_target: {deductive._last_target}")
    print(f"  action_history: {deductive._action_history}")

    # Simulate correct prediction
    predicted = {}
    for call in expected_calls:
        predicted.update(call)

    # Simulate what _deductive_observe would do
    targets = []
    cwd = state_tracker.get_cwd()
    for fname, args in predicted.items():
        if not isinstance(args, dict):
            continue
        if fname == "mkdir":
            dir_name = args.get("dir_name", "")
            if dir_name:
                targets.append(cwd + "/" + dir_name)
                print(f"  → mkdir target: {cwd}/{dir_name}")
        elif fname == "mv":
            dest = args.get("destination", "")
            if dest:
                dest_path = cwd + "/" + dest
                if dest_path in state_tracker._all_paths or dest in state_tracker.get_dirs_at_cwd():
                    targets.append(dest_path)
                    print(f"  → mv target: {dest_path}")
                else:
                    print(f"  → mv dest '{dest}' not in known dirs: {state_tracker.get_dirs_at_cwd()}")
        elif fname == "cd":
            folder = args.get("folder", "")
            if folder:
                state_tracker.update_from_prediction({"cd": args})
                print(f"  → cd to {folder}, new CWD: {state_tracker.get_cwd()}")

    # Update deductive with the targets
    actions = list(expected_funcs)
    deductive.observe(
        state=state_tracker.get_cwd(),
        actions=actions,
        targets=targets if targets else None,
    )
    print(f"  After observe: target_history={len(deductive._target_history)}, last_target={deductive._last_target}")

    # Update state tracker for remaining ops
    for fname, args in predicted.items():
        if fname != "cd" and isinstance(args, dict):
            state_tracker.update_from_prediction({fname: args})

    print()
