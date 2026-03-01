"""Debug: trace cd-missing failures to see what signals the LLM gets."""
import sys, json
sys.path.insert(0, ".")

from handler import GlyphhBFCLHandler
from llm_client import get_client
from multi_turn_handler import (
    get_available_functions, extract_turn_query, parse_ground_truth_step,
    _extract_fs_hints, _apply_pathway_boost, _predict_next_func,
    _GFS_NAV_PATTERNS, _GFS_NAV_TRIGGER,
)
from state_tracker import ConversationStateTracker
from glyphh import Encoder
from glyphh.core.types import Concept
from glyphh.state import ConversationState
from multi_turn_handler import SEQUENCE_CONFIG

handler = GlyphhBFCLHandler(threshold=0.15)

with open("data/bfcl/BFCL_v4_multi_turn_base.json") as f:
    entries = [json.loads(l) for l in f]
with open("data/bfcl/possible_answer/BFCL_v4_multi_turn_base.json") as f:
    gts = [json.loads(l) for l in f]

# Pick entry 0 — first failure is turn 1: expected=['cd','grep'] got=['grep']
entry = entries[0]
gt = gts[0]
funcs = get_available_functions(entry)
func_map = {fd["name"]: fd for fd in funcs}

seq_enc = Encoder(SEQUENCE_CONFIG)
func_glyphs = {
    fdef["name"]: seq_enc.encode(Concept(name=fdef["name"], attributes={"func_name": fdef["name"]}))
    for fdef in funcs
}

conv_state = ConversationState(dimension=10000, seed=73, decay=0.75)
for pat_name, func_seq in _GFS_NAV_PATTERNS:
    seq_glyphs = [func_glyphs[f] for f in func_seq if f in func_glyphs]
    if len(seq_glyphs) == len(func_seq):
        conv_state.add_pathway(pat_name, seq_glyphs)
if "cd" in func_glyphs:
    conv_state.add_pathway(_GFS_NAV_TRIGGER, [func_glyphs["cd"]])

state_tracker = ConversationStateTracker()
state_tracker.init_from_config(entry.get("initial_config", {}))

print(f"Entry 0: {len(entry['question'])} turns")
print(f"Initial CWD: {state_tracker.get_cwd()}")
print(f"Files: {state_tracker.get_files_at_cwd()}")
print(f"Dirs: {state_tracker.get_dirs_at_cwd()}")
print()

# Simulate turns 0 and 1
turns = entry.get("question", [])
ground_truth = gt.get("ground_truth", gt) if isinstance(gt, dict) else gt

for turn_idx in range(min(2, len(turns))):
    query = extract_turn_query(turns[turn_idx])
    expected_step = ground_truth[turn_idx] if turn_idx < len(ground_truth) else []
    expected_calls = parse_ground_truth_step(expected_step)
    expected_funcs = set()
    for call in expected_calls:
        expected_funcs.update(call.keys())

    print(f"=== Turn {turn_idx} ===")
    print(f"Query: {query}")
    print(f"Expected: {sorted(expected_funcs)}")

    # Model A scores
    route = handler.route(query, funcs)
    scores = route.get("all_scores", [])
    score_map = {s["function"]: s["score"] for s in scores}
    print(f"Model A top 5: {[(s['function'], round(s['score'],3)) for s in scores[:5]]}")
    print(f"  cd score: {score_map.get('cd', 0):.3f}")

    # Model B prediction
    prediction_boost = _predict_next_func([], func_glyphs)

    # Intent hints
    cd_hint, touch_hint, search_hint = _extract_fs_hints(query)
    print(f"IntentExtractor cd_hint: {cd_hint}")

    # State hint
    state_hint = state_tracker.get_state_hint(query)
    print(f"StateTracker: cwd={state_hint['cwd']}, needs_cd={state_hint['needs_cd_signal']}")
    print(f"  mentions_child={state_hint.get('query_mentions_child')}, mentions_other={state_hint.get('query_mentions_other_dir')}")

    # Pathway state
    active = conv_state.active_pathways(top_k=5)
    print(f"ConversationState depth={conv_state.depth}, active={[(n, round(s,3)) for n,s in active if s > 0.1]}")

    # Apply boosts
    boosted = _apply_pathway_boost(scores, conv_state, cd_hint, touch_hint, search_hint, state_hint)
    boosted_map = {s["function"]: s["score"] for s in boosted}
    print(f"After boost top 5: {[(s['function'], round(s['score'],3)) for s in boosted[:5]]}")
    print(f"  cd score after boost: {boosted_map.get('cd', 0):.3f}")

    # Simulate correct prediction for next turn
    for f in sorted(expected_funcs):
        if f in func_glyphs:
            conv_state.update([func_glyphs[f]])
    state_tracker.update_from_prediction({f: {} for f in expected_funcs})
    print()
