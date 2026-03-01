"""Debug: trace all HDC signals for cd-missing failure turns."""
import json, re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from handler import GlyphhBFCLHandler
from glyphh.intent import IntentExtractor
from glyphh.state import DeductiveLayer, InductiveLayer
from state_tracker import ConversationStateTracker
from multi_turn_handler import (
    extract_turn_query, parse_ground_truth_step,
    get_available_functions, _extract_fs_hints, _get_fs_extractor,
)

DATA_DIR = Path("data/bfcl")
entries = [json.loads(l) for l in open(DATA_DIR / "BFCL_v4_multi_turn_base.json")]
answers = {}
for l in open(DATA_DIR / "possible_answer/BFCL_v4_multi_turn_base.json"):
    ae = json.loads(l)
    answers[ae["id"]] = ae

handler = GlyphhBFCLHandler()
fs_extractor = _get_fs_extractor()

# Focus on entries with cd-missing failures
for entry_idx in [0, 1, 2, 9, 10]:
    entry = entries[entry_idx]
    eid = entry["id"]
    gt = answers.get(eid, {}).get("ground_truth", [])
    turns = entry.get("question", [])
    available = get_available_functions(entry)
    func_names = [f["name"] for f in available]

    print(f"\n{'='*70}")
    print(f"Entry {entry_idx} ({eid}) — functions: {func_names}")
    print(f"{'='*70}")

    # Set up layers
    deductive = DeductiveLayer(dimension=10000, seed=89, packs=["filesystem"])
    inductive = InductiveLayer(dimension=10000, seed=97, packs=["filesystem"])
    state_tracker = ConversationStateTracker()
    state_tracker.init_from_config(entry.get("initial_config", {}))

    deductive.observe(state=state_tracker.get_cwd(), actions=[])

    for t_idx in range(len(turns)):
        query = extract_turn_query(turns[t_idx])
        if not query:
            continue

        step = gt[t_idx] if t_idx < len(gt) else []
        calls = parse_ground_truth_step(step)
        expected_funcs = set()
        for c in calls:
            expected_funcs.update(c.keys())

        has_cd = "cd" in expected_funcs

        # 1. Model A scores
        result = handler.route(query, available)
        all_scores = result.get("all_scores", [])
        cd_score = next((s["score"] for s in all_scores if s["function"] == "cd"), None)
        top3 = [(s["function"], round(s["score"], 4)) for s in all_scores[:3]]

        # 2. DeductiveLayer
        cwd = state_tracker.get_cwd()
        deduction = deductive.deduce(query=query, current_state=cwd)

        # 3. InductiveLayer
        query_tokens = " ".join(re.sub(r"[^a-z0-9\s]", "", query.lower()).split()[:20])
        ind_result = inductive.predict(features={"query_tokens": query_tokens})

        # 4. IntentExtractor (cd hint)
        cd_hint, touch_hint, search_hint = _extract_fs_hints(query)

        marker = " *** CD NEEDED ***" if has_cd else ""
        print(f"\n  Turn {t_idx}: expected={sorted(expected_funcs)}{marker}")
        print(f"    Query: \"{query[:100]}\"")
        print(f"    CWD: {cwd}")
        print(f"    Model A — cd_score={cd_score}, top3={top3}")
        print(f"    Deductive — prereqs={deduction['prerequisites']}, conf={deduction['confidence']}, mismatch={deduction['mismatch_score']}")
        print(f"    Inductive — label={ind_result['label']}, conf={ind_result['confidence']}, scores={ind_result['scores']}")
        print(f"    IntentExt — cd_hint={cd_hint}")

        # Update deductive with expected actions (simulate ground truth path)
        # Extract target dirs from ground truth for deductive observe
        targets = []
        for c in calls:
            for fname, args in c.items():
                if fname == "cd" and isinstance(args, dict):
                    path = args.get("dir_name", args.get("folder", ""))
                    if path:
                        state_tracker.update_from_prediction({fname: args})
                elif fname in ("mkdir", "mv", "cp", "touch") and isinstance(args, dict):
                    for v in args.values():
                        if isinstance(v, str) and "/" in v:
                            targets.append(v)
                    try:
                        state_tracker.update_from_prediction({fname: args})
                    except Exception:
                        pass

        deductive.observe(
            state=state_tracker.get_cwd(),
            actions=list(expected_funcs),
            targets=targets if targets else None,
        )
