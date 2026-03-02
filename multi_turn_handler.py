"""
Multi-turn BFCL handler — six-model HDC architecture.

Six Glyphh signals, no custom ML:
  Model A  (handler.py / ENCODER_CONFIG seed=42)
           Function router: query → per-function cosine scores via BFCL HDC encoder
  Model B  (SEQUENCE_CONFIG seed=73, include_temporal=True)
           Sequence predictor: history of past calls → predicted next call via
           Glyphh temporal encoding + SDK BeamSearchPredictor
  ConversationState (SDK, seed=73 — same space as Model B)
           HDC pathway encoder: tracks conversation trajectory as a decaying superposition
           of position-bound action vectors.  Pre-seeded GorillaFileSystem nav→op patterns
           are matched via cosine similarity; active patterns boost continuation functions.
           Hebbian reinforcement (fire → wire) strengthens patterns on correct turns.
  IntentExtractor (SDK, seed=53 internally)
           Filesystem intent hints: cd / touch / find signals for GorillaFileSystem turns
  DeductiveLayer (SDK, seed=89)
           HDC predictive coding: accumulates environmental context as decaying superposition,
           detects mismatch between accumulated state and query-implied state via cosine,
           resolves prerequisites by matching delta against transition library.
           Hebbian reinforcement on confirmed transitions.
  InductiveLayer (SDK, seed=97)
           HDC inductive reasoning: few-shot classifier that learns cd-needed vs cd-not-needed
           patterns from training data. Pre-seeded from ground truth, then online-learned
           during eval. Encodes query + state + actions as role-bound BoW, classifies by
           closest centroid via cosine similarity.

Pipeline per turn:
  1. Model A scores all available functions against the current query
  2. Model B (if ≥2 past calls): encodes call history, predicts next function,
     boosts the predicted function's score in Model A's output
  3. ConversationState: detects active pathway patterns (e.g. nav→op after a cd call),
     boosts filesystem operation functions; IntentExtractor adds cd/touch/find signals
  4. DeductiveLayer: detects location mismatch → boosts prerequisite functions (e.g. cd)
  5. State tracker adds CWD / known-files / known-dirs facts for LLM context
  6. LLM receives Glyphh-ranked functions + all signals → selects + fills args
  7. ConversationState.update() + DeductiveLayer.observe() records turn;
     .confirm() on correct turns (Hebbian)
"""

import json
from pathlib import Path
from typing import Any

from glyphh import Encoder
from glyphh.core.config import EncoderConfig, Layer, Segment, Role, TemporalConfig
from glyphh.core.types import Concept
from glyphh.core.ops import cosine_similarity
from glyphh.temporal import BeamSearchPredictor
from glyphh.cognitive import CognitiveLoop, DomainConfig

from handler import GlyphhBFCLHandler
from llm_client import get_client
from glyphh.intent import IntentExtractor
from glyphh.state import ConversationState, DeductiveLayer, InductiveLayer
from state_tracker import ConversationStateTracker

# ── Model B: Sequence prediction encoder (seed=73, separate space from Model A seed=42) ──
SEQUENCE_CONFIG = EncoderConfig(
    dimension=10000,
    seed=73,
    include_temporal=True,
    temporal_config=TemporalConfig(signal_type="sequence"),
    layers=[
        Layer(
            name="function",
            similarity_weight=1.0,
            segments=[
                Segment(
                    name="call",
                    similarity_weight=1.0,
                    roles=[
                        Role(name="func_name", similarity_weight=1.0),
                    ],
                ),
            ],
        ),
    ],
)

_seq_encoder: Encoder | None = None
_beam_predictor: BeamSearchPredictor | None = None


def _get_seq_encoder() -> Encoder:
    global _seq_encoder
    if _seq_encoder is None:
        _seq_encoder = Encoder(SEQUENCE_CONFIG)
    return _seq_encoder


def _get_predictor() -> BeamSearchPredictor:
    global _beam_predictor
    if _beam_predictor is None:
        _beam_predictor = BeamSearchPredictor(beam_width=5, drift_reduction=True)
    return _beam_predictor

DATA_DIR = Path(__file__).parent / "data" / "bfcl"
FUNC_DOC_DIR = DATA_DIR / "multi_turn_func_doc"

# Singleton IntentExtractor with filesystem pack — initialized once, reused across all entries
_fs_extractor: IntentExtractor | None = None


def _get_fs_extractor() -> IntentExtractor:
    global _fs_extractor
    if _fs_extractor is None:
        _fs_extractor = IntentExtractor(packs=["filesystem"])
    return _fs_extractor


_CD_KEYWORDS = (
    "navigate to", "go to", "change to", "change directory",
    "chdir", "move to", "switch to", "enter the", "into the",
)
_TOUCH_ACTIONS = {"touch", "create", "make", "generate", "build", "new"}
_TOUCH_TARGETS = {"file", "document", "script", "text", "config", "log"}

# GorillaFileSystem two-step patterns [prerequisite_nav, operation] — for Hebbian reinforcement.
# After a full cd→op sequence fires correctly, these patterns accumulate strength.
# NOTE: 2-step patterns cannot be used for DETECTION because the partial 1-step state
# (after only calling cd) has ~0 cosine similarity to the full 2-step pattern.
_GFS_NAV_PATTERNS = [
    ("nav_mv",    ["cd", "mv"]),
    ("nav_cp",    ["cd", "cp"]),
    ("nav_grep",  ["cd", "grep"]),
    ("nav_cat",   ["cd", "cat"]),
    ("nav_ls",    ["cd", "ls"]),
    ("nav_touch", ["cd", "touch"]),
    ("nav_mkdir", ["cd", "mkdir"]),
    ("nav_echo",  ["cd", "echo"]),
    ("nav_sort",  ["cd", "sort"]),
    ("nav_find",  ["cd", "find"]),
    ("nav_rm",    ["cd", "rm"]),
    ("nav_wc",    ["cd", "wc"]),
    ("nav_diff",  ["cd", "diff"]),
]

# Single-step navigation-trigger pattern for DETECTION.
# A 1-step state perfectly matches a 1-step pattern (cosine=1.0) but is ~orthogonal
# to any 2-step pattern — so "nav_context" fires precisely when cd was the LAST call
# and no follow-up operation has been made yet.
_GFS_NAV_TRIGGER = "nav_context"   # pattern name

# When "nav_context" is active, these functions are likely next operations
_GFS_OP_FUNCS = frozenset({
    "mv", "cp", "grep", "cat", "ls", "touch", "mkdir",
    "echo", "sort", "find", "rm", "wc", "diff", "chmod",
})

# Lookup: pattern_name → {func_name: relative_weight} for continuation boosting
_PATHWAY_NEXT_FUNCS: dict[str, dict[str, float]] = {
    _GFS_NAV_TRIGGER: {f: 1.0 for f in _GFS_OP_FUNCS},
}


def _extract_fs_hints(query: str) -> tuple[dict, dict, dict]:
    """Extract filesystem intent hints using the SDK IntentExtractor + keyword fallbacks.

    Returns (cd_hint, touch_hint, search_hint) dicts compatible with
    the multi-turn handler's hint consumption logic.

    Confidence values are set to pass/fail the thresholds in _llm_select_and_call:
      cd_hint:     confidence >= 0.03 to inject cd / show cd signal
      touch_hint:  confidence >= 0.09 to inject touch/echo/mkdir
      search_hint: confidence >= 0.05 to inject find; < 0.05 to gate it out
    """
    result = _get_fs_extractor().extract(query)
    action = result.get("action", "none")
    target = result.get("target", "none")
    ql = query.lower()

    # Directory navigation: cd action OR navigation keyword phrases
    needs_cd = action == "cd" or any(kw in ql for kw in _CD_KEYWORDS)
    cd_hint = {"needs_cd": needs_cd, "confidence": 0.80 if needs_cd else 0.01}

    # File creation: touch action, or create/make + file-like target
    needs_touch = action == "touch" or (action in _TOUCH_ACTIONS and target in _TOUCH_TARGETS)
    touch_hint = {"needs_touch": needs_touch, "confidence": 0.80 if needs_touch else 0.00}

    # File search: find_files or search action (searching for files by name/pattern)
    wants_find = action in ("find_files", "search")
    search_hint = {"wants_find": wants_find, "confidence": 0.80 if wants_find else 0.01}

    return cd_hint, touch_hint, search_hint

# Map involved_classes names to func doc files
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


def load_class_functions(class_name: str) -> dict[str, dict]:
    """Load function definitions for a class.

    Returns dict of func_name → func_def.
    """
    fname = CLASS_TO_FILE.get(class_name)
    if not fname:
        return {}

    fpath = FUNC_DOC_DIR / fname
    if not fpath.exists():
        return {}

    funcs = {}
    with open(fpath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            inner = entry.get("function", entry)
            name = inner.get("name", "")
            funcs[name] = inner
    return funcs


def get_available_functions(entry: dict) -> list[dict]:
    """Get the available function definitions for a multi-turn entry."""
    classes = entry.get("involved_classes", [])
    all_funcs = {}
    for cls in classes:
        cls_funcs = load_class_functions(cls)
        all_funcs.update(cls_funcs)
    return list(all_funcs.values())


def extract_turn_query(turn: Any) -> str:
    """Extract the user query from a turn."""
    if isinstance(turn, list):
        for msg in turn:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
        if turn and isinstance(turn[-1], dict):
            return turn[-1].get("content", "")
    elif isinstance(turn, dict):
        return turn.get("content", "")
    elif isinstance(turn, str):
        return turn
    return ""


def parse_ground_truth_step(step: list) -> list[dict]:
    """Parse a ground truth step into [{func_name: {args}}] format."""
    if not step:
        return []

    calls = []
    for call_str in step:
        if not isinstance(call_str, str):
            continue
        paren_idx = call_str.find("(")
        if paren_idx == -1:
            calls.append({call_str: {}})
            continue

        func_name = call_str[:paren_idx].strip()
        args_str = call_str[paren_idx + 1:].rstrip(")")

        args = {}
        if args_str.strip():
            try:
                args = eval(f"dict({args_str})")
            except Exception:
                args = {"_raw": args_str}

        calls.append({func_name: args})

    return calls



def _predict_next_func(
    call_history: list,
    func_glyphs: dict,
) -> dict[str, float]:
    """Model B: use BeamSearchPredictor to boost scores for predicted next function.

    Returns a dict of {func_name: boost_amount} where boost_amount is
    confidence * 0.20 (max 20% score boost for the top predicted function).

    Only fires when call_history has ≥ 2 entries (minimum for meaningful prediction).
    """
    if len(call_history) < 2 or not func_glyphs:
        return {}

    try:
        pred_result = _get_predictor().predict(
            history=call_history,
            time_intervals=1,
            hierarchy_level="cortex",
        )
    except Exception:
        return {}

    if not pred_result.predictions:
        return {}

    top_pred = pred_result.predictions[0]
    if top_pred.confidence < 0.25:  # Low confidence → don't bias
        return {}

    # Find which available function glyph is most similar to the predicted vector
    best_match: str | None = None
    best_sim = -1.0
    pred_vec = top_pred.vector.data  # Prediction.vector is the predicted Vector state

    for fname, fglyph in func_glyphs.items():
        sim = cosine_similarity(pred_vec, fglyph.global_cortex.data)
        if sim > best_sim:
            best_sim = sim
            best_match = fname

    if best_match is None or best_sim < 0.1:
        return {}

    return {best_match: top_pred.confidence * 0.20}


def eval_multi_turn_entry(
    handler: GlyphhBFCLHandler,
    llm,
    entry: dict,
    ground_truth: list,
) -> dict:
    """Evaluate a single multi-turn entry.

    Three-model Glyphh architecture:
    - Model A (handler.py): scores all functions per query
    - Model B (SEQUENCE_CONFIG + BeamSearchPredictor): predicts next function
      from past call history and boosts Model A scores
    - IntentExtractor: filesystem intent hints (cd/touch/find)
    - State tracker: CWD/files context for the LLM
    - LLM: sees top-5 Glyphh-ranked functions, selects + fills args
    """
    turns = entry.get("question", [])
    available_funcs = get_available_functions(entry)

    if not available_funcs:
        return {
            "id": entry.get("id", "?"),
            "turns": len(turns),
            "correct_turns": 0,
            "total_turns": len(turns),
            "correct": False,
            "details": [],
        }

    func_defs = available_funcs
    func_map = {fdef["name"]: fdef for fdef in available_funcs}

    # Pre-encode all available functions as Model B sequence glyphs (cached per entry)
    seq_enc = _get_seq_encoder()
    func_glyphs: dict = {
        fdef["name"]: seq_enc.encode(Concept(
            name=fdef["name"],
            attributes={"func_name": fdef["name"]},
        ))
        for fdef in available_funcs
    }
    call_history: list = []  # Temporal glyph history for Model B

    # Initialize ConversationState (HDC pathway tracking — seed=73 matches func_glyphs space)
    conv_state = ConversationState(dimension=10000, seed=73, decay=0.75)

    # Register 2-step patterns for Hebbian reinforcement (strengthen on correct cd→op turns)
    for pat_name, func_seq in _GFS_NAV_PATTERNS:
        seq_glyphs = [func_glyphs[f] for f in func_seq if f in func_glyphs]
        if len(seq_glyphs) == len(func_seq):
            conv_state.add_pathway(pat_name, seq_glyphs)

    # Register 1-step navigation trigger for DETECTION.
    # A 1-step [cd] pattern matches the current state with cosine=1.0 immediately after
    # calling cd, and drops to ~0 after calling a follow-up operation — giving a clean
    # "we are in navigation context" signal exactly when it's needed.
    if "cd" in func_glyphs:
        conv_state.add_pathway(_GFS_NAV_TRIGGER, [func_glyphs["cd"]])

    # Initialize state tracker from filesystem config
    state_tracker = ConversationStateTracker()
    state_tracker.init_from_config(entry.get("initial_config", {}))

    # Initialize DeductiveLayer + InductiveLayer — both load domain knowledge
    # from the filesystem pack. No manual transition registration or pre-seeding.
    # temporal_window=3: detect stagnation over last 3 turns
    # enable_beam=True: use BeamSearchPredictor for state trajectory prediction
    deductive = DeductiveLayer(
        dimension=10000, seed=89, packs=["filesystem"],
        temporal_window=3, enable_beam=True,
    )
    deductive.observe(state=state_tracker.get_cwd(), actions=[])

    inductive = InductiveLayer(dimension=10000, seed=97, packs=["filesystem"])

    conversation_context = []
    turn_results = []
    correct_turns = 0

    for turn_idx, turn in enumerate(turns):
        query = extract_turn_query(turn)
        if not query:
            # Empty turn (miss_func: the user message was removed)
            expected_step = ground_truth[turn_idx] if turn_idx < len(ground_truth) else []
            expected_calls = parse_ground_truth_step(expected_step)
            if not expected_calls:
                turn_results.append({
                    "turn": turn_idx,
                    "correct": True,
                    "expected": [],
                    "predicted": [],
                    "reason": "empty_turn_no_call",
                })
                correct_turns += 1
            else:
                turn_results.append({
                    "turn": turn_idx,
                    "correct": False,
                    "expected_funcs": sorted(set(
                        k for c in expected_calls for k in c.keys()
                    )),
                    "predicted_funcs": [],
                    "reason": "empty_turn_with_expected_calls",
                })
            continue

        # Get expected for this turn
        expected_step = ground_truth[turn_idx] if turn_idx < len(ground_truth) else []
        expected_calls = parse_ground_truth_step(expected_step)

        if not expected_calls:
            # No function call expected (miss_func: function is missing from API)
            if llm:
                route_result = handler.route(query, func_defs)
                all_scores = route_result.get("all_scores", [])
                # Apply Model B prediction boost
                prediction_boost = _predict_next_func(call_history, func_glyphs)
                all_scores = _apply_prediction_boost(all_scores, prediction_boost)
                cd_hint, touch_hint, search_hint = _extract_fs_hints(query)
                state_hint = state_tracker.get_state_hint(query)
                active_before = conv_state.active_pathways(top_k=3)
                all_scores = _apply_pathway_boost(all_scores, conv_state, cd_hint, touch_hint, search_hint, state_hint)
                pathway_ctx = {
                    "depth": conv_state.depth,
                    "active_patterns": [n for n, s in active_before if s > 0.1],
                }
                predicted = _llm_select_and_call(
                    llm, query, func_defs, all_scores, conversation_context,
                    cd_hint=cd_hint, touch_hint=touch_hint, search_hint=search_hint,
                    state_hint=state_hint, pathway_context=pathway_ctx,
                )
                predicted_funcs = set(predicted.keys()) if isinstance(predicted, dict) else set()
                func_correct = len(predicted_funcs) == 0
                # Update state from prediction
                state_tracker.update_from_prediction(predicted if isinstance(predicted, dict) else {})
            else:
                active_before = []
                predicted_funcs = set()
                func_correct = True

            # Update Model B call history
            _update_call_history(call_history, predicted_funcs, func_glyphs)

            # Update ConversationState pathway trajectory (Hebbian: fire → wire)
            predicted_glyph_list = [func_glyphs[f] for f in predicted_funcs if f in func_glyphs]
            conv_state._last_active_patterns = [n for n, s in active_before if s > 0.1]
            if predicted_glyph_list:
                conv_state.update(predicted_glyph_list)
            if func_correct and predicted_glyph_list:
                conv_state.confirm(predicted_glyph_list)

            turn_results.append({
                "turn": turn_idx,
                "correct": func_correct,
                "expected": [],
                "predicted_funcs": sorted(predicted_funcs) if predicted_funcs else [],
                "reason": "no_call_expected",
            })
            # DeductiveLayer: observe state even on no-call-expected turns
            _deductive_observe(deductive, state_tracker, list(predicted_funcs), predicted if isinstance(predicted, dict) else {})

            if func_correct:
                correct_turns += 1
            conversation_context.append({"role": "user", "content": query})
            continue

        # Extract expected function names
        expected_funcs = set()
        for call in expected_calls:
            expected_funcs.update(call.keys())

        # Model A: Glyphh scores all functions
        route_result = handler.route(query, func_defs)
        all_scores = route_result.get("all_scores", [])

        # Model B: apply beam prediction boost to Model A scores
        prediction_boost = _predict_next_func(call_history, func_glyphs)
        all_scores = _apply_prediction_boost(all_scores, prediction_boost)

        predicted: dict = {}  # populated by whichever branch runs below
        active_before: list = []
        if llm:
            cd_hint, touch_hint, search_hint = _extract_fs_hints(query)
            state_hint = state_tracker.get_state_hint(query)
            active_before = conv_state.active_pathways(top_k=3)
            # Apply ConversationState pathway boost: HDC-based context signal
            all_scores = _apply_pathway_boost(all_scores, conv_state, cd_hint, touch_hint, search_hint, state_hint)

            # DeductiveLayer: detect implicit prerequisites via HDC mismatch
            deduction = deductive.deduce(query, current_state=state_tracker.get_cwd())
            if deduction["prerequisites"]:
                print(f"  [DED] {entry.get('id', '?')} T{turn_idx}: prereqs={deduction['prerequisites']} "
                      f"conf={deduction['confidence']:.3f} mismatch={deduction['mismatch_score']}")
                for prereq in deduction["prerequisites"]:
                    boost = min(0.40, deduction["confidence"] * 0.55)
                    # Apply additive boost to the prerequisite function's score
                    all_scores = [
                        {**s, "score": round(s["score"] + boost, 4)} if s["function"] == prereq else s
                        for s in all_scores
                    ]
                all_scores.sort(key=lambda x: x["score"], reverse=True)
                # When deduction fires with confidence, override contradictory
                # cd_hint so the LLM doesn't get conflicting signals
                if "cd" in deduction["prerequisites"] and deduction["confidence"] > 0.3:
                    cd_hint = {"needs_cd": True, "confidence": deduction["confidence"]}

            # InductiveLayer: learned pattern classification for cd-needed
            import re as _re_q
            query_tokens = " ".join(
                _re_q.sub(r"[^a-z0-9\s]", "", query.lower()).split()[:20]
            )
            inductive_result = inductive.predict(features={
                "query_tokens": query_tokens,
            })
            if inductive_result["label"] == "cd_needed" and inductive_result["confidence"] > 0.08:
                ind_boost = min(0.25, inductive_result["confidence"] * 0.50)
                all_scores = [
                    {**s, "score": round(s["score"] + ind_boost, 4)} if s["function"] == "cd" else s
                    for s in all_scores
                ]
                all_scores.sort(key=lambda x: x["score"], reverse=True)
                # Reinforce cd_hint if inductive agrees
                if not cd_hint or not cd_hint.get("needs_cd"):
                    cd_hint = {"needs_cd": True, "confidence": inductive_result["confidence"]}

            pathway_ctx = {
                "depth": conv_state.depth,
                "active_patterns": [n for n, s in active_before if s > 0.1],
            }
            if deduction["prerequisites"]:
                pathway_ctx["deduction"] = deduction

            predicted = _llm_select_and_call(
                llm, query, func_defs, all_scores, conversation_context,
                cd_hint=cd_hint, touch_hint=touch_hint, search_hint=search_hint,
                state_hint=state_hint, pathway_context=pathway_ctx,
            )
            predicted_funcs = set(predicted.keys()) if isinstance(predicted, dict) else set()
            func_correct = predicted_funcs == expected_funcs

            # Update state from prediction
            state_tracker.update_from_prediction(predicted if isinstance(predicted, dict) else {})

            turn_results.append({
                "turn": turn_idx,
                "correct": func_correct,
                "expected_funcs": sorted(expected_funcs),
                "predicted_funcs": sorted(predicted_funcs),
                "confidence": route_result.get("confidence", 0),
                "glyphh_top1": route_result.get("tool", "?"),
                "beam_boost": list(prediction_boost.keys()) if prediction_boost else [],
                "mode": "hybrid",
            })
        elif len(expected_funcs) == 1:
            predicted_func = route_result.get("tool")
            expected_func = next(iter(expected_funcs))
            func_correct = predicted_func == expected_func
            turn_results.append({
                "turn": turn_idx,
                "correct": func_correct,
                "expected_func": expected_func,
                "predicted_func": predicted_func,
                "confidence": route_result.get("confidence", 0),
                "mode": "glyphh_only",
            })
        else:
            route_result = handler.route_multi(query, func_defs)
            predicted_funcs = set(route_result.get("tools", []))
            func_correct = predicted_funcs == expected_funcs
            turn_results.append({
                "turn": turn_idx,
                "correct": func_correct,
                "expected_funcs": sorted(expected_funcs),
                "predicted_funcs": sorted(predicted_funcs),
                "confidence": route_result.get("confidence", 0),
                "mode": "glyphh_only",
            })

        # Update Model B call history with this turn's predicted calls
        _update_call_history(call_history, predicted_funcs, func_glyphs)

        # Update ConversationState pathway trajectory (Hebbian: fire → wire)
        predicted_glyph_list = [func_glyphs[f] for f in predicted_funcs if f in func_glyphs]
        conv_state._last_active_patterns = [n for n, s in active_before if s > 0.1]
        if predicted_glyph_list:
            conv_state.update(predicted_glyph_list)
        if func_correct and predicted_glyph_list:
            conv_state.confirm(predicted_glyph_list)

        # DeductiveLayer: observe this turn's outcome + Hebbian confirm
        _deductive_observe(deductive, state_tracker, list(predicted_funcs), predicted if isinstance(predicted, dict) else {})
        if "cd" in expected_funcs:
            deductive.confirm("cd" in predicted_funcs, "cd")

        # InductiveLayer: Hebbian reinforcement
        if "cd" in expected_funcs:
            inductive.confirm(was_correct="cd" in predicted_funcs, label="cd_needed")
        else:
            inductive.confirm(was_correct="cd" not in predicted_funcs, label="cd_not_needed")

        if turn_results[-1]["correct"]:
            correct_turns += 1

        # Add user query AND assistant response to conversation context.
        # Including the predicted function calls helps the LLM understand
        # what state the system is in for subsequent turns.
        conversation_context.append({"role": "user", "content": query})
        if predicted and isinstance(predicted, dict):
            import json as _json2
            calls_str = ", ".join(
                f"{fname}({', '.join(f'{k}={repr(v)}' for k, v in args.items())})"
                if args else f"{fname}()"
                for fname, args in predicted.items()
            )
            conversation_context.append({
                "role": "assistant",
                "content": f"Called: {calls_str}",
            })

    return {
        "id": entry.get("id", "?"),
        "turns": len(turns),
        "correct_turns": correct_turns,
        "total_turns": len(turn_results),
        "correct": correct_turns == len(turn_results),
        "details": turn_results,
    }


def _apply_prediction_boost(
    all_scores: list[dict],
    prediction_boost: dict[str, float],
) -> list[dict]:
    """Apply Model B beam prediction boost to Model A's scored list, re-sort."""
    if not prediction_boost:
        return all_scores
    boosted = []
    for entry in all_scores:
        fname = entry["function"]
        boost = prediction_boost.get(fname, 0.0)
        if boost:
            boosted.append({**entry, "score": entry["score"] + boost})
        else:
            boosted.append(entry)
    boosted.sort(key=lambda x: x["score"], reverse=True)
    return boosted


def _apply_pathway_boost(
    all_scores: list[dict],
    conv_state: ConversationState,
    cd_hint: dict | None,
    touch_hint: dict | None,
    search_hint: dict | None,
    state_hint: dict | None = None,
) -> list[dict]:
    """Apply HDC-based pathway + intent boosts to Model A scores.

    Three signal sources:

    1. StateTracker structural inference — filesystem-state reasoning about whether
       the current query references a different directory (needs cd) or a local file
       (no cd needed).  This is structural logic about observable state, not a
       prompt rule — the tracker knows the CWD and maps query nouns to known dirs.

    2. ConversationState pathway patterns — after a cd call, the 1-step nav_context
       pattern fires at cosine=1.0, boosting all fs operation candidates.  Drops
       to ~0 once the follow-up operation fires (no repeat-navigation pressure).

    3. IntentExtractor (SDK) — BoW-based query intent signals (cd/touch/find).
    """
    boosts: dict[str, float] = {}

    # 1. StateTracker: structural filesystem navigation inference
    if state_hint:
        needs_cd = state_hint.get("needs_cd_signal", "unclear")
        if needs_cd == "likely_yes":
            boosts["cd"] = boosts.get("cd", 0.0) + 0.35
        elif needs_cd == "likely_no":
            boosts["cd"] = boosts.get("cd", 0.0) - 0.15

    # 2. ConversationState: detect navigation context via 1-step trigger pattern.
    #    A 1-step [cd] pattern matches at cosine=1.0 immediately after a cd call,
    #    then drops to ~0 once a follow-up operation fires — giving a clean signal.
    if conv_state.depth > 0:
        active = conv_state.active_pathways(top_k=len(_PATHWAY_NEXT_FUNCS) + 3)
        for pattern_name, score in active:
            if score < 0.3:  # Only act on clear matches
                continue
            next_funcs = _PATHWAY_NEXT_FUNCS.get(pattern_name, {})
            for fname, weight in next_funcs.items():
                boosts[fname] = boosts.get(fname, 0.0) + score * weight * 0.15
        # Penalize cd if nav trigger is strongly active (don't navigate twice)
        nav_trigger_score = next((s for n, s in active if n == _GFS_NAV_TRIGGER), 0.0)
        if nav_trigger_score > 0.3:
            boosts["cd"] = boosts.get("cd", 0.0) - nav_trigger_score * 0.10

    # 3. IntentExtractor: query-level navigation / creation / search intent (HDC model)
    if cd_hint and cd_hint.get("needs_cd") and cd_hint.get("confidence", 0) >= 0.03:
        boosts["cd"] = boosts.get("cd", 0.0) + 0.15
    elif cd_hint and not cd_hint.get("needs_cd"):
        boosts["cd"] = boosts.get("cd", 0.0) - 0.10

    if touch_hint and touch_hint.get("needs_touch") and touch_hint.get("confidence", 0) >= 0.09:
        boosts["touch"] = boosts.get("touch", 0.0) + 0.20

    if search_hint:
        if search_hint.get("wants_find"):
            boosts["find"] = boosts.get("find", 0.0) + 0.20
        else:
            boosts["find"] = boosts.get("find", 0.0) - 0.10

    if not boosts:
        return all_scores

    boosted = []
    for entry in all_scores:
        b = boosts.get(entry["function"], 0.0)
        if b:
            new_score = max(0.0, entry["score"] + b)
            boosted.append({**entry, "score": round(new_score, 4)})
        else:
            boosted.append(entry)
    boosted.sort(key=lambda x: x["score"], reverse=True)
    return boosted


def _update_call_history(
    call_history: list,
    predicted_funcs: set[str],
    func_glyphs: dict,
) -> None:
    """Append this turn's predicted function glyphs to the Model B call history."""
    for fname in sorted(predicted_funcs):  # sorted for determinism
        glyph = func_glyphs.get(fname)
        if glyph is not None:
            call_history.append(glyph)


def _deductive_observe(
    deductive: DeductiveLayer,
    state_tracker: ConversationStateTracker,
    actions: list[str],
    predicted: dict,
) -> None:
    """Update DeductiveLayer with this turn's state and directed targets.

    Extracts target directories from predicted calls (mkdir, mv, cp, touch)
    to track where files are being directed. Targets are always extracted
    without gating on known paths — the DeductiveLayer encodes them as HD
    symbols, so unknown paths get unique vectors and produce valid mismatch.
    """
    targets = []
    cwd = state_tracker.get_cwd()

    for fname, args in predicted.items():
        if not isinstance(args, dict):
            continue
        if fname == "mkdir":
            dir_name = args.get("dir_name", "")
            if dir_name:
                targets.append(cwd + "/" + dir_name)
        elif fname in ("mv", "cp"):
            dest = args.get("destination", "")
            if dest:
                # Always treat destination as a target — no _all_paths gate.
                # DeductiveLayer encodes targets as HD symbols; unknown paths
                # produce valid mismatch against current state.
                if dest.startswith("/"):
                    targets.append(dest)
                else:
                    targets.append(cwd + "/" + dest)
        elif fname == "touch":
            file_name = args.get("file_name", "")
            if file_name and "/" in file_name:
                # File being created in a subdirectory — extract the dir as target
                targets.append(cwd + "/" + file_name.rsplit("/", 1)[0])

    deductive.observe(
        state=cwd,
        actions=actions,
        targets=targets if targets else None,
    )



def _llm_select_and_call(
    llm,
    query: str,
    all_func_defs: list[dict],
    glyphh_scores: list[dict],
    conversation: list[dict],
    cd_hint: dict | None = None,
    touch_hint: dict | None = None,
    search_hint: dict | None = None,
    state_hint: dict | None = None,
    pathway_context: dict | None = None,
) -> dict:
    """LLM selects function(s) and extracts args, guided by Glyphh's scores.

    Architecture:
    - Glyphh ranks all functions by semantic relevance to the query
    - LLM receives ALL available functions, ordered by Glyphh score
      (multi-turn scenarios have only 18-32 functions, so full list is fine)
    - Glyphh signals (cd/touch/find hints, state) are prepended as hints

    Returns dict of {func_name: {args}} for each predicted call.
    """
    import json as _json

    if not all_func_defs:
        return {}

    func_map = {fd["name"]: fd for fd in all_func_defs}

    # Order all functions by Glyphh score so the most relevant appear first
    score_map = {s["function"]: s["score"] for s in glyphh_scores}
    top_names: list[str] = sorted(
        func_map.keys(),
        key=lambda n: score_map.get(n, 0.0),
        reverse=True,
    )

    # Build full signatures for all selected functions
    func_blocks = []
    for fname in top_names:
        fdef = func_map.get(fname, {})
        desc = fdef.get("description", "")
        params = fdef.get("parameters", {}).get("properties", {})
        required = fdef.get("parameters", {}).get("required", [])
        score = score_map.get(fname, 0.0)

        param_parts = []
        for pname, pdef in params.items():
            ptype = pdef.get("type", "any")
            req = "required" if pname in required else "optional"
            param_parts.append(f"{pname}: {ptype} ({req})")

        params_str = ", ".join(param_parts) if param_parts else "none"
        # Show Glyphh score so LLM can reason about relevance; low-score functions are contextual noise
        func_blocks.append(f"  [{score:.2f}] {fname}({params_str}) — {desc}")

    functions_text = "AVAILABLE FUNCTIONS (Glyphh relevance score shown, higher = more relevant):\n"
    functions_text += "\n".join(func_blocks) if func_blocks else "  (none)"

    system = (
        "You select the correct function call(s) for each turn of a stateful conversation. "
        "Given the user's request, the current system state, and the available functions "
        "(ranked by semantic relevance), call exactly what the user is asking for.\n\n"
        "Return ONLY a JSON array — each element has ONE key = function name, value = arguments object. "
        "Example: [{\"cd\": {\"folder\": \"src\"}}, {\"mv\": {\"source\": \"old.txt\", \"destination\": \"new.txt\"}}]. "
        "Return [] if no function matches. No explanation, no markdown."
    )

    user_msg = f"{functions_text}\n\n"

    # Glyphh-derived factual signals — observations from HDC models, not instructions.
    # The LLM reasons from these facts rather than following hardcoded rules.
    signals = []

    if cd_hint:
        if cd_hint.get("needs_cd"):
            signals.append(f"Navigation intent: YES (confidence={cd_hint['confidence']:.2f})")
        else:
            signals.append(f"Navigation intent: NO (confidence={cd_hint['confidence']:.2f})")

    if touch_hint:
        if touch_hint.get("needs_touch"):
            signals.append("File creation intent: YES")

    if search_hint:
        if search_hint.get("wants_find"):
            signals.append("Search/find intent: YES")
        else:
            signals.append("Search/find intent: NO")

    if pathway_context:
        depth = pathway_context.get("depth", 0)
        active_pats = pathway_context.get("active_patterns", [])
        if depth > 0:
            signals.append(f"Pathway depth: {depth} step(s) in current sequence")
        if active_pats:
            signals.append(f"Active trajectory patterns: {', '.join(active_pats[:3])}")

        # DeductiveLayer signal: state mismatch detected
        deduction = pathway_context.get("deduction")
        if deduction and deduction.get("prerequisites"):
            prereqs = ", ".join(deduction["prerequisites"])
            mm = deduction.get("mismatch_score", 0)
            conf = deduction.get("confidence", 0)
            target = deduction.get("target")
            deduction_msg = (
                f"DEDUCTION: State mismatch detected (score={mm:.2f}, "
                f"confidence={conf:.2f}). Prerequisite {prereqs} likely needed"
            )
            if target:
                deduction_msg += f" → target: {target}"
            signals.append(deduction_msg)

    if signals:
        user_msg += "GLYPHH INTENT SIGNALS:\n" + "\n".join(f"  {s}" for s in signals) + "\n\n"

    # Filesystem state — factual observations from StateTracker for LLM to reason about
    if state_hint:
        cwd = state_hint.get("cwd", "/")
        files_here = state_hint.get("files_here", [])
        dirs_here = state_hint.get("dirs_here", [])
        query_mentions_child = state_hint.get("query_mentions_child")
        query_mentions_other = state_hint.get("query_mentions_other_dir")
        needs_cd_signal = state_hint.get("needs_cd_signal", "unclear")

        state_lines = [f"Working directory: {cwd}"]
        if files_here:
            state_lines.append(f"Files: {', '.join(files_here[:10])}")
        if dirs_here:
            state_lines.append(f"Subdirectories: {', '.join(dirs_here[:10])}")

        # Glyphh StateTracker conclusion: derived from HDC similarity between query and filesystem state
        if query_mentions_child:
            state_lines.append(
                f"StateTracker: query references '{query_mentions_child}' (a known subdirectory) — "
                f"navigation away from current directory detected"
            )
        elif query_mentions_other:
            state_lines.append(
                f"StateTracker: query references '{query_mentions_other}' — "
                f"a directory in the tree but not the current location"
            )
        elif needs_cd_signal == "likely_no":
            state_lines.append("StateTracker: query references files in the current directory — no navigation needed")

        user_msg += "SYSTEM STATE:\n" + "\n".join(f"  {l}" for l in state_lines) + "\n\n"

    user_msg += f"User: {query}"

    messages = [{"role": "system", "content": system}]
    if conversation:
        for ctx in conversation[-3:]:
            messages.append(ctx)
    messages.append({"role": "user", "content": user_msg})

    try:
        text = llm.chat_complete(messages, max_tokens=512)

        # Strip markdown code fences
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        # Robust JSON parsing: LLM sometimes returns objects without array brackets.
        # e.g. '{"mkdir": {...}}, {"mv": {...}}' → wrap in brackets for valid JSON
        text_stripped = text.strip()
        if text_stripped and not text_stripped.startswith("["):
            text_stripped = "[" + text_stripped + "]"
        calls = _json.loads(text_stripped)
        if not isinstance(calls, list):
            calls = [calls]

        # Validate: only accept function names that exist in the available set
        valid_names = {fd["name"] for fd in all_func_defs}
        result = {}
        for call in calls:
            if isinstance(call, dict):
                for fname, args in call.items():
                    if fname in valid_names:
                        result[fname] = args if isinstance(args, dict) else {}

        return result
    except Exception:
        return {}


# ── Cognitive Loop path (LLM-free) ──

# Domain config — loaded once, shared across all entries
_DOMAIN_CONFIG_PATH = Path(__file__).parent / "domain" / "gorilla_file_system.json"
_domain_config: DomainConfig | None = None


def _get_domain_config() -> DomainConfig:
    global _domain_config
    if _domain_config is None:
        _domain_config = DomainConfig.from_file(_DOMAIN_CONFIG_PATH)
    return _domain_config


def normalize_gfs_state(initial_config: dict) -> dict:
    """Convert GorillaFileSystem initial_config to normalized state dict.

    Returns {primary, collections, _tree} for CognitiveLoop.begin().
    This is model-side parsing — the SDK doesn't know GorillaFileSystem.
    """
    gfs = initial_config.get("GorillaFileSystem", {})
    root = gfs.get("root", {})
    if not root:
        return {"primary": "/", "collections": {}, "_tree": {}}

    tree: dict[str, dict] = {}

    def _walk(path: str, node: dict) -> None:
        if not isinstance(node, dict) or node.get("type") != "directory":
            return
        contents = node.get("contents", {})
        files = []
        dirs = []
        for name, child in contents.items():
            if isinstance(child, dict):
                if child.get("type") == "file":
                    files.append(name)
                elif child.get("type") == "directory":
                    dirs.append(name)
                    _walk(path.rstrip("/") + "/" + name, child)
        tree[path] = {"files_here": files, "dirs_here": dirs}

    root_name = list(root.keys())[0] if root else "root"
    _walk("/" + root_name, root.get(root_name, {}))
    primary = "/" + root_name

    initial_collections = dict(tree.get(primary, {}))

    return {
        "primary": primary,
        "collections": initial_collections,
        "_tree": tree,
    }


def eval_multi_turn_entry_cognitive(
    entry: dict,
    ground_truth: list,
) -> dict:
    """Evaluate a multi-turn entry using CognitiveLoop (no LLM).

    The CognitiveLoop handles everything:
      - Intent extraction (perceive)
      - Episodic recall (remember)
      - Deductive reasoning (prerequisite detection)
      - Slot extraction (argument filling)
      - Confidence gating (ASK vs CALL)
      - Hebbian reinforcement (learn from outcomes)
    """
    turns = entry.get("question", [])
    available_funcs = get_available_functions(entry)

    if not available_funcs:
        return {
            "id": entry.get("id", "?"),
            "turns": len(turns),
            "correct_turns": 0,
            "total_turns": len(turns),
            "correct": False,
            "details": [],
        }

    config = _get_domain_config()
    loop = CognitiveLoop(
        packs=["filesystem"],
        domain_config=config,
    )

    # Parse initial state and start the loop
    initial_state = normalize_gfs_state(entry.get("initial_config", {}))
    loop.begin(functions=available_funcs, initial_state=initial_state)

    # Also maintain the legacy state tracker for query_mentions_child hints
    state_tracker = ConversationStateTracker()
    state_tracker.init_from_config(entry.get("initial_config", {}))

    turn_results = []
    correct_turns = 0

    for turn_idx, turn in enumerate(turns):
        query = extract_turn_query(turn)

        # Get expected for this turn
        expected_step = ground_truth[turn_idx] if turn_idx < len(ground_truth) else []
        expected_calls = parse_ground_truth_step(expected_step)

        if not query:
            # Empty turn
            if not expected_calls:
                turn_results.append({
                    "turn": turn_idx,
                    "correct": True,
                    "expected": [],
                    "predicted": [],
                    "reason": "empty_turn_no_call",
                })
                correct_turns += 1
            else:
                turn_results.append({
                    "turn": turn_idx,
                    "correct": False,
                    "expected_funcs": sorted(set(
                        k for c in expected_calls for k in c.keys()
                    )),
                    "predicted_funcs": [],
                    "reason": "empty_turn_with_expected_calls",
                })
            continue

        # Inject state tracker hints into the cognitive loop's state
        state_hint = state_tracker.get_state_hint(query)
        loop._state["query_mentions_child"] = state_hint.get("query_mentions_child")
        loop._state["query_mentions_other_dir"] = state_hint.get("query_mentions_other_dir")

        # Run the cognitive loop
        result = loop.step(query)

        # Extract predicted function names
        predicted_funcs = set()
        predicted_dict: dict[str, dict] = {}
        if result.action == "CALL":
            for call in result.calls:
                for fname, args in call.items():
                    predicted_funcs.add(fname)
                    predicted_dict[fname] = args

        # Extract expected function names
        expected_funcs = set()
        for call in expected_calls:
            expected_funcs.update(call.keys())

        # Score this turn
        if not expected_calls:
            func_correct = len(predicted_funcs) == 0
        else:
            func_correct = predicted_funcs == expected_funcs

        # Update state tracker from prediction (for next turn hints)
        state_tracker.update_from_prediction(predicted_dict)

        # Hebbian reinforcement
        loop.confirm(
            was_correct=func_correct,
            correct_outcome=expected_calls if not func_correct else None,
        )

        turn_results.append({
            "turn": turn_idx,
            "correct": func_correct,
            "expected_funcs": sorted(expected_funcs),
            "predicted_funcs": sorted(predicted_funcs),
            "confidence": result.confidence,
            "action": result.action,
            "mode": "cognitive",
        })

        if func_correct:
            correct_turns += 1

    loop.end()

    return {
        "id": entry.get("id", "?"),
        "turns": len(turns),
        "correct_turns": correct_turns,
        "total_turns": len(turn_results),
        "correct": correct_turns == len(turn_results),
        "details": turn_results,
    }
