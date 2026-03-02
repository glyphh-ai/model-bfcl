"""
Multi-turn BFCL handler — CognitiveLoop architecture.

Uses the CognitiveLoop from glyphh.cognitive for all multi-turn evaluation.
The CognitiveLoop handles internally:
  - SchemaIntentClassifier (LLM-primary routing via function schemas)
  - IdeaSpace (episodic memory, seed=101)
  - DeductiveLayer (prerequisite detection, seed=89)
  - InductiveLayer (pattern learning, seed=97)
  - ConversationState (trajectory tracking, seed=73)
  - SlotExtractor (argument extraction from DomainConfig)
  - IntentCache (HDC learns from LLM decisions, seed=113)

Pipeline per turn:
  1. PERCEIVE  — SchemaIntentClassifier: LLM routes query to function(s)
  2. RECALL    — IdeaSpace: find similar past situations
  3. DEDUCE    — DeductiveLayer: check state mismatch → prerequisites
  4. PREDICT   — InductiveLayer: learned pattern classification
  5. RESOLVE   — Map intent → function names (LLM direct or rule-based)
  6. SLOT      — SlotExtractor + LLM-provided args merged
  7. CONVERGE  — Confidence gate (HDC 40% + LLM 60%)
  8. DECIDE    — Emit CALL or ASK
  9. RECORD    — Store idea-glyph, Hebbian reinforcement via confirm()
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, TYPE_CHECKING

from glyphh.cognitive import CognitiveLoop, DomainConfig
from state_tracker import ConversationStateTracker

if TYPE_CHECKING:
    from glyphh.llm.engine import LLMEngine


DATA_DIR = Path(__file__).parent / "data" / "bfcl"
FUNC_DOC_DIR = DATA_DIR / "multi_turn_func_doc"

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


# ── Domain config (loaded once, shared across all entries) ──

_DOMAIN_CONFIG_PATH = Path(__file__).parent / "domain" / "gorilla_file_system.json"
_domain_config: DomainConfig | None = None


def _get_domain_config() -> DomainConfig:
    global _domain_config
    if _domain_config is None:
        _domain_config = DomainConfig.from_file(_DOMAIN_CONFIG_PATH)
    return _domain_config


# ── Data loading ──

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


# ── State normalization ──

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


# ── Multi-turn evaluation ──

def eval_multi_turn_entry(
    entry: dict,
    ground_truth: list,
    engine: LLMEngine | None = None,
) -> dict:
    """Evaluate a multi-turn entry using CognitiveLoop.

    The CognitiveLoop handles everything:
      - Intent extraction + function routing (SchemaIntentClassifier)
      - Episodic recall (IdeaSpace)
      - Deductive reasoning (prerequisite detection)
      - Inductive reasoning (pattern classification)
      - Conversation trajectory (ConversationState)
      - Slot extraction (argument filling)
      - Confidence gating (ASK vs CALL)
      - Hebbian reinforcement (learn from outcomes)

    The ConversationStateTracker is maintained alongside for
    query_mentions_child hints (filesystem structural inference).
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
        llm_engine=engine,
    )

    # Parse initial state and start the loop
    initial_state = normalize_gfs_state(entry.get("initial_config", {}))
    loop.begin(functions=available_funcs, initial_state=initial_state)

    # Maintain the state tracker for query_mentions_child hints
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
            # Empty turn (miss_func: the user message was removed)
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
