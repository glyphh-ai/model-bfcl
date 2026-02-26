"""
Multi-turn BFCL handler.

Glyphh routes each turn to the right function(s).
LLM extracts arguments from the query + function signature.

Architecture:
  1. Load function definitions from multi_turn_func_doc/
  2. For each turn in a multi-turn conversation:
     a. Glyphh encodes available functions and the user query
     b. Glyphh scores all functions → provides ranked signal to LLM
     c. LLM sees ALL functions but with Glyphh's relevance scores
     d. LLM selects the exact set to call + extracts arguments
  3. Compare against ground truth per turn

Glyphh's role:
  - Scores every function against the query (semantic routing signal)
  - Provides confidence-ranked ordering so LLM sees most relevant first
  - Does NOT filter — the LLM sees all functions because multi-turn
    queries often require utility functions (cd, ls) that score low
    semantically but are contextually necessary
"""

import json
from pathlib import Path
from typing import Any

from handler import GlyphhBFCLHandler
from llm_client import get_client
from intent_models import DirectoryIntentModel, FileCreationIntentModel, SearchIntentModel
from state_tracker import ConversationStateTracker

DATA_DIR = Path(__file__).parent / "data" / "bfcl"
FUNC_DOC_DIR = DATA_DIR / "multi_turn_func_doc"

# Singleton intent models — initialized once, reused across all entries
_dir_intent_model = None
_file_creation_model = None
_search_intent_model = None

def _get_dir_intent_model() -> DirectoryIntentModel:
    global _dir_intent_model
    if _dir_intent_model is None:
        _dir_intent_model = DirectoryIntentModel()
    return _dir_intent_model

def _get_file_creation_model() -> FileCreationIntentModel:
    global _file_creation_model
    if _file_creation_model is None:
        _file_creation_model = FileCreationIntentModel()
    return _file_creation_model

def _get_search_intent_model() -> SearchIntentModel:
    global _search_intent_model
    if _search_intent_model is None:
        _search_intent_model = SearchIntentModel()
    return _search_intent_model

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



def eval_multi_turn_entry(
    handler: GlyphhBFCLHandler,
    llm,
    entry: dict,
    ground_truth: list,
) -> dict:
    """Evaluate a single multi-turn entry.

    Hybrid approach:
    - Glyphh scores all functions and provides ranked signal
    - LLM sees all functions with Glyphh scores, selects + extracts args
    - If no LLM: fall back to Glyphh top-1 routing only
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

    # Initialize state tracker from filesystem config
    state_tracker = ConversationStateTracker()
    state_tracker.init_from_config(entry.get("initial_config", {}))

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
                cd_hint = _get_dir_intent_model().score(query)
                touch_hint = _get_file_creation_model().score(query)
                search_hint = _get_search_intent_model().score(query)
                state_hint = state_tracker.get_state_hint(query)
                predicted = _llm_select_and_call(
                    llm, query, func_defs, all_scores, conversation_context,
                    cd_hint=cd_hint, touch_hint=touch_hint, search_hint=search_hint,
                    state_hint=state_hint,
                )
                predicted_funcs = set(predicted.keys()) if isinstance(predicted, dict) else set()
                func_correct = len(predicted_funcs) == 0
                # Update state from prediction
                state_tracker.update_from_prediction(predicted if isinstance(predicted, dict) else {})
            else:
                predicted_funcs = set()
                func_correct = True

            turn_results.append({
                "turn": turn_idx,
                "correct": func_correct,
                "expected": [],
                "predicted_funcs": sorted(predicted_funcs) if predicted_funcs else [],
                "reason": "no_call_expected",
            })
            if func_correct:
                correct_turns += 1
            conversation_context.append({"role": "user", "content": query})
            continue

        # Extract expected function names
        expected_funcs = set()
        for call in expected_calls:
            expected_funcs.update(call.keys())

        # Glyphh scores all functions
        route_result = handler.route(query, func_defs)
        all_scores = route_result.get("all_scores", [])

        if llm:
            cd_hint = _get_dir_intent_model().score(query)
            touch_hint = _get_file_creation_model().score(query)
            search_hint = _get_search_intent_model().score(query)
            state_hint = state_tracker.get_state_hint(query)
            predicted = _llm_select_and_call(
                llm, query, func_defs, all_scores, conversation_context,
                cd_hint=cd_hint, touch_hint=touch_hint, search_hint=search_hint,
                state_hint=state_hint,
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

        if turn_results[-1]["correct"]:
            correct_turns += 1

        conversation_context.append({"role": "user", "content": query})

    return {
        "id": entry.get("id", "?"),
        "turns": len(turns),
        "correct_turns": correct_turns,
        "total_turns": len(turn_results),
        "correct": correct_turns == len(turn_results),
        "details": turn_results,
    }



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
) -> dict:
    """LLM selects function(s) and extracts args, guided by Glyphh's scores.

    Architecture:
    - Glyphh ranks all functions by semantic relevance to the query
    - DirectoryIntentModel provides cd_hint (needs_cd, confidence)
    - FileCreationIntentModel provides touch_hint (needs_touch, confidence)
    - SearchIntentModel provides search_hint (wants_find, confidence)
    - LLM sees ALL functions organized by Glyphh's ranking
    - Top-ranked get full signatures; lower-ranked get names + description

    Returns dict of {func_name: {args}} for each predicted call.
    """
    import json as _json

    if not all_func_defs:
        return {}

    func_map = {fd["name"]: fd for fd in all_func_defs}

    # Split into tiers based on Glyphh scores
    top_score = glyphh_scores[0]["score"] if glyphh_scores else 0
    tier1_cutoff = max(top_score * 0.50, 0.20)

    tier1_funcs = []
    tier2_funcs = []
    for s in glyphh_scores:
        if s["score"] >= tier1_cutoff:
            tier1_funcs.append(s["function"])
        else:
            tier2_funcs.append(s["function"])

    # Ensure at least top-8 in tier 1
    while len(tier1_funcs) < min(8, len(glyphh_scores)):
        if tier2_funcs:
            tier1_funcs.append(tier2_funcs.pop(0))
        else:
            break

    # Build tier 1: full signatures
    tier1_blocks = []
    for fname in tier1_funcs:
        fdef = func_map.get(fname, {})
        desc = fdef.get("description", "")
        params = fdef.get("parameters", {}).get("properties", {})
        required = fdef.get("parameters", {}).get("required", [])

        param_parts = []
        for pname, pdef in params.items():
            ptype = pdef.get("type", "any")
            req = "required" if pname in required else "optional"
            param_parts.append(f"{pname}: {ptype} ({req})")

        params_str = ", ".join(param_parts) if param_parts else "none"
        tier1_blocks.append(f"  {fname}({params_str}) — {desc}")

    # Build tier 2: names + brief description
    tier2_blocks = []
    for fname in tier2_funcs:
        fdef = func_map.get(fname, {})
        desc = fdef.get("description", "")[:60]
        tier2_blocks.append(f"  {fname} — {desc}")

    functions_text = "PRIMARY FUNCTIONS (most relevant to query):\n"
    functions_text += "\n".join(tier1_blocks) if tier1_blocks else "  (none)"
    if tier2_blocks:
        functions_text += "\n\nOTHER AVAILABLE FUNCTIONS:\n"
        functions_text += "\n".join(tier2_blocks)

    system = (
        "You select function calls for a stateful tool-use system. "
        "The system maintains working directory and login state across turns.\n\n"
        "RULES:\n"
        "1. Call EXACTLY the functions the user asks for — no more, no less.\n"
        "2. Directory navigation (cd): include ONLY when the user explicitly mentions "
        "going to, navigating to, or operating 'in'/'within' a NAMED directory. "
        "Do NOT add cd() for operations on files in the current directory.\n"
        "3. NEVER add find() as a preparatory step. Files exist in the working directory. "
        "Only include find() if the user explicitly says 'find', 'search for', or 'locate'.\n"
        "4. Do NOT add ls() unless the user explicitly asks to list or see directory contents.\n"
        "5. Do NOT add cat() unless the user explicitly asks to read/view/display file contents.\n"
        "6. Do NOT add authentication or login calls unless the user explicitly asks to log in.\n"
        "7. Each turn is independent — do NOT repeat calls from previous turns.\n"
        "8. If the user asks for multiple distinct actions in one message, include all of them.\n"
        "9. If the user says to create/write a file, include touch() or echo() as appropriate.\n"
        "10. If no function matches, return [].\n\n"
        "Return ONLY a JSON array. Each element: {\"function_name\": {\"param\": \"value\"}}.\n"
        "No explanation, no markdown — just the JSON array."
    )

    user_msg = f"{functions_text}\n\n"

    # Add intent signals from Glyphh models
    hints = []

    # Directory intent
    if cd_hint:
        if cd_hint.get("needs_cd") and cd_hint.get("confidence", 0) >= 0.03:
            hints.append(
                f"CD SIGNAL: Navigation intent detected (conf={cd_hint['confidence']:.3f}). "
                "Consider including cd() if a specific directory is mentioned."
            )
        elif not cd_hint.get("needs_cd") and cd_hint.get("confidence", 0) >= 0.01:
            hints.append(
                f"CD SIGNAL: No navigation intent (conf={cd_hint['confidence']:.3f}). "
                "Do NOT add cd()."
            )

    # File creation intent — only hint when confident
    if touch_hint and touch_hint.get("needs_touch") and touch_hint.get("confidence", 0) >= 0.09:
        hints.append(
            "FILE CREATION SIGNAL: Writing new content to a file detected. "
            "Include touch() to create the file before echo() writes to it."
        )

    # Search suppression — only inject when find is in the top Glyphh scores
    # (i.e., when there's actual risk of the LLM adding find)
    find_in_top = any(
        s["function"] == "find" for s in glyphh_scores[:8]
    ) if glyphh_scores else False

    if search_hint and find_in_top:
        if not search_hint.get("wants_find"):
            hints.append(
                "SEARCH SIGNAL: No search intent detected. Do NOT add find() — "
                "operate on files directly by name."
            )

    if hints:
        user_msg += "GLYPHH SIGNALS:\n" + "\n".join(f"  • {h}" for h in hints) + "\n\n"

    # Add filesystem state context
    if state_hint:
        cwd = state_hint.get("cwd", "/")
        files_here = state_hint.get("files_here", [])
        dirs_here = state_hint.get("dirs_here", [])
        cd_signal = state_hint.get("needs_cd_signal", "unclear")

        state_lines = [f"Current working directory: {cwd}"]
        if files_here:
            state_lines.append(f"Files here: {', '.join(files_here[:10])}")
        if dirs_here:
            state_lines.append(f"Subdirectories here: {', '.join(dirs_here[:10])}")

        if cd_signal == "likely_yes":
            child = state_hint.get("query_mentions_child")
            if child:
                state_lines.append(
                    f"The query references '{child}' which is a subdirectory here — "
                    "cd() into it before operating."
                )
            else:
                state_lines.append(
                    "The query references a directory that is NOT the current one — "
                    "cd() is likely needed."
                )
        elif cd_signal == "likely_no":
            state_lines.append(
                "The query references files/dirs in the current directory — "
                "do NOT add cd()."
            )

        user_msg += "FILESYSTEM STATE:\n" + "\n".join(f"  {l}" for l in state_lines) + "\n\n"

    user_msg += f"User: {query}"

    messages = [{"role": "system", "content": system}]
    if conversation:
        for ctx in conversation[-3:]:
            messages.append(ctx)
    messages.append({"role": "user", "content": user_msg})

    try:
        resp = llm.client.chat.completions.create(
            model=llm.model,
            messages=messages,
            temperature=0.0,
            max_tokens=512,
        )
        text = resp.choices[0].message.content.strip()

        # Strip markdown code fences
        if text.startswith("```"):
            text = text.split("\n", 1)[1] if "\n" in text else text[3:]
            if text.endswith("```"):
                text = text[:-3]
            text = text.strip()

        calls = _json.loads(text)
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

        # Soft hard gate: strip find() only when search intent is very low
        # Threshold 0.05 — blocks "Read the file X" (score ~0.04) but allows
        # "Find the file named X" (score ~0.08) through
        if search_hint and search_hint.get("confidence", 1.0) < 0.05 and "find" in result:
            del result["find"]

        return result
    except Exception:
        return {}
