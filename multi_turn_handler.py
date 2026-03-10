"""Glyphh Ada 1.1 — Multi-turn handler.

Architecture:
  HDC (BFCLModelScorer)  -> routes which functions to call (deterministic, top-N)
  CognitiveLoop          -> tracks state (CWD), injects prerequisites (cd before grep)
  LLM (Claude Haiku)     -> picks from top-N candidates + extracts arg values (scoped task)

The HDC acts as a guardrail — the LLM only sees functions the scorer
deems relevant, so it can't hallucinate random function calls.

Usage:
    handler = MultiTurnHandler()
    handler.setup(entry, func_defs)
    for query in turn_queries:
        call_strings, raw_calls = handler.process_turn(query)
        # call_strings: ["cd(folder='document')", "mkdir(dir_name='temp')"]
        # raw_calls:    [{"cd": {"folder": "document"}}, {"mkdir": {"dir_name": "temp"}}]
    handler.reset()
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

from glyphh import CognitiveLoop

sys.path.insert(0, str(Path(__file__).parent))

from scorer import BFCLModelScorer, _CLASS_DIR_MAP
from domain_config import CLASS_DOMAIN_CONFIGS
from llm_extractor import LLMArgumentExtractor


# -- Func def loading (shared with test_multi_turn.py) ---------------------

_FUNC_DOC_DIR = Path(__file__).parent / "data" / "bfcl" / "multi_turn_func_doc"

_CLASS_TO_FILE = {
    "GorillaFileSystem":  "gorilla_file_system.json",
    "MathAPI":            "math_api.json",
    "MessageAPI":         "message_api.json",
    "PostingAPI":         "posting_api.json",
    "TicketAPI":          "ticket_api.json",
    "TradingBot":         "trading_bot.json",
    "TravelAPI":          "travel_booking.json",
    "TravelBookingAPI":   "travel_booking.json",
    "VehicleControlAPI":  "vehicle_control.json",
    "TwitterAPI":         "posting_api.json",
}


def _load_jsonl(path: Path) -> list[dict]:
    results = []
    with open(path) as f:
        for line in f:
            if line.strip():
                results.append(json.loads(line))
    return results


def load_func_defs(involved_classes: list[str], excluded: list[str] | None = None) -> list[dict]:
    """Load function definitions with class-prefixed names."""
    excluded = set(excluded or [])
    func_defs = []
    for cls in involved_classes:
        fname = _CLASS_TO_FILE.get(cls)
        if not fname:
            continue
        fpath = _FUNC_DOC_DIR / fname
        if not fpath.exists():
            continue
        for func in _load_jsonl(fpath):
            if func["name"] not in excluded:
                func = dict(func)
                func["name"] = f"{cls}.{func['name']}"
                func_defs.append(func)
    return func_defs


def _strip_class_prefix(func_name: str) -> str:
    """GorillaFileSystem.cd -> cd"""
    return func_name.split(".")[-1] if "." in func_name else func_name


# -- State normalization ---------------------------------------------------

def _normalize_fs_tree(root: dict, prefix: str = "") -> dict:
    """Convert BFCL initial_config filesystem tree to CognitiveLoop state."""
    tree = {}

    def _walk(node: dict, path: str):
        items = []
        locations = []
        for name, info in node.items():
            if isinstance(info, dict):
                if info.get("type") == "directory":
                    locations.append(name)
                    child_path = f"{path}/{name}" if path else name
                    _walk(info.get("contents", {}), child_path)
                elif info.get("type") == "file":
                    items.append(name)
                else:
                    locations.append(name)
                    child_path = f"{path}/{name}" if path else name
                    _walk(info, child_path)
        tree[path] = {"items_here": items, "locations_here": locations}

    _walk(root, "")
    return tree


def _build_initial_state(initial_config: dict, class_name: str) -> dict | None:
    """Build CognitiveLoop initial state from BFCL entry initial_config."""
    config = initial_config.get(class_name, {})
    if not config:
        return None

    root = config.get("root", {})
    if not root:
        return None

    tree = _normalize_fs_tree(root)
    root_dir_name = list(root.keys())[0] if root else ""
    root_collections = tree.get(root_dir_name, {"items_here": [], "locations_here": []})

    return {
        "primary": root_dir_name,
        "collections": dict(root_collections),
        "_tree": tree,
    }


# -- Handler ---------------------------------------------------------------

class MultiTurnHandler:
    """HDC routing + CognitiveLoop state + LLM arg extraction."""

    def __init__(self):
        self._scorers: dict[str, BFCLModelScorer] = {}
        self._loops: dict[str, CognitiveLoop] = {}
        self._llm = LLMArgumentExtractor()
        self._func_defs: dict[str, dict] = {}  # func_name -> func_def
        self._class_func_defs: dict[str, list[dict]] = {}  # class_name -> [func_defs]
        self._involved_classes: list[str] = []
        self._history: list[str] = []
        self._previous_calls: list[list[dict]] = []
        self._initial_config: dict = {}

    def setup(self, entry: dict, func_defs: list[dict]) -> None:
        """Initialize scorers + CognitiveLoops for an entry."""
        self.reset()

        self._involved_classes = entry.get("involved_classes", [])
        self._initial_config = entry.get("initial_config", {})

        # Index func defs
        for fd in func_defs:
            self._func_defs[fd["name"]] = fd
            class_name = fd["name"].split(".")[0] if "." in fd["name"] else ""
            self._class_func_defs.setdefault(class_name, []).append(fd)

        # Create scorer + CognitiveLoop per involved class
        for class_name in self._involved_classes:
            class_dir = _CLASS_DIR_MAP.get(class_name)
            if not class_dir:
                continue

            scorer = BFCLModelScorer()
            scorer.configure_from_db(class_dir)
            self._scorers[class_name] = scorer

            domain_config = CLASS_DOMAIN_CONFIGS.get(class_name)
            loop = CognitiveLoop(
                domain_config=domain_config,
                model_scorer=scorer,
                confidence_threshold=0.15,
            )

            initial_state = _build_initial_state(self._initial_config, class_name)
            class_funcs = self._class_func_defs.get(class_name, [])
            func_schemas = []
            for fd in class_funcs:
                func_schemas.append({
                    "name": fd["name"],
                    "description": fd.get("description", ""),
                    "parameters": fd.get("parameters", {}),
                })

            loop.begin(functions=func_schemas, initial_state=initial_state)
            self._loops[class_name] = loop

    def add_functions(self, func_defs: list[dict]) -> None:
        """Add held-out functions back (for miss_func holdout turns)."""
        for fd in func_defs:
            self._func_defs[fd["name"]] = fd
            class_name = fd["name"].split(".")[0] if "." in fd["name"] else ""
            self._class_func_defs.setdefault(class_name, []).append(fd)

    def _detect_keyword_functions(self, query: str, class_name: str) -> list[str]:
        """Detect functions implied by keywords in the query using DomainConfig."""
        domain_config = CLASS_DOMAIN_CONFIGS.get(class_name)
        if not domain_config:
            return []

        query_lower = query.lower()
        detected = []
        for func_name, keywords in domain_config.multi_action_keywords.items():
            for kw in keywords:
                if kw.lower() in query_lower:
                    detected.append(func_name)
                    break
        return detected

    def process_turn(self, query: str) -> tuple[list[str], list[dict]]:
        """Process one turn.

        Returns:
            (call_strings, raw_calls) where:
            - call_strings: ["cd(folder='document')", "mkdir(dir_name='temp')"]
            - raw_calls: [{"cd": {"folder": "document"}}, {"mkdir": {"dir_name": "temp"}}]

        Pipeline:
          1. HDC scoring per class -> Level 1 class filtering
          2. HDC top-N + keyword detection -> candidate functions
          3. CognitiveLoop state -> CWD context
          4. LLM picks from candidates + extracts args (route_and_extract_turn)
          5. CognitiveLoop apply_state -> update CWD
          6. Format as call strings
        """
        # Step 1: HDC route per class
        class_scores: dict[str, list[tuple[str, float]]] = {}
        for class_name in self._involved_classes:
            scorer = self._scorers.get(class_name)
            if not scorer:
                continue

            result = scorer.score(query)
            if not result.all_scores:
                continue

            top = [(s["function"], s["score"]) for s in result.all_scores[:5]
                   if s["score"] > 0.10]
            if top:
                class_scores[class_name] = top

        # Level 1: pick relevant class(es)
        class_max = {cn: max(s for _, s in funcs) for cn, funcs in class_scores.items()}
        if not class_max:
            self._history.append(query)
            self._previous_calls.append([])
            return [], []

        best_score = max(class_max.values())
        relevant_classes = [cn for cn, ms in class_max.items()
                           if ms >= best_score - 0.15]

        # Step 2: Build candidate list -- HDC top-N + keyword-detected functions
        candidate_funcs: list[str] = []
        seen: set[str] = set()

        for class_name in relevant_classes:
            kw_funcs = self._detect_keyword_functions(query, class_name)
            for fn in kw_funcs:
                bare = _strip_class_prefix(fn)
                if bare not in seen:
                    seen.add(bare)
                    candidate_funcs.append(fn)

            for fn, sim in class_scores.get(class_name, []):
                bare = _strip_class_prefix(fn)
                if bare not in seen:
                    seen.add(bare)
                    candidate_funcs.append(fn)

        if not candidate_funcs:
            self._history.append(query)
            self._previous_calls.append([])
            return [], []

        # Step 3: Get current state from CognitiveLoop (CWD)
        current_cwd = "/"
        for class_name in list(relevant_classes) + [c for c in self._involved_classes if c not in relevant_classes]:
            loop = self._loops.get(class_name)
            if loop and hasattr(loop, '_state'):
                primary = loop._state.get("primary", "")
                if primary:
                    current_cwd = primary
                    break

        # Step 4: Build func_defs for the LLM (only the candidates)
        candidate_fdefs = []
        for fn in candidate_funcs:
            fd = self._func_defs.get(fn)
            if fd:
                candidate_fdefs.append(fd)

        # Step 5: LLM picks from candidates + extracts args
        if candidate_fdefs and query:
            calls = self._llm.route_and_extract_turn(
                query=query,
                all_func_defs=candidate_fdefs,
                conversation_history=self._history,
                previous_calls=self._previous_calls,
                initial_config=self._initial_config,
                current_cwd=current_cwd,
            )
        else:
            calls = []

        # Step 6: Update CognitiveLoop state from the result
        for class_name in relevant_classes:
            loop = self._loops.get(class_name)
            if loop:
                prefixed_calls = []
                for call in calls:
                    for fn, args in call.items():
                        prefixed_fn = f"{class_name}.{fn}"
                        prefixed_calls.append({prefixed_fn: args if isinstance(args, dict) else {}})
                try:
                    loop.apply_state(prefixed_calls)
                except Exception:
                    pass

        # Step 7: Format as call strings + keep raw calls
        call_strings = []
        for call in calls:
            for fn, args in call.items():
                call_strings.append(_format_call(fn, args if isinstance(args, dict) else {}))

        self._history.append(query)
        self._previous_calls.append(calls)
        return call_strings, calls

    def get_filtered_tools_and_cwd(self, query: str) -> tuple[list[dict], str]:
        """HDC route + keyword detection + CognitiveLoop CWD.

        Returns (filtered_func_defs, current_cwd) for use with external LLM loop.
        Falls back to ALL func_defs if HDC finds nothing.
        """
        class_scores: dict[str, list[tuple[str, float]]] = {}
        for class_name in self._involved_classes:
            scorer = self._scorers.get(class_name)
            if not scorer:
                continue
            result = scorer.score(query)
            if not result.all_scores:
                continue
            top = [(s["function"], s["score"]) for s in result.all_scores[:5]
                   if s["score"] > 0.10]
            if top:
                class_scores[class_name] = top

        # Level 1: pick relevant class(es)
        class_max = {cn: max(s for _, s in funcs) for cn, funcs in class_scores.items()}
        if not class_max:
            return list(self._func_defs.values()), self._get_cwd()

        best_score = max(class_max.values())
        relevant_classes = [cn for cn, ms in class_max.items()
                           if ms >= best_score - 0.15]

        # Build candidate list — HDC top-N + keyword-detected functions
        candidate_funcs: list[str] = []
        seen: set[str] = set()

        for class_name in relevant_classes:
            kw_funcs = self._detect_keyword_functions(query, class_name)
            for fn in kw_funcs:
                bare = _strip_class_prefix(fn)
                if bare not in seen:
                    seen.add(bare)
                    candidate_funcs.append(fn)

            for fn, sim in class_scores.get(class_name, []):
                bare = _strip_class_prefix(fn)
                if bare not in seen:
                    seen.add(bare)
                    candidate_funcs.append(fn)

        # Also include all funcs from relevant classes (HDC may miss some)
        relevant_class_set = set(relevant_classes)
        for fd in self._func_defs.values():
            full_name = fd["name"]
            cls = full_name.split(".")[0] if "." in full_name else ""
            bare = _strip_class_prefix(full_name)
            if cls in relevant_class_set and bare not in seen:
                seen.add(bare)
                candidate_funcs.append(full_name)

        candidate_fdefs = []
        for fn in candidate_funcs:
            fd = self._func_defs.get(fn)
            if fd:
                candidate_fdefs.append(fd)

        return candidate_fdefs if candidate_fdefs else list(self._func_defs.values()), self._get_cwd()

    def _get_cwd(self) -> str:
        """Get current working directory from CognitiveLoop state."""
        for class_name in self._involved_classes:
            loop = self._loops.get(class_name)
            if loop and hasattr(loop, '_state'):
                primary = loop._state.get("primary", "")
                if primary:
                    return primary
        return "/"

    def update_state(self, raw_calls: list[dict]) -> None:
        """Update CognitiveLoop state from executed calls.

        Args:
            raw_calls: [{bare_func_name: {args}}, ...]
        """
        for class_name in self._involved_classes:
            loop = self._loops.get(class_name)
            if not loop:
                continue
            prefixed_calls = []
            for call in raw_calls:
                for fn, args in call.items():
                    prefixed_fn = f"{class_name}.{fn}"
                    prefixed_calls.append({prefixed_fn: args if isinstance(args, dict) else {}})
            try:
                loop.apply_state(prefixed_calls)
            except Exception:
                pass

    def record_turn(self, query: str, raw_calls: list[dict]) -> None:
        """Record turn in history (for conversation context in subsequent turns)."""
        self._history.append(query)
        self._previous_calls.append(raw_calls)

    def reset(self) -> None:
        """Reset between entries."""
        for loop in self._loops.values():
            try:
                loop.end()
            except Exception:
                pass
        self._scorers.clear()
        self._loops.clear()
        self._func_defs.clear()
        self._class_func_defs.clear()
        self._involved_classes.clear()
        self._history.clear()
        self._previous_calls.clear()
        self._initial_config.clear()


def _format_call(func_name: str, args: dict) -> str:
    """Format as func_name(arg1='val1', arg2='val2')."""
    if not args:
        return f"{func_name}()"
    parts = []
    for k, v in args.items():
        if isinstance(v, str):
            parts.append(f"{k}='{v}'")
        elif isinstance(v, bool):
            parts.append(f"{k}={v}")
        else:
            parts.append(f"{k}={v}")
    return f"{func_name}({', '.join(parts)})"
