"""
BFCL handler — HDC routing + argument extraction.

BFCLHandler ties together:
  BFCLScorer        — HDC function routing (which function to call)
  Extractor         — argument extraction (rule-based or LLM-assisted)
  CognitiveLoop     — multi-turn routing with episodic memory + deduction

Exports:
  BFCLHandler.route(query, func_defs)            → single-function routing result
  BFCLHandler.route_multi(query, func_defs)      → multi-function routing result
  BFCLHandler.route_multi_turn(entry, func_defs) → multi-turn routing via CognitiveLoop
  BFCLHandler.is_irrelevant(query, func_defs)    → irrelevance detection
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from glyphh import CognitiveLoop

from scorer import BFCLScorer
from extractor import ArgumentExtractor
from domain_config import CLASS_DOMAIN_CONFIGS
from intent import extract_api_class, extract_pack_actions

# Per-class exemplar directories
_CLASSES_DIR = Path(__file__).parent / "classes"

# Class name → folder name
_CLASS_TO_FOLDER = {
    "GorillaFileSystem": "gorilla_file_system",
    "TwitterAPI":        "twitter_api",
    "MessageAPI":        "message_api",
    "PostingAPI":        "posting_api",
    "TicketAPI":         "ticket_api",
    "MathAPI":           "math_api",
    "TradingBot":        "trading_bot",
    "TravelAPI":         "travel_booking",
    "TravelBookingAPI":  "travel_booking",
    "VehicleControlAPI": "vehicle_control",
}

# Class name → SDK pack names for CognitiveLoop
_CLASS_TO_PACKS: dict[str, list[str]] = {
    "GorillaFileSystem": ["filesystem"],
    "TwitterAPI":        ["social"],
    "MessageAPI":        ["social"],
    "PostingAPI":        ["social"],
    "TicketAPI":         [],
    "MathAPI":           ["math"],
    "TradingBot":        ["trading"],
    "TravelAPI":         ["travel"],
    "TravelBookingAPI":  ["travel"],
    "VehicleControlAPI": ["vehicle"],
}

# Pack canonical action → class-prefixed function name (overrides for non-obvious mappings)
# For GorillaFileSystem, pack canonicals ARE bare function names — auto-derived.
# For other classes, explicit mappings where pack canonical != bare function name.
_PACK_FUNC_OVERRIDES: dict[str, dict[str, str]] = {
    "TwitterAPI": {
        "post":    "TwitterAPI.post_tweet",
        "repost":  "TwitterAPI.retweet",
        "comment": "TwitterAPI.comment",
        "mention": "TwitterAPI.mention",
    },
    "MessageAPI": {
        "dm_social": "MessageAPI.send_message",
    },
    "PostingAPI": {
        "post":    "PostingAPI.post",
        "comment": "PostingAPI.comment",
        "repost":  "PostingAPI.share",
    },
    "MathAPI": {
        "add":        "MathAPI.add",
        "subtract":   "MathAPI.subtract",
        "multiply":   "MathAPI.multiply",
        "divide":     "MathAPI.divide",
        "power":      "MathAPI.power",
        "sqrt":       "MathAPI.square_root",
        "log":        "MathAPI.logarithm",
        "statistics": "MathAPI.mean",
    },
    "TradingBot": {
        "buy":           "TradingBot.buy",
        "sell":          "TradingBot.sell",
        "get_quote":     "TradingBot.get_quote",
        "get_balance":   "TradingBot.get_balance",
        "get_history":   "TradingBot.get_history",
        "place_order":   "TradingBot.place_order",
    },
    "TravelBookingAPI": {
        "book_flight":      "TravelBookingAPI.book_flight",
        "book_hotel":       "TravelBookingAPI.book_hotel",
        "cancel_booking":   "TravelBookingAPI.cancel_booking",
        "check_in":         "TravelBookingAPI.check_in",
        "get_flight_status": "TravelBookingAPI.get_flight_status",
    },
    "TravelAPI": {
        "book_flight":      "TravelAPI.book_flight",
        "book_hotel":       "TravelAPI.book_hotel",
        "cancel_booking":   "TravelAPI.cancel_booking",
        "check_in":         "TravelAPI.check_in",
        "get_flight_status": "TravelAPI.get_flight_status",
    },
    "VehicleControlAPI": {
        "accelerate":     "VehicleControlAPI.accelerate",
        "brake":          "VehicleControlAPI.brake",
        "lock":           "VehicleControlAPI.lock",
        "unlock":         "VehicleControlAPI.unlock",
        "start_engine":   "VehicleControlAPI.start_engine",
        "stop_engine":    "VehicleControlAPI.stop_engine",
        "set_climate":    "VehicleControlAPI.set_climate",
        "set_navigation": "VehicleControlAPI.set_navigation",
    },
}


class BFCLHandler:
    """HDC routing + pluggable argument extraction.

    Usage:
        handler = BFCLHandler()                          # rule-based extraction
        handler = BFCLHandler(extractor=LLMExtractor())  # LLM-assisted extraction
        result  = handler.route(query, func_defs)
        # result["tool"]       — predicted function name (or None if irrelevant)
        # result["args"]       — extracted argument dict
        # result["confidence"] — HDC similarity score
        # result["top_k"]      — top-5 scored functions for debugging
    """

    def __init__(
        self,
        confidence_threshold: float = 0.22,
        irrelevance_threshold: float = 0.60,
        extractor: Any | None = None,
    ) -> None:
        self._scorer    = BFCLScorer()
        self._extractor = extractor or ArgumentExtractor()
        self._threshold = confidence_threshold
        self._irr_threshold = irrelevance_threshold
        self.language   = "python"  # Set per-category by runner

    def _extract_args(self, query: str, func_def: dict) -> dict:
        """Call extractor with language awareness."""
        if hasattr(self._extractor, 'extract') and 'language' in self._extractor.extract.__code__.co_varnames:
            return self._extractor.extract(query, func_def, language=self.language)
        return self._extract_args(query, func_def)

    # ── Public API ───────────────────────────────────────────────────────

    def route(
        self,
        query: str,
        func_defs: list[dict],
        force: bool = False,
    ) -> dict[str, Any]:
        """Route a query to the single best matching function.

        Args:
            query:     Natural language query.
            func_defs: List of available function definitions.
            force:     If True, always return the best match without threshold
                       check. Use for routing categories where a correct function
                       is guaranteed to exist. If False, applies the confidence
                       threshold (use for irrelevance detection categories).

        Returns a result dict with:
          tool         (str | None) — predicted function name, None if irrelevant
          args         (dict)       — extracted argument values
          confidence   (float)      — HDC similarity score [0, 1]
          is_irrelevant (bool)      — True when no function matches
          latency_ms   (float)      — wall-clock time in milliseconds
          top_k        (list)       — top-5 [{function, score}] for inspection
        """
        t0 = time.perf_counter()

        self._scorer.configure(func_defs)
        result = self._scorer.score(query)

        elapsed = (time.perf_counter() - t0) * 1000

        if force:
            # Routing category: always pick best — a correct function is guaranteed
            irrelevant = False
            func_name  = result.functions[0] if result.functions else None
        else:
            irrelevant = result.is_irrelevant or result.confidence < self._threshold
            func_name  = result.functions[0] if result.functions and not irrelevant else None
        args: dict = {}

        if func_name:
            func_def = self._find_def(func_defs, func_name)
            if func_def:
                args = self._extract_args(query, func_def)

        return {
            "tool":          func_name,
            "args":          args,
            "confidence":    result.confidence,
            "is_irrelevant": irrelevant,
            "latency_ms":    elapsed,
            "top_k":         result.all_scores[:5],
        }

    def route_multi(self, query: str, func_defs: list[dict], force: bool = False) -> dict[str, Any]:
        """Route a query to potentially multiple functions.

        Returns a result dict with:
          tools        (list[str])  — selected function names (empty if irrelevant)
          args         (dict)       — {func_name: {param: value, ...}}
          confidence   (float)      — average HDC similarity of selected functions
          is_irrelevant (bool)      — True when no function matches
          latency_ms   (float)      — wall-clock time in milliseconds
          top_k        (list)       — top-5 [{function, score}] for inspection
        """
        t0 = time.perf_counter()

        self._scorer.configure(func_defs)
        result = self._scorer.score_multi(query)

        elapsed = (time.perf_counter() - t0) * 1000

        if not force and result.is_irrelevant:
            return {
                "tools":         [],
                "args":          {},
                "confidence":    result.confidence,
                "is_irrelevant": True,
                "latency_ms":    elapsed,
                "top_k":         result.all_scores[:5],
            }

        all_args: dict[str, dict] = {}
        for fname in result.functions:
            func_def = self._find_def(func_defs, fname)
            if func_def:
                all_args[fname] = self._extract_args(query, func_def)

        return {
            "tools":         result.functions,
            "args":          all_args,
            "confidence":    result.confidence,
            "is_irrelevant": False,
            "latency_ms":    elapsed,
            "top_k":         result.all_scores[:5],
        }

    def is_irrelevant(self, query: str, func_defs: list[dict]) -> tuple[bool, dict]:
        """Determine whether a query is irrelevant (no function should be called).

        Uses the higher irrelevance_threshold for stricter filtering.

        Returns:
            (is_irrelevant: bool, route_result: dict)
        """
        t0 = time.perf_counter()

        self._scorer.configure(func_defs)
        result = self._scorer.score(query)

        elapsed = (time.perf_counter() - t0) * 1000

        irrelevant = result.is_irrelevant or result.confidence < self._irr_threshold
        func_name = result.functions[0] if result.functions and not irrelevant else None
        args: dict = {}

        if func_name:
            func_def = self._find_def(func_defs, func_name)
            if func_def:
                args = self._extract_args(query, func_def)

        route_result = {
            "tool":          func_name,
            "args":          args,
            "confidence":    result.confidence,
            "is_irrelevant": irrelevant,
            "latency_ms":    elapsed,
            "top_k":         result.all_scores[:5],
        }
        return route_result["is_irrelevant"], route_result

    def route_multi_turn(
        self,
        entry: dict,
        func_defs: list[dict],
    ) -> dict[str, Any]:
        """Multi-turn routing via CognitiveLoop with SDK packs and pack-based routing.

        Pipeline per turn:
          1. Detect API class from query (extract_api_class via pack domain_signals)
          2. Pack canonical routing: extract_pack_actions() → map to class functions
          3. DomainConfig multi_action_keywords → additional functions
          4. HDC scorer fallback when pack matching finds nothing
          5. Exclusion rules filter confusable pairs

        CognitiveLoop with packs provides:
          - DeductiveLayer: prerequisite detection from state pack transitions
          - InductiveLayer: learned patterns from state pack patterns
          - IdeaSpace: episodic memory across turns
          - ConversationState: Hebbian pathway tracking

        Args:
            entry:     Multi-turn entry dict with "question", "involved_classes".
            func_defs: Function definitions (class-prefixed, e.g. "GorillaFileSystem.mv").

        Returns:
            per_turn   (list[dict]) — [{functions: [str], confidence: float}, ...]
            latency_ms (float)      — total wall-clock time
        """
        t0 = time.perf_counter()

        involved_classes = entry.get("involved_classes", [])

        # Group func_defs by class prefix
        class_funcs: dict[str, list[dict]] = {}
        for f in func_defs:
            cls = f["name"].split(".")[0]
            class_funcs.setdefault(cls, []).append(f)

        # Build available function name sets per class
        class_available: dict[str, set[str]] = {
            cls: {f["name"] for f in funcs}
            for cls, funcs in class_funcs.items()
        }

        # Create per-class CognitiveLoops with packs + scorers as fallback
        loops: dict[str, CognitiveLoop] = {}
        scorers: dict[str, BFCLScorer] = {}
        for cls in class_funcs:
            # Scorer for fallback
            scorer = BFCLScorer()
            exemplars = self._load_exemplars(cls)
            if exemplars:
                scorer.configure_from_exemplars(exemplars)
            else:
                scorer.configure(class_funcs[cls])
            scorers[cls] = scorer

            # CognitiveLoop with domain packs
            domain_config = CLASS_DOMAIN_CONFIGS.get(cls)
            packs = _CLASS_TO_PACKS.get(cls, [])
            loop = CognitiveLoop(
                packs=packs,
                domain_config=domain_config,
                model_scorer=scorer,
                confidence_threshold=0.10,
            )
            loop.begin(functions=self._to_loop_functions(class_funcs[cls]))
            loops[cls] = loop

        turns = entry.get("question", [])
        per_turn: list[dict] = []

        for turn_messages in turns:
            # miss_func: empty turn → no function calls expected
            if not turn_messages:
                per_turn.append({"functions": [], "confidence": 0.0})
                continue

            query = self._extract_turn_query(turn_messages)
            if not query:
                per_turn.append({"functions": [], "confidence": 0.0})
                continue

            # Stage 1: Detect API class from query
            matched_cls = extract_api_class(query, involved_classes)
            available = class_available.get(matched_cls, set())

            # Stage 2: Pack canonical routing
            # Intent packs provide rich NL synonym/phrase matching.
            # extract_pack_actions returns canonical action names (e.g., "cat", "tail").
            pack_canonicals = extract_pack_actions(query)
            overrides = _PACK_FUNC_OVERRIDES.get(matched_cls, {})

            pack_funcs: list[str] = []
            for canonical in pack_canonicals:
                # Check override mapping first (non-obvious function names)
                if canonical in overrides:
                    fname = overrides[canonical]
                    if fname in available and fname not in pack_funcs:
                        pack_funcs.append(fname)
                    continue
                # Auto-derive: {class}.{canonical} for direct matches
                fname = f"{matched_cls}.{canonical}"
                if fname in available and fname not in pack_funcs:
                    pack_funcs.append(fname)

            # Stage 3: DomainConfig multi_action_keywords (secondary)
            domain_config = CLASS_DOMAIN_CONFIGS.get(matched_cls)
            if domain_config and domain_config.multi_action_keywords:
                query_lower = query.lower()
                for fname, patterns in domain_config.multi_action_keywords.items():
                    if fname in pack_funcs or fname not in available:
                        continue
                    for pat in patterns:
                        if pat in query_lower:
                            pack_funcs.append(fname)
                            break

            # Stage 4: HDC scorer fallback when pack + keywords find nothing
            if not pack_funcs:
                scorer = scorers.get(matched_cls)
                if scorer:
                    result = scorer.score(query)
                    if result.functions:
                        pack_funcs = [result.functions[0]]

            # Stage 5: Exclusion rules
            if domain_config and domain_config.exclusion_rules:
                to_remove: set[str] = set()
                for specific, generics in domain_config.exclusion_rules.items():
                    if specific in pack_funcs:
                        for g in generics:
                            if g in pack_funcs:
                                to_remove.add(g)
                pack_funcs = [f for f in pack_funcs if f not in to_remove]

            # Feed the query through CognitiveLoop for state tracking
            loop = loops.get(matched_cls)
            if loop:
                loop.step(query)

            per_turn.append({
                "functions": pack_funcs,
                "confidence": 0.8 if pack_funcs else 0.0,
            })

        # End all loops
        for loop in loops.values():
            loop.end()

        elapsed = (time.perf_counter() - t0) * 1000

        return {
            "per_turn":   per_turn,
            "latency_ms": elapsed,
        }

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _load_exemplars(cls: str) -> list[dict]:
        """Load pre-built exemplars for a class from classes/{folder}/exemplars.jsonl."""
        folder = _CLASS_TO_FOLDER.get(cls)
        if not folder:
            return []
        path = _CLASSES_DIR / folder / "exemplars.jsonl"
        if not path.exists():
            return []
        exemplars = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    exemplars.append(json.loads(line))
        return exemplars

    @staticmethod
    def _to_loop_functions(func_defs: list[dict]) -> list[dict]:
        """Convert func_defs to CognitiveLoop format."""
        return [
            {
                "name": f["name"],
                "description": f.get("description", ""),
                "parameters": f.get("parameters", {}),
            }
            for f in func_defs
        ]

    @staticmethod
    def _extract_turn_query(turn_messages: list) -> str:
        """Extract user query from a turn's message list."""
        for msg in reversed(turn_messages):
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "")
        if turn_messages:
            last = turn_messages[-1]
            return last.get("content", "") if isinstance(last, dict) else ""
        return ""

    @staticmethod
    def _find_def(func_defs: list[dict], name: str | None) -> dict | None:
        if not name:
            return None
        return next((f for f in func_defs if f.get("name") == name), None)
