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
from intent import extract_api_class

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
                args = self._extractor.extract(query, func_def)

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
                all_args[fname] = self._extractor.extract(query, func_def)

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
                args = self._extractor.extract(query, func_def)

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
        """Two-stage multi-turn routing with per-class exemplar-based HDC scoring.

        Stage 1: Detect which API class a turn's query targets.
        Stage 2: Score against that class's exemplar Glyphs + DomainConfig keywords.

        Each class has pre-built exemplars (3 weighted BoW variants per function)
        that provide diverse matching surfaces for HDC cosine similarity.

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

        # Create per-class scorers — prefer exemplar-based, fallback to func_def
        scorers: dict[str, BFCLScorer] = {}
        for cls in class_funcs:
            scorer = BFCLScorer()
            exemplars = self._load_exemplars(cls)
            if exemplars:
                scorer.configure_from_exemplars(exemplars)
            else:
                scorer.configure(class_funcs[cls])
            scorers[cls] = scorer

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

            # Stage 2: Score against class's exemplar Glyphs
            scorer = scorers.get(matched_cls)
            if scorer is None:
                per_turn.append({"functions": [], "confidence": 0.0})
                continue

            result = scorer.score_multi(query)
            funcs = list(result.functions)  # multiple matches via gap analysis

            # Multi-function: check DomainConfig multi_action_keywords
            domain_config = CLASS_DOMAIN_CONFIGS.get(matched_cls)
            if domain_config and domain_config.multi_action_keywords:
                available = {f["name"] for f in class_funcs.get(matched_cls, [])}
                query_lower = query.lower()
                for fname, patterns in domain_config.multi_action_keywords.items():
                    if fname in funcs or fname not in available:
                        continue
                    for pat in patterns:
                        if pat in query_lower:
                            funcs.append(fname)
                            break

            per_turn.append({
                "functions": funcs,
                "confidence": result.confidence,
            })

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
