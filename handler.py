"""
BFCL handler — pure HDC routing + schema-guided argument extraction.

BFCLHandler ties together:
  BFCLScorer   — HDC function routing (which function to call)
  ArgumentExtractor — schema-guided arg extraction (what values to pass)

No LLM. No external services. All computation is local and deterministic.

Exports:
  BFCLHandler.route(query, func_defs)       → single-function routing result
  BFCLHandler.route_multi(query, func_defs) → multi-function routing result
  BFCLHandler.is_irrelevant(query, func_defs) → irrelevance detection
"""

from __future__ import annotations

import time
from typing import Any

from scorer import BFCLScorer
from extractor import ArgumentExtractor


class BFCLHandler:
    """Pure HDC BFCL handler. No LLM.

    Usage:
        handler = BFCLHandler()
        result  = handler.route(query, func_defs)
        # result["tool"]       — predicted function name (or None if irrelevant)
        # result["args"]       — extracted argument dict
        # result["confidence"] — HDC similarity score
        # result["top_k"]      — top-5 scored functions for debugging
    """

    def __init__(self, confidence_threshold: float = 0.22) -> None:
        self._scorer    = BFCLScorer()
        self._extractor = ArgumentExtractor()
        self._threshold = confidence_threshold

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

        Returns:
            (is_irrelevant: bool, route_result: dict)
        """
        result = self.route(query, func_defs)
        return result["is_irrelevant"], result

    # ── Helpers ──────────────────────────────────────────────────────────

    @staticmethod
    def _find_def(func_defs: list[dict], name: str | None) -> dict | None:
        if not name:
            return None
        return next((f for f in func_defs if f.get("name") == name), None)
