"""
BFCL Handler using CognitiveLoop + optional HDC ModelScorer + optional LLM.

Three modes:
  1. Full hybrid (default): HDC scorer routes, LLM extracts args + arbitrates
  2. HDC-only (--hdc-only): HDC scorer routes, no LLM (sub-ms per query)
  3. LLM-only (use_scorer=False): Original LLM-primary behavior

The HDC scorer (BFCLModelScorer) uses the BFCL Model A encoder to score
queries against function definitions in 10000-dim vector space. The
CognitiveLoop's SchemaIntentClassifier handles the three-tier waterfall:
  ModelScorer (domain HDC) → IntentCache (learned HDC) → LLM (generative)
"""

from __future__ import annotations

import time
from typing import Any, TYPE_CHECKING

from glyphh.cognitive import CognitiveLoop

if TYPE_CHECKING:
    from glyphh.llm.engine import LLMEngine

# Irrelevance detection: if confidence is below this threshold
# AND no functions were resolved, classify as irrelevant.
IRRELEVANCE_THRESHOLD = 0.15


class CognitiveBFCLHandler:
    """BFCL handler backed by CognitiveLoop + HDC scorer + LLM.

    Usage:
        from glyphh.llm import LLMEngine

        engine = LLMEngine()
        handler = CognitiveBFCLHandler(engine=engine, use_scorer=True)

        # Single-turn routing
        result = handler.route(query, func_defs)

        # Multi-function routing
        result = handler.route_multi(query, func_defs)

        # Irrelevance detection
        is_irr = handler.is_irrelevant_query(query, func_defs)

        # HDC-only (no LLM needed):
        handler = CognitiveBFCLHandler(engine=None, use_scorer=True)
    """

    def __init__(
        self,
        engine: LLMEngine | None = None,
        confidence_threshold: float = 0.25,
        use_scorer: bool = True,
    ):
        self._engine = engine
        self._threshold = confidence_threshold
        self._use_scorer = use_scorer
        self._scorer = None

        if use_scorer:
            from model_scorer import BFCLModelScorer
            self._scorer = BFCLModelScorer()

    def _make_loop(self, func_defs: list[dict]) -> CognitiveLoop:
        """Create and begin a fresh CognitiveLoop for one entry."""
        loop = CognitiveLoop(
            domain_config=None,
            llm_engine=self._engine,
            confidence_threshold=self._threshold,
            model_scorer=self._scorer,
        )
        loop.begin(functions=func_defs)
        return loop

    # ── Single-function routing ──

    def route(self, query: str, func_defs: list[dict]) -> dict[str, Any]:
        """Route a query to the best-matching function.

        Returns dict compatible with the existing eval pipeline:
            {tool, confidence, latency_ms, top_k, all_scores, calls}
        """
        start = time.perf_counter()
        loop = self._make_loop(func_defs)
        result = loop.step(query)
        loop.end()
        elapsed_ms = (time.perf_counter() - start) * 1000

        # Extract the primary predicted function
        predicted = None
        calls = []
        if result.action == "CALL" and result.calls:
            calls = result.calls
            predicted = next(iter(result.calls[0].keys()), None)

        # Build all_scores from classification signals
        classification = result.signals.get("classification", {})
        all_scores = self._build_scores(classification, func_defs)

        return {
            "tool": predicted,
            "confidence": round(result.confidence, 4),
            "latency_ms": elapsed_ms,
            "top_k": all_scores[:3],
            "all_scores": all_scores,
            "calls": calls,
            "source": classification.get("source", "unknown"),
        }

    # ── Multi-function routing ──

    def route_multi(self, query: str, func_defs: list[dict]) -> dict[str, Any]:
        """Route a query to ALL matching functions (parallel_multiple).

        When the HDC scorer is active, uses score_multi() for gap analysis.
        """
        start = time.perf_counter()

        # If scorer is active, use its gap analysis for multi-function
        if self._scorer is not None:
            from glyphh.cognitive.model_scorer import ScorerResult
            self._scorer.configure(func_defs)
            scorer_result = self._scorer.score_multi(query)
            elapsed_ms = (time.perf_counter() - start) * 1000

            # If scorer found multiple functions, use them
            if scorer_result.functions:
                calls = [{f: {}} for f in scorer_result.functions]

                # Try LLM for args if available
                if self._engine is not None and scorer_result.functions:
                    loop = self._make_loop(func_defs)
                    step_result = loop.step(query)
                    loop.end()
                    elapsed_ms = (time.perf_counter() - start) * 1000

                    # Merge LLM args with scorer routing
                    if step_result.action == "CALL":
                        llm_args = {}
                        for c in step_result.calls:
                            llm_args.update(c)
                        calls = []
                        for fname in scorer_result.functions:
                            args = llm_args.get(fname, {})
                            calls.append({fname: args})

                return {
                    "tools": scorer_result.functions,
                    "confidence": round(scorer_result.confidence, 4),
                    "latency_ms": elapsed_ms,
                    "top_k": scorer_result.all_scores[:5],
                    "all_scores": scorer_result.all_scores,
                    "calls": calls,
                    "source": "model_scorer",
                }

        # Fallback: CognitiveLoop handles everything
        loop = self._make_loop(func_defs)
        result = loop.step(query)
        loop.end()
        elapsed_ms = (time.perf_counter() - start) * 1000

        tools = []
        calls = []
        if result.action == "CALL" and result.calls:
            calls = result.calls
            for call in result.calls:
                tools.extend(call.keys())

        classification = result.signals.get("classification", {})
        all_scores = self._build_scores(classification, func_defs)

        return {
            "tools": tools,
            "confidence": round(result.confidence, 4),
            "latency_ms": elapsed_ms,
            "top_k": all_scores[:5],
            "all_scores": all_scores,
            "calls": calls,
            "source": classification.get("source", "unknown"),
        }

    # ── Irrelevance detection ──

    def is_irrelevant_query(
        self,
        query: str,
        func_defs: list[dict],
    ) -> tuple[bool, dict[str, Any]]:
        """Detect if a query is irrelevant to the available functions.

        When the HDC scorer is active, check for scorer-flagged irrelevance
        first (uses role-level analysis, sub-ms).

        Returns (is_irrelevant, route_result).
        """
        route_result = self.route(query, func_defs)

        # Check scorer-flagged irrelevance
        if route_result.get("source") == "model_scorer_irrelevant":
            return True, route_result

        # If no function was resolved and confidence is low → irrelevant
        if route_result["tool"] is None:
            return True, route_result

        if route_result["confidence"] < IRRELEVANCE_THRESHOLD:
            return True, route_result

        return False, route_result

    # ── BFCL evaluation interface ──

    def predict(
        self,
        query: str,
        func_defs: list[dict],
        category: str = "simple",
    ) -> list[dict]:
        """BFCL-compatible prediction interface."""
        if category in ("parallel_multiple", "live_parallel_multiple"):
            result = self.route_multi(query, func_defs)
            return result.get("calls", [])

        if category in ("irrelevance", "live_irrelevance"):
            is_irr, result = self.is_irrelevant_query(query, func_defs)
            if is_irr:
                return []
            return result.get("calls", [])

        result = self.route(query, func_defs)
        return result.get("calls", [])

    def decode_ast(self, result: list[dict]) -> list[dict]:
        """Convert prediction result to BFCL AST format."""
        return result

    def decode_execute(self, result: list[dict]) -> list[str]:
        """Convert prediction result to BFCL executable format."""
        calls = []
        for call in result:
            for func_name, args in call.items():
                if args:
                    arg_str = ", ".join(
                        f"{k}={repr(v)}" for k, v in args.items()
                    )
                    calls.append(f"{func_name}({arg_str})")
                else:
                    calls.append(f"{func_name}()")
        return calls

    # ── Internal ──

    @staticmethod
    def _build_scores(
        classification: dict,
        func_defs: list[dict],
    ) -> list[dict]:
        """Build an all_scores list from classification signals.

        When model_scorer is the source, uses the all_scores from the scorer.
        Otherwise builds from LLM classification confidence.
        """
        # Use scorer's all_scores directly if available
        scorer_scores = classification.get("all_scores")
        if scorer_scores:
            return scorer_scores

        # Fallback: LLM-based scores
        conf = classification.get("confidence", 0.0)
        selected = set(classification.get("functions", []))

        scores = []
        for fd in func_defs:
            fname = fd["name"]
            score = conf if fname in selected else 0.0
            scores.append({"function": fname, "score": round(score, 4)})

        scores.sort(key=lambda x: x["score"], reverse=True)
        return scores
