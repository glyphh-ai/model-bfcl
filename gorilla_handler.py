"""
Gorilla-compatible handler for BFCL leaderboard submission.

Wraps BFCLHandler to produce decode_ast / decode_execute outputs
matching the gorilla evaluation framework's expected interface.

No LLM calls — pure HDC routing + CognitiveLoop + rule-based arg extraction.

Integration:
  1. Copy this file into the gorilla repo's model_handler/ directory
  2. Register in model_config.py as a ModelConfig entry
  3. Add to SUPPORTED_MODELS

See: https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/CONTRIBUTING.md
"""

from __future__ import annotations

from typing import Any

from handler import BFCLHandler
from run_bfcl import load_multi_turn_func_defs


class GlyphhBFCLHandler:
    """Gorilla-compatible handler. Pure HDC + CognitiveLoop, no LLM."""

    def __init__(self, confidence_threshold: float = 0.22) -> None:
        self.handler = BFCLHandler(confidence_threshold=confidence_threshold)

    # ── Inference ─────────────────────────────────────────────────────────

    def inference_single_turn(
        self,
        question: list[list[dict]],
        func_defs: list[dict],
    ) -> dict[str, Any]:
        """Process a single-turn query. Returns raw route result."""
        query = self._extract_query(question)
        return self.handler.route(query, func_defs, force=True)

    def inference_multi_turn(
        self,
        question: list[list[dict]],
        func_defs: list[dict],
        initial_config: dict | None = None,
        involved_classes: list[str] | None = None,
        excluded_function: list[str] | None = None,
    ) -> dict[str, Any]:
        """Process a multi-turn query. Returns per-turn routing results."""
        # If func_defs not pre-loaded, load from multi_turn_func_doc/
        if not func_defs and involved_classes:
            func_defs = load_multi_turn_func_defs(
                involved_classes, excluded_function or [],
            )

        entry = {
            "question": question,
            "initial_config": initial_config or {},
        }
        return self.handler.route_multi_turn(entry, func_defs)

    # ── Decode: AST format ────────────────────────────────────────────────

    def decode_ast(
        self,
        result: dict[str, Any],
        language: str = "Python",
        has_tool_call_tag: bool = False,
    ) -> list[dict] | list[list[dict]]:
        """Convert result to AST format: [{"func_name": {"param": val}}]

        For multi-turn results (per_turn key present), returns a list of
        per-turn AST lists.
        """
        if "per_turn" in result:
            # Multi-turn: list of per-turn call lists
            all_turns = []
            for turn in result["per_turn"]:
                calls = []
                for func in turn.get("functions", []):
                    calls.append({func: {}})
                all_turns.append(calls)
            return all_turns

        # Single-turn
        if result.get("tool"):
            return [{result["tool"]: result.get("args", {})}]
        return []

    # ── Decode: Execute format ────────────────────────────────────────────

    def decode_execute(
        self,
        result: dict[str, Any],
        has_tool_call_tag: bool = False,
    ) -> list[str] | list[list[str]]:
        """Convert result to execute format: ["func_name(param=val, ...)"]

        For multi-turn results, returns a list of per-turn call string lists.
        """
        if "per_turn" in result:
            all_turns = []
            for turn in result["per_turn"]:
                calls = []
                for func in turn.get("functions", []):
                    calls.append(f"{func}()")
                all_turns.append(calls)
            return all_turns

        # Single-turn
        if result.get("tool"):
            args = result.get("args", {})
            args_str = ", ".join(f"{k}={repr(v)}" for k, v in args.items())
            return [f"{result['tool']}({args_str})"]
        return []

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _extract_query(question: list[list[dict]]) -> str:
        """Extract the user query from gorilla question format."""
        for turn in question:
            if isinstance(turn, list):
                for msg in reversed(turn):
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        return msg.get("content", "")
            elif isinstance(turn, dict) and turn.get("role") == "user":
                return turn.get("content", "")
        return ""
