"""
Gorilla BFCL CLI-compatible handler for Glyphh.

Implements the gorilla BaseHandler interface so Glyphh can be run via:
  bfcl generate --model glyphh --test-category all
  bfcl evaluate --model glyphh --test-category all

Architecture:
  - Routing + args: CognitiveLoop (SchemaIntentClassifier + SlotExtractor)
  - LLM:           Local LLMEngine (Qwen3-4B via llama-cpp-python)
  - Irrelevance:   Confidence threshold on CognitiveLoop output
  - Multi-turn:    CognitiveLoop with DomainConfig

Usage (from gorilla repo root):
  bfcl generate --model glyphh --test-category simple
  bfcl evaluate --model glyphh --test-category simple
"""

import json
import os
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Gorilla BaseHandler import (with graceful fallback for standalone use)
# ---------------------------------------------------------------------------

try:
    from bfcl.model_handler.base_handler import BaseHandler
    from bfcl.constant import GORILLA_TO_OPENAPI
    _GORILLA_AVAILABLE = True
except ImportError:
    # Fallback for standalone testing outside the gorilla repo
    class BaseHandler:  # type: ignore
        def __init__(self, model_name, temperature=0.001, top_p=1, max_tokens=1200):
            self.model_name = model_name
            self.temperature = temperature
            self.top_p = top_p
            self.max_tokens = max_tokens

        def write(self, result, file_to_open):
            os.makedirs(os.path.dirname(file_to_open), exist_ok=True)
            with open(file_to_open, "a") as f:
                f.write(json.dumps(result) + "\n")

    _GORILLA_AVAILABLE = False

# ---------------------------------------------------------------------------
# Local imports (from bfcl model directory)
# ---------------------------------------------------------------------------

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE))

from cognitive_handler import CognitiveBFCLHandler

# ---------------------------------------------------------------------------
# Category groupings
# ---------------------------------------------------------------------------

IRRELEVANCE_CATS = {
    "irrelevance", "live_irrelevance",
}

MULTI_CATS = {
    "parallel_multiple", "live_parallel_multiple",
}

PARALLEL_CATS = {
    "parallel", "live_parallel",
}

MULTI_TURN_CATS = {
    "multi_turn_base",
    "multi_turn_miss_func",
    "multi_turn_miss_param",
    "multi_turn_long_context",
    "multi_turn_composite",
    "multi_turn_parallel",
    "multi_turn_parallel_multiple",
    "multi_turn_irrelevance",
}


# ---------------------------------------------------------------------------
# GlyphhHandler — the gorilla-compatible handler
# ---------------------------------------------------------------------------

class GlyphhHandler(BaseHandler):
    """Gorilla BFCL-compatible handler using CognitiveLoop + local LLM.

    The CognitiveLoop handles all routing and argument extraction via
    SchemaIntentClassifier backed by LLMEngine (local Qwen3-4B GGUF).
    """

    def __init__(self, model_name: str, temperature: float = 0.0, **kwargs):
        super().__init__(model_name, temperature=temperature, **kwargs)

        # Initialize local LLM engine
        self._engine = None
        try:
            from glyphh.llm import LLMEngine
            self._engine = LLMEngine()
        except Exception as e:
            print(f"Warning: LLMEngine init failed ({e}), running HDC-only")

        self.handler = CognitiveBFCLHandler(engine=self._engine, use_scorer=True)

    # ── Gorilla interface ────────────────────────────────────────────────────

    def inference(
        self,
        test_question: list[dict],
        functions: list = [],
        model: str = "",
        test_category: str = "simple",
    ) -> list[dict] | str:
        """Route query and fill arguments via CognitiveLoop."""
        # Extract user query from test_question messages
        query = self._extract_query(test_question)
        if not query:
            return []

        # Multi-turn categories — delegate to multi_turn_handler
        if test_category in MULTI_TURN_CATS:
            return self._handle_multi_turn(test_question, functions, test_category)

        # Normalize functions to list of dicts
        func_defs = self._normalize_functions(functions)
        if not func_defs:
            return []

        # Use CognitiveBFCLHandler for all categories
        return self.handler.predict(query, func_defs, test_category)

    def decode_ast(self, result, language: str = "Python"):
        """Convert inference result to gorilla AST format."""
        if not result:
            return result
        if isinstance(result, str):
            return result
        return result

    def decode_execute(self, result):
        """Convert inference result to executable function call strings."""
        if not result:
            return result
        if isinstance(result, str):
            return result
        calls = []
        for call in result:
            if isinstance(call, dict):
                for func_name, args in call.items():
                    if args:
                        arg_str = ", ".join(
                            f"{k}={repr(v)}" for k, v in args.items()
                        )
                        calls.append(f"{func_name}({arg_str})")
                    else:
                        calls.append(f"{func_name}()")
        return calls

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _extract_query(self, test_question: list[dict]) -> str:
        """Extract the user query string from gorilla test_question format."""
        if not test_question:
            return ""
        for msg in test_question:
            if isinstance(msg, dict):
                if msg.get("role") == "user":
                    return msg.get("content", "")
        # Fallback: last message content
        last = test_question[-1]
        if isinstance(last, dict):
            return last.get("content", "")
        return str(last)

    def _normalize_functions(self, functions) -> list[dict]:
        """Normalize gorilla functions input to list of JSON Schema dicts."""
        if not functions:
            return []
        if isinstance(functions, list):
            result = []
            for f in functions:
                if isinstance(f, dict):
                    if "function" in f:
                        result.append(f["function"])
                    else:
                        result.append(f)
                elif isinstance(f, str):
                    try:
                        result.append(json.loads(f))
                    except Exception:
                        pass
            return result
        return []

    def _handle_multi_turn(
        self,
        test_question: list[dict],
        functions,
        test_category: str,
    ) -> list[dict]:
        """Delegate multi-turn handling to multi_turn_handler."""
        try:
            from multi_turn_handler import eval_multi_turn_entry
            entry = {"question": test_question, "involved_classes": []}
            result = eval_multi_turn_entry(entry, ground_truth=[], engine=self._engine)
            details = result.get("details", [])
            if details:
                last = details[-1]
                predicted_funcs = last.get("predicted_funcs", [])
                if isinstance(predicted_funcs, list) and predicted_funcs:
                    return [{fn: {}} for fn in predicted_funcs]
        except Exception:
            pass
        return []


# ---------------------------------------------------------------------------
# Standalone test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test GlyphhHandler locally")
    parser.add_argument("--model", default="glyphh")
    parser.add_argument("--query", default="Get the weather for San Francisco")
    parser.add_argument("--category", default="simple")
    args = parser.parse_args()

    handler = GlyphhHandler(args.model)

    test_funcs = [
        {
            "name": "get_weather",
            "description": "Get current weather for a location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "City name"},
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"], "description": "Temperature unit"},
                },
                "required": ["location"],
            },
        },
        {
            "name": "search_hotels",
            "description": "Search for hotels in a city",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string"},
                    "checkin": {"type": "string"},
                },
                "required": ["city"],
            },
        },
    ]

    result = handler.inference(
        test_question=[{"role": "user", "content": args.query}],
        functions=test_funcs,
        test_category=args.category,
    )
    print(f"Query:    {args.query}")
    print(f"Category: {args.category}")
    print(f"Result:   {json.dumps(result, indent=2)}")
