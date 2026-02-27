"""
Gorilla BFCL CLI-compatible handler for Glyphh.

Implements the gorilla BaseHandler interface so Glyphh can be run via:
  bfcl generate --model glyphh-anthropic-claude-haiku-4-5-20251001 --test-category all
  bfcl evaluate --model glyphh-anthropic-claude-haiku-4-5-20251001 --test-category all

Architecture:
  - Routing:    Glyphh HDC (0 tokens)
  - Arg fill:   LLM sees ONLY the matched function schema (~150 tokens vs 2000+)
  - Parallel:   route_multi → fill args per matched function
  - Irrelevance: is_irrelevant() → return []
  - Multi-turn: delegate to multi_turn_handler

Model name convention:
  glyphh-{provider}-{model}
  Examples:
    glyphh-anthropic-claude-haiku-4-5-20251001
    glyphh-openai-gpt-4.1-nano
    glyphh-google-gemini-2.0-flash-lite

Usage (from gorilla repo root):
  export ANTHROPIC_API_KEY=...
  bfcl generate --model glyphh-anthropic-claude-haiku-4-5-20251001 --test-category simple
  bfcl evaluate --model glyphh-anthropic-claude-haiku-4-5-20251001 --test-category simple
"""

import json
import os
import re
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

from handler import GlyphhBFCLHandler
from llm_client import OpenAIClient, GeminiClient

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
    "multi_turn_long_context",
    "multi_turn_composite",
    "multi_turn_parallel",
    "multi_turn_parallel_multiple",
    "multi_turn_irrelevance",
}

# All categories that need arg filling via LLM
_NEEDS_LLM = True  # We always use LLM for arg extraction


# ---------------------------------------------------------------------------
# Model name parsing
# ---------------------------------------------------------------------------

def _parse_model_name(model_name: str) -> tuple[str, str]:
    """Parse 'glyphh-{provider}-{model}' into (provider, model).

    Examples:
      glyphh-anthropic-claude-haiku-4-5-20251001 → (anthropic, claude-haiku-4-5-20251001)
      glyphh-openai-gpt-4.1-nano                → (openai, gpt-4.1-nano)
      glyphh-google-gemini-2.0-flash-lite        → (google, gemini-2.0-flash-lite)
    """
    prefix = "glyphh-"
    if model_name.startswith(prefix):
        rest = model_name[len(prefix):]
    else:
        rest = model_name

    # Known providers
    for provider in ("anthropic", "openai", "google"):
        if rest.startswith(provider + "-"):
            return provider, rest[len(provider) + 1:]

    # Default: first token is provider
    parts = rest.split("-", 1)
    if len(parts) == 2:
        return parts[0], parts[1]
    return "openai", rest


def _build_llm_client(provider: str, model: str):
    """Instantiate the appropriate LLM client."""
    if provider == "anthropic":
        # Use OpenAI-compatible client with Anthropic's API
        try:
            import anthropic as _ant  # noqa: F401
            return AnthropicArgClient(model)
        except ImportError:
            raise RuntimeError("pip install anthropic to use the Anthropic provider")
    elif provider == "google":
        return GeminiClient(model)
    else:
        # openai or any compatible
        return OpenAIClient(model)


# ---------------------------------------------------------------------------
# Anthropic arg extraction client (mirrors OpenAIClient interface)
# ---------------------------------------------------------------------------

class AnthropicArgClient:
    """Thin Anthropic client for argument extraction.

    Uses claude-* models via Anthropic SDK. Exposes the same extract_args /
    extract_multi_call_args interface as OpenAIClient.
    """

    def __init__(self, model: str):
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import anthropic
            self._client = anthropic.Anthropic()
        return self._client

    def _call(self, system: str, user_msg: str) -> str:
        resp = self.client.messages.create(
            model=self.model,
            max_tokens=512,
            temperature=0.0,
            system=system,
            messages=[{"role": "user", "content": user_msg}],
        )
        return resp.content[0].text.strip()

    def extract_args(
        self,
        query: str,
        func_name: str,
        func_def: dict,
        context: list | None = None,
    ) -> dict:
        params = func_def.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])

        param_lines = []
        for pname, pdef in props.items():
            ptype = pdef.get("type", "any")
            req = "(required)" if pname in required else "(optional)"
            default = f", default={pdef['default']}" if "default" in pdef else ""
            enum = f", enum={pdef['enum']}" if "enum" in pdef else ""
            param_lines.append(f"  - {pname}: {ptype} {req} — {pdef.get('description', '')}{default}{enum}")

        param_block = "\n".join(param_lines) if param_lines else "  (no parameters)"

        system = (
            "You extract function call arguments from a user query. "
            "Return ONLY a JSON object with the argument names and values. "
            "No explanation, no markdown, no wrapping — just the raw JSON object. "
            "Use the exact parameter names from the function signature. "
            "Only include parameters that the user explicitly or implicitly specifies. "
            "For required parameters with no clear value, use reasonable defaults."
        )
        user_msg = f"Function: {func_name}\nParameters:\n{param_block}\n\nQuery: {query}"

        try:
            text = self._call(system, user_msg)
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            return json.loads(text)
        except Exception:
            return {}

    def extract_multi_call_args(
        self,
        query: str,
        func_names: list[str],
        func_defs: list[dict],
        context: list | None = None,
    ) -> list[dict]:
        func_blocks = []
        for fname, fdef in zip(func_names, func_defs):
            params = fdef.get("parameters", {}).get("properties", {})
            required = fdef.get("parameters", {}).get("required", [])
            param_lines = []
            for pname, pdef in params.items():
                ptype = pdef.get("type", "any")
                req = "(required)" if pname in required else "(optional)"
                param_lines.append(f"    - {pname}: {ptype} {req}")
            func_blocks.append(f"  {fname}:\n" + "\n".join(param_lines))

        system = (
            "You extract function call arguments from a user query. "
            "The user wants to call multiple functions. "
            "Return a JSON array of objects, each with the function name as key "
            "and a dict of arguments as value. Example: "
            '[{"func_a": {"x": 1}}, {"func_b": {"y": 2}}]. '
            "No explanation, no markdown — just the raw JSON array."
        )
        user_msg = "Functions:\n" + "\n".join(func_blocks) + f"\n\nQuery: {query}"

        try:
            text = self._call(system, user_msg)
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            return json.loads(text)
        except Exception:
            return [{name: {}} for name in func_names]


# ---------------------------------------------------------------------------
# GlyphhHandler — the gorilla-compatible handler
# ---------------------------------------------------------------------------

class GlyphhHandler(BaseHandler):
    """Gorilla BFCL-compatible handler using Glyphh HDC routing + LLM arg fill.

    Glyphh handles ALL routing decisions (zero tokens).
    The LLM fills arguments for the single matched function schema (~150 tokens).
    For parallel/parallel_multiple, each matched function gets its own arg-fill call.
    """

    def __init__(self, model_name: str, temperature: float = 0.0, **kwargs):
        super().__init__(model_name, temperature=temperature, **kwargs)
        self.glyphh = GlyphhBFCLHandler(threshold=0.15)
        provider, llm_model = _parse_model_name(model_name)
        self.llm = _build_llm_client(provider, llm_model)

    # ── Gorilla interface ────────────────────────────────────────────────────

    def inference(
        self,
        test_question: list[dict],
        functions: list = [],
        model: str = "",
        test_category: str = "simple",
    ) -> list[dict] | str:
        """Route query and fill arguments.

        Returns list of {func_name: {args}} or [] for irrelevance.
        """
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

        # Irrelevance categories
        if test_category in IRRELEVANCE_CATS:
            result = self.glyphh.route(query, func_defs)
            q_glyph = result.get("_query_glyph")
            f_glyph = result.get("_best_func_glyph")
            if q_glyph and f_glyph and self.glyphh.is_irrelevant(
                q_glyph, f_glyph, result["confidence"]
            ):
                return []
            # Relevant — fall through to single-function arg fill
            tool = result.get("tool")
            if not tool:
                return []
            return self._fill_single_args(query, tool, func_defs)

        # Parallel_multiple — route to multiple distinct functions
        if test_category in MULTI_CATS:
            result = self.glyphh.route_multi(query, func_defs)
            tools = result.get("tools", [])
            if not tools:
                return []
            return self._fill_multi_args(query, tools, func_defs)

        # Parallel — same function called with different args
        if test_category in PARALLEL_CATS:
            result = self.glyphh.route(query, func_defs)
            tool = result.get("tool")
            if not tool:
                return []
            # For parallel: LLM extracts all call instances
            return self._fill_parallel_args(query, tool, func_defs)

        # Simple / multiple / live variants — single best function
        result = self.glyphh.route(query, func_defs)
        tool = result.get("tool")
        if not tool:
            return []
        return self._fill_single_args(query, tool, func_defs)

    def decode_ast(self, result, language: str = "Python"):
        """Convert inference result to gorilla AST format."""
        if not result:
            return result
        if isinstance(result, str):
            return result
        # Already in [{func_name: {args}}] format — gorilla expects this
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
                    # Gorilla wraps functions as {"type": "function", "function": {...}}
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

    def _fill_single_args(
        self, query: str, func_name: str, func_defs: list[dict]
    ) -> list[dict]:
        """LLM fills args for one matched function. LLM sees ONLY this schema."""
        matched = next((f for f in func_defs if f.get("name") == func_name), None)
        if not matched:
            return [{func_name: {}}]
        args = self.llm.extract_args(query, func_name, matched)
        return [{func_name: args}]

    def _fill_multi_args(
        self, query: str, func_names: list[str], func_defs: list[dict]
    ) -> list[dict]:
        """LLM fills args for multiple matched functions (parallel_multiple)."""
        matched_defs = []
        for fname in func_names:
            fdef = next((f for f in func_defs if f.get("name") == fname), {"name": fname})
            matched_defs.append(fdef)

        calls = self.llm.extract_multi_call_args(query, func_names, matched_defs)
        if not calls:
            return [{name: {}} for name in func_names]
        return calls

    def _fill_parallel_args(
        self, query: str, func_name: str, func_defs: list[dict]
    ) -> list[dict]:
        """LLM extracts all parallel calls for the same function.

        For parallel category: the user wants to call the same function
        multiple times with different arguments.
        """
        matched = next((f for f in func_defs if f.get("name") == func_name), None)
        if not matched:
            return [{func_name: {}}]

        params = matched.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])

        param_lines = []
        for pname, pdef in props.items():
            ptype = pdef.get("type", "any")
            req = "(required)" if pname in required else "(optional)"
            param_lines.append(f"  - {pname}: {ptype} {req} — {pdef.get('description', '')}")

        system = (
            "You extract multiple parallel function call arguments from a user query. "
            "The user wants to call the same function multiple times with different arguments. "
            "Return a JSON array of objects, each with the function name as key and a dict of args as value. "
            f'Example: [{{"func": {{"x": 1}}}}, {{"func": {{"x": 2}}}}]. '
            "No explanation, no markdown — just the raw JSON array."
        )
        user_msg = (
            f"Function: {func_name}\n"
            f"Parameters:\n" + "\n".join(param_lines) + f"\n\nQuery: {query}"
        )

        try:
            if hasattr(self.llm, "_call"):
                text = self.llm._call(system, user_msg)
            else:
                # OpenAI-compatible
                resp = self.llm.client.chat.completions.create(
                    model=self.llm.model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.0,
                    max_tokens=512,
                )
                text = resp.choices[0].message.content.strip()

            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            calls = json.loads(text)
            if isinstance(calls, list):
                return calls
            if isinstance(calls, dict):
                return [{func_name: calls}]
        except Exception:
            pass

        return [{func_name: {}}]

    def _handle_multi_turn(
        self,
        test_question: list[dict],
        functions,
        test_category: str,
    ) -> list[dict]:
        """Delegate multi-turn handling to multi_turn_handler."""
        try:
            from multi_turn_handler import eval_multi_turn_entry
            func_defs = self._normalize_functions(functions)
            # Gorilla passes the full entry as test_question for multi-turn
            entry = {"question": test_question, "involved_classes": []}
            result = eval_multi_turn_entry(self.glyphh, self.llm, entry, ground_truth=[])
            # Return last turn's prediction
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
    parser.add_argument("--model", default="glyphh-anthropic-claude-haiku-4-5-20251001")
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
