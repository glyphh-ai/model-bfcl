"""
LLM-assisted argument extraction for BFCL.

Uses Claude to extract function argument values from natural language queries,
given the function schema. Drop-in replacement for ArgumentExtractor.

HDC handles routing (which function). LLM handles extraction (what values).

Requires:
    pip install anthropic
    export ANTHROPIC_API_KEY=sk-ant-...

Exports:
    LLMArgumentExtractor.extract(query, func_def) → dict[str, Any]
"""

from __future__ import annotations

import json
import os
from typing import Any

import anthropic


class LLMArgumentExtractor:
    """Extract function arguments using Claude."""

    def __init__(self, model: str = "claude-haiku-4-5-20251001", temperature: float = 0.0) -> None:
        self._client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
        self._model = model
        self._temperature = temperature
        # Token usage tracking
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_calls = 0
        self._snapshot_input = 0
        self._snapshot_output = 0
        self._snapshot_calls = 0

    def snapshot(self) -> None:
        """Save current counters so category deltas can be computed."""
        self._snapshot_input = self.total_input_tokens
        self._snapshot_output = self.total_output_tokens
        self._snapshot_calls = self.total_calls

    def get_category_usage(self) -> dict:
        """Return token usage since last snapshot()."""
        return {
            "input_tokens": self.total_input_tokens - self._snapshot_input,
            "output_tokens": self.total_output_tokens - self._snapshot_output,
            "calls": self.total_calls - self._snapshot_calls,
        }

    def get_total_usage(self) -> dict:
        """Return cumulative token usage."""
        return {
            "input_tokens": self.total_input_tokens,
            "output_tokens": self.total_output_tokens,
            "calls": self.total_calls,
        }

    def extract(self, query: str, func_def: dict) -> dict[str, Any]:
        """Extract arguments from a query given a function schema.

        Same interface as ArgumentExtractor.extract().

        Args:
            query:    Natural language query string.
            func_def: BFCL function definition dict with 'parameters' schema.

        Returns:
            Dict mapping parameter names to extracted values.
        """
        params = func_def.get("parameters", {})
        props = params.get("properties", {}) if isinstance(params, dict) else {}
        if not props:
            return {}

        schema_str = json.dumps({
            "name": func_def.get("name", ""),
            "description": func_def.get("description", ""),
            "parameters": params,
        }, indent=2)

        prompt = (
            "Extract the function argument values from the user query.\n\n"
            f"Function schema:\n```json\n{schema_str}\n```\n\n"
            f"User query: {query}\n\n"
            "Return ONLY a JSON object mapping parameter names to their extracted values.\n"
            "IMPORTANT RULES:\n"
            "- Include ALL parameters defined in the schema, not just those explicitly mentioned.\n"
            "- For parameters not mentioned in the query, use the default value from the schema.\n"
            "- If no default is specified and the parameter is not mentioned, use a sensible default "
            "for the type (0 for numbers, \"\" for strings, false for booleans, [] for arrays).\n"
            "- Use the exact types specified in the schema (integer, number, string, boolean, array, dict).\n"
            "- For enum parameters, pick the value that best matches the query context, or the first enum value as default.\n"
            "- For mathematical expressions, use Python syntax: ** for exponentiation (not ^), "
            "e.g. 'x**2' not 'x^2', '3x**2 + 2x - 1' not '3x^2 + 2x - 1'.\n"
            "- When a parameter expects a variable name or identifier referenced in the query, "
            "use that exact name (e.g. if query says 'using mapController', use 'mapController').\n\n"
            "JSON:"
        )

        try:
            response = self._client.messages.create(
                model=self._model,
                temperature=self._temperature,
                max_tokens=1024,
                messages=[{"role": "user", "content": prompt}],
            )
            # Track token usage
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
            self.total_calls += 1

            text = response.content[0].text.strip()

            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            result = json.loads(text)
            if not isinstance(result, dict):
                return {}

            # Coerce types to match schema
            return self._coerce_types(result, props)

        except Exception:
            return {}

    def extract_parallel(self, query: str, func_defs: list[dict]) -> list[dict]:
        """Extract multiple parallel function calls from a query.

        The query may require the same function called multiple times with
        different args, or multiple different functions each called once or more.

        Returns:
            List of {func_name: {arg: value}} dicts — one per function call.
        """
        schemas = []
        for fd in func_defs:
            params = fd.get("parameters", {})
            schemas.append({
                "name": fd.get("name", ""),
                "description": fd.get("description", ""),
                "parameters": params,
            })

        schema_str = json.dumps(schemas, indent=2)

        prompt = (
            "The user query requires one or more function calls, possibly calling "
            "the same function multiple times with different arguments.\n\n"
            f"Available functions:\n```json\n{schema_str}\n```\n\n"
            f"User query: {query}\n\n"
            "Return ONLY a JSON array of function call objects. Each object has the "
            "function name as key and an object of arguments as value.\n"
            "Example: [{\"func_name\": {\"arg1\": \"val1\"}}, {\"func_name\": {\"arg2\": \"val2\"}}]\n\n"
            "IMPORTANT RULES:\n"
            "- If the query asks to do something for multiple items (e.g., multiple cities, "
            "multiple people), make a SEPARATE function call for each item.\n"
            "- Include ALL parameters defined in the schema for each call.\n"
            "- For parameters not mentioned, use the default value from the schema.\n"
            "- If no default, use a sensible default (0 for numbers, \"\" for strings, false for booleans, [] for arrays).\n"
            "- For enum parameters, pick the best match or the first enum value.\n\n"
            "JSON:"
        )

        try:
            response = self._client.messages.create(
                model=self._model,
                temperature=self._temperature,
                max_tokens=2048,
                messages=[{"role": "user", "content": prompt}],
            )
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
            self.total_calls += 1

            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()

            result = json.loads(text)
            if not isinstance(result, list):
                return []

            # Coerce types per function schema
            props_map = {}
            for fd in func_defs:
                params = fd.get("parameters", {})
                props_map[fd.get("name", "")] = (
                    params.get("properties", {}) if isinstance(params, dict) else {}
                )

            coerced = []
            for call in result:
                if not isinstance(call, dict):
                    continue
                for fname, args in call.items():
                    if isinstance(args, dict) and fname in props_map:
                        args = self._coerce_types(args, props_map[fname])
                    coerced.append({fname: args})
            return coerced

        except Exception:
            return []

    @staticmethod
    def _coerce_types(result: dict, props: dict) -> dict[str, Any]:
        """Coerce extracted values to match schema types."""
        coerced = {}
        for key, value in result.items():
            if key not in props:
                continue
            ptype = props[key].get("type", "string")
            try:
                if ptype in ("integer", "int"):
                    coerced[key] = int(value) if not isinstance(value, int) else value
                elif ptype in ("number", "float"):
                    coerced[key] = float(value) if not isinstance(value, float) else value
                elif ptype == "boolean":
                    coerced[key] = bool(value)
                elif ptype == "array":
                    if isinstance(value, list):
                        coerced[key] = value
                    elif isinstance(value, str):
                        coerced[key] = [value]  # wrap string, don't split into chars
                    else:
                        coerced[key] = [value]
                elif ptype == "dict":
                    coerced[key] = dict(value) if not isinstance(value, dict) else value
                else:
                    coerced[key] = value
            except (ValueError, TypeError):
                coerced[key] = value
        return coerced
