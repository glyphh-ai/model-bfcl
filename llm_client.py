"""
Thin LLM client for argument extraction.

Glyphh handles routing. This module only extracts function arguments
from the query + function signature. It's intentionally minimal —
the LLM is a dumb pipe, not the brain.

Supports: OpenAI-compatible APIs (GPT-4.1-nano, etc.), Google Gemini.
"""

import json
import os
from typing import Any

# Load .env if present
_env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", ".env")
if os.path.exists(_env_path):
    with open(_env_path) as _f:
        for _line in _f:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _key, _val = _line.split("=", 1)
                os.environ.setdefault(_key.strip(), _val.strip())


def get_client(provider: str = "openai", model: str = "gpt-4.1-nano"):
    """Get an LLM client for argument extraction."""
    if provider == "openai":
        return OpenAIClient(model)
    elif provider == "gemini":
        return GeminiClient(model)
    else:
        raise ValueError(f"Unknown provider: {provider}")


class OpenAIClient:
    """OpenAI-compatible API client."""

    def __init__(self, model: str = "gpt-4.1-nano"):
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI
            self._client = OpenAI()
        return self._client

    def extract_args(
        self,
        query: str,
        func_name: str,
        func_def: dict,
        context: list[dict] | None = None,
    ) -> dict:
        """Extract function arguments from query + function signature.

        Args:
            query: The user's natural language query
            func_name: The function Glyphh routed to
            func_def: The full function definition (name, description, parameters)
            context: Optional previous turns for multi-turn

        Returns:
            Dict of argument name → value
        """
        params = func_def.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])

        # Build a concise param description
        param_lines = []
        for pname, pdef in props.items():
            ptype = pdef.get("type", "any")
            pdesc = pdef.get("description", "")
            req = "(required)" if pname in required else "(optional)"
            default = f", default={pdef['default']}" if "default" in pdef else ""
            enum = f", enum={pdef['enum']}" if "enum" in pdef else ""
            param_lines.append(f"  - {pname}: {ptype} {req} — {pdesc}{default}{enum}")

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

        messages = [{"role": "system", "content": system}]
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": user_msg})

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=512,
            )
            text = resp.choices[0].message.content.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            return json.loads(text)
        except Exception as e:
            return {}

    def extract_multi_call_args(
        self,
        query: str,
        func_names: list[str],
        func_defs: list[dict],
        context: list[dict] | None = None,
    ) -> list[dict]:
        """Extract arguments for multiple function calls (parallel/parallel_multiple).

        Returns list of {func_name: {args}} dicts.
        """
        func_blocks = []
        for fname, fdef in zip(func_names, func_defs):
            params = fdef.get("parameters", {})
            props = params.get("properties", {})
            required = params.get("required", [])
            param_lines = []
            for pname, pdef in props.items():
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

        user_msg = f"Functions:\n" + "\n".join(func_blocks) + f"\n\nQuery: {query}"

        messages = [{"role": "system", "content": system}]
        if context:
            messages.extend(context)
        messages.append({"role": "user", "content": user_msg})

        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=0.0,
                max_tokens=1024,
            )
            text = resp.choices[0].message.content.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            return json.loads(text)
        except Exception:
            return [{name: {}} for name in func_names]

    def multi_turn_step(
        self,
        query: str,
        func_name: str,
        func_def: dict,
        conversation: list[dict],
        state_summary: str = "",
    ) -> dict:
        """Handle a single multi-turn step.

        Glyphh already picked the function. LLM extracts args
        given the conversation history and current state.
        """
        return self.extract_args(
            query=query,
            func_name=func_name,
            func_def=func_def,
            context=conversation,
        )


class GeminiClient:
    """Google Gemini API client."""

    def __init__(self, model: str = "gemini-2.0-flash-lite"):
        self.model = model
        self._client = None

    @property
    def client(self):
        if self._client is None:
            import google.generativeai as genai
            genai.configure(api_key=os.environ.get("GOOGLE_API_KEY"))
            self._client = genai.GenerativeModel(self.model)
        return self._client

    def extract_args(
        self,
        query: str,
        func_name: str,
        func_def: dict,
        context: list[dict] | None = None,
    ) -> dict:
        """Extract function arguments using Gemini."""
        params = func_def.get("parameters", {})
        props = params.get("properties", {})
        required = params.get("required", [])

        param_lines = []
        for pname, pdef in props.items():
            ptype = pdef.get("type", "any")
            req = "(required)" if pname in required else "(optional)"
            param_lines.append(f"  - {pname}: {ptype} {req} — {pdef.get('description', '')}")

        prompt = (
            f"Extract function call arguments from this query.\n"
            f"Function: {func_name}\n"
            f"Parameters:\n" + "\n".join(param_lines) + "\n\n"
            f"Query: {query}\n\n"
            f"Return ONLY a JSON object with argument names and values. No explanation."
        )

        try:
            resp = self.client.generate_content(prompt)
            text = resp.text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            return json.loads(text)
        except Exception:
            return {}

    def extract_multi_call_args(self, query, func_names, func_defs, context=None):
        """Extract arguments for multiple function calls using Gemini."""
        # Simplified — same pattern as OpenAI
        results = []
        for fname, fdef in zip(func_names, func_defs):
            args = self.extract_args(query, fname, fdef, context)
            results.append({fname: args})
        return results

    def multi_turn_step(self, query, func_name, func_def, conversation, state_summary=""):
        return self.extract_args(query, func_name, func_def, conversation)
