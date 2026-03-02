"""
Local LLM client adapter — wraps glyphh.llm.LLMEngine to provide the
chat_complete() interface consumed by the BFCL eval pipeline.

Replaces llm_client.py (OpenAI/Anthropic/Gemini API clients).
"""

from __future__ import annotations

import json
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from glyphh.llm.engine import LLMEngine


class LocalLLMClient:
    """Wraps LLMEngine with the chat_complete() interface.

    Usage:
        from glyphh.llm import LLMEngine

        engine = LLMEngine()
        client = LocalLLMClient(engine)

        text = client.chat_complete(
            messages=[
                {"role": "system", "content": "Pick the function."},
                {"role": "user", "content": "List files in /tmp"},
            ],
            max_tokens=256,
        )
    """

    def __init__(self, engine: LLMEngine):
        self._engine = engine

    def chat_complete(
        self,
        messages: list[dict[str, str]],
        max_tokens: int = 512,
    ) -> str:
        """Generate a response from a conversation.

        Flattens the multi-message format into (system, user) pair
        for LLMEngine.generate(). Conversation history is concatenated
        into the user prompt.
        """
        system = ""
        conversation_parts: list[str] = []

        for msg in messages:
            role = msg.get("role", "")
            content = msg.get("content", "")
            if role == "system":
                system = content
            elif role == "user":
                conversation_parts.append(f"User: {content}")
            elif role == "assistant":
                conversation_parts.append(f"Assistant: {content}")

        user = "\n".join(conversation_parts)

        return self._engine.generate(
            system=system,
            user=user,
            max_tokens=max_tokens,
        )

    def extract_args(
        self,
        query: str,
        func_name: str,
        func_def: dict[str, Any],
        context: list[dict] | None = None,
    ) -> dict[str, Any]:
        """Extract arguments for a single function call via structured generation.

        Returns dict of {param_name: value}.
        """
        params = func_def.get("parameters", {}).get("properties", {})
        required = func_def.get("parameters", {}).get("required", [])

        # Build tool schema for structured generation
        tools = [{
            "type": "function",
            "function": {
                "name": "fill_args",
                "description": f"Extract arguments for {func_name}",
                "parameters": {
                    "type": "object",
                    "properties": params,
                    "required": required,
                },
            },
        }]

        system = (
            f"Extract the arguments for the function '{func_name}' from the user query. "
            f"Description: {func_def.get('description', '')}. "
            "Return ONLY the arguments object."
        )

        user_msg = f"Query: {query}"
        if context:
            ctx_lines = []
            for msg in context[-3:]:
                ctx_lines.append(f"{msg['role']}: {msg['content']}")
            user_msg = "\n".join(ctx_lines) + f"\n\nQuery: {query}"

        try:
            result = self._engine.structured_generate(
                system=system,
                user=user_msg,
                tools=tools,
                max_tokens=256,
            )
            return result.data if isinstance(result.data, dict) else {}
        except Exception:
            return {}

    def extract_multi_call_args(
        self,
        query: str,
        calls: list[dict[str, Any]],
        func_defs: dict[str, dict],
        context: list[dict] | None = None,
    ) -> list[dict[str, Any]]:
        """Extract arguments for multiple function calls.

        Args:
            calls: [{func_name: {partial_args}}, ...]
            func_defs: {func_name: func_def}

        Returns list of [{func_name: {full_args}}, ...]
        """
        result_calls = []
        for call in calls:
            for fname, partial_args in call.items():
                fdef = func_defs.get(fname, {})
                if not fdef:
                    result_calls.append({fname: partial_args})
                    continue

                # If args already fully extracted by CognitiveLoop, use them
                if partial_args and self._has_required_args(partial_args, fdef):
                    result_calls.append({fname: partial_args})
                    continue

                # Extract remaining args via LLM
                full_args = self.extract_args(query, fname, fdef, context)
                # Merge: partial_args take precedence
                merged = {**full_args, **partial_args}
                result_calls.append({fname: merged})

        return result_calls

    @staticmethod
    def _has_required_args(args: dict, func_def: dict) -> bool:
        """Check if all required parameters are present."""
        required = func_def.get("parameters", {}).get("required", [])
        return all(r in args for r in required)
