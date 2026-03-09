"""Glyphh Ada — Tool Execution handler.

Hybrid HDC + LLM architecture:
  - HDC scorer: routes query → candidate functions (no hallucination)
  - CognitiveLoop: tracks conversation state (CWD, recent actions, entities)
  - LLM (Claude): extracts arguments given routed functions + state context

The HDC acts as a guardrail — the LLM only sees functions the scorer
deems relevant, so it can't hallucinate random function calls. The LLM
just fills in the blanks (argument values from NL).

Usage:
    handler = ToolExecutionHandler(scorer, func_defs)
    calls = handler.handle(query, prior_turns, initial_config)
    # → ["cd(folder='document')", "mkdir(dir_name='temp')", ...]
"""

from __future__ import annotations

import importlib.util
import json
import re
from pathlib import Path
from typing import Any

from .llm_extractor import LLMArgumentExtractor

# Explicit import to avoid module shadowing
_DIR = Path(__file__).parent
_intent_spec = importlib.util.spec_from_file_location("_te_intent", _DIR / "intent.py")
_intent_mod = importlib.util.module_from_spec(_intent_spec)
_intent_spec.loader.exec_module(_intent_mod)
_VERB_MAP = _intent_mod._VERB_MAP
extract_entities = _intent_mod.extract_entities


# ── Shared LLM extractor (reused across handlers) ────────────────────

_llm_extractor: LLMArgumentExtractor | None = None


def _get_llm_extractor() -> LLMArgumentExtractor:
    global _llm_extractor
    if _llm_extractor is None:
        _llm_extractor = LLMArgumentExtractor()
    return _llm_extractor


# ── Handler ───────────────────────────────────────────────────────────

class ToolExecutionHandler:
    """Hybrid HDC routing + LLM arg extraction handler."""

    def __init__(self, scorer, func_defs: list[dict]):
        self.scorer = scorer
        self.func_defs_list = func_defs
        self.func_defs = {fd["name"]: fd for fd in func_defs}

        # bare_name → full_name mapping
        self._bare_to_full: dict[str, str] = {}
        for name in self.func_defs:
            bare = name.split(".")[-1] if "." in name else name
            self._bare_to_full[bare] = name

    def handle(
        self,
        query: str,
        prior_turns: list[dict] | None = None,
        initial_config: dict | None = None,
        current_cwd: str | None = None,
    ) -> list[str]:
        """Produce ordered list of call strings.

        Args:
            query: Current turn's NL query.
            prior_turns: [{query, ground_truth}, ...] from earlier turns.
            initial_config: Initial state of API instances (filesystem tree, etc.).
            current_cwd: Current working directory from CognitiveLoop state.

        Returns:
            List of call strings like ["cd(folder='doc')", "mv(source='a', destination='b')"]
        """
        llm = _get_llm_extractor()

        # Build conversation history and previous calls from prior turns
        conversation_history = []
        previous_calls = []
        if prior_turns:
            for pt in prior_turns:
                conversation_history.append(pt["query"])
                # Parse ground truth call strings into {func: {args}} dicts
                calls = _parse_call_strings(pt["ground_truth"])
                previous_calls.append(calls)

        # Use LLM for full routing + extraction (HDC routing TBD)
        result = llm.route_and_extract_turn(
            query=query,
            all_func_defs=self.func_defs_list,
            conversation_history=conversation_history,
            previous_calls=previous_calls,
            initial_config=initial_config,
            current_cwd=current_cwd,
        )

        # Format as call strings
        return [_format_call(fn, args) for call in result for fn, args in call.items()]


def _parse_call_strings(call_strings: list[str]) -> list[dict]:
    """Parse ground truth call strings into {func: {args}} dicts.

    E.g., "cd(folder='document')" → {"cd": {"folder": "document"}}
    """
    calls = []
    for cs in call_strings:
        m = re.match(r"(\w+)\((.*)\)$", cs, re.DOTALL)
        if not m:
            continue
        func_name = m.group(1)
        args_str = m.group(2).strip()
        if not args_str:
            calls.append({func_name: {}})
            continue

        # Parse kwargs
        args = {}
        # Use a simple approach: split on comma-separated key=value pairs
        # Handle nested quotes carefully
        try:
            # Build a dict by eval-ing kwargs (safe since ground truth is trusted)
            args = eval(f"dict({args_str})")
        except Exception:
            pass
        calls.append({func_name: args})
    return calls


def _format_call(func_name: str, args: Any) -> str:
    """Format as func_name(arg1='val1', arg2='val2')."""
    if not isinstance(args, dict) or not args:
        return f"{func_name}()"
    parts = []
    for k, v in args.items():
        if isinstance(v, str):
            parts.append(f"{k}='{v}'")
        elif isinstance(v, bool):
            parts.append(f"{k}={v}")
        else:
            parts.append(f"{k}={v}")
    return f"{func_name}({','.join(parts)})"
