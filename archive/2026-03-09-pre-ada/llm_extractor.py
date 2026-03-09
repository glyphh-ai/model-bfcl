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

    def extract(self, query: str, func_def: dict, language: str = "python") -> dict[str, Any]:
        """Extract arguments from a query given a function schema.

        Same interface as ArgumentExtractor.extract().

        Args:
            query:    Natural language query string.
            func_def: BFCL function definition dict with 'parameters' schema.
            language: Target language ("python", "java", "javascript").

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

        lang_lower = language.lower()
        if lang_lower == "java":
            lang_rules = (
                "- ALL values MUST be Java string representations.\n"
                "- Booleans: use \"true\" or \"false\" (lowercase strings).\n"
                "- Integers: use plain number strings like \"42\".\n"
                "- Longs: append L suffix, e.g. \"42L\".\n"
                "- Floats: append f suffix, e.g. \"3.14f\".\n"
                "- Doubles: plain decimal strings like \"3.14\".\n"
                "- Strings: use the plain string value (no extra quotes).\n"
                "- Arrays: use Java syntax like \"new int[]{1, 2, 3}\" or \"new String[]{\\\"a\\\", \\\"b\\\"}\".\n"
                "- ArrayLists: use \"new ArrayList<>(Arrays.asList(1, 2, 3))\" or with strings \"new ArrayList<>(Arrays.asList(\\\"a\\\", \\\"b\\\"))\".\n"
                "- HashMaps: use 'new HashMap<String, String>() {{ put(\"key\", \"value\"); }}'.\n"
                "- For typed references (Path, File, Enum constants, etc.), use the Java constructor/constant syntax "
                "exactly as it would appear in Java code, e.g. \"new Path('/backup/data.txt')\" or \"WinReg.HKEY_LOCAL_MACHINE\".\n"
                "- When the schema type is 'any', the value should still be a string.\n"
            )
        elif lang_lower == "javascript":
            lang_rules = (
                "- ALL values MUST be JavaScript string representations.\n"
                "- Booleans: use \"true\" or \"false\" (lowercase strings).\n"
                "- Numbers: use plain number strings like \"42\" or \"3.14\".\n"
                "- Strings: use the plain string value.\n"
                "- Arrays: use JavaScript syntax like \"[1, 2, 3]\" or \"['a', 'b']\".\n"
                "- Objects: use JavaScript syntax like \"{key: 'value'}\".\n"
                "- For typed references, use JavaScript-style constructors or constants.\n"
            )
        else:
            lang_rules = (
                "- Use the exact types specified in the schema (integer, number, string, boolean, array, dict).\n"
                "- For mathematical expressions, use Python syntax: ** for exponentiation (not ^), "
                "e.g. 'x**2' not 'x^2', '3x**2 + 2x - 1' not '3x^2 + 2x - 1'.\n"
            )

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
            f"{lang_rules}"
            "- For enum parameters, pick the value that best matches the query context, or the first enum value as default.\n"
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

            # For Java/JS, skip type coercion — keep values as strings
            if lang_lower in ("java", "javascript"):
                return {k: v for k, v in result.items() if k in props}

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

    def extract_multi_turn(
        self,
        query: str,
        func_defs: list[dict],
        conversation_history: list[str],
        initial_config: dict | None = None,
        current_state: dict | None = None,
    ) -> list[dict]:
        """Extract arguments for multiple functions in a multi-turn context.

        Unlike extract(), this method:
          - Receives ALL functions routed for this turn (not one at a time)
          - Receives conversation history for context (file names, directories, etc.)
          - Receives initial_config so exact file/folder names are known
          - Receives current_state (CWD etc.) from CognitiveLoop
          - Returns a list of {bare_func_name: {args}} dicts ready for gorilla output

        Args:
            query: Current turn's user query.
            func_defs: Function definitions routed for this turn (class-prefixed names).
            conversation_history: Previous turn queries for context.
            initial_config: Initial state of API instances (filesystem tree, etc.).
            current_state: Current CognitiveLoop state (e.g. {"primary": "workspace/document"}).

        Returns:
            List of {bare_func_name: {param: value}} dicts.
        """
        schemas = []
        for fd in func_defs:
            params = fd.get("parameters", {})
            # Use bare name (strip class prefix) for the output
            full_name = fd.get("name", "")
            bare_name = full_name.split(".")[-1] if "." in full_name else full_name
            schemas.append({
                "name": bare_name,
                "full_name": full_name,
                "description": fd.get("description", ""),
                "parameters": params,
            })

        schema_str = json.dumps(schemas, indent=2)

        # Build conversation context
        context = ""
        if conversation_history:
            context = "Previous conversation turns:\n"
            for ti, prev_q in enumerate(conversation_history):
                context += f"  Turn {ti + 1}: {prev_q}\n"
            context += "\n"

        # Build filesystem/state context from initial_config
        state_context = ""
        if initial_config:
            state_context = "Initial system state (use EXACT names from here):\n"
            for cls_name, config in initial_config.items():
                if "root" in config:
                    state_context += f"  {cls_name} filesystem:\n"
                    state_context += self._format_fs_tree(config["root"], indent=4)
                else:
                    state_context += f"  {cls_name}: {json.dumps(config, indent=2)}\n"
            state_context += "\n"

        # Current state from CognitiveLoop (e.g. CWD)
        cwd_context = ""
        if current_state:
            primary = current_state.get("primary", "")
            if primary:
                cwd_context = f"CURRENT WORKING DIRECTORY: {primary}\n\n"

        prompt = (
            "You are extracting function arguments for a multi-turn tool-calling system.\n"
            "The HDC model has already decided WHICH functions to call. Your job is to "
            "extract the correct argument values from the conversation.\n\n"
            f"{state_context}"
            f"{cwd_context}"
            f"{context}"
            f"Current query: {query}\n\n"
            f"Functions to call (in order):\n```json\n{schema_str}\n```\n\n"
            "Return ONLY a JSON array of function call objects. Each object has the "
            "bare function name as key and an object of arguments as value.\n"
            "Example: [{\"cd\": {\"folder\": \"documents\"}}, {\"mv\": {\"source\": \"a.txt\", \"destination\": \"b\"}}]\n\n"
            "IMPORTANT RULES:\n"
            "- You MUST call exactly these functions in this order. Do not add or remove functions.\n"
            "- Extract argument values from the current query AND conversation context.\n"
            "- Use EXACT file and directory names from the initial system state — do NOT guess or modify names.\n"
            "- cd is RELATIVE to the current working directory shown above. Use '..' to go up.\n"
            "- File names, directory names, and paths often come from earlier turns.\n"
            "- For parameters not mentioned anywhere, use the default value from the schema.\n"
            "- If no default, use a sensible default (0 for numbers, \"\" for strings, false for booleans).\n"
            "- Use the exact parameter types specified in the schema.\n\n"
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
                return self._fallback_extract(query, func_defs)

            # Build props map for type coercion
            props_map = {}
            for fd in func_defs:
                full_name = fd.get("name", "")
                bare_name = full_name.split(".")[-1] if "." in full_name else full_name
                params = fd.get("parameters", {})
                props_map[bare_name] = (
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
            return coerced if coerced else self._fallback_extract(query, func_defs)

        except Exception:
            return self._fallback_extract(query, func_defs)

    def route_and_extract_turn(
        self,
        query: str,
        all_func_defs: list[dict],
        conversation_history: list[str],
        previous_calls: list[list[dict]],
        initial_config: dict | None = None,
    ) -> list[dict]:
        """Full function selection + argument extraction for one multi-turn step.

        Unlike extract_multi_turn(), this method decides WHICH functions to call
        (not just what args). HDC handles class routing; LLM handles per-turn
        function selection within the available functions.

        Args:
            query: Current turn's user query.
            all_func_defs: ALL available function definitions for the entry.
            conversation_history: Previous turn queries.
            previous_calls: Previous turns' function calls [[{fn: args}, ...], ...]
            initial_config: Initial state of API instances.

        Returns:
            List of {bare_func_name: {param: value}} dicts.
        """
        # Build function schema list with bare names
        schemas = []
        props_map = {}
        for fd in all_func_defs:
            full_name = fd.get("name", "")
            bare_name = full_name.split(".")[-1] if "." in full_name else full_name
            params = fd.get("parameters", {})
            schemas.append({
                "name": bare_name,
                "description": fd.get("description", ""),
                "parameters": params,
            })
            props_map[bare_name] = (
                params.get("properties", {}) if isinstance(params, dict) else {}
            )

        schema_str = json.dumps(schemas, indent=2)

        # Build state context
        state_context = ""
        if initial_config:
            state_context = "Initial system state (use EXACT names from here):\n"
            for cls_name, config in initial_config.items():
                if "root" in config:
                    state_context += f"  {cls_name} filesystem:\n"
                    state_context += self._format_fs_tree(config["root"], indent=4)
                else:
                    state_context += f"  {cls_name}: {json.dumps(config)}\n"
            state_context += "\n"

        # Compute current working directory by simulating cd calls
        cwd = self._compute_cwd(initial_config, previous_calls)

        # Build conversation + action history
        history = ""
        if conversation_history or previous_calls:
            history = "Conversation history:\n"
            for ti in range(len(conversation_history)):
                history += f"  Turn {ti + 1} query: {conversation_history[ti]}\n"
                if ti < len(previous_calls) and previous_calls[ti]:
                    calls_str = ", ".join(
                        f"{fn}({', '.join(f'{k}={repr(v)}' for k, v in args.items())})"
                        for call in previous_calls[ti]
                        for fn, args in call.items()
                        if isinstance(args, dict)
                    )
                    history += f"  Turn {ti + 1} calls: {calls_str}\n"
            history += "\n"

        cwd_context = f"CURRENT WORKING DIRECTORY: {cwd}\n\n" if cwd else ""

        prompt = (
            "You are a function-calling agent in a multi-turn conversation.\n"
            "Given the current query, conversation history, and available functions, "
            "decide which functions to call and with what arguments.\n\n"
            f"{state_context}"
            f"{cwd_context}"
            f"{history}"
            f"Current query: {query}\n\n"
            f"Available functions:\n```json\n{schema_str}\n```\n\n"
            "Return ONLY a JSON array of function call objects. Each object has the "
            "function name as key and an object of arguments as value.\n"
            "Example: [{\"cd\": {\"folder\": \"documents\"}}, {\"mkdir\": {\"dir_name\": \"new\"}}, "
            "{\"mv\": {\"source\": \"a.txt\", \"destination\": \"new\"}}]\n\n"
            "CRITICAL RULES:\n"
            "1. STATE PERSISTS ACROSS TURNS. You are currently in the directory shown above. "
            "Do NOT cd to a directory you are already in.\n"
            "2. cd is RELATIVE — it moves one level at a time from the CURRENT directory. "
            "Use cd('..') to go up. You cannot cd to an absolute path.\n"
            "3. Use EXACT file and directory names from the initial system state. "
            "If the query says 'document folder' but the filesystem has 'documents', use 'documents'.\n"
            "4. Track file locations: if a file was moved in a previous turn, it is now in the "
            "new location. You must cd to that location to operate on it.\n"
            "5. If the query asks to create a directory, include mkdir BEFORE moving files into it.\n"
            "6. Every turn MUST have at least one function call. If unsure what the query needs, "
            "pick the most relevant function(s) from the available list.\n"
            "7. Include ALL required parameters for each function.\n"
            "8. Order matters: cd before operations in that directory, mkdir before mv into it.\n"
            "9. For parameters not mentioned, use the default from schema or sensible defaults.\n\n"
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

    def _fallback_extract(self, query: str, func_defs: list[dict]) -> list[dict]:
        """Fallback: extract per-function when multi-turn fails."""
        calls = []
        for fd in func_defs:
            full_name = fd.get("name", "")
            bare_name = full_name.split(".")[-1] if "." in full_name else full_name
            args = self.extract(query, fd)
            calls.append({bare_name: args})
        return calls

    @staticmethod
    def _compute_cwd(
        initial_config: dict | None,
        previous_calls: list[list[dict]],
    ) -> str:
        """Simulate cd calls to compute current working directory."""
        # Find root directory name from initial_config
        root = ""
        if initial_config:
            for cls_name, config in initial_config.items():
                if "root" in config and isinstance(config["root"], dict):
                    root = list(config["root"].keys())[0] if config["root"] else ""
                    break

        cwd_parts = [root] if root else []

        # Replay all cd calls from previous turns
        for turn_calls in previous_calls:
            for call in turn_calls:
                if isinstance(call, dict):
                    for fn, args in call.items():
                        if fn == "cd" and isinstance(args, dict):
                            folder = args.get("folder", "")
                            if folder == ".." and cwd_parts:
                                cwd_parts.pop()
                            elif folder and folder != ".":
                                cwd_parts.append(folder)

        return "/".join(cwd_parts) if cwd_parts else "/"

    @staticmethod
    def _format_fs_tree(node: dict, indent: int = 0) -> str:
        """Format a filesystem tree from initial_config for the prompt."""
        lines = []
        prefix = " " * indent
        for name, info in node.items():
            if isinstance(info, dict) and info.get("type") == "directory":
                contents = info.get("contents", {})
                child_names = list(contents.keys()) if contents else []
                lines.append(f"{prefix}{name}/ (directory, contains: {child_names})\n")
                if contents:
                    lines.append(LLMArgumentExtractor._format_fs_tree(contents, indent + 2))
            elif isinstance(info, dict) and info.get("type") == "file":
                lines.append(f"{prefix}{name} (file)\n")
            elif isinstance(info, dict):
                # Nested directory without type field (root level)
                lines.append(f"{prefix}{name}/\n")
                lines.append(LLMArgumentExtractor._format_fs_tree(info, indent + 2))
        return "".join(lines)

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
