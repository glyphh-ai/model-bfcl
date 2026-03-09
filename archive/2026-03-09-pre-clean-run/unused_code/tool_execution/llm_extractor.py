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
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
import anthropic

# Load .env from bfcl directory
_ENV_PATH = Path(__file__).parent.parent / ".env"
load_dotenv(_ENV_PATH)


def _extract_json(text: str) -> Any:
    """Extract JSON array or object from LLM response text.

    Handles: raw JSON, ```json fences, reasoning text before/after JSON.
    """
    text = text.strip()

    # Try raw parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Strip markdown code fences
    if "```" in text:
        # Find content between first ``` and last ```
        parts = text.split("```")
        for part in parts[1:]:
            # Skip the language tag line
            content = part.split("\n", 1)[1] if "\n" in part else part
            content = content.strip()
            try:
                return json.loads(content)
            except json.JSONDecodeError:
                continue

    # Find JSON array or object in the text
    # Look for [ ... ] or { ... }
    import re
    for pattern in [r'\[[\s\S]*\]', r'\{[\s\S]*\}']:
        m = re.search(pattern, text)
        if m:
            try:
                return json.loads(m.group())
            except json.JSONDecodeError:
                continue

    return None


class LLMArgumentExtractor:
    """Extract function arguments using Claude."""

    def __init__(self, model: str = "claude-sonnet-4-20250514", temperature: float = 0.0) -> None:
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

    @staticmethod
    def _to_anthropic_tool(fd: dict) -> dict:
        """Convert BFCL func def to Anthropic native tool schema."""
        full_name = fd.get("name", "")
        bare_name = full_name.split(".")[-1] if "." in full_name else full_name
        params = fd.get("parameters", {})

        # Convert BFCL "dict" type to JSON Schema "object"
        input_schema = dict(params)
        if input_schema.get("type") == "dict":
            input_schema["type"] = "object"
        # Ensure properties exist
        if "properties" not in input_schema:
            input_schema["properties"] = {}
        # Convert property types too
        for prop_name, prop in input_schema.get("properties", {}).items():
            if isinstance(prop, dict) and prop.get("type") == "dict":
                prop["type"] = "object"
            if isinstance(prop, dict) and prop.get("type") == "array":
                items = prop.get("items", {})
                if isinstance(items, dict) and items.get("type") == "dict":
                    items["type"] = "object"

        return {
            "name": bare_name,
            "description": fd.get("description", ""),
            "input_schema": input_schema,
        }

    def route_and_extract_turn(
        self,
        query: str,
        all_func_defs: list[dict],
        conversation_history: list[str],
        previous_calls: list[list[dict]],
        initial_config: dict | None = None,
        current_cwd: str | None = None,
    ) -> list[dict]:
        """Full function selection + argument extraction using native tool_use.

        Uses Anthropic's native tool calling (not JSON prompt) for structured
        output. HDC handles class routing; LLM picks from candidates via tools.

        Returns:
            List of {bare_func_name: {param: value}} dicts.
        """
        # Build native Anthropic tools
        tools = []
        seen_names = set()
        for fd in all_func_defs:
            tool = self._to_anthropic_tool(fd)
            if tool["name"] not in seen_names:
                tools.append(tool)
                seen_names.add(tool["name"])

        if not tools:
            return []

        # Build system prompt with state context
        system_parts = []

        system_parts.append(
            "You are an expert in composing functions. You are given a question and a set of possible functions. "
            "Based on the question, you will need to make one or more function/tool calls to achieve the purpose. "
            "If none of the functions can be used, point it out and do not make any tool calls.\n\n"
            "At each turn, you should try your best to complete the tasks requested by the user within the current turn. "
            "Continue to output functions to call until you have fulfilled the user's request to the best of your ability. "
            "Only call functions that are directly needed — do not add extra exploratory calls like pwd(), ls(), or find() "
            "unless the user specifically asks for them."
        )

        # Filesystem state context
        if initial_config:
            state_context = "\n\nSystem state (use EXACT names from here):\n"
            for cls_name, config in initial_config.items():
                if "root" in config:
                    root = config["root"]
                    root_name = list(root.keys())[0] if root else ""
                    root_info = root.get(root_name, {})
                    contents = root_info.get("contents", root_info) if isinstance(root_info, dict) else {}
                    state_context += f"  {cls_name} filesystem (contents of {root_name}/):\n"
                    state_context += self._format_fs_tree(contents, indent=4)
                else:
                    state_context += f"  {cls_name}: {json.dumps(config)}\n"
            system_parts.append(state_context)

        # CWD context
        cwd = current_cwd if current_cwd is not None else self._compute_cwd(initial_config, previous_calls)
        if cwd:
            system_parts.append(
                f"\nCURRENT WORKING DIRECTORY: {cwd}\n"
                f"You are ALREADY inside '{cwd}'. Do NOT cd into '{cwd}'."
            )

        system_prompt = "\n".join(system_parts)

        # Build messages: conversation history + current query
        messages = []
        for ti in range(len(conversation_history)):
            messages.append({"role": "user", "content": conversation_history[ti]})
            # Add previous turn's calls as assistant tool_use blocks
            if ti < len(previous_calls) and previous_calls[ti]:
                assistant_content = []
                tool_results = []
                for ci, call in enumerate(previous_calls[ti]):
                    for fn, args in call.items():
                        tool_use_id = f"call_{ti}_{ci}"
                        assistant_content.append({
                            "type": "tool_use",
                            "id": tool_use_id,
                            "name": fn,
                            "input": args if isinstance(args, dict) else {},
                        })
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": tool_use_id,
                            "content": "OK",
                        })
                if assistant_content:
                    messages.append({"role": "assistant", "content": assistant_content})
                    messages.append({"role": "user", "content": tool_results})

        # Current turn query
        messages.append({"role": "user", "content": query})

        try:
            response = self._client.messages.create(
                model=self._model,
                temperature=self._temperature,
                max_tokens=4096,
                system=system_prompt,
                tools=tools,
                messages=messages,
            )
            self.total_input_tokens += response.usage.input_tokens
            self.total_output_tokens += response.usage.output_tokens
            self.total_calls += 1

            # Extract tool_use blocks from response
            calls = []
            for block in response.content:
                if block.type == "tool_use":
                    calls.append({block.name: block.input})

            return calls

        except Exception as e:
            import traceback
            traceback.print_exc()
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
