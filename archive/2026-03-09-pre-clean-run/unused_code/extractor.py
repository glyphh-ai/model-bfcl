"""
Schema-guided argument extractor for BFCL. No LLM.

Extracts argument values from natural language queries using:
  1. Named-anchor matching — "base of 10", "a = 1", "radius is 5"
  2. Enum matching        — fuzzy match against allowed values
  3. Positional numbers   — assign extracted numbers left-to-right to params
  4. Boolean triggers     — yes/no/true/false and flag words
  5. String extraction    — quoted strings, quoted values, last noun phrase

Exports:
  ArgumentExtractor.extract(query, func_def) → dict[str, Any]
"""

from __future__ import annotations

import re
from typing import Any


class ArgumentExtractor:
    """Schema-guided argument extraction from NL queries. No LLM."""

    def extract(self, query: str, func_def: dict) -> dict[str, Any]:
        """Extract arguments from a query given a function schema.

        Args:
            query:    Natural language query string.
            func_def: BFCL function definition dict with 'parameters' schema.

        Returns:
            Dict mapping parameter names to extracted values.
            Only includes parameters for which a value was found.
        """
        params   = func_def.get("parameters", {})
        props    = params.get("properties", {}) if isinstance(params, dict) else {}
        required = params.get("required", []) if isinstance(params, dict) else []

        if not props:
            return {}

        result: dict[str, Any] = {}

        # Pass 1: named-anchor and enum extraction (highest confidence)
        for pname, pschema in props.items():
            value = self._extract_named(query, pname, pschema)
            if value is not None:
                result[pname] = value

        # Pass 2: fill remaining required params positionally from numbers
        missing_numeric = [
            pname for pname in required
            if pname not in result
            and props.get(pname, {}).get("type") in ("integer", "number", "float")
        ]
        if missing_numeric:
            self._fill_positional_numbers(query, missing_numeric, props, result)

        # Pass 3: fill remaining required string params
        for pname in required:
            if pname not in result:
                pschema = props.get(pname, {})
                ptype   = pschema.get("type", "string")
                if ptype == "string" and "enum" not in pschema:
                    value = self._extract_string_fallback(query, pname, pschema)
                    if value is not None:
                        result[pname] = value

        return result

    # ── Per-type extraction ──────────────────────────────────────────────

    def _extract_named(self, query: str, pname: str, pschema: dict) -> Any:
        """Try to extract a value by matching the parameter name near a value."""
        ptype = pschema.get("type", "string")

        # Enum values: try fuzzy match regardless of type
        if "enum" in pschema:
            v = self._match_enum(query, pschema["enum"])
            if v is not None:
                return v

        if ptype in ("integer", "int"):
            return self._extract_number_near_name(query, pname, integer=True)

        if ptype in ("number", "float"):
            return self._extract_number_near_name(query, pname, integer=False)

        if ptype == "boolean":
            return self._extract_boolean(query, pname, pschema)

        if ptype == "array":
            return self._extract_array(query, pname, pschema)

        if ptype == "string":
            # Quoted strings first
            v = self._extract_quoted(query)
            if v is not None:
                return v
            return self._extract_number_near_name(query, pname, integer=False, as_string=True)

        return None

    def _extract_number_near_name(
        self,
        query: str,
        pname: str,
        integer: bool,
        as_string: bool = False,
    ) -> Any:
        """Extract a number that appears near the parameter name in the query.

        Handles patterns:
          - "base of 10", "radius is 5", "a = 1"
          - "base 10", "radius=5"
          - "a=1, b=-3, c=2"  (equation-style)
        """
        q_lower = query.lower()
        clean_name = re.sub(r"[^a-z0-9 ]", "", pname.lower().replace("_", " "))

        # Pattern: param_name OP number
        # OP = of, is, =, :, (, space
        for part in clean_name.split():
            pattern = rf"\b{re.escape(part)}\s*(?:of|is|=|:|\()?\s*(-?\d+\.?\d*)"
            m = re.search(pattern, q_lower)
            if m:
                raw = m.group(1)
                return self._coerce(raw, integer, as_string)

        # Pattern: number OP param_name  ("4 for base", "5 as height")
        for part in clean_name.split():
            pattern = rf"(-?\d+\.?\d*)\s*(?:for|as|is)\s*{re.escape(part)}\b"
            m = re.search(pattern, q_lower)
            if m:
                return self._coerce(m.group(1), integer, as_string)

        return None

    def _fill_positional_numbers(
        self,
        query: str,
        param_names: list[str],
        props: dict,
        result: dict,
    ) -> None:
        """Assign all numbers in the query positionally to missing numeric params."""
        numbers = re.findall(r"-?\d+\.?\d*", query)
        idx = 0
        for pname in param_names:
            if idx >= len(numbers):
                break
            ptype = props.get(pname, {}).get("type", "integer")
            integer = ptype == "integer"
            result[pname] = self._coerce(numbers[idx], integer)
            idx += 1

    def _match_enum(self, query: str, enum_values: list) -> Any:
        """Fuzzy match query words against enum values (case-insensitive)."""
        q_lower = query.lower()
        for val in enum_values:
            if isinstance(val, str) and val.lower() in q_lower:
                return val
        # Try partial word match
        words = re.findall(r"\b\w+\b", q_lower)
        for val in enum_values:
            if isinstance(val, str):
                val_lower = val.lower()
                if any(w in val_lower or val_lower in w for w in words if len(w) > 2):
                    return val
        return None

    def _extract_boolean(self, query: str, pname: str, pschema: dict) -> bool | None:
        """Extract a boolean value from the query."""
        q_lower = query.lower()
        clean_name = pname.lower().replace("_", " ")

        # Look for explicit true/false/yes/no near the param name
        for part in clean_name.split():
            pattern = rf"\b{re.escape(part)}\s*(?:is|=|:)?\s*(true|false|yes|no)\b"
            m = re.search(pattern, q_lower)
            if m:
                return m.group(1) in ("true", "yes")

        # Presence of the param name in a positive context
        if clean_name in q_lower:
            return True

        # Check default value
        default = pschema.get("default")
        if default is not None and isinstance(default, bool):
            return default

        return None

    def _extract_array(self, query: str, pname: str, pschema: dict) -> list | None:
        """Extract an array value (comma-separated or 'and'-joined numbers/strings)."""
        item_type = pschema.get("items", {}).get("type", "string")

        if item_type in ("integer", "number", "float"):
            # All numbers in the query
            numbers = re.findall(r"-?\d+\.?\d*", query)
            if numbers:
                if item_type == "integer":
                    return [int(float(n)) for n in numbers]
                return [float(n) for n in numbers]

        # Try to extract a sequence of quoted strings
        quoted = re.findall(r'"([^"]*)"', query)
        if quoted:
            return quoted

        return None

    def _extract_quoted(self, query: str) -> str | None:
        """Extract first quoted string from query."""
        m = re.search(r'"([^"]*)"', query)
        if m:
            return m.group(1)
        m = re.search(r"'([^']+)'", query)
        if m:
            return m.group(1)
        return None

    def _extract_string_fallback(self, query: str, pname: str, pschema: dict) -> str | None:
        """Last resort: extract a meaningful string value."""
        # Try to find a proper noun (capitalized word not at sentence start)
        words = query.split()
        for i, w in enumerate(words):
            if i > 0 and w[0].isupper() and len(w) > 1 and w.isalpha():
                return w
        return None

    # ── Utilities ────────────────────────────────────────────────────────

    @staticmethod
    def _coerce(raw: str, integer: bool, as_string: bool = False) -> Any:
        """Convert a raw number string to the appropriate Python type."""
        if as_string:
            return raw
        try:
            if integer:
                return int(float(raw))
            return float(raw)
        except (ValueError, TypeError):
            return None
