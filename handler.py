"""
BFCL Handler for Glyphh HDC.

This module implements the interface that the BFCL evaluation harness expects.
It can be used in two ways:

1. As a standalone runner against BFCL data files (for local testing)
2. As a handler class that can be integrated into the gorilla BFCL framework

The handler:
  - Takes BFCL test entries (query + function definitions)
  - Dynamically encodes all function definitions into HDC vectors
  - Encodes the query
  - Returns the best-matching function name via cosine similarity

For BFCL categories that require arguments (simple, multiple, parallel),
we pair Glyphh routing with an LLM for argument extraction (same as our
S3 strategy: Glyphh routes, LLM fills args).
"""

import json
import os
import time
from typing import Any

from encoder import ENCODER_CONFIG, encode_function, encode_query
from glyphh.core.types import Concept
from glyphh.core.ops import cosine_similarity
from glyphh.encoder import Encoder


class GlyphhBFCLHandler:
    """Glyphh HDC handler for BFCL evaluation."""

    def __init__(self, threshold: float = 0.45, use_llm_for_args: bool = True):
        self.threshold = threshold
        self.use_llm_for_args = use_llm_for_args
        self.encoder = Encoder(ENCODER_CONFIG)
        self._openai_client = None

    # ── Core routing ──

    def route(self, query: str, func_defs: list[dict]) -> dict[str, Any]:
        """Route a query to the best-matching function from the provided definitions.

        Args:
            query: Natural language user query.
            func_defs: List of BFCL function definition dicts.

        Returns:
            Dict with tool, confidence, latency_ms, top_k matches.
        """
        start = time.perf_counter()

        # Encode all function definitions
        func_glyphs = []
        func_names = []
        for fd in func_defs:
            concept_dict = encode_function(fd)
            concept = Concept(
                name=concept_dict["name"],
                attributes=concept_dict["attributes"],
            )
            glyph = self.encoder.encode(concept)
            func_glyphs.append(glyph)
            func_names.append(fd["name"])

        # Encode the query
        q_dict = encode_query(query, func_defs)
        q_concept = Concept(name="query", attributes=q_dict["attributes"])
        q_glyph = self.encoder.encode(q_concept)

        # Role-level weighted similarity
        role_weights = {
            "action": 1.0,
            "function_name": 0.9,
            "description": 0.8,
            "parameters": 0.6,
        }

        q_roles = self._extract_roles(q_glyph)

        scores = []
        for i, fg in enumerate(func_glyphs):
            f_roles = self._extract_roles(fg)
            score = self._weighted_similarity(q_roles, f_roles, role_weights)
            scores.append((score, i))

        scores.sort(key=lambda x: x[0], reverse=True)
        elapsed_ms = (time.perf_counter() - start) * 1000

        top_k = [
            {"function": func_names[i], "score": round(s, 4)}
            for s, i in scores[:3]
        ]

        if scores and scores[0][0] >= self.threshold:
            best_score, best_idx = scores[0]
            return {
                "tool": func_names[best_idx],
                "confidence": round(best_score, 4),
                "latency_ms": elapsed_ms,
                "top_k": top_k,
            }

        return {
            "tool": None,
            "confidence": round(scores[0][0], 4) if scores else 0.0,
            "latency_ms": elapsed_ms,
            "top_k": top_k,
        }

    def _extract_roles(self, glyph) -> dict:
        """Extract role vectors from a glyph."""
        roles = {}
        for layer in glyph.layers.values():
            for seg in layer.segments.values():
                for rname, rvec in seg.roles.items():
                    roles[rname] = rvec
        return roles

    def _weighted_similarity(self, q_roles: dict, f_roles: dict, weights: dict) -> float:
        """Compute weighted cosine similarity across roles."""
        weighted_sum = 0.0
        weight_total = 0.0
        for rname, w in weights.items():
            if rname in q_roles and rname in f_roles:
                sim = float(cosine_similarity(
                    q_roles[rname].data, f_roles[rname].data
                ))
                weighted_sum += sim * w
                weight_total += w
        return weighted_sum / weight_total if weight_total > 0 else 0.0

    # ── LLM argument extraction (optional, for S3-style eval) ──

    def _get_openai_client(self):
        if self._openai_client is None:
            import openai
            self._openai_client = openai.OpenAI(
                api_key=os.environ.get("OPENAI_API_KEY", "")
            )
        return self._openai_client

    def extract_args(self, query: str, func_def: dict, model: str = "gpt-4o-mini") -> dict:
        """Use an LLM to extract arguments for a routed function.

        This is the S3 strategy: Glyphh picks the function, LLM fills args.
        """
        schema = func_def.get("parameters", {})
        props = schema.get("properties", {})
        required = schema.get("required", [])

        param_lines = []
        for pname, pdef in props.items():
            req = " REQUIRED" if pname in required else ""
            ptype = pdef.get("type", "any")
            desc = pdef.get("description", "")
            enum = pdef.get("enum")
            extra = f" enum={enum}" if enum else ""
            param_lines.append(f"  {pname} ({ptype}{req}){extra}: {desc}")

        tool_desc = (
            f"{func_def['name']}: {func_def.get('description', '')}\n"
            f"Parameters:\n" + "\n".join(param_lines)
        )

        system = (
            f"You are an argument extractor. The function has been selected: {func_def['name']}\n\n"
            f"FUNCTION:\n{tool_desc}\n\n"
            "Given the user query, return ONLY the arguments as a JSON object.\n"
            "Rules:\n"
            "- Match the schema exactly (correct types, required fields)\n"
            "- Do NOT include optional fields with null values\n"
            "- Use enum values exactly as specified\n"
            "Respond with ONLY valid JSON."
        )

        client = self._get_openai_client()
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": query},
            ],
            max_tokens=300,
        )

        content = resp.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("\n", 1)[1] if "\n" in content else content[3:]
            if content.endswith("```"):
                content = content[:-3]
            content = content.strip()

        try:
            return json.loads(content)
        except json.JSONDecodeError:
            return {}

    # ── BFCL evaluation interface ──

    def predict(
        self,
        query: str,
        func_defs: list[dict],
        with_args: bool = False,
        llm_model: str = "gpt-4o-mini",
    ) -> list[dict]:
        """BFCL-compatible prediction interface.

        Returns a list of function calls in BFCL format:
            [{"func_name": {"param1": "val1", ...}}]

        For routing-only evaluation (irrelevance detection), returns
        empty list if no function matches.
        """
        result = self.route(query, func_defs)
        tool = result["tool"]

        if tool is None:
            return []

        if with_args and self.use_llm_for_args:
            # Find the matching function definition
            func_def = next((f for f in func_defs if f["name"] == tool), None)
            if func_def:
                args = self.extract_args(query, func_def, model=llm_model)
                return [{tool: args}]

        return [{tool: {}}]

    def decode_ast(self, result: list[dict]) -> list[dict]:
        """Convert prediction result to BFCL AST format.

        BFCL expects: [{"func1": {"param1": "val1"}}]
        Our predict() already returns this format.
        """
        return result

    def decode_execute(self, result: list[dict]) -> list[str]:
        """Convert prediction result to BFCL executable format.

        BFCL expects: ["func1(param1=val1, param2=val2)"]
        """
        calls = []
        for call in result:
            for func_name, args in call.items():
                if args:
                    arg_str = ", ".join(
                        f"{k}={repr(v)}" for k, v in args.items()
                    )
                    calls.append(f"{func_name}({arg_str})")
                else:
                    calls.append(f"{func_name}()")
        return calls
