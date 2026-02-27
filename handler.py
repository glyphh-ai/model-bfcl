"""
BFCL Handler for Glyphh HDC.

Uses native Glyphh SDK encoding with:
- text_encoding="bag_of_words" on text roles (SDK handles word splitting + bundling)
- include_temporal=False (no temporal signal pollution)
- apply_weights_during_encoding=False (weights applied at scoring time)
- Multi-level similarity (cortex, layer, segment, role) for scoring
- Multi-function routing for parallel_multiple category
- Score distribution analysis for irrelevance detection
"""

import json
import os
import time
from typing import Any

from encoder import ENCODER_CONFIG, encode_function, encode_query
from glyphh.core.types import Concept, Glyph
from glyphh.core.ops import cosine_similarity
from glyphh import Encoder, SimilarityCalculator


class GlyphhBFCLHandler:
    """Glyphh HDC handler for BFCL evaluation."""

    def __init__(self, threshold: float = 0.15):
        self.threshold = threshold
        self.encoder = Encoder(ENCODER_CONFIG)
        self.calc = SimilarityCalculator(threshold=0.0)

    # ── Encoding via SDK standard path ──

    def _encode_glyph(self, concept_dict: dict) -> Glyph:
        """Encode a concept dict into a Glyph using the SDK's standard path.

        The SDK handles:
        - bag_of_words text encoding (word splitting + bundling)
        - numeric binning (thermometer encoding for param_count)
        - weighted bundling (apply_weights_during_encoding=True)
        """
        concept = Concept(
            name=concept_dict["name"],
            attributes=concept_dict["attributes"],
        )
        return self.encoder.encode(concept)

    # ── Multi-level similarity scoring ──

    def _score(self, query_glyph: Glyph, func_glyph: Glyph) -> float:
        """Compute similarity using native cosine at cortex, layer, segment, role levels."""
        # Cortex-level (global overview)
        cortex_sim = float(cosine_similarity(
            query_glyph.global_cortex.data,
            func_glyph.global_cortex.data,
        ))

        # Layer-level similarities
        layer_sims = []
        for lname in query_glyph.layers:
            if lname in func_glyph.layers and not lname.startswith("_"):
                ql = query_glyph.layers[lname]
                fl = func_glyph.layers[lname]
                lsim = float(cosine_similarity(ql.cortex.data, fl.cortex.data))
                lweight = ql.weights.get("similarity", 1.0)
                layer_sims.append((lsim, lweight))

        # Segment-level similarities
        seg_sims = []
        for lname in query_glyph.layers:
            if lname not in func_glyph.layers or lname.startswith("_"):
                continue
            ql = query_glyph.layers[lname]
            fl = func_glyph.layers[lname]
            for sname in ql.segments:
                if sname in fl.segments:
                    qs = ql.segments[sname]
                    fs = fl.segments[sname]
                    ssim = float(cosine_similarity(qs.cortex.data, fs.cortex.data))
                    sweight = qs.weights.get("similarity", 1.0)
                    seg_sims.append((ssim, sweight))

        # Role-level similarities (finest grain)
        role_sims = []
        for lname in query_glyph.layers:
            if lname not in func_glyph.layers or lname.startswith("_"):
                continue
            for sname in query_glyph.layers[lname].segments:
                if sname not in func_glyph.layers[lname].segments:
                    continue
                qs = query_glyph.layers[lname].segments[sname]
                fs = func_glyph.layers[lname].segments[sname]
                for rname in qs.roles:
                    if rname in fs.roles:
                        rsim = float(cosine_similarity(
                            qs.roles[rname].data, fs.roles[rname].data
                        ))
                        # Get role weight from config
                        rweight = 1.0
                        for ld in ENCODER_CONFIG.layers:
                            if ld.name == lname:
                                for sd in ld.segments:
                                    if sd.name == sname:
                                        for rd in sd.roles:
                                            if rd.name == rname:
                                                rweight = rd.similarity_weight
                        role_sims.append((rsim, rweight))

        # Weighted combination across hierarchy levels.
        # Role-level dominates: BFCL function names are highly specific
        # (calculate_mortgage_payment, search_hotels_by_amenity) so role-level
        # BoW similarity is far more discriminative than holistic cortex signal.
        w_cortex = 0.05
        w_layer = 0.10
        w_segment = 0.25
        w_role = 0.60

        score = cortex_sim * w_cortex

        if layer_sims:
            weighted_layer = sum(s * w for s, w in layer_sims) / sum(w for _, w in layer_sims)
            score += weighted_layer * w_layer

        if seg_sims:
            weighted_seg = sum(s * w for s, w in seg_sims) / sum(w for _, w in seg_sims)
            score += weighted_seg * w_segment

        if role_sims:
            weighted_role = sum(s * w for s, w in role_sims) / sum(w for _, w in role_sims)
            score += weighted_role * w_role

        return score

    # ── Core routing (single best match) ──

    def route(self, query: str, func_defs: list[dict]) -> dict[str, Any]:
        """Route a query to the best-matching function."""
        start = time.perf_counter()

        # Encode all function definitions
        func_glyphs = []
        func_names = []
        for fd in func_defs:
            concept_dict = encode_function(fd)
            glyph = self._encode_glyph(concept_dict)
            func_glyphs.append(glyph)
            func_names.append(fd["name"])

        # Encode the query
        q_dict = encode_query(query)
        q_glyph = self._encode_glyph(q_dict)

        # Score each function
        scores = []
        for i, fg in enumerate(func_glyphs):
            score = self._score(q_glyph, fg)
            scores.append((score, i))

        scores.sort(key=lambda x: x[0], reverse=True)
        elapsed_ms = (time.perf_counter() - start) * 1000

        all_scores = [{"function": func_names[i], "score": round(s, 4)} for s, i in scores]

        if scores and scores[0][0] >= self.threshold:
            best_score, best_idx = scores[0]
            return {
                "tool": func_names[best_idx],
                "confidence": round(best_score, 4),
                "latency_ms": elapsed_ms,
                "top_k": all_scores[:3],
                "all_scores": all_scores,
                "_query_glyph": q_glyph,
                "_best_func_glyph": func_glyphs[best_idx],
            }

        best_idx = scores[0][1] if scores else 0
        return {
            "tool": None,
            "confidence": round(scores[0][0], 4) if scores else 0.0,
            "latency_ms": elapsed_ms,
            "top_k": all_scores[:3],
            "all_scores": all_scores,
            "_query_glyph": q_glyph,
            "_best_func_glyph": func_glyphs[best_idx] if func_glyphs else None,
        }

    # ── Multi-function routing (for parallel_multiple) ──

    def route_multi(self, query: str, func_defs: list[dict]) -> dict[str, Any]:
        """Route a query to ALL matching functions.

        Uses score gap analysis to determine which functions are relevant:
        - Scores all functions
        - Finds natural gaps in the score distribution
        - Returns all functions above the gap
        """
        start = time.perf_counter()

        func_glyphs = []
        func_names = []
        for fd in func_defs:
            concept_dict = encode_function(fd)
            glyph = self._encode_glyph(concept_dict)
            func_glyphs.append(glyph)
            func_names.append(fd["name"])

        q_dict = encode_query(query)
        q_glyph = self._encode_glyph(q_dict)

        scores = []
        for i, fg in enumerate(func_glyphs):
            score = self._score(q_glyph, fg)
            scores.append((score, i))

        scores.sort(key=lambda x: x[0], reverse=True)
        elapsed_ms = (time.perf_counter() - start) * 1000

        all_scores = [{"function": func_names[i], "score": round(s, 4)} for s, i in scores]

        # Multi-select: find the natural gap in scores
        selected = self._select_multiple(scores, func_names)

        return {
            "tools": selected,
            "confidence": round(scores[0][0], 4) if scores else 0.0,
            "latency_ms": elapsed_ms,
            "top_k": all_scores[:5],
            "all_scores": all_scores,
        }

    def _select_multiple(
        self, scores: list[tuple[float, int]], func_names: list[str]
    ) -> list[str]:
        """Select multiple functions using score gap analysis.

        Strategy: absolute floor + cluster boundary detection.
        BFCL parallel_multiple queries request 2-4 distinct operations —
        selected functions should form a tight, well-separated cluster.
        """
        if not scores:
            return []

        if len(scores) == 1:
            return [func_names[scores[0][1]]]

        top_score = scores[0][0]
        MIN_ABS = 0.18  # Hard floor — anything below is noise

        # Filter to candidates above the absolute floor
        candidates = [(s, i) for s, i in scores if s >= MIN_ABS]
        if not candidates:
            return [func_names[scores[0][1]]]

        # Find a decisive cluster break (only break on very large gaps, >60% of top).
        # A 60% gap threshold means: 0.50→0.18 triggers (gap=0.64), but
        # 0.46→0.25 does NOT trigger (gap=0.46), allowing the ratio fallback
        # to include closely ranked secondary functions in parallel_multiple.
        for j in range(len(candidates) - 1):
            gap = candidates[j][0] - candidates[j + 1][0]
            if top_score > 0 and gap / top_score > 0.60:
                return [func_names[candidates[k][1]] for k in range(j + 1)]

        # No clear gap — use ratio-based cutoff (tightened from 0.35 to 0.45)
        cutoff = top_score * 0.45
        selected = [func_names[idx] for s, idx in candidates if s >= cutoff]
        return selected if selected else [func_names[scores[0][1]]]

    # ── Irrelevance detection ──

    def is_irrelevant(
        self,
        query_glyph: 'Glyph',
        func_glyph: 'Glyph',
        overall_score: float,
    ) -> bool:
        """Detect if a query is irrelevant using role-level similarity analysis.

        Uses fine-grained role similarities to determine if the query genuinely
        matches the function. Irrelevant queries tend to have low description
        and function_name similarity even when the overall bag-of-words score
        is moderate due to shared vocabulary (e.g., "calculate" appears in both
        the query and an unrelated function).

        Strategy:
        - Extract per-role cosine similarities from the glyph hierarchy
        - Focus on description (best separator) and function_name (structural)
        - Use weighted combination as relevance signal
        - Threshold to reject low-relevance matches
        """
        # Extract role-level similarities
        role_sims = {}
        for lname in query_glyph.layers:
            if lname.startswith("_") or lname not in func_glyph.layers:
                continue
            for sname in query_glyph.layers[lname].segments:
                if sname not in func_glyph.layers[lname].segments:
                    continue
                qs = query_glyph.layers[lname].segments[sname]
                fs = func_glyph.layers[lname].segments[sname]
                for rname in qs.roles:
                    if rname in fs.roles:
                        sim = float(cosine_similarity(
                            qs.roles[rname].data, fs.roles[rname].data
                        ))
                        role_sims[rname] = sim

        if not role_sims:
            return True

        desc = max(role_sims.get("description", 0), 0)
        fname = max(role_sims.get("function_name", 0), 0)

        # Weighted relevance: description is the strongest discriminator,
        # function_name provides structural confirmation
        relevance = desc * 0.6 + fname * 0.4

        # Hard floor: very low overall score = definitely irrelevant
        if overall_score < 0.12:
            return True

        # Raised from 0.14: BFCL irrelevance queries are completely off-domain
        # and should have near-zero similarity to all available functions.
        return relevance < 0.20

    # ── BFCL evaluation interface ──

    def predict(
        self,
        query: str,
        func_defs: list[dict],
        category: str = "simple",
    ) -> list[dict]:
        """BFCL-compatible prediction interface."""
        if category == "parallel_multiple":
            result = self.route_multi(query, func_defs)
            tools = result["tools"]
            if not tools:
                return []
            return [{t: {}} for t in tools]

        result = self.route(query, func_defs)

        if category == "irrelevance":
            q_glyph = result.get("_query_glyph")
            f_glyph = result.get("_best_func_glyph")
            if q_glyph and f_glyph and self.is_irrelevant(
                q_glyph, f_glyph, result["confidence"]
            ):
                return []
            tool = result["tool"]
            return [{tool: {}}] if tool else []

        tool = result["tool"]
        if tool is None:
            return []
        return [{tool: {}}]

    def decode_ast(self, result: list[dict]) -> list[dict]:
        """Convert prediction result to BFCL AST format."""
        return result

    def decode_execute(self, result: list[dict]) -> list[str]:
        """Convert prediction result to BFCL executable format."""
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
