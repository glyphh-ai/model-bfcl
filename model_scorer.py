"""
BFCLModelScorer — HDC encoder-based scoring for BFCL function routing.

Implements the ModelScorer protocol from glyphh.cognitive so the BFCL
HDC encoder (Model A) can be plugged into the CognitiveLoop as the
primary classification path.

Pipeline: encode functions → encode query → hierarchical similarity → rank.
Sub-millisecond per query, no LLM needed for routing.
"""

from __future__ import annotations

from typing import Any

from glyphh import Encoder
from glyphh.core.config import EncoderConfig
from glyphh.core.ops import cosine_similarity
from glyphh.core.types import Concept, Glyph
from glyphh.cognitive.model_scorer import ScorerResult

from encoder import ENCODER_CONFIG, encode_function, encode_query


class BFCLModelScorer:
    """HDC model scorer for BFCL function routing.

    Encodes function definitions and queries using the BFCL encoder
    (10000-dim, seed=42, two-layer hierarchy) and scores via
    hierarchical cosine similarity.

    Implements the ModelScorer protocol:
        scorer.configure(func_defs)
        scorer.score(query) → ScorerResult
        scorer.score_multi(query) → ScorerResult
    """

    # Hierarchical similarity weights
    _W_CORTEX = 0.05    # Global cortex
    _W_LAYER = 0.10     # Per-layer cortex
    _W_SEGMENT = 0.25   # Per-segment cortex
    _W_ROLE = 0.60      # Per-role (leaf level)

    # Irrelevance threshold
    _IRRELEVANCE_THRESHOLD = 0.20

    # Multi-function selection
    _MULTI_GAP_RATIO = 0.60      # Relative gap > 60% → single function
    _MULTI_FALLBACK_RATIO = 0.45  # Absolute ratio < 45% → single function

    def __init__(self):
        self._encoder = Encoder(ENCODER_CONFIG)
        self._func_glyphs: dict[str, Glyph] = {}
        self._func_defs: dict[str, dict] = {}

    def configure(self, func_defs: list[dict[str, Any]]) -> None:
        """Encode function definitions into Glyphs."""
        self._func_glyphs.clear()
        self._func_defs = {f["name"]: f for f in func_defs}

        for func_def in func_defs:
            concept_dict = encode_function(func_def)
            concept = Concept(
                name=concept_dict["name"],
                attributes=concept_dict["attributes"],
            )
            glyph = self._encoder.encode(concept)
            self._func_glyphs[func_def["name"]] = glyph

    def score(self, query: str) -> ScorerResult:
        """Score a query against configured functions. Returns best match."""
        if not self._func_glyphs:
            return ScorerResult()

        query_dict = encode_query(query)
        query_concept = Concept(
            name=query_dict["name"],
            attributes=query_dict["attributes"],
        )
        query_glyph = self._encoder.encode(query_concept)

        # Score against all functions
        scores = []
        for fname, fglyph in self._func_glyphs.items():
            sim = self._hierarchical_similarity(query_glyph, fglyph)
            scores.append({"function": fname, "score": sim})

        scores.sort(key=lambda x: x["score"], reverse=True)

        # Check irrelevance
        if self._is_irrelevant(query_glyph, scores):
            return ScorerResult(
                confidence=scores[0]["score"] if scores else 0.0,
                all_scores=scores,
                is_irrelevant=True,
            )

        # Best match
        best = scores[0] if scores else {"function": "", "score": 0.0}
        return ScorerResult(
            functions=[best["function"]] if best["function"] else [],
            confidence=best["score"],
            all_scores=scores,
        )

    def score_multi(self, query: str) -> ScorerResult:
        """Score for multiple matching functions using gap analysis."""
        if not self._func_glyphs:
            return ScorerResult()

        query_dict = encode_query(query)
        query_concept = Concept(
            name=query_dict["name"],
            attributes=query_dict["attributes"],
        )
        query_glyph = self._encoder.encode(query_concept)

        # Score against all functions
        scores = []
        for fname, fglyph in self._func_glyphs.items():
            sim = self._hierarchical_similarity(query_glyph, fglyph)
            scores.append({"function": fname, "score": sim})

        scores.sort(key=lambda x: x["score"], reverse=True)

        # Check irrelevance
        if self._is_irrelevant(query_glyph, scores):
            return ScorerResult(
                confidence=scores[0]["score"] if scores else 0.0,
                all_scores=scores,
                is_irrelevant=True,
            )

        # Select multiple functions via gap analysis
        selected = self._select_multiple(scores)

        avg_confidence = (
            sum(s["score"] for s in scores if s["function"] in selected) / len(selected)
            if selected else 0.0
        )

        return ScorerResult(
            functions=selected,
            confidence=avg_confidence,
            all_scores=scores,
        )

    # ── Hierarchical similarity ──────────────────────────────────────────

    def _hierarchical_similarity(self, q: Glyph, f: Glyph) -> float:
        """Compute weighted hierarchical similarity between two Glyphs.

        Four levels:
          cortex  (5%)  — global cortex
          layer   (10%) — per-layer cortex average
          segment (25%) — per-segment cortex average
          role    (60%) — per-role average (leaf level)
        """
        # Global cortex
        cortex_sim = float(cosine_similarity(
            q.global_cortex.data, f.global_cortex.data,
        ))

        # Layer-level
        layer_sims = []
        for lname in q.layers:
            if lname in f.layers:
                ql = q.layers[lname].cortex.data
                fl = f.layers[lname].cortex.data
                layer_sims.append(float(cosine_similarity(ql, fl)))
        layer_sim = sum(layer_sims) / len(layer_sims) if layer_sims else 0.0

        # Segment-level
        seg_sims = []
        for lname in q.layers:
            if lname not in f.layers:
                continue
            for sname in q.layers[lname].segments:
                if sname in f.layers[lname].segments:
                    qs = q.layers[lname].segments[sname].cortex.data
                    fs = f.layers[lname].segments[sname].cortex.data
                    seg_sims.append(float(cosine_similarity(qs, fs)))
        seg_sim = sum(seg_sims) / len(seg_sims) if seg_sims else 0.0

        # Role-level
        role_sims = []
        for lname in q.layers:
            if lname not in f.layers:
                continue
            for sname in q.layers[lname].segments:
                if sname not in f.layers[lname].segments:
                    continue
                for rname in q.layers[lname].segments[sname].roles:
                    if rname in f.layers[lname].segments[sname].roles:
                        qr = q.layers[lname].segments[sname].roles[rname].data
                        fr = f.layers[lname].segments[sname].roles[rname].data
                        role_sims.append(float(cosine_similarity(qr, fr)))
        role_sim = sum(role_sims) / len(role_sims) if role_sims else 0.0

        return (
            self._W_CORTEX * cortex_sim
            + self._W_LAYER * layer_sim
            + self._W_SEGMENT * seg_sim
            + self._W_ROLE * role_sim
        )

    # ── Irrelevance detection ────────────────────────────────────────────

    def _is_irrelevant(self, query_glyph: Glyph, scores: list[dict]) -> bool:
        """Check if query doesn't match any function meaningfully.

        Uses role-level analysis: description similarity * 0.6 + function
        name similarity * 0.4 must exceed threshold for at least one function.
        """
        if not scores:
            return True

        best_relevance = 0.0
        for score_entry in scores[:3]:  # Check top 3
            fname = score_entry["function"]
            fglyph = self._func_glyphs.get(fname)
            if fglyph is None:
                continue

            # Extract role-level similarities
            desc_sim = 0.0
            fname_sim = 0.0
            try:
                qd = query_glyph.layers["semantics"].segments["context"].roles["description"].data
                fd = fglyph.layers["semantics"].segments["context"].roles["description"].data
                desc_sim = float(cosine_similarity(qd, fd))
            except (KeyError, AttributeError):
                pass
            try:
                qn = query_glyph.layers["signature"].segments["identity"].roles["function_name"].data
                fn = fglyph.layers["signature"].segments["identity"].roles["function_name"].data
                fname_sim = float(cosine_similarity(qn, fn))
            except (KeyError, AttributeError):
                pass

            relevance = desc_sim * 0.6 + fname_sim * 0.4
            best_relevance = max(best_relevance, relevance)

        return best_relevance < self._IRRELEVANCE_THRESHOLD

    # ── Multi-function selection ─────────────────────────────────────────

    def _select_multiple(self, scores: list[dict]) -> list[str]:
        """Select multiple functions using gap analysis.

        Walks scores from best to worst. A function is included if:
        1. Its score is within the gap threshold of the previous score
        2. Falls back to ratio analysis if gap is ambiguous

        Returns ordered list of selected function names.
        """
        if not scores:
            return []
        if len(scores) == 1:
            return [scores[0]["function"]]

        selected = [scores[0]["function"]]
        best_score = scores[0]["score"]

        if best_score <= 0:
            return []

        for i in range(1, len(scores)):
            curr = scores[i]["score"]
            prev = scores[i - 1]["score"]

            if prev <= 0:
                break

            # Relative gap from previous
            gap = (prev - curr) / prev
            if gap > self._MULTI_GAP_RATIO:
                break  # Big gap — stop including

            # Absolute ratio to best
            ratio = curr / best_score
            if ratio < self._MULTI_FALLBACK_RATIO:
                break  # Too far from best — stop

            selected.append(scores[i]["function"])

        return selected
