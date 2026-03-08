"""
Pure HDC scorer for BFCL function routing.

BFCLScorer implements the ModelScorer protocol from glyphh.cognitive.
Functions are first-class Glyphs encoded into the Glyphh HDC space.
No LLM anywhere.

Scoring: 4-level hierarchical cosine similarity
  cortex  (5%)  — global bundled representation
  layer  (10%)  — per-layer cortex
  segment (25%) — per-segment cortex
  role   (60%)  — per-role vectors (leaf, most discriminating)
"""

from __future__ import annotations

from typing import Any

from glyphh import Encoder
from glyphh.core.ops import cosine_similarity
from glyphh.core.types import Concept, Glyph
from glyphh.cognitive.model_scorer import ScorerResult

from encoder import ENCODER_CONFIG, encode_function, encode_exemplar, encode_query


# ---------------------------------------------------------------------------
# BFCLScoringStrategy — 4-level hierarchical similarity
# ---------------------------------------------------------------------------

class BFCLScoringStrategy:
    """4-level hierarchical similarity between two Glyphs.

    Weights: cortex 5% + layer 10% + segment 25% + role 60%

    The role level (60%) is the most discriminating: it compares each
    named role vector individually (action, target, function_name,
    description, parameters), giving a nuanced per-field similarity.
    """

    _W_CORTEX  = 0.05
    _W_LAYER   = 0.10
    _W_SEGMENT = 0.25
    _W_ROLE    = 0.60

    # Per-role weights: action gets 2x to better distinguish Find vs Buy/Book
    _ROLE_WEIGHTS: dict[str, float] = {
        "action":        2.0,
        "target":        1.0,
        "function_name": 1.5,
        "description":   1.0,
        "parameters":    0.8,
    }

    def score_pair(self, q: Glyph, f: Glyph) -> float:
        """Compute weighted hierarchical similarity between query and function Glyphs."""
        # Global cortex
        cortex_sim = float(cosine_similarity(
            q.global_cortex.data, f.global_cortex.data,
        ))

        # Layer-level cortex average
        layer_sims = []
        for lname in q.layers:
            if lname in f.layers:
                layer_sims.append(float(cosine_similarity(
                    q.layers[lname].cortex.data,
                    f.layers[lname].cortex.data,
                )))
        layer_sim = sum(layer_sims) / len(layer_sims) if layer_sims else 0.0

        # Segment-level cortex average
        seg_sims = []
        for lname in q.layers:
            if lname not in f.layers:
                continue
            for sname in q.layers[lname].segments:
                if sname in f.layers[lname].segments:
                    seg_sims.append(float(cosine_similarity(
                        q.layers[lname].segments[sname].cortex.data,
                        f.layers[lname].segments[sname].cortex.data,
                    )))
        seg_sim = sum(seg_sims) / len(seg_sims) if seg_sims else 0.0

        # Role-level weighted average (most discriminating)
        role_weighted_sum = 0.0
        role_weight_total = 0.0
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
                        sim = float(cosine_similarity(qr, fr))
                        w = self._ROLE_WEIGHTS.get(rname, 1.0)
                        role_weighted_sum += sim * w
                        role_weight_total += w
        role_sim = role_weighted_sum / role_weight_total if role_weight_total > 0 else 0.0

        return (
            self._W_CORTEX  * cortex_sim
            + self._W_LAYER   * layer_sim
            + self._W_SEGMENT * seg_sim
            + self._W_ROLE    * role_sim
        )


# ---------------------------------------------------------------------------
# BFCLScorer — ModelScorer implementation
# ---------------------------------------------------------------------------

class BFCLScorer:
    """Pure HDC function scorer for BFCL.

    Pipeline:
      configure(func_defs) → encode each function as a Glyph
      score(query)         → encode query as a Glyph, score against all functions
      score_multi(query)   → same but select multiple functions via gap analysis

    No LLM. No external service calls. All computation is deterministic HDC.
    """

    # Queries below this similarity are considered irrelevant
    IRRELEVANCE_THRESHOLD = 0.22

    # Multi-function gap analysis
    MULTI_GAP_RATIO      = 0.75   # relative drop > 75% → stop selecting
    MULTI_FALLBACK_RATIO = 0.20   # absolute ratio to best < 20% → stop

    def __init__(self) -> None:
        self._encoder   = Encoder(ENCODER_CONFIG)
        self._strategy  = BFCLScoringStrategy()
        self._func_glyphs: dict[str, Glyph] = {}
        self._func_defs:   dict[str, dict]  = {}

    # ── ModelScorer protocol ─────────────────────────────────────────────

    def configure(self, func_defs: list[dict[str, Any]]) -> None:
        """Encode function definitions into Glyphs."""
        self._func_defs = {f["name"]: f for f in func_defs}
        self._func_glyphs.clear()

        for func_def in func_defs:
            concept_dict = encode_function(func_def)
            glyph = self._encoder.encode(Concept(
                name=concept_dict["name"],
                attributes=concept_dict["attributes"],
            ))
            self._func_glyphs[func_def["name"]] = glyph

    def configure_from_exemplars(self, exemplars: list[dict]) -> None:
        """Encode pre-built exemplars into Glyphs.

        Multiple exemplars per function: each variant encodes into a separate
        Glyph. score() returns the function name from the best-matching
        exemplar Glyph (stripped of variant suffix).
        """
        self._func_glyphs.clear()
        self._func_defs.clear()

        for exemplar in exemplars:
            concept_dict = encode_exemplar(exemplar)
            glyph = self._encoder.encode(Concept(
                name=concept_dict["name"],
                attributes=concept_dict["attributes"],
            ))
            # Key by "func_name__v{N}" to allow multiple Glyphs per function
            func_name = exemplar["function_name"]
            variant = exemplar.get("variant", 1)
            key = f"{func_name}__v{variant}"
            self._func_glyphs[key] = glyph

    def score(self, query: str) -> ScorerResult:
        """Score a query against configured functions. Returns single best match.

        Always populates `functions` with the top match so callers can use it
        with force=True. `is_irrelevant` is flagged when confidence falls below
        IRRELEVANCE_THRESHOLD — callers decide whether to heed this flag.
        """
        if not self._func_glyphs:
            return ScorerResult()

        q_glyph = self._encode_query(query)
        scores  = self._score_all(q_glyph)

        if not scores:
            return ScorerResult()

        best = scores[0]
        return ScorerResult(
            functions=[best["function"]] if best["function"] else [],
            confidence=best["score"],
            all_scores=scores,
            is_irrelevant=best["score"] < self.IRRELEVANCE_THRESHOLD,
        )

    def score_multi(self, query: str) -> ScorerResult:
        """Score for multiple matching functions using gap analysis.

        Always runs gap analysis and returns candidate functions regardless of
        confidence level — is_irrelevant is flagged but callers with force=True
        can still use the candidates.
        """
        if not self._func_glyphs:
            return ScorerResult()

        q_glyph = self._encode_query(query)
        scores  = self._score_all(q_glyph)

        if not scores:
            return ScorerResult()

        is_irrelevant = scores[0]["score"] < self.IRRELEVANCE_THRESHOLD
        selected = self._select_multiple(scores)
        avg_conf = (
            sum(s["score"] for s in scores if s["function"] in set(selected))
            / len(selected)
            if selected else scores[0]["score"]
        )

        return ScorerResult(
            functions=selected,
            confidence=avg_conf,
            all_scores=scores,
            is_irrelevant=is_irrelevant,
        )

    def encode_query(self, query: str) -> Glyph:
        """Encode a query into a Glyph (ModelScorer protocol)."""
        return self._encode_query(query)

    def get_func_glyphs(self) -> dict[str, Glyph]:
        """Return encoded function Glyphs (ModelScorer protocol)."""
        return dict(self._func_glyphs)

    def scoring_strategy(self) -> BFCLScoringStrategy:
        """Return the scoring strategy (ModelScorer protocol)."""
        return self._strategy

    # ── Internals ────────────────────────────────────────────────────────

    def _encode_query(self, query: str) -> Glyph:
        q_dict = encode_query(query)
        return self._encoder.encode(Concept(
            name=q_dict["name"],
            attributes=q_dict["attributes"],
        ))

    @staticmethod
    def _strip_variant(name: str) -> str:
        """Strip variant suffix: 'GorillaFileSystem.mv__v1' → 'GorillaFileSystem.mv'."""
        idx = name.find("__v")
        return name[:idx] if idx >= 0 else name

    def _score_all(self, q_glyph: Glyph) -> list[dict]:
        """Score query against all function Glyphs, sorted descending.

        When multiple exemplar variants exist for the same function,
        keeps only the best-scoring variant per function.
        """
        raw_scores = [
            {"function": fname, "score": self._strategy.score_pair(q_glyph, fglyph)}
            for fname, fglyph in self._func_glyphs.items()
        ]
        raw_scores.sort(key=lambda x: x["score"], reverse=True)

        # Deduplicate: keep best variant per function
        seen: set[str] = set()
        scores: list[dict] = []
        for entry in raw_scores:
            func = self._strip_variant(entry["function"])
            if func not in seen:
                seen.add(func)
                scores.append({"function": func, "score": entry["score"]})

        return scores

    def _select_multiple(self, scores: list[dict]) -> list[str]:
        """Select multiple functions from a sorted score list via gap analysis."""
        if not scores:
            return []
        if len(scores) == 1:
            return [scores[0]["function"]]

        selected   = [scores[0]["function"]]
        best_score = scores[0]["score"]

        if best_score <= 0:
            return []

        for i in range(1, len(scores)):
            curr = scores[i]["score"]
            prev = scores[i - 1]["score"]

            if prev <= 0:
                break

            gap   = (prev - curr) / prev
            ratio = curr / best_score

            if gap > self.MULTI_GAP_RATIO or ratio < self.MULTI_FALLBACK_RATIO:
                break

            selected.append(scores[i]["function"])

        return selected
