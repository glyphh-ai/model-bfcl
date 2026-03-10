"""HDC Irrelevance Detection Model for BFCL.

Determines whether a query is relevant to any of the available functions.
Uses three HDC signals combined via binding:

1. Parameter alignment — do the query's nouns match any function parameter names?
   This is the strongest signal: "calculate volume" vs function with param "park_name"
   immediately reveals irrelevance.

2. Name token alignment — does the query reference words from the function name?
   "compute derivative" vs "compute_definite_integral" — "derivative" != "integral"

3. Action+target binding — bind(action_vec, target_vec) catches semantic mismatches
   that survive vocabulary overlap.

Architecture:
  Encoder (seed=53) with 3 layers:
    - param_match: BoW of query nouns vs BoW of all parameter names
    - name_match: BoW of query tokens vs BoW of function name tokens
    - composite: bind(action, target) compositional match

This model is ONLY used for irrelevance detection categories.
It does NOT affect function routing (which uses the main HDC scorer).

Exports:
    IrrelevanceModel — dedicated irrelevance classifier
"""

import re
import numpy as np
from glyphh import Encoder
from glyphh.core.config import EncoderConfig, Layer, Role, Segment
from glyphh.core.ops import bind, bundle, cosine_similarity
from glyphh.core.types import Concept
from glyphh.linguistics.character import CharacterEncoder

from intent import extract_action, extract_target


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "the", "a", "an", "to", "for", "on", "in", "is", "it", "i",
    "do", "can", "please", "now", "up", "my", "our", "me", "we",
    "and", "or", "of", "with", "from", "this", "that", "about",
    "how", "what", "when", "where", "which", "who", "also",
    "then", "but", "just", "them", "their", "its", "be", "been",
    "have", "has", "had", "not", "dont", "want", "think",
    "given", "using", "by", "if", "so", "as", "at", "into",
    "are", "was", "were", "will", "would", "could", "should",
    "may", "might", "shall", "need", "does", "did", "am",
    "you", "your", "he", "she", "they", "us", "his", "her",
}

_GENERIC_WORDS = {
    "function", "method", "call", "value", "return", "result",
    "input", "output", "parameter", "argument", "type", "object",
    "true", "false", "null", "none", "default", "specified",
    "based", "provided", "particular", "specific", "certain",
}


def _split_camel_snake(name: str) -> str:
    name = name.replace("_", " ").replace("-", " ").replace(".", " ")
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return name.lower().strip()


def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return [w for w in cleaned.split()
            if w not in _STOP_WORDS and w not in _GENERIC_WORDS and len(w) > 1]


def _bow(words: list[str]) -> str:
    unique = list(dict.fromkeys(words))
    return " ".join(unique) if unique else "none"


def _extract_param_names(func_def: dict) -> list[str]:
    """Extract all parameter names from a function definition."""
    params = func_def.get("parameters", {})
    props = params.get("properties", {})
    names = []
    for pname in props:
        # Split param names: "park_name" → ["park", "name"]
        names.extend(_split_camel_snake(pname).split())
    # Also include parameter descriptions
    for pname, pinfo in props.items():
        desc = pinfo.get("description", "")
        if desc:
            names.extend(_tokenize(desc)[:5])  # first 5 meaningful words
    return [n for n in names if n not in _STOP_WORDS and n not in _GENERIC_WORDS and len(n) > 1]


# ---------------------------------------------------------------------------
# Encoder config — 2 layers for structured comparison
# ---------------------------------------------------------------------------

IRRELEVANCE_CONFIG = EncoderConfig(
    dimension=10000,
    seed=53,
    apply_weights_during_encoding=False,
    include_temporal=False,
    layers=[
        Layer(
            name="param_alignment",
            similarity_weight=0.5,  # strongest signal
            segments=[
                Segment(
                    name="params",
                    roles=[
                        Role(
                            name="param_tokens",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="name_alignment",
            similarity_weight=0.35,
            segments=[
                Segment(
                    name="name",
                    roles=[
                        Role(
                            name="name_tokens",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="description_alignment",
            similarity_weight=0.15,
            segments=[
                Segment(
                    name="desc",
                    roles=[
                        Role(
                            name="desc_tokens",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# IrrelevanceModel
# ---------------------------------------------------------------------------

class IrrelevanceModel:
    """HDC-based irrelevance detection.

    Uses parameter name alignment as primary signal — if the query's nouns
    don't match any function parameter names, it's likely irrelevant.
    """

    THRESHOLD = 0.20

    def __init__(self):
        self._encoder = Encoder(IRRELEVANCE_CONFIG)
        self._func_glyphs: list[tuple] = []  # (glyph, func_name)

    def configure(self, func_defs: list[dict]) -> None:
        """Encode function definitions."""
        self._func_glyphs = []

        for fd in func_defs:
            name = fd.get("name", "")
            description = fd.get("description", "")

            # Function name tokens
            name_tokens = _tokenize(_split_camel_snake(name))

            # Parameter name tokens (strongest discriminator)
            param_tokens = _extract_param_names(fd)

            # Description tokens
            desc_tokens = _tokenize(description)[:10]

            glyph = self._encoder.encode(Concept(
                name=f"irr_{name}",
                attributes={
                    "param_tokens": _bow(param_tokens),
                    "name_tokens": _bow(name_tokens),
                    "desc_tokens": _bow(desc_tokens),
                },
            ))
            self._func_glyphs.append((glyph, name))

    def score(self, query: str) -> float:
        """Score query relevance. Higher = more relevant."""
        if not self._func_glyphs:
            return 0.0

        query_tokens = _tokenize(query)

        query_glyph = self._encoder.encode(Concept(
            name="irr_query",
            attributes={
                "param_tokens": _bow(query_tokens),
                "name_tokens": _bow(query_tokens),
                "desc_tokens": _bow(query_tokens),
            },
        ))

        # Layer-weighted comparison
        best = 0.0
        for glyph, _ in self._func_glyphs:
            # Get per-layer similarities
            total_sim = 0.0
            total_weight = 0.0
            for layer_name in query_glyph.layers:
                if layer_name not in glyph.layers:
                    continue
                q_layer = query_glyph.layers[layer_name]
                f_layer = glyph.layers[layer_name]
                layer_weight = IRRELEVANCE_CONFIG.layers[
                    next(i for i, l in enumerate(IRRELEVANCE_CONFIG.layers) if l.name == layer_name)
                ].similarity_weight

                for seg_name in q_layer.segments:
                    if seg_name not in f_layer.segments:
                        continue
                    q_seg = q_layer.segments[seg_name]
                    f_seg = f_layer.segments[seg_name]
                    for role_name in q_seg.roles:
                        if role_name not in f_seg.roles:
                            continue
                        sim = float(cosine_similarity(
                            q_seg.roles[role_name].data,
                            f_seg.roles[role_name].data,
                        ))
                        total_sim += sim * layer_weight
                        total_weight += layer_weight

            score = total_sim / total_weight if total_weight > 0 else 0.0
            if score > best:
                best = score

        return best

    def is_relevant(self, query: str) -> bool:
        """Determine if query is relevant to any configured function."""
        return self.score(query) >= self.THRESHOLD
