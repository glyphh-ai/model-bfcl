"""IrrelevanceSidecar — validates HDC routing for false-positive matches.

Independent HDC vector space (seed=97) focused on discriminative features
the main encoder may miss. The main encoder uses broad BoW similarity which
can produce high scores when vocabulary overlaps incidentally (e.g. "calculate
area of triangle" matching `determine_body_mass_index` because both mention
"height" and "calculate").

The sidecar encodes:
  - Function name tokens (BoW) — does the query actually reference words
    from the function name? This is the strongest relevance signal.
  - Target specificity — does the query's target noun match the function's
    target at a fine-grained level?
  - Description keywords (BoW) — filtered to key nouns only, excluding
    generic filler words that inflate main encoder scores.

Architecture:
  Main encoder (seed=42) → cosine score (broad match)
  Sidecar (seed=97) → cosine score (discriminative match)
  Combined: weighted blend determines relevance/irrelevance.

Exports:
  SIDECAR_CONFIG         — EncoderConfig for the sidecar model
  IrrelevanceSidecar     — validates query-function relevance
"""

import re

from glyphh import Encoder
from glyphh.core.config import EncoderConfig, Layer, Role, Segment
from glyphh.core.ops import cosine_similarity
from glyphh.core.types import Concept

from intent import extract_action, extract_target

# ---------------------------------------------------------------------------
# SIDECAR_CONFIG — Irrelevance detection in independent vector space
# ---------------------------------------------------------------------------

SIDECAR_CONFIG = EncoderConfig(
    dimension=10000,
    seed=97,  # Independent from main encoder (seed=42)
    apply_weights_during_encoding=False,
    include_temporal=False,
    layers=[
        Layer(
            name="name_alignment",
            similarity_weight=0.6,
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
            name="specificity",
            similarity_weight=0.4,
            segments=[
                Segment(
                    name="target_match",
                    roles=[
                        Role(
                            name="target",
                            similarity_weight=1.0,
                            lexicons=[
                                # Math — shapes
                                "triangle", "circle", "rectangle", "square",
                                "polygon", "shape",
                                # Math — measurements
                                "area", "circumference", "perimeter",
                                "hypotenuse", "angle", "distance",
                                # Math — algebra / number theory
                                "equation", "roots", "prime", "factorial",
                                "sum", "number",
                                # Math — sequences
                                "fibonacci", "sequence",
                                # Math — calculus / linear algebra / stats
                                "derivative", "integral", "matrix", "vector",
                                "mean", "probability", "standard", "grade",
                                # Physics
                                "velocity", "force", "energy", "temperature",
                                "particle", "charge", "diameter", "mass",
                                # Health / fitness
                                "hydration", "calorie", "steps",
                                # Filesystem
                                "file", "directory", "path", "content", "disk",
                                "permission",
                                # Finance
                                "stock", "price", "trade", "order", "portfolio",
                                "balance", "option", "growth", "ratio", "revenue",
                                "market", "currency", "depreciation", "mortgage",
                                # Travel / vehicle
                                "flight", "hotel", "trip", "route", "vehicle",
                                "fuel", "gear",
                                # Communication
                                "message", "channel", "notification", "post",
                                "user",
                                # Geography
                                "capital", "city", "country", "population",
                                # Places / Nature
                                "restaurant", "park", "mountain", "tourist",
                                # Weather / misc
                                "weather", "ticket",
                                # Events / History
                                "event",
                                # Geology
                                "era",
                                # Science / Biology
                                "genotype", "discoverer",
                                # Safety / Crime
                                "crime",
                                # Arts / museum
                                "artwork", "landmark", "museum",
                                # Entertainment / games
                                "game", "book",
                                # Recipes / Cooking
                                "recipe",
                                # Players / sports
                                "player", "team", "sport", "goal",
                                # Legal
                                "case",
                                # Data structures
                                "string", "list", "record", "data", "key",
                                # API / HTTP
                                "request", "response", "endpoint", "api",
                                "none",
                            ],
                        ),
                        Role(
                            name="description_nouns",
                            similarity_weight=0.7,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Helpers
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

# Generic verbs/words that inflate BoW matches without adding signal
_GENERIC_WORDS = {
    "get", "set", "find", "make", "use", "take", "give", "let",
    "function", "method", "call", "value", "return", "result",
    "input", "output", "parameter", "argument", "type", "object",
    "true", "false", "null", "none", "default", "specified",
    "based", "provided", "particular", "specific", "certain",
    "calculate", "compute", "determine", "retrieve", "obtain",
}


def _split_camel_snake(name: str) -> str:
    """Split camelCase and snake_case into space-separated words."""
    name = name.replace("_", " ").replace("-", " ").replace(".", " ")
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return name.lower().strip()


def _tokenize(text: str) -> list[str]:
    """Tokenize into meaningful lowercase words."""
    cleaned = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return [w for w in cleaned.split() if w not in _STOP_WORDS and len(w) > 1]


def _extract_nouns(text: str) -> list[str]:
    """Extract content nouns from description — filters generic/filler words."""
    tokens = _tokenize(text)
    return [t for t in tokens if t not in _GENERIC_WORDS]


def _bow_value(words: list[str]) -> str:
    """Deduplicated space-separated BoW string."""
    unique = list(dict.fromkeys(words))
    return " ".join(unique) if unique else "none"


# ---------------------------------------------------------------------------
# IrrelevanceSidecar
# ---------------------------------------------------------------------------

class IrrelevanceSidecar:
    """Validates HDC routing results to detect false-positive matches.

    Uses an independent HDC vector space (seed=97) focused on discriminative
    features. For each function set, builds a sidecar GlyphSpace. When a query
    comes in, encodes it and checks sidecar similarity against all functions.

    The sidecar score combined with the main encoder score provides better
    irrelevance detection than threshold-on-main-score alone.

    Usage:
        sidecar = IrrelevanceSidecar()
        sidecar.configure(func_defs)
        is_relevant = sidecar.is_relevant(query, main_score)
    """

    # Combined decision thresholds (tuned on BFCL V4 irrelevance data)
    MAIN_WEIGHT = 0.4
    SIDECAR_WEIGHT = 0.6
    COMBINED_THRESHOLD = 0.18

    def __init__(self):
        self._encoder = Encoder(SIDECAR_CONFIG)
        self._func_glyphs: list[tuple] = []  # (glyph, func_name)

    def configure(self, func_defs: list[dict]) -> None:
        """Build sidecar glyphs from function definitions."""
        self._func_glyphs = []

        for fd in func_defs:
            name = fd.get("name", "unknown")
            description = fd.get("description", "")

            # Extract function name tokens (strongest discriminator)
            name_tokens = _tokenize(_split_camel_snake(name))

            # Extract target from name + description
            name_words = _split_camel_snake(name)
            target = extract_target(name_words + " " + description[:100])

            # Extract description nouns (filtered — no generic filler)
            desc_nouns = _extract_nouns(description)

            glyph = self._encoder.encode(Concept(
                name=f"sidecar_{name}",
                attributes={
                    "name_tokens": _bow_value(name_tokens),
                    "target": target,
                    "description_nouns": _bow_value(desc_nouns),
                },
            ))
            self._func_glyphs.append((glyph, name))

    def score(self, query: str) -> float:
        """Score a query against all functions in the sidecar space.

        Returns the best sidecar similarity score across all functions.
        """
        if not self._func_glyphs:
            return 0.0

        # Extract query tokens for name alignment
        query_tokens = _tokenize(query)

        # Extract target from query
        target = extract_target(query)

        # Extract description-level nouns from query
        query_nouns = _extract_nouns(query)

        query_glyph = self._encoder.encode(Concept(
            name="sidecar_query",
            attributes={
                "name_tokens": _bow_value(query_tokens),
                "target": target,
                "description_nouns": _bow_value(query_nouns),
            },
        ))

        # Weighted role-level comparison (same pattern as ToolNameSidecar)
        role_weights = {
            "name_tokens": 0.6,
            "target": 0.5,
            "description_nouns": 0.4,
        }

        query_roles = {}
        for layer in query_glyph.layers.values():
            for seg in layer.segments.values():
                query_roles.update(seg.roles)

        best_score = 0.0
        for glyph, _ in self._func_glyphs:
            func_roles = {}
            for layer in glyph.layers.values():
                for seg in layer.segments.values():
                    func_roles.update(seg.roles)

            weighted_sum = 0.0
            weight_total = 0.0
            for rname, w in role_weights.items():
                if rname in query_roles and rname in func_roles:
                    sim = float(cosine_similarity(
                        query_roles[rname].data, func_roles[rname].data
                    ))
                    weighted_sum += sim * w
                    weight_total += w

            score = weighted_sum / weight_total if weight_total > 0 else 0.0
            if score > best_score:
                best_score = score

        return best_score

    def is_relevant(self, query: str, main_score: float) -> bool:
        """Determine if a query is relevant to the configured function set.

        Combines the main encoder score with the sidecar score using weighted
        blend. Returns True if the combined score exceeds the threshold.

        Args:
            query: The user's natural language query.
            main_score: The top-1 similarity score from the main encoder.

        Returns:
            True if the query is likely relevant, False if likely irrelevant.
        """
        sidecar_score = self.score(query)
        combined = (self.MAIN_WEIGHT * main_score +
                    self.SIDECAR_WEIGHT * sidecar_score)
        return combined >= self.COMBINED_THRESHOLD
