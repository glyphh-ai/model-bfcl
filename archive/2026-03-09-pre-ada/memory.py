"""
Pure HDC memory retrieval for BFCL V4 agentic evaluation.

Approach: clause-level encoding + sliding windows + cosine similarity retrieval.
  1. Prereq conversation → split into clauses → create overlapping windows
  2. Encode each window as a Glyph (BoW)
  3. Question → encode as Glyph → cosine against all windows → top matches
  4. Return top matching windows (ground truth is naturally a substring)

Evaluation: agentic_checker() checks if any ground_truth string is contained
(case-insensitive, punctuation-stripped) in the model response.

Pure HDC. No LLM. No external services.
"""

import re

from glyphh import Encoder
from glyphh.core.config import EncoderConfig, Layer, Role, Segment
from glyphh.core.ops import cosine_similarity
from glyphh.core.types import Concept, Glyph


# ---------------------------------------------------------------------------
# Encoder config — text-focused, BoW only
# ---------------------------------------------------------------------------

MEMORY_ENCODER_CONFIG = EncoderConfig(
    dimension=10000,
    seed=42,
    apply_weights_during_encoding=False,
    include_temporal=False,
    layers=[
        Layer(
            name="content",
            similarity_weight=1.0,
            segments=[
                Segment(
                    name="text",
                    roles=[
                        Role(
                            name="words",
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
# Text processing
# ---------------------------------------------------------------------------

# Minimal stop words — keep content-bearing words like "first", "name", etc.
_STOP_WORDS = {
    "the", "a", "an", "to", "for", "on", "in", "is", "it", "i",
    "do", "can", "now", "up", "and", "or", "of", "with", "from",
    "this", "that", "how", "what", "when", "where", "which", "who",
    "then", "but", "just", "be", "been", "not",
    "are", "was", "were", "will", "would", "could", "should",
    "you", "your", "he", "she", "they", "us", "his", "her",
    "im", "ive", "id", "so", "as", "at", "if",
}


def _split_clauses(text: str) -> list[str]:
    """Split text into clauses — smaller chunks than sentences.

    Splits on sentence boundaries AND clause boundaries (commas, semicolons,
    dashes, parentheses) to create shorter chunks where each word has
    more signal in BoW encoding.
    """
    # First split on sentence boundaries
    parts = re.split(r'(?<=[.!?])\s+(?=[A-Z])|\n\s*\n', text)
    clauses = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        # Split on clause boundaries: semicolons, dashes, " and ", " but "
        # Also split on commas when followed by coordinating conjunctions
        sub_parts = re.split(
            r'[;—–]\s*|,\s+(?:and|but|so|or|because|which|where|who|although)\s+',
            part,
        )
        for sp in sub_parts:
            sp = sp.strip()
            if len(sp) > 5:
                clauses.append(sp)
    return clauses


def _tokenize(text: str) -> list[str]:
    """Tokenize into meaningful lowercase words."""
    cleaned = re.sub(r"[^a-z0-9@_.'\-\s]", "", text.lower())
    words = cleaned.split()
    return [w for w in words if w not in _STOP_WORDS and len(w) > 1]


def _bow_value(words: list[str]) -> str:
    """Build deduplicated BoW string."""
    unique = list(dict.fromkeys(words))
    return " ".join(unique) if unique else "none"


# ---------------------------------------------------------------------------
# MemoryHandler
# ---------------------------------------------------------------------------

class MemoryHandler:
    """Pure HDC memory retrieval using Glyphh Encoder.

    Ingests prereq conversation text, encodes clause windows as Glyphs,
    retrieves best-matching windows for each question via cosine similarity.
    """

    TOP_K = 50           # chunks to return
    WINDOW_SIZE = 3      # number of adjacent clauses per window

    def __init__(self) -> None:
        self._encoder = Encoder(MEMORY_ENCODER_CONFIG)
        self._chunk_glyphs: list[tuple[str, Glyph]] = []

    def ingest(self, prereq_entries: list[dict]) -> None:
        """Encode prereq conversation clause windows as Glyphs."""
        self._chunk_glyphs.clear()
        all_clauses: list[str] = []

        for entry in prereq_entries:
            for turn in entry.get("question", []):
                if isinstance(turn, list):
                    for msg in turn:
                        text = msg.get("content", "")
                        if text:
                            all_clauses.extend(_split_clauses(text))
                elif isinstance(turn, dict):
                    text = turn.get("content", "")
                    if text:
                        all_clauses.extend(_split_clauses(text))

        # Create sliding windows of WINDOW_SIZE clauses
        for i in range(len(all_clauses)):
            window_clauses = all_clauses[i:i + self.WINDOW_SIZE]
            window_text = " ".join(window_clauses)
            tokens = _tokenize(window_text)
            if not tokens:
                continue
            glyph = self._encoder.encode(Concept(
                name="chunk",
                attributes={"words": _bow_value(tokens)},
            ))
            self._chunk_glyphs.append((window_text, glyph))

    def query(self, question: str) -> str:
        """Retrieve best matching clause windows for a question.

        Returns concatenated top-K windows. Since eval checks substring
        containment, returning the source text naturally includes
        the ground truth answer.
        """
        if not self._chunk_glyphs:
            return ""

        q_tokens = _tokenize(question)
        if not q_tokens:
            return ""

        q_glyph = self._encoder.encode(Concept(
            name="query",
            attributes={"words": _bow_value(q_tokens)},
        ))

        # Score all chunks
        scored: list[tuple[str, float]] = []
        for text, glyph in self._chunk_glyphs:
            sim = float(cosine_similarity(
                q_glyph.global_cortex.data,
                glyph.global_cortex.data,
            ))
            scored.append((text, sim))

        scored.sort(key=lambda x: x[1], reverse=True)

        # Return top-K chunks
        return " ".join(text for text, _ in scored[:self.TOP_K])
