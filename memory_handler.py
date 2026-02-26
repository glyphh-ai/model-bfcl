"""
Glyphh Memory Handler for BFCL V4 Agentic — Memory Category.

Glyphh IS the memory backend. No external vector store, no FAISS, no embeddings.
Same HDC substrate as the function router, different concept structure.

Architecture:
  1. Parse prereq conversations into individual fact sentences
  2. Encode each fact as a Glyph (bag_of_words on content + topic role)
  3. When a question comes in, encode it as a query glyph
  4. Cosine similarity retrieval against all stored fact glyphs
  5. Return the source text of the best-matching fact
  6. LLM extracts the precise answer from the source text

The Glyphh memory is the brain. The LLM is just formatting the answer.
"""

import json
import re
import os
from pathlib import Path
from typing import Any

from glyphh.core.config import EncoderConfig, Layer, Role, Segment
from glyphh.core.types import Concept, Glyph
from glyphh.core.ops import cosine_similarity
from glyphh import Encoder

# ---------------------------------------------------------------------------
# Memory encoder config — Model M (memory specialist)
# ---------------------------------------------------------------------------

MEMORY_CONFIG = EncoderConfig(
    dimension=10000,
    seed=211,  # Unique seed for memory model
    apply_weights_during_encoding=False,
    include_temporal=False,
    layers=[
        Layer(
            name="content",
            similarity_weight=0.70,
            segments=[
                Segment(
                    name="fact",
                    roles=[
                        Role(
                            name="text",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="context",
            similarity_weight=0.30,
            segments=[
                Segment(
                    name="meta",
                    roles=[
                        Role(
                            name="topic",
                            similarity_weight=0.6,
                            text_encoding="bag_of_words",
                        ),
                        Role(
                            name="domain",
                            similarity_weight=0.4,
                            lexicons=[
                                "customer", "healthcare", "finance",
                                "student", "notetaker", "general",
                            ],
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
    "im", "ive", "id", "thats", "its", "dont", "wont", "cant",
    "really", "pretty", "actually", "basically", "like", "kind",
    "thing", "things", "lot", "much", "very", "quite", "some",
}


def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return [w for w in cleaned.split() if w not in _STOP_WORDS and len(w) > 1]


def _bow(words: list[str]) -> str:
    unique = list(dict.fromkeys(words))
    return " ".join(unique) if unique else "none"


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences, keeping meaningful chunks."""
    # Split on sentence boundaries
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    # Merge very short fragments with previous
    merged = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if merged and len(p.split()) < 5:
            merged[-1] = merged[-1] + " " + p
        else:
            merged.append(p)
    return merged


# ---------------------------------------------------------------------------
# Fact extraction from prereq conversations
# ---------------------------------------------------------------------------

def extract_facts(prereq_entries: list[dict], scenario: str) -> list[dict]:
    """Extract individual facts from prereq conversation entries.

    Each fact is a sentence or clause from a user turn that contains
    retrievable information (names, numbers, preferences, etc.).

    Returns list of {"text": str, "topic": str, "scenario": str}
    """
    facts = []
    for entry in prereq_entries:
        topic = entry.get("topic", "general")
        turns = entry.get("question", [])

        for turn in turns:
            # Extract user content
            content = ""
            if isinstance(turn, list):
                for msg in turn:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        content = msg.get("content", "")
            elif isinstance(turn, dict):
                if turn.get("role") == "user":
                    content = turn.get("content", "")
            elif isinstance(turn, str):
                content = turn

            if not content:
                continue

            # Split into sentences — each sentence is a potential fact
            sentences = _split_sentences(content)
            for sent in sentences:
                tokens = _tokenize(sent)
                if len(tokens) < 3:
                    continue  # Too short to be a useful fact
                facts.append({
                    "text": sent,
                    "topic": topic,
                    "scenario": scenario,
                })

    return facts


# ---------------------------------------------------------------------------
# GlyphhMemory — the memory model
# ---------------------------------------------------------------------------

class GlyphhMemory:
    """Glyphh HDC memory backend for BFCL V4.

    Stores facts as glyphs. Retrieves by cosine similarity.
    This IS the memory — no external vector store needed.
    """

    def __init__(self):
        self.encoder = Encoder(MEMORY_CONFIG)
        self.fact_glyphs: list[Glyph] = []
        self.fact_texts: list[str] = []  # Source text for each glyph

    def store(self, text: str, topic: str = "general", domain: str = "general"):
        """Store a fact in memory. One-shot encoding, instant."""
        tokens = _tokenize(text)
        topic_tokens = _tokenize(topic)

        concept = Concept(
            name="fact",
            attributes={
                "text": _bow(tokens),
                "topic": _bow(topic_tokens),
                "domain": domain,
            },
        )
        glyph = self.encoder.encode(concept)
        self.fact_glyphs.append(glyph)
        self.fact_texts.append(text)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        """Retrieve the most relevant facts for a query.

        Uses multi-level similarity: global cortex + content role level.
        """
        tokens = _tokenize(query)
        q_concept = Concept(
            name="query",
            attributes={
                "text": _bow(tokens),
                "topic": _bow(tokens),
                "domain": "general",
            },
        )
        q_glyph = self.encoder.encode(q_concept)

        scores = []
        for i, fg in enumerate(self.fact_glyphs):
            # Global cortex similarity
            cortex_sim = float(cosine_similarity(
                q_glyph.global_cortex.data,
                fg.global_cortex.data,
            ))

            # Content role-level similarity (finer grain)
            role_sim = cortex_sim  # fallback
            try:
                q_text_role = q_glyph.layers["content"].segments["fact"].roles["text"]
                f_text_role = fg.layers["content"].segments["fact"].roles["text"]
                role_sim = float(cosine_similarity(q_text_role.data, f_text_role.data))
            except (KeyError, AttributeError):
                pass

            # Weight role-level higher — it's the content match
            sim = cortex_sim * 0.3 + role_sim * 0.7
            scores.append((sim, i))

        scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for sim, idx in scores[:top_k]:
            results.append({
                "text": self.fact_texts[idx],
                "score": round(sim, 4),
            })
        return results

    def clear(self):
        self.fact_glyphs.clear()
        self.fact_texts.clear()

    @property
    def size(self) -> int:
        return len(self.fact_glyphs)


# ---------------------------------------------------------------------------
# Chunked fact extraction — overlapping windows for better retrieval
# ---------------------------------------------------------------------------

def extract_facts_chunked(
    prereq_entries: list[dict],
    scenario: str,
    chunk_size: int = 3,
    overlap: int = 1,
) -> list[dict]:
    """Extract facts using overlapping sentence windows.

    Instead of individual sentences, use windows of chunk_size sentences
    with overlap. This preserves more context around each fact.

    Also stores the full turn text as a single fact for broad matching.
    """
    facts = []

    for entry in prereq_entries:
        topic = entry.get("topic", "general")
        turns = entry.get("question", [])

        for turn in turns:
            content = ""
            if isinstance(turn, list):
                for msg in turn:
                    if isinstance(msg, dict) and msg.get("role") == "user":
                        content = msg.get("content", "")
            elif isinstance(turn, dict):
                if turn.get("role") == "user":
                    content = turn.get("content", "")
            elif isinstance(turn, str):
                content = turn

            if not content or len(content.split()) < 3:
                continue

            # Store full turn as one fact (broad matching)
            facts.append({
                "text": content,
                "topic": topic,
                "scenario": scenario,
            })

            # Also store sentence-level chunks with overlap
            sentences = _split_sentences(content)
            if len(sentences) <= 1:
                continue

            step = max(1, chunk_size - overlap)
            for i in range(0, len(sentences), step):
                chunk = " ".join(sentences[i:i + chunk_size])
                tokens = _tokenize(chunk)
                if len(tokens) < 3:
                    continue
                facts.append({
                    "text": chunk,
                    "topic": topic,
                    "scenario": scenario,
                })

    return facts
