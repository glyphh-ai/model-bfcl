"""
Web Search Handler for BFCL V4 Agentic — Web Search Category.

Architecture (Glyphh-routed + structural verification):
  1. LLM decomposes multihop question into sub-queries
  2. For each hop: search DDG, fetch pages
  3. Encode all text chunks as glyphs in SearchMemory (bag-of-words)
  4. Glyphh retrieves best-matching chunks for the sub-question
  5. LLM extracts answer from Glyphh-curated context
  6. Glyphh structural model verifies: does the extracted answer's
     sentence context structurally match what the question asks?
  7. If verification fails, try next-best Glyphh chunks

Glyphh = attention layer (routes questions to text) + verification layer
LLM = decomposer + extractor from curated context (local LLMEngine)
"""

from __future__ import annotations

import json
import re
import os
import time
from typing import Any, TYPE_CHECKING

from glyphh.core.config import EncoderConfig, Layer, Role, Segment
from glyphh.core.types import Concept, Glyph
from glyphh.core.ops import cosine_similarity
from glyphh import Encoder

if TYPE_CHECKING:
    from glyphh.llm.engine import LLMEngine


# ---------------------------------------------------------------------------
# Shared helpers
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

def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return [w for w in cleaned.split() if w not in _STOP_WORDS and len(w) > 1]

def _bow(words: list[str]) -> str:
    unique = list(dict.fromkeys(words))
    return " ".join(unique) if unique else "none"

def _extract_entities(text: str) -> list[str]:
    caps = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
    nums = re.findall(r'\b\d{2,}\b', text)
    tokens = []
    for c in caps:
        tokens.extend(c.lower().split())
    tokens.extend(nums)
    return list(dict.fromkeys(tokens))


# ---------------------------------------------------------------------------
# Glyphh Search Memory — chunk-level routing (what worked at 38%)
# ---------------------------------------------------------------------------

SEARCH_CONFIG = EncoderConfig(
    dimension=10000,
    seed=317,
    apply_weights_during_encoding=False,
    include_temporal=False,
    layers=[
        Layer(
            name="content",
            similarity_weight=0.65,
            segments=[
                Segment(
                    name="text",
                    roles=[
                        Role(
                            name="keywords",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="topic",
            similarity_weight=0.35,
            segments=[
                Segment(
                    name="domain",
                    roles=[
                        Role(
                            name="entities",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
    ],
)


class SearchMemory:
    """Glyphh chunk-level search memory. Routes questions to text chunks."""

    def __init__(self):
        self.encoder = Encoder(SEARCH_CONFIG)
        self.snippet_glyphs: list[Glyph] = []
        self.snippet_texts: list[str] = []
        self.snippet_sources: list[str] = []

    def store(self, text: str, source: str = ""):
        tokens = _tokenize(text)
        entities = _extract_entities(text)
        if not tokens:
            return
        concept = Concept(
            name="snippet",
            attributes={
                "keywords": _bow(tokens),
                "entities": _bow(entities) if entities else _bow(tokens[:10]),
            },
        )
        glyph = self.encoder.encode(concept)
        self.snippet_glyphs.append(glyph)
        self.snippet_texts.append(text)
        self.snippet_sources.append(source)

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        if not self.snippet_glyphs:
            return []
        tokens = _tokenize(query)
        entities = _extract_entities(query)
        q_concept = Concept(
            name="query",
            attributes={
                "keywords": _bow(tokens),
                "entities": _bow(entities) if entities else _bow(tokens[:10]),
            },
        )
        q_glyph = self.encoder.encode(q_concept)
        scores = []
        for i, sg in enumerate(self.snippet_glyphs):
            cortex_sim = float(cosine_similarity(
                q_glyph.global_cortex.data, sg.global_cortex.data,
            ))
            role_sim = cortex_sim
            try:
                qr = q_glyph.layers["content"].segments["text"].roles["keywords"]
                sr = sg.layers["content"].segments["text"].roles["keywords"]
                role_sim = float(cosine_similarity(qr.data, sr.data))
            except (KeyError, AttributeError):
                pass
            sim = cortex_sim * 0.3 + role_sim * 0.7
            scores.append((sim, i))
        scores.sort(key=lambda x: x[0], reverse=True)
        return [
            {"text": self.snippet_texts[i], "source": self.snippet_sources[i],
             "score": round(s, 4)}
            for s, i in scores[:top_k]
        ]

    def clear(self):
        self.snippet_glyphs.clear()
        self.snippet_texts.clear()
        self.snippet_sources.clear()

    @property
    def size(self) -> int:
        return len(self.snippet_glyphs)


# ---------------------------------------------------------------------------
# Glyphh Structural Verifier — sentence-level answer verification
# ---------------------------------------------------------------------------

VERIFY_CONFIG = EncoderConfig(
    dimension=10000,
    seed=419,
    apply_weights_during_encoding=False,
    include_temporal=False,
    layers=[
        Layer(
            name="relation",
            similarity_weight=0.55,
            segments=[
                Segment(
                    name="predicate",
                    roles=[
                        Role(
                            name="verbs",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="subject",
            similarity_weight=0.45,
            segments=[
                Segment(
                    name="nouns",
                    roles=[
                        Role(
                            name="content_words",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
    ],
)

# Common verbs/relation words
_VERB_WORDS = {
    "produces", "produce", "producing", "produced", "founded", "found",
    "born", "won", "winning", "attended", "studied", "directed", "wrote",
    "located", "situated", "based", "plays", "played", "starring",
    "awarded", "received", "created", "built", "established", "opened",
    "published", "performed", "graduated", "enrolled", "leads", "led",
    "serves", "served", "named", "called", "known", "became", "elected",
    "appointed", "largest", "biggest", "smallest", "tallest", "richest",
    "most", "expensive", "popular", "famous", "first", "last", "oldest",
}


class StructuralVerifier:
    """Verifies that an extracted answer structurally matches the question.

    Encodes the question and the sentence containing the answer into
    relation + subject roles. If the structural similarity is too low,
    the answer is likely wrong (extracted from a tangentially related sentence).
    """

    def __init__(self):
        self.encoder = Encoder(VERIFY_CONFIG)

    def _encode(self, text: str) -> Glyph:
        words = re.sub(r"[^a-z0-9\s]", "", text.lower()).split()
        verbs = [w for w in words if w in _VERB_WORDS]
        nouns = [w for w in words if w not in _STOP_WORDS and w not in _VERB_WORDS and len(w) > 1]
        concept = Concept(
            name="sentence",
            attributes={
                "verbs": _bow(verbs) if verbs else "none",
                "content_words": _bow(nouns) if nouns else "none",
            },
        )
        return self.encoder.encode(concept)

    def verify(self, question: str, answer: str, context: str) -> float:
        """Score how well the answer's context structurally matches the question.

        Returns similarity score 0-1. Higher = better structural match.
        """
        # Find the sentence containing the answer in the context
        answer_sentence = ""
        for sent in re.split(r'(?<=[.!?])\s+', context):
            if answer.lower() in sent.lower():
                answer_sentence = sent
                break

        if not answer_sentence:
            return 0.5  # Can't verify, neutral score

        q_glyph = self._encode(question)
        a_glyph = self._encode(answer_sentence)

        return float(cosine_similarity(
            q_glyph.global_cortex.data, a_glyph.global_cortex.data,
        ))


# ---------------------------------------------------------------------------
# DDG search + URL fetch + chunking
# ---------------------------------------------------------------------------

def _search_ddg(query: str, max_results: int = 5) -> list[dict]:
    try:
        from ddgs import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
            return [{"title": r.get("title", ""), "url": r.get("href", ""),
                     "snippet": r.get("body", "")} for r in results]
    except ImportError:
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
                return [{"title": r.get("title", ""), "url": r.get("href", ""),
                         "snippet": r.get("body", "")} for r in results]
        except Exception as e:
            return [{"error": str(e)}]
    except Exception as e:
        return [{"error": str(e)}]


def _fetch_url(url: str, max_chars: int = 6000) -> str:
    import urllib.request
    try:
        req = urllib.request.Request(url, headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        with urllib.request.urlopen(req, timeout=8) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
        text = re.sub(r'<script[^>]*>.*?</script>', '', html, flags=re.DOTALL)
        text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', ' ', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text[:max_chars]
    except Exception:
        return ""


def _chunk_text(text: str, chunk_size: int = 200, overlap: int = 40) -> list[str]:
    words = text.split()
    if len(words) <= chunk_size:
        return [text] if len(words) > 15 else []
    chunks = []
    step = max(1, chunk_size - overlap)
    for i in range(0, len(words), step):
        chunk = " ".join(words[i:i + chunk_size])
        if len(chunk.split()) > 15:
            chunks.append(chunk)
    return chunks


# ---------------------------------------------------------------------------
# WebSearchAgent — Glyphh-routed + structurally verified
# ---------------------------------------------------------------------------

class WebSearchAgent:
    """Multihop web search with Glyphh routing + structural verification.

    Glyphh does two jobs:
      1. SearchMemory routes sub-questions to the best text chunks
      2. StructuralVerifier checks if the LLM's extracted answer
         structurally matches what the question is asking

    If verification fails, we try extracting from the next-best chunks.
    """

    def __init__(self, engine: LLMEngine | None = None):
        self._engine = engine
        self.verifier = StructuralVerifier()

    def _llm(self, system: str, user: str, max_tokens: int = 300) -> str:
        if self._engine is None:
            return ""
        try:
            return self._engine.generate(
                system=system,
                user=user,
                max_tokens=max_tokens,
            )
        except Exception:
            return ""

    def answer(self, question: str, max_hops: int = 6) -> str:
        """Answer a multihop question with Glyphh routing + verification."""
        memory = SearchMemory()
        sub_questions = self._decompose(question)
        findings = {}

        for i, sq in enumerate(sub_questions[:max_hops]):
            resolved = sq
            for step_num, ans in findings.items():
                resolved = resolved.replace(f"[STEP_{step_num}]", ans)

            # Search and ingest into Glyphh memory
            self._search_and_ingest(resolved, memory)

            # Glyphh retrieves best chunks
            matched = memory.retrieve(resolved, top_k=8)

            # Try extraction with structural verification
            ans = self._extract_with_verification(resolved, matched)

            # Retry with broader search if needed
            if not ans or ans.lower() in ("unknown", "not found", ""):
                self._search_and_ingest(resolved + " wikipedia", memory, fetch_pages=1)
                matched2 = memory.retrieve(resolved, top_k=8)
                ans = self._extract_with_verification(resolved, matched2)

            findings[i + 1] = ans if ans and ans.lower() not in ("unknown", "") else "unknown"
            time.sleep(0.15)

        # Final: retrieve from ALL accumulated context
        final_matched = memory.retrieve(question, top_k=8)
        final_context = "\n\n".join(m["text"] for m in final_matched)

        chain = "\n".join(f"Step {k}: {v}" for k, v in sorted(findings.items()))
        final = self._llm(
            "Using the research findings AND supporting text, answer the question.\n"
            "Return ONLY the final answer — a specific name, number, date, or place.\n"
            "No explanation.",
            f"Question: {question}\n\nFindings:\n{chain}\n\nSupporting text:\n{final_context[:3000]}",
        )
        return final if final else findings.get(len(findings), "unknown")

    def _extract_with_verification(self, question: str, matched: list[dict]) -> str:
        """Extract answer from Glyphh-matched chunks, verify structurally.

        Try the top chunks first. If the extracted answer doesn't
        structurally match the question, try the next batch.
        """
        if not matched:
            return ""

        # First attempt: top 5 chunks
        context1 = "\n\n".join(m["text"] for m in matched[:5])
        ans1 = self._extract(question, context1)

        if ans1 and ans1.lower() not in ("unknown", "not found", ""):
            # Structural verification
            score = self.verifier.verify(question, ans1, context1)
            if score > 0.15:
                return ans1

            # Low structural match — try next batch of chunks
            if len(matched) > 5:
                context2 = "\n\n".join(m["text"] for m in matched[3:8])
                ans2 = self._extract(question, context2)
                if ans2 and ans2.lower() not in ("unknown", ""):
                    score2 = self.verifier.verify(question, ans2, context2)
                    if score2 > score:
                        return ans2

            return ans1  # Return first answer even if verification is low

        return ans1 or ""

    def _search_and_ingest(self, query: str, memory: SearchMemory, fetch_pages: int = 1):
        results = _search_ddg(query, max_results=5)
        good = [r for r in results if "error" not in r and r.get("snippet")]
        for r in good:
            memory.store(r["snippet"], source=r.get("title", ""))
        for r in good[:fetch_pages]:
            url = r.get("url", "")
            if not url:
                continue
            page = _fetch_url(url)
            if page:
                for chunk in _chunk_text(page):
                    memory.store(chunk, source=r.get("title", ""))

    def _extract(self, question: str, context: str) -> str:
        if not context.strip():
            return ""
        return self._llm(
            "Extract the specific factual answer to the question from the text.\n"
            "- Return ONLY the answer: a name, number, date, or place\n"
            "- For people, return their full name\n"
            "- For numbers/years, return just the number\n"
            "- If you cannot find the answer, return 'unknown'",
            f"Question: {question}\n\nText:\n{context[:4000]}",
        )

    def _decompose(self, question: str) -> list[str]:
        text = self._llm(
            "Break this complex question into simple sub-questions. "
            "Each sub-question must be answerable with one web search.\n"
            "Use [STEP_N] as placeholder for the answer from step N.\n"
            "Return ONLY a JSON array of strings.\n"
            "Include specific details (years, 'according to Forbes', etc.).\n"
            "Example: [\"What is the world's most expensive tea?\", "
            "\"Which country produces [STEP_1]?\", "
            "\"According to Forbes April 2025, who is the richest billionaire from [STEP_2]?\"]",
            question,
            max_tokens=600,
        )
        try:
            if text.startswith("```"):
                text = text.split("\n", 1)[1] if "\n" in text else text[3:]
                if text.endswith("```"):
                    text = text[:-3]
                text = text.strip()
            return json.loads(text)
        except Exception:
            return [question]
