# BFCL Pre-Ada Archive — 2026-03-09

Snapshot of the BFCL multi-turn implementation before the Glyphh Ada 1.0 rewrite.

## Results at time of archive

- **Internal scoring**: 95% on 20 multi-turn base entries (HDC routing + Haiku arg extraction)
- **Gorilla eval**: 10% (2/20 pass) — state execution mismatch

### V4 scores (internal, pure HDC routing):
- Non-Live 97.5%
- Hallucination 99.9%
- Live 71.8%
- Multi-Turn 97.0%
- Agentic (memory) 91.6%
- Overall 92.7% on 100% V4 weight

## Architecture

Two-tier routing: TaskScorer (query -> function sequence from exemplars) -> CognitiveLoop (PERCEIVE+RECALL+DEDUCE+RESOLVE) -> BFCLScorer fallback.

LLM (Claude Haiku) handles argument extraction.

## Why archived

The TaskScorer approach matched whole function sequences from ground truth exemplars,
which was too rigid — each entry has different available functions, state, and requirements.
Cross-class exemplar pollution and extra-function injection caused 10% gorilla eval accuracy
despite 95% internal routing accuracy.

Starting fresh with Glyphh Ada 1.0.
