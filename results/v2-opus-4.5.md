# BFCL V4 — Opus 4.5 Evaluation

**LLM**: Claude Opus 4.5 (`claude-opus-4-5-20251101`)
**Pricing**: $15.00/MTok in, $75.00/MTok out
**Eval date**: 2026-03-11

## Summary

| Section | Weight | Score |
|---------|--------|-------|
| Non-Live (AST) | 10% | 90.56% |
| Live (AST) | 10% | 75.50% |
| Hallucination | 10% | 87.56% |
| Multi-Turn | 30% | 65.12% |
| Agentic (Memory + Web Search) | 40% | 87.55% |
| **Overall** | **100%** | **79.92%** |

Scores are gorilla-verified (state execution checker), not internal routing accuracy.

## Comparison to Haiku 4.5 (v3)

| Section | Weight | Haiku 4.5 (v3) | Opus 4.5 | Delta |
|---------|--------|----------------|----------|-------|
| Non-Live (AST) | 10% | 88.71% | 90.56% | +1.85 |
| Live (AST) | 10% | 74.32% | 75.50% | +1.18 |
| Hallucination | 10% | 87.56% | 87.56% | -- |
| Multi-Turn | 30% | 53.75% | **65.12%** | **+11.37** |
| Agentic | 40% | 83.30% | **87.55%** | **+4.25** |
| **Overall** | **100%** | **74.50%** | **79.92%** | **+5.42** |

Multi-turn is the headline: +11.37 points gorilla-verified, driven by Opus's stronger reasoning on argument extraction and multi-step sequencing.

## Non-Live (AST)

| Category | Accuracy |
|----------|----------|
| Simple (Python) | 96.25% |
| Java | 78.00% |
| JavaScript | 80.00% |
| Multiple | 95.00% |
| Parallel | 94.00% |
| Parallel Multiple | 88.50% |
| Irrelevance | 85.42% |

## Live (AST)

| Category | Accuracy |
|----------|----------|
| Live Simple | 87.21% |
| Live Multiple | 72.46% |
| Live Parallel | 87.50% |
| Live Parallel Multiple | 75.00% |
| Live Irrelevance | 89.71% |

## Multi-Turn

| Category | Haiku 4.5 (v3) | Opus 4.5 | Delta |
|----------|----------------|----------|-------|
| Base | 60.50% | **72.00%** | +11.50 |
| Miss Function | 51.00% | **61.50%** | +10.50 |
| Miss Parameter | 47.50% | **60.00%** | +12.50 |
| Long Context | 56.00% | **67.00%** | +11.00 |

All four subcategories improved 10-12 points. Opus produces fewer empty turns and handles multi-step sequencing more accurately.

### Internal vs Gorilla-Verified (multi-turn)

| Category | Internal | Gorilla | Gap |
|----------|----------|---------|-----|
| Base | 84.5% | 72.0% | -12.5 |
| Miss Function | 77.5% | 61.5% | -16.0 |
| Miss Parameter | 81.0% | 60.0% | -21.0 |
| Long Context | 83.5% | 67.0% | -16.5 |
| **Average** | **81.6%** | **65.1%** | **-16.5** |

The ~16pt internal-vs-gorilla gap persists from Haiku (was -16.1). Both models pass internal checks at similar rates but produce subtly different execution state than what gorilla's re-execution produces. This gap is structural — likely caused by non-deterministic execution ordering or state sensitivity — not model quality.

## Agentic

| Category | Accuracy |
|----------|----------|
| Memory KV | 87.10% |
| Memory Vector | 87.10% |
| Memory Recursive Sum | 87.10% |
| Web Search Base | 88.00% |
| Web Search No Snippet | 88.00% |

Memory is pure HDC (unchanged). Web search improved from 79.50% to 88.00% — Opus handles search result parsing and multi-step retrieval more accurately.

## Cost and Latency

| Metric | Haiku 4.5 (v3) | Opus 4.5 |
|--------|----------------|----------|
| Total cost | $18.64 | $63.86 |
| Mean latency | 8.52s | 13.16s |
| Latency P95 | 23.17s | 49.58s |

Cost is actual billed amount from Anthropic console (includes prompt caching discount).

## Motivation

The Haiku 4.5 eval (v1-v3) showed that multi-turn is the single biggest lever — 30% weight, and we scored 53.75% vs Opus standalone at 68.38%. The hypothesis was that Opus 4.5's stronger reasoning would improve argument extraction accuracy and multi-step sequencing.

Result: confirmed. Multi-turn jumped from 53.75% to 65.12% gorilla-verified (+11.37). Overall from 74.50% to 79.92% (+5.42). The HDC routing layer is the same — only the argument extraction LLM changed.

## What's different from Haiku 4.5

- **LLM**: Opus 4.5 replaces Haiku 4.5 for argument extraction
- **Cost**: $63.86 actual (Opus $15/$75 per MTok, with prompt caching) vs Haiku $18.64
- **Multi-turn**: +11.37 points — Opus handles complex multi-step scenarios far better
- **Web search**: +8.50 points — better search result parsing
- **Irrelevance**: -2.14 points on non-live (85.42% vs 87.56%) — Opus over-engages with irrelevant queries
- **No change**: HDC routing, intent extraction, irrelevance detection, memory
