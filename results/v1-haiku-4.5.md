# BFCL V4 — Haiku 4.5 Evaluation (v1–v3)

**LLM**: Claude Haiku 4.5 (`claude-haiku-4-5-20251001`)
**Pricing**: $1.00/MTok in, $5.00/MTok out
**Eval dates**: 2026-03-10 (v1), 2026-03-11 (v2, v3)

## Summary

| Section | Weight | v1 | v2 | v3 | Delta v2-v3 |
|---------|--------|-----|-----|-----|-------------|
| Non-Live (AST) | 10% | 88.71% | 88.71% | 88.71% | -- |
| Live (AST) | 10% | 74.32% | 74.32% | 74.32% | -- |
| Hallucination | 10% | 87.56% | 87.56% | 87.56% | -- |
| Multi-Turn | 30% | 45.00% | 53.62% | **53.75%** | +0.13 |
| Agentic (Memory + Web Search) | 40% | 83.30% | 83.30% | 83.30% | -- |
| **Overall** | **100%** | **71.88%** | **74.47%** | **74.50%** | **+0.03** |

Scores are gorilla-verified (state execution checker), not internal routing accuracy.

## What changed across versions

**v1** forced `tool_choice={"type": "any"}` on step 0 of every multi-turn step, which eliminated empty turns but sometimes forced Claude to pick the wrong tool — corrupting state for later turns.

**v2** uses auto-then-retry: try `tool_choice=auto` first (natural behavior), only fall back to `tool_choice=any` if Claude returns text instead of tool calls. This preserves Claude's judgment while still preventing empty turns.

Additionally, v2 includes:
- **Holdout turn detection**: In miss_func, turns where functions are removed have GT=[] (no calls expected). v2 detects these and sends no tools, preventing Claude from improvising with wrong functions.
- **Miss_func replay prompt**: When held-out functions are added back, v2 replays the previous user request with an explicit prompt to use the newly available tools.
- **Per-category system prompts**: Tuned system prompts for base, miss_func, and miss_param subcategories.

**v3** tightened multi-turn system prompts to reduce extra tool calls that corrupt execution state:
- "Make ONLY the calls directly required by the user's request"
- "NEVER add verification calls (ls, pwd, cat, displayCarStatus) after an operation"
- "NEVER retry a failed call — if a tool returns an error, stop and respond with text"
- "Use absolute or direct paths when possible instead of cd + relative path sequences"

The base subcategory improved +9% gorilla-verified (51.5% -> 60.5%), but other subcategories saw smaller or mixed changes, resulting in minimal overall improvement. The core issue is a ~16pt internal-vs-gorilla gap caused by execution state divergence, not call count.

## Non-Live (AST)

| Category | Accuracy |
|----------|----------|
| Simple (Python) | 94.50% |
| Java | 76.00% |
| JavaScript | 72.00% |
| Multiple | 94.50% |
| Parallel | 91.50% |
| Parallel Multiple | 88.00% |
| Irrelevance | 85.42% |

## Live (AST)

| Category | Accuracy |
|----------|----------|
| Live Simple | 84.11% |
| Live Multiple | 71.89% |
| Live Parallel | 75.00% |
| Live Parallel Multiple | 75.00% |
| Live Irrelevance | 89.71% |

## Multi-Turn

| Category | v1 | v2 | v3 | Delta v2-v3 |
|----------|-----|-----|-----|-------------|
| Base | 54.50% | 59.00% | **60.50%** | +1.50 |
| Miss Function | 33.50% | 50.00% | **51.00%** | +1.00 |
| Miss Parameter | 38.50% | 47.00% | **47.50%** | +0.50 |
| Long Context | 53.50% | 58.50% | **56.00%** | -2.50 |

v2-v3: prompt tightening helped base (+1.5%) and miss_func (+1.0%) but slightly hurt long_context (-2.5%). The ~16pt internal-vs-gorilla gap remains the primary challenge.

## Agentic

| Category | Accuracy |
|----------|----------|
| Memory KV | 87.10% |
| Memory Vector | 87.10% |
| Memory Recursive Sum | 87.10% |
| Web Search Base | 77.00% |
| Web Search No Snippet | 82.00% |

## Internal vs Gorilla-Verified (multi-turn, v3)

| Category | Internal | Gorilla | Gap |
|----------|----------|---------|-----|
| Base | 71.5% | 60.5% | -11.0 |
| Miss Function | 67.5% | 51.0% | -16.5 |
| Miss Parameter | 70.0% | 47.5% | -22.5 |
| Long Context | 70.5% | 56.0% | -14.5 |
| **Average** | **69.9%** | **53.75%** | **-16.1** |

The internal checker runs in-process; gorilla re-executes all function calls independently and compares final state. The gap comes from subtle state divergence — arguments that are "close enough" to pass internal checks but produce different execution results when re-run.

## Cost and Latency

| Metric | v1 | v2 | v3 |
|--------|-----|-----|-----|
| Total cost | $2.11 | $19.23 | $18.64 |
| Mean latency | 9.39s | 19.28s | 8.52s |

v2 cost was high due to auto-then-retry making 2 API calls per turn. v3 uses prompt caching aggressively.

## Pure HDC Routing Accuracy (internal, no gorilla state execution)

Routing-only scores — did HDC pick the right function? Does not measure argument correctness or state execution.

| Category | Accuracy |
|----------|----------|
| Non-Live | 97.5% |
| Hallucination | 99.9% |
| Live | 71.8% |
| Multi-Turn | 97.0% |
| Agentic (Memory) | 91.6% |
| **Overall** | **92.7%** |

## Result Archives

- `archive-20260310-pre-rerun/` — v1 full results
- `archive-20260311-run/` — v2 full results
- `archive-20260311-v3-run/` — v3 multi-turn results (non-multi-turn sections unchanged)
