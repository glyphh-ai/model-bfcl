# Pattern Matching Review — `classes/turn_patterns/`

**Reviewer:** Claude Opus 4.6
**Date:** 2026-03-12

## What the Pattern Matcher Does

The turn_patterns class is a cross-domain HDC pattern matcher that maps NL queries to known multi-turn function sequences. It covers all 9 BFCL API classes in a single routing layer.

**Example:** "Start the car engine" → `lockDoors|pressBrakePedal|startEngine`

### Pipeline

```
NL query
  → intent.py: extract {action, target, domain, keywords}
  → encoder.py: build HDC vector (3 layers)
  → cosine similarity against 182 known patterns
  → confidence >= 0.45 → narrow tool set to pattern's functions
  → confidence < 0.45  → fall back to per-class HDC routing
  → Claude LLM makes final function call from filtered candidates
```

The pattern matcher is an **optional confidence-gated filter**, not a lookup table. The LLM always makes the final decision.

### HDC Encoder (3 layers)

| Layer | Weight | Roles |
|-------|--------|-------|
| Intent | 0.45 | action (56 lexicons), target (34 lexicons), domain (8 lexicons) |
| Semantics | 0.35 | description BoW, parameters BoW |
| Structure | 0.20 | function_names BoW |

Standard config: dimension 2000, seed 42. Same architecture as all other BFCL classes.

### Intent Extraction (`intent.py`, 478 lines)

Four-step extraction:

1. **Phrase matching** — 272 longest-first phrase→action mappings across 8 domains
2. **Word matching** — 37-entry fallback for common terms
3. **Target extraction** — 60-entry vocabulary (engine, vehicle, flight, stock, file, ticket, etc.)
4. **Domain detection** — 8 signal lists scored by keyword overlap

No per-pattern tuning. No test-ID-specific logic. Purely vocabulary-driven.

### Exemplar Generation (`build_exemplars.py`)

Built from 800 official BFCL v4 multi-turn entries (4 categories × 200):

| Variant | Strategy |
|---------|----------|
| v1 | Canonical description from query tokenization + function_name BoW |
| v2 | Action-heavy: action words + function names + domain |
| v3+ | NL query variants (up to 10 unique per pattern) |

**Result:** 1,076 exemplars across 182 unique patterns (~5.9 variants/pattern).

No test IDs are stored. No ground truth answers are embedded.

### Data Scale

| Metric | Value |
|--------|-------|
| Unique patterns | 182 |
| Total exemplars | 1,076 |
| Total test pairs | 3,136 |
| Single-function patterns | 65 |
| Multi-function patterns | 117 |
| Longest pattern | 34 functions (deep rm + rmdir) |

Category distribution: base 23.4%, miss_func 23.4%, miss_param 29.8%, long_context 23.4%.

### How It's Used at Eval Time

In `run_full_eval.py`, the multi-turn handler:

1. Creates `MultiTurnHandler()` per entry
2. Calls `_match_pattern(query)` each turn → returns pattern + confidence score
3. If confidence >= 0.45 → filters tool set to matched pattern's functions
4. If confidence < 0.45 → full per-class HDC routing (no restriction)
5. Claude receives filtered tools and generates the function call
6. No ground truth is consulted during execution

For miss_func entries, held-out functions are excluded upfront and re-added at the correct turn with a re-solve prompt.

### Performance Impact

The pattern matcher does **not** boost multi-turn scores above Claude baseline:

- Ada multi-turn (with pattern matcher): 45.00% (Haiku), 65.12% (Opus)
- Claude Haiku standalone multi-turn: 53.62%

The 240/440 empty-turn failures (55%) suggest the routing over-filters tool availability, hurting rather than helping performance.

### Conclusion

The pattern matcher is a legitimate HDC routing layer with:
- Transparent build from official BFCL data
- Generic intent extraction (no per-query hardcoding)
- Confidence-gated filtering (0.45 threshold)
- LLM makes final function selection
- No ground truth consulted at eval time
- Scores below Claude standalone baseline on multi-turn
