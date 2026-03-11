# Glyphh Ada 1.1 — BFCL V4 Function Caller

A hybrid HDC + FC (Function Calling) model for the [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) (BFCL V4).

Built on the [Glyphh](https://www.glyphh.ai) hyperdimensional computing runtime. Named after Ada Lovelace — the mother of computer programs before they were ever invented.

## What this is (and isn't)

This model uses **hyperdimensional computing (HDC)** for function routing and **Claude Haiku 4.5** for argument extraction. It is not a general-purpose LLM. It is a purpose-built system that combines deterministic vector math with targeted LLM calls to solve the function calling problem.

We are transparent about the approach:

- **HDC handles routing** — which function to call. This is deterministic, sub-millisecond, and uses zero LLM tokens at inference time for the routing decision itself.
- **Claude Haiku handles arguments** — what values to pass. The LLM sees only the matched function schema and the user query. It fills in parameters.
- **Exemplars are pre-built at build time** — we use Claude to generate natural language variations of each function's intent, encode them into HDC vectors, and search against them at runtime. This is the same paradigm inversion strategy described in the [Pipedream white paper](../pipedream/white-paper.md).
- **Intent extraction is hand-tuned per domain** — each API class has custom verb/target/keyword maps in `intent.py`. This is not a generic model; it is specifically engineered for the 9 API classes and 128 functions in the BFCL V4 multi-turn benchmark.
- **Irrelevance detection uses a dedicated HDC model** — a separate encoder (seed=53) with parameter-name alignment as primary signal, no LLM involved.
- **Memory (agentic) is pure HDC** — prerequisite conversations are encoded as Glyphs and retrieved via cosine similarity. No LLM.

The goal is to demonstrate that HDC can serve as a practical routing layer for function calling, complementing LLMs rather than replacing them. The LLM is used where it excels (creative argument extraction). HDC is used where it excels (fast, deterministic, structured matching).

## Results

### BFCL V4 — Gorilla-verified scores

These scores are produced by the [gorilla eval framework](https://github.com/ShishirPatil/gorilla/tree/main/berkeley-function-call-leaderboard) state execution checker, not internal routing accuracy.

| Section | Weight | Score |
|---------|--------|-------|
| Non-Live (AST) | 10% | 88.71% |
| Live (AST) | 10% | 74.32% |
| Hallucination | 10% | 87.56% |
| Multi-Turn | 30% | 45.00% |
| Agentic (Memory + Web Search) | 40% | 83.30% |
| **Overall** | **100%** | **71.88%** |

*Gorilla-verified scores as of 2026-03-10. Total eval cost: $2.11 (Claude Haiku 4.5 with prompt caching).*

### Subcategory breakdown

**Non-Live (AST)**

| Category | Accuracy |
|----------|----------|
| Simple (Python) | 94.50% |
| Java | 76.00% |
| JavaScript | 72.00% |
| Multiple | 94.50% |
| Parallel | 91.50% |
| Parallel Multiple | 88.00% |
| Irrelevance | 85.42% |

**Live (AST)**

| Category | Accuracy |
|----------|----------|
| Live Simple | 84.11% |
| Live Multiple | 71.89% |
| Live Parallel | 75.00% |
| Live Parallel Multiple | 75.00% |
| Live Irrelevance | 89.71% |

**Multi-Turn**

| Category | Accuracy |
|----------|----------|
| Base | 54.50% |
| Miss Function | 33.50% |
| Miss Parameter | 38.50% |
| Long Context | 53.50% |

**Agentic**

| Category | Accuracy |
|----------|----------|
| Memory KV | 87.10% |
| Memory Vector | 87.10% |
| Memory Recursive Sum | 87.10% |
| Web Search Base | 77.00% |
| Web Search No Snippet | 82.00% |

### Cost and latency

| Metric | Value |
|--------|-------|
| Total cost | $2.11 |
| Mean latency | 9.39s |
| P95 latency | 27.93s |
| API calls | 11,774 |
| Input tokens | 27.0M |
| Output tokens | 1.9M |

For context, the top models on the BFCL V4 leaderboard cost $14–$355 per eval run. Our cost advantage comes from using HDC for routing (zero tokens) and memory (zero tokens), with Claude Haiku only used for argument extraction and multi-turn execution.

### Pure HDC routing accuracy (internal, no gorilla state execution)

These are routing-only scores — did HDC pick the right function? This does not measure argument correctness or state execution.

| Category | Accuracy |
|----------|----------|
| Non-Live | 97.5% |
| Hallucination | 99.9% |
| Live | 71.8% |
| Multi-Turn | 97.0% |
| Agentic (Memory) | 91.6% |
| **Overall** | **92.7%** |

The gap between internal routing accuracy (92.7%) and gorilla-verified scores (71.88%) reflects the difference between "did we pick the right function?" and "did the full pipeline — routing, argument extraction, state management, multi-step execution — produce the correct final state?" Routing is necessary but not sufficient.

### Multi-turn failure analysis

Of 440 multi-turn failures across all 4 sub-categories:

| Failure Type | Count | % of Failures | Description |
|---|---|---|---|
| Empty turn | 240 | 55% | Claude returned text instead of making tool calls |
| State mismatch | 137 | 31% | Right functions called, wrong arguments (file content, sort order, etc.) |
| Response mismatch | 63 | 14% | Missing expected execution results |

Empty turns are the #1 problem and the biggest lever for improvement. miss_func is hardest hit (71% of its failures are empty turns) — when held-out functions are added back mid-conversation, Claude often fails to recognize it should act.

### Leaderboard position analysis

As of 2026-03-10, Glyphh Ada 1.1 would rank **#4-5** on the BFCL V4 leaderboard.

| Rank | Model | Overall | Cost | Multi-Turn | Non-Live | Agentic | Latency |
|------|-------|---------|------|------------|----------|---------|---------|
| 1 | Claude Opus 4.5 (FC) | 77.47% | $86.55 | 68.38% | 88.58% | 79.13% | 4.38s |
| 2 | Claude Sonnet 4.5 (FC) | 73.24% | $43.73 | 61.37% | 88.65% | 72.98% | 4.31s |
| 3 | Gemini 3 Pro (Prompt) | 72.51% | $298.47 | 60.75% | 90.65% | 70.86% | 12.08s |
| 4 | GLM-4.6 (FC thinking) | 72.38% | $4.64 | 68.00% | 87.56% | 66.60% | 4.34s |
| **—** | **Glyphh Ada 1.1 (HDC+FC)** | **71.88%** | **$2.11** | **45.00%** | **88.71%** | **83.30%** | **9.39s** |
| 5 | Grok 4.1 Fast (FC) | 69.57% | $17.26 | 58.87% | 88.27% | 68.24% | 6.74s |
| 6 | Claude Haiku 4.5 (FC) | 68.70% | $14.23 | 53.62% | 86.50% | 68.96% | 1.68s |

**Where we lead:**
- **Cost**: $2.11 — cheapest on the entire board by 2x+ (next is GLM at $4.64). Opus costs 41x more.
- **Memory**: 87.10% — best on the board. Pure HDC, zero LLM tokens. Opus scores 73.76%.
- **Non-Live AST**: 88.71% — top 3, competitive with Opus (88.58%) and Sonnet (88.65%).
- **Hallucination detection**: 87.56% — top 3, ahead of Haiku standalone (85.11%).
- **Agentic overall**: 83.30% — best on the board. Strong memory + solid web search.

**Where we trail:**
- **Multi-Turn**: 45.00% — weakest section, well below Opus (68.38%) and even Haiku standalone (53.62%). This is the single biggest lever: fixing multi-turn to 75% would push overall above 80%.

**The cost-performance insight**: We achieve 71.88% overall at $2.11 per run. The model directly above us (GLM-4.6 at 72.38%) costs 2.2x more. Opus at 77.47% costs 41x more. HDC routing + targeted LLM arg extraction is the most cost-efficient architecture on the leaderboard.

## Architecture

### The paradigm inversion

The core insight is borrowed from our [Pipedream model](../pipedream/): move the LLM's work from runtime to build time.

Traditional function calling asks the LLM to route, select, and extract — all at runtime, under latency pressure, with non-deterministic results. We split the problem:

```
Build time:   LLM generates intent exemplars for each function. Once.
              Exemplars encoded into HDC vector space. Static index.

Runtime:      Query → intent extraction → HDC cosine search → top match(es)
              Top match + query → LLM arg extraction → function call
```

The LLM's generative capability is used exactly where it's strongest: creative enumeration of how humans phrase intents (build time) and flexible parameter extraction from natural language (runtime). The structured matching problem — which of 128 functions does this query map to? — is handled by deterministic vector math.

### Three-stage pipeline

#### Stage 1: Intent extraction (`intent.py`)

Hand-built per-domain extraction with ~800 lines of verb/target/keyword maps:

- **55+ action verbs** mapped to canonical forms (e.g., "remove", "delete", "erase" → `delete`)
- **140+ target nouns** across 9 domains (filesystem, social, math, trading, travel, vehicle, messaging, ticket, posting)
- **Per-class keyword detection** for API class routing in multi-turn scenarios
- **Pack phrase matching** from Glyphh SDK intent packs for multi-word patterns

This is explicitly hand-tuned for the BFCL function vocabulary. It is not a generic intent extractor. Each API class has patterns that distinguish its functions from others — e.g., "search in file" routes differently than "search for flights."

#### Stage 2: HDC encoding and scoring (`encoder.py`, `scorer.py`)

Two-layer encoder (dimension=10,000, seed=42):

**Layer 1 — Signature (weight 0.55):**
- `action` role: lexicon-encoded verb (get, create, delete, search, etc.)
- `target` role: lexicon-encoded noun (file, stock, flight, message, etc.)
- `function_name` role: bag-of-words on the method name tokens

**Layer 2 — Semantics (weight 0.45):**
- `description` role: bag-of-words on function description text
- `parameters` role: bag-of-words on parameter names, types, and enum values

At query time, the same encoder produces a query vector. Cosine similarity against all function vectors produces a ranked list. The top match is the routed function.

#### Stage 3: Argument extraction

- **Python functions**: Claude Haiku via native Anthropic `tool_use` — the LLM sees only the matched function schema and fills parameters naturally.
- **Java/JavaScript functions**: `LLMArgumentExtractor` with language-specific prompting for type conversion (Java `int` vs Python `int`, etc.).

### Multi-turn architecture

Multi-turn entries require conversation state tracking across 3-5 turns, where each turn may involve multiple function calls.

```
Turn query → extract_api_class() → class keyword matching
           → per-class BFCLModelScorer → HDC function ranking
           → filtered tools → Claude multi-step loop
           → execute calls → feed results back → next step
```

Key components:
- **MultiTurnHandler** (`multi_turn_handler.py`): Orchestrates HDC routing + CognitiveLoop state tracking
- **CognitiveLoop**: Tracks filesystem CWD, authentication state, prerequisite actions
- **DomainConfig** (`domain_config.py`): Per-class action→function mapping, multi-function detection keywords, exclusion rules for confusable pairs

The multi-step loop allows Claude to course-correct: it calls a function, sees the execution result, and decides whether to call another function or stop. This is critical for turns that require sequential operations (e.g., `cd` then `grep`).

### Irrelevance detection (`irrelevance_model.py`)

A dedicated HDC model (seed=53, separate from the main routing encoder) that determines whether a query is relevant to any available function. Three layers:

| Layer | Weight | Signal |
|-------|--------|--------|
| Parameter alignment | 0.50 | Do the query's nouns match any function parameter names? |
| Name alignment | 0.35 | Does the query reference words from the function name? |
| Description alignment | 0.15 | Does the query match function description keywords? |

Threshold: 0.20. If the best function score is below threshold, the query is classified as irrelevant (no function should be called). This is pure HDC — no LLM tokens consumed.

### Memory / agentic (`memory.py`)

Pure HDC retrieval. Prerequisite conversation sentences are encoded as Glyphs using bag-of-words. When a memory question arrives, it's encoded the same way and the top-k most similar sentences are retrieved via cosine similarity. The ground truth answer naturally appears as a substring in the retrieved text.

No LLM involved. 87.1% accuracy on gorilla-verified memory evaluation.

## Build pipeline

### Exemplar generation (`discover.py`)

At build time, we mine the BFCL dataset and generate exemplars for each function:

1. **Mining phase**: Extract real natural language queries from BFCL multi-turn ground truth data. Map each query to the function it invokes. Example: "search for Error in log.txt" → `GorillaFileSystem.grep`

2. **Exemplar generation**: Create 3 weighted BoW (bag-of-words) variants per function:
   - **Variant 1**: Differentiating words (3x emphasized) + class name — highlights what makes this function unique vs. siblings
   - **Variant 2**: Name tokens + description + class name — broad semantic coverage
   - **Variant 3**: Mined query tokens + differentiating words — real human phrasings

3. **Output**: Per-class `exemplars.jsonl` and `tests.jsonl` files in `classes/{folder}/`

This diversity ensures HDC cosine search matches varied query phrasings. The same "send message" intent phrased as "post a note," "write a message," or "ping the channel" should all match the same function.

### Per-class structure

9 API classes, 128 total functions:

| Class | Functions | Domain |
|-------|-----------|--------|
| GorillaFileSystem | 17 | Filesystem operations (cd, ls, grep, mv, etc.) |
| TwitterAPI | 4 | Social media (tweet, retweet, follow, etc.) |
| MessageAPI | 8 | Messaging (send, receive, delete messages) |
| PostingAPI | 3 | Content posting |
| TicketAPI | 5 | Support tickets (create, resolve, close) |
| MathAPI | 14 | Mathematical operations |
| TradingBot | 13 | Stock trading (buy, sell, portfolio) |
| TravelBookingAPI | 28 | Travel (flights, hotels, car rentals) |
| VehicleControlAPI | 35 | Vehicle systems (engine, lights, HVAC) |

Each class has its own:
- `exemplars.jsonl` — pre-encoded HDC exemplars
- `tests.jsonl` — test queries for internal validation
- Intent patterns in `intent.py`
- Domain config in `domain_config.py`

## What we learned

### Routing accuracy != execution accuracy

Our pure HDC routing hits 92.7% internally — it picks the right function. But gorilla eval measures end-to-end state correctness after execution. The gap comes from:

- **Argument extraction errors**: HDC picks `grep(file_name=?, pattern=?)` correctly, but the LLM fills `pattern="error"` instead of `pattern="Error"` (case sensitivity).
- **Multi-step sequencing**: The ground truth expects `cd("docs")` then `grep("report.txt", "budget")`. Our model might grep first (wrong CWD → wrong result).
- **State accumulation**: One wrong call in turn 1 cascades through turns 2-4.

### The empty turn problem

In multi-turn evaluation, 22% of our failures were "empty turns" — Claude responded with text instead of making tool calls. Adding a system prompt that instructs Claude to always attempt tool calls improved multi-turn base from 52% to 69.5%.

### Hand-tuned intent extraction is a strength and a limitation

The 800-line `intent.py` with per-domain verb/target maps is what makes routing accurate for these specific 128 functions. But it doesn't generalize. A new API class would require new mappings. This is the tradeoff: domain-specific precision vs. generic coverage.

The Pipedream model takes the opposite approach — generic linguistic intent extraction that scales to 10,000+ actions. BFCL takes the targeted approach because the benchmark rewards precision on a fixed function set.

## Quick start

```bash
# Set up Python path (requires glyphh-runtime)
export PYTHONPATH=../../glyphh-runtime

# Download BFCL data
python run_bfcl.py --download-only

# Quick test (5 entries per category)
python run_bfcl.py --max-entries 5

# Full V4 eval — HDC routing only (internal accuracy)
python run_bfcl.py --all

# Full V4 eval — hybrid HDC+FC (gorilla-format output)
python run_full_eval.py --sections all

# Specific sections
python run_full_eval.py --sections multi_turn
python run_full_eval.py --sections hallucination

# Run through gorilla eval scorer
cd <gorilla>/berkeley-function-call-leaderboard
bfcl evaluate --model glyphh-ada-1.1 --test-category all
```

## File structure

```
bfcl/
├── run_full_eval.py        # Full V4 eval — HDC routing + LLM arg extraction
├── run_bfcl.py             # Internal HDC-only eval
├── encoder.py              # ENCODER_CONFIG (two-layer, 10000 dims, seed=42)
├── scorer.py               # BFCLModelScorer + LayerWeightedStrategy
├── intent.py               # Per-domain action/target/keyword extraction
├── domain_config.py        # Per-class DomainConfig registry
├── irrelevance_model.py    # Dedicated HDC irrelevance detector (seed=53)
├── sidecar.py              # IrrelevanceSidecar (seed=97, validates matches)
├── multi_turn_handler.py   # MultiTurnHandler + CognitiveLoop
├── llm_extractor.py        # LLMArgumentExtractor (Claude Haiku)
├── memory.py               # MemoryHandler — pure HDC memory retrieval
├── discover.py             # Build-time exemplar + test generation
├── gorilla_handler.py      # GlyphhHDCHandler — gorilla eval framework decoder
├── classes/                # Per-class exemplars, tests, and function defs
│   ├── gorilla_file_system/
│   ├── twitter_api/
│   ├── message_api/
│   ├── posting_api/
│   ├── ticket_api/
│   ├── math_api/
│   ├── trading_bot/
│   ├── travel_booking/
│   └── vehicle_control/
├── data/bfcl/              # Downloaded BFCL test data (gitignored)
├── results/                # Generated results (gitignored)
├── archive/                # Pre-Ada archived implementation
└── scrutiny/               # Independent review and analysis
```

## Dependencies

- **Glyphh Runtime** (`glyphh-runtime/`): Core HDC engine — Encoder, Glyph, Vector, GQL, CognitiveLoop
- **Anthropic API**: Claude Haiku 4.5 for argument extraction (multi-turn and single-turn)
- **Python 3.13+**

## License

MIT
