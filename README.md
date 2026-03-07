# BFCL Function Caller

Glyphh HDC model for the [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) (BFCL V4).

Pure hyperdimensional computing — no LLM, no neural network. Deterministic HDC routing + schema-guided argument extraction.

## Results (V4)

| Category | Accuracy | Weight |
|----------|----------|--------|
| Non-Live | 97.5% | 10% |
| Hallucination | 99.9% | 10% |
| Live | 71.8% | 10% |
| Multi-Turn | 97.0% | 30% |
| Agentic (Memory) | 91.6% | 40% |
| **Overall** | **92.7%** | 100% |

## Architecture

```
User Query → encode_query() → HDC vector
                                        } cosine similarity → best match
Function Defs → encode_function() → HDC vectors
```

**Single-turn**: BFCLScorer encodes query + function defs into HDC vectors, picks best match via cosine similarity, then ArgumentExtractor fills parameters from the query using schema-guided extraction.

**Multi-turn**: Two-stage routing — `extract_api_class()` detects which API class a turn targets, then per-class BFCLScorer with pre-built exemplars scores via `score_multi()` gap analysis. DomainConfig keywords catch additional multi-function calls.

**Memory (Agentic)**: MemoryHandler encodes prerequisite conversation sentences as Glyphs, then retrieves top matches for each question via cosine similarity. Ground truth naturally appears as substring in retrieved text.

## Quick start

```bash
# Set up Python path (requires glyphh-runtime)
export PYTHONPATH=../../glyphh-runtime

# Download BFCL data
python run_bfcl.py --download-only

# Quick test (5 entries per category)
python run_bfcl.py --max-entries 5

# Full V4 eval (all categories)
python run_bfcl.py --all

# Specific categories
python run_bfcl.py --categories simple multiple multi_turn_base memory_kv

# With verbose output (show mismatches)
python run_bfcl.py --all --verbose

# Save results to disk
python run_bfcl.py --all --output results/
```

## Categories

| Category | Type | Description |
|----------|------|-------------|
| `simple` | Non-Live | Single function, Python |
| `java` | Non-Live | Single function, Java |
| `javascript` | Non-Live | Single function, JavaScript |
| `multiple` | Non-Live | Pick one from multiple candidates |
| `parallel` | Non-Live | Call multiple functions |
| `parallel_multiple` | Non-Live | Multiple calls from multiple candidates |
| `irrelevance` | Hallucination | No function should be called |
| `live_simple` | Live | Single function, real APIs |
| `live_multiple` | Live | Pick one, real APIs |
| `live_parallel` | Live | Multiple calls, real APIs |
| `live_parallel_multiple` | Live | Multiple from multiple, real APIs |
| `live_irrelevance` | Hallucination | No call, real APIs |
| `multi_turn_base` | Multi-Turn | Multi-step conversations |
| `multi_turn_miss_func` | Multi-Turn | Missing function variant |
| `multi_turn_miss_param` | Multi-Turn | Missing parameter variant |
| `multi_turn_long_context` | Multi-Turn | Long context variant |
| `memory_kv` | Agentic | Memory retrieval (KV backend) |
| `memory_vector` | Agentic | Memory retrieval (vector backend) |
| `memory_rec_sum` | Agentic | Memory retrieval (recursive summary) |

## Gorilla leaderboard submission

### Generate result files

```bash
# Generate gorilla-format JSONL files for all V4 categories
python run_bfcl.py --all --output-gorilla results/glyphh-hdc-v1

# This produces:
# results/glyphh-hdc-v1/
#   non_live/    (7 files: simple_python, java, javascript, multiple, parallel, parallel_multiple, irrelevance)
#   live/        (5 files: live_simple, live_multiple, live_parallel, live_parallel_multiple, live_irrelevance)
#   multi_turn/  (4 files: multi_turn_base, miss_func, miss_param, long_context)
#   agentic/     (3 files: memory_kv, memory_vector, memory_rec_sum)
```

### Wire into gorilla eval framework

1. **Clone the gorilla repo**:
   ```bash
   git clone https://github.com/ShishirPatil/gorilla.git
   cd gorilla/berkeley-function-call-leaderboard
   pip install -e .
   ```

2. **Copy the handler**:
   ```bash
   cp gorilla_handler.py \
      <gorilla>/berkeley-function-call-leaderboard/bfcl_eval/model_handler/api_inference/glyphh.py
   ```

3. **Copy pre-computed result files**:
   ```bash
   cp -r results/glyphh-hdc-v1/ \
      <gorilla>/berkeley-function-call-leaderboard/result/glyphh-hdc-v1/
   ```

4. **Register in model_config.py** (`bfcl_eval/constants/model_config.py`):
   ```python
   from bfcl_eval.model_handler.api_inference.glyphh import GlyphhHDCHandler

   # Add to api_inference_model_map:
   "glyphh-hdc-v1": ModelConfig(
       model_name="glyphh-hdc-v1",
       display_name="Glyphh HDC v1 (Prompt)",
       url="https://glyphh.com",
       org="Glyphh",
       license="Proprietary",
       model_handler=GlyphhHDCHandler,
       input_price=None,
       output_price=None,
       is_fc_model=False,
       underscore_to_dot=False,
   ),
   ```

5. **Register in supported_models.py** (`bfcl_eval/constants/supported_models.py`):
   ```python
   # Add to SUPPORTED_MODELS list:
   "glyphh-hdc-v1",
   ```

6. **Run evaluation**:
   ```bash
   bfcl evaluate --model glyphh-hdc-v1 --test-category all
   bfcl scores
   ```

### Handler details

`gorilla_handler.py` contains `GlyphhHDCHandler` — a standalone decoder that parses pre-computed result strings into AST and execute formats:

- `decode_ast(result)` — Parses JSON string `'[{"func": {"param": "val"}}]'` into `[{"func": {"param": "val"}}]`
- `decode_execute(result)` — Converts to executable format `["func(param='val')"]`
- Empty strings (irrelevance) → `[]`
- Multi-turn results are nested `list[list[str]]` where each inner list contains a JSON string of function calls for that turn

No inference happens in the handler — all computation is pre-computed by `run_bfcl.py --output-gorilla`.

## File structure

```
bfcl/
├── run_bfcl.py          # Main eval script + gorilla output generation
├── handler.py           # BFCLHandler — HDC routing + arg extraction
├── scorer.py            # BFCLScorer — HDC function scoring
├── encoder.py           # Encoder config for BFCL function defs
├── extractor.py         # ArgumentExtractor — schema-guided arg extraction
├── intent.py            # extract_api_class() for multi-turn routing
├── domain_config.py     # Per-class domain configs (multi-action keywords)
├── memory.py            # MemoryHandler — HDC memory retrieval for agentic
├── discover.py          # Exemplar + test query generation
├── gorilla_handler.py   # GlyphhHDCHandler — gorilla eval framework decoder
├── classes/             # Per-class exemplars and tests
│   ├── gorilla_file_system/
│   ├── twitter_api/
│   ├── message_api/
│   ├── posting_api/
│   ├── ticket_api/
│   ├── math_api/
│   ├── trading_bot/
│   ├── travel_booking/
│   └── vehicle_control/
├── data/bfcl/           # Downloaded BFCL test data (gitignored)
└── results/             # Generated results (gitignored)
```

## License

MIT
