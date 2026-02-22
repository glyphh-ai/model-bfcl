# BFCL Function Caller

Glyphh HDC model for the [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html) (BFCL).

## What this does

Unlike the SaaS Tool Router (which uses fixed domain exemplars), this model dynamically encodes **any** function definition into HDC vectors at query time. Given a set of function definitions and a natural language query, it routes to the correct function via cosine similarity — zero tokens, sub-10ms.

## Architecture

```
User Query → encode_query() → HDC vector
                                          } cosine similarity → best match
Function Defs → encode_function() → HDC vectors
```

The encoder extracts four signals from each function definition:
- **action** — the primary verb (get, create, calculate, etc.)
- **function_name** — camelCase/snake_case split into words
- **description** — keywords from the function description
- **parameters** — parameter names, types, enum values

## Evaluation strategies

| Strategy | Description | Tokens |
|----------|-------------|--------|
| Routing only | Glyphh picks the function, no args | 0 |
| Glyphh + LLM args | Glyphh routes, LLM fills arguments | minimal |

## Quick start

```bash
# Install glyphh
pip install glyphh

# Download BFCL data and run routing eval
python run_bfcl.py --download-only
python run_bfcl.py --routing-only --categories simple_python multiple irrelevance

# Quick test (first 20 entries per category)
python run_bfcl.py --max-entries 20

# Full eval with LLM args (needs OPENAI_API_KEY)
export OPENAI_API_KEY=sk-...
python run_bfcl.py --with-args --categories simple_python multiple parallel
```

## BFCL submission

To submit to the official BFCL leaderboard, integrate the `GlyphhBFCLHandler` class from `handler.py` into the gorilla evaluation framework. See the [BFCL Contributing Guide](https://github.com/ShishirPatil/gorilla/blob/main/berkeley-function-call-leaderboard/CONTRIBUTING.md).

## License

MIT
