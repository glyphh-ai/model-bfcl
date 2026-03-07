"""Per-class intent overrides for MathAPI."""

CLASS_ALIASES = ["MathAPI", "math", "mathematics", "calculator"]
CLASS_DOMAIN = "math"

ACTION_SYNONYMS = {
    "compute": "calculate", "figure": "calculate", "work out": "calculate",
    "determine": "calculate", "evaluate": "calculate",
    "average": "mean", "sum": "add", "total": "add",
}

TARGET_OVERRIDES = {
    "value": "number", "result": "number", "answer": "number",
    "calculation": "number",
}
