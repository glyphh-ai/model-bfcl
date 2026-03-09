"""Per-class intent overrides for MathAPI.

Maps NL verbs/targets to the 18 math functions.
Used by the per-class encoder for Stage 2 routing.
"""

CLASS_ALIASES = ["MathAPI", "math", "mathematics", "calculator", "compute"]
CLASS_DOMAIN = "math"

# NL verb → canonical function name (bare, without class prefix)
ACTION_SYNONYMS = {
    # General compute
    "compute": "add", "figure": "add", "work out": "add",
    # Addition
    "sum": "add", "total": "add", "plus": "add",
    # Subtraction
    "minus": "subtract", "take away": "subtract", "difference": "subtract",
    # Multiplication
    "times": "multiply", "product": "multiply",
    # Division
    "divided": "divide", "quotient": "divide", "ratio": "divide",
    # Square root
    "sqrt": "square_root", "square root": "square_root", "root": "square_root",
    # Power
    "raise": "power", "exponent": "power", "to the power": "power",
    # Logarithm
    "log": "logarithm", "natural log": "logarithm",
    # Mean
    "average": "mean", "avg": "mean",
    # Percentage
    "percent": "percentage",
    # Absolute value
    "abs": "absolute_value", "absolute": "absolute_value",
    # Rounding
    "round": "round_number", "truncate": "round_number",
    # GCD / LCM
    "greatest common": "gcd", "gcd": "gcd",
    "least common": "lcm", "lcm": "lcm",
    # Max / Min
    "maximum": "max_value", "highest": "max_value",
    "minimum": "min_value", "lowest": "min_value", "smallest": "min_value",
    # Conversion
    "convert": "si_unit_conversion", "convert units": "si_unit_conversion",
    "imperial": "imperial_si_conversion",
    # Standard deviation
    "std dev": "standard_deviation", "deviation": "standard_deviation",
    "spread": "standard_deviation",
}

# NL noun → canonical target for encoder lexicon matching
TARGET_OVERRIDES = {
    "value": "number", "result": "number", "answer": "number",
    "calculation": "number", "sum": "number", "product": "number",
    "quotient": "number", "remainder": "number", "output": "number",
}

# Function name → (action, target) for encoder
FUNCTION_INTENTS = {
    "absolute_value":           ("calculate", "number"),
    "add":                      ("calculate", "number"),
    "divide":                   ("calculate", "number"),
    "gcd":                      ("calculate", "number"),
    "imperial_si_conversion":   ("convert", "number"),
    "lcm":                      ("calculate", "number"),
    "logarithm":                ("calculate", "number"),
    "max_value":                ("find", "number"),
    "mean":                     ("calculate", "number"),
    "min_value":                ("find", "number"),
    "multiply":                 ("calculate", "number"),
    "percentage":               ("calculate", "number"),
    "power":                    ("calculate", "number"),
    "round_number":             ("calculate", "number"),
    "si_unit_conversion":       ("convert", "number"),
    "square_root":              ("calculate", "number"),
    "standard_deviation":       ("calculate", "number"),
    "subtract":                 ("calculate", "number"),
}
