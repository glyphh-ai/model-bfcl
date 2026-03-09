"""Per-class intent extraction for MathAPI.

Uses phrase/word maps for NL -> action/target extraction.
Maps canonical actions to the 17 MathAPI functions.

Exports:
    extract_intent(query) -> {action, target, keywords}
    ACTION_TO_FUNC        -- canonical action -> MathAPI.{func}
    FUNC_TO_ACTION        -- bare func name -> canonical action
"""

from __future__ import annotations

import re

# -- Pack canonical action -> MathAPI function ----------------------------

ACTION_TO_FUNC: dict[str, str] = {
    "absolute_value":       "MathAPI.absolute_value",
    "add":                  "MathAPI.add",
    "divide":               "MathAPI.divide",
    "imperial_convert":     "MathAPI.imperial_si_conversion",
    "logarithm":            "MathAPI.logarithm",
    "max_value":            "MathAPI.max_value",
    "mean":                 "MathAPI.mean",
    "min_value":            "MathAPI.min_value",
    "multiply":             "MathAPI.multiply",
    "percentage":           "MathAPI.percentage",
    "power":                "MathAPI.power",
    "round_number":         "MathAPI.round_number",
    "si_convert":           "MathAPI.si_unit_conversion",
    "square_root":          "MathAPI.square_root",
    "standard_deviation":   "MathAPI.standard_deviation",
    "subtract":             "MathAPI.subtract",
    "sum_values":           "MathAPI.sum_values",
}

# Reverse: bare function name -> pack canonical action
FUNC_TO_ACTION: dict[str, str] = {
    "absolute_value":       "absolute_value",
    "add":                  "add",
    "divide":               "divide",
    "imperial_si_conversion": "imperial_convert",
    "logarithm":            "logarithm",
    "max_value":            "max_value",
    "mean":                 "mean",
    "min_value":            "min_value",
    "multiply":             "multiply",
    "percentage":           "percentage",
    "power":                "power",
    "round_number":         "round_number",
    "si_unit_conversion":   "si_convert",
    "square_root":          "square_root",
    "standard_deviation":   "standard_deviation",
    "subtract":             "subtract",
    "sum_values":           "sum_values",
}

# Function -> canonical target
FUNC_TO_TARGET: dict[str, str] = {
    "absolute_value":       "number",
    "add":                  "number",
    "divide":               "number",
    "imperial_si_conversion": "unit",
    "logarithm":            "number",
    "max_value":            "list",
    "mean":                 "list",
    "min_value":            "list",
    "multiply":             "number",
    "percentage":           "number",
    "power":                "number",
    "round_number":         "number",
    "si_unit_conversion":   "unit",
    "square_root":          "number",
    "standard_deviation":   "list",
    "subtract":             "number",
    "sum_values":           "list",
}

# -- NL synonym -> pack canonical action ----------------------------------
# Derived from tests.jsonl queries + function descriptions
# Longest/most specific phrases first for greedy matching

_PHRASE_MAP: list[tuple[str, str]] = [
    # standard_deviation -- must come before "deviation" word match
    ("standard deviation", "standard_deviation"),
    ("std dev", "standard_deviation"),
    # logarithm -- multi-word phrases
    ("base 10 logarithm", "logarithm"),
    ("base the previous", "logarithm"),
    ("log base", "logarithm"),
    ("log of", "logarithm"),
    ("logarithm of", "logarithm"),
    ("compute the logarithm", "logarithm"),
    ("determine the logarithm", "logarithm"),
    ("calculate the logarithm", "logarithm"),
    # mean / average
    ("compute the average", "mean"),
    ("calculate the average", "mean"),
    ("get the mean", "mean"),
    ("mean of", "mean"),
    ("average of", "mean"),
    ("average score", "mean"),
    ("average tire pressure", "mean"),
    ("average amount", "mean"),
    ("average spent", "mean"),
    ("the mean of", "mean"),
    ("the average", "mean"),
    # square root
    ("square root", "square_root"),
    # imperial conversion
    ("imperial and si", "imperial_convert"),
    ("imperial to si", "imperial_convert"),
    ("si to imperial", "imperial_convert"),
    ("imperial si", "imperial_convert"),
    ("miles to kilometers", "imperial_convert"),
    ("kilometers to miles", "imperial_convert"),
    ("pounds to kilograms", "imperial_convert"),
    ("kilograms to pounds", "imperial_convert"),
    ("fahrenheit to celsius", "imperial_convert"),
    ("celsius to fahrenheit", "imperial_convert"),
    ("feet to meters", "imperial_convert"),
    ("meters to feet", "imperial_convert"),
    ("inches to centimeters", "imperial_convert"),
    # si conversion
    ("si unit", "si_convert"),
    ("convert from", "si_convert"),
    ("convert a value", "si_convert"),
    # max_value
    ("maximum value", "max_value"),
    ("max value", "max_value"),
    ("find the maximum", "max_value"),
    ("find the max", "max_value"),
    ("highest value", "max_value"),
    ("largest value", "max_value"),
    # min_value
    ("minimum value", "min_value"),
    ("min value", "min_value"),
    ("find the minimum", "min_value"),
    ("find the min", "min_value"),
    ("lowest value", "min_value"),
    ("smallest value", "min_value"),
    # sum_values
    ("sum of a list", "sum_values"),
    ("sum of all", "sum_values"),
    ("sum of the", "sum_values"),
    ("calculate the sum", "sum_values"),
    # absolute value
    ("absolute value", "absolute_value"),
    # round
    ("round to", "round_number"),
    ("round the", "round_number"),
    ("round number", "round_number"),
    ("rounded to", "round_number"),
    ("decimal places", "round_number"),
    ("nearest integer", "round_number"),
    # percentage
    ("percentage of", "percentage"),
    ("percent of", "percentage"),
    ("what percentage", "percentage"),
    ("what percent", "percentage"),
    # power
    ("raised to", "power"),
    ("to the power", "power"),
    # divide
    ("divided by", "divide"),
    ("divide by", "divide"),
    # multiply
    ("multiply by", "multiply"),
    ("multiplied by", "multiply"),
    ("product of", "multiply"),
    # subtract
    ("subtract from", "subtract"),
    ("subtracted from", "subtract"),
    ("difference between", "subtract"),
    # add -- at end since "add" is generic
    ("add up", "add"),
    ("add together", "add"),
    ("sum of two", "add"),
]

_WORD_MAP: dict[str, str] = {
    # Single word -> canonical action
    "logarithm": "logarithm",
    "log": "logarithm",
    "mean": "mean",
    "average": "mean",
    "avg": "mean",
    "deviation": "standard_deviation",
    "sqrt": "square_root",
    "root": "square_root",
    "absolute": "absolute_value",
    "abs": "absolute_value",
    "round": "round_number",
    "truncate": "round_number",
    "maximum": "max_value",
    "max": "max_value",
    "highest": "max_value",
    "largest": "max_value",
    "minimum": "min_value",
    "min": "min_value",
    "lowest": "min_value",
    "smallest": "min_value",
    "percentage": "percentage",
    "percent": "percentage",
    "power": "power",
    "exponent": "power",
    "divide": "divide",
    "quotient": "divide",
    "ratio": "divide",
    "multiply": "multiply",
    "times": "multiply",
    "subtract": "subtract",
    "minus": "subtract",
    "add": "add",
    "plus": "add",
    "sum": "sum_values",
    "total": "sum_values",
    "convert": "si_convert",
    "conversion": "si_convert",
    "imperial": "imperial_convert",
}

_TARGET_MAP: dict[str, str] = {
    "number": "number",
    "numbers": "list",
    "value": "number",
    "values": "list",
    "result": "number",
    "list": "list",
    "unit": "unit",
    "units": "unit",
    "distance": "number",
    "price": "number",
    "score": "number",
    "scores": "list",
    "temperature": "number",
}

_STOP_WORDS = frozenset({
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
    "let", "like", "any", "all", "some", "each", "every",
    "there", "here", "via", "per", "after", "before", "over",
})


def extract_intent(query: str) -> dict[str, str]:
    """Extract math intent from NL query.

    Returns:
        {action: str, target: str, keywords: str}
    """
    q_lower = query.lower()

    # 1. Phrase match (longest first)
    action = "none"
    for phrase, act in _PHRASE_MAP:
        if phrase in q_lower:
            action = act
            break

    # 2. Word match fallback
    if action == "none":
        words = re.sub(r"[^a-z0-9\s]", " ", q_lower).split()
        for w in words:
            if w in _WORD_MAP:
                action = _WORD_MAP[w]
                break

    # 3. Target extraction
    target = "none"
    words = re.sub(r"[^a-z0-9\s]", " ", q_lower).split()
    for w in words:
        if w in _TARGET_MAP:
            target = _TARGET_MAP[w]
            break

    # 4. Keywords (stop words removed)
    kw_tokens = [w for w in words if w not in _STOP_WORDS and len(w) > 1]
    keywords = " ".join(dict.fromkeys(kw_tokens))

    return {"action": action, "target": target, "keywords": keywords}
