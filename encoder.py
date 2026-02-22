"""
Encoder for the BFCL Function Caller model.

Unlike the SaaS tool router (which has fixed domain exemplars), this encoder
dynamically builds HDC vectors from arbitrary JSON Schema function definitions
provided at query time — exactly what BFCL requires.

Flow:
  1. Parse BFCL function definitions → extract action, name, description, params
  2. Encode each function as an HDC glyph
  3. Encode the user query as an HDC glyph
  4. Cosine similarity match → return top function(s)

Exports:
  ENCODER_CONFIG — EncoderConfig for the BFCL model
  encode_function(func_def) — JSON Schema function → Concept dict
  encode_query(query, func_defs) — NL query → Concept dict (context-aware)
"""

import re
from glyphh.core.config import EncoderConfig, Layer, Role, Segment

# ---------------------------------------------------------------------------
# ENCODER_CONFIG
# ---------------------------------------------------------------------------

ENCODER_CONFIG = EncoderConfig(
    dimension=10000,
    seed=42,
    layers=[
        Layer(
            name="signature",
            similarity_weight=0.6,
            segments=[
                Segment(
                    name="identity",
                    roles=[
                        Role(
                            name="action",
                            similarity_weight=1.0,
                            lexicons=[
                                "get", "set", "create", "update", "delete",
                                "find", "search", "list", "calculate", "convert",
                                "check", "validate", "send", "add", "remove",
                                "fetch", "compute", "sort", "filter", "count",
                                "parse", "format", "generate", "build", "make",
                                "open", "close", "start", "stop", "run",
                                "read", "write", "load", "save", "download",
                                "upload", "move", "copy", "rename", "merge",
                                "split", "join", "compare", "match", "replace",
                                "extract", "transform", "encode", "decode",
                                "encrypt", "decrypt", "compress", "decompress",
                                "none",
                            ],
                        ),
                        Role(
                            name="function_name",
                            similarity_weight=0.9,
                            lexicons=[],  # dynamic — populated from func names
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="semantics",
            similarity_weight=0.4,
            segments=[
                Segment(
                    name="context",
                    roles=[
                        Role(
                            name="description",
                            similarity_weight=0.8,
                            lexicons=[],  # dynamic — populated from descriptions
                        ),
                        Role(
                            name="parameters",
                            similarity_weight=0.6,
                            lexicons=[],  # dynamic — populated from param names
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# NL extraction helpers
# ---------------------------------------------------------------------------

_STOP_WORDS = {
    "the", "a", "an", "to", "for", "on", "in", "is", "it", "i",
    "do", "can", "please", "now", "up", "my", "our", "me", "we",
    "and", "or", "of", "with", "from", "this", "that", "about",
    "how", "what", "when", "where", "which", "who", "also",
    "then", "but", "just", "them", "their", "its", "be", "been",
    "have", "has", "had", "not", "dont", "want", "think",
    "use", "call", "run", "execute", "tool", "function", "api",
    "given", "using", "by", "if", "so", "as", "at", "into",
}

_ACTION_MAP = {
    "get": "get", "fetch": "get", "retrieve": "get", "obtain": "get",
    "show": "get", "display": "get", "return": "get", "look": "get",
    "find": "find", "search": "search", "lookup": "find", "query": "search",
    "locate": "find", "discover": "find",
    "create": "create", "make": "create", "new": "create", "build": "create",
    "generate": "create", "construct": "create", "initialize": "create",
    "add": "add", "append": "add", "insert": "add", "include": "add",
    "update": "update", "modify": "update", "change": "update", "edit": "update",
    "set": "set", "assign": "set", "configure": "set",
    "delete": "delete", "remove": "remove", "drop": "delete", "erase": "delete",
    "clear": "delete", "destroy": "delete",
    "list": "list", "enumerate": "list", "show": "list",
    "calculate": "calculate", "compute": "calculate", "evaluate": "calculate",
    "determine": "calculate", "measure": "calculate", "estimate": "calculate",
    "convert": "convert", "transform": "convert", "translate": "convert",
    "check": "check", "verify": "check", "validate": "validate",
    "test": "check", "confirm": "check", "ensure": "check",
    "send": "send", "post": "send", "submit": "send", "transmit": "send",
    "sort": "sort", "order": "sort", "rank": "sort", "arrange": "sort",
    "filter": "filter", "select": "filter", "pick": "filter",
    "count": "count", "tally": "count",
    "parse": "parse", "extract": "extract", "read": "read",
    "format": "format", "render": "format",
    "open": "open", "close": "close", "start": "start", "stop": "stop",
    "run": "run", "execute": "run",
    "write": "write", "save": "save", "store": "save",
    "load": "load", "download": "download", "upload": "upload",
    "move": "move", "copy": "copy", "rename": "rename",
    "merge": "merge", "split": "split", "join": "join",
    "compare": "compare", "match": "match", "replace": "replace",
    "encode": "encode", "decode": "decode",
    "encrypt": "encrypt", "decrypt": "decrypt",
    "compress": "compress", "decompress": "decompress",
}


def _extract_action(text: str) -> str:
    """Extract the primary action verb from text."""
    words = re.sub(r"[^a-z\s]", "", text.lower()).split()
    for w in words:
        if w in _ACTION_MAP:
            return _ACTION_MAP[w]
    return "none"


def _split_camel_snake(name: str) -> str:
    """Split camelCase and snake_case into space-separated words."""
    # snake_case → spaces
    name = name.replace("_", " ").replace("-", " ")
    # camelCase → spaces
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return name.lower().strip()


def _extract_keywords(text: str) -> str:
    """Extract meaningful keywords from text, removing stop words."""
    cleaned = re.sub(r"[^a-z0-9\s]", "", text.lower())
    words = [w for w in cleaned.split() if w not in _STOP_WORDS and len(w) > 1]
    return " ".join(words)


def _extract_param_names(func_def: dict) -> str:
    """Extract parameter names from a BFCL function definition."""
    params = func_def.get("parameters", {})
    if isinstance(params, dict):
        props = params.get("properties", {})
        if props:
            names = []
            for pname, pdef in props.items():
                names.append(_split_camel_snake(pname))
                # Also grab enum values as keywords
                if "enum" in pdef:
                    names.extend(str(v).lower() for v in pdef["enum"][:5])
                # Grab description keywords
                if "description" in pdef:
                    names.append(_extract_keywords(pdef["description"]))
            return " ".join(names)
    return ""


# ---------------------------------------------------------------------------
# encode_function — BFCL function definition → Concept dict
# ---------------------------------------------------------------------------

def encode_function(func_def: dict) -> dict:
    """Convert a BFCL function definition into a Concept-compatible dict.

    BFCL function defs look like:
    {
        "name": "calculate_triangle_area",
        "description": "Calculate the area of a triangle given its base and height.",
        "parameters": {
            "type": "object",
            "properties": {
                "base": {"type": "number", "description": "The base of the triangle."},
                "height": {"type": "number", "description": "The height of the triangle."}
            },
            "required": ["base", "height"]
        }
    }
    """
    name = func_def.get("name", "unknown")
    description = func_def.get("description", "")
    name_words = _split_camel_snake(name)

    # Extract action from function name first, then description
    action = _extract_action(name_words)
    if action == "none":
        action = _extract_action(description)

    desc_keywords = _extract_keywords(description)
    param_keywords = _extract_param_names(func_def)

    return {
        "name": f"func_{name}",
        "attributes": {
            "action": action,
            "function_name": name_words,
            "description": desc_keywords,
            "parameters": param_keywords,
        },
    }


# ---------------------------------------------------------------------------
# encode_query — NL query → Concept dict
# ---------------------------------------------------------------------------

def encode_query(query: str, func_defs: list[dict] | None = None) -> dict:
    """Convert a user query into a Concept-compatible dict for function matching.

    Args:
        query: The natural language query from the user.
        func_defs: Optional list of function definitions for context
                   (not used in encoding, but available for future use).
    """
    cleaned = re.sub(r"[^a-z0-9\s]", "", query.lower())
    words = cleaned.split()

    action = _extract_action(query)
    keywords = _extract_keywords(query)

    return {
        "name": "query",
        "attributes": {
            "action": action,
            "function_name": keywords,  # query keywords match against func names
            "description": keywords,
            "parameters": keywords,
        },
    }
