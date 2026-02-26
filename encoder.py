"""
Encoder for the BFCL Function Caller model.

Uses native Glyphh HDC encoding with:
- text_encoding="bag_of_words" on text roles (SDK-native, no manual bundling)
- include_temporal=False (no temporal signal pollution)
- apply_weights_during_encoding=False (weights applied at scoring time)
- Lexicons for categorical action role

Exports:
  ENCODER_CONFIG — EncoderConfig for the BFCL model
  encode_function(func_def) — JSON Schema function → Concept dict
  encode_query(query) — NL query → Concept dict
"""

import re
from glyphh.core.config import EncoderConfig, Layer, Role, Segment

# ---------------------------------------------------------------------------
# ENCODER_CONFIG
# ---------------------------------------------------------------------------

ENCODER_CONFIG = EncoderConfig(
    dimension=10000,
    seed=42,
    apply_weights_during_encoding=False,
    include_temporal=False,
    layers=[
        Layer(
            name="signature",
            similarity_weight=0.55,
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
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="semantics",
            similarity_weight=0.45,
            segments=[
                Segment(
                    name="context",
                    roles=[
                        Role(
                            name="description",
                            similarity_weight=0.8,
                            text_encoding="bag_of_words",
                        ),
                        Role(
                            name="parameters",
                            similarity_weight=0.6,
                            text_encoding="bag_of_words",
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
    "given", "using", "by", "if", "so", "as", "at", "into",
    "are", "was", "were", "will", "would", "could", "should",
    "may", "might", "shall", "need", "does", "did", "am",
    "you", "your", "he", "she", "they", "us", "his", "her",
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
    "list": "list", "enumerate": "list",
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
    name = name.replace("_", " ").replace("-", " ").replace(".", " ")
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return name.lower().strip()


def _tokenize(text: str) -> list[str]:
    """Tokenize text into meaningful words, removing stop words and short tokens."""
    cleaned = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return [w for w in cleaned.split() if w not in _STOP_WORDS and len(w) > 1]


def _extract_param_tokens(func_def: dict) -> list[str]:
    """Extract parameter tokens from a BFCL function definition."""
    params = func_def.get("parameters", {})
    tokens = []
    if isinstance(params, dict):
        props = params.get("properties", {})
        if props:
            for pname, pdef in props.items():
                tokens.extend(_split_camel_snake(pname).split())
                if "enum" in pdef:
                    for v in pdef["enum"][:8]:
                        tokens.extend(_tokenize(str(v)))
                if "description" in pdef:
                    tokens.extend(_tokenize(pdef["description"]))
                ptype = pdef.get("type", "")
                if ptype and ptype not in ("object", "dict", "string", "str"):
                    tokens.append(ptype)
    return [t for t in tokens if t not in _STOP_WORDS and len(t) > 1]


def _bow_value(words: list[str]) -> str:
    """Create a bag-of-words value string for a role attribute.

    The SDK's text_encoding="bag_of_words" will split this into words
    and bundle their symbol vectors.
    """
    unique = list(dict.fromkeys(words))  # dedupe preserving order
    return " ".join(unique) if unique else "none"


# ---------------------------------------------------------------------------
# encode_function — BFCL function definition → Concept dict
# ---------------------------------------------------------------------------

def encode_function(func_def: dict) -> dict:
    """Convert a BFCL function definition into a Concept-compatible dict.

    Returns a dict with 'name' and 'attributes' suitable for Concept().
    Text role values are space-separated keyword tokens — the SDK's
    text_encoding="bag_of_words" handles the actual encoding.
    """
    name = func_def.get("name", "unknown")
    description = func_def.get("description", "")
    name_words = _split_camel_snake(name)

    # Action: from function name first, then description
    action = _extract_action(name_words)
    if action == "none":
        action = _extract_action(description)

    # Function name tokens
    name_tokens = _tokenize(name_words)

    # Description tokens
    desc_tokens = _tokenize(description)

    # Parameter tokens
    param_tokens = _extract_param_tokens(func_def)

    return {
        "name": f"func_{name}",
        "attributes": {
            "action": action,
            "function_name": _bow_value(name_tokens),
            "description": _bow_value(desc_tokens),
            "parameters": _bow_value(param_tokens),
        },
    }


# ---------------------------------------------------------------------------
# encode_query — NL query → Concept dict
# ---------------------------------------------------------------------------

def encode_query(query: str) -> dict:
    """Convert a user query into a Concept-compatible dict.

    All text roles get the same keyword tokens from the query so they
    can match against function_name, description, and parameters.
    """
    action = _extract_action(query)
    keywords = _tokenize(query)
    kw_value = _bow_value(keywords)

    return {
        "name": "query",
        "attributes": {
            "action": action,
            "function_name": kw_value,
            "description": kw_value,
            "parameters": kw_value,
        },
    }
