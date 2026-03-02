"""
Intent extraction for the BFCL function call model.

Provides action and target extraction for both NL queries and function
definitions. Tuned for BFCL's diverse function space: math, filesystem,
APIs, physics, finance, travel, etc.

Exports:
  extract_action(text) → str   — canonical action verb (or "none")
  extract_target(text) → str   — canonical target object (or "none")
  extract_keywords(text) → str — space-separated keyword tokens
  extract_intent(text) → dict  — {"action", "target", "keywords"}
"""

import re


# ---------------------------------------------------------------------------
# Verb → canonical action
# Lexicon mirrors the action role in ENCODER_CONFIG exactly.
# ---------------------------------------------------------------------------

_VERB_MAP: dict[str, str] = {
    # get family
    "get": "get", "fetch": "get", "retrieve": "get", "obtain": "get",
    "show": "get", "display": "get", "return": "get", "look": "get",
    "view": "get", "what": "get", "whats": "get", "give": "get",
    # read family
    "read": "read",
    # find / search family
    "find": "find", "search": "search", "lookup": "find", "query": "search",
    "locate": "find", "discover": "find",
    # calculate family — covers math, physics, stats
    "calculate": "calculate", "compute": "calculate", "evaluate": "calculate",
    "determine": "calculate", "measure": "calculate", "estimate": "calculate",
    "solve": "calculate", "derive": "calculate", "integrate": "calculate",
    "differentiate": "calculate",
    # create family
    "create": "create", "make": "create", "generate": "create", "build": "create",
    "construct": "create", "initialize": "create", "new": "create",
    # update family
    "update": "update", "modify": "update", "change": "update", "edit": "update",
    # set family
    "set": "set", "assign": "set", "configure": "set",
    # delete family
    "delete": "delete", "remove": "remove", "drop": "delete", "erase": "delete",
    "clear": "delete", "destroy": "delete",
    # list family
    "list": "list", "enumerate": "list",
    # convert family
    "convert": "convert", "transform": "convert", "translate": "convert",
    # check / validate family
    "check": "check", "verify": "check", "validate": "validate",
    "test": "check", "confirm": "check", "ensure": "check",
    # send family
    "send": "send", "post": "send", "submit": "send", "transmit": "send",
    # add family
    "add": "add", "append": "add", "insert": "add", "include": "add",
    # sort family
    "sort": "sort", "order": "sort", "rank": "sort", "arrange": "sort",
    # filter / select family
    "filter": "filter", "select": "filter", "pick": "filter",
    # count family
    "count": "count", "tally": "count",
    # parse / extract family
    "parse": "parse", "extract": "extract",
    # format family
    "format": "format", "render": "format",
    # open / close / start / stop
    "open": "open", "close": "close", "start": "start", "stop": "stop",
    "run": "run", "execute": "run",
    # write / save family
    "write": "write", "save": "save", "store": "save",
    # load / download / upload
    "load": "load", "download": "download", "upload": "upload",
    # file ops
    "move": "move", "copy": "copy", "rename": "rename",
    "merge": "merge", "split": "split", "join": "join",
    # compare / match / replace
    "compare": "compare", "match": "match", "replace": "replace",
    # encoding / crypto / compression
    "encode": "encode", "decode": "decode",
    "encrypt": "encrypt", "decrypt": "decrypt",
    "compress": "compress", "decompress": "decompress",
    # fetch forms
    "fetches": "get", "retrieves": "get", "gets": "get", "calculates": "calculate",
    "computes": "calculate", "returns": "get",
    # provide / discover forms
    "provide": "get", "provides": "get",
    "discover": "find", "discovers": "find", "discovered": "find",
    # W-question words → get (who discovered..., when was..., what is...)
    "who": "get", "when": "get", "where": "get",
}

# Impact ranking: when multiple verbs found, pick highest-impact one.
# Only used when no leading-verb priority match is found.
_IMPACT_RANK: dict[str, int] = {
    "delete": 10, "destroy": 10,
    "remove": 9,
    "create": 8, "generate": 8, "build": 8, "make": 8,
    "update": 7, "modify": 7, "set": 7, "replace": 7,
    "calculate": 6, "compute": 6, "convert": 6,
    "send": 6, "upload": 6, "download": 6, "encrypt": 6, "decrypt": 6,
    "validate": 5, "check": 5, "verify": 5,
    "search": 5, "find": 5, "filter": 5,
    "add": 4, "insert": 4, "append": 4,
    "get": 3, "fetch": 3, "retrieve": 3, "list": 3, "read": 3,
    "open": 2, "close": 2, "start": 2, "stop": 2, "run": 2,
}


# ---------------------------------------------------------------------------
# Noun → canonical target
# Lexicon mirrors the target role in ENCODER_CONFIG exactly.
# ---------------------------------------------------------------------------

_TARGET_MAP: dict[str, str] = {
    # Math — geometry (shapes)
    "triangle": "triangle", "triangles": "triangle",
    "circle": "circle", "circles": "circle",
    "rectangle": "rectangle", "square": "square", "polygon": "polygon",
    "shape": "shape",
    # Math — geometry (measurements)
    "area": "area", "areas": "area",
    "circumference": "circumference",
    "perimeter": "perimeter",
    "hypotenuse": "hypotenuse",
    "distance": "distance", "displacement": "distance",
    "angle": "angle", "angles": "angle",
    # Math — algebra / number theory
    "equation": "equation", "equations": "equation",
    "root": "roots", "roots": "roots",
    "factor": "prime", "factors": "prime", "prime": "prime",
    "gcd": "sum", "lcm": "sum",
    "number": "number", "digit": "number", "integer": "number",
    # Math — sequences / series
    "fibonacci": "fibonacci", "sequence": "sequence", "series": "sequence",
    # Math — calculus
    "derivative": "derivative", "integral": "integral",
    "factorial": "factorial",
    # Math — linear algebra / stats
    "matrix": "matrix", "vector": "vector",
    "mean": "mean", "average": "mean", "median": "mean", "mode": "mean",
    "standard": "standard", "deviation": "standard",
    "probability": "probability", "variance": "probability",
    "sum": "sum", "product": "sum", "total": "sum",
    "grade": "grade", "grades": "grade", "score": "grade",
    # Physics
    "velocity": "velocity", "speed": "speed", "acceleration": "velocity",
    "force": "force", "energy": "energy", "power": "energy",
    "temperature": "temperature", "pressure": "temperature",
    # Physics — particles / atomic
    "neutron": "particle", "proton": "particle", "electron": "particle",
    "particle": "particle", "atom": "particle", "atomic": "particle",
    "charge": "charge", "diameter": "diameter", "mass": "mass",
    # Health / fitness
    "hydration": "hydration", "water": "hydration",
    "calorie": "calorie", "calories": "calorie",
    "steps": "steps", "step": "steps",
    "weight": "mass",
    # Filesystem
    "file": "file", "files": "file",
    "directory": "directory", "folder": "directory", "dir": "directory",
    "path": "path", "content": "content", "contents": "content",
    "disk": "disk", "permission": "permission",
    # Finance / Trading
    "stock": "stock", "stocks": "stock", "share": "stock", "shares": "stock",
    "price": "price", "prices": "price", "cost": "price",
    "trade": "trade", "trades": "trade", "order": "order", "orders": "order",
    "portfolio": "portfolio", "balance": "balance", "profit": "balance",
    "option": "option", "fund": "fund",
    "revenue": "revenue", "revenues": "revenue",
    "market": "market",
    "growth": "growth", "ratio": "ratio",
    "currency": "currency", "euro": "currency", "euros": "currency",
    "dollar": "currency", "dollars": "currency",
    "usd": "currency", "eur": "currency", "forex": "currency",
    "exchange": "currency",
    "depreciation": "depreciation", "depreciated": "depreciation",
    "mortgage": "mortgage", "loan": "mortgage",
    # Travel / Transport
    "flight": "flight", "flights": "flight",
    "hotel": "hotel", "hotels": "hotel",
    "trip": "trip", "route": "route", "destination": "destination",
    "booking": "booking", "reservation": "booking",
    # Vehicle
    "vehicle": "vehicle", "car": "vehicle", "automobile": "vehicle",
    "fuel": "fuel", "gear": "gear", "engine": "vehicle",
    # Communication
    "message": "message", "messages": "message",
    "channel": "channel", "notification": "notification",
    "post": "post", "tweet": "post", "email": "message",
    # Users / entities
    "user": "user", "users": "user",
    "contact": "user", "customer": "user", "person": "user",
    # Geography / general entities
    "capital": "capital", "city": "city", "country": "country",
    "population": "population",
    # Weather
    "weather": "weather", "forecast": "weather",
    # Tickets / issues
    "ticket": "ticket", "issue": "ticket", "bug": "ticket",
    # Arts / museum (sculpture IS an artwork — same canonical target)
    "artwork": "artwork", "painting": "artwork", "art": "artwork",
    "sculpture": "artwork", "sculptor": "artwork",
    "statue": "artwork",
    "landmark": "landmark", "monument": "landmark",
    "museum": "museum",
    # Entertainment / games
    "game": "game", "games": "game",
    "board": "game", "card": "game", "trivia": "game",
    "book": "book", "novel": "book",
    # Players / sports
    "player": "player", "team": "team", "athlete": "player",
    "championship": "team", "sport": "sport",
    "goal": "goal", "goals": "goal",
    # Weather / air quality
    "air": "weather",
    # Legal
    "case": "case", "law": "case", "court": "case", "legal": "case",
    # Data structures
    "string": "string", "text": "string",
    "list": "list", "array": "list",
    "record": "record", "data": "data",
    "key": "key", "hash": "key", "dictionary": "key",
    # Recipes / Cooking
    "recipe": "recipe", "recipes": "recipe",
    # Places / Geography / Nature
    "restaurant": "restaurant",
    "park": "park",
    "mountain": "mountain", "mountains": "mountain",
    "tourist": "tourist", "attraction": "tourist",
    # Events / History
    "event": "event", "events": "event",
    "treaty": "event", "election": "event", "signing": "event",
    # Geology
    "era": "era", "epoch": "era", "geology": "era",
    # Science / Biology
    "genotype": "genotype", "allele": "genotype",
    "discoverer": "discoverer",
    # Statistics
    "statistical": "statistics", "statistics": "statistics",
    "statistic": "statistics", "significance": "statistics",
    # Safety / Crime
    "crime": "crime",
}


# ---------------------------------------------------------------------------
# Stop words for keyword extraction
# ---------------------------------------------------------------------------

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
    "again", "more", "much", "many", "less", "next", "last",
    "other", "too", "ok", "out", "back", "use", "call",
    "find", "get",  # too generic for BoW
})


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _split_camel_snake(text: str) -> str:
    """Split camelCase and snake_case into space-separated lowercase words."""
    text = text.replace("_", " ").replace("-", " ").replace(".", " ")
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    return text.lower().strip()


def _clean(text: str) -> str:
    """Lowercase, split camel/snake, strip non-alpha."""
    return re.sub(r"[^a-z\s]", " ", _split_camel_snake(text))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_action(text: str) -> str:
    """Extract the canonical action from text (query or function name/description).

    Algorithm:
      1. Leading-verb priority: if first word maps to an action, use it.
      2. Impact-ranked fallback: pick highest-impact mapped verb.

    Returns "none" if no action verb is found.
    """
    words = _clean(text).split()
    if not words:
        return "none"

    # Leading-verb priority
    if words[0] in _VERB_MAP:
        return _VERB_MAP[words[0]]

    # Impact-ranked fallback
    best_action, best_impact = "none", -1
    for w in words:
        if w in _VERB_MAP:
            mapped = _VERB_MAP[w]
            impact = _IMPACT_RANK.get(mapped, 1)
            if impact > best_impact:
                best_impact = impact
                best_action = mapped

    return best_action


def extract_target(text: str) -> str:
    """Extract the canonical target object from text.

    Scans tokens left-to-right and returns the first mapped noun.
    Returns "none" if no target is found.
    """
    words = _clean(text).split()
    for w in words:
        if w in _TARGET_MAP:
            return _TARGET_MAP[w]
    return "none"


def extract_keywords(text: str) -> str:
    """Extract meaningful keyword tokens for BoW encoding.

    Returns space-separated tokens with stop words removed.
    """
    cleaned = re.sub(r"[^a-z0-9\s]", " ", _split_camel_snake(text))
    tokens = cleaned.split()
    return " ".join(t for t in tokens if t not in _STOP_WORDS and len(t) > 1)


def extract_intent(text: str) -> dict:
    """Extract action, target, and keywords from text.

    Returns:
        {"action": str, "target": str, "keywords": str}
    All strings are lowercase. "none" when field not found.
    """
    return {
        "action":   extract_action(text),
        "target":   extract_target(text),
        "keywords": extract_keywords(text),
    }
