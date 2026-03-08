"""
Intent extraction for the BFCL function call model.

Provides action and target extraction for both NL queries and function
definitions. Tuned for BFCL's diverse function space: math, filesystem,
APIs, physics, finance, travel, etc.

Uses SDK intent packs (filesystem, social, math, trading, travel, vehicle)
for phrase-level matching — a priority layer above single-word verb matching.
Pack phrases provide discriminative signal for multi-turn queries:
  "occurrence of the keyword" → grep (pack phrase) vs "investigate" → find (verb)

Exports:
  extract_action(text) → str   — canonical action verb (or "none")
  extract_target(text) → str   — canonical target object (or "none")
  extract_keywords(text) → str — space-separated keyword tokens
  extract_intent(text) → dict  — {"action", "target", "keywords"}
  extract_pack_actions(text) → list[str] — pack canonical actions for BoW injection
  extract_api_class(query, involved_classes) → str — Stage 1 class detection via pack domain_signals
"""

import json
import re
from pathlib import Path


# ---------------------------------------------------------------------------
# Pack loading — SDK intent packs provide phrase-level matching
# ---------------------------------------------------------------------------

def _get_packs_dir() -> Path:
    """Resolve pack data directory from SDK installation."""
    try:
        import glyphh
        packs_dir = Path(glyphh.__file__).parent / "intent" / "data" / "packs"
        if packs_dir.exists():
            return packs_dir
    except ImportError:
        pass
    # Fallback for development
    return (
        Path(__file__).resolve().parents[2]
        / "glyphh-runtime" / "glyphh" / "intent" / "data" / "packs"
    )


# Map pack canonical actions → BFCL action role values (encoder lexicon)
_PACK_ACTION_MAP: dict[str, str] = {
    # filesystem
    "ls": "list", "cd": "find", "mkdir": "create", "rm": "delete",
    "cp": "copy", "mv": "move", "cat": "read", "chmod": "set",
    "grep": "search", "touch": "create", "wc": "count", "pwd": "get",
    "find_files": "find", "tail": "get", "head": "get", "echo": "write",
    "diff": "compare", "sort": "sort", "pipe": "run",
    "kill_process": "stop", "ssh": "open", "sudo": "run",
    "tar": "compress", "curl": "download",
    # social
    "post": "send", "repost": "send", "comment": "send",
    "like": "add", "follow": "add", "unfollow": "remove",
    "mention": "send", "dm_social": "send", "block": "remove",
    "report": "send", "create_story": "create",
    "schedule_post": "create", "analyze_engagement": "get",
    # math
    "add": "add", "subtract": "calculate", "multiply": "calculate",
    "divide": "calculate", "power": "calculate", "sqrt": "calculate",
    "log": "calculate", "derivative": "calculate", "integral": "calculate",
    "solve": "calculate", "simplify": "calculate",
    "convert_units": "convert", "statistics": "calculate",
    "matrix_op": "calculate", "modulo": "calculate",
    # trading
    "buy": "create", "sell": "delete", "place_order": "create",
    "cancel_order": "delete", "get_quote": "get", "get_portfolio": "get",
    "get_balance": "get", "transfer": "move", "get_history": "get",
    "set_alert": "set", "analyze_market": "get",
    # travel
    "book_flight": "create", "book_hotel": "create", "book_car": "create",
    "check_itinerary": "get", "cancel_booking": "delete",
    "check_in": "run", "get_flight_status": "get", "upgrade": "update",
    "add_baggage": "add", "search_flights": "search",
    "get_visa_info": "get", "get_weather": "get",
    # vehicle
    "accelerate": "run", "brake": "stop", "steer": "move",
    "set_cruise": "set", "park": "stop", "lock": "close",
    "unlock": "open", "start_engine": "start", "stop_engine": "stop",
    "set_navigation": "find", "check_fuel": "get", "honk": "run",
    "set_climate": "set", "check_diagnostics": "get", "open_trunk": "open",
}

# Pack phrase data: list of (phrase, bfcl_action, pack_canonical)
# Sorted by phrase length desc for longest-match-first
_PACK_PHRASES: list[tuple[str, str, str]] = []
_PACKS: dict[str, dict] = {}  # raw pack data keyed by pack name
_PACKS_LOADED = False

# Map BFCL API class → SDK pack name for Stage 1 class detection
_CLASS_TO_PACK: dict[str, str] = {
    "GorillaFileSystem": "filesystem",
    "TwitterAPI":        "social",
    "MessageAPI":        "social",
    "PostingAPI":        "social",
    "TicketAPI":         "saas",
    "MathAPI":           "math",
    "TradingBot":        "trading",
    "TravelBookingAPI":  "travel",
    "TravelAPI":         "travel",
    "VehicleControlAPI": "vehicle",
}


def _load_packs() -> None:
    """Load all relevant intent packs from SDK for phrase matching."""
    global _PACK_PHRASES, _PACKS, _PACKS_LOADED
    if _PACKS_LOADED:
        return
    _PACKS_LOADED = True

    packs_dir = _get_packs_dir()
    if not packs_dir.exists():
        return

    pack_names = ["filesystem", "social", "math", "trading", "travel", "vehicle", "saas", "churn"]
    phrases: list[tuple[str, str, str]] = []

    for pack_name in pack_names:
        path = packs_dir / f"{pack_name}.json"
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        _PACKS[pack_name] = data

        # Top-level phrase→action mappings
        for p in data.get("phrases", []):
            canonical = p["action"]
            bfcl_action = _PACK_ACTION_MAP.get(canonical)
            if bfcl_action:
                phrases.append((p["phrase"].lower(), bfcl_action, canonical))

        # Action-level phrases and synonyms (both multi-word and single-word)
        for action_def in data.get("actions", []):
            canonical = action_def["canonical"]
            bfcl_action = _PACK_ACTION_MAP.get(canonical)
            if not bfcl_action:
                continue
            for phrase in action_def.get("phrases", []):
                phrases.append((phrase.lower(), bfcl_action, canonical))
            for syn in action_def.get("synonyms", []):
                syn_lower = syn.lower()
                # Include single-word synonyms (>2 chars to avoid noise)
                # Critical for matching paraphrased BFCL queries like
                # "display the files" → "display" → cat canonical
                if len(syn_lower) > 2:
                    phrases.append((syn_lower, bfcl_action, canonical))

    # Dedupe and sort by length desc (longest match first)
    seen = set()
    unique = []
    for phrase, action, canonical in phrases:
        key = (phrase, canonical)
        if key not in seen:
            seen.add(key)
            unique.append((phrase, action, canonical))
    _PACK_PHRASES = sorted(unique, key=lambda x: len(x[0]), reverse=True)


def _match_pack_phrase(text: str) -> tuple[str, str] | None:
    """Match text against pack phrases.

    Returns (bfcl_action, pack_canonical) or None.
    Longest phrase match wins.
    """
    _load_packs()
    text_lower = text.lower()
    for phrase, bfcl_action, canonical in _PACK_PHRASES:
        if phrase in text_lower:
            return bfcl_action, canonical
    return None


def extract_pack_actions(text: str) -> list[str]:
    """Extract pack canonical action names from text using phrase matching.

    Returns list of pack canonical names (e.g., ["grep", "tail"]).
    Used by encoder to inject into function_name BoW for direct matching.
    """
    _load_packs()
    text_lower = text.lower()
    matched: list[str] = []
    for phrase, _, canonical in _PACK_PHRASES:
        if phrase in text_lower and canonical not in matched:
            matched.append(canonical)
    return matched


# BFCL-specific class detection keywords — NL patterns for each API class.
# Pack domain_signals are CLI-oriented; these cover natural language used in
# BFCL multi-turn queries. Word matches get +3, phrase matches get +5.
_CLASS_KEYWORDS: dict[str, dict[str, list[str]]] = {
    "GorillaFileSystem": {
        "words": ["file", "files", "folder", "directory", "txt", "pdf", "csv",
                  "copy", "copied", "move", "moved", "rename", "renamed",
                  "sort", "sorted", "grep", "diff", "tail", "contents",
                  "archive", "backup", "document", "disk"],
        "phrases": ["into the file", "to the file", "in the file",
                    "to the directory", "in the directory", "into the folder",
                    "create a file", "create a document", "create a directory",
                    "list of files", "file system", "file name", "go into",
                    "copy it", "move it", "compare the", "search for",
                    "jot down", "write into", "read the content",
                    "in the folder", "to the folder"],
    },
    "TwitterAPI": {
        "words": ["tweet", "twitter", "retweet", "hashtag", "mention"],
        "phrases": ["social media", "post on twitter", "draft a tweet",
                    "post a tweet", "share on twitter", "comment underneath",
                    "share the sorted result"],
    },
    "MessageAPI": {
        "words": ["message", "inbox", "dm"],
        "phrases": ["send a message", "send message", "view messages"],
    },
    "TicketAPI": {
        "words": ["ticket", "jira", "bug", "issue"],
        "phrases": ["create ticket", "close ticket", "resolve ticket",
                    "support ticket", "open a ticket", "ticket status"],
    },
    "MathAPI": {
        "words": ["calculate", "average", "logarithm", "sqrt", "percentage",
                  "mean", "sum", "multiply", "divide", "subtract"],
        "phrases": ["square root", "log base", "standard deviation"],
    },
    "TradingBot": {
        "words": ["stock", "shares", "portfolio", "trade", "ticker", "buy",
                  "sell", "balance", "quote"],
        "phrases": ["stock price", "buy shares", "sell shares",
                    "trading account", "market order"],
    },
    "TravelBookingAPI": {
        "words": ["flight", "hotel", "booking", "reservation", "airline",
                  "airport", "itinerary", "baggage", "travel", "trip",
                  "destination", "budget", "invoice", "insurance"],
        "phrases": ["book a flight", "book a hotel", "cancel booking",
                    "flight status", "check in", "travel from", "fly to",
                    "budget limit", "exchange rate", "customer support",
                    "nearest airport", "zipcode"],
    },
    "TravelAPI": {
        "words": ["flight", "hotel", "booking", "reservation", "airline",
                  "airport", "itinerary", "baggage", "travel", "trip",
                  "destination", "budget", "invoice", "insurance"],
        "phrases": ["book a flight", "book a hotel", "cancel booking",
                    "flight status", "check in", "travel from", "fly to",
                    "budget limit", "exchange rate", "customer support",
                    "nearest airport", "zipcode"],
    },
    "VehicleControlAPI": {
        "words": ["vehicle", "car", "engine", "brake", "accelerate",
                  "cruise", "steering", "fuel"],
        "phrases": ["start the car", "start the engine", "cruise control",
                    "turn on the", "set temperature", "lock the car"],
    },
    "PostingAPI": {
        "words": ["post", "publish"],
        "phrases": ["create a post", "share post"],
    },
}


def extract_api_class(query: str, involved_classes: list[str]) -> str:
    """Stage 1: Detect which BFCL API class a query targets.

    Combines BFCL-specific class keywords with SDK pack domain_signals.
    Phrase matches are weighted higher than word matches.

    Args:
        query:            Natural language query text.
        involved_classes: List of API class names available for this entry.

    Returns:
        Best-matching class name. Falls back to first involved class.
    """
    if len(involved_classes) <= 1:
        return involved_classes[0] if involved_classes else ""

    _load_packs()
    query_lower = query.lower()
    query_words = set(re.sub(r"[^a-z0-9\s]", " ", query_lower).split())
    scores: dict[str, int] = {}

    for cls in involved_classes:
        score = 0

        # 1. BFCL-specific class keywords (highest priority)
        cls_kw = _CLASS_KEYWORDS.get(cls, {})
        for word in cls_kw.get("words", []):
            if word in query_words:
                score += 3
        for phrase in cls_kw.get("phrases", []):
            if phrase in query_lower:
                score += 5

        # 2. SDK pack domain_signals (secondary)
        pack_name = _CLASS_TO_PACK.get(cls)
        if pack_name and pack_name in _PACKS:
            pack = _PACKS[pack_name]
            for _domain_key, keywords in pack.get("domain_signals", {}).items():
                for keyword, weight in keywords.items():
                    if " " in keyword:
                        if keyword in query_lower:
                            score += weight
                    else:
                        if keyword in query_words:
                            score += weight

        scores[cls] = score

    if not scores:
        return involved_classes[0]

    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else involved_classes[0]


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
    "locate": "find", "discover": "find", "gather": "find", "identify": "find",
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
    # Multi-turn domain verbs (filesystem, messaging, social)
    "navigate": "find", "investigate": "search", "peek": "get",
    "draft": "create", "jot": "write", "craft": "create",
    "broadcast": "send", "dispatch": "send", "tweet": "send",
    "transfer": "move", "duplicate": "copy", "secure": "save",
    "organize": "sort", "inspect": "check",
    "archive": "save", "pop": "find",
    # Live function matching — verbs that map to search/find family
    "attend": "find", "attending": "find",
    "browse": "find", "browsing": "find",
    "explore": "find", "exploring": "find",
    "interested": "find",
    "looking": "find",
    "recommend": "find", "suggest": "find",
    "available": "find",
    "hire": "find", "hiring": "find",
    "need": "find",
    # Live function matching — verbs that map to create/action family
    "buy": "create", "buying": "create",
    "purchase": "create", "purchasing": "create",
    "reserve": "create", "reserving": "create",
    "schedule": "create", "scheduling": "create",
    "rent": "create", "renting": "create",
    "pay": "send", "paying": "send",
    "request": "send", "requesting": "send",
    # Live function matching — media/interaction verbs
    "play": "run", "playing": "run",
    "watch": "get", "watching": "get",
    "listen": "get", "listening": "get",
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
    "write": 5, "save": 5,
    "validate": 5, "check": 5, "verify": 5,
    "search": 5, "find": 5, "filter": 5,
    "add": 4, "insert": 4, "append": 4,
    "get": 3, "fetch": 3, "retrieve": 3, "list": 3, "read": 3,
    "open": 2, "close": 2, "start": 2, "stop": 2, "run": 2,
    # Live action verbs
    "buy": 8, "purchase": 8, "reserve": 8, "schedule": 7,
    "rent": 7, "hire": 5, "attend": 4, "browse": 4,
    "play": 4, "watch": 3, "listen": 3, "pay": 6, "request": 5,
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
    "line": "line", "lines": "line", "word": "word", "words": "word",
    "character": "character", "characters": "character",
    "document": "document", "report": "document", "note": "document",
    "notes": "document", "summary": "document",
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
      1. Pack phrase matching (highest priority — multi-word patterns).
      2. Leading-verb priority: if first word maps to an action, use it.
      3. Impact-ranked fallback: pick highest-impact mapped verb.

    Returns "none" if no action verb is found.
    """
    # 1. Pack phrase matching — longest phrase match wins
    pack_match = _match_pack_phrase(text)
    if pack_match:
        return pack_match[0]  # bfcl_action

    words = _clean(text).split()
    if not words:
        return "none"

    # 2. Leading-verb priority
    if words[0] in _VERB_MAP:
        return _VERB_MAP[words[0]]

    # 3. Impact-ranked fallback
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
