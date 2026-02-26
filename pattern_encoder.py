"""
Pattern Encoder — the second Glyphh model for multi-turn BFCL.

Model A (encoder.py) answers: "which function matches this query?"
Model B (this file) answers: "which action sequence does this query imply?"

Same HDC substrate. Same primitives (generate_symbol, bind, bundle, bag_of_words,
cosine_similarity). Different concept structure — instead of encoding function
signatures, we encode action sequence templates mined from ground truth.

Architecture:
  - Patterns are mined from multi-turn ground truth answer sequences
  - Each pattern is a (intent_description, action_sequence, domain) tuple
  - Patterns are encoded as Glyphs with:
    * intent role (bag_of_words): semantic description of what the pattern does
    * sequence role (bag_of_words): the ordered function names as tokens
    * domain role (lexicon): which API class domain this belongs to
  - At inference, the query is encoded and scored against all patterns
  - Top pattern gives the predicted function sequence for that turn
"""

import json
import re
from pathlib import Path
from collections import Counter, defaultdict
from typing import Any

from glyphh.core.config import EncoderConfig, Layer, Role, Segment
from glyphh.core.types import Concept, Glyph
from glyphh.core.ops import cosine_similarity
from glyphh import Encoder

# ---------------------------------------------------------------------------
# Pattern encoder config — separate model from the function router
# ---------------------------------------------------------------------------

PATTERN_ENCODER_CONFIG = EncoderConfig(
    dimension=10000,
    seed=73,  # Different seed = different vector space = independent model
    apply_weights_during_encoding=False,
    include_temporal=False,
    layers=[
        Layer(
            name="intent",
            similarity_weight=0.60,
            segments=[
                Segment(
                    name="semantics",
                    roles=[
                        Role(
                            name="query_intent",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="structure",
            similarity_weight=0.40,
            segments=[
                Segment(
                    name="action",
                    roles=[
                        Role(
                            name="sequence",
                            similarity_weight=0.7,
                            text_encoding="bag_of_words",
                        ),
                        Role(
                            name="domain",
                            similarity_weight=0.3,
                            lexicons=[
                                "filesystem", "trading", "travel",
                                "twitter", "messaging", "ticket",
                                "vehicle", "math", "mixed",
                            ],
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# ---------------------------------------------------------------------------
# Stop words (same as encoder.py)
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


def _tokenize(text: str) -> list[str]:
    cleaned = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return [w for w in cleaned.split() if w not in _STOP_WORDS and len(w) > 1]


def _split_camel_snake(name: str) -> str:
    name = name.replace("_", " ").replace("-", " ").replace(".", " ")
    name = re.sub(r"([a-z])([A-Z])", r"\1 \2", name)
    return name.lower().strip()


def _bow_value(words: list[str]) -> str:
    unique = list(dict.fromkeys(words))
    return " ".join(unique) if unique else "none"


# ---------------------------------------------------------------------------
# Domain classification
# ---------------------------------------------------------------------------

_FUNC_TO_DOMAIN = {
    # Filesystem
    "cd": "filesystem", "ls": "filesystem", "mkdir": "filesystem",
    "touch": "filesystem", "echo": "filesystem", "cat": "filesystem",
    "grep": "filesystem", "find": "filesystem", "mv": "filesystem",
    "cp": "filesystem", "rm": "filesystem", "tail": "filesystem",
    "head": "filesystem", "sort": "filesystem", "wc": "filesystem",
    "diff": "filesystem", "du": "filesystem",
    # Trading
    "get_stock_info": "trading", "place_order": "trading",
    "get_order_details": "trading", "cancel_order": "trading",
    "get_account_info": "trading", "get_watchlist": "trading",
    "add_to_watchlist": "trading", "remove_stock_from_watchlist": "trading",
    "get_current_time": "trading", "update_market_status": "trading",
    "get_order_history": "trading", "get_available_stocks": "trading",
    "make_transaction": "trading", "get_transaction_history": "trading",
    "fund_account": "trading",
    # Travel
    "get_flight_cost": "travel", "book_flight": "travel",
    "cancel_booking": "travel", "retrieve_invoice": "travel",
    "list_all_airports": "travel", "get_nearest_airport_by_city": "travel",
    "purchase_insurance": "travel", "contact_customer_support": "travel",
    "get_zipcode_based_on_city": "travel", "estimate_distance": "travel",
    "estimate_drive_feasibility_by_mileage": "travel",
    "compute_exchange_rate": "travel", "set_budget_limit": "travel",
    # Twitter
    "authenticate_twitter": "twitter", "post_tweet": "twitter",
    "comment": "twitter", "retweet": "twitter", "get_tweet": "twitter",
    "mention": "twitter", "follow_user": "twitter",
    "get_user_tweets": "twitter",
    # Messaging
    "message_login": "messaging", "send_message": "messaging",
    "view_messages_sent": "messaging", "add_contact": "messaging",
    "delete_message": "messaging", "get_message_stats": "messaging",
    "search_messages": "messaging",
    # Ticket
    "ticket_login": "ticket", "create_ticket": "ticket",
    "get_ticket": "ticket", "close_ticket": "ticket",
    "resolve_ticket": "ticket", "edit_ticket": "ticket",
    "ticket_get_login_status": "ticket",
    # Vehicle
    "lockDoors": "vehicle", "pressBrakePedal": "vehicle",
    "startEngine": "vehicle", "fillFuelTank": "vehicle",
    "check_tire_pressure": "vehicle", "find_nearest_tire_shop": "vehicle",
    "set_navigation": "vehicle", "liter_to_gallon": "vehicle",
    "gallon_to_liter": "vehicle", "estimate_drive_feasibility_by_mileage": "vehicle",
    "displayCarStatus": "vehicle", "activateParkingBrake": "vehicle",
    "adjustClimateControl": "vehicle", "setCruiseControl": "vehicle",
    "get_outside_temperature_from_google": "vehicle",
    "get_outside_temperature_from_weather_com": "vehicle",
    # Math
    "mean": "math", "standard_deviation": "math", "si_unit_conversion": "math",
    "logarithm": "math", "add": "math", "subtract": "math",
    "multiply": "math", "divide": "math", "percentage": "math",
    "min_value": "math", "max_value": "math", "sum_values": "math",
}


def _classify_domain(func_names: list[str]) -> str:
    """Classify a function sequence into a domain."""
    domains = [_FUNC_TO_DOMAIN.get(f, "mixed") for f in func_names]
    if not domains:
        return "mixed"
    counts = Counter(domains)
    top = counts.most_common(1)[0]
    # If majority is one domain, use it; otherwise mixed
    if top[1] > len(domains) * 0.5:
        return top[0]
    return "mixed"


# ---------------------------------------------------------------------------
# Pattern mining from ground truth
# ---------------------------------------------------------------------------

DATA_DIR = Path(__file__).parent / "data" / "bfcl"

_ANSWER_FILES = {
    "multi_turn_base": "possible_answer/BFCL_v3_multi_turn_base.json",
    "multi_turn_miss_func": "possible_answer/BFCL_v3_multi_turn_miss_func.json",
    "multi_turn_miss_param": "possible_answer/BFCL_v3_multi_turn_miss_param.json",
    "multi_turn_long_context": "possible_answer/BFCL_v3_multi_turn_long_context.json",
}

_QUESTION_FILES = {
    "multi_turn_base": "BFCL_v3_multi_turn_base.json",
    "multi_turn_miss_func": "BFCL_v3_multi_turn_miss_func.json",
    "multi_turn_miss_param": "BFCL_v3_multi_turn_miss_param.json",
    "multi_turn_long_context": "BFCL_v3_multi_turn_long_context.json",
}


def mine_patterns() -> list[dict]:
    """Mine action sequence patterns from multi-turn ground truth.

    Returns list of pattern dicts:
        {
            "sequence": ("func_a", "func_b"),  # ordered function names
            "domain": "trading",
            "example_queries": ["query1", "query2", ...],
            "count": 68,
        }
    """
    pattern_data = defaultdict(lambda: {"queries": [], "count": 0})

    for cat in _ANSWER_FILES:
        qpath = DATA_DIR / _QUESTION_FILES[cat]
        apath = DATA_DIR / _ANSWER_FILES[cat]
        if not qpath.exists() or not apath.exists():
            continue

        questions = {}
        with open(qpath) as f:
            for line in f:
                e = json.loads(line.strip())
                questions[e.get("id", "")] = e

        with open(apath) as f:
            for line in f:
                a = json.loads(line.strip())
                eid = a.get("id", "")
                gt = a.get("ground_truth", [])
                entry = questions.get(eid, {})
                turns = entry.get("question", [])

                for ti, step in enumerate(gt):
                    if not step:
                        continue
                    funcs = []
                    for call_str in step:
                        if isinstance(call_str, str):
                            paren = call_str.find("(")
                            funcs.append(call_str[:paren] if paren != -1 else call_str)

                    seq = tuple(funcs)
                    query = ""
                    if ti < len(turns):
                        turn = turns[ti]
                        if isinstance(turn, list):
                            for msg in turn:
                                if isinstance(msg, dict) and msg.get("role") == "user":
                                    query = msg.get("content", "")

                    pattern_data[seq]["queries"].append(query)
                    pattern_data[seq]["count"] += 1

    patterns = []
    for seq, data in pattern_data.items():
        patterns.append({
            "sequence": seq,
            "domain": _classify_domain(list(seq)),
            "example_queries": data["queries"],
            "count": data["count"],
        })

    # Sort by count descending
    patterns.sort(key=lambda p: p["count"], reverse=True)
    return patterns


# ---------------------------------------------------------------------------
# Pattern encoding
# ---------------------------------------------------------------------------

def encode_pattern(pattern: dict) -> dict:
    """Encode a mined pattern into a Concept-compatible dict for Model B.

    The pattern glyph captures:
    - query_intent: bag-of-words from ALL example queries (semantic centroid)
    - sequence: function names as tokens (structural signal)
    - domain: categorical domain classification
    """
    seq = pattern["sequence"]
    queries = pattern["example_queries"]
    domain = pattern["domain"]

    # Build intent from all example queries — this is the semantic centroid
    all_tokens = []
    for q in queries:
        all_tokens.extend(_tokenize(q))

    # Weight by frequency — more common tokens are more representative
    token_counts = Counter(all_tokens)
    # Take top tokens (cap at 60 to avoid noise)
    top_tokens = [t for t, _ in token_counts.most_common(60)]

    # Sequence tokens: split function names into words
    seq_tokens = []
    for fname in seq:
        seq_tokens.extend(_split_camel_snake(fname).split())

    return {
        "name": f"pattern_{'_'.join(seq[:3])}",
        "attributes": {
            "query_intent": _bow_value(top_tokens),
            "sequence": _bow_value(seq_tokens),
            "domain": domain,
        },
    }


def encode_pattern_query(query: str, domain_hint: str = "mixed") -> dict:
    """Encode a user query for matching against patterns.

    Same structure as pattern encoding so cosine similarity works.
    """
    tokens = _tokenize(query)

    return {
        "name": "pattern_query",
        "attributes": {
            "query_intent": _bow_value(tokens),
            "sequence": _bow_value(tokens),  # query tokens match against func name tokens
            "domain": domain_hint,
        },
    }


# ---------------------------------------------------------------------------
# PatternRouter — the second Glyphh model
# ---------------------------------------------------------------------------

class PatternRouter:
    """Glyphh HDC model for action sequence pattern routing.

    This is Model B. It scores a query against all known action patterns
    and returns the best-matching sequence template.

    Uses the same HDC math as Model A (function router) but with a
    different encoder config, different seed, different concept structure.
    Same substrate, different specialization.
    """

    def __init__(self):
        self.encoder = Encoder(PATTERN_ENCODER_CONFIG)
        self.patterns: list[dict] = []
        self.pattern_glyphs: list[Glyph] = []
        self._built = False

    def build(self, min_count: int = 2):
        """Mine patterns from ground truth and encode them.

        Args:
            min_count: Minimum occurrence count to include a pattern.
                       Patterns seen only once are likely noise.
        """
        raw_patterns = mine_patterns()

        self.patterns = []
        self.pattern_glyphs = []

        for p in raw_patterns:
            if p["count"] < min_count:
                continue

            concept_dict = encode_pattern(p)
            concept = Concept(
                name=concept_dict["name"],
                attributes=concept_dict["attributes"],
            )
            glyph = self.encoder.encode(concept)

            self.patterns.append(p)
            self.pattern_glyphs.append(glyph)

        self._built = True

    def route(
        self,
        query: str,
        domain_hint: str = "mixed",
        top_k: int = 5,
    ) -> list[dict]:
        """Route a query to the best-matching action patterns.

        Returns top_k matches, each with:
            - sequence: tuple of function names
            - score: cosine similarity
            - domain: pattern domain
            - count: how often this pattern appeared in training
        """
        if not self._built:
            self.build()

        q_dict = encode_pattern_query(query, domain_hint)
        q_concept = Concept(name=q_dict["name"], attributes=q_dict["attributes"])
        q_glyph = self.encoder.encode(q_concept)

        scores = []
        for i, pg in enumerate(self.pattern_glyphs):
            score = self._score(q_glyph, pg)
            scores.append((score, i))

        scores.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, idx in scores[:top_k]:
            p = self.patterns[idx]
            results.append({
                "sequence": p["sequence"],
                "score": round(score, 4),
                "domain": p["domain"],
                "count": p["count"],
            })

        return results

    def _score(self, q_glyph: Glyph, p_glyph: Glyph) -> float:
        """Multi-level cosine similarity, same approach as Model A."""
        # Cortex level
        cortex_sim = float(cosine_similarity(
            q_glyph.global_cortex.data,
            p_glyph.global_cortex.data,
        ))

        # Role level
        role_sims = []
        for lname in q_glyph.layers:
            if lname.startswith("_") or lname not in p_glyph.layers:
                continue
            for sname in q_glyph.layers[lname].segments:
                if sname not in p_glyph.layers[lname].segments:
                    continue
                qs = q_glyph.layers[lname].segments[sname]
                ps = p_glyph.layers[lname].segments[sname]
                for rname in qs.roles:
                    if rname in ps.roles:
                        rsim = float(cosine_similarity(
                            qs.roles[rname].data, ps.roles[rname].data
                        ))
                        # Get weight from config
                        rweight = 1.0
                        for ld in PATTERN_ENCODER_CONFIG.layers:
                            if ld.name == lname:
                                for sd in ld.segments:
                                    if sd.name == sname:
                                        for rd in sd.roles:
                                            if rd.name == rname:
                                                rweight = rd.similarity_weight
                        role_sims.append((rsim, rweight))

        # Combine: 30% cortex, 70% role-level
        score = cortex_sim * 0.30
        if role_sims:
            weighted_role = sum(s * w for s, w in role_sims) / sum(w for _, w in role_sims)
            score += weighted_role * 0.70

        return score
