"""
DeductiveLayer — HDC deductive reasoning for implicit prerequisite detection.

Mirrors predictive coding in the brain:
  ACCUMULATE → COMPARE → RESOLVE
  (observe)    (mismatch)  (transition)

The core insight: when a user says "grep for budget analysis in the file"
while CWD is /workspace/document/temp/, the *query itself* doesn't mention
any directory — but the accumulated environmental context (we were just
working in temp/, now we need something in reports/) creates a detectable
mismatch. That mismatch IS the deduction that a prerequisite cd() is needed.

Uses seed=89 (independent vector space from Model A=42, IntentExtractor=53,
Model B/ConversationState=73).

Pure HDC operations: bind, bundle, cosine_similarity, generate_symbol.
No per-function rules — the transition library is pattern-based and learnable.
"""

import re

import numpy as np

from glyphh.core.ops import bind, bundle, cosine_similarity, generate_symbol


# ── Constants ──

_DIM = 10000
_SEED = 89
_DECAY = 0.7

# Mismatch threshold: location_mismatch > this triggers prerequisite search
_MISMATCH_THRESHOLD = 0.12

# Minimum confidence for transition match to produce a boost
_RESOLVE_THRESHOLD = 0.05


# ── Role symbols (deterministic, seed=89) ──

ROLE_LOCATION = generate_symbol(_SEED, "role_location", _DIM)
ROLE_ENTITY = generate_symbol(_SEED, "role_entity", _DIM)
ROLE_ACTION = generate_symbol(_SEED, "role_action", _DIM)


def _weighted_bundle(pairs: list[tuple[np.ndarray, float]], dimension: int) -> np.ndarray:
    """Weighted majority vote of bipolar vectors.

    Same implementation as glyphh.state.pathway._weighted_bundle.
    """
    total = np.zeros(dimension, dtype=np.float32)
    for vec, weight in pairs:
        total += vec.astype(np.float32) * weight
    return np.where(total >= 0, 1, -1).astype(np.int8)


def _encode_observation(
    location: str,
    entities: list[str],
    actions: list[str],
) -> np.ndarray:
    """Encode a (location, entities, actions) triple as a single HDC vector.

    Binds each signal to its role, then bundles into one observation vector.
    """
    components = []

    # Location: bind role to location symbol
    if location:
        loc_vec = generate_symbol(_SEED, f"loc_{location}", _DIM)
        components.append(bind(ROLE_LOCATION, loc_vec))

    # Entities: BoW-style bundle of entity symbols, then bind to role
    if entities:
        ent_vecs = [generate_symbol(_SEED, f"ent_{e}", _DIM) for e in entities[:8]]
        ent_bundle = bundle(ent_vecs) if len(ent_vecs) > 1 else ent_vecs[0]
        components.append(bind(ROLE_ENTITY, ent_bundle))

    # Actions: BoW-style bundle of action symbols, then bind to role
    if actions:
        act_vecs = [generate_symbol(_SEED, f"act_{a}", _DIM) for a in actions]
        act_bundle = bundle(act_vecs) if len(act_vecs) > 1 else act_vecs[0]
        components.append(bind(ROLE_ACTION, act_bundle))

    if not components:
        return np.ones(_DIM, dtype=np.int8)  # identity-ish fallback

    return bundle(components) if len(components) > 1 else components[0]


def _encode_query_environment(
    query: str,
    state_hint: dict,
) -> np.ndarray:
    """Encode what the query implies about the environment.

    Extracts location signals from the query text and state_hint,
    then encodes them in the same (location, entity, action) space
    so cosine comparison with env_context is meaningful.
    """
    query_lower = query.lower()
    words = set(re.sub(r"[^a-z0-9\s]", "", query_lower).split())

    # Location signal: what directory does the query reference?
    implied_location = ""
    if state_hint:
        # If query mentions a child dir → that's where we need to be
        child = state_hint.get("query_mentions_child")
        if child:
            cwd = state_hint.get("cwd", "/")
            implied_location = cwd + "/" + child
        # If query mentions another known dir → navigation needed
        elif state_hint.get("query_mentions_other_dir"):
            implied_location = state_hint["query_mentions_other_dir"]
        else:
            # Default: query implies current location
            implied_location = state_hint.get("cwd", "/")

    # Entity signal: any file-like nouns in the query
    entities = []
    for w in words:
        if "." in w and len(w) > 2:  # looks like a filename
            entities.append(w)

    # Action signal: extract verb-like words
    action_words = []
    action_verbs = {
        "grep", "find", "search", "move", "copy", "rename", "delete",
        "remove", "create", "list", "cat", "echo", "touch", "diff",
        "sort", "wc", "head", "tail", "mv", "cp", "ls", "cd",
    }
    for w in words:
        if w in action_verbs:
            action_words.append(w)

    return _encode_observation(implied_location, entities, action_words)


class DeductiveLayer:
    """HDC deductive reasoning for implicit prerequisite detection.

    Three-phase cycle per turn:
      1. ACCUMULATE (observe): Record location/entity/action as decaying superposition
      2. COMPARE (deduce): Detect mismatch between env_context and query_env
      3. RESOLVE (deduce): Match mismatch against transition library → prerequisites

    The transition library is pre-seeded with common patterns and strengthened
    via Hebbian reinforcement (confirm) on correct predictions.
    """

    def __init__(self, dimension: int = _DIM, seed: int = _SEED, decay: float = _DECAY):
        self._dim = dimension
        self._seed = seed
        self._decay = decay
        self._observations: list[np.ndarray] = []
        self._transition_library: dict[str, dict] = {}
        self._seed_transitions()

    def _seed_transitions(self):
        """Pre-seed the transition library with common prerequisite patterns.

        Each transition has:
          - vector: HDC encoding of the "delta" signature for this transition type
          - prerequisite: function name to inject
          - strength: Hebbian weight (grows with confirmed correct use)
        """
        # Location change: the delta between "I'm at location A" and "I need location B"
        # We encode this as the signature of a location-change event
        loc_change_examples = [
            ("loc_/workspace/src", "loc_/workspace/docs"),
            ("loc_/home/projects", "loc_/home/projects/src"),
            ("loc_/workspace/document/temp", "loc_/workspace/document/reports"),
        ]
        delta_vecs = []
        for from_loc, to_loc in loc_change_examples:
            from_vec = bind(ROLE_LOCATION, generate_symbol(self._seed, from_loc, self._dim))
            to_vec = bind(ROLE_LOCATION, generate_symbol(self._seed, to_loc, self._dim))
            delta_vecs.append(bind(to_vec, from_vec))
        loc_change_vec = bundle(delta_vecs)

        self._transition_library["location_change"] = {
            "vector": loc_change_vec,
            "prerequisite": "cd",
            "strength": 1.0,
        }

        # File creation: delta from "no file" to "file exists"
        creation_examples = [
            ("act_create", "act_touch"),
            ("act_new", "act_touch"),
            ("act_make", "act_touch"),
        ]
        creation_vecs = []
        for act_from, act_to in creation_examples:
            from_vec = bind(ROLE_ACTION, generate_symbol(self._seed, act_from, self._dim))
            to_vec = bind(ROLE_ACTION, generate_symbol(self._seed, act_to, self._dim))
            creation_vecs.append(bind(to_vec, from_vec))
        creation_vec = bundle(creation_vecs)

        self._transition_library["file_creation"] = {
            "vector": creation_vec,
            "prerequisite": "touch",
            "strength": 1.0,
        }

    def observe(self, location: str, entities: list[str], actions: list[str]):
        """ACCUMULATE: Record what happened this turn.

        Encodes the observation as an HDC vector and appends to history.
        Old observations decay exponentially in the superposition.
        """
        obs = _encode_observation(location, entities, actions)
        self._observations.append(obs)

        # Keep bounded history (last 10 turns)
        if len(self._observations) > 10:
            self._observations = self._observations[-10:]

    def get_env_context(self) -> np.ndarray | None:
        """Return the current environmental context as a decaying superposition.

        Recent observations dominate, older ones fade. This encodes
        "where we've been and what we've done" as a single HD vector.
        """
        if not self._observations:
            return None

        n = len(self._observations)
        pairs = []
        for i, obs in enumerate(self._observations):
            # Most recent = weight 1.0, older = decay^(n-1-i)
            weight = self._decay ** (n - 1 - i)
            pairs.append((obs, weight))

        return _weighted_bundle(pairs, self._dim)

    def deduce(self, query: str, state_hint: dict) -> dict:
        """COMPARE + RESOLVE: Detect mismatch, identify prerequisites.

        Phase 2 (COMPARE): Encodes the query's implied environment and compares
        it to the accumulated env_context via cosine similarity. A high mismatch
        means the query implies a different state than where we currently are.

        Phase 3 (RESOLVE): If mismatch exceeds threshold, matches the delta
        against the transition library to identify which prerequisite to inject.

        Returns:
            {
                "prerequisites": ["cd"],        # functions to inject (empty if no mismatch)
                "mismatch_score": 0.34,         # overall environment mismatch
                "location_mismatch": 0.41,      # location-specific mismatch
                "target_dir": "reports",         # deduced target directory (if found)
                "confidence": 0.78,             # transition match confidence
            }
        """
        result = {
            "prerequisites": [],
            "mismatch_score": 0.0,
            "location_mismatch": 0.0,
            "target_dir": None,
            "confidence": 0.0,
        }

        env_context = self.get_env_context()
        if env_context is None:
            return result

        # Encode what the query implies about the environment
        query_env = _encode_query_environment(query, state_hint)

        # COMPARE: overall mismatch
        overall_sim = cosine_similarity(env_context, query_env)
        mismatch = 1.0 - overall_sim
        result["mismatch_score"] = round(mismatch, 4)

        # COMPARE: location-specific mismatch via unbinding
        implied_location = bind(ROLE_LOCATION, query_env)
        current_location = bind(ROLE_LOCATION, env_context)
        loc_sim = cosine_similarity(implied_location, current_location)
        loc_mismatch = 1.0 - loc_sim
        result["location_mismatch"] = round(loc_mismatch, 4)

        # Only proceed to RESOLVE if location mismatch exceeds threshold
        if loc_mismatch <= _MISMATCH_THRESHOLD:
            return result

        # RESOLVE: compute delta and match against transition library
        delta = bind(query_env, env_context)

        best_transition = None
        best_score = 0.0

        for name, transition in self._transition_library.items():
            sim = cosine_similarity(delta, transition["vector"])
            weighted_sim = sim * transition["strength"]
            if weighted_sim > best_score and weighted_sim > _RESOLVE_THRESHOLD:
                best_score = weighted_sim
                best_transition = name

        if best_transition:
            transition = self._transition_library[best_transition]
            result["prerequisites"] = [transition["prerequisite"]]
            result["confidence"] = round(best_score, 4)

            # Try to extract the target directory from state_hint
            if transition["prerequisite"] == "cd" and state_hint:
                child = state_hint.get("query_mentions_child")
                other = state_hint.get("query_mentions_other_dir")
                if child:
                    result["target_dir"] = child
                elif other:
                    result["target_dir"] = other

        return result

    def confirm(self, was_correct: bool, prerequisite: str):
        """Hebbian reinforcement: strengthen or weaken the transition that fired.

        If the deduced prerequisite was correct (ground truth included it),
        strengthen the transition. If wrong, weaken it (but don't drop below 0.3
        to allow recovery).
        """
        for name, transition in self._transition_library.items():
            if transition["prerequisite"] == prerequisite:
                if was_correct:
                    # Strengthen with diminishing returns (like PathwayLibrary)
                    transition["strength"] = min(
                        3.0,
                        transition["strength"] + 0.1 * (1.0 / transition["strength"]),
                    )
                else:
                    # Weaken but don't kill
                    transition["strength"] = max(0.3, transition["strength"] * 0.85)

    def reset(self):
        """Reset state for a new conversation."""
        self._observations = []
        for transition in self._transition_library.values():
            transition["strength"] = 1.0
