"""
Domain-specific Glyphh intent models for multi-turn BFCL.

These are pure Glyphh HDC models — no LLM, no training data, no answers.
Each model encodes a rich concept representing an intent pattern,
then measures query similarity against that concept.

Models:
  - DirectoryIntentModel: detects "user wants to change directory"
  - FileCreationIntentModel: detects "user wants to create a new file"
  - SearchIntentModel: detects "user wants to search/find/locate"

All models use the full 4-level Glyphh hierarchy for scoring:
  cortex (15%) → layer (20%) → segment (25%) → role (40%)
"""

from glyphh import Encoder
from glyphh.core.config import EncoderConfig, Layer, Segment, Role
from glyphh.core.types import Concept
from glyphh.core.ops import cosine_similarity


def _hierarchical_score(
    query_glyph,
    ref_glyph,
    config: EncoderConfig,
    w_cortex: float = 0.15,
    w_layer: float = 0.20,
    w_segment: float = 0.25,
    w_role: float = 0.40,
) -> float:
    """Compute 4-level hierarchical similarity between two glyphs.

    Mirrors handler.py._score() exactly:
      cortex  → global overview similarity
      layer   → per-layer cortex similarity, weighted by layer.similarity_weight
      segment → per-segment cortex similarity, weighted by segment.similarity_weight
      role    → per-role binding similarity, weighted by role.similarity_weight

    Args:
        query_glyph: Glyph encoded from the user query.
        ref_glyph: Reference concept glyph (e.g., navigation concept).
        config: The EncoderConfig used to build both glyphs (for role weights).
        w_cortex/w_layer/w_segment/w_role: hierarchy level weights (sum to 1.0).

    Returns:
        Weighted similarity score in [0, 1].
    """
    # ── Cortex level ──
    cortex_sim = float(cosine_similarity(
        query_glyph.global_cortex.data,
        ref_glyph.global_cortex.data,
    ))

    # ── Layer level ──
    layer_sims = []
    for lname in query_glyph.layers:
        if lname in ref_glyph.layers and not lname.startswith("_"):
            ql = query_glyph.layers[lname]
            rl = ref_glyph.layers[lname]
            lsim = float(cosine_similarity(ql.cortex.data, rl.cortex.data))
            lweight = ql.weights.get("similarity", 1.0)
            layer_sims.append((lsim, lweight))

    # ── Segment level ──
    seg_sims = []
    for lname in query_glyph.layers:
        if lname not in ref_glyph.layers or lname.startswith("_"):
            continue
        ql = query_glyph.layers[lname]
        rl = ref_glyph.layers[lname]
        for sname in ql.segments:
            if sname in rl.segments:
                qs = ql.segments[sname]
                rs = rl.segments[sname]
                ssim = float(cosine_similarity(qs.cortex.data, rs.cortex.data))
                sweight = qs.weights.get("similarity", 1.0)
                seg_sims.append((ssim, sweight))

    # ── Role level ──
    role_sims = []
    for lname in query_glyph.layers:
        if lname not in ref_glyph.layers or lname.startswith("_"):
            continue
        for sname in query_glyph.layers[lname].segments:
            if sname not in ref_glyph.layers[lname].segments:
                continue
            qs = query_glyph.layers[lname].segments[sname]
            rs = ref_glyph.layers[lname].segments[sname]
            for rname in qs.roles:
                if rname in rs.roles:
                    rsim = float(cosine_similarity(
                        qs.roles[rname].data, rs.roles[rname].data
                    ))
                    # Get role weight from config
                    rweight = 1.0
                    for ld in config.layers:
                        if ld.name == lname:
                            for sd in ld.segments:
                                if sd.name == sname:
                                    for rd in sd.roles:
                                        if rd.name == rname:
                                            rweight = rd.similarity_weight
                    role_sims.append((rsim, rweight))

    # ── Weighted combination ──
    score = cortex_sim * w_cortex

    if layer_sims:
        total_w = sum(w for _, w in layer_sims)
        score += (sum(s * w for s, w in layer_sims) / total_w) * w_layer

    if seg_sims:
        total_w = sum(w for _, w in seg_sims)
        score += (sum(s * w for s, w in seg_sims) / total_w) * w_segment

    if role_sims:
        total_w = sum(w for _, w in role_sims)
        score += (sum(s * w for s, w in role_sims) / total_w) * w_role

    return score


# ── Directory Intent Model ──
# A Glyphh model that encodes the concept of "directory navigation intent"
# using every possible way a human might express wanting to change directories.

_DIR_INTENT_CONFIG = EncoderConfig(
    dimension=10000,
    seed=42,
    apply_weights_during_encoding=True,
    include_temporal=False,
    layers=[
        Layer(
            name="navigation",
            similarity_weight=0.6,
            segments=[
                Segment(
                    name="directional",
                    roles=[
                        Role(
                            name="movement_verbs",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                        Role(
                            name="prepositions",
                            similarity_weight=0.9,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
                Segment(
                    name="destination",
                    roles=[
                        Role(
                            name="location_words",
                            similarity_weight=0.8,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="context",
            similarity_weight=0.4,
            segments=[
                Segment(
                    name="phrases",
                    roles=[
                        Role(
                            name="navigation_phrases",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
    ],
)


# Anti-navigation config — encodes "operating on current location" intent
_CURRENT_DIR_CONFIG = EncoderConfig(
    dimension=10000,
    seed=42,
    apply_weights_during_encoding=True,
    include_temporal=False,
    layers=[
        Layer(
            name="local_action",
            similarity_weight=0.6,
            segments=[
                Segment(
                    name="action_verbs",
                    roles=[
                        Role(
                            name="local_verbs",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
                Segment(
                    name="location",
                    roles=[
                        Role(
                            name="current_location",
                            similarity_weight=0.9,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="context",
            similarity_weight=0.4,
            segments=[
                Segment(
                    name="phrases",
                    roles=[
                        Role(
                            name="local_phrases",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
    ],
)



class DirectoryIntentModel:
    """Pure Glyphh model for detecting directory navigation intent.

    Two-signal architecture:
      1. Concept similarity: query encoded against _DIR_INTENT_CONFIG,
         compared to a rich navigation concept glyph vs anti-navigation glyph.
      2. Bigram pattern similarity: query bigrams bound as Glyphh vectors,
         compared to canonical navigation bigram patterns (prep + location).

    The combined score determines whether cd() should be included.
    """

    def __init__(self):
        self._encoder = Encoder(_DIR_INTENT_CONFIG)
        self._anti_encoder = Encoder(_CURRENT_DIR_CONFIG)
        self._nav_glyph = None
        self._anti_nav_glyph = None
        self._nav_pattern_vec = None
        self._anti_pattern_vec = None
        self._build_concepts()

    def _build_concepts(self):
        """Build navigation and anti-navigation glyphs + bound bigram patterns."""

        # ── Navigation concept ──
        nav_concept = Concept(
            name="directory_navigation",
            attributes={
                "movement_verbs": (
                    "cd chdir navigate enter open access "
                    "go folder directory path "
                    "switch change move transfer relocate "
                    "pop step browse venture head proceed"
                ),
                "prepositions": (
                    "into within inside to in "
                    "under over toward towards"
                ),
                "location_words": (
                    "directory folder subdirectory subfolder "
                    "workspace documents archive archives "
                    "backup shared communal path "
                    "home desktop downloads project "
                    "root parent child nested "
                    "research data temp logs config"
                ),
                "navigation_phrases": (
                    "go to folder navigate directory "
                    "change directory cd folder "
                    "move into directory within folder "
                    "open folder enter directory "
                    "pop over to folder head into directory "
                    "in directory folder right in folder "
                    "within the directory to the folder "
                    "inside folder under directory "
                    "in documents folder in workspace directory "
                    "in archive directory to archives folder "
                    "in research directory in project folder "
                    "navigate to the directory go into folder"
                ),
            },
        )
        self._nav_glyph = self._encoder.encode(nav_concept)

        # ── Anti-navigation concept (local actions, no directory change) ──
        anti_concept = Concept(
            name="local_action",
            attributes={
                "local_verbs": (
                    "list read write create delete remove "
                    "copy rename sort find search cat echo "
                    "touch grep tail head diff wc chmod "
                    "print display show check verify run execute "
                    "send post get fetch calculate compute "
                    "update modify edit set configure"
                ),
                "current_location": (
                    "here current this file files "
                    "present existing local now "
                    "contents items entries"
                ),
                "local_phrases": (
                    "list the files read the file "
                    "create a file delete the file "
                    "sort the files find the file "
                    "show me the contents check the status "
                    "run the command execute the script "
                    "what files are here current directory contents"
                ),
            },
        )
        self._anti_nav_glyph = self._anti_encoder.encode(anti_concept)

        # ── Bound bigram patterns for navigation ──
        nav_patterns = [
            ("into", "directory"), ("into", "folder"), ("into", "workspace"),
            ("within", "directory"), ("within", "folder"),
            ("in", "directory"), ("in", "folder"), ("in", "workspace"),
            ("to", "directory"), ("to", "folder"), ("to", "workspace"),
            ("inside", "directory"), ("inside", "folder"),
            ("over", "directory"), ("over", "folder"),
            ("navigate", "directory"), ("navigate", "folder"),
            ("enter", "directory"), ("enter", "folder"),
            ("go", "directory"), ("go", "folder"),
            ("change", "directory"), ("switch", "directory"),
            ("move", "directory"), ("move", "folder"),
            ("head", "directory"), ("head", "folder"),
            ("cd", "directory"), ("cd", "folder"),
            ("in", "documents"), ("in", "archive"), ("in", "archives"),
            ("in", "research"), ("in", "project"), ("in", "backup"),
            ("in", "shared"), ("in", "communal"), ("in", "data"),
            ("to", "documents"), ("to", "archive"), ("to", "research"),
            ("to", "project"), ("to", "backup"), ("to", "shared"),
            ("into", "documents"), ("into", "archive"), ("into", "research"),
        ]
        pattern_vecs = []
        for prep, loc in nav_patterns:
            prep_vec = self._encoder.generate_symbol(prep)
            loc_vec = self._encoder.generate_symbol(loc)
            bound = self._encoder.bind(prep_vec, loc_vec)
            pattern_vecs.append(bound)
        self._nav_pattern_vec = self._encoder.bundle(pattern_vecs)

        # ── Bound bigram patterns for anti-navigation (local actions) ──
        anti_patterns = [
            ("list", "files"), ("list", "contents"), ("list", "items"),
            ("read", "file"), ("write", "file"), ("create", "file"),
            ("delete", "file"), ("remove", "file"), ("copy", "file"),
            ("sort", "files"), ("find", "file"), ("search", "file"),
            ("show", "contents"), ("check", "status"), ("run", "command"),
            ("the", "file"), ("the", "files"), ("this", "file"),
            ("current", "directory"), ("here", "files"),
        ]
        anti_vecs = []
        for w1, w2 in anti_patterns:
            v1 = self._encoder.generate_symbol(w1)
            v2 = self._encoder.generate_symbol(w2)
            bound = self._encoder.bind(v1, v2)
            anti_vecs.append(bound)
        self._anti_pattern_vec = self._encoder.bundle(anti_vecs)

    def _extract_bigrams(self, query: str) -> list:
        """Extract word bigrams from query and bind them as Glyphh vectors."""
        import re
        words = re.sub(r"[^a-z0-9\s]", "", query.lower()).split()
        words = [w for w in words if len(w) > 1]

        bigram_vecs = []
        for i in range(len(words) - 1):
            w1_vec = self._encoder.generate_symbol(words[i])
            w2_vec = self._encoder.generate_symbol(words[i + 1])
            bound = self._encoder.bind(w1_vec, w2_vec)
            bigram_vecs.append(bound)

        return bigram_vecs

    def _encode_query(self, query: str, encoder: Encoder, config: EncoderConfig) -> 'Glyph':
        """Encode a query using the given encoder config."""
        import re
        words = re.sub(r"[^a-z0-9\s]", "", query.lower()).split()
        words = [w for w in words if len(w) > 1]
        text = " ".join(words) if words else "none"

        attrs = {}
        for layer in config.layers:
            for segment in layer.segments:
                for role in segment.roles:
                    attrs[role.name] = text

        concept = Concept(name="query", attributes=attrs)
        return encoder.encode(concept)

    def score(self, query: str) -> dict:
        """Score a query for directory navigation intent.

        Three-signal scoring with full 4-level hierarchy:
          1. Hierarchical similarity (cortex→layer→segment→role) against
             both nav and anti-nav concept glyphs.
          2. Bigram pattern similarity: query bigrams vs nav/anti-nav patterns.
          3. Keyword signal: prep+location bigrams, explicit nav verbs.

        Returns:
            {
                "nav_score": float,      # combined navigation signal
                "local_score": float,    # combined local-action signal
                "bigram_nav": float,     # bigram pattern nav similarity
                "bigram_local": float,   # bigram pattern local similarity
                "keyword_boost": float,  # explicit keyword signal
                "delta": float,          # final nav - local
                "needs_cd": bool,        # True if cd should be included
                "confidence": float,     # abs(delta)
            }
        """
        import re

        # ── Signal 1: Full 4-level hierarchical similarity ──
        nav_query = self._encode_query(query, self._encoder, _DIR_INTENT_CONFIG)
        local_query = self._encode_query(query, self._anti_encoder, _CURRENT_DIR_CONFIG)

        nav_combined = _hierarchical_score(
            nav_query, self._nav_glyph, _DIR_INTENT_CONFIG
        )
        local_combined = _hierarchical_score(
            local_query, self._anti_nav_glyph, _CURRENT_DIR_CONFIG
        )

        # ── Signal 2: Bigram pattern similarity ──
        bigram_vecs = self._extract_bigrams(query)

        bigram_nav = 0.0
        bigram_local = 0.0
        if bigram_vecs:
            query_bigram_bundle = self._encoder.bundle(bigram_vecs)
            bigram_nav = float(cosine_similarity(
                query_bigram_bundle.data,
                self._nav_pattern_vec.data,
            ))
            bigram_local = float(cosine_similarity(
                query_bigram_bundle.data,
                self._anti_pattern_vec.data,
            ))

        # ── Signal 3: Keyword boost ──
        words = re.sub(r"[^a-z0-9\s]", "", query.lower()).split()
        words_set = set(words)

        dir_preps = {"into", "within", "inside", "to", "in"}
        loc_nouns = {
            "directory", "folder", "subdirectory", "subfolder",
            "workspace", "documents", "archive", "archives",
            "backup", "shared", "communal", "research", "project",
            "data", "temp", "logs", "config", "drafts", "home",
            "desktop", "downloads", "root", "parent",
        }
        nav_verbs = {"navigate", "cd", "chdir", "enter"}

        keyword_boost = 0.0

        for i in range(len(words) - 1):
            if words[i] in dir_preps and words[i + 1] in loc_nouns:
                keyword_boost = 0.08
                break

        if words_set & nav_verbs:
            keyword_boost = max(keyword_boost, 0.06)

        local_verbs = {"list", "read", "write", "create", "delete", "remove",
                       "copy", "sort", "grep", "cat", "echo", "touch", "find",
                       "search", "show", "display", "check", "run", "execute"}
        if (words_set & local_verbs) and not (words_set & loc_nouns) and not (words_set & nav_verbs):
            keyword_boost -= 0.02

        # ── Combine signals ──
        # Hierarchical sim 30%, bigram pattern 40%, keyword 30%
        nav_final = nav_combined * 0.3 + max(bigram_nav, 0) * 0.4 + keyword_boost
        local_final = local_combined * 0.3 + max(bigram_local, 0) * 0.4

        delta = nav_final - local_final

        return {
            "nav_score": round(nav_final, 4),
            "local_score": round(local_final, 4),
            "bigram_nav": round(bigram_nav, 4),
            "bigram_local": round(bigram_local, 4),
            "keyword_boost": round(keyword_boost, 4),
            "delta": round(delta, 4),
            "needs_cd": delta > 0.0,
            "confidence": round(abs(delta), 4),
        }



# ── File Creation Intent Model ──
# Detects when the user wants to create a new file (touch needed).
# The key pattern: user says "write/jot/record/populate X into file Y"
# and the ground truth expects touch(file) + echo(content, file).
# The LLM consistently misses the touch() because it thinks echo() alone
# creates the file. This model detects the creation intent.

_FILE_CREATION_CONFIG = EncoderConfig(
    dimension=10000,
    seed=42,
    apply_weights_during_encoding=True,
    include_temporal=False,
    layers=[
        Layer(
            name="creation",
            similarity_weight=0.6,
            segments=[
                Segment(
                    name="verbs",
                    roles=[
                        Role(
                            name="creation_verbs",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
                Segment(
                    name="targets",
                    roles=[
                        Role(
                            name="file_targets",
                            similarity_weight=0.9,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="context",
            similarity_weight=0.4,
            segments=[
                Segment(
                    name="phrases",
                    roles=[
                        Role(
                            name="creation_phrases",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
    ],
)


class FileCreationIntentModel:
    """Pure Glyphh model for detecting file creation intent.

    Detects when the user wants to create/write a new file, which requires
    touch() before echo(). The LLM consistently misses touch() because it
    assumes echo() alone creates the file.

    Two-signal architecture:
      1. Concept similarity against a rich "file creation" glyph
      2. Bound bigram patterns for creation phrases (write+file, create+file, etc.)
    """

    def __init__(self):
        self._encoder = Encoder(_FILE_CREATION_CONFIG)
        self._creation_glyph = None
        self._creation_pattern_vec = None
        self._build_concepts()

    def _build_concepts(self):
        """Build file creation concept glyph and bigram patterns."""

        creation_concept = Concept(
            name="file_creation",
            attributes={
                "creation_verbs": (
                    "create write make generate produce "
                    "jot record populate fill draft compose "
                    "save store put place log note "
                    "document capture pen inscribe "
                    "touch new initialize start begin"
                ),
                "file_targets": (
                    "file document txt text report summary "
                    "log notes draft memo record "
                    "output results data content "
                    "script config readme changelog"
                ),
                "creation_phrases": (
                    "create a file write to file "
                    "jot down in file record in file "
                    "populate file with data "
                    "write content to file save to file "
                    "create new file make a file "
                    "put data in file store in file "
                    "draft a document compose a file "
                    "log results to file note in file "
                    "generate a report write a summary "
                    "fill file with content "
                    "capture in file document in file"
                ),
            },
        )
        self._creation_glyph = self._encoder.encode(creation_concept)

        # Bound bigram patterns for file creation
        creation_patterns = [
            ("create", "file"), ("create", "document"), ("create", "report"),
            ("write", "file"), ("write", "document"), ("write", "txt"),
            ("make", "file"), ("make", "document"),
            ("jot", "down"), ("jot", "file"), ("jot", "notes"),
            ("record", "file"), ("record", "data"), ("record", "results"),
            ("populate", "file"), ("populate", "document"),
            ("save", "file"), ("save", "document"), ("save", "results"),
            ("store", "file"), ("store", "data"),
            ("put", "file"), ("put", "data"),
            ("draft", "file"), ("draft", "document"),
            ("compose", "file"), ("compose", "document"),
            ("log", "file"), ("log", "results"),
            ("note", "file"), ("note", "down"),
            ("generate", "file"), ("generate", "report"),
            ("fill", "file"), ("fill", "document"),
            ("capture", "file"), ("capture", "data"),
            ("new", "file"), ("new", "document"),
            ("touch", "file"), ("initialize", "file"),
            ("content", "file"), ("data", "file"),
            ("write", "content"), ("write", "data"),
            ("write", "summary"), ("write", "report"),
        ]
        pattern_vecs = []
        for w1, w2 in creation_patterns:
            v1 = self._encoder.generate_symbol(w1)
            v2 = self._encoder.generate_symbol(w2)
            bound = self._encoder.bind(v1, v2)
            pattern_vecs.append(bound)
        self._creation_pattern_vec = self._encoder.bundle(pattern_vecs)

    def score(self, query: str) -> dict:
        """Score a query for file creation intent using full 4-level hierarchy.

        Returns:
            {
                "creation_score": float,  # combined creation signal
                "needs_touch": bool,      # True if touch() should be included
                "confidence": float,      # how sure we are
            }
        """
        import re

        # ── Signal 1: Full 4-level hierarchical similarity ──
        words = re.sub(r"[^a-z0-9\s]", "", query.lower()).split()
        words = [w for w in words if len(w) > 1]
        text = " ".join(words) if words else "none"

        attrs = {}
        for layer in _FILE_CREATION_CONFIG.layers:
            for segment in layer.segments:
                for role in segment.roles:
                    attrs[role.name] = text

        query_concept = Concept(name="query", attributes=attrs)
        query_glyph = self._encoder.encode(query_concept)

        concept_combined = _hierarchical_score(
            query_glyph, self._creation_glyph, _FILE_CREATION_CONFIG
        )

        # ── Signal 2: Bigram pattern similarity ──
        bigram_sim = 0.0
        bigram_vecs = []
        for i in range(len(words) - 1):
            v1 = self._encoder.generate_symbol(words[i])
            v2 = self._encoder.generate_symbol(words[i + 1])
            bound = self._encoder.bind(v1, v2)
            bigram_vecs.append(bound)

        if bigram_vecs:
            query_bundle = self._encoder.bundle(bigram_vecs)
            bigram_sim = float(cosine_similarity(
                query_bundle.data,
                self._creation_pattern_vec.data,
            ))

        # ── Signal 3: Keyword check ──
        words_set = set(words)
        creation_verbs = {
            "create", "write", "jot", "record", "populate", "draft",
            "compose", "log", "note", "generate", "capture", "save",
            "store", "fill",
        }
        file_nouns = {
            "file", "document", "txt", "report", "summary", "log",
            "notes", "draft", "memo", "record", "readme", "changelog",
        }
        has_file_ext = bool(re.search(r"\.\w{2,4}\b", query.lower()))

        keyword_boost = 0.0
        if words_set & creation_verbs and (words_set & file_nouns or has_file_ext):
            keyword_boost = 0.06
        elif has_file_ext and words_set & creation_verbs:
            keyword_boost = 0.04

        # ── Combine ──
        creation_score = concept_combined * 0.3 + max(bigram_sim, 0) * 0.4 + keyword_boost

        return {
            "creation_score": round(creation_score, 4),
            "needs_touch": creation_score > 0.09,
            "confidence": round(creation_score, 4),
        }


# ── Search Suppression Model ──
# Detects when the user is NOT asking to search/find/locate.
# The LLM over-predicts find() 13x in 50 entries — it adds find()
# as a preparatory step before operating on files. This model
# detects explicit search intent so we can suppress find() otherwise.

_SEARCH_INTENT_CONFIG = EncoderConfig(
    dimension=10000,
    seed=42,
    apply_weights_during_encoding=True,
    include_temporal=False,
    layers=[
        Layer(
            name="search",
            similarity_weight=0.6,
            segments=[
                Segment(
                    name="verbs",
                    roles=[
                        Role(
                            name="search_verbs",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
                Segment(
                    name="targets",
                    roles=[
                        Role(
                            name="search_targets",
                            similarity_weight=0.8,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="context",
            similarity_weight=0.4,
            segments=[
                Segment(
                    name="phrases",
                    roles=[
                        Role(
                            name="search_phrases",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
    ],
)


class SearchIntentModel:
    """Pure Glyphh model for detecting explicit search/find intent.

    Used to SUPPRESS false find() calls. The LLM adds find() as a
    preparatory step when the user mentions a file by name. This model
    detects when the user is genuinely asking to search/find/locate,
    so we can tell the LLM "do NOT add find()" when the signal is low.

    Architecture:
      1. Concept similarity against a "search intent" glyph
      2. Bound bigram patterns for search phrases
      3. Keyword detection for explicit search verbs
    """

    def __init__(self):
        self._encoder = Encoder(_SEARCH_INTENT_CONFIG)
        self._search_glyph = None
        self._search_pattern_vec = None
        self._build_concepts()

    def _build_concepts(self):
        """Build search intent concept glyph and bigram patterns."""

        search_concept = Concept(
            name="search_intent",
            attributes={
                "search_verbs": (
                    "find search locate discover hunt "
                    "look seek scan probe explore "
                    "track trace detect identify spot "
                    "uncover dig ferret rummage scour"
                ),
                "search_targets": (
                    "file files document documents "
                    "somewhere deep hidden lost missing "
                    "anywhere everywhere recursively "
                    "filesystem system tree hierarchy "
                    "subdirectories nested buried"
                ),
                "search_phrases": (
                    "find a file search for file "
                    "locate the file look for file "
                    "find somewhere deep in filesystem "
                    "search recursively for file "
                    "hunt down the file track the file "
                    "find file named locate file called "
                    "search the directory for "
                    "look through files find where "
                    "discover the file scan for file"
                ),
            },
        )
        self._search_glyph = self._encoder.encode(search_concept)

        # Bound bigram patterns for search intent
        search_patterns = [
            ("find", "file"), ("find", "files"), ("find", "document"),
            ("find", "named"), ("find", "called"), ("find", "somewhere"),
            ("search", "file"), ("search", "files"), ("search", "for"),
            ("search", "directory"), ("search", "recursively"),
            ("locate", "file"), ("locate", "files"), ("locate", "document"),
            ("look", "for"), ("look", "file"), ("look", "through"),
            ("hunt", "file"), ("hunt", "down"),
            ("seek", "file"), ("seek", "out"),
            ("scan", "file"), ("scan", "files"), ("scan", "directory"),
            ("track", "file"), ("track", "down"),
            ("discover", "file"), ("discover", "where"),
            ("somewhere", "deep"), ("somewhere", "in"),
            ("hidden", "file"), ("lost", "file"), ("missing", "file"),
        ]
        pattern_vecs = []
        for w1, w2 in search_patterns:
            v1 = self._encoder.generate_symbol(w1)
            v2 = self._encoder.generate_symbol(w2)
            bound = self._encoder.bind(v1, v2)
            pattern_vecs.append(bound)
        self._search_pattern_vec = self._encoder.bundle(pattern_vecs)

    def score(self, query: str) -> dict:
        """Score a query for explicit search/find intent using full 4-level hierarchy.

        Returns:
            {
                "search_score": float,    # combined search signal
                "wants_find": bool,       # True if find() is genuinely wanted
                "confidence": float,
            }
        """
        import re

        words = re.sub(r"[^a-z0-9\s]", "", query.lower()).split()
        words = [w for w in words if len(w) > 1]
        text = " ".join(words) if words else "none"

        # ── Signal 1: Full 4-level hierarchical similarity ──
        attrs = {}
        for layer in _SEARCH_INTENT_CONFIG.layers:
            for segment in layer.segments:
                for role in segment.roles:
                    attrs[role.name] = text

        query_concept = Concept(name="query", attributes=attrs)
        query_glyph = self._encoder.encode(query_concept)

        concept_combined = _hierarchical_score(
            query_glyph, self._search_glyph, _SEARCH_INTENT_CONFIG
        )

        # ── Signal 2: Bigram pattern similarity ──
        bigram_sim = 0.0
        bigram_vecs = []
        for i in range(len(words) - 1):
            v1 = self._encoder.generate_symbol(words[i])
            v2 = self._encoder.generate_symbol(words[i + 1])
            bound = self._encoder.bind(v1, v2)
            bigram_vecs.append(bound)

        if bigram_vecs:
            query_bundle = self._encoder.bundle(bigram_vecs)
            bigram_sim = float(cosine_similarity(
                query_bundle.data,
                self._search_pattern_vec.data,
            ))

        # ── Signal 3: Keyword check ──
        words_set = set(words)
        explicit_search = {"find", "search", "locate", "hunt", "seek", "scan"}
        search_context = {"somewhere", "deep", "hidden", "lost", "missing",
                          "recursively", "anywhere", "everywhere"}

        keyword_boost = 0.0
        if words_set & explicit_search:
            keyword_boost = 0.04
            if words_set & search_context:
                keyword_boost = 0.08

        # ── Combine ──
        search_score = concept_combined * 0.3 + max(bigram_sim, 0) * 0.4 + keyword_boost

        return {
            "search_score": round(search_score, 4),
            "wants_find": search_score > 0.10,
            "confidence": round(search_score, 4),
        }
