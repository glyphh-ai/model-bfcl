"""Glyphh Ada 1.0 — BFCL Model Scorer.

Implements ModelScorer protocol using GlyphSpace + per-class encoders.
Each API class has its own intent.py + encoder.py; this scorer detects
the class from the function definitions and delegates accordingly.

Usage:
    scorer = BFCLModelScorer()
    scorer.configure(func_defs)
    result = scorer.score("list files in the directory")
"""

from __future__ import annotations

import importlib.util
import pickle
import sys
from pathlib import Path
from typing import Any

import psycopg2

from glyphh import Encoder
from glyphh.core.ops import cosine_similarity
from glyphh.core.types import Concept, Glyph
from glyphh.cognitive.glyph_space import GlyphSpace, ScoringStrategy
from glyphh.cognitive.model_scorer import ScorerResult

DB_DSN = "postgresql://postgres:postgres@localhost:5434/bfcl"

_CLASSES_DIR = Path(__file__).parent / "classes"

# Map class prefixes to directory names
_CLASS_DIR_MAP = {
    "GorillaFileSystem": "gorilla_file_system",
    "TwitterAPI": "twitter_api",
    "PostingAPI": "posting_api",
    "MessageAPI": "message_api",
    "TicketAPI": "ticket_api",
    "MathAPI": "math_api",
    "TradingBot": "trading_bot",
    "TravelAPI": "travel_booking",
    "VehicleControlAPI": "vehicle_control",
}

# Reverse: directory name → class prefix
_DIR_TO_CLASS = {v: k for k, v in _CLASS_DIR_MAP.items()}

# Func doc JSON files per class
_FUNC_DOC_DIR = Path(__file__).parent / "data" / "bfcl" / "multi_turn_func_doc"

# Module cache — load each class module only once
_MODULE_CACHE: dict[str, Any] = {}


def _load_class_module(class_dir: str, module_name: str):
    """Load a module from a class directory using importlib (no sys.path pollution).

    Critical: each class's encoder.py does `from intent import ...` which uses
    the plain 'intent' module name. We must evict stale modules before loading
    a new class to prevent cross-class contamination.
    """
    cache_key = f"{class_dir}_{module_name}"
    if cache_key in _MODULE_CACHE:
        return _MODULE_CACHE[cache_key]

    mod_path = _CLASSES_DIR / class_dir / f"{module_name}.py"
    if not mod_path.exists():
        raise FileNotFoundError(f"No {module_name}.py in {class_dir}")

    unique_name = f"_bfcl_{class_dir}_{module_name}"
    spec = importlib.util.spec_from_file_location(unique_name, mod_path)
    mod = importlib.util.module_from_spec(spec)

    # Evict plain-name modules that encoder.py imports via `from intent import ...`
    # to prevent cross-class contamination between different class directories.
    _stale = {}
    for plain_name in ("intent", "encoder"):
        if plain_name in sys.modules:
            _stale[plain_name] = sys.modules.pop(plain_name)

    class_path = str(_CLASSES_DIR / class_dir)
    inserted = class_path not in sys.path
    if inserted:
        sys.path.insert(0, class_path)
    try:
        spec.loader.exec_module(mod)
    finally:
        if inserted and class_path in sys.path:
            sys.path.remove(class_path)

    sys.modules[unique_name] = mod
    _MODULE_CACHE[cache_key] = mod
    return mod


def _detect_class(func_defs: list[dict], class_dir: str | None = None) -> str | None:
    """Detect API class from function name prefixes or explicit class_dir."""
    if class_dir:
        return _DIR_TO_CLASS.get(class_dir)
    for fd in func_defs:
        name = fd.get("name", "")
        if "." in name:
            prefix = name.split(".")[0]
            if prefix in _CLASS_DIR_MAP:
                return prefix
    return None


class LayerWeightedStrategy:
    """Layer-weighted scoring using EncoderConfig weights.

    Computes per-layer scores (weighted average of role similarities),
    then combines layers using their similarity_weight.
    """

    def __init__(self, encoder_config):
        self._config = encoder_config

    def score_pair(self, query_glyph: Glyph, target_glyph: Glyph) -> float:
        layer_configs = {lc.name: lc for lc in self._config.layers}
        total_score = 0.0
        total_weight = 0.0

        for lname in query_glyph.layers:
            if lname not in target_glyph.layers:
                continue
            lc = layer_configs.get(lname)
            if not lc:
                continue
            lw = lc.similarity_weight

            role_score = 0.0
            role_weight = 0.0
            for seg_cfg in lc.segments:
                sname = seg_cfg.name
                ql = query_glyph.layers[lname]
                fl = target_glyph.layers[lname]
                if sname not in ql.segments or sname not in fl.segments:
                    continue
                for role_cfg in seg_cfg.roles:
                    rname = role_cfg.name
                    qs = ql.segments[sname]
                    fs = fl.segments[sname]
                    if rname in qs.roles and rname in fs.roles:
                        sim = float(cosine_similarity(
                            qs.roles[rname].data, fs.roles[rname].data))
                        rw = role_cfg.similarity_weight
                        role_score += sim * rw
                        role_weight += rw

            if role_weight > 0:
                layer_sim = role_score / role_weight
                total_score += layer_sim * lw
                total_weight += lw

        return total_score / total_weight if total_weight > 0 else 0.0


class BFCLModelScorer:
    """BFCL model scorer — routes queries to functions using per-class Glyphh models.

    Implements the ModelScorer protocol:
        configure(func_defs) → encode functions into GlyphSpace
        score(query) → ScorerResult with ranked functions
    """

    IRRELEVANCE_THRESHOLD = 0.15

    def __init__(self):
        self._encoder: Encoder | None = None
        self._space: GlyphSpace | None = None
        self._encode_query_fn = None
        self._class_prefix: str = ""
        self._class_dir: str = ""
        self._func_glyphs: dict[str, Glyph] = {}

    def configure(self, func_defs: list[dict[str, Any]], class_dir: str | None = None) -> None:
        """Encode function definitions into GlyphSpace."""
        class_name = _detect_class(func_defs, class_dir=class_dir)
        if not class_name:
            raise ValueError("Cannot detect API class from function definitions")

        self._class_prefix = class_name
        self._class_dir = _CLASS_DIR_MAP[class_name]

        encoder_mod = _load_class_module(self._class_dir, "encoder")
        config = encoder_mod.ENCODER_CONFIG
        encode_fn = encoder_mod.encode_function
        self._encode_query_fn = encoder_mod.encode_query

        self._encoder = Encoder(config)
        strategy = LayerWeightedStrategy(config)
        self._space = GlyphSpace(scoring_strategy=strategy)

        self._func_glyphs = {}
        for fd in func_defs:
            concept_dict = encode_fn(fd)
            glyph = self._encoder.encode(
                Concept(name=concept_dict["name"], attributes=concept_dict["attributes"])
            )
            self._func_glyphs[fd["name"]] = glyph

        self._space.configure(self._func_glyphs)

    def score(self, query: str) -> ScorerResult:
        """Score a query against configured functions."""
        if not self._space or not self._encoder or not self._encode_query_fn:
            return ScorerResult()

        concept_dict = self._encode_query_fn(query)
        query_glyph = self._encoder.encode(
            Concept(name=concept_dict["name"], attributes=concept_dict["attributes"])
        )
        return self._space.find_similar(query_glyph)

    def encode_query(self, query: str) -> Glyph | None:
        if not self._encoder or not self._encode_query_fn:
            return None
        concept_dict = self._encode_query_fn(query)
        return self._encoder.encode(
            Concept(name=concept_dict["name"], attributes=concept_dict["attributes"])
        )

    def configure_from_db(self, class_dir: str) -> None:
        """Load pre-encoded function glyphs from pgvector instead of re-encoding."""
        class_name = _DIR_TO_CLASS.get(class_dir)
        if not class_name:
            raise ValueError(f"Unknown class_dir: {class_dir}")

        self._class_prefix = class_name
        self._class_dir = class_dir

        encoder_mod = _load_class_module(class_dir, "encoder")
        config = encoder_mod.ENCODER_CONFIG
        self._encode_query_fn = encoder_mod.encode_query

        self._encoder = Encoder(config)
        strategy = LayerWeightedStrategy(config)
        self._space = GlyphSpace(scoring_strategy=strategy)

        conn = psycopg2.connect(DB_DSN)
        cur = conn.cursor()
        cur.execute(
            "SELECT func_name, glyph_pickle FROM function_glyphs WHERE class_dir = %s",
            (class_dir,),
        )
        self._func_glyphs = {}
        for func_name, glyph_bytes in cur.fetchall():
            self._func_glyphs[func_name] = pickle.loads(bytes(glyph_bytes))
        cur.close()
        conn.close()

        self._space.configure(self._func_glyphs)

    def get_func_glyphs(self) -> dict[str, Glyph]:
        return dict(self._func_glyphs)

    def scoring_strategy(self) -> LayerWeightedStrategy | None:
        if self._class_dir:
            encoder_mod = _load_class_module(self._class_dir, "encoder")
            return LayerWeightedStrategy(encoder_mod.ENCODER_CONFIG)
        return None
