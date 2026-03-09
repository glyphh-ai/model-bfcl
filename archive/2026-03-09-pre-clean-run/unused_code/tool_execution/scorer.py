"""Glyphh Ada — Tool Execution scorer.

Unified scorer that works across ALL API classes simultaneously.
Encodes all available function definitions into one GlyphSpace,
then scores queries against the full pool.

Usage:
    scorer = ToolExecutionScorer()
    scorer.configure(func_defs)  # all funcs from all involved classes
    result = scorer.score("Move final_report.pdf to temp directory")
    # result.functions = ["GorillaFileSystem.mv", "GorillaFileSystem.cp", ...]
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import psycopg2

from glyphh import Encoder
from glyphh.core.types import Concept, Glyph
from glyphh.cognitive.glyph_space import GlyphSpace
from glyphh.cognitive.model_scorer import ScorerResult
from glyphh.core.ops import cosine_similarity

DB_DSN = "postgresql://postgres:postgres@localhost:5434/bfcl"

import importlib.util
import sys

# Explicit import from this directory to avoid bfcl/encoder.py shadowing
_THIS_DIR = Path(__file__).parent
_enc_spec = importlib.util.spec_from_file_location("_te_encoder", _THIS_DIR / "encoder.py")
_enc_mod = importlib.util.module_from_spec(_enc_spec)
_enc_spec.loader.exec_module(_enc_mod)
ENCODER_CONFIG = _enc_mod.ENCODER_CONFIG
encode_query = _enc_mod.encode_query
encode_function = _enc_mod.encode_function

_BFCL_DIR = Path(__file__).parent.parent
_FUNC_DOC_DIR = _BFCL_DIR / "data" / "bfcl" / "multi_turn_func_doc"

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
_DIR_TO_CLASS = {v: k for k, v in _CLASS_DIR_MAP.items()}

_FUNC_DOC_FILES = {
    "gorilla_file_system": "gorilla_file_system.json",
    "twitter_api": "posting_api.json",
    "posting_api": "posting_api.json",
    "message_api": "message_api.json",
    "ticket_api": "ticket_api.json",
    "math_api": "math_api.json",
    "trading_bot": "trading_bot.json",
    "travel_booking": "travel_booking.json",
    "vehicle_control": "vehicle_control.json",
}


class LayerWeightedStrategy:
    """Layer-weighted scoring using EncoderConfig weights."""

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


def load_func_defs(class_dirs: list[str]) -> list[dict]:
    """Load function definitions for the given class directories."""
    all_defs = []
    seen = set()
    for class_dir in class_dirs:
        fname = _FUNC_DOC_FILES.get(class_dir)
        if not fname:
            continue
        class_prefix = _DIR_TO_CLASS[class_dir]
        path = _FUNC_DOC_DIR / fname
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                fd = json.loads(line)
                name = fd.get("name", "")
                # Normalize class prefix
                if "." in name:
                    bare = name.split(".", 1)[1]
                    fd["name"] = f"{class_prefix}.{bare}"
                else:
                    fd["name"] = f"{class_prefix}.{name}"
                # Deduplicate (twitter_api and posting_api share func doc)
                if fd["name"] not in seen:
                    seen.add(fd["name"])
                    all_defs.append(fd)
    return all_defs


class ToolExecutionScorer:
    """Scores queries against all available functions using the 3-layer model."""

    def __init__(self):
        self._encoder = Encoder(ENCODER_CONFIG)
        self._strategy = LayerWeightedStrategy(ENCODER_CONFIG)
        self._space = GlyphSpace(scoring_strategy=self._strategy)
        self._func_glyphs: dict[str, Glyph] = {}
        self._func_defs: dict[str, dict] = {}  # name → func_def (for arg extraction)

    def configure(self, func_defs: list[dict]) -> None:
        """Encode function definitions into GlyphSpace."""
        self._func_glyphs = {}
        self._func_defs = {}
        for fd in func_defs:
            concept_dict = encode_function(fd)
            glyph = self._encoder.encode(
                Concept(name=concept_dict["name"], attributes=concept_dict["attributes"])
            )
            self._func_glyphs[fd["name"]] = glyph
            self._func_defs[fd["name"]] = fd
        self._space.configure(self._func_glyphs)

    def score(self, query: str) -> ScorerResult:
        """Score a query against all configured functions."""
        concept_dict = encode_query(query)
        query_glyph = self._encoder.encode(
            Concept(name=concept_dict["name"], attributes=concept_dict["attributes"])
        )
        return self._space.find_similar(query_glyph)

    def configure_from_db(self, class_dirs: list[str] | None = None) -> None:
        """Load pre-encoded function glyphs from pgvector.

        If class_dirs is None, loads ALL tool_execution glyphs.
        If class_dirs is provided, filters to only functions from those classes.
        """
        conn = psycopg2.connect(DB_DSN)
        cur = conn.cursor()
        cur.execute(
            "SELECT func_name, glyph_pickle FROM function_glyphs WHERE class_dir = %s",
            ("tool_execution",),
        )
        self._func_glyphs = {}
        if class_dirs:
            # Filter: only include functions whose class prefix matches
            allowed_prefixes = {_DIR_TO_CLASS[cd] for cd in class_dirs if cd in _DIR_TO_CLASS}
        else:
            allowed_prefixes = None

        for func_name, glyph_bytes in cur.fetchall():
            if allowed_prefixes:
                prefix = func_name.split(".")[0] if "." in func_name else ""
                if prefix not in allowed_prefixes:
                    continue
            self._func_glyphs[func_name] = pickle.loads(bytes(glyph_bytes))

        cur.close()
        conn.close()
        self._space.configure(self._func_glyphs)

    def get_func_def(self, func_name: str) -> dict | None:
        """Get the function definition for argument extraction."""
        return self._func_defs.get(func_name)
