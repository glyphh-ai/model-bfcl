"""Encode all 9 BFCL class function glyphs and store in pgvector.

Run once (or after encoder changes) to populate the database:
    cd glyphh-models
    PYTHONPATH=../glyphh-runtime python bfcl/load_db.py

Then tests query the DB instead of re-encoding every run.
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import psycopg2

# Setup paths
_BFCL_DIR = Path(__file__).parent
sys.path.insert(0, str(_BFCL_DIR))

from scorer import _CLASS_DIR_MAP, _DIR_TO_CLASS, _CLASSES_DIR, _load_class_module, _MODULE_CACHE

# Runtime
_RUNTIME = _BFCL_DIR.parent.parent / "glyphh-runtime"
if str(_RUNTIME) not in sys.path:
    sys.path.insert(0, str(_RUNTIME))

from glyphh import Encoder
from glyphh.core.types import Concept

_FUNC_DOC_DIR = _BFCL_DIR / "data" / "bfcl" / "multi_turn_func_doc"

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

DB_DSN = "postgresql://postgres:postgres@localhost:5434/bfcl"

CREATE_TABLE = """
CREATE TABLE IF NOT EXISTS function_glyphs (
    id SERIAL PRIMARY KEY,
    class_dir TEXT NOT NULL,
    func_name TEXT NOT NULL,
    cortex vector(2000),
    glyph_pickle BYTEA NOT NULL,
    UNIQUE (class_dir, func_name)
);
"""


def _load_func_defs(class_dir: str) -> list[dict]:
    fname = _FUNC_DOC_FILES[class_dir]
    path = _FUNC_DOC_DIR / fname
    with open(path) as f:
        func_defs = [json.loads(line) for line in f if line.strip()]
    class_prefix = _DIR_TO_CLASS[class_dir]
    for fd in func_defs:
        name = fd.get("name", "")
        if "." in name:
            old_prefix = name.split(".")[0]
            if old_prefix != class_prefix:
                fd["name"] = f"{class_prefix}.{name.split('.', 1)[1]}"
    return func_defs


def main():
    conn = psycopg2.connect(DB_DSN)
    conn.autocommit = True
    cur = conn.cursor()

    # Create table
    cur.execute(CREATE_TABLE)
    cur.execute("DELETE FROM function_glyphs")  # fresh load

    total = 0
    for class_dir in _CLASS_DIR_MAP.values():
        # Clear module cache to avoid cross-class contamination
        _MODULE_CACHE.clear()
        for name in list(sys.modules.keys()):
            if name in ("intent", "encoder"):
                del sys.modules[name]

        encoder_mod = _load_class_module(class_dir, "encoder")
        config = encoder_mod.ENCODER_CONFIG
        encode_fn = encoder_mod.encode_function

        encoder = Encoder(config)

        func_defs = _load_func_defs(class_dir)
        count = 0
        for fd in func_defs:
            concept_dict = encode_fn(fd)
            glyph = encoder.encode(
                Concept(name=concept_dict["name"], attributes=concept_dict["attributes"])
            )

            # Cortex vector as list for pgvector
            cortex_list = glyph.global_cortex.data.tolist()
            # Pad to 2000 if needed
            while len(cortex_list) < 2000:
                cortex_list.append(0)

            glyph_bytes = pickle.dumps(glyph)

            cur.execute(
                """INSERT INTO function_glyphs (class_dir, func_name, cortex, glyph_pickle)
                   VALUES (%s, %s, %s, %s)
                   ON CONFLICT (class_dir, func_name) DO UPDATE
                   SET cortex = EXCLUDED.cortex, glyph_pickle = EXCLUDED.glyph_pickle""",
                (class_dir, fd["name"], str(cortex_list), glyph_bytes),
            )
            count += 1

        print(f"  {class_dir}: {count} functions")
        total += count

    cur.close()
    conn.close()
    print(f"\nLoaded {total} function glyphs into pgvector.")


if __name__ == "__main__":
    main()
