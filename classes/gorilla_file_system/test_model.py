"""Tests for the GorillaFileSystem Glyphh Ada model.

Verifies:
  1. Intent extraction: NL queries → correct action/target
  2. HDC encoding: queries and functions encode into compatible Glyphs
  3. Routing accuracy: each function is the top match for its representative queries
"""

import json
import sys
from pathlib import Path

import pytest

# Setup path — local dir MUST come first to shadow top-level intent.py
_DIR = Path(__file__).parent
_BFCL_DIR = _DIR.parent.parent
sys.path.insert(0, str(_DIR))

from intent import extract_intent, ACTION_TO_FUNC, FUNC_TO_ACTION, FUNC_TO_TARGET
from encoder import ENCODER_CONFIG, encode_query, encode_function

from glyphh import Encoder
from glyphh.core.types import Concept, Glyph
from glyphh.core.ops import cosine_similarity


def _layer_weighted_score(q: Glyph, f: Glyph) -> float:
    """Layer-weighted similarity using ENCODER_CONFIG weights.

    Computes per-layer scores (weighted-average of role similarities within
    each layer), then combines layers using their similarity_weight.
    This lets the intent layer's action role dominate over noisy BoW roles.
    """
    config = ENCODER_CONFIG
    layer_configs = {lc.name: lc for lc in config.layers}

    total_score = 0.0
    total_weight = 0.0

    for lname in q.layers:
        if lname not in f.layers:
            continue
        lc = layer_configs.get(lname)
        if not lc:
            continue
        lw = lc.similarity_weight

        # Weighted average of role similarities within this layer
        role_score = 0.0
        role_weight = 0.0
        for seg_cfg in lc.segments:
            sname = seg_cfg.name
            if sname not in q.layers[lname].segments or sname not in f.layers[lname].segments:
                continue
            for role_cfg in seg_cfg.roles:
                rname = role_cfg.name
                qs = q.layers[lname].segments[sname]
                fs = f.layers[lname].segments[sname]
                if rname in qs.roles and rname in fs.roles:
                    sim = float(cosine_similarity(qs.roles[rname].data, fs.roles[rname].data))
                    rw = role_cfg.similarity_weight
                    role_score += sim * rw
                    role_weight += rw

        if role_weight > 0:
            layer_sim = role_score / role_weight
            total_score += layer_sim * lw
            total_weight += lw

    return total_score / total_weight if total_weight > 0 else 0.0


# ── Intent extraction tests ──────────────────────────────────────────────

class TestIntentExtraction:
    """Verify NL queries extract correct filesystem actions."""

    @pytest.mark.parametrize("query, expected_action", [
        # ls
        ("List all files in the current directory", "ls"),
        ("Show me the list of files", "ls"),
        ("Display all the available files located within the '/temp' directory", "ls"),
        # cd
        ("Go to the workspace directory", "cd"),
        ("Navigate to the documents folder", "cd"),
        ("Pop on over to the Documents directory", "cd"),
        # mkdir
        ("Create a new directory called 'backup'", "mkdir"),
        ("Make sure to create the directory first", "mkdir"),
        # touch
        ("Create a file called 'notes.txt'", "touch"),
        ("Craft a new file dubbed 'summary.txt'", "touch"),
        # cat
        ("Display the contents of 'readme.txt'", "cat"),
        ("Read the content of the file", "cat"),
        ("Output the complete content of the first file", "cat"),
        # grep
        ("Search for the keyword 'Error' in log.txt", "grep"),
        ("Identify sections pertaining to 'budget analysis'", "grep"),
        ("Investigate within 'log.txt' for the occurrence of the keyword 'Error'", "grep"),
        # sort
        ("Sort the file by line", "sort"),
        ("Arrange the contents alphabetically", "sort"),
        # diff
        ("Compare the two files", "diff"),
        ("Juxtapose it with 'previous_report.pdf'", "diff"),
        # mv
        ("Move 'final_report.pdf' to the 'temp' directory", "mv"),
        ("Rename the file to 'backup.txt'", "mv"),
        # cp
        ("Copy the file to the backup directory", "cp"),
        ("Duplicate the document", "cp"),
        # rm
        ("Delete the temporary file", "rm"),
        ("Remove the old log", "rm"),
        # tail
        ("Show the last 20 lines of the file", "tail"),
        ("Finally, show the last 20 lines the file", "tail"),
        # wc
        ("Count the number of lines in the file", "wc"),
        ("How many words are in the document?", "wc"),
        # echo
        ("Write 'hello world' into the file", "echo"),
        ("Jot down some notes in the file", "echo"),
        # find
        ("Find all files with 'test' in their name", "find_files"),
        ("Locate the configuration file", "find_files"),
        ("Gather files that have 'test' in their name", "find_files"),
        # pwd
        ("Check the current working directory", "pwd"),
        ("Where am I right now?", "pwd"),
        # du
        ("Show the disk usage of the directory", "du"),
        ("How much storage is being used?", "du"),
        # diff
        ("Find the differences between file1.txt and file2.txt", "diff"),
    ])
    def test_action_extraction(self, query: str, expected_action: str):
        result = extract_intent(query)
        assert result["action"] == expected_action, (
            f"Query: '{query}'\n"
            f"Expected action: {expected_action}\n"
            f"Got: {result['action']}"
        )

    def test_all_actions_have_functions(self):
        """Every action in the lexicon maps to a real function."""
        for action, func in ACTION_TO_FUNC.items():
            assert func.startswith("GorillaFileSystem."), f"{action} → {func}"

    def test_all_functions_have_actions(self):
        """Every GorillaFileSystem function has a reverse action mapping."""
        expected_funcs = {
            "ls", "cd", "mkdir", "rm", "rmdir", "cp", "mv", "cat",
            "grep", "touch", "wc", "pwd", "find", "tail", "echo",
            "diff", "sort", "du",
        }
        for func in expected_funcs:
            assert func in FUNC_TO_ACTION, f"Missing FUNC_TO_ACTION for {func}"
            assert func in FUNC_TO_TARGET, f"Missing FUNC_TO_TARGET for {func}"


# ── HDC encoding tests ──────────────────────────────────────────────────

class TestEncoding:
    """Verify Glyphs encode correctly and score as expected."""

    @pytest.fixture
    def encoder(self):
        return Encoder(ENCODER_CONFIG)

    @pytest.fixture
    def func_defs(self):
        """Load actual function definitions from func_doc."""
        path = _BFCL_DIR / "data" / "bfcl" / "multi_turn_func_doc" / "gorilla_file_system.json"
        with open(path) as f:
            return [json.loads(line) for line in f if line.strip()]

    @pytest.fixture
    def func_glyphs(self, encoder, func_defs):
        """Encode all function defs into Glyphs."""
        glyphs = {}
        for fd in func_defs:
            cd = encode_function(fd)
            glyph = encoder.encode(Concept(name=cd["name"], attributes=cd["attributes"]))
            glyphs[fd["name"]] = glyph
        return glyphs

    def _score(self, encoder, func_glyphs, query: str) -> list[tuple[str, float]]:
        """Score a query against all function Glyphs using hierarchical scoring.

        Uses 4-level weighted similarity (cortex 5% + layer 10% + segment 25% + role 60%)
        matching the BFCLScoringStrategy. This respects the layer weights defined in
        ENCODER_CONFIG — intent layer vs semantics layer.
        """
        qd = encode_query(query)
        q_glyph = encoder.encode(Concept(name=qd["name"], attributes=qd["attributes"]))

        scores = []
        for fname, fg in func_glyphs.items():
            sim = _layer_weighted_score(q_glyph, fg)
            scores.append((fname, sim))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def test_all_functions_encoded(self, func_glyphs):
        """All 18 functions should be encoded."""
        assert len(func_glyphs) == 18

    @pytest.mark.parametrize("query, expected_func", [
        # Each function must be top-1 for at least one representative query
        ("List all files in the directory", "ls"),
        ("Change to the documents folder", "cd"),
        ("Create a new directory called backup", "mkdir"),
        ("Delete the old file", "rm"),
        ("Remove the empty directory", "rmdir"),
        ("Copy report.txt to backup", "cp"),
        ("Move final_report.pdf to temp", "mv"),
        ("Display the contents of readme.txt", "cat"),
        ("Search for 'error' in the log file", "grep"),
        ("Create a new file called notes.txt", "touch"),
        ("Count the lines in data.csv", "wc"),
        ("What is the current directory?", "pwd"),
        ("Find all files named 'config'", "find"),
        ("Show the last 20 lines of server.log", "tail"),
        ("Write 'hello world' to output.txt", "echo"),
        ("Compare file1.txt and file2.txt", "diff"),
        ("Sort the contents of names.txt", "sort"),
        ("Check disk usage of the home directory", "du"),
    ])
    def test_function_routing(self, encoder, func_glyphs, query: str, expected_func: str):
        """Each function should be the top match for its representative query."""
        scores = self._score(encoder, func_glyphs, query)
        top_func = scores[0][0]
        top_score = scores[0][1]
        second_score = scores[1][1] if len(scores) > 1 else 0.0

        assert top_func == expected_func, (
            f"Query: '{query}'\n"
            f"Expected: {expected_func}\n"
            f"Got: {top_func} (score={top_score:.4f})\n"
            f"Top-3: {[(f, round(s, 4)) for f, s in scores[:3]]}"
        )

    @pytest.mark.parametrize("query, expected_func", [
        # Multi-turn context queries (from actual BFCL entries)
        ("Perform a detailed search using grep to identify sections in the file pertaining to 'budget analysis'", "grep"),
        ("Upon identifying the requisite content, sort the 'final_report.pdf' by line", "sort"),
        ("Show the last 20 lines the file", "tail"),
        ("For clarity, output the complete content of the first file you find on the terminal", "cat"),
        ("Pop on over to the 'Documents' directory and craft a new file dubbed 'summary.txt'", "cd"),
        ("Display all the available files located within the '/temp' directory including hidden ones", "ls"),
        ("Investigate within 'log.txt' for the occurrence of the keyword 'Error'", "grep"),
    ])
    def test_multi_turn_queries(self, encoder, func_glyphs, query: str, expected_func: str):
        """Queries from actual multi-turn entries should route correctly."""
        scores = self._score(encoder, func_glyphs, query)
        top_func = scores[0][0]
        assert top_func == expected_func, (
            f"Query: '{query}'\n"
            f"Expected: {expected_func}\n"
            f"Got: {top_func}\n"
            f"Top-3: {[(f, round(s, 4)) for f, s in scores[:3]]}"
        )

    def test_separation(self, encoder, func_glyphs):
        """Top match should have meaningful separation from second match."""
        test_queries = [
            "List files in directory",
            "Move the file to backup",
            "Search for error in log",
            "Sort the file alphabetically",
            "Compare these two files",
        ]
        for query in test_queries:
            scores = self._score(encoder, func_glyphs, query)
            top = scores[0][1]
            second = scores[1][1]
            gap = top - second
            assert gap > 0.01, (
                f"Query: '{query}' — insufficient separation\n"
                f"Top: {scores[0][0]}={top:.4f}, Second: {scores[1][0]}={second:.4f}, Gap={gap:.4f}"
            )
