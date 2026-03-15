#!/usr/bin/env python3
"""Test SlotExtractor against BFCL multi-turn ground truth for GorillaFileSystem.

Evaluates whether the data-driven slot filling (zero LLM) can extract
correct argument values from natural language queries.

Three key improvements over v1:
  - Gap 1: Claim-aware extraction — multiple calls per turn don't fight over quotes
  - Gap 2: Cross-turn state replay — later turns see CWD/fs changes from earlier turns
  - Gap 3: Type-aware disambiguation — quoted_filtered prevents cross-type assignment

Usage:
    cd glyphh-models/bfcl
    PYTHONPATH=../../glyphh-runtime python test_slot_extraction.py
"""

from __future__ import annotations

import copy
import json
import re
from collections import defaultdict
from pathlib import Path

from glyphh.cognitive.slots import SlotExtractor
from domain_config import FS_DOMAIN_CONFIG

_BFCL_DIR = Path(__file__).parent
_DATA_DIR = _BFCL_DIR / "data" / "bfcl"


# ── Filesystem state simulator ──────────────────────────────────────

class FSState:
    """Simulate GorillaFileSystem state for cross-turn tracking.

    Maintains a filesystem tree and CWD. After each ground-truth call,
    apply_call() mutates the tree so the next turn's extraction sees
    accurate items_here / locations_here collections.
    """

    def __init__(self, root: dict, cwd_path: list[str] | None = None):
        self._root = copy.deepcopy(root)
        self._cwd_path = list(cwd_path) if cwd_path else []

    def get_state(self) -> dict:
        """Return CognitiveLoop-compatible state dict."""
        node = self._get_node(self._cwd_path)
        items = []
        locations = []
        contents = node.get("contents", {}) if isinstance(node, dict) else {}
        for name, child in contents.items():
            if isinstance(child, dict) and child.get("type") == "directory":
                locations.append(name)
            else:
                items.append(name)
        return {
            "primary": "/".join(self._cwd_path),
            "collections": {
                "items_here": items,
                "locations_here": locations,
            },
        }

    def apply_call(self, func_name: str, args: dict):
        """Apply a ground-truth call to mutate the filesystem state."""
        bare = func_name.split(".")[-1] if "." in func_name else func_name

        if bare == "cd":
            folder = args.get("folder", "")
            if folder == "..":
                if self._cwd_path:
                    self._cwd_path.pop()
            elif folder:
                self._cwd_path.append(folder)

        elif bare == "mkdir":
            dir_name = args.get("dir_name", "")
            if dir_name:
                node = self._get_node(self._cwd_path)
                contents = node.setdefault("contents", {})
                contents[dir_name] = {"type": "directory", "contents": {}}

        elif bare == "touch":
            file_name = args.get("file_name", "")
            if file_name:
                node = self._get_node(self._cwd_path)
                contents = node.setdefault("contents", {})
                contents[file_name] = {"type": "file", "content": ""}

        elif bare == "echo":
            file_name = args.get("file_name", "")
            if file_name:
                node = self._get_node(self._cwd_path)
                contents = node.setdefault("contents", {})
                contents[file_name] = {"type": "file", "content": args.get("content", "")}

        elif bare == "rm":
            file_name = args.get("file_name", "")
            if file_name:
                node = self._get_node(self._cwd_path)
                node.get("contents", {}).pop(file_name, None)

        elif bare == "rmdir":
            dir_name = args.get("dir_name", "")
            if dir_name:
                node = self._get_node(self._cwd_path)
                node.get("contents", {}).pop(dir_name, None)

        elif bare == "mv":
            source = args.get("source", "")
            destination = args.get("destination", "")
            if source and destination:
                node = self._get_node(self._cwd_path)
                contents = node.get("contents", {})
                if source in contents:
                    item = contents.pop(source)
                    # Check if destination is an existing directory
                    if destination in contents and isinstance(contents[destination], dict) \
                       and contents[destination].get("type") == "directory":
                        contents[destination].setdefault("contents", {})[source] = item
                    else:
                        contents[destination] = item

        elif bare == "cp":
            source = args.get("source", "")
            destination = args.get("destination", "")
            if source and destination:
                node = self._get_node(self._cwd_path)
                contents = node.get("contents", {})
                if source in contents:
                    contents[destination] = copy.deepcopy(contents[source])

    def _get_node(self, path: list[str]) -> dict:
        """Navigate to a node in the tree by path components."""
        node = self._root
        for comp in path:
            contents = node.get("contents", {}) if isinstance(node, dict) else {}
            node = contents.get(comp, {})
        return node if isinstance(node, dict) else {}


# ── Data loading ────────────────────────────────────────────────────

# Known GorillaFileSystem function names
_FS_BARE = {"cd", "ls", "mkdir", "rm", "cp", "mv", "cat", "grep", "touch",
            "wc", "pwd", "find", "tail", "head", "echo", "diff", "sort",
            "du", "rmdir"}
_FS_FUNCS = {f"GorillaFileSystem.{f}" for f in _FS_BARE}


def _parse_call(call_str: str) -> tuple[str, dict] | None:
    """Parse 'func(arg1='val1', arg2=val2)' into (func_name, {arg: value})."""
    m = re.match(r"(\w[\w.]*)\((.*)\)$", call_str, re.DOTALL)
    if not m:
        return None
    func_name = m.group(1)
    args_str = m.group(2).strip()
    if not args_str:
        return (func_name, {})
    try:
        args = eval(f"dict({args_str})")
        return (func_name, args)
    except Exception:
        pass
    # Positional args
    try:
        vals = eval(f"({args_str},)")
        args = {f"_pos_{i}": v for i, v in enumerate(vals)}
        return (func_name, args)
    except Exception:
        return (func_name, {"_raw": args_str})


def _normalize_func_name(name: str) -> str:
    """Ensure function name has GorillaFileSystem. prefix."""
    if "." in name:
        return name
    return f"GorillaFileSystem.{name}"


def _load_scenarios() -> dict[str, dict]:
    """Load all multi-turn scenario entries, keyed by ID."""
    scenarios = {}
    for cat in ["multi_turn_base", "multi_turn_miss_func",
                "multi_turn_miss_param", "multi_turn_long_context"]:
        q_path = _DATA_DIR / f"BFCL_v4_{cat}.json"
        a_path = _DATA_DIR / "possible_answer" / f"BFCL_v4_{cat}.json"
        if not q_path.exists() or not a_path.exists():
            continue

        with open(q_path) as f:
            entries = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}
        with open(a_path) as f:
            answers = {json.loads(line)["id"]: json.loads(line) for line in f if line.strip()}

        for sid, entry in entries.items():
            gt = answers.get(sid, {}).get("ground_truth", [])
            involved = entry.get("involved_classes", [])
            if "GorillaFileSystem" not in involved:
                continue
            entry["_ground_truth"] = gt
            scenarios[sid] = entry

    return scenarios


def _build_fs_root(fs_config: dict) -> dict:
    """Build the FSState root node from BFCL initial_config.GorillaFileSystem."""
    root_data = fs_config.get("root", {})
    if not root_data:
        return {"type": "directory", "contents": {}}

    # The root typically has one top-level dir (the user's home)
    # We wrap it so FSState can navigate into it
    return {"type": "directory", "contents": root_data}


def _initial_cwd(fs_config: dict) -> list[str]:
    """Determine the initial CWD path from BFCL config.

    BFCL scenarios start CWD at the user's home directory (first top-level dir).
    """
    root = fs_config.get("root", {})
    for name in root:
        return [name]
    return []


def _load_func_schemas() -> dict[str, dict]:
    """Load function schemas from multi_turn_func_doc."""
    doc_path = _DATA_DIR / "multi_turn_func_doc" / "gorilla_file_system.json"
    func_schemas = {}
    if doc_path.exists():
        with open(doc_path) as f:
            for line in f:
                if not line.strip():
                    continue
                fd = json.loads(line)
                name = fd.get("name", "")
                func_schemas[f"GorillaFileSystem.{name}"] = fd
    return func_schemas


# ── Main evaluation ─────────────────────────────────────────────────

def main():
    scenarios = _load_scenarios()
    func_schemas = _load_func_schemas()
    extractor = SlotExtractor(FS_DOMAIN_CONFIG)

    print(f"Loaded {len(scenarios)} GorillaFileSystem scenarios")
    print(f"Loaded {len(func_schemas)} function schemas\n")

    # Stats
    total = 0
    correct = 0
    partial = 0
    wrong = 0
    skipped = 0
    errors_by_param: dict[str, int] = {}
    errors_by_func: dict[str, int] = {}
    sample_failures: list[dict] = []

    for sid, entry in scenarios.items():
        gt_turns = entry.get("_ground_truth", [])
        questions = entry.get("question", [])
        initial_config = entry.get("initial_config", {})
        fs_config = initial_config.get("GorillaFileSystem", {})

        # Build FSState from initial_config
        fs_root = _build_fs_root(fs_config)
        cwd = _initial_cwd(fs_config)
        fs_state = FSState(fs_root, cwd)

        for turn_idx, (turn_msgs, turn_gt) in enumerate(zip(questions, gt_turns)):
            # Extract user query
            query = ""
            for msg in turn_msgs:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    query = msg.get("content", "")
                    break
            if not query:
                continue

            # Parse all ground truth calls for this turn
            gt_calls = []
            for call_str in turn_gt:
                parsed = _parse_call(call_str)
                if parsed is None:
                    continue
                fname, args = parsed
                fname = _normalize_func_name(fname)
                if fname not in _FS_FUNCS:
                    # Check bare name
                    bare = fname.split(".")[-1]
                    if bare in _FS_BARE:
                        fname = f"GorillaFileSystem.{bare}"
                    else:
                        continue
                gt_calls.append((fname, args))

            if not gt_calls:
                continue

            # Get current state from FSState
            state = fs_state.get_state()

            # Extract all functions for this turn
            func_names = [fname for fname, _ in gt_calls]
            all_schemas = {fname: func_schemas.get(fname, {}) for fname in func_names}

            filled = extractor.extract(
                query, func_names, all_schemas, state,
            )

            # Evaluate each call
            for call_idx, (fname, expected) in enumerate(gt_calls):
                if not expected:
                    skipped += 1
                    continue

                # Skip positional-only args
                if any(k.startswith("_") for k in expected):
                    skipped += 1
                    continue

                schema = func_schemas.get(fname, {})
                if not schema:
                    skipped += 1
                    continue

                extracted = filled.get(fname, {})
                total += 1

                # Compare
                all_match = True
                any_match = False
                for pname, expected_val in expected.items():
                    got = extracted.get(pname)
                    if got == expected_val:
                        any_match = True
                    else:
                        # Also accept string/int coercion matches
                        if str(got) == str(expected_val):
                            any_match = True
                        else:
                            all_match = False
                            errors_by_param[pname] = errors_by_param.get(pname, 0) + 1
                            bare = fname.split(".")[-1]
                            errors_by_func[bare] = errors_by_func.get(bare, 0) + 1

                if all_match:
                    correct += 1
                elif any_match:
                    partial += 1
                else:
                    wrong += 1

                if not all_match and len(sample_failures) < 30:
                    sample_failures.append({
                        "query": query[:120],
                        "func": fname.split(".")[-1],
                        "expected": expected,
                        "got": extracted,
                        "scenario": sid,
                        "turn": turn_idx,
                        "state_items": state["collections"].get("items_here", [])[:5],
                        "state_locs": state["collections"].get("locations_here", [])[:5],
                    })

            # Apply ground truth calls to FSState for next turn's state
            for fname, args in gt_calls:
                fs_state.apply_call(fname, args)

    # ── Report ──
    print(f"Results (with cross-turn state + claim-aware + type-filtered):")
    print(f"  Total:   {total}")
    print(f"  Correct: {correct} ({100*correct/total:.1f}%)" if total else "  Correct: 0")
    print(f"  Partial: {partial} ({100*partial/total:.1f}%)" if total else "  Partial: 0")
    print(f"  Wrong:   {wrong} ({100*wrong/total:.1f}%)" if total else "  Wrong: 0")
    print(f"  Skipped: {skipped}")

    print(f"\nErrors by parameter:")
    for pname, count in sorted(errors_by_param.items(), key=lambda x: -x[1]):
        print(f"  {pname}: {count}")

    print(f"\nErrors by function:")
    for fname, count in sorted(errors_by_func.items(), key=lambda x: -x[1]):
        print(f"  {fname}: {count}")

    print(f"\nSample failures (first 15):")
    for f in sample_failures[:15]:
        print(f"  [{f['scenario']} t{f['turn']}] {f['func']}")
        print(f"    Query: {f['query']}")
        print(f"    Expected: {f['expected']}")
        print(f"    Got:      {f['got']}")
        print(f"    State: items={f['state_items']}, locs={f['state_locs']}")
        print()


if __name__ == "__main__":
    main()
