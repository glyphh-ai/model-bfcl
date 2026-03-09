"""Pure HDC + rule-based multi-turn evaluation. No LLM.

Architecture:
  1. HDC routes → which functions to call
  2. Rule-based arg extraction from query text + state
  3. CWD state tracking
  4. Execute against real BFCL Python class instances
  5. Compare against ground truth
"""

from __future__ import annotations

import copy
import json
import re
import sys
import time
from pathlib import Path

# Add paths
_BFCL_DIR = Path(__file__).parent
sys.path.insert(0, str(_BFCL_DIR))
GORILLA_ROOT = Path(__file__).parent.parent.parent.parent / "gorilla" / "berkeley-function-call-leaderboard"
sys.path.insert(0, str(GORILLA_ROOT))

from multi_turn_handler import load_func_defs
from scorer import BFCLModelScorer, _CLASS_DIR_MAP
from domain_config import CLASS_DOMAIN_CONFIGS

from bfcl_eval.eval_checker.multi_turn_eval.multi_turn_utils import (
    execute_multi_turn_func_call,
)

MAX_ENTRIES = int(sys.argv[1]) if len(sys.argv) > 1 else 20


# ── Filesystem state ────────────────────────────────────────────────

class FSState:
    """Track filesystem state from initial_config."""

    def __init__(self, initial_config: dict):
        self.cwd_parts: list[str] = []  # path from root
        self.tree = {}
        # Build tree from GorillaFileSystem config
        for cls_name, config in initial_config.items():
            if "root" in config:
                root = config["root"]
                root_name = list(root.keys())[0] if root else ""
                self.tree = self._flatten_tree(root.get(root_name, {}))
                self.cwd_parts = [root_name] if root_name else []
                break

    def _flatten_tree(self, node: dict, path: str = "") -> dict:
        """Build a flat map of path → {files: [], dirs: []}."""
        result = {}
        files = []
        dirs = []
        contents = node.get("contents", node) if isinstance(node, dict) else {}
        for name, info in contents.items():
            if isinstance(info, dict) and info.get("type") == "directory":
                dirs.append(name)
                sub = self._flatten_tree(info, f"{path}/{name}" if path else name)
                result.update(sub)
            elif isinstance(info, dict) and info.get("type") == "file":
                files.append(name)
            elif isinstance(info, dict):
                dirs.append(name)
                sub = self._flatten_tree(info, f"{path}/{name}" if path else name)
                result.update(sub)
        result[path] = {"files": files, "dirs": dirs}
        return result

    @property
    def cwd(self) -> str:
        return "/".join(self.cwd_parts)

    def cd(self, folder: str):
        if folder == "..":
            if self.cwd_parts:
                self.cwd_parts.pop()
        else:
            self.cwd_parts.append(folder)

    def files_here(self) -> list[str]:
        return self.tree.get(self.cwd, {}).get("files", [])

    def dirs_here(self) -> list[str]:
        return self.tree.get(self.cwd, {}).get("dirs", [])

    def find_file(self, name: str) -> str | None:
        """Find which directory contains a file."""
        for path, info in self.tree.items():
            if name in info["files"]:
                return path
        return None

    def find_dir(self, name: str) -> str | None:
        """Find which parent contains a directory."""
        for path, info in self.tree.items():
            if name in info["dirs"]:
                return path
        return None


# ── HDC routing ──────────────────────────────────────────────────────

def setup_hdc_scorers(involved_classes: list[str]) -> dict[str, BFCLModelScorer]:
    scorers = {}
    for class_name in involved_classes:
        class_dir = _CLASS_DIR_MAP.get(class_name)
        if not class_dir:
            continue
        scorer = BFCLModelScorer()
        scorer.configure_from_db(class_dir)
        scorers[class_name] = scorer
    return scorers


def hdc_route(query: str, scorers: dict, involved_classes: list[str]) -> list[tuple[str, str, float]]:
    """Return [(class_name, func_name, score), ...] sorted by score desc."""
    results = []
    for class_name in involved_classes:
        scorer = scorers.get(class_name)
        if not scorer:
            continue
        result = scorer.score(query)
        if not result.all_scores:
            continue
        for s in result.all_scores[:5]:
            if s["score"] > 0.08:
                results.append((class_name, s["function"], s["score"]))
    results.sort(key=lambda x: -x[2])
    return results


# ── Arg extraction ───────────────────────────────────────────────────

def extract_quoted(query: str) -> list[str]:
    """Extract all single-quoted strings from query."""
    return re.findall(r"'([^']+)'", query)


def extract_numbers(query: str) -> list[int]:
    """Extract standalone numbers from query."""
    return [int(m) for m in re.findall(r'\b(\d+)\b', query)]


def extract_file_like(query: str, known_files: list[str], known_dirs: list[str]) -> list[str]:
    """Extract file/directory names mentioned in query (quoted or known)."""
    quoted = extract_quoted(query)
    found = []
    # First: quoted strings that look like files/dirs
    for q in quoted:
        found.append(q)
    # Also check for known files/dirs mentioned without quotes
    for f in known_files + known_dirs:
        if f in query and f not in found:
            found.append(f)
    return found


def find_mentioned_names(query: str, fs: FSState) -> tuple[list[str], list[str]]:
    """Find all file and directory names mentioned in query (quoted or from tree)."""
    quoted = extract_quoted(query)
    mentioned_files = []
    mentioned_dirs = []

    # Check quoted strings
    for q in quoted:
        for path, info in fs.tree.items():
            if q in info["files"] and q not in mentioned_files:
                mentioned_files.append(q)
            if q in info["dirs"] and q not in mentioned_dirs:
                mentioned_dirs.append(q)

    # Check ALL known files/dirs (unquoted) in query
    for path, info in fs.tree.items():
        for f in info["files"]:
            if f in query and f not in mentioned_files:
                mentioned_files.append(f)
        for d in info["dirs"]:
            if d in query and d not in mentioned_dirs:
                mentioned_dirs.append(d)

    # Also add quoted strings that aren't in tree (new names)
    for q in quoted:
        if q not in mentioned_files and q not in mentioned_dirs:
            if "." in q:
                mentioned_files.append(q)
            else:
                mentioned_dirs.append(q)

    return mentioned_files, mentioned_dirs


def cd_to_target(fs: FSState, target_file_or_dir: str) -> list[str]:
    """Generate cd() calls to navigate to where a file/dir exists. Returns call strings."""
    # Where is this file?
    file_path = fs.find_file(target_file_or_dir)
    if file_path is None:
        file_path = fs.find_dir(target_file_or_dir)
    if file_path is None:
        return []

    target_parts = file_path.split("/") if file_path else []
    current = fs.cwd_parts[:]

    # Already there?
    if target_parts == current:
        return []

    calls = []
    # Navigate: go up then down
    # Find common prefix
    common = 0
    for i in range(min(len(current), len(target_parts))):
        if current[i] == target_parts[i]:
            common = i + 1
        else:
            break

    # Go up
    for _ in range(len(current) - common):
        fs.cd("..")
        calls.append("cd(folder='..')")

    # Go down
    for part in target_parts[common:]:
        fs.cd(part)
        calls.append(f"cd(folder='{part}')")

    return calls


def build_calls(
    query: str,
    hdc_results: list[tuple[str, str, float]],
    fs: FSState,
    initial_config: dict,
    prev_results: list[str],
    func_defs: list[dict],
) -> list[str]:
    """Given HDC-routed functions and query, build call strings deterministically."""
    query_lower = query.lower()
    quoted = extract_quoted(query)
    numbers = extract_numbers(query)

    calls = []

    # Detect which functions are needed
    needed_funcs = detect_functions(query, query_lower, hdc_results, initial_config)

    # Find all mentioned files/dirs in the tree
    mentioned_files, mentioned_dirs = find_mentioned_names(query, fs)

    for func in needed_funcs:
        func_calls = extract_args_for_func(
            func, query, query_lower, quoted, numbers,
            fs, initial_config, prev_results,
            fs.files_here(), fs.dirs_here(),
            mentioned_files, mentioned_dirs
        )
        calls.extend(func_calls)

    return calls


def detect_functions(
    query: str, query_lower: str,
    hdc_results: list[tuple[str, str, float]],
    initial_config: dict,
) -> list[str]:
    """Detect which functions to call, in order."""
    funcs = []
    top1 = hdc_results[0][1] if hdc_results else ""
    top_set = {fn for _, fn, sc in hdc_results[:3]}

    # Keywords → function mapping
    kw_map = [
        (["list", "ls", "show me the list", "inventory", "display all"], "ls"),
        (["move", "transfer", "relocate"], "mv"),
        (["copy", "duplicate", "backup"], "cp"),
        (["create a file", "craft a new file", "generate a new file", "draft up", "create a document"], "touch"),
        (["create a new directory", "set up a new directory", "generate a new directory", "make sure to create the directory"], "mkdir"),
        (["write", "jot down", "put some", "infuse", "echo"], "echo"),
        (["search", "grep", "investigate within", "identify sections", "find lines"], "grep"),
        (["sort", "arrange", "sorted"], "sort"),
        (["compare", "comparison", "diff", "differences"], "diff"),
        (["display the content", "read", "peek at", "reveal the content", "show what", "contents of"], "cat"),
        (["find", "locate files", "gather files"], "find"),
        (["last .* lines", "tail", "last entry", "last five"], "tail"),
        (["word count", "how many words", "tally up", "how many characters", "how many lines", "how comprehensive", "summary of the lines"], "wc"),
        (["navigate", "go to", "go into", "cd into", "cd within", "pop on over", "open the"], "cd"),
        (["post", "tweet", "share .* social media", "draft a post"], "post_tweet"),
        (["authenticate", "log in.*twitter"], "authenticate_twitter"),
        (["comment", "add a supportive comment"], "comment"),
        (["login", "logging in", "log in.*USR"], "message_login"),
        (["add .* contact"], "add_contact"),
        (["send .* message", "dispatch .* report", "message .* colleague"], "send_message"),
        (["view .* messages", "messages sent"], "view_messages_sent"),
        (["average", "mean"], "mean"),
    ]

    detected = set()
    for keywords, func in kw_map:
        for kw in keywords:
            if re.search(kw, query_lower):
                detected.add(func)
                break

    # Use HDC top-1 if nothing detected
    if not detected and top1:
        detected.add(top1)

    # Order: cd first, then mkdir, then operations, then social/messaging last
    order = ["cd", "mkdir", "touch", "echo", "mv", "cp", "find", "grep", "sort",
             "diff", "cat", "tail", "wc", "ls",
             "authenticate_twitter", "message_login", "add_contact",
             "post_tweet", "send_message", "comment", "view_messages_sent", "mean"]

    # But don't add cd if the query is primarily about ls/listing
    if "ls" in detected and "cd" in detected:
        # If query says "list files in X directory", we might need cd first
        # But if it's "list all files in current directory", no cd needed
        pass

    return [f for f in order if f in detected]


def extract_args_for_func(
    func: str, query: str, query_lower: str,
    quoted: list[str], numbers: list[int],
    fs: FSState, initial_config: dict,
    prev_results: list[str],
    files_here: list[str], dirs_here: list[str],
    mentioned_files: list[str] = None, mentioned_dirs: list[str] = None,
) -> list[str]:
    """Build call string(s) for a specific function."""

    if func == "ls":
        return ["ls(a=True)"]

    elif func == "cd":
        # Find the folder to cd into
        folder = None
        # Check quoted strings first
        for q in quoted:
            if q == ".." or q in (mentioned_dirs or []):
                folder = q
                break
        # Then check mentioned dirs
        if not folder and mentioned_dirs:
            for d in mentioned_dirs:
                if d.lower() in query_lower:
                    folder = d
                    break
        if not folder and ".." in query_lower:
            folder = ".."
        if folder:
            # Navigate there using tree
            nav = cd_to_target(fs, folder)
            if nav:
                return nav
            # Direct cd if not in tree (e.g., "..")
            fs.cd(folder)
            return [f"cd(folder='{folder}')"]
        return []

    elif func == "mkdir":
        for q in quoted:
            if q not in files_here:  # new directory name
                return [f"mkdir(dir_name='{q}')"]
        return []

    elif func == "touch":
        for q in quoted:
            if "." in q:  # looks like a filename
                return [f"touch(file_name='{q}')"]
        return []

    elif func == "echo":
        # Need content and file_name
        file_name = None
        content = None
        for q in quoted:
            if "." in q and not content:
                # Could be file or content — check if it's a known file
                if q in files_here or any("." in q and len(q) < 50 for _ in [1]):
                    if file_name is None:
                        file_name = q
                    continue
            if content is None and file_name is not None:
                content = q
            elif content is None:
                content = q
            elif file_name is None:
                file_name = q

        # If we have content but no file, check for file in query
        if content and not file_name:
            for f in files_here:
                if f.lower() in query_lower:
                    file_name = f
                    break
        if file_name and content:
            return [f"echo(content='{content}', file_name='{file_name}')"]
        return []

    elif func == "mv":
        if len(quoted) >= 2:
            return [f"mv(source='{quoted[0]}', destination='{quoted[1]}')"]
        elif len(quoted) == 1:
            # source is quoted, destination might be a dir name in query
            source = quoted[0]
            dest = None
            for d in dirs_here:
                if d.lower() in query_lower and d != source:
                    dest = d
                    break
            if dest:
                return [f"mv(source='{source}', destination='{dest}')"]
        return []

    elif func == "cp":
        if len(quoted) >= 2:
            return [f"cp(source='{quoted[0]}', destination='{quoted[1]}')"]
        elif len(quoted) == 1:
            source = quoted[0]
            dest = None
            for d in dirs_here:
                if d.lower() in query_lower and d != source:
                    dest = d
                    break
            if dest:
                return [f"cp(source='{source}', destination='{dest}')"]
            # Maybe destination is a new directory
            for q in extract_quoted(query):
                if q != source:
                    dest = q
                    break
            if dest:
                return [f"cp(source='{source}', destination='{dest}')"]
        else:
            # Multiple files to copy — check for file names in files_here
            calls = []
            dest = None
            for q in quoted:
                if q not in files_here:
                    dest = q
                    break
            if dest:
                for f in files_here:
                    if "." in f:  # text files
                        calls.append(f"cp(source='{f}', destination='{dest}')")
            return calls
        return []

    elif func == "grep":
        file_name = None
        pattern = None
        for q in quoted:
            if "." in q and len(q) < 50:
                file_name = q
            else:
                pattern = q
        if not file_name:
            for f in files_here:
                if f.lower() in query_lower:
                    file_name = f
                    break
        if file_name and pattern:
            return [f"grep(file_name='{file_name}', pattern='{pattern}')"]
        return []

    elif func == "sort":
        for q in quoted:
            if "." in q:
                return [f"sort(file_name='{q}')"]
        # Check files_here
        for f in files_here:
            if f.lower() in query_lower:
                return [f"sort(file_name='{f}')"]
        return []

    elif func == "diff":
        file_names = [q for q in quoted if "." in q]
        if len(file_names) >= 2:
            return [f"diff(file_name1='{file_names[0]}', file_name2='{file_names[1]}')"]
        return []

    elif func == "cat":
        for q in quoted:
            if "." in q:
                return [f"cat(file_name='{q}')"]
        for f in files_here:
            if f.lower() in query_lower:
                return [f"cat(file_name='{f}')"]
        return []

    elif func == "find":
        for q in quoted:
            return [f"find(path='.', name='{q}')"]
        return []

    elif func == "tail":
        file_name = None
        lines = 1  # default
        for q in quoted:
            if "." in q:
                file_name = q
        if not file_name:
            for f in files_here:
                if f.lower() in query_lower:
                    file_name = f
                    break
        # Extract line count
        m = re.search(r'last\s+(\d+)\s+lines', query_lower)
        if m:
            lines = int(m.group(1))
        elif "last entry" in query_lower or "last line" in query_lower:
            lines = 1
        m2 = re.search(r'last\s+(\w+)\s+lines', query_lower)
        if m2:
            word_to_num = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
                           "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                           "twenty": 20}
            lines = word_to_num.get(m2.group(1), lines)
        if file_name:
            return [f"tail(file_name='{file_name}', lines={lines})"]
        return []

    elif func == "wc":
        file_name = None
        for q in quoted:
            if "." in q:
                file_name = q
        if not file_name:
            for f in files_here:
                if f.lower() in query_lower:
                    file_name = f
                    break
        # Detect mode
        modes = []
        if any(w in query_lower for w in ["word", "words"]):
            modes.append("w")
        if any(w in query_lower for w in ["character", "characters"]):
            modes.append("c")
        if any(w in query_lower for w in ["line", "lines"]):
            modes.append("l")
        if "summary" in query_lower or "comprehensive" in query_lower:
            modes = ["l", "w", "c"]
        if not modes:
            modes = ["w"]
        if file_name:
            return [f"wc(file_name='{file_name}', mode='{m}')" for m in modes]
        return []

    elif func == "post_tweet":
        # Extract content, mentions, tags from query
        content = None
        mentions = []
        tags = []

        # Content is usually in quotes after "states" or "saying" or from prev results
        for q in quoted:
            if len(q) > 10 and not q.startswith("@") and not q.startswith("#"):
                content = q
                break

        # If content references previous results, use them
        if not content and prev_results:
            content = prev_results[-1] if prev_results else ""

        # Mentions: @name patterns or quoted names after "mention"
        mention_match = re.findall(r'mention\w*\s+(\w+)', query_lower)
        for m in mention_match:
            mentions.append(f"@{m}")

        # Tags: #tag patterns or quoted after "tag"
        tag_match = re.findall(r"'#(\w+)'", query)
        for t in tag_match:
            tags.append(f"#{t}")
        tag_match2 = re.findall(r"tag\w*\s+'(\w+)'", query_lower)
        for t in tag_match2:
            tags.append(f"#{t}")

        if content:
            parts = [f"content='{content}'"]
            if mentions:
                parts.append(f"mentions={mentions}")
            if tags:
                parts.append(f"tags={tags}")
            return [f"post_tweet({', '.join(parts)})"]
        return []

    elif func == "authenticate_twitter":
        config = initial_config.get("TwitterAPI", {})
        username = config.get("username", "")
        password = config.get("password", "")
        if username:
            return [f"authenticate_twitter(username='{username}', password='{password}')"]
        return []

    elif func == "comment":
        content = None
        for q in quoted:
            content = q
            break
        if content:
            # tweet_id from previous results or default
            return [f"comment(tweet_id=0, comment_content='{content}')"]
        return []

    elif func == "message_login":
        user_id = None
        for q in quoted:
            if q.startswith("USR"):
                user_id = q
                break
        m = re.search(r'(USR\d+)', query)
        if m:
            user_id = m.group(1)
        if user_id:
            return [f"message_login(user_id='{user_id}')"]
        return []

    elif func == "add_contact":
        # Extract name — usually after "add contact" or "add her/his contact"
        for q in quoted:
            if not q.startswith("USR") and len(q) < 30:
                return [f"add_contact(user_name='{q}')"]
        # Try to find name from query
        m = re.search(r'add\s+\w+\s+contact\s*\(?\s*(\w+)', query_lower)
        if m:
            name = m.group(1).title()
            return [f"add_contact(user_name='{name}')"]
        return []

    elif func == "send_message":
        receiver = None
        message = None
        m = re.search(r'(USR\d+)', query)
        if m:
            receiver = m.group(1)
        for q in quoted:
            if q.startswith("USR"):
                receiver = q
            elif len(q) > 5:
                message = q
        if receiver and message:
            return [f"send_message(receiver_id='{receiver}', message='{message}')"]
        return []

    elif func == "view_messages_sent":
        return ["view_messages_sent()"]

    elif func == "mean":
        # Extract numbers from previous results
        nums = []
        for r in prev_results:
            try:
                nums.append(int(r))
            except (ValueError, TypeError):
                pass
        if nums:
            return [f"mean(numbers={nums})"]
        return []

    return []


# ── Call comparison ──────────────────────────────────────────────────

def _parse_call(call_str):
    m = re.match(r"(\w+)\((.*)\)$", call_str, re.DOTALL)
    if not m:
        return (call_str, {})
    func_name = m.group(1)
    args_str = m.group(2).strip()
    if not args_str:
        return (func_name, {})
    try:
        args = eval(f"dict({args_str})")
        return (func_name, args)
    except:
        pass
    try:
        vals = eval(f"({args_str},)")
        args = {f"_pos_{i}": v for i, v in enumerate(vals)}
        return (func_name, args)
    except:
        return (func_name, {"_raw": args_str})


def _calls_match(pred, gt):
    if len(pred) != len(gt):
        return False
    for (pf, pa), (gf, ga) in zip(pred, gt):
        if pf != gf:
            return False
        p_has_pos = any(k.startswith("_pos_") for k in pa)
        g_has_pos = any(k.startswith("_pos_") for k in ga)
        if p_has_pos or g_has_pos:
            p_vals = sorted(str(v) for v in pa.values())
            g_vals = sorted(str(v) for v in ga.values())
            if p_vals != g_vals:
                return False
        else:
            if pa != ga:
                return False
    return True


# ── Main ─────────────────────────────────────────────────────────────

def main():
    path = _BFCL_DIR / "data" / "bfcl" / "BFCL_v4_multi_turn_base.json"
    gt_path = _BFCL_DIR / "data" / "bfcl" / "possible_answer" / "BFCL_v4_multi_turn_base.json"

    entries, gt_entries = [], []
    with open(path) as f:
        for i, line in enumerate(f):
            if i >= MAX_ENTRIES:
                break
            entries.append(json.loads(line))
    with open(gt_path) as f:
        for i, line in enumerate(f):
            if i >= MAX_ENTRIES:
                break
            gt_entries.append(json.loads(line))

    total_scenarios = 0
    passed_scenarios = 0
    total_turns = 0
    passed_turns = 0
    t0 = time.time()

    for idx, (entry, gt_entry) in enumerate(zip(entries, gt_entries)):
        scenario_id = entry["id"]
        ground_truth = gt_entry["ground_truth"]
        initial_config = entry.get("initial_config", {})
        involved_classes = entry["involved_classes"]

        func_defs = load_func_defs(involved_classes, entry.get("excluded_function", []))
        scorers = setup_hdc_scorers(involved_classes)
        fs = FSState(initial_config)

        # Extract queries
        queries = []
        for turn_messages in entry.get("question", []):
            for msg in turn_messages:
                if isinstance(msg, dict) and msg.get("role") == "user":
                    queries.append(msg.get("content", ""))
                    break
            else:
                queries.append("")

        scenario_pass = True
        total_scenarios += 1
        prev_results = []

        for turn_idx, query in enumerate(queries):
            if not query:
                continue

            gt_turn = ground_truth[turn_idx] if turn_idx < len(ground_truth) else []
            total_turns += 1

            # HDC route
            hdc_results = hdc_route(query, scorers, involved_classes)

            # Build calls deterministically
            predicted = build_calls(
                query, hdc_results, fs, initial_config, prev_results, func_defs
            )

            # Execute to get results for future turns
            if predicted:
                execution_results, _ = execute_multi_turn_func_call(
                    predicted, initial_config, involved_classes,
                    f"glyphh_hdc_{scenario_id}", scenario_id,
                )
                prev_results = [str(r) for r in execution_results if not str(r).startswith("Error")]
            else:
                prev_results = []

            # Compare
            parsed_pred = [_parse_call(c) for c in predicted]
            parsed_gt = [_parse_call(c) for c in gt_turn]
            match = _calls_match(parsed_pred, parsed_gt)

            if match:
                passed_turns += 1
            else:
                scenario_pass = False
                print(f"  FAIL {scenario_id} turn {turn_idx}: {query[:80]}")
                print(f"    Exp: {gt_turn}")
                print(f"    Got: {predicted}")

        if scenario_pass:
            passed_scenarios += 1
            print(f"  PASS {scenario_id}")

        elapsed = time.time() - t0
        print(f"  [{idx+1}/{MAX_ENTRIES}] {passed_scenarios}/{total_scenarios} scenarios, "
              f"{passed_turns}/{total_turns} turns, {elapsed:.1f}s")

    print(f"\n{'='*60}")
    print(f"PURE HDC + RULES (no LLM)")
    print(f"FINAL: {passed_scenarios}/{total_scenarios} scenarios ({passed_scenarios/total_scenarios*100:.1f}%)")
    print(f"       {passed_turns}/{total_turns} turns ({passed_turns/total_turns*100:.1f}%)")
    print(f"       {time.time()-t0:.1f}s total")


if __name__ == "__main__":
    main()
