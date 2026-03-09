"""Per-class intent extraction for GorillaFileSystem.

Uses the SDK filesystem pack for NL → action/target extraction.
Maps pack canonical actions to the 18 GorillaFileSystem functions.

Exports:
    extract_intent(query) → {action, target, keywords}
    ACTION_TO_FUNC        — canonical action → GorillaFileSystem.{func}
    FUNC_TO_ACTION        — bare func name → canonical action
"""

from __future__ import annotations

import re

# ── Pack canonical action → GorillaFileSystem function ────────────────────

ACTION_TO_FUNC: dict[str, str] = {
    "ls":         "GorillaFileSystem.ls",
    "cd":         "GorillaFileSystem.cd",
    "mkdir":      "GorillaFileSystem.mkdir",
    "rm":         "GorillaFileSystem.rm",
    "rmdir":      "GorillaFileSystem.rmdir",
    "cp":         "GorillaFileSystem.cp",
    "mv":         "GorillaFileSystem.mv",
    "cat":        "GorillaFileSystem.cat",
    "grep":       "GorillaFileSystem.grep",
    "touch":      "GorillaFileSystem.touch",
    "wc":         "GorillaFileSystem.wc",
    "pwd":        "GorillaFileSystem.pwd",
    "find_files": "GorillaFileSystem.find",
    "tail":       "GorillaFileSystem.tail",
    "head":       "GorillaFileSystem.head",
    "echo":       "GorillaFileSystem.echo",
    "diff":       "GorillaFileSystem.diff",
    "sort":       "GorillaFileSystem.sort",
    "du":         "GorillaFileSystem.du",
}

# Reverse: bare function name → pack canonical action
FUNC_TO_ACTION: dict[str, str] = {
    "ls":    "ls",
    "cd":    "cd",
    "mkdir": "mkdir",
    "rm":    "rm",
    "rmdir": "rmdir",
    "cp":    "cp",
    "mv":    "mv",
    "cat":   "cat",
    "grep":  "grep",
    "touch": "touch",
    "wc":    "wc",
    "pwd":   "pwd",
    "find":  "find_files",
    "tail":  "tail",
    "head":  "head",
    "echo":  "echo",
    "diff":  "diff",
    "sort":  "sort",
    "du":    "du",
}

# Function → canonical target
FUNC_TO_TARGET: dict[str, str] = {
    "ls":    "directory",
    "cd":    "directory",
    "mkdir": "directory",
    "rm":    "file",
    "rmdir": "directory",
    "cp":    "file",
    "mv":    "file",
    "cat":   "file",
    "grep":  "file",
    "touch": "file",
    "wc":    "file",
    "pwd":   "directory",
    "find":  "file",
    "tail":  "file",
    "head":  "file",
    "echo":  "file",
    "diff":  "file",
    "sort":  "file",
    "du":    "directory",
}

# ── NL synonym → pack canonical action ──────────────────────────────────
# Derived from filesystem pack actions[].synonyms + phrases

_PHRASE_MAP: list[tuple[str, str]] = [
    # Longest/most specific phrases first for greedy matching
    # ls — must come before "current directory" (pwd) to avoid false match
    ("list of files", "ls"),
    ("list all files", "ls"),
    ("list all the", "ls"),
    ("list the files", "ls"),
    ("available files", "ls"),
    ("files located within", "ls"),
    ("show me the list", "ls"),
    ("inventory of", "ls"),
    # mkdir — must come before single-word "create" (touch)
    ("new directory", "mkdir"),
    ("create directory", "mkdir"),
    ("make directory", "mkdir"),
    ("new folder", "mkdir"),
    ("create folder", "mkdir"),
    ("make a new folder", "mkdir"),
    ("make sure to create", "mkdir"),
    ("create the directory", "mkdir"),
    ("set up a new directory", "mkdir"),
    ("generate a new directory", "mkdir"),
    # rmdir
    ("delete directory", "rmdir"),
    ("remove directory", "rmdir"),
    ("remove the empty directory", "rmdir"),
    ("remove empty directory", "rmdir"),
    ("delete folder", "rmdir"),
    ("remove folder", "rmdir"),
    # cd — navigation phrases before touch so "go to X and create file" → cd
    ("change directory", "cd"),
    ("navigate to", "cd"),
    ("go to", "cd"),
    ("go into", "cd"),
    ("pop on over", "cd"),
    ("head over to", "cd"),
    ("switch to", "cd"),
    # touch — after mkdir and cd phrases
    ("create file", "touch"),
    ("new file", "touch"),
    ("craft a new file", "touch"),
    ("create a document", "touch"),
    # du
    ("disk usage", "du"),
    ("storage usage", "du"),
    ("directory size", "du"),
    ("how much space", "du"),
    ("how much storage", "du"),
    # wc
    ("word count", "wc"),
    ("line count", "wc"),
    ("count lines", "wc"),
    ("count words", "wc"),
    ("how many lines", "wc"),
    ("how many words", "wc"),
    ("how many characters", "wc"),
    # find
    ("search for file", "find_files"),
    ("search for a file", "find_files"),
    ("locate file", "find_files"),
    ("find file", "find_files"),
    ("find files", "find_files"),
    ("gather files", "find_files"),
    ("working directory", "pwd"),
    ("current directory", "pwd"),
    ("where am i", "pwd"),
    ("current path", "pwd"),
    ("last lines", "tail"),
    ("last line", "tail"),
    ("last 20 lines", "tail"),
    ("last 10 lines", "tail"),
    ("show the last", "tail"),
    ("end of file", "tail"),
    ("move the file", "mv"),
    ("rename the file", "mv"),
    ("copy the file", "cp"),
    ("duplicate the file", "cp"),
    ("make a copy", "cp"),
    ("delete the file", "rm"),
    ("remove the file", "rm"),
    ("compare the", "diff"),
    ("differences between", "diff"),
    ("juxtapose", "diff"),
    # head — must come before cat phrases ("show" falls through to cat otherwise)
    ("first lines", "head"),
    ("first line", "head"),
    ("first 10 lines", "head"),
    ("first 5 lines", "head"),
    ("first 20 lines", "head"),
    ("beginning of", "head"),
    ("top of the file", "head"),
    ("top lines", "head"),
    ("head of the file", "head"),
    ("read the content", "cat"),
    ("display contents", "cat"),
    ("show contents", "cat"),
    ("view its contents", "cat"),
    ("output the complete content", "cat"),
    ("take a peek", "cat"),
    ("peek at what", "cat"),
    ("complete content of", "cat"),
    ("search in file", "grep"),
    ("search for text", "grep"),
    ("search the file", "grep"),
    ("identify sections", "grep"),
    ("occurrence of", "grep"),
    ("scan the file", "grep"),
    ("search for keyword", "grep"),
    ("search for the keyword", "grep"),
    ("sort the file", "sort"),
    ("sort the output", "sort"),
    ("jot down", "echo"),
    ("write into", "echo"),
]

_WORD_MAP: dict[str, str] = {
    # Single word → canonical action
    "list": "ls",
    "ls": "ls",
    "dir": "ls",
    "navigate": "cd",
    "enter": "cd",
    "copy": "cp",
    "duplicate": "cp",
    "clone": "cp",
    "move": "mv",
    "rename": "mv",
    "relocate": "mv",
    "transfer": "mv",
    "delete": "rm",
    "remove": "rm",
    "erase": "rm",
    "compare": "diff",
    "differences": "diff",
    "diff": "diff",
    "search": "grep",
    "grep": "grep",
    "investigate": "grep",
    "sort": "sort",
    "order": "sort",
    "arrange": "sort",
    "display": "cat",
    "show": "cat",
    "view": "cat",
    "read": "cat",
    "print": "cat",
    "cat": "cat",
    "write": "echo",
    "echo": "echo",
    "find": "find_files",
    "locate": "find_files",
    "gather": "find_files",
    "create": "touch",
    "craft": "touch",
    "produce": "touch",
    "touch": "touch",
    "tail": "tail",
    "count": "wc",
    "tally": "wc",
    "pwd": "pwd",
    "storage": "du",
}

_TARGET_MAP: dict[str, str] = {
    "file": "file",
    "files": "file",
    "document": "file",
    "doc": "file",
    "folder": "directory",
    "directory": "directory",
    "dir": "directory",
    "path": "directory",
}

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
})


def extract_intent(query: str) -> dict[str, str]:
    """Extract filesystem intent from NL query.

    Returns:
        {action: str, target: str, keywords: str}
    """
    q_lower = query.lower()

    # 1. Phrase match (longest first)
    action = "none"
    for phrase, act in _PHRASE_MAP:
        if phrase in q_lower:
            action = act
            break

    # 2. Word match fallback
    if action == "none":
        words = re.sub(r"[^a-z0-9\s]", " ", q_lower).split()
        for w in words:
            if w in _WORD_MAP:
                action = _WORD_MAP[w]
                break

    # 3. Target extraction
    target = "none"
    words = re.sub(r"[^a-z0-9\s]", " ", q_lower).split()
    for w in words:
        if w in _TARGET_MAP:
            target = _TARGET_MAP[w]
            break

    # 4. Keywords (stop words removed)
    kw_tokens = [w for w in words if w not in _STOP_WORDS and len(w) > 1]
    keywords = " ".join(dict.fromkeys(kw_tokens))

    return {"action": action, "target": target, "keywords": keywords}
