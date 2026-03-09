"""Per-class intent overrides for GorillaFileSystem.

Maps NL verbs/targets to the 18 filesystem functions.
Used by the per-class encoder for Stage 2 routing.
"""

CLASS_ALIASES = ["GorillaFileSystem", "filesystem", "file system", "fs", "files"]
CLASS_DOMAIN = "filesystem"

# NL verb → canonical function name (bare, without class prefix)
ACTION_SYNONYMS = {
    # cat — read/display content
    "display": "cat", "show": "cat", "print": "cat", "read": "cat", "view": "cat",
    # cd — change directory
    "change directory": "cd", "go to": "cd", "navigate": "cd", "enter": "cd",
    # cp — copy
    "copy": "cp", "duplicate": "cp", "clone": "cp",
    # diff — compare
    "compare": "diff", "difference": "diff",
    # du — disk usage
    "disk usage": "du", "size": "du", "space": "du",
    # echo — write/output
    "write": "echo", "output": "echo", "append": "echo",
    # find — locate files
    "find": "find", "locate": "find", "search files": "find",
    # grep — search content
    "grep": "grep", "search": "grep", "search for": "grep",
    "look for": "grep", "investigate": "grep",
    # ls — list
    "list": "ls", "dir": "ls", "directory listing": "ls",
    # mkdir — create directory
    "create directory": "mkdir", "make directory": "mkdir", "new folder": "mkdir",
    # mv — move/rename
    "move": "mv", "rename": "mv", "relocate": "mv",
    # pwd — current directory
    "current directory": "pwd", "where am i": "pwd", "working directory": "pwd",
    # rm — delete file
    "delete": "rm", "remove": "rm", "erase": "rm",
    # rmdir — delete directory
    "delete directory": "rmdir", "remove directory": "rmdir",
    # sort — sort content
    "sort": "sort", "order": "sort", "arrange": "sort",
    # tail — read end of file
    "last lines": "tail", "end of": "tail", "bottom": "tail",
    # touch — create file
    "create file": "touch", "new file": "touch",
    # wc — word/line count
    "word count": "wc", "count lines": "wc", "count words": "wc", "line count": "wc",
}

# NL noun → canonical target for encoder lexicon matching
TARGET_OVERRIDES = {
    "document": "file", "doc": "file",
    "folder": "directory", "dir": "directory", "path": "directory",
    "text": "content", "output": "content", "data": "content",
    "storage": "disk", "space": "disk",
}

# Function name → (action, target) for encoder
FUNCTION_INTENTS = {
    "cat":    ("read", "content"),
    "cd":     ("navigate", "directory"),
    "cp":     ("copy", "file"),
    "diff":   ("compare", "file"),
    "du":     ("check", "disk"),
    "echo":   ("write", "content"),
    "find":   ("find", "file"),
    "grep":   ("search", "content"),
    "ls":     ("list", "file"),
    "mkdir":  ("create", "directory"),
    "mv":     ("move", "file"),
    "pwd":    ("check", "directory"),
    "rm":     ("delete", "file"),
    "rmdir":  ("delete", "directory"),
    "sort":   ("sort", "content"),
    "tail":   ("read", "content"),
    "touch":  ("create", "file"),
    "wc":     ("count", "content"),
}
