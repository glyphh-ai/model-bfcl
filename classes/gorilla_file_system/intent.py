"""Per-class intent overrides for GorillaFileSystem."""

CLASS_ALIASES = ["GorillaFileSystem", "gorilla", "file system", "filesystem",
                 "gorilla file system"]
CLASS_DOMAIN = "filesystem"

ACTION_SYNONYMS = {
    "navigate": "cd", "gather": "find", "locate": "find", "identify": "find",
    "craft": "touch", "draft": "touch", "produce": "touch", "generate": "touch",
    "jot": "echo", "infuse": "echo", "put": "echo",
    "juxtapose": "diff", "spot": "diff",
    "tally": "wc", "count": "wc",
    "peek": "cat", "reveal": "cat", "display": "cat", "view": "cat",
    "arrange": "sort", "organize": "sort",
    "transfer": "mv", "shift": "mv", "rename": "mv",
    "duplicate": "cp", "backup": "cp",
    "inventory": "ls",
}

TARGET_OVERRIDES = {
    "folder": "directory", "dir": "directory",
    "doc": "file", "document": "file",
    "txt": "file", "pdf": "file", "csv": "file",
    "report": "file", "archive": "directory",
}
