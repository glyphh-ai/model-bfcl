"""
Glyphh-based conversation state tracker for multi-turn BFCL.

Encodes the current filesystem state (cwd, known files, directory tree)
as Glyphh vectors. Provides similarity-based signals about whether
the user's query references the current directory or a different one.

This is NOT training to answers — it's encoding observable state from
the conversation (predicted function calls) and the initial filesystem
config, then using Glyphh similarity to inform the next turn.
"""

import re
from glyphh import Encoder
from glyphh.core.config import EncoderConfig, Layer, Segment, Role
from glyphh.core.types import Concept
from glyphh.core.ops import cosine_similarity


_STATE_CONFIG = EncoderConfig(
    dimension=10000,
    seed=42,
    apply_weights_during_encoding=True,
    include_temporal=False,
    layers=[
        Layer(
            name="location",
            similarity_weight=0.6,
            segments=[
                Segment(
                    name="current_dir",
                    roles=[
                        Role(
                            name="cwd_words",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
                Segment(
                    name="available_dirs",
                    roles=[
                        Role(
                            name="child_dirs",
                            similarity_weight=0.8,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
        Layer(
            name="files",
            similarity_weight=0.4,
            segments=[
                Segment(
                    name="visible",
                    roles=[
                        Role(
                            name="file_names",
                            similarity_weight=1.0,
                            text_encoding="bag_of_words",
                        ),
                    ],
                ),
            ],
        ),
    ],
)


class ConversationStateTracker:
    """Tracks filesystem state across multi-turn conversations.

    Parses the initial_config to build a directory tree, then updates
    cwd based on predicted cd() calls after each turn. Provides:
      - Exact cwd path (for LLM prompt)
      - Files visible at cwd (for LLM prompt)
      - Child directories at cwd (for LLM prompt)
      - Glyphh similarity between query mentions and cwd vs other dirs
        (for cd intent refinement)
    """

    def __init__(self):
        self._encoder = Encoder(_STATE_CONFIG)
        self._dir_tree = {}      # full directory tree: path -> {files: [], dirs: []}
        self._cwd = "/"          # current working directory
        self._all_paths = set()  # all known directory paths
        self._last_actions = []  # last N predicted function names

    def init_from_config(self, initial_config: dict):
        """Parse initial_config to build the directory tree and set initial cwd."""
        self._dir_tree = {}
        self._all_paths = set()

        gfs = initial_config.get("GorillaFileSystem", {})
        root = gfs.get("root", {})

        if not root:
            self._cwd = "/"
            return

        # The root has one top-level directory (e.g., "workspace", "alex")
        # That's the initial cwd
        root_name = list(root.keys())[0]
        self._cwd = "/" + root_name

        # Recursively parse the tree
        self._parse_tree(root[root_name], "/" + root_name)

    def _parse_tree(self, node: dict, path: str):
        """Recursively parse a filesystem node into the dir_tree."""
        if not isinstance(node, dict):
            return

        files = []
        dirs = []
        contents = node.get("contents", node) if node.get("type") == "directory" else node

        if not isinstance(contents, dict):
            return

        for name, child in contents.items():
            if name in ("type", "content", "contents"):
                continue
            if isinstance(child, dict):
                if child.get("type") == "file":
                    files.append(name)
                elif child.get("type") == "directory":
                    dirs.append(name)
                    self._parse_tree(child, path + "/" + name)
                elif "contents" in child:
                    # Implicit directory
                    dirs.append(name)
                    self._parse_tree(child, path + "/" + name)

        self._dir_tree[path] = {"files": files, "dirs": dirs}
        self._all_paths.add(path)

    def get_cwd(self) -> str:
        """Return current working directory path."""
        return self._cwd

    def get_files_at_cwd(self) -> list[str]:
        """Return files visible at current working directory."""
        info = self._dir_tree.get(self._cwd, {})
        return info.get("files", [])

    def get_dirs_at_cwd(self) -> list[str]:
        """Return child directories at current working directory."""
        info = self._dir_tree.get(self._cwd, {})
        return info.get("dirs", [])

    def update_from_prediction(self, predicted: dict):
        """Update state based on predicted function calls from a turn.

        Tracks cd() calls to update cwd, mkdir() to add new directories,
        touch() to add new files, mv()/cp() for file movements.
        """
        if not isinstance(predicted, dict):
            return

        self._last_actions = list(predicted.keys())

        for fname, args in predicted.items():
            if not isinstance(args, dict):
                continue

            if fname == "cd":
                folder = args.get("folder", "")
                if folder == "..":
                    # Go up one level
                    parts = self._cwd.rsplit("/", 1)
                    if len(parts) > 1 and parts[0]:
                        self._cwd = parts[0]
                    # else: already at root, stay
                elif folder:
                    new_path = self._cwd + "/" + folder
                    self._cwd = new_path
                    # Ensure it exists in our tree
                    if new_path not in self._dir_tree:
                        self._dir_tree[new_path] = {"files": [], "dirs": []}
                        self._all_paths.add(new_path)

            elif fname == "mkdir":
                dir_name = args.get("dir_name", "")
                if dir_name:
                    new_path = self._cwd + "/" + dir_name
                    if new_path not in self._dir_tree:
                        self._dir_tree[new_path] = {"files": [], "dirs": []}
                        self._all_paths.add(new_path)
                    # Add to parent's dirs list
                    if self._cwd in self._dir_tree:
                        if dir_name not in self._dir_tree[self._cwd]["dirs"]:
                            self._dir_tree[self._cwd]["dirs"].append(dir_name)

            elif fname == "touch":
                file_name = args.get("file_name", "")
                if file_name and self._cwd in self._dir_tree:
                    if file_name not in self._dir_tree[self._cwd]["files"]:
                        self._dir_tree[self._cwd]["files"].append(file_name)

            elif fname == "mv":
                source = args.get("source", "")
                dest = args.get("destination", "")
                if source and dest and self._cwd in self._dir_tree:
                    # Remove from current location
                    files = self._dir_tree[self._cwd]["files"]
                    if source in files:
                        files.remove(source)
                    # If dest is a directory name, file moves there
                    dest_path = self._cwd + "/" + dest
                    if dest_path in self._dir_tree:
                        self._dir_tree[dest_path]["files"].append(source)

            elif fname == "cp":
                source = args.get("source", "")
                dest = args.get("destination", "")
                if source and dest:
                    dest_path = self._cwd + "/" + dest
                    if dest_path in self._dir_tree:
                        self._dir_tree[dest_path]["files"].append(source)

    def get_state_hint(self, query: str) -> dict:
        """Generate state-aware hints for the LLM prompt.

        Returns:
            {
                "cwd": str,                    # current working directory
                "cwd_short": str,              # just the last component
                "files_here": list[str],       # files at cwd
                "dirs_here": list[str],        # child dirs at cwd
                "query_mentions_cwd": bool,    # query mentions current dir name
                "query_mentions_child": str|None,  # which child dir query mentions
                "query_mentions_parent": bool, # query mentions parent dir
                "needs_cd_signal": str,        # "likely_yes", "likely_no", "unclear"
            }
        """
        query_lower = query.lower()
        query_words = set(re.sub(r"[^a-z0-9\s]", "", query_lower).split())

        cwd_parts = self._cwd.strip("/").split("/")
        cwd_short = cwd_parts[-1] if cwd_parts else ""
        parent_short = cwd_parts[-2] if len(cwd_parts) >= 2 else ""

        files_here = self.get_files_at_cwd()
        dirs_here = self.get_dirs_at_cwd()

        # Check if query mentions the current directory name
        query_mentions_cwd = cwd_short.lower() in query_lower if cwd_short else False

        # Check if query mentions a child directory
        query_mentions_child = None
        for d in dirs_here:
            if d.lower() in query_lower:
                query_mentions_child = d
                break

        # Check if query mentions parent/ancestor directory
        query_mentions_parent = parent_short.lower() in query_lower if parent_short else False

        # Check if query mentions a file that exists here
        query_mentions_local_file = False
        for f in files_here:
            fname_lower = f.lower().replace(".", " ").split()
            if any(w in query_words for w in fname_lower if len(w) > 2):
                query_mentions_local_file = True
                break

        # Check if query mentions a directory we know about but isn't cwd or child
        query_mentions_other_dir = None
        for path in self._all_paths:
            dirname = path.strip("/").split("/")[-1].lower()
            if dirname and dirname in query_lower:
                if dirname != cwd_short.lower() and dirname not in [d.lower() for d in dirs_here]:
                    query_mentions_other_dir = dirname
                    break

        # Determine cd signal
        if query_mentions_child:
            # Query mentions a child directory → likely needs cd into it
            needs_cd = "likely_yes"
        elif query_mentions_other_dir:
            # Query mentions a dir that's not cwd or child → needs navigation
            needs_cd = "likely_yes"
        elif query_mentions_parent and not query_mentions_child:
            # Query mentions parent dir → might need to go up
            needs_cd = "likely_yes"
        elif query_mentions_local_file and not query_mentions_other_dir:
            # Query mentions a file that's here and no other dir → no cd
            needs_cd = "likely_no"
        elif query_mentions_cwd and not query_mentions_other_dir and not query_mentions_parent:
            # Query mentions current dir by name, nothing else → already here
            needs_cd = "likely_no"
        else:
            needs_cd = "unclear"

        return {
            "cwd": self._cwd,
            "cwd_short": cwd_short,
            "files_here": files_here,
            "dirs_here": dirs_here,
            "query_mentions_cwd": query_mentions_cwd,
            "query_mentions_child": query_mentions_child,
            "query_mentions_parent": query_mentions_parent,
            "query_mentions_other_dir": query_mentions_other_dir,
            "needs_cd_signal": needs_cd,
        }

    def reset(self):
        """Reset state for a new conversation."""
        self._dir_tree = {}
        self._cwd = "/"
        self._all_paths = set()
        self._last_actions = []
