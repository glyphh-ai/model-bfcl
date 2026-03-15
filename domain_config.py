"""
Per-class DomainConfig for BFCL multi-turn APIs.

Each BFCL API class gets its own DomainConfig for use with per-class
CognitiveLoop instances. This enables:
  - action_to_func: direct action → function mapping in RESOLVE stage
  - multi_action_keywords: multi-function detection per turn
  - exclusion_rules: prevent confusable function pairs

CLASS_DOMAIN_CONFIGS maps class name → DomainConfig (or None).
"""

from glyphh import DomainConfig


# ── GorillaFileSystem ─────────────────────────────────────────────────────

FS_DOMAIN_CONFIG = DomainConfig.from_dict({
    "domain": "gorilla_filesystem",
    "slot_definitions": {
        # Navigation target — match against known subdirs, then quoted (skip filenames)
        "folder": {
            "type": "entity",
            "strategies": ["state_match:locations_here", "quoted_filtered:items_here", "fallback_words"],
            "fallback_words": {"parent": "..", "back": "..", "up": ".."},
        },
        # Directory name for mkdir/rmdir — usually a new name, so quoted (skip existing files)
        "dir_name": {
            "type": "text",
            "strategies": ["quoted_filtered:items_here"],
        },
        # File name — match against known files in CWD, then quoted (skip dir names), then implicit
        "file_name": {
            "type": "entity",
            "strategies": ["state_match:items_here", "quoted_filtered:locations_here", "implicit_single:items_here"],
        },
        # Diff: first file
        "file_name1": {
            "type": "entity",
            "strategies": ["quoted:0", "state_match:items_here"],
        },
        # Diff: second file
        "file_name2": {
            "type": "entity",
            "strategies": ["quoted:1"],
        },
        # Move/copy source — before transfer word, or match against files
        "source": {
            "type": "entity",
            "strategies": ["positional_before_transfer", "state_match:items_here", "quoted:0"],
        },
        # Move/copy destination — after transfer word, or match against dirs
        "destination": {
            "type": "entity",
            "strategies": ["positional_after_transfer", "state_match:locations_here", "quoted:1"],
        },
        # Echo content — quoted string (usually the longest one)
        "content": {
            "type": "text",
            "strategies": ["quoted:0"],
        },
        # Grep pattern — quoted string
        "pattern": {
            "type": "text",
            "strategies": ["quoted"],
        },
        # Find name — quoted string
        "name": {
            "type": "text",
            "strategies": ["quoted"],
        },
        # Find path — usually "." (current dir)
        "path": {
            "type": "text",
            "strategies": ["fallback_words", "quoted"],
            "fallback_words": {"current": ".", "here": ".", "this": "."},
        },
        # Tail/head line count
        "lines": {
            "type": "number",
            "strategies": ["number"],
        },
        # ls -a flag
        "a": {
            "type": "boolean",
            "strategies": ["bool_triggers"],
            "triggers": {"all": True, "hidden": True, "-a": True, "including": True},
        },
        # du human-readable flag
        "human_readable": {
            "type": "boolean",
            "strategies": ["bool_triggers"],
            "triggers": {"human": True, "readable": True, "human-readable": True},
        },
        # wc mode
        "mode": {
            "type": "enum",
            "strategies": ["keyword_map"],
            "keyword_map": {"words": "w", "word": "w", "lines": "l", "line": "l",
                            "characters": "c", "character": "c", "chars": "c"},
        },
    },
    "action_to_func": {
        "list":        "GorillaFileSystem.ls",
        "navigate":    "GorillaFileSystem.cd",
        "find":        "GorillaFileSystem.find",
        "read":        "GorillaFileSystem.cat",
        "write":       "GorillaFileSystem.echo",
        "create_file": "GorillaFileSystem.touch",
        "create_dir":  "GorillaFileSystem.mkdir",
        "move":        "GorillaFileSystem.mv",
        "copy":        "GorillaFileSystem.cp",
        "delete":      "GorillaFileSystem.rm",
        "delete_dir":  "GorillaFileSystem.rmdir",
        "compare":     "GorillaFileSystem.diff",
        "search":      "GorillaFileSystem.grep",
        "sort":        "GorillaFileSystem.sort",
        "tail":        "GorillaFileSystem.tail",
        "count":       "GorillaFileSystem.wc",
        "disk_usage":  "GorillaFileSystem.du",
        "cwd":         "GorillaFileSystem.pwd",
    },
    "multi_action_keywords": {
        "GorillaFileSystem.cd":    ["navigate", "go to", "cd ", "open folder",
                                     "pop on over", "go into", "cd into",
                                     "head over to", "switch to",
                                     "in the directory", "in tmp directory",
                                     "open the", "directly open",
                                     "in our communal", "in our shared",
                                     "in the existing", "over to the",
                                     "to the folder", "into a '",
                                     "into the folder", "in documents directory",
                                     "navigate to"],
        "GorillaFileSystem.find":  ["find every", "find all", "find file", "find any",
                                     "locate a ", "locate any", "locate the ",
                                     "search for file", "search for a file",
                                     "gather files", "gather file", "identify file",
                                     "have in their name", "in its file name",
                                     "in its name", "in their file name",
                                     "nestled within"],
        "GorillaFileSystem.mv":    ["move", "transfer", "shift", "rename"],
        "GorillaFileSystem.cp":    ["copy", "duplicate", "backup", "make a copy",
                                     "safely copied", "copied into", "copy it over",
                                     "transfer a duplicate"],
        "GorillaFileSystem.sort":  ["sort", "arrange", "organize", "alphabetically",
                                     "alphabetical order", "sorted",
                                     "having its contents sorted",
                                     "benefit from having its"],
        "GorillaFileSystem.diff":  ["compare", "juxtapose", "differences",
                                     "distinctions", "disparities",
                                     "delve into the difference", "spot the difference",
                                     "comparison between", "draw a comparison",
                                     "pinpoint any", "difference of the",
                                     "are identical", "verify both"],
        "GorillaFileSystem.grep":  ["grep", "search in", "find lines",
                                     "identify sections", "occurrence of",
                                     "search the file for", "search for the keyword",
                                     "scan the file", "pattern in",
                                     "identify any line", "any line with"],
        "GorillaFileSystem.wc":    ["count", "tally", "how many words",
                                     "how many lines", "how many characters",
                                     "word count"],
        "GorillaFileSystem.cat":   ["display contents", "show contents",
                                     "read the content", "view its contents",
                                     "what's inside", "reveal the contents",
                                     "take a peek", "peek at what",
                                     "output the complete content",
                                     "have a look at what",
                                     "may i have a look",
                                     "complete content of"],
        "GorillaFileSystem.ls":    ["list of files", "inventory", "available files",
                                     "show me the list", "list the files",
                                     "list all files", "files located within"],
        "GorillaFileSystem.echo":  ["write", "jot down", "put some", "infuse",
                                     "write into", "echo", "store the",
                                     "here are the things i want to put",
                                     "put together", "laid out"],
        "GorillaFileSystem.touch": ["create a file", "craft a new file",
                                     "produce a file", "generate a file",
                                     "create a document", "draft up",
                                     "file creation", "initiate a file",
                                     "producing a file", "producing a fil",
                                     "new file call"],
        "GorillaFileSystem.mkdir": ["create a directory", "new directory",
                                     "set up a new directory",
                                     "generate a new directory",
                                     "create the directory",
                                     "make sure to create",
                                     "creating the directory",
                                     "ensure the directory",
                                     "make sure the"],
        "GorillaFileSystem.tail":  ["last lines", "last line", "last part", "tail",
                                     "last few lines", "last entry",
                                     "last 20 lines", "last 10 lines",
                                     "show the last"],
        "GorillaFileSystem.du":    ["disk usage", "storage usage",
                                     "human readable", "human readible",
                                     "how much space", "log the storage"],
    },
    "exclusion_rules": {
        "GorillaFileSystem.rm":    ["GorillaFileSystem.rmdir"],
        "GorillaFileSystem.rmdir": ["GorillaFileSystem.rm"],
        "GorillaFileSystem.cat":   ["GorillaFileSystem.tail"],
        "GorillaFileSystem.tail":  ["GorillaFileSystem.cat"],
        "GorillaFileSystem.touch": ["GorillaFileSystem.echo"],
    },
    "state_effects": {
        "GorillaFileSystem.cd": [
            {"op": "set_primary", "from_arg": "folder", "parent_keyword": ".."},
        ],
    },
    "state_format": {
        "separator": "/",
        "default_primary": "",
        "tree_key": "_tree",
    },
    "transitions": [
        {
            "name": "context_switch",
            "directing_actions": ["mv", "cp", "touch", "mkdir", "echo", "rm", "rmdir"],
            "operating_actions": ["cat", "grep", "sort", "diff", "tail", "wc", "ls", "find_files", "head"],
            "prerequisite": "cd",
        },
    ],
})


# ── TwitterAPI ────────────────────────────────────────────────────────────

TWITTER_DOMAIN_CONFIG = DomainConfig.from_dict({
    "domain": "twitter_api",
    "action_to_func": {
        "tweet":   "TwitterAPI.post_tweet",
        "comment": "TwitterAPI.comment",
        "retweet": "TwitterAPI.retweet",
        "mention": "TwitterAPI.mention",
    },
    "multi_action_keywords": {
        "TwitterAPI.post_tweet": ["tweet", "post on twitter", "broadcast",
                                   "share on twitter", "draft a tweet",
                                   "post a tweet", "crafting a tweet",
                                   "on social media", "share on social",
                                   "share the result", "share the verbatim",
                                   "posting a summary"],
        "TwitterAPI.comment":    ["comment", "reply", "underneath",
                                   "comment underneath", "supportive comment",
                                   "add a comment", "add a supportive"],
        "TwitterAPI.retweet":    ["retweet", "share the tweet", "amplify"],
        "TwitterAPI.mention":    ["mention", "mentioning"],
        "TwitterAPI.authenticate_twitter": ["authenticate", "log in to twitter",
                                              "sign in to twitter"],
    },
})


# ── MessageAPI ────────────────────────────────────────────────────────────

MESSAGE_DOMAIN_CONFIG = DomainConfig.from_dict({
    "domain": "message_api",
    "action_to_func": {
        "send_message":  "MessageAPI.send_message",
        "view_messages": "MessageAPI.view_messages_received",
    },
    "multi_action_keywords": {
        "MessageAPI.send_message":          ["send message", "dispatch", "message my",
                                              "message to", "send a message"],
        "MessageAPI.view_messages_received": ["view messages", "check messages",
                                               "received messages", "inbox"],
    },
})


# ── TicketAPI ─────────────────────────────────────────────────────────────

TICKET_DOMAIN_CONFIG = DomainConfig.from_dict({
    "domain": "ticket_api",
    "action_to_func": {
        "create_ticket":  "TicketAPI.create_ticket",
        "resolve_ticket": "TicketAPI.resolve_ticket",
        "get_ticket":     "TicketAPI.get_ticket",
        "close_ticket":   "TicketAPI.close_ticket",
        "edit_ticket":    "TicketAPI.edit_ticket",
    },
    "multi_action_keywords": {
        "TicketAPI.create_ticket":  ["create ticket", "draft a ticket",
                                      "support ticket", "new ticket",
                                      "open a ticket"],
        "TicketAPI.resolve_ticket": ["resolve", "check it off", "mark as resolved"],
        "TicketAPI.get_ticket":     ["get ticket", "retrieve ticket",
                                      "ticket details", "ticket status"],
        "TicketAPI.close_ticket":   ["close ticket", "close the ticket"],
        "TicketAPI.edit_ticket":    ["edit ticket", "update ticket",
                                      "modify ticket"],
    },
})


# ── MathAPI ───────────────────────────────────────────────────────────────

MATH_DOMAIN_CONFIG = DomainConfig.from_dict({
    "domain": "math_api",
    "action_to_func": {
        "mean":       "MathAPI.mean",
        "logarithm":  "MathAPI.logarithm",
        "add_nums":   "MathAPI.add",
        "subtract":   "MathAPI.subtract",
        "multiply":   "MathAPI.multiply",
        "divide":     "MathAPI.divide",
        "sqrt":       "MathAPI.square_root",
        "power":      "MathAPI.power",
        "percentage": "MathAPI.percentage",
    },
    "multi_action_keywords": {
        "MathAPI.mean":        ["average", "mean of", "compute the average"],
        "MathAPI.logarithm":   ["logarithm", "log of", "log base"],
        "MathAPI.add":         ["add", "sum of", "total of"],
        "MathAPI.subtract":    ["subtract", "minus", "difference"],
        "MathAPI.multiply":    ["multiply", "product of", "times"],
        "MathAPI.divide":      ["divide", "quotient", "divided by"],
        "MathAPI.square_root": ["square root", "sqrt"],
        "MathAPI.power":       ["power", "raised to", "exponent"],
        "MathAPI.percentage":  ["percentage", "percent of"],
    },
})


# ── TradingBot ────────────────────────────────────────────────────────────

TRADING_DOMAIN_CONFIG = DomainConfig.from_dict({
    "domain": "trading_bot",
    "action_to_func": {
        "buy":     "TradingBot.buy",
        "sell":    "TradingBot.sell",
        "balance": "TradingBot.get_balance",
        "history": "TradingBot.get_history",
        "quote":   "TradingBot.get_quote",
    },
    "multi_action_keywords": {
        "TradingBot.buy":         ["buy", "purchase", "acquire"],
        "TradingBot.sell":        ["sell", "liquidate", "dump"],
        "TradingBot.get_balance": ["balance", "funds", "buying power"],
        "TradingBot.get_history": ["history", "past trades", "trade log"],
        "TradingBot.get_quote":   ["quote", "price of", "current price"],
    },
})


# ── TravelBookingAPI ─────────────────────────────────────────────────────

TRAVEL_DOMAIN_CONFIG = DomainConfig.from_dict({
    "domain": "travel_booking",
    "action_to_func": {
        "book_flight":   "TravelBookingAPI.book_flight",
        "book_hotel":    "TravelBookingAPI.book_hotel",
        "cancel":        "TravelBookingAPI.cancel_booking",
        "check_in":      "TravelBookingAPI.check_in",
        "flight_status": "TravelBookingAPI.get_flight_status",
    },
    "multi_action_keywords": {
        "TravelBookingAPI.book_flight":       ["book flight", "fly to", "plane ticket"],
        "TravelBookingAPI.book_hotel":        ["book hotel", "reserve room", "hotel in"],
        "TravelBookingAPI.cancel_booking":    ["cancel booking", "cancel flight",
                                                "cancel hotel"],
        "TravelBookingAPI.check_in":          ["check in", "checkin"],
        "TravelBookingAPI.get_flight_status": ["flight status", "is my flight",
                                                "flight delayed"],
    },
})


# ── VehicleControlAPI ────────────────────────────────────────────────────

VEHICLE_DOMAIN_CONFIG = DomainConfig.from_dict({
    "domain": "vehicle_control",
    "action_to_func": {
        "accelerate":  "VehicleControlAPI.accelerate",
        "brake":       "VehicleControlAPI.brake",
        "steer":       "VehicleControlAPI.steer",
        "lock":        "VehicleControlAPI.lock",
        "unlock":      "VehicleControlAPI.unlock",
        "start":       "VehicleControlAPI.start_engine",
        "stop":        "VehicleControlAPI.stop_engine",
        "set_climate": "VehicleControlAPI.set_climate",
        "navigate":    "VehicleControlAPI.set_navigation",
    },
    "multi_action_keywords": {
        "VehicleControlAPI.accelerate":     ["accelerate", "speed up", "go faster"],
        "VehicleControlAPI.brake":          ["brake", "slow down", "decelerate"],
        "VehicleControlAPI.lock":           ["lock the", "lock doors"],
        "VehicleControlAPI.unlock":         ["unlock the", "unlock doors"],
        "VehicleControlAPI.start_engine":   ["start engine", "start the car",
                                              "turn on the car"],
        "VehicleControlAPI.stop_engine":    ["stop engine", "turn off", "shut down"],
        "VehicleControlAPI.set_climate":    ["temperature", "air conditioning",
                                              "climate control"],
        "VehicleControlAPI.set_navigation": ["navigate to", "directions to",
                                              "take me to"],
    },
})


# ── PostingAPI (same structure as Twitter, different class prefix) ────────

POSTING_DOMAIN_CONFIG = DomainConfig.from_dict({
    "domain": "posting_api",
    "action_to_func": {
        "post":    "PostingAPI.post",
        "comment": "PostingAPI.comment",
        "share":   "PostingAPI.share",
    },
    "multi_action_keywords": {
        "PostingAPI.post":    ["post", "create post", "publish"],
        "PostingAPI.comment": ["comment", "reply"],
        "PostingAPI.share":   ["share", "repost"],
    },
})


# ── Registry ──────────────────────────────────────────────────────────────

CLASS_DOMAIN_CONFIGS: dict[str, DomainConfig | None] = {
    "GorillaFileSystem": FS_DOMAIN_CONFIG,
    "TwitterAPI":        TWITTER_DOMAIN_CONFIG,
    "MessageAPI":        MESSAGE_DOMAIN_CONFIG,
    "PostingAPI":        POSTING_DOMAIN_CONFIG,
    "TicketAPI":         TICKET_DOMAIN_CONFIG,
    "MathAPI":           MATH_DOMAIN_CONFIG,
    "TradingBot":        TRADING_DOMAIN_CONFIG,
    "TravelBookingAPI":  TRAVEL_DOMAIN_CONFIG,
    "TravelAPI":         TRAVEL_DOMAIN_CONFIG,
    "VehicleControlAPI": VEHICLE_DOMAIN_CONFIG,
}
