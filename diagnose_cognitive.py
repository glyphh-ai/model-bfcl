"""Diagnostic script for cognitive loop failures.

Runs entries and shows detailed per-turn breakdown:
  - Query text
  - IntentExtractor output (action, target, domain)
  - Resolved functions
  - Multi-action keyword matches
  - Expected vs predicted
"""

import json
import sys
from pathlib import Path

# Add glyphh-runtime to path
sys.path.insert(0, str(Path(__file__).parent.parent / "glyphh-runtime"))

from multi_turn_handler import (
    get_available_functions,
    extract_turn_query,
    parse_ground_truth_step,
    normalize_gfs_state,
    ConversationStateTracker,
)
from run_bfcl import load_bfcl_file, DATA_DIR, BFCL_FILES, BFCL_ANSWER_FILES
from glyphh.cognitive import CognitiveLoop, DomainConfig
from glyphh.intent import IntentExtractor
import re


def diagnose(max_entries=10):
    config = DomainConfig.from_file(Path(__file__).parent / "domain" / "gorilla_file_system.json")

    cat = "multi_turn_base"
    entries = load_bfcl_file(DATA_DIR / BFCL_FILES[cat])[:max_entries]

    # Load answers
    answer_entries = load_bfcl_file(DATA_DIR / BFCL_ANSWER_FILES[cat])
    answer_map = {ae.get("id", ""): ae for ae in answer_entries}
    answers = []
    for entry in entries:
        entry_id = entry.get("id", "")
        ae = answer_map.get(entry_id, {})
        answers.append(ae.get("ground_truth", []))

    extractor = IntentExtractor(packs=["filesystem"])

    total_turns = 0
    correct_turns = 0
    failure_categories = {
        "unmapped_action": 0,
        "extra_cd": 0,
        "extra_other": 0,
        "non_filesystem": 0,
        "wrong_func": 0,
        "missing_func": 0,
    }

    for idx, entry in enumerate(entries):
        gt = answers[idx] if idx < len(answers) else []
        available_funcs = get_available_functions(entry)
        available_names = {f["name"] for f in available_funcs}

        # Check if this entry has non-filesystem functions
        non_fs_funcs = available_names - {
            "ls", "cd", "cp", "mv", "cat", "grep", "sort", "touch", "mkdir",
            "rm", "rmdir", "diff", "echo", "find", "wc", "tail", "head",
            "pwd", "du", "chmod",
        }

        loop = CognitiveLoop(packs=["filesystem"], domain_config=config)
        initial_state = normalize_gfs_state(entry.get("initial_config", {}))
        loop.begin(functions=available_funcs, initial_state=initial_state)

        state_tracker = ConversationStateTracker()
        state_tracker.init_from_config(entry.get("initial_config", {}))

        turns = entry.get("question", [])
        entry_correct = True

        for turn_idx, turn in enumerate(turns):
            total_turns += 1
            query = extract_turn_query(turn)
            expected_step = gt[turn_idx] if turn_idx < len(gt) else []
            expected_calls = parse_ground_truth_step(expected_step)
            expected_funcs = set()
            for call in expected_calls:
                expected_funcs.update(call.keys())

            if not query:
                if not expected_calls:
                    correct_turns += 1
                continue

            # Get intent
            intent = extractor.extract(query)
            action = intent.get("action", "")
            target = intent.get("target", "")
            domain = intent.get("domain", "")

            # Inject state hints
            state_hint = state_tracker.get_state_hint(query)
            loop._state["query_mentions_child"] = state_hint.get("query_mentions_child")
            loop._state["query_mentions_other_dir"] = state_hint.get("query_mentions_other_dir")

            # Run cognitive step
            result = loop.step(query)
            predicted_funcs = set()
            predicted_dict = {}
            if result.action == "CALL":
                for call in result.calls:
                    for fname, args in call.items():
                        predicted_funcs.add(fname)
                        predicted_dict[fname] = args

            func_correct = predicted_funcs == expected_funcs

            # Update state tracker
            state_tracker.update_from_prediction(predicted_dict)

            # Hebbian reinforcement
            loop.confirm(
                was_correct=func_correct,
                correct_outcome=expected_calls if not func_correct else None,
            )

            if func_correct:
                correct_turns += 1
            else:
                entry_correct = False
                # Categorize failure
                missing = expected_funcs - predicted_funcs
                extra = predicted_funcs - expected_funcs

                has_non_fs = bool(expected_funcs & non_fs_funcs)

                # Check if action was in action_to_func
                mapped_func = config.action_to_func.get(action)

                # Detect which multi_action_keywords matched
                query_lower = query.lower()
                matched_keywords = {}
                for fname, patterns in config.multi_action_keywords.items():
                    for pat in patterns:
                        if re.search(pat, query_lower):
                            matched_keywords[fname] = pat
                            break

                print(f"\n  [{idx}] turn {turn_idx}: FAIL")
                print(f"    query: {query[:120]}")
                print(f"    intent: action={action!r} target={target!r} domain={domain!r}")
                print(f"    mapped_func: {mapped_func!r}")
                print(f"    keyword_matches: {matched_keywords}")
                print(f"    expected: {sorted(expected_funcs)}")
                print(f"    got:      {sorted(predicted_funcs)}")
                print(f"    missing:  {sorted(missing)}")
                print(f"    extra:    {sorted(extra)}")
                print(f"    conf:     {result.confidence:.3f}")
                if result.signals.get("recall"):
                    print(f"    recall:   {result.signals['recall'][:3]}")

                # Categorize
                if has_non_fs:
                    failure_categories["non_filesystem"] += 1
                elif not predicted_funcs and not mapped_func and not matched_keywords:
                    failure_categories["unmapped_action"] += 1
                elif "cd" in extra:
                    failure_categories["extra_cd"] += 1
                elif extra:
                    failure_categories["extra_other"] += 1
                elif missing and not extra:
                    failure_categories["missing_func"] += 1
                else:
                    failure_categories["wrong_func"] += 1

        loop.end()

    print(f"\n{'='*60}")
    print(f"  Turn accuracy: {correct_turns}/{total_turns} = {100*correct_turns/total_turns:.1f}%")
    print(f"\n  Failure categories:")
    for cat, count in sorted(failure_categories.items(), key=lambda x: -x[1]):
        if count > 0:
            print(f"    {cat}: {count}")
    print(f"{'='*60}")


if __name__ == "__main__":
    max_entries = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    diagnose(max_entries)
