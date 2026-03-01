"""Analyze all cd-missing failures from the 20-entry run to find patterns."""
import sys, json
sys.path.insert(0, ".")

# The error output from the run shows all failures — let's categorize them
failures = [
    # (entry, turn, expected, got)
    (0, 1, ['cd', 'grep'], ['grep']),
    (0, 3, ['cd', 'diff', 'mv'], ['diff', 'mv']),
    (1, 0, ['ls'], ['ls', 'pwd']),           # over-prediction, not cd-missing
    (1, 1, ['cd', 'mv'], ['cd', 'mkdir', 'mv']),  # over-prediction
    (1, 2, ['cd', 'grep'], ['grep']),
    (2, 3, ['cd', 'cp', 'mv'], ['cp']),      # cd + mv missing
    (3, 1, ['cd', 'cp'], ['find']),           # wrong func entirely
    (5, 0, ['cd', 'mv'], ['find']),           # wrong func entirely
    (5, 3, ['comment'], ['authenticate_twitter', 'post_tweet']),  # wrong func
    (6, 4, ['cd', 'echo', 'touch'], ['echo', 'wc']),  # cd + touch missing
    (7, 1, ['find'], ['ls']),                 # wrong func
    (8, 2, ['authenticate_twitter', 'post_tweet'], ['authenticate_twitter', 'diff']),
    (8, 3, ['comment'], ['diff']),
    (9, 1, ['cd', 'cp', 'mv'], ['cp']),      # cd + mv missing
    (10, 1, ['cd', 'mv'], ['mv']),           # cd missing
]

# Categorize
cd_missing = [f for f in failures if 'cd' in f[2] and 'cd' not in f[3]]
over_pred = [f for f in failures if len(f[3]) > len(f[2]) or any(x not in f[2] for x in f[3])]
wrong_func = [f for f in failures if not any(x in f[2] for x in f[3])]
cd_present_but_wrong = [f for f in failures if 'cd' not in f[2] and 'cd' not in f[3] and f not in wrong_func and f not in over_pred]

print("=== FAILURE ANALYSIS (20 entries, 70 turns) ===")
print(f"Total failures: {len(failures)}")
print(f"Correct turns: {70 - len(failures)} / 70 = {(70-len(failures))/70:.1%}")
print()

print(f"CD-MISSING ({len(cd_missing)} failures):")
for entry, turn, exp, got in cd_missing:
    ops_correct = [x for x in got if x in exp and x != 'cd']
    ops_missing = [x for x in exp if x not in got and x != 'cd']
    print(f"  entry {entry} turn {turn}: expected {exp} got {got}")
    print(f"    ops correct: {ops_correct}, ops missing: {ops_missing}")
print()

print(f"OVER-PREDICTION ({len([f for f in over_pred if f not in cd_missing])} failures):")
for f in over_pred:
    if f not in cd_missing:
        print(f"  entry {f[0]} turn {f[1]}: expected {f[2]} got {f[3]}")
print()

print(f"WRONG FUNCTION ({len(wrong_func)} failures):")
for f in wrong_func:
    print(f"  entry {f[0]} turn {f[1]}: expected {f[2]} got {f[3]}")
print()

print("=== IMPACT ===")
print(f"If cd-missing were fixed: {len(cd_missing)} more turns correct")
print(f"New turn accuracy: {(70-len(failures)+len(cd_missing))/70:.1%} (from {(70-len(failures))/70:.1%})")

# How many entries would flip to correct?
# Entry is correct only if ALL turns are correct
# Count per-entry failures
entry_failures = {}
for entry, turn, exp, got in failures:
    entry_failures.setdefault(entry, []).append((turn, exp, got))

cd_only_entries = []
for eid, turns in entry_failures.items():
    all_cd_missing = all(
        'cd' in exp and 'cd' not in got and all(x in exp for x in got)
        for turn, exp, got in turns
    )
    if all_cd_missing:
        cd_only_entries.append(eid)

print(f"\nEntries where ALL failures are cd-missing: {cd_only_entries}")
print(f"Fixing cd would add {len(cd_only_entries)} more correct entries → {3+len(cd_only_entries)}/20 = {(3+len(cd_only_entries))/20:.0%}")
