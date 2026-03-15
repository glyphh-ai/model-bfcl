#!/usr/bin/env python3
"""Quick test: run N miss_func entries and export for gorilla eval.

Usage:
    python test_miss_func.py [count] [offset]
    python test_miss_func.py 20 180    # entries 180-199
"""
import sys
from run_full_eval import run_multi_turn_categories, export_all_results, export_gorilla_results

n = int(sys.argv[1]) if len(sys.argv) > 1 else 10
offset = int(sys.argv[2]) if len(sys.argv) > 2 else 0

results = run_multi_turn_categories(["multi_turn_miss_func"], workers=3, max_entries=n, offset=offset)
for cr in results:
    print(f"\n{cr.category}: {cr.correct}/{cr.total} ({100*cr.correct/cr.total:.1f}%)")

export_all_results(results)
export_gorilla_results(results)
print("\nDone — files exported.")
