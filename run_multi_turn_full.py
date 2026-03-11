#!/usr/bin/env python3
"""Run all multi-turn categories with per-section result files and logs.

Usage:
    python run_multi_turn_full.py
"""
from run_full_eval import run_multi_turn_categories, export_all_results, export_gorilla_results, V4_MULTI_TURN_CATS

# Run each category separately so results flush per-section
for cat in V4_MULTI_TURN_CATS:
    print(f"\n{'='*60}")
    print(f"  Running: {cat}")
    print(f"{'='*60}")
    results = run_multi_turn_categories([cat], workers=3)
    for cr in results:
        print(f"\n  {cr.category}: {cr.correct}/{cr.total} ({100*cr.correct/cr.total:.1f}%)")
    export_all_results(results)
    export_gorilla_results(results)
    print(f"  {cat} — files exported.\n")

print("\nAll multi-turn categories complete.")
