"""Eval multi_turn_base with all 3 Glyphh intent models."""
import sys, time, json
from pathlib import Path
from collections import Counter
sys.path.insert(0, str(Path(__file__).parent))

from handler import GlyphhBFCLHandler
from llm_client import get_client
from run_bfcl import run_multi_turn_category


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    model = sys.argv[2] if len(sys.argv) > 2 else "gpt-4.1"

    print(f"Running multi_turn_base: {n} entries, LLM={model}")
    print(f"Models: DirectoryIntent + FileCreation + SearchSuppression")
    print()

    handler = GlyphhBFCLHandler(threshold=0.15)
    llm = get_client("openai", model)

    start = time.time()
    result = run_multi_turn_category(handler, "multi_turn_base", llm=llm, max_entries=n)
    elapsed = time.time() - start

    print(f"\n{'='*60}")
    print(f"RESULTS ({n} entries)")
    print(f"{'='*60}")
    print(f"Entry: {result['correct']}/{result['total']} = {result['accuracy']:.3f}")
    print(f"Turn:  {result['correct_turns']}/{result['total_turns']} = {result['turn_accuracy']:.3f}")
    print(f"Time:  {elapsed:.1f}s")

    func_over = Counter()
    func_under = Counter()
    for r in result["results"]:
        for d in r.get("details", []):
            if not d.get("correct", True):
                pred = set(d.get("predicted_funcs", []))
                exp = set(d.get("expected_funcs", []))
                for f in (pred - exp): func_over[f] += 1
                for f in (exp - pred): func_under[f] += 1

    print(f"\nOver:  {dict(func_over.most_common(10))}")
    print(f"Under: {dict(func_under.most_common(10))}")


if __name__ == "__main__":
    main()
