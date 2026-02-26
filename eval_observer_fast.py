#!/usr/bin/env python3
"""
Fast Observer eval — build once, test all 4 multi-turn categories.
One build, one eval pass. ~7 min total.
"""
import json, sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from handler import GlyphhBFCLHandler
from pattern_encoder import PatternRouter
from observer import Observer, _domain_from_classes
from multi_turn_handler import get_available_functions, extract_turn_query, parse_ground_truth_step

DATA_DIR = Path(__file__).parent / "data" / "bfcl"
CATS = ["multi_turn_base", "multi_turn_miss_func", "multi_turn_miss_param", "multi_turn_long_context"]
QFILES = {c: f"BFCL_v3_{c}.json" for c in CATS}
AFILES = {c: f"possible_answer/BFCL_v3_{c}.json" for c in CATS}

def load_jsonl(p):
    with open(p) as f:
        return [json.loads(l.strip()) for l in f if l.strip()]

def eval_cat(obs, cat, top_k=3):
    qs = {e["id"]: e for e in load_jsonl(DATA_DIR / QFILES[cat])}
    ans = {a["id"]: a for a in load_jsonl(DATA_DIR / AFILES[cat])}
    te = ce = tt = ct = sc = st = mc = mt = 0
    ids = list(ans.keys())
    for idx, eid in enumerate(ids):
        print(f"\r    [{idx+1}/{len(ids)}]", end="", flush=True)
        entry = qs.get(eid, {}); gt = ans[eid].get("ground_truth", [])
        turns = entry.get("question", []); af = get_available_functions(entry)
        if not af or not gt: continue
        te += 1; an = {f["name"] for f in af}
        dom = _domain_from_classes(entry.get("involved_classes", []))
        ok = True
        for ti, step in enumerate(gt):
            if not step: continue
            q = extract_turn_query(turns[ti] if ti < len(turns) else "")
            if not q: continue
            ec = parse_ground_truth_step(step)
            if not ec: continue
            ef = []; [ef.extend(c.keys()) for c in ec]
            tt += 1; multi = len(ef) > 1
            if multi: mt += 1
            else: st += 1
            pf = obs.decide(q, af, an, domain_hint=dom, top_k=top_k)
            if list(pf) == list(ef):
                ct += 1
                if multi: mc += 1
                else: sc += 1
            else: ok = False
        if ok: ce += 1
    print()
    return {"cat": cat, "entries": te, "e_acc": ce/te if te else 0, "ce": ce,
            "turns": tt, "t_acc": ct/tt if tt else 0, "ct": ct,
            "st": st, "s_acc": sc/st if st else 0, "mt": mt, "m_acc": mc/mt if mt else 0}

print("=" * 60)
print("  OBSERVER — ALL MULTI-TURN CATEGORIES")
print("=" * 60)
fr = GlyphhBFCLHandler(threshold=0.15)
pr = PatternRouter(); pr.build(min_count=2)
print(f"Model B: {len(pr.patterns)} patterns")
t0 = time.perf_counter()
obs = Observer(fr, pr); obs.build()
print(f"Observer: {len(obs.ref_glyphs)} refs in {time.perf_counter()-t0:.0f}s")

results = []
for cat in CATS:
    print(f"\n  {cat}:")
    t1 = time.perf_counter()
    r = eval_cat(obs, cat, top_k=3)
    r["time"] = time.perf_counter() - t1
    results.append(r)
    print(f"    Entry: {r['e_acc']:.1%} ({r['ce']}/{r['entries']})")
    print(f"    Turn:  {r['t_acc']:.1%} ({r['ct']}/{r['turns']})")
    print(f"    Single: {r['s_acc']:.1%} ({r['st']})  Multi: {r['m_acc']:.1%} ({r['mt']})")
    print(f"    Time: {r['time']:.1f}s")

tt = sum(r["turns"] for r in results)
ct = sum(r["ct"] for r in results)
te = sum(r["entries"] for r in results)
ce = sum(r["ce"] for r in results)
v4_mt = sum(r["e_acc"] for r in results) / len(results)

print(f"\n{'=' * 60}")
print(f"  TOTALS")
print(f"  Entry: {ce}/{te} = {ce/te:.1%}")
print(f"  Turn:  {ct}/{tt} = {ct/tt:.1%}")
print(f"  V4 Multi-Turn Score: {v4_mt:.1%}")
print(f"  Contributes {v4_mt * 0.30:.1%} to V4 overall (30% weight)")
print(f"  Total time: {time.perf_counter()-t0:.0f}s")
print(f"{'=' * 60}")
