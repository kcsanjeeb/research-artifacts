#!/usr/bin/env python3
"""Task 1: McNemar paired significance tests, DECAF E3-Phase2 (RVS-Ego, 72B judge)."""
import json
from scipy import stats

B = "/data3/zhuotaotian2_e2/runs/20260928_0315_e3_phase2_compB/"
P0 = "/data3/zhuotaotian2_e2/runs/20260927_0000_e3_phase0/"
W25 = "/data3/zhuotaotian2_e2/runs/20260926_1455_w25_judge_harness/"

ARMS = {
    "deferred": B + "judged_answers_7b_b_deferred_72b.json",
    "writetime": B + "judged_answers_7b_b_writetime_72b.json",
    "writetime_rawbias": B + "judged_answers_7b_b_writetime_rawbias_72b.json",
    "deferred_tax075": B + "judged_answers_7b_b_deferred_tax075_72b.json",
    "writetime_tax075": B + "judged_answers_7b_b_writetime_tax075_72b.json",
    "mukv_paper": P0 + "judged_7b_paper_72b.json",
    "stock_rekv_w25": W25 + "judged_7b_72b.json",
}

def load_ordered(path):
    """Return list of (key, correct) preserving record order; key includes occurrence
    index so duplicate (video_id,question) records pair up consistently across files."""
    recs = json.load(open(path))["judged"]
    seen = {}
    out = []
    for r in recs:
        k = (r["video_id"], r["question"])
        i = seen.get(k, 0)
        seen[k] = i + 1
        out.append(((k, i), 1 if r["judge_pred"] == "yes" else 0))
    return out

def as_map(ordered):
    return dict(ordered)

def mcnemar(a_map, b_map, name_a, name_b):
    keys = sorted(set(a_map) & set(b_map))
    only_a = set(a_map) - set(b_map)
    only_b = set(b_map) - set(a_map)
    both = bb = aw = bw = ab = ba = 0
    for k in keys:
        x, y = a_map[k], b_map[k]
        if x == 1 and y == 1:
            both += 1
        elif x == 0 and y == 0:
            bb += 1
        elif x == 1 and y == 0:
            ab += 1  # a right, b wrong
        else:
            ba += 1  # b right, a wrong
    n = len(keys)
    # exact: binomial test on discordant counts, two-sided
    d = ab + ba
    if d > 0:
        p_exact = stats.binomtest(ab, d, 0.5).pvalue
        chi2 = (abs(ab - ba) - 1) ** 2 / d
        p_cc = stats.chi2.sf(chi2, 1)
    else:
        p_exact = p_cc = 1.0
        chi2 = 0.0
    acc_a = (both + ab) / n
    acc_b = (both + ba) / n
    return dict(n=n, excluded=len(only_a) + len(only_b),
                both_correct=both, both_wrong=bb,
                a_right_b_wrong=ab, b_right_a_wrong=ba,
                chi2_cc=chi2, p_cc=p_cc, p_exact=p_exact,
                acc_a=acc_a, acc_b=acc_b, diff=acc_a - acc_b)

PAIRS = [
    ("deferred", "mukv_paper", "deferred vs MuKV-paper (headline +0.9 claim)"),
    ("deferred", "writetime", "deferred vs writetime (Gate-2: indistinguishable?)"),
    ("deferred_tax075", "writetime_tax075", "deferred_tax075 vs writetime_tax075 (Gate-2 tax075)"),
    ("deferred", "stock_rekv_w25", "deferred vs stock-ReKV W2.5 (-2.3 deficit)"),
    ("writetime", "writetime_rawbias", "writetime-debias vs writetime-rawbias (debias null, paired)"),
]

maps = {k: as_map(load_ordered(v)) for k, v in ARMS.items()}
order = {k: load_ordered(v) for k, v in ARMS.items()}
for k, v in order.items():
    n_rec = len(v)
    n_uniq = len({kk for kk, _ in v})
    print(f"arm {k}: {n_rec} records, {n_uniq} unique keys, acc(records)="
          f"{sum(c for _, c in v)/n_rec:.4f}")

print("\n" + "=" * 88)
print(f"{'pair':<46}{'n':>5}{'disc(a/b)':>11}{'chi2cc':>8}{'p_cc':>9}{'p_exact':>10}{'diff pt':>9}")
results = {}
for a, b, label in PAIRS:
    r = mcnemar(maps[a], maps[b], a, b)
    results[(a, b)] = r
    print(f"{label:<46}{r['n']:>5}{r['a_right_b_wrong']:>5}/{r['b_right_a_wrong']:<5}"
          f"{r['chi2_cc']:>8.3f}{r['p_cc']:>9.5f}{r['p_exact']:>10.5f}{100*r['diff']:>8.2f}")

print("\ndetail:")
for a, b, label in PAIRS:
    r = results[(a, b)]
    print(f"\n[{label}]")
    for k, v in r.items():
        print(f"  {k}: {v if not isinstance(v, float) else round(v, 6)}")

json.dump({f"{a}__{b}": r for (a, b), r in results.items()},
          open("/data3/zhuotaotian2_e2/deliverables/e3_phase2/mcnemar_results.json", "w"), indent=2)
print("\nwrote mcnemar_results.json")
