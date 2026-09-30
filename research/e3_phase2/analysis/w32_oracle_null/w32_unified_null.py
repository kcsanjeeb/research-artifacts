#!/usr/bin/env python3
"""W3.2 protocol reconciliation: recompute the oracle-gap permutation null for
RVS / StreamingBench / OVO-Bench under ONE identical protocol, from the
per-question judged jsons.

Canonical protocol (= the RVS/SB null2 within-question exchangeability scheme):
  - null construction: for each question, independently permute the 4 policy
    correctness values across the 4 policy labels (per-question multiset fixed;
    policy marginals NOT preserved across questions).
  - gap under the null: oracle accuracy (per-question max) minus best-fixed
    policy accuracy (max column mean under the permuted matrix). Oracle is
    invariant under this null.
  - 10,000 iterations, seed 2024 (a fresh stream per benchmark).
  - one-sided upper-tail p with +1 correction: (sum(null_gap >= obs) + 1)/(N+1).
"""
import json
import numpy as np

SEED = 2024
N_PERM = 10_000
POLS = ["seg", "frame", "patch", "default"]

B = "/data3/zhuotaotian2_e2/runs/20260928_0315_e3_phase2_compB/"
XB = "/data3/zhuotaotian2_e2/runs/20260928_2105_e3_phase2_xbench/"
OVO = "/data3/zhuotaotian2_e2/runs/20260929_1215_e3_phase2_ovo/"
OUT = "/data3/zhuotaotian2_e2/deliverables/e3_phase2/analysis/w32_oracle_null/"

def load_rvs():
    M = []
    for p in POLS:
        recs = json.load(open(B + f"judged_oracle_{p}_72b.json"))["judged"]
        M.append([1 if r["judge_pred"] == "yes" else 0 for r in recs])
    sysv = json.load(open(B + "judged_answers_7b_b_writetime_72b.json"))["judged"]
    return np.array(M).T, np.mean([1 if r["judge_pred"] == "yes" else 0 for r in sysv])

def load_xb():
    M = []
    for p in POLS:
        recs = json.load(open(XB + f"judged_oracle_{p}_letter.json"))["records"]
        M.append([1 if r["correct"] else 0 for r in recs])
    sysv = json.load(open(XB + "judged_answers_7b_writetime_letter.json"))["records"]
    return np.array(M).T, np.mean([1 if r["correct"] else 0 for r in sysv])

def load_ovo():
    M = []
    for p in POLS:
        recs = json.load(open(OVO + f"judged_oracle_{p}_letter.json"))["records"]
        M.append([1 if r["correct"] else 0 for r in recs])
    sysv = json.load(open(OVO + "judged_answers_7b_writetime_letter.json"))["records"]
    return np.array(M).T, np.mean([1 if r["correct"] else 0 for r in sysv])

def unified_null(name, M, sys_acc):
    n = M.shape[0]
    assert M.shape[1] == 4
    col_acc = M.mean(0)
    best_i = int(np.argmax(col_acc))
    oracle = M.any(1).mean()
    best_fixed = col_acc[best_i]
    obs_gap = oracle - best_fixed
    rng = np.random.default_rng(SEED)
    null_bf = np.empty(N_PERM)
    for t in range(N_PERM):
        perm = np.argsort(rng.random((n, 4)), axis=1)
        Mp = np.take_along_axis(M, perm, axis=1)
        null_bf[t] = Mp.mean(0).max()
    null_gap = oracle - null_bf
    p = float((np.sum(null_gap >= obs_gap - 1e-12) + 1) / (N_PERM + 1))
    pct = float(100 * np.mean(null_gap < obs_gap))
    z = float((obs_gap - null_gap.mean()) / null_gap.std())
    return {
        "benchmark": name, "n": n,
        "per_policy_acc": {p: float(M[:, i].mean()) for i, p in enumerate(POLS)},
        "oracle_acc": float(oracle), "best_fixed": POLS[best_i],
        "best_fixed_acc": float(best_fixed), "system_acc": float(sys_acc),
        "obs_gap_oracle_minus_best_fixed": float(obs_gap),
        "n_perm": N_PERM, "seed": SEED,
        "null_gap_mean": float(null_gap.mean()), "null_gap_sd": float(null_gap.std()),
        "null_gap_q05": float(np.percentile(null_gap, 5)),
        "null_gap_q95": float(np.percentile(null_gap, 95)),
        "obs_gap_percentile_in_null": pct,
        "p_one_sided_upper_tail": p, "z": z,
    }

def holm(pvals, alpha=0.05):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        val = (m - rank) * pvals[i]
        running = max(running, val)
        adj[i] = min(running, 1.0)
    rej = adj < alpha  # Holm step-up rejection
    return adj, rej

def bh(pvals, alpha=0.05):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 1.0
    for rank in range(m - 1, -1, -1):
        i = order[rank]
        running = min(running, m * pvals[i] / (rank + 1))
        adj[i] = min(running, 1.0)
    return adj, adj < alpha

results = {
    "rvs_subset": unified_null("rvs_subset", *load_rvs()),
    "streamingbench_rt": unified_null("streamingbench_rt", *load_xb()),
    "ovo_bench": unified_null("ovo_bench", *load_ovo()),
}
names = list(results)
pvals = [results[k]["p_one_sided_upper_tail"] for k in names]
holm_adj, holm_rej = holm(pvals)
bh_adj, bh_rej = bh(pvals)
results["multiplicity"] = {
    "method_note": "3 tests, family = oracle-gap permutation tests across benchmarks",
    "holm_adjusted": {k: float(a) for k, a in zip(names, holm_adj)},
    "holm_reject_alpha0.05": {k: bool(r) for k, r in zip(names, holm_rej)},
    "bh_adjusted": {k: float(a) for k, a in zip(names, bh_adj)},
    "bh_reject_alpha0.05": {k: bool(r) for k, r in zip(names, bh_rej)},
}
json.dump(results, open(OUT + "w32_unified_null.json", "w"), indent=1)

for k in names:
    r = results[k]
    print(f"== {k} (n={r['n']}) ==")
    print("  policies " + "  ".join(f"{p} {100*v:.1f}" for p, v in r["per_policy_acc"].items()))
    print(f"  system {100*r['system_acc']:.1f} | best-fixed {r['best_fixed']} {100*r['best_fixed_acc']:.1f} | oracle {100*r['oracle_acc']:.1f}")
    print(f"  obs gap {100*r['obs_gap_oracle_minus_best_fixed']:.2f} pt | null mean {100*r['null_gap_mean']:.2f} sd {100*r['null_gap_sd']:.2f}")
    print(f"  obs percentile in null {r['obs_gap_percentile_in_null']:.2f}th | p(one-sided upper) {r['p_one_sided_upper_tail']:.4f} | z {r['z']:+.2f}")
print("== multiplicity ==")
for k in names:
    print(f"  {k}: Holm adj p={results['multiplicity']['holm_adjusted'][k]:.4f} reject={results['multiplicity']['holm_reject_alpha0.05'][k]} | BH adj p={results['multiplicity']['bh_adjusted'][k]:.4f} reject={results['multiplicity']['bh_reject_alpha0.05'][k]}")
