#!/usr/bin/env python3
"""W3.2 — random-selector baseline + permutation nulls for the oracle gaps.

Benchmarks:
  RVS 2-video subset: n=316 records (314 unique questions), 72B judged,
    policies seg/frame/patch/default; system = writetime full run filtered
    to the subset videos; block->time: 4 s/block (effective 0.25 fps after
    double decimation of the 0.5-fps npy).
  StreamingBench RT: n=180, letter-match judge, 2 s/block (0.5 fps mp4 path);
    system = answers_7b_writetime (73.9).

(a) Random-per-question selector floor: expected accuracy = mean_q(k_q/4)
    (analytic) + seeded simulation (10,000 draws) with 95% CI.
(b) Permutation null for the oracle gap (10,000 perms, seed 2024):
    Null 1 (primary): independently shuffle each policy's correctness column
      across questions (preserves marginals, destroys per-question
      correlation). Null oracle = per-question max of shuffled columns;
      null best-fixed = max shuffled column accuracy. Gaps:
      oracle - system, oracle - best-fixed. Empirical p vs observed.
    Null 2 (secondary): within each question, permute the 4 correctness
      values across policy labels (per-question multiset fixed). Oracle is
      invariant under this null; only oracle - best-fixed varies.

Outputs: w32_random_floor.json, w32_permutation_null.json
"""
import json, os
import numpy as np
from scipy import stats

B = "/data3/zhuotaotian2_e2/runs/20260928_0315_e3_phase2_compB/"
XB = "/data3/zhuotaotian2_e2/runs/20260928_2105_e3_phase2_xbench/"
OUT = "/data3/zhuotaotian2_e2/deliverables/e3_phase2/analysis/w32_oracle_null/"
os.makedirs(OUT, exist_ok=True)
SEED = 2024
N_PERM = 10_000
N_SIM = 10_000

POLS = ["seg", "frame", "patch", "default"]


def load_rvs():
    mats = {}
    for p in POLS:
        recs = json.load(open(B + f"judged_oracle_{p}_72b.json"))["judged"]
        mats[p] = np.array([1 if r["judge_pred"] == "yes" else 0 for r in recs])
    # system (writetime) on the subset, occurrence-aligned to oracle file order
    sub = {v["video_id"] for v in json.load(open(B + "oracle_subset_anno.json"))}
    wt = {}
    recs = json.load(open(B + "judged_answers_7b_b_writetime_72b.json"))["judged"]
    for r in recs:
        if r["video_id"] in sub:
            k = (r["video_id"], r["question"])
            wt.setdefault(k, []).append(1 if r["judge_pred"] == "yes" else 0)
    seen, sysv = {}, []
    orecs = json.load(open(B + "judged_oracle_seg_72b.json"))["judged"]
    for r in orecs:
        k = (r["video_id"], r["question"])
        i = seen.get(k, 0); seen[k] = i + 1
        sysv.append(wt[k][i])
    mats["system"] = np.array(sysv)
    return mats


def load_xb():
    mats = {}
    for p in POLS:
        recs = json.load(open(XB + f"judged_oracle_{p}_letter.json"))["records"]
        mats[p] = np.array([1 if r["correct"] else 0 for r in recs])
    recs = json.load(open(XB + "judged_answers_7b_writetime_letter.json"))["records"]
    mats["system"] = np.array([1 if r["correct"] else 0 for r in recs])
    return mats


def analyze(name, mats):
    M = np.stack([mats[p] for p in POLS], axis=1)          # (n, 4)
    n = M.shape[0]
    sysv = mats["system"]
    k = M.sum(1)
    oracle = (k > 0).astype(int)
    col_acc = M.mean(0)
    best_fixed_i = int(np.argmax(col_acc))
    out = {
        "n": n,
        "per_policy_acc": {p: float(M[:, i].mean()) for i, p in enumerate(POLS)},
        "oracle_acc": float(oracle.mean()),
        "best_fixed": POLS[best_fixed_i],
        "best_fixed_acc": float(col_acc[best_fixed_i]),
        "system_acc": float(sysv.mean()),
        "gap_oracle_minus_system_pt": float(100 * (oracle.mean() - sysv.mean())),
        "gap_oracle_minus_best_fixed_pt": float(100 * (oracle.mean() - col_acc[best_fixed_i])),
        "all_right": int((k == 4).sum()),
        "all_wrong": int((k == 0).sum()),
        "mixed": int(((k > 0) & (k < 4)).sum()),
    }

    # ---- (a) random floor -------------------------------------------------
    p_q = k / 4.0
    exp_acc = float(p_q.mean())
    rng = np.random.default_rng(SEED)
    # simulate: for each draw, per-question Bernoulli(p_q)
    draws = rng.random((N_SIM, n)) < p_q[None, :]
    sim_acc = draws.mean(1)
    out["random_floor"] = {
        "analytic_expected_acc": exp_acc,
        "sim_mean_acc": float(sim_acc.mean()),
        "sim_sd_acc": float(sim_acc.std()),
        "sim_ci95": [float(np.percentile(sim_acc, 2.5)), float(np.percentile(sim_acc, 97.5))],
        "gap_system_minus_floor_pt": float(100 * (sysv.mean() - exp_acc)),
        "gap_oracle_minus_floor_pt": float(100 * (oracle.mean() - exp_acc)),
    }

    # ---- (b) permutation nulls -------------------------------------------
    obs_gap_sys = oracle.mean() - sysv.mean()
    obs_gap_bf = oracle.mean() - col_acc[best_fixed_i]

    # Null 1: column shuffle across questions
    null_oracle = np.empty(N_PERM); null_bf = np.empty(N_PERM); null_sysgap = np.empty(N_PERM)
    for t in range(N_PERM):
        S = np.empty_like(M)
        for j in range(4):
            S[:, j] = rng.permutation(M[:, j])
        no = (S.sum(1) > 0).mean()
        nbf = S.mean(0).max()
        null_oracle[t] = no
        null_bf[t] = nbf
        null_sysgap[t] = no - sysv.mean()
    # oracle - best-fixed under shuffle: pair the same perm
    null_bfgap = null_oracle - null_bf
    p_sys = float((np.sum(null_sysgap >= obs_gap_sys - 1e-12) + 1) / (N_PERM + 1))
    p_bf = float((np.sum(null_bfgap >= obs_gap_bf - 1e-12) + 1) / (N_PERM + 1))

    # Null 2: within-question label permutation (oracle invariant)
    null2_bf = np.empty(N_PERM)
    for t in range(N_PERM):
        S = M.copy()
        for i in range(n):
            S[i] = M[i, rng.permutation(4)]
        null2_bf[t] = S.mean(0).max()
    obs2 = oracle.mean()
    null2_bfgap = obs2 - null2_bf
    p_bf2 = float((np.sum(null2_bfgap >= obs_gap_bf - 1e-12) + 1) / (N_PERM + 1))

    out["permutation_null"] = {
        "n_perm": N_PERM, "seed": SEED,
        "null1_column_shuffle": {
            "oracle_acc_mean": float(null_oracle.mean()),
            "oracle_acc_sd": float(null_oracle.std()),
            "oracle_acc_q95": float(np.percentile(null_oracle, 95)),
            "oracle_acc_max": float(null_oracle.max()),
            "observed_oracle_acc": float(oracle.mean()),
            "observed_gap_oracle_minus_system": float(obs_gap_sys),
            "empirical_p_gap_vs_system": p_sys,
            "observed_gap_oracle_minus_best_fixed": float(obs_gap_bf),
            "empirical_p_gap_vs_best_fixed": p_bf,
            "null_gap_bf_mean": float(null_bfgap.mean()),
            "null_gap_bf_sd": float(null_bfgap.std()),
            "null_gap_bf_q95": float(np.percentile(null_bfgap, 95)),
            "null_gap_bf_max": float(null_bfgap.max()),
        },
        "null2_within_question_label_perm": {
            "observed_gap_oracle_minus_best_fixed": float(obs_gap_bf),
            "empirical_p_gap_vs_best_fixed": p_bf2,
            "null_bf_acc_mean": float(null2_bf.mean()),
            "null_bf_acc_max": float(null2_bf.max()),
        },
    }
    return out


results = {}
results["rvs_subset"] = analyze("rvs_subset", load_rvs())
results["streamingbench_rt"] = analyze("streamingbench_rt", load_xb())
json.dump(results, open(OUT + "w32_results.json", "w"), indent=1)

for name, r in results.items():
    print(f"\n== {name} (n={r['n']}) ==")
    print(f"  policies: " + "  ".join(f"{p} {100*v:.1f}" for p, v in r["per_policy_acc"].items()))
    print(f"  system {r['system_acc']*100:.1f} | best-fixed {r['best_fixed']} {r['best_fixed_acc']*100:.1f} | oracle {r['oracle_acc']*100:.1f}")
    print(f"  partition: all-right {r['all_right']} | mixed {r['mixed']} | all-wrong {r['all_wrong']}")
    rf = r["random_floor"]
    print(f"  random floor: analytic {100*rf['analytic_expected_acc']:.2f} | sim {100*rf['sim_mean_acc']:.2f} +- {100*rf['sim_sd_acc']:.2f} | CI { [round(100*x,2) for x in rf['sim_ci95']] }")
    print(f"  system-floor gap {rf['gap_system_minus_floor_pt']:+.2f} pt | oracle-floor gap {rf['gap_oracle_minus_floor_pt']:+.2f} pt")
    pn = r["permutation_null"]["null1_column_shuffle"]
    print(f"  null1: oracle null mean {100*pn['oracle_acc_mean']:.2f} sd {100*pn['oracle_acc_sd']:.2f} q95 {100*pn['oracle_acc_q95']:.2f} max {100*pn['oracle_acc_max']:.2f} | observed {100*pn['observed_oracle_acc']:.2f}")
    print(f"  null1: gap-vs-system obs {100*pn['observed_gap_oracle_minus_system']:.2f} p={pn['empirical_p_gap_vs_system']:.4f} | gap-vs-bestfixed obs {100*pn['observed_gap_oracle_minus_best_fixed']:.2f} p={pn['empirical_p_gap_vs_best_fixed']:.4f}")
    n2 = r["permutation_null"]["null2_within_question_label_perm"]
    print(f"  null2: gap-vs-bestfixed p={n2['empirical_p_gap_vs_best_fixed']:.4f} (null bf max {100*n2['null_bf_acc_max']:.2f})")
