"""T1 analysis: ReKV/RVS-Ego side, pooled across sharded run artifacts.
Complements t1_rekv_analyze.py (per-block Gini) with per-video stats and an
additive uniform-vs-oracle headroom estimate over LOO-sampled blocks.

Bytes per block are identical (196 tok x 24 layers x 2 KV x 2 kv-heads x 64
dim x fp16 = 2,408,448 B), so Gini(v/byte) == Gini(v).

Usage: python t1_rekv_pooled_analysis.py RUN_DIR OUT_PNG SUMMARY_JSON
"""
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np

BLOCK_BYTES = 196 * 24 * 2 * 2 * 64 * 2


def gini(x):
    x = np.sort(np.maximum(np.asarray(x, dtype=float), 0))
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) @ x) / (n * x.sum()) - (n + 1) / n)


def lorenz(x):
    x = np.sort(np.maximum(np.asarray(x, dtype=float), 0))
    cum = np.concatenate([[0.0], np.cumsum(x)])
    if cum[-1] > 0:
        cum = cum / cum[-1]
    return np.arange(len(cum)) / (len(cum) - 1), cum


def main():
    run_dir, out_prefix, summary_p = sys.argv[1], sys.argv[2], sys.argv[3]
    logs, loos = [], []
    seen_q, seen_l = set(), set()
    for p in sorted(glob.glob(os.path.join(run_dir, "artifacts",
                                           "t1_rekv_log_shard*.jsonl"))):
        for l in open(p):
            r = json.loads(l)
            k = (r["video_id"], r["question"])
            if k not in seen_q:
                seen_q.add(k)
                logs.append(r)
    for p in sorted(glob.glob(os.path.join(run_dir, "artifacts",
                                           "t1_rekv_loo_shard*.jsonl"))):
        for l in open(p):
            r = json.loads(l)
            k = (r["video_id"], r["question"], r["block"])
            if k not in seen_l:
                seen_l.add(k)
                loos.append(r)
    print(f"{len(logs)} query logs, {len(loos)} LOO evals (deduped)")

    by_vid = defaultdict(list)
    for r in logs:
        by_vid[r["video_id"]].append(r)
    drops_vb = defaultdict(lambda: defaultdict(list))
    drops_vq = defaultdict(dict)
    for r in loos:
        drops_vb[r["video_id"]][r["block"]].append(r["drop"])
        drops_vq[r["video_id"]].setdefault(r["question"], {})[r["block"]] = r["drop"]

    f_all, v_all, sampled_v, per_video = [], [], [], {}
    for vid, rs in sorted(by_vid.items()):
        n_blocks = rs[0]["n_blocks"]
        f = np.zeros(n_blocks)
        for r in rs:
            for b in set(r["retrieved_blocks"]):
                f[b] += 1
        v = np.zeros(n_blocks)
        for b, dr in drops_vb[vid].items():
            v[b] = float(np.mean(dr))
        f_all.append(f)
        v_all.append(v)
        sampled_v += [float(np.mean(dr)) for dr in drops_vb[vid].values()]
        per_video[vid] = {
            "n_blocks": int(n_blocks), "n_queries": len(rs),
            "mean_token_f1": float(np.mean([r["token_f1"] for r in rs])),
            "frac_never_retrieved": float(np.mean(f == 0)),
            "gini_f": gini(f), "gini_v": gini(v),
            "n_loo_blocks": len(drops_vb[vid]),
        }
    f_all = np.concatenate(f_all)
    v_all = np.concatenate(v_all)
    sampled_v = np.array(sampled_v)
    mean_f1 = float(np.mean([per_video[v]["mean_token_f1"] for v in per_video]))

    budgets = [0.75, 0.5, 0.25, 0.1]
    uw = {"budgets": budgets, "uniform": {}, "oracle": {},
          "note": ("additive approximation over LOO-sampled blocks, drops "
                   "clipped at 0 (negative LOO deltas = noise, treated as zero "
                   "value, same clip0 convention as the SMB side): query F1 "
                   "with removed set M = max(0, F1_full - sum of clipped drops "
                   "of removed blocks); uniform = mean of 10 random subsets of "
                   "sampled blocks; oracle = keep highest-value sampled blocks. "
                   "Never-retrieved blocks (v=0) dropped first by both.")}
    drops_vb0 = {vid: {b: max(0.0, float(np.mean(dr)))
                       for b, dr in bb.items()}
                 for vid, bb in drops_vb.items()}
    for b in budgets:
        uni_all, ora_all = [], []
        for vid, rs in sorted(by_vid.items()):
            samp = sorted(drops_vb0[vid])
            if not samp:
                continue
            k = max(1, int(round(len(samp) * b)))
            order = sorted(samp, key=lambda j: -drops_vb0[vid][j])
            oracle_keep = set(order[:k])
            rng = np.random.RandomState(0)
            uni_sets = [set(rng.choice(samp, size=k, replace=False).tolist())
                        for _ in range(10)]
            d0 = {q: {j: max(0.0, dr) for j, dr in dd.items()}
                  for q, dd in drops_vq[vid].items()}
            for r in rs:
                full = r["token_f1"]
                d = d0.get(r["question"], {})
                uvals = [max(0.0, full - sum(d[j] for j in d if j not in ks))
                         for ks in uni_sets]
                uni_all.append(float(np.mean(uvals)))
                ora_all.append(max(0.0, full - sum(d[j] for j in d
                                                   if j not in oracle_keep)))
        uw["uniform"][str(b)] = round(float(np.mean(uni_all)), 4)
        uw["oracle"][str(b)] = round(float(np.mean(ora_all)), 4)
        print("budget %s: uniform F1=%s oracle F1=%s"
              % (b, uw["uniform"][str(b)], uw["oracle"][str(b)]), flush=True)

    # query-level concentration: share of a query's total clipped sampled
    # value carried by its single most valuable retrieved block
    conc = []
    for vid in drops_vq:
        for q, dd in drops_vq[vid].items():
            pos = [max(0.0, dr) for dr in dd.values()]
            if sum(pos) > 0:
                conc.append(max(pos) / sum(pos))
    query_concentration = {"mean_top1_share": float(np.mean(conc)) if conc else None,
                           "n_queries_with_positive_value": len(conc)}

    # null model: Gini(f) under uniform-random retrieval of the same top-k
    # (structural skew from retrieving 64 of ~1800 blocks over 12 queries)
    rng = np.random.RandomState(0)
    f_null = []
    for vid, rs in sorted(by_vid.items()):
        n_blocks = rs[0]["n_blocks"]
        fn = np.zeros(n_blocks)
        for _ in rs:
            fn[rng.choice(n_blocks, size=64, replace=False)] += 1
        f_null.append(fn)
    f_null = np.concatenate(f_null)

    out = {
        "n_videos": len(by_vid), "n_queries": len(logs), "n_loo_evals": len(loos),
        "mean_token_f1_full_memory": mean_f1,
        "block_bytes": BLOCK_BYTES,
        "bytes_note": "all blocks identical KV size -> Gini(v/byte) == Gini(v)",
        "gini_f_pooled": gini(f_all), "gini_v_pooled": gini(v_all),
        "gini_f_null_uniform_retrieval": gini(f_null),
        "gini_v_sampled": gini(sampled_v),
        "gini_f_per_video_mean": float(np.mean([p["gini_f"] for p in per_video.values()])),
        "gini_v_per_video_mean": float(np.mean([p["gini_v"] for p in per_video.values()])),
        "frac_never_retrieved_pooled": float(np.mean(f_all == 0)),
        "frac_v_negative_sampled": float(np.mean(sampled_v < 0)),
        "frac_v_zero_sampled": float(np.mean(sampled_v == 0)),
        "query_value_concentration": query_concentration,
        "uniform_vs_oracle_additive": uw,
        "per_video": per_video,
    }
    json.dump(out, open(summary_p, "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    nz = sampled_v[sampled_v != 0]
    if len(nz):
        ax.hist(np.sign(nz) * np.log10(np.abs(nz) + 1e-4), bins=40,
                color="tab:blue", alpha=0.8)
    ax.set_xlabel("sign(v)*log10(|v|+1e-4), LOO-sampled blocks")
    ax.set_ylabel("# blocks")
    ax.set_title("ReKV/RVS-Ego marginal value (token-F1 drop)\n"
                 "Gini(f)=%.2f, Gini(v)=%.2f; %.0f%% never retrieved"
                 % (out["gini_f_pooled"], out["gini_v_pooled"],
                    out["frac_never_retrieved_pooled"] * 100))
    ax = axes[1]
    for arr, name in ((f_all, "retrieval freq f_j"), (v_all, "marginal value v_j")):
        lx, ly = lorenz(arr)
        ax.plot(lx, ly, label="%s (Gini=%.2f)" % (name, gini(arr)))
    ax.plot([0, 1], [0, 1], "k--", lw=0.5)
    ax.set_xlabel("fraction of blocks (sorted)")
    ax.set_ylabel("cumulative share")
    ax.legend(fontsize=8)
    ax.set_title("Lorenz curves (pooled, %d videos)" % len(by_vid))
    ax = axes[2]
    ub = [uw["uniform"][str(b)] for b in budgets]
    ob = [uw["oracle"][str(b)] for b in budgets]
    ax.plot([b * 100 for b in budgets], ub, marker="o",
            label="uniform removal (10 seeds)")
    ax.plot([b * 100 for b in budgets], ob, marker="s",
            label="oracle water-fill (LOO value)")
    ax.axhline(mean_f1, color="gray", ls=":", lw=0.8,
               label="full memory (%.3f)" % mean_f1)
    ax.set_xlabel("retention budget within sampled blocks (%)")
    ax.set_ylabel("mean token-F1 (additive approx)")
    ax.legend(fontsize=8)
    ax.set_title("ReKV/RVS-Ego: uniform vs oracle (additive)")
    fig.tight_layout()
    fig.savefig(out_prefix, dpi=150)
    print("fig saved", out_prefix)


if __name__ == "__main__":
    main()
