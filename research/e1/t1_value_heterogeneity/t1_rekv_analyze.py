"""T1 ReKV-side analysis: per-block retrieval frequency + LOO value on RVS-Ego.
Bytes per block are uniform (KV block = 196 tokens x 24 layers x 2x2x64 fp16
= 2.4 MB at 0.5B), so value-per-byte is proportional to value; reported as such.
"""
import json
import os
import sys
from collections import Counter

import numpy as np

def gini_nonneg(x):
    x = np.maximum(np.asarray(x, dtype=float), 0)
    x = np.sort(x)
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) @ x) / (n * x.sum()) - (n + 1) / n)


def main(log_path, loo_path, out_json, out_png):
    recs = [json.loads(l) for l in open(log_path)]
    n_q = len(recs)
    total_blocks = sum(r["n_blocks"] for r in recs)  # per-query video blocks; use per-video
    vids = {}
    for r in recs:
        vids.setdefault(r["video_id"], r["n_blocks"])
    n_blocks_total = sum(vids.values())

    freq = Counter()
    for r in recs:
        for b in set(r["retrieved_blocks"]):
            freq[(r["video_id"], b)] += 1
    f = np.array([freq.get((v, b), 0) for v, nb in vids.items()
                  for b in range(nb)])
    loo = [json.loads(l) for l in open(loo_path)] if os.path.getsize(loo_path) else []
    v = np.zeros_like(f, dtype=float)
    # map LOO drops back to block ids
    vid_list = list(vids)
    offset = {}
    o = 0
    for vv in vid_list:
        offset[vv] = o
        o += vids[vv]
    drop_sum, drop_n = Counter(), Counter()
    for r in loo:
        drop_sum[(r["video_id"], r["block"])] += r["drop"]
        drop_n[(r["video_id"], r["block"])] += 1
    for key, s in drop_sum.items():
        v_, b_ = key
        v[offset[v_] + b_] = s / drop_n[key]   # mean token-F1 drop

    out = {
        "n_queries": n_q, "n_videos": len(vids), "n_blocks_total": int(n_blocks_total),
        "topk_per_query": 64,
        "bytes_per_block_mb": 2.4,
        "bytes_uniform_note": "KV bytes uniform per block -> v/bytes proportional to v",
        "gini_f": gini_nonneg(f),
        "gini_v_sampled": gini_nonneg(v[v != 0]) if (v != 0).any() else 0.0,
        "frac_blocks_never_retrieved": float(np.mean(f == 0)),
        "mean_drop_sampled": float(np.mean([r["drop"] for r in loo])) if loo else None,
        "frac_drop_positive": float(np.mean([r["drop"] > 0 for r in loo])) if loo else None,
        "n_loo": len(loo),
        "f": f.tolist(), "v_sampled": v.tolist(),
    }
    json.dump(out, open(out_json, "w"), indent=1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    ax.hist(np.log10(f[f > 0]), bins=30, color="tab:blue", alpha=0.8)
    ax.set_xlabel("log10(retrieval frequency f_j), retrieved blocks only")
    ax.set_title(f"ReKV/RVS-Ego block retrieval freq\n"
                 f"Gini(f)={out['gini_f']:.2f}; "
                 f"{out['frac_blocks_never_retrieved']*100:.0f}% blocks never retrieved")
    ax = axes[1]
    xs = np.sort(np.maximum(v, 0))
    cum = np.concatenate([[0.0], np.cumsum(xs)])
    if cum[-1] > 0:
        cum /= cum[-1]
    ax.plot(np.arange(len(cum)) / max(1, len(cum) - 1), cum, color="tab:orange")
    ax.plot([0, 1], [0, 1], "k--", lw=0.5)
    ax.set_xlabel("fraction of blocks (sorted)")
    ax.set_ylabel("cumulative value share")
    ax.set_title(f"Lorenz curve, sampled marginal value (token-F1 drop)\n"
                 f"Gini(v)={out['gini_v_sampled']:.2f} over sampled+all-zero blocks")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    print(json.dumps({k: out[k] for k in ("gini_f", "gini_v_sampled",
                                          "frac_blocks_never_retrieved",
                                          "mean_drop_sampled",
                                          "frac_drop_positive", "n_loo")},
                     indent=1))


if __name__ == "__main__":
    main(*sys.argv[1:5])
