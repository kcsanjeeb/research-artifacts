#!/usr/bin/env python3
"""W2.2 drift test: do blocks' selection-frequency groups drift across T1 queries?

Split each video's 12 queries (log order) into disjoint early (q_idx 0-5) and
late (q_idx 6-11) halves; assign blocks to Cold/Warm/Hot terciles by frequency
rank independently per half (over the union of blocks used in either half;
unused-in-a-half blocks get freq 0 and still take a rank); measure the
transition matrix P(late group | early group), the cold->hot fraction, and
Spearman rho between early and late frequency. TTKV reference: 7-8% cold->hot,
rho = 0.64-0.78 within one inference session.
"""
import json
from collections import Counter, defaultdict

import numpy as np

LOG = "/home/san/e1/runs/20260925_2237_t1_rekv_sharded/artifacts/t1_rekv_log.jsonl"
OUT = "/home/san/e1/w22_sensitivity/w22_drift.json"
GROUPS = ["Cold", "Warm", "Hot"]


def assign_groups(blocks, freq):
    order = sorted(blocks, key=lambda b: (freq.get(b, 0), b))
    n = len(order)
    return {b: min(2, r * 3 // n) for r, b in enumerate(order)}


def main():
    rows = [json.loads(l) for l in open(LOG)]
    by_vid = defaultdict(list)
    for r in rows:
        by_vid[r["video_id"]].append(r)

    trans = np.zeros((3, 3), dtype=np.int64)  # [early][late]
    rho_per_video = []
    cold_to_hot = []
    per_video = {}
    for vid, qrows in sorted(by_vid.items()):
        assert len(qrows) == 12, f"{vid}: {len(qrows)} queries"
        early, late = qrows[:6], qrows[6:]
        fe = Counter()
        for r in early:
            fe.update(set(r["retrieved_blocks"]))
        fl = Counter()
        for r in late:
            fl.update(set(r["retrieved_blocks"]))
        blocks = sorted(set(fe) | set(fl))
        ge = assign_groups(blocks, fe)
        gl = assign_groups(blocks, fl)
        for b in blocks:
            trans[ge[b], gl[b]] += 1
        x = np.array([fe.get(b, 0) for b in blocks], dtype=float)
        y = np.array([fl.get(b, 0) for b in blocks], dtype=float)
        rho = float(np.corrcoef(np.argsort(np.argsort(x)),
                                np.argsort(np.argsort(y)))[0, 1])  # Spearman
        rho_per_video.append(rho)
        ce = [b for b in blocks if ge[b] == 0]
        ch = float(np.mean([gl[b] == 2 for b in ce])) if ce else 0.0
        cold_to_hot.append(ch)
        per_video[vid] = {"n_blocks": len(blocks), "spearman": rho,
                          "cold_to_hot": ch}

    pooled = trans / trans.sum(axis=1, keepdims=True)
    out = {
        "n_videos": len(by_vid),
        "transition_counts_early_x_late": trans.tolist(),
        "transition_matrix_p_late_given_early": np.round(pooled, 4).tolist(),
        "groups": GROUPS,
        "cold_to_hot_frac_mean": float(np.mean(cold_to_hot)),
        "cold_to_hot_frac_per_video": {v[:8]: c for v, c in
                                       zip(sorted(by_vid), cold_to_hot)},
        "spearman_mean": float(np.mean(rho_per_video)),
        "spearman_per_video": {v[:8]: r for v, r in
                               zip(sorted(by_vid), rho_per_video)},
        "ttkv_reference": {"cold_to_hot": "0.07-0.08",
                           "rho": "0.64-0.78"},
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
