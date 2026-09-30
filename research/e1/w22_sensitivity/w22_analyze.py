#!/usr/bin/env python3
"""W2.2 analysis: quantization distortion by selection-frequency quintile.

Aggregates w22_quant_shard*.jsonl: per-quintile distortion at 2/4/8-bit,
2->4-bit reduction, Hot-vs-Cold ratio, and the PASS/FAIL gate
(PASS = Hot 2-bit distortion >= 1.5x Cold).

Two aggregation levels (both reported):
  eval-level: every (block, query, bits) record equally weighted.
  block-level: mean drop per block first, then quintile mean over blocks
              (de-correlates repeated queries on the same block).
"""
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RUN = "/home/san/e1/runs/20260926_1445_w22_quant_sensitivity"
OUT = "/home/san/e1/w22_sensitivity"
QLABELS = {0: "Q1 Cold", 1: "Q2", 2: "Q3", 3: "Q4", 4: "Q5 Hot"}


def main():
    recs = []
    for p in sorted(glob.glob(os.path.join(RUN, "artifacts/w22_quant_shard*.jsonl"))):
        for l in open(p):
            recs.append(json.loads(l))
    print(f"{len(recs)} eval records")

    bits_list = sorted({r["bits"] for r in recs})
    quints = sorted({r["quintile"] for r in recs})

    def agg(key_of):
        out = {}
        for q in quints:
            for b in bits_list:
                vals = [key_of(r) for r in recs
                        if r["quintile"] == q and r["bits"] == b]
                out[(q, b)] = (float(np.mean(vals)), float(np.std(vals)) / max(1, len(vals) ** .5),
                               len(vals))
        return out

    eval_level = agg(lambda r: r["drop"])
    eval_outf1 = agg(lambda r: r["out_f1"])

    # block-level: mean drop per block
    block_drop = {}
    for r in recs:
        block_drop.setdefault((r["video_id"], r["block"], r["quintile"]), {}).setdefault(r["bits"], []).append(r["drop"])
    block_level = {}
    for (vid, blk, q), d in block_drop.items():
        for b, vals in d.items():
            block_level.setdefault((q, b), []).append(float(np.mean(vals)))
    block_agg = {k: (float(np.mean(v)), float(np.std(v)) / max(1, len(v) ** .5), len(v))
                 for k, v in block_level.items()}

    # positive-share at 2-bit (noise check, cf T1's 36-46% negative LOO deltas)
    pos_share = {q: float(np.mean([r["drop"] > 0 for r in recs
                                   if r["quintile"] == q and r["bits"] == 2]))
                 for q in quints}

    table = {}
    for q in quints:
        row = {}
        for b in bits_list:
            m_eval, se_eval, n_eval = eval_level[(q, b)]
            m_blk, se_blk, n_blk = block_agg[(q, b)]
            row[f"bits{b}"] = {
                "drop_mean_eval": m_eval, "drop_se_eval": se_eval, "n_eval": n_eval,
                "drop_mean_block": m_blk, "drop_se_block": se_blk, "n_blocks": n_blk,
                "out_f1_mean": eval_outf1[(q, b)][0],
            }
        d2 = eval_level[(q, 2)][0]
        d4 = eval_level[(q, 4)][0]
        b2 = block_agg[(q, 2)][0]
        b4 = block_agg[(q, 4)][0]
        row["reduction_2to4_eval"] = d2 - d4
        row["reduction_2to4_block"] = b2 - b4
        row["ratio_4to2_eval"] = d2 / d4 if d4 else None
        table[QLABELS[q]] = row

    hot_cold = {
        "ratio_2bit_eval": eval_level[(4, 2)][0] / max(1e-9, eval_level[(0, 2)][0]),
        "ratio_2bit_block": block_agg[(4, 2)][0] / max(1e-9, block_agg[(0, 2)][0]),
        "ratio_4bit_block": block_agg[(4, 4)][0] / max(1e-9, block_agg[(0, 4)][0]),
    }
    verdict = "PASS" if hot_cold["ratio_2bit_block"] >= 1.5 else "FAIL"

    summary = {
        "n_evals": len(recs),
        "n_blocks": len(block_drop),
        "quintile_table": table,
        "hot_over_cold": hot_cold,
        "pos_share_2bit": {QLABELS[q]: pos_share[q] for q in quints},
        "gate": "Hot 2-bit distortion >= 1.5x Cold (block-level mean drop)",
        "verdict": verdict,
    }
    with open(os.path.join(OUT, "w22_sensitivity.json"), "w") as f:
        json.dump(summary, f, indent=1)

    # figure: per-quintile mean drop by bits (block-level), with SE bars
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    x = np.arange(len(quints))
    width = 0.25
    for i, b in enumerate(bits_list):
        ms = [block_agg[(q, b)][0] for q in quints]
        ses = [block_agg[(q, b)][1] for q in quints]
        ax.bar(x + (i - 1) * width, ms, width, yerr=ses,
               label=f"{b}-bit", capsize=3)
    ax.set_xticks(x, [QLABELS[q] for q in quints], rotation=20)
    ax.set_ylabel("mean token-F1 drop (block-level)")
    ax.set_title("per-block quantization distortion by quintile")
    ax.legend()
    ax = axes[1]
    bp_data = [[np.mean(d) for (vid, blk, q), d in block_drop.items() if q == qu
                and 2 in d] for qu in quints]
    ax.boxplot(bp_data, labels=[QLABELS[q] for q in quints], showfliers=False)
    ax.set_ylabel("per-block mean 2-bit drop")
    ax.set_title(f"2-bit: Hot/Cold = {hot_cold['ratio_2bit_block']:.2f}x "
                 f"(gate >= 1.5 -> {verdict})")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "w22_sensitivity.png"), dpi=150)
    print(json.dumps({"hot_over_cold": hot_cold, "verdict": verdict,
                      "table": table}, indent=1))


if __name__ == "__main__":
    main()
