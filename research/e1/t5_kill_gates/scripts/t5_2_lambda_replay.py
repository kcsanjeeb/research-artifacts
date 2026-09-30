"""T5.2 — Does lambda oscillate under dual ascent?

Replay the LOGGED fleet-v1 arrival stream (metrics.jsonl from the fleet_v1
sweep cells, run 20260919_0100_fleet_v1_sweep) through a minimal SIMULATED
dual-ascent price controller. This is a measurement of controller dynamics on
the real arrival process, not a mechanism build.

Controller (textbook):
  admission:  admit arrival i at time t iff salience_i >= lambda_t
  spend_W:    sum of real_gpu_s of admitted arrivals in window W
  update:     lambda <- max(0, lambda + eta * (spend_W - B_W) / B_W)

Budget grounded in the sweep cell: 3 GPU lanes, virtual accel 2.4
(horizon 3600 vt-s / wall 1500 s) -> capacity 1.25 GPU-s per virtual-s.
B_W = target_util * capacity * W with target_util = 0.9, W = 60 vt-s.

PASS = visible sustained oscillation (limit cycle) in the lambda trace ->
build hysteresis M2. FAIL = lambda converges smoothly -> kill M2.

Quantitative oscillation criteria (post-transient = second half of horizon):
  - peak-to-peak amplitude of lambda >= 10% of its mean, AND
  - >= 6 sign changes of d(lambda) per 100 windows (persistent direction
    reversals, not a monotone approach).
"""
import glob
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.expanduser("~/e1/fleetmem"))
from make_run_dir import make_run_dir

SWEEP = os.path.expanduser("~/e1/runs/20260919_0100_fleet_v1_sweep")
CELLS = ["fcfs_N16", "fcfs_N24"]         # logged arrival streams (seed 0)
# fcfs_N16: 338/341 arrivals logged, complete to vt=3520 (backlog_end=3).
# fcfs_N24: truncated by wall-clock at vt=2726 (358 of 472 arrivals logged);
# replay restricted to the logged horizon (per-cell, see HORIZON override).
W = 60.0                                   # window, virtual seconds
N_LANES = 3
ACCEL = 2.4                                # from cell_summary (3600 vt-s / 1500 wall-s)
TARGET_UTIL = 0.9
ETA_GRID = [0.05, 0.2, 0.5, 1.0]
LAM0 = 0.0
AMP_CRIT = 0.10                            # peak-to-peak >= 10% of mean
SIGN_CRIT = 6.0                            # sign changes per 100 windows


def load_arrivals(cell):
    path = os.path.join(SWEEP, cell, "metrics.jsonl")
    arr = []
    for l in open(path):
        r = json.loads(l)
        arr.append({"t": r["arrival_vt"], "sal": r["salience"],
                    "cost": r["real_gpu_s"]})
    arr.sort(key=lambda a: a["t"])
    return arr


def run_controller(arr, eta, horizon):
    cap = N_LANES / ACCEL                    # GPU-s per virtual-s
    B_W = TARGET_UTIL * cap * W
    n_w = int(np.ceil(horizon / W))
    lam = LAM0
    lams, spends, admits = [], [], []
    wi = 0
    spend = 0.0
    n_admit = 0
    for a in arr:
        while a["t"] >= (wi + 1) * W and wi < n_w:
            lams.append(lam)
            spends.append(spend)
            admits.append(n_admit)
            lam = max(0.0, lam + eta * (spend - B_W) / B_W)
            wi += 1
            spend = 0.0
            n_admit = 0
        if wi >= n_w:
            break
        if a["sal"] >= lam:
            spend += a["cost"]
            n_admit += 1
    while wi < n_w:
        lams.append(lam)
        spends.append(spend)
        admits.append(n_admit)
        lam = max(0.0, lam + eta * (spend - B_W) / B_W)
        wi += 1
        spend = 0.0
        n_admit = 0
    return np.array(lams), np.array(spends), np.array(admits), B_W


def oscillation_stats(lams):
    half = len(lams) // 2
    seg = lams[half:]
    mean = float(np.mean(seg))
    p2p = float(np.max(seg) - np.min(seg))
    amp_rel = p2p / mean if mean > 0 else float("inf") if p2p > 0 else 0.0
    d = np.diff(seg)
    signs = np.sign(d[np.abs(d) > 1e-12])
    n_sc = int(np.sum(signs[1:] * signs[:-1] < 0)) if len(signs) > 1 else 0
    sc_per_100 = n_sc / len(seg) * 100.0
    osc = (amp_rel >= AMP_CRIT) and (sc_per_100 >= SIGN_CRIT)
    return {"mean": mean, "p2p": p2p, "amp_rel": amp_rel,
            "sign_changes_per_100w": sc_per_100, "oscillates": bool(osc)}


def main():
    results = {}
    fig, axes = plt.subplots(len(CELLS), 1, figsize=(11, 4 * len(CELLS)),
                             sharex=True)
    if len(CELLS) == 1:
        axes = [axes]
    for ax, cell in zip(axes, CELLS):
        arr = load_arrivals(cell)
        horizon = float(np.ceil(max(a["t"] for a in arr) / W) * W)
        results[cell] = {"n_arrivals": len(arr), "horizon_vt": horizon,
                         "demand_gpu_s_per_vt_s": float(sum(a["cost"] for a in arr) / horizon),
                         "salience_min": min(a["sal"] for a in arr),
                         "salience_p05": float(np.percentile([a["sal"] for a in arr], 5)),
                         "etas": {}}
        for eta in ETA_GRID:
            lams, spends, admits, B_W = run_controller(arr, eta, horizon)
            st = oscillation_stats(lams)
            st["B_W"] = B_W
            st["mean_spend_frac_budget"] = float(np.mean(spends[len(spends)//2:]) / B_W)
            results[cell]["etas"][str(eta)] = st
            ax.plot(np.arange(len(lams)) * W, lams, label=f"eta={eta}")
        ax.set_title(f"{cell}: lambda trace (dual ascent, W={W:.0f}s)")
        ax.set_ylabel("lambda")
        ax.legend()
        ax.grid(alpha=0.3)
    axes[-1].set_xlabel("virtual time (s)")
    fig.tight_layout()

    rd = make_run_dir("t5_2_lambda_replay", {
        "task": "T5.2", "arrival_source": SWEEP, "cells": CELLS,
        "controller": "admit iff salience >= lambda; "
                      "lambda <- max(0, lambda + eta*(spend_W - B_W)/B_W)",
        "W_virtual_s": W, "n_lanes": N_LANES, "accel": ACCEL,
        "target_util": TARGET_UTIL, "eta_grid": ETA_GRID, "lambda0": LAM0,
        "osc_criteria": {"amp_rel_min": AMP_CRIT,
                         "sign_changes_per_100w_min": SIGN_CRIT,
                         "segment": "second half of horizon"}})
    fig.savefig(os.path.join(rd, "artifacts", "lambda_traces.png"), dpi=120)

    any_osc = any(st["oscillates"] for c in results.values()
                  for st in c["etas"].values())
    verdict = "PASS (lambda oscillates -> build hysteresis M2)" if any_osc \
        else "FAIL (lambda converges smoothly -> kill M2)"
    metrics = {"results": results, "any_eta_oscillates": any_osc,
               "verdict": verdict}
    with open(os.path.join(rd, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    lines = ["# T5.2 — Does lambda oscillate under dual ascent?", "",
             f"**Verdict: {verdict}**", "",
             "Arrival streams replayed verbatim from the logged fleet-v1 sweep "
             f"({SWEEP}), cells {', '.join(CELLS)}. Controller: admit iff "
             "salience >= lambda; lambda <- max(0, lambda + eta*(spend_W - "
             "B_W)/B_W) per 60 vt-s window; B_W = 0.9 * (3 lanes / 2.4 accel) "
             "* 60 = 67.5 GPU-s. Lambda is a price in salience units "
             "(saliences here are peak tier-1 scores, strongly concentrated "
             ">= 0.99 — see salience_p05 per cell).", "",
             "Oscillation criteria (post-transient second half): peak-to-peak "
             ">= 10% of mean AND >= 6 direction reversals per 100 windows.", ""]
    for cell, c in results.items():
        lines.append(f"## {cell} ({c['n_arrivals']} arrivals over "
                     f"{c['horizon_vt']:.0f} vt-s, demand "
                     f"{c['demand_gpu_s_per_vt_s']:.2f} GPU-s/vt-s vs capacity "
                     f"{N_LANES / ACCEL:.2f}, "
                     f"salience min={c['salience_min']:.4f}, "
                     f"p05={c['salience_p05']:.4f})")
        lines.append("")
        lines.append("| eta | mean lam | p2p | amp/mean | reversals/100w | "
                     "spend/budget | oscillates |")
        lines.append("|---|---|---|---|---|---|---|")
        for eta, st in c["etas"].items():
            lines.append(f"| {eta} | {st['mean']:.4f} | {st['p2p']:.4f} | "
                         f"{st['amp_rel']:.3f} | {st['sign_changes_per_100w']:.1f} | "
                         f"{st['mean_spend_frac_budget']:.3f} | {st['oscillates']} |")
        lines.append("")
    lines.append("Trace plot: artifacts/lambda_traces.png")
    lines.append("")
    lines.append("Note: spend/budget != 1.0 in sustained oscillation is "
                 "expected — the controller cannot hold spend exactly at "
                 "budget when admission is a hard threshold on a concentrated "
                 "salience distribution; it overshoots and corrects, which is "
                 "exactly the limit cycle M2's hysteresis is meant to damp.")
    with open(os.path.join(rd, "REPORT.md"), "w") as f:
        f.write("\n".join(lines))
    print(rd)
    print(json.dumps({c: {e: {k: round(v, 4) if isinstance(v, float) else v
                              for k, v in st.items()}
                          for e, st in r["etas"].items()}
                      for c, r in results.items()}, indent=2))
    print(verdict)


if __name__ == "__main__":
    main()
