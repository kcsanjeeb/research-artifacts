"""Generate E0 figures from the three result JSONs into results/figures/."""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common  # noqa: E402


def fig_capacity(cap, figdir):
    m = cap["models"]["HolmesVAU-2B"]
    sweep = [r for r in m["batch_sweep"] if not r.get("oom")]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    b = [r["batch"] for r in sweep]
    tp = [r["clips_per_s"] for r in sweep]
    ax1.plot(b, tp, "o-", label="throughput (clips/s)")
    ax1.set_xlabel("batch size (clips)")
    ax1.set_ylabel("clips/s")
    ax1.set_xscale("log", base=2)
    ax2 = ax1.twinx()
    ax2.plot(b, [r["p95_ms"] for r in sweep], "s--", color="tab:red",
             label="p95 latency (ms)")
    ax2.set_ylabel("p95 latency (ms)")
    ax1.set_title("E0.1 HolmesVAU-2B on 1x V100 (fp16, eager)")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "capacity_batch_sweep.png"), dpi=150)
    plt.close(fig)

    reps = [r for r in m.get("replica_sweep", []) if not r.get("failed")]
    if reps:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot([r["replicas"] for r in reps],
                [r["agg_clips_per_s"] for r in reps], "o-")
        ax.set_xlabel("replicas per GPU")
        ax.set_ylabel("aggregate clips/s")
        ax.set_title("E0.2 replica scaling (1x V100)")
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "capacity_replicas.png"), dpi=150)
        plt.close(fig)


def fig_demand(dem, figdir):
    curve = dem["curve"]
    periods = sorted(set(c["clip_period_s"] for c in curve))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for p in periods:
        cs = sorted([c for c in curve if c["clip_period_s"] == p],
                    key=lambda c: c["N"])
        ax.plot([c["N"] for c in cs], [c["rho_p95"] for c in cs], "o-",
                label=f"p95, period={p}s")
        ax.plot([c["N"] for c in cs], [c["rho_mean"] for c in cs], "o:",
                alpha=0.5, label=f"mean, period={p}s")
    ax.axhline(1.0, color="k", ls="--", lw=1, label="rho = 1 (capacity)")
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("N streams")
    ax.set_ylabel("load factor rho = demand / CAPACITY")
    ax.set_title("E0.2 escalation demand vs capacity")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "demand_load_factor.png"), dpi=150)
    plt.close(fig)


def fig_redundancy(red, figdir):
    alphas = red["alphas"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for a, r in alphas.items():
        curve = [c for c in r["curve"] if c["label_agreement"] is not None]
        axes[0].plot([c["theta"] for c in curve],
                     [c["hit_rate"] for c in curve], label=f"alpha={a}")
        axes[1].plot([c["theta"] for c in curve],
                     [c["label_agreement"] for c in curve], label=f"alpha={a}")
    axes[0].set_xlabel("theta")
    axes[0].set_ylabel("causal hit rate")
    axes[0].legend(fontsize=8)
    axes[1].axhline(0.95, color="k", ls="--", lw=1)
    axes[1].set_xlabel("theta")
    axes[1].set_ylabel("label agreement")
    axes[1].set_ylim(0.5, 1.01)
    fig.suptitle("E0.3 safe redundancy (causal NN)")
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, "redundancy_curves.png"), dpi=150)
    plt.close(fig)

    op = red.get("operating_point")
    if op and op.get("by_type"):
        fig, ax = plt.subplots(figsize=(5, 4))
        types = ["T1", "T2", "T3"]
        ax.bar(types, [op["by_type"].get(t, 0) for t in types])
        ax.set_ylabel("hit rate (fraction of all clips)")
        ax.set_title(f"safe hit rate by type @ theta={op['theta']}")
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "redundancy_by_type.png"), dpi=150)
        plt.close(fig)


def main():
    figdir = os.path.join(common.RESULTS_DIR, "figures")
    os.makedirs(figdir, exist_ok=True)
    cap_p = os.path.join(common.RESULTS_DIR, "capacity.json")
    dem_p = os.path.join(common.RESULTS_DIR, "demand.json")
    red_p = os.path.join(common.RESULTS_DIR, "redundancy.json")
    if os.path.exists(cap_p):
        fig_capacity(json.load(open(cap_p)), figdir)
    if os.path.exists(dem_p):
        fig_demand(json.load(open(dem_p)), figdir)
    if os.path.exists(red_p):
        fig_redundancy(json.load(open(red_p)), figdir)
    print("figures written to", figdir)


if __name__ == "__main__":
    main()
