#!/usr/bin/env python3
"""W2.4 — Is there chain structure in retrieval to exploit?

From t1_rekv_log.jsonl (per-query top-64 retrieved block indices, ReKV-0.5B,
RVS-Ego): measure temporal clustering of retrieved blocks — mean run length of
consecutive block indices and number of distinct contiguous segments — and
compare against a uniform-random null over the same block count drawn from the
same video's block range.

PASS = retrieved blocks form substantially longer contiguous runs than null.
FAIL = scattered like null.
"""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "/home/san/e1/runs/20260925_2237_t1_rekv_sharded/artifacts/t1_rekv_log.jsonl"
OUT_JSON = "/home/san/e1/w24_chain_structure/chain_structure.json"
OUT_FIG = "/home/san/e1/w24_chain_structure/chain_structure.png"
RNG_SEED = 20260926
N_NULL_PER_QUERY = 500


def run_stats(blocks):
    """Given sorted unique block indices: (n_segments, mean_run_len, max_run, coverage_len).

    A run = maximal set of consecutive integers (gap 1).
    coverage_len = span covered counting only in-run indices / total retrieved.
    """
    b = np.asarray(sorted(set(blocks)), dtype=np.int64)
    n = len(b)
    if n == 0:
        return 0, 0.0, 0, 0.0
    gaps = np.diff(b)
    n_segments = int(1 + np.sum(gaps > 1))
    run_ids = np.concatenate([[0], np.cumsum(gaps > 1)])
    run_lengths = np.bincount(run_ids)
    mean_run = n / n_segments
    max_run = int(run_lengths.max())
    in_run = int(run_lengths[run_lengths > 1].sum()) if n_segments < n else n
    coverage = in_run / n
    return n_segments, mean_run, max_run, coverage


def main():
    rng = np.random.default_rng(RNG_SEED)
    queries = [json.loads(l) for l in open(LOG)]
    rows = []
    for q in queries:
        blocks = q["retrieved_blocks"]
        nb = q["n_blocks"]
        k = len(blocks)
        ns, mrl, mr, cov = run_stats(blocks)
        row = {
            "video_id": q["video_id"], "question": q["question"], "k": k,
            "n_blocks": nb, "n_segments": ns, "mean_run_len": mrl,
            "max_run": mr, "frac_in_runs_ge2": cov,
        }
        # null: uniform sample of k blocks from [0, nb), N_NULL_PER_QUERY draws
        null_ns, null_mrl, null_mr, null_cov = [], [], [], []
        for _ in range(N_NULL_PER_QUERY):
            samp = rng.choice(nb, size=k, replace=False)
            a, b, c, d = run_stats(samp)
            null_ns.append(a); null_mrl.append(b); null_mr.append(c); null_cov.append(d)
        row["null_n_segments_mean"] = float(np.mean(null_ns))
        row["null_n_segments_std"] = float(np.std(null_ns))
        row["null_mean_run_len_mean"] = float(np.mean(null_mrl))
        row["null_mean_run_len_std"] = float(np.std(null_mrl))
        row["null_max_run_mean"] = float(np.mean(null_mr))
        row["null_frac_in_runs_ge2_mean"] = float(np.mean(null_cov))
        row["z_mean_run_len"] = (mrl - np.mean(null_mrl)) / (np.std(null_mrl) + 1e-12)
        rows.append(row)

    def pooled(key):
        return float(np.mean([r[key] for r in rows]))

    summary = {
        "n_queries": len(rows),
        "n_videos": len({r["video_id"] for r in rows}),
        "k_min": min(r["k"] for r in rows),
        "k_max": max(r["k"] for r in rows),
        "actual": {
            "mean_run_len": pooled("mean_run_len"),
            "n_segments": pooled("n_segments"),
            "max_run": pooled("max_run"),
            "frac_in_runs_ge2": pooled("frac_in_runs_ge2"),
        },
        "null": {
            "mean_run_len": pooled("null_mean_run_len_mean"),
            "n_segments": pooled("null_n_segments_mean"),
            "max_run": pooled("null_max_run_mean"),
            "frac_in_runs_ge2": pooled("null_frac_in_runs_ge2_mean"),
        },
        "ratio_mean_run_len": pooled("mean_run_len") / pooled("null_mean_run_len_mean"),
        "ratio_n_segments": pooled("n_segments") / pooled("null_n_segments_mean"),
        "queries_longer_than_null_95pct": float(np.mean(
            [r["mean_run_len"] > r["null_mean_run_len_mean"] + 1.96 * r["null_mean_run_len_std"]
             for r in rows])),
        "queries_shorter_than_null_95pct": float(np.mean(
            [r["mean_run_len"] < r["null_mean_run_len_mean"] - 1.96 * r["null_mean_run_len_std"]
             for r in rows])),
        "mean_z_mean_run_len": pooled("z_mean_run_len"),
    }

    with open(OUT_JSON, "w") as f:
        json.dump({"summary": summary, "per_query": rows}, f, indent=1)

    # figures
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    ax = axes[0]
    ax.scatter([r["null_mean_run_len_mean"] for r in rows],
               [r["mean_run_len"] for r in rows], alpha=0.5)
    lim = max(max(r["mean_run_len"] for r in rows),
              max(r["null_mean_run_len_mean"] for r in rows)) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlabel("null mean run length"); ax.set_ylabel("actual mean run length")
    ax.set_title(f"run length: actual {summary['actual']['mean_run_len']:.2f} vs "
                 f"null {summary['null']['mean_run_len']:.2f} "
                 f"(x{summary['ratio_mean_run_len']:.1f})")
    ax = axes[1]
    ax.scatter([r["null_n_segments_mean"] for r in rows],
               [r["n_segments"] for r in rows], alpha=0.5)
    lim = max(max(r["n_segments"] for r in rows),
              max(r["null_n_segments_mean"] for r in rows)) * 1.1
    ax.plot([0, lim], [0, lim], "k--", lw=1)
    ax.set_xlabel("null n segments"); ax.set_ylabel("actual n segments")
    ax.set_title(f"segments: actual {summary['actual']['n_segments']:.1f} vs "
                 f"null {summary['null']['n_segments']:.1f}")
    ax = axes[2]
    ax.hist([r["z_mean_run_len"] for r in rows], bins=25)
    ax.axvline(1.96, color="r", ls="--", lw=1)
    ax.axvline(-1.96, color="r", ls="--", lw=1)
    ax.set_xlabel("z (mean run length vs null)")
    ax.set_title(f"z>1.96 for {summary['queries_longer_than_null_95pct']*100:.0f}% of queries")
    fig.tight_layout()
    fig.savefig(OUT_FIG, dpi=150)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
