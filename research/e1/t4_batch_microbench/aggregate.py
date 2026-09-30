"""Aggregate T4 metrics.jsonl -> summary tables, plots, cost.json."""
import json, glob, os, statistics as st

RUN = os.path.expanduser("~/e1/runs/20260926_1248_t4_batch_microbench")
OUT = os.path.expanduser("~/e1/t4_batch_microbench")

def load():
    recs = []
    for f in glob.glob(os.path.join(RUN, "metrics_*.jsonl")):
        with open(f) as fh:
            for line in fh:
                recs.append(json.loads(line))
    return recs

def summarize(recs):
    pts = {}
    for r in recs:
        pts.setdefault(r["tag"], []).append(r)
    rows = []
    for tag, rs in sorted(pts.items()):
        walls = [r["wall_s"] for r in rs]
        jobs = sum(r["jobs"] for r in rs)
        total_s = sum(walls)
        per_job = [r["wall_s"] / r["jobs"] for r in rs]
        rows.append({
            "tag": tag,
            "batch_size": rs[0]["batch_size"],
            "n_4f": rs[0]["n_4f"], "n_8f": rs[0]["n_8f"],
            "batches": len(rs), "jobs": jobs,
            "gpu_s_job_mean": total_s / jobs,
            "gpu_s_job_p50": st.median(per_job),
            "gpu_s_job_p95": sorted(per_job)[int(0.95 * (len(per_job) - 1))] if len(per_job) > 1 else per_job[0],
            "jobs_per_min": jobs / (total_s / 60),
            "batch_wall_mean": st.mean(walls),
            "peak_vram_mb": max(r["peak_vram_mb"] for r in rs),
            "gen_tokens_per_job": sum(r["gen_tokens_total"] for r in rs) / jobs,
            "gpu_s_per_1k_padded_tokens": total_s / (sum(
                r["jobs"] * (r["padded_input_tokens"]
                             + r["gen_tokens_total"] / r["jobs"]) for r in rs) / 1000.0),
        })
    return rows

def main():
    recs = load()
    rows = summarize(recs)
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(rows, f, indent=2)
    hdr = "%-10s %4s %6s %9s %9s %9s %10s %9s %8s" % (
        "point", "bs", "jobs", "gpu_s/job", "p50", "p95", "jobs/min", "peakVRAM", "gen_tok")
    print(hdr)
    for r in rows:
        print("%-10s %4d %6d %9.3f %9.3f %9.3f %10.2f %8.0fMB %8.0f" % (
            r["tag"], r["batch_size"], r["jobs"], r["gpu_s_job_mean"],
            r["gpu_s_job_p50"], r["gpu_s_job_p95"], r["jobs_per_min"],
            r["peak_vram_mb"], r["gen_tokens_per_job"]))
    # cost
    total_gpu_s = sum(r["jobs"] * r["gpu_s_job_mean"] for r in rows)
    cost = {"total_gpu_s_all_points": round(total_gpu_s, 1),
            "total_gpu_h": round(total_gpu_s / 3600, 3),
            "gpus_used": [1, 2, 3],
            "model_load_gpu_s_per_process": "~120 (3 processes)"}
    with open(os.path.join(RUN, "cost.json"), "w") as f:
        json.dump(cost, f, indent=2)
    print(json.dumps(cost))

if __name__ == "__main__":
    main()
