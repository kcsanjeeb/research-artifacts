"""Plots for T4: throughput-vs-batch, VRAM-vs-batch, mixed-vs-homogeneous."""
import json, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.expanduser("~/e1/t4_batch_microbench")
rows = json.load(open(os.path.join(OUT, "summary.json")))
by = {r["tag"]: r for r in rows}
BS = [1, 2, 4, 8, 16]

def series(prefix):
    gsj, jpm, vram = [], [], []
    for b in BS:
        r = by["%s_b%d" % (prefix, b)]
        gsj.append(r["gpu_s_job_mean"])
        jpm.append(r["jobs_per_min"])
        vram.append(r["peak_vram_mb"] / 1024)
    return gsj, jpm, vram

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

ax = axes[0]
for prefix, label, color in (("8f", "8 frames/job", "tab:blue"),
                             ("4f", "4 frames/job", "tab:orange")):
    gsj, jpm, _ = series(prefix)
    ax.plot(BS, jpm, "o-", color=color, label=label)
ax.set_xscale("log", base=2)
ax.set_xticks(BS)
ax.set_xticklabels(BS)
ax.set_xlabel("batch size (fixed)")
ax.set_ylabel("throughput (jobs/min)")
ax.set_title("Throughput vs batch size")
ax.grid(alpha=0.3)
ax.legend()

ax = axes[1]
for prefix, label, color in (("8f", "8 frames/job", "tab:blue"),
                             ("4f", "4 frames/job", "tab:orange"),
                             ("mixed", "mixed 4+8 (b8,b16)", "tab:green")):
    bs_list = BS if prefix != "mixed" else [8, 16]
    vram = [by["%s_b%d" % (prefix, b)]["peak_vram_mb"] / 1024 for b in bs_list]
    ax.plot(bs_list, vram, "o-", color=color, label=label)
ax.set_xscale("log", base=2)
ax.set_xticks(BS)
ax.set_xticklabels(BS)
ax.set_xlabel("batch size (fixed)")
ax.set_ylabel("peak VRAM (GiB)")
ax.set_title("Peak VRAM vs batch size")
ax.grid(alpha=0.3)
ax.legend()

ax = axes[2]
# mixed vs homogeneous at same total job count (b8 and b16)
cats = ["b8", "b16"]
homo8 = [by["8f_b8"]["gpu_s_job_mean"], by["8f_b16"]["gpu_s_job_mean"]]
homo4 = [by["4f_b8"]["gpu_s_job_mean"], by["4f_b16"]["gpu_s_job_mean"]]
bucket = [(homo8[i] + homo4[i]) / 2 for i in range(2)]
mixed = [by["mixed_b8"]["gpu_s_job_mean"], by["mixed_b16"]["gpu_s_job_mean"]]
x = range(2)
w = 0.25
ax.bar([i - w for i in x], homo8, w, label="homogeneous 8-frame", color="tab:blue")
ax.bar([i for i in x], homo4, w, label="homogeneous 4-frame", color="tab:orange")
ax.bar([i + w for i in x], bucket, w, label="perfect bucketing (avg)", color="tab:gray")
ax.bar([i + 2 * w for i in x], mixed, w, label="mixed batch (padded)", color="tab:green")
ax.set_xticks([i + w / 2 for i in x])
ax.set_xticklabels(cats)
ax.set_ylabel("GPU-s / job")
ax.set_title("Mixed vs homogeneous (same total jobs)")
ax.grid(alpha=0.3, axis="y")
ax.legend(fontsize=8)

fig.tight_layout()
fig.savefig(os.path.join(OUT, "t4_plots.png"), dpi=150)
print("wrote t4_plots.png")
