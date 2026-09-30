#!/usr/bin/env python3
"""W4.3 — window coverage vs planner gain across the three benchmarks.

Coverage definition = W3.3 (w33_ceiling.py): overlap of retrieved frame-blocks
with the question's evidence window [start,end], PLUS the local-cache addendum
(the last n_local=15000/196=76 tokens = 76 frames resident in the local cache
at answer time count as context-present), over the streamed history up to the
question — capped at 1. Measurement is against the gold window; allocation in
W4.2 must never use it.

Block->time: RVS 4 s/block (0.25 fps effective, double-decimated npy path,
verified in W3.3); StreamingBench / OVO 2 s/block (0.5 fps mp4 path,
--sample_fps 0.5 in run_arm.sh; max retrieved block ~ duration/2 verified).

Two coverage populations per benchmark:
  (a) ALL questions of the system writetime arm (the deployment reality);
  (b) the all-wrong subset (partition from the 4 oracle arms), using the
      best-fixed-policy oracle arm's retrieval — W3.3's convention, so RVS/SB
      numbers reproduce w33_allwrong_breakdown.json (RVS mean 0.209 /
      median 0.126, n=152; SB 1.0/1.0, n=42).

Outputs: w43_coverage.json, gain_vs_coverage.png, printed 4-column table.
"""
import json, csv, os, collections
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/data3/zhuotaotian2_e2"
B = ROOT + "/runs/20260928_0315_e3_phase2_compB/"
XB = ROOT + "/runs/20260928_2105_e3_phase2_xbench/"
OVO = ROOT + "/runs/20260929_1215_e3_phase2_ovo/"
RVS_ANNO = ROOT + "/data/rvs_annotations/test_qa_ego4d_realtime.json"
OUT = ROOT + "/deliverables/e3_phase2/analysis/w43_coverage/"
os.makedirs(OUT, exist_ok=True)

POLS = ["seg", "frame", "patch", "default"]
N_LOCAL = 15000 // 196          # 76 most-recent frames stay in the local cache


def blocks_overlap(blocks, start, end, spb):
    cov = 0.0
    for b in blocks:
        bs, be = b * spb, (b + 1) * spb
        cov += max(0.0, min(be, end) - max(bs, start))
    return cov


def load_csv_blocks(path):
    rows = list(csv.DictReader(open(path)))
    return rows, [[int(x) for x in r["retrieved_blocks"].split(";")] if r["retrieved_blocks"] else []
                  for r in rows]


def coverage(blocks, start, end, spb, frontier, dur=None):
    """W3.3 coverage: retrieved-block overlap + local-cache addendum, capped at 1."""
    win = max(1e-9, end - start)
    fr = min(frontier, dur) if dur else frontier
    local_lo = max(0.0, fr - N_LOCAL * spb)
    ov = blocks_overlap(blocks, start, end, spb)
    ov += max(0.0, min(end, fr) - max(start, local_lo))
    return min(ov, win) / win


def stats(v):
    v = list(v)
    return dict(n=len(v), mean=float(np.mean(v)), median=float(np.median(v)),
                frac_zero=float(np.mean([x <= 0 for x in v])),
                frac_ge_half=float(np.mean([x >= 0.5 for x in v])))


# ---------------- StreamingBench ----------------
def load_xb_like(run, anno_f, judged_sys_f):
    anno = json.load(open(run + anno_f))
    flat = [(v["video_id"], c) for v in anno for c in v["conversations"]]
    rows, blocks = load_csv_blocks(run + "answers_7b_writetime/1_0.csv")
    recs = json.load(open(run + judged_sys_f))["records"]
    assert len(rows) == len(flat) == len(recs), (len(rows), len(flat), len(recs))
    assert [r["video_id"] for r in rows] == [r["video_id"] for r in recs]
    return flat, blocks, recs, {v["video_id"]: v["duration"] for v in anno}


def oracle_partition(run, n, suffix="_letter"):
    mats = {}
    for p in POLS:
        recs = json.load(open(run + f"judged_oracle_{p}{suffix}.json"))["records"]
        assert len(recs) == n, (p, len(recs))
        mats[p] = [1 if r["correct"] else 0 for r in recs]
    best = max(POLS, key=lambda p: sum(mats[p]))
    k = [sum(mats[p][i] for p in POLS) for i in range(n)]
    return k, best


def bench(run, anno_f, judged_sys_f, spb, name, oracle_retr_dir=None):
    flat, sys_blocks, recs, durs = load_xb_like(run, anno_f, judged_sys_f)
    n = len(flat)
    # (a) all questions, system arm retrieval
    cov_all = [coverage(sys_blocks[i], flat[i][1]["start_time"], flat[i][1]["end_time"],
                        spb, frontier=flat[i][1]["end_time"]) for i in range(n)]
    # (b) all-wrong subset, oracle best-policy retrieval (W3.3 convention)
    k, best = oracle_partition(run, n)
    aw = [i for i in range(n) if k[i] == 0]
    if oracle_retr_dir is None:
        oracle_retr_dir = run + f"oracle_{best}/1_0.csv"
    orows, oblocks = load_csv_blocks(oracle_retr_dir)
    assert len(orows) == n, (len(orows), n)
    cov_aw = [coverage(oblocks[i], flat[i][1]["start_time"], flat[i][1]["end_time"],
                       spb, frontier=flat[i][1]["end_time"]) for i in aw]
    return dict(name=name, n=n, best_policy=best, all_wrong_idx=aw,
                all_questions=stats(cov_all), all_wrong=stats(cov_aw),
                window_s=stats([f[1]["end_time"] - f[1]["start_time"] for f in flat]))


xb = bench(XB, "xbench_anno.json", "judged_answers_7b_writetime_letter.json", 2.0, "StreamingBench")
ovo = bench(OVO, "ovo_anno.json", "judged_answers_7b_writetime_letter.json", 2.0, "OVO-Bench")

# ---------------- RVS ----------------
# System arm covers all 1465 questions / 9 videos; restrict to the W3.3 oracle
# subset (2 videos, 316 occurrence-indexed records). Match per-video positionally
# (both orderings iterate the same annotation json per video) with text asserts.
anno = {(a["video_name"], a["question"]): a for a in json.load(open(RVS_ANNO))}
sub = json.load(open(B + "oracle_subset_anno.json"))
sub_videos = [v["video_id"] for v in sub]
sub_conv = {v["video_id"]: v["conversations"] for v in sub}
srows, sblocks = load_csv_blocks(B + "answers_7b_b_writetime/1_0.csv")
sys_by_video = collections.defaultdict(list)
for i, r in enumerate(srows):
    if r["video_id"] in set(sub_videos):
        sys_by_video[r["video_id"]].append(i)
dur_by_video = {v["video_id"]: v["duration"] for v in json.load(open(B + "ego4d_oe_npy.json"))}

SPB_RVS = 4.0
cov_all_rvs = []
flat_rvs = []
for v in sub_videos:
    convs = sub_conv[v]
    idxs = sys_by_video[v]
    assert len(idxs) == len(convs), (v, len(idxs), len(convs))
    for pos, (c, i) in enumerate(zip(convs, idxs)):
        assert srows[i]["question"] == c["question"], (v, pos)
        a = anno[(v, c["question"])]
        start, end = a["start_time"], a["end_time"]
        flat_rvs.append((v, c, i))
        cov_all_rvs.append(coverage(sblocks[i], start, end, SPB_RVS,
                                    frontier=2 * end, dur=dur_by_video[v]))
assert len(cov_all_rvs) == 316

# all-wrong partition (W3.3): oracle arm judged files, patch retrieval
mats_r = {}
for p in POLS:
    recs = json.load(open(B + f"judged_oracle_{p}_72b.json"))["judged"]
    assert len(recs) == 316
    mats_r[p] = [1 if r["judge_pred"] == "yes" else 0 for r in recs]
k_r = [sum(mats_r[p][i] for p in POLS) for i in range(316)]
aw_r = [i for i in range(316) if k_r[i] == 0]
prows, pblocks = load_csv_blocks(B + "oracle_patch/1_0.csv")
assert len(prows) == 316
cov_aw_rvs = []
for i in aw_r:
    v, c, _ = flat_rvs[i]
    a = anno[(v, c["question"])]
    cov_aw_rvs.append(coverage(pblocks[i], a["start_time"], a["end_time"], SPB_RVS,
                               frontier=2 * a["end_time"], dur=dur_by_video[v]))
# W3.3 reproduction check
w33 = json.load(open(ROOT + "/deliverables/e3_phase2/analysis/w33_ceiling/w33_allwrong_breakdown.json"))
w33_rvs = w33["rvs"]["overlap_ratio_all_wrong"]
repro = dict(mean_ref=w33_rvs["mean"], median_ref=w33_rvs["median"],
             mean_here=float(np.mean(cov_aw_rvs)), median_here=float(np.median(cov_aw_rvs)))

rvs = dict(name="RVS-Ego (oracle subset)", n=316, best_policy="patch", all_wrong_idx=aw_r,
           all_questions=stats(cov_all_rvs),
           all_wrong=stats(cov_aw_rvs),
           window_s=stats([anno[(v, c["question"])]["end_time"] - anno[(v, c["question"])]["start_time"]
                           for v, c, _ in flat_rvs]),
           w33_reproduction=repro)

# ---------------- planner gain & system acc (w32_results.json / ORACLE_OVO.md) ----------------
w32 = json.load(open(ROOT + "/deliverables/e3_phase2/analysis/w32_oracle_null/w32_results.json"))
table = []
rows_t = [
    # gap_system_minus_floor_pt in w32_results.json is ALREADY in percentage points
    ("RVS-Ego", rvs, round(100 * w32["rvs_subset"]["system_acc"], 1),
     round(w32["rvs_subset"]["random_floor"]["gap_system_minus_floor_pt"], 1)),
    ("StreamingBench", xb, round(100 * w32["streamingbench_rt"]["system_acc"], 1),
     round(w32["streamingbench_rt"]["random_floor"]["gap_system_minus_floor_pt"], 1)),
    # OVO numbers from ORACLE_OVO.md (w32_unified_null reports the same test but
    # the table below needs the random floor, which lives in the run's verdict)
    ("OVO-Bench", ovo, 39.4, 39.4 - 38.5),
]
for name, b, acc, gain in rows_t:
    table.append(dict(benchmark=name, system_acc=round(acc, 1),
                      planner_gain_over_random_floor_pt=round(gain, 1),
                      coverage_all_q_mean=round(b["all_questions"]["mean"], 3),
                      coverage_all_q_median=round(b["all_questions"]["median"], 3),
                      n_all_q=b["all_questions"]["n"],
                      coverage_allwrong_mean=round(b["all_wrong"]["mean"], 3),
                      coverage_allwrong_median=round(b["all_wrong"]["median"], 3),
                      n_all_wrong=b["all_wrong"]["n"],
                      mean_window_s=round(b["window_s"]["mean"], 1)))

out = dict(definition=__doc__.split("\n\n")[0].strip(),
           proxies=dict(block_time=dict(rvs_s=4.0, streamingbench_s=2.0, ovo_s=2.0),
                        n_local_frames=N_LOCAL,
                        local_cache_s=dict(rvs=304.0, streamingbench=152.0, ovo=152.0),
                        frontier=dict(rvs="min(2*end, video duration) — 2x-end encode quirk (W3.3)",
                                      streamingbench="end (cumulative encode)",
                                      ovo="end (cumulative encode)")),
           benchmarks={b["name"]: {k: v for k, v in b.items() if k != "all_wrong_idx"}
                       for b in (rvs, xb, ovo)},
           sources=dict(rvs="runs/20260928_0315_e3_phase2_compB (system arm answers_7b_b_writetime "
                            "retrieval + oracle_patch for the all-wrong subset)",
                        streamingbench="runs/20260928_2105_e3_phase2_xbench",
                        ovo="runs/20260929_1215_e3_phase2_ovo"),
           table=table)
json.dump(out, open(OUT + "w43_coverage.json", "w"), indent=1)

# ---------------- plot ----------------
fig, ax = plt.subplots(figsize=(7, 5))
markers = {"RVS-Ego": ("o", "#b03a2e"), "StreamingBench": ("s", "#1a5276"),
           "OVO-Bench": ("^", "#148f77")}
for name, b, acc, gain in rows_t:
    m, c = markers[name]
    aq, aw = b["all_questions"], b["all_wrong"]
    ax.plot(aq["mean"], gain, m, color=c, ms=11, zorder=3)
    ax.plot(aw["mean"], gain, m, color=c, ms=11, mfc="white", zorder=3)
    ax.plot([aq["mean"], aw["mean"]], [gain, gain], "-", color=c, lw=1, alpha=0.5, zorder=2)
    off = (14, -6) if name != "StreamingBench" else (-215, -45)
    ax.annotate(f"{name}\nall q: cov={aq['mean']:.2f} (n={aq['n']})\n"
                f"all-wrong: cov={aw['mean']:.2f} (n={aw['n']})",
                (aq["mean"], gain), textcoords="offset points", xytext=off,
                fontsize=8, color=c)
ax.axhline(0, color="gray", lw=0.8, ls="--")
ax.set_xlabel("mean window coverage (retrieved blocks + local cache, W3.3 definition)")
ax.set_ylabel("planner gain over random-selector floor (pt)")
ax.set_title("W4.3 — planner gain vs window coverage (solid: all questions, system arm;\n"
             "hollow: all-wrong subset, oracle best-policy retrieval)")
ax.set_xlim(-0.05, 1.15)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT + "gain_vs_coverage.png", dpi=160)

print(json.dumps(table, indent=1))
print("\nRVS W3.3 reproduction (all-wrong, oracle patch):", json.dumps(repro))
print("\nwrote", OUT + "w43_coverage.json", "and gain_vs_coverage.png")
