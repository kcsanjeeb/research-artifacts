#!/usr/bin/env python3
"""W3.3 — ceiling analysis: partition all-right / all-wrong / mixed, then
characterize the all-wrong set with documented proxies.

RVS subset (n=316 occurrence-indexed):
  window [start,end] (s) from data/rvs_annotations/test_qa_ego4d_realtime.json
  block->time: sampled frame b covers [4b, 4b+4) s (effective 0.25 fps: npy
    decoded at 0.5 fps then linspace-resampled at 0.5 again; verified against
    max block id ~ duration/4 and store n_segments=225 for 3600 s).
  stream frontier at question time: end*0.5 sampled frames = 2*end seconds
    (rekv_stream_vqa encodes until temporal_windows[-1] = end*sample_fps),
    so the annotated window is ALWAYS within the streamed range on RVS.
  store retention: per-segment budget (TAX*784), NO cross-segment eviction
    (decaf_b.py) -> every streamed segment retains >=1 grain at tax0.5;
    true zero-coverage is structurally impossible. Coverage proxy: window
    segment ids vs the union of seg_ids logged in the arm's commit_log
    (seg_scores top-32 + commit lists) across all questions of that video.

  Categories (best fixed policy = patch, 45.6):
    (i)  retrieval failure: retrieved blocks ∩ window = ∅
    (ii) store-coverage (logged) failure: not (i), and none of the window's
         segments appear in any logged store interaction for that video
    (iii) model-capability failure: everything else (context present in
         retrieval or logs, all 4 policies wrong)
  Overlap ratio = |union(retrieved blocks) ∩ [start,end]| / (end-start).

StreamingBench (n=180):
  block->time: [2b, 2b+2) (0.5 fps mp4 path); frontier = end exactly.
  No seg-level store logging on xbench -> only retrieval-failure vs
  model-capability split (coverage not assessable from logs).

Outputs: w33_partition.json, w33_allwrong_breakdown.json, w33_examples.json,
         w33_mixed_by_type.json
"""
import json, csv, os, re, collections
import numpy as np

B = "/data3/zhuotaotian2_e2/runs/20260928_0315_e3_phase2_compB/"
XB = "/data3/zhuotaotian2_e2/runs/20260928_2105_e3_phase2_xbench/"
RVS_ANNO = "/data3/zhuotaotian2_e2/data/rvs_annotations/test_qa_ego4d_realtime.json"
OUT = "/data3/zhuotaotian2_e2/deliverables/e3_phase2/analysis/w33_ceiling/"
os.makedirs(OUT, exist_ok=True)

POLS = ["seg", "frame", "patch", "default"]
SECONDS_PER_BLOCK_RVS = 4.0
SECONDS_PER_BLOCK_XB = 2.0

TAXONOMY = [
    ("temporal-firstlast", re.compile(r"\b(first|last|earlier|beginning|end of|final)\b", re.I)),
    ("temporal-beforeafter", re.compile(r"\b(before|after|prior to|following)\b", re.I)),
    ("duration", re.compile(r"\bhow long\b|\bhow much time\b|\bhow many (seconds|minutes|hours)\b", re.I)),
    ("counting", re.compile(r"\bhow many\b", re.I)),
    ("task", re.compile(r"what task\b", re.I)),
    ("step", re.compile(r"what step\b", re.I)),
    ("overview", re.compile(r"overview|summar|describe the overall\b", re.I)),
    ("location", re.compile(r"\bwhere\b", re.I)),
    ("reason-why", re.compile(r"\bwhy\b", re.I)),
    ("action-doing", re.compile(r"doing|happen", re.I)),
    ("other-what", re.compile(r".", re.I)),
]

def qtype(q):
    for name, rx in TAXONOMY:
        if rx.search(q):
            return name
    return "other-what"


def blocks_overlap(blocks, start, end, spb):
    """covered seconds of [start,end] by retrieved frame-block ids."""
    cov = 0.0
    for b in blocks:
        bs, be = b * spb, (b + 1) * spb
        cov += max(0.0, min(be, end) - max(bs, start))
    return cov


def load_rvs():
    anno = {(a["video_name"], a["question"]): a
            for a in json.load(open(RVS_ANNO))}
    sub_videos = {v["video_id"] for v in json.load(open(B + "oracle_subset_anno.json"))}
    mats, metas = {}, []
    for p in POLS:
        recs = json.load(open(B + f"judged_oracle_{p}_72b.json"))["judged"]
        mats[p] = [1 if r["judge_pred"] == "yes" else 0 for r in recs]
    # retrieval: best policy = patch; rows aligned to judged order (verified)
    retr = {}
    for p in POLS:
        rows = list(csv.DictReader(open(B + f"oracle_{p}/1_0.csv")))
        retr[p] = [[int(x) for x in r["retrieved_blocks"].split(";")] if r["retrieved_blocks"] else []
                   for r in rows]
    # logged store seg ids per arm per video
    logged = {p: collections.defaultdict(set) for p in POLS}
    for p in POLS:
        for line in open(B + f"oracle_{p}/commit_log.jsonl"):
            r = json.loads(line)
            v = r["video_id"]
            for e in r.get("pass1", {}).get("seg_scores", []):
                logged[p][v].add(int(e[0]))
            for e in r.get("pass1", {}).get("commit", []):
                logged[p][v].add(int(e[0]))
    orecs = json.load(open(B + "judged_oracle_seg_72b.json"))["judged"]
    for i, r in enumerate(orecs):
        a = anno[(r["video_id"], r["question"])]
        metas.append(dict(video_id=r["video_id"], question=r["question"],
                          answer=r["answer"], start=a["start_time"], end=a["end_time"],
                          answer_type=a.get("answer_type", "?"),
                          qtype=qtype(r["question"]),
                          pred={p: json.load(open(B + f"judged_oracle_{p}_72b.json"))["judged"][i]["pred_answer"] for p in []}))
    dur_by_video = {v["video_id"]: v["duration"] for v in json.load(open(B + "ego4d_oe_npy.json"))}
    return mats, retr, logged, metas, sub_videos, dur_by_video


def load_xb():
    # ordered flatten of anno conversations; records are in the same order
    # (letter-judge records truncate the question text, so index-key)
    flat = []
    for v in json.load(open(XB + "xbench_anno.json")):
        for c in v["conversations"]:
            flat.append((v["video_id"], c))
    mats, retr, metas = {}, {}, []
    for p in POLS:
        recs = json.load(open(XB + f"judged_oracle_{p}_letter.json"))["records"]
        mats[p] = [1 if r["correct"] else 0 for r in recs]
        rows = list(csv.DictReader(open(XB + f"oracle_{p}/1_0.csv")))
        retr[p] = [[int(x) for x in r["retrieved_blocks"].split(";")] if r["retrieved_blocks"] else []
                   for r in rows]
    recs = json.load(open(XB + "judged_oracle_seg_letter.json"))["records"]
    assert len(flat) == len(recs) == 180, (len(flat), len(recs))
    for i, r in enumerate(recs):
        v, c = flat[i]
        assert v == r["video_id"], (i, v, r["video_id"])
        metas.append(dict(video_id=v, question=c["question"],
                          answer=r["answer"], start=c["start_time"], end=c["end_time"],
                          answer_type="MC", qtype="MC"))
    return mats, retr, metas


def partition(mats, n):
    k = [sum(mats[p][i] for p in POLS) for i in range(n)]
    return k

# ---------------- RVS ----------------
mats, retr, logged, metas, sub_videos, dur_by_video = load_rvs()
n = len(metas)
k = partition(mats, n)
best_pol = max(POLS, key=lambda p: sum(mats[p]))
seg_logged = logged[best_pol]

part = collections.Counter("all-right" if k[i] == 4 else "all-wrong" if k[i] == 0 else "mixed"
                           for i in range(n))
partition_out = {
    "n": n, "counts": dict(part),
    "pct": {c: round(100 * part[c] / n, 1) for c in part},
    "per_policy_acc": {p: round(100 * sum(mats[p]) / n, 1) for p in POLS},
    "best_fixed_policy": best_pol,
    "note": "occurrence-indexed records (316); the question 'What is the setting of the video clip?' appears 3x identically",
}

cats = collections.Counter()
examples = collections.defaultdict(list)
overlap_allwrong = []
type_cat = collections.defaultdict(collections.Counter)
win_lens = collections.defaultdict(list)
N_LOCAL_FRAMES = 15000 // 196          # 76 most-recent frames stay in the local cache
for i in range(n):
    if k[i] != 0:
        continue
    m = metas[i]
    start, end, v = m["start"], m["end"], m["video_id"]
    win = max(1e-9, end - start)
    blocks = retr[best_pol][i]
    ov = blocks_overlap(blocks, start, end, SECONDS_PER_BLOCK_RVS)
    # local-cache coverage: stream frontier (2*end s, capped at video length) minus
    # the last n_local tokens are addressable WITHOUT retrieval
    frontier_s = min(2 * end, dur_by_video[v])
    local_lo = max(0.0, frontier_s - N_LOCAL_FRAMES * SECONDS_PER_BLOCK_RVS)
    ov_total = ov + max(0.0, min(end, frontier_s) - max(start, local_lo))
    ov_total = min(ov_total, win)
    overlap_allwrong.append(ov_total / win)
    win_lens["all-wrong"].append(win)
    wsegs = set(range(int(start // 16), int(np.ceil(end / 16))))
    if ov_total <= 0:
        cat = "retrieval-failure"
    elif not (wsegs & seg_logged[v]):
        cat = "store-coverage(logged)-failure"
    else:
        cat = "model-capability-failure"
    cats[cat] += 1
    type_cat[m["qtype"]][cat] += 1
    if len(examples[cat]) < 8:
        examples[cat].append(dict(
            video_id=v, question=m["question"][:110], answer=m["answer"][:80],
            window=[start, end], answer_type=m["answer_type"], qtype=m["qtype"],
            retrieved_block_range=[min(blocks) if blocks else None, max(blocks) if blocks else None],
            n_retrieved=len(blocks),
            overlap_ratio=round(ov_total / win, 3)))

for i in range(n):
    if k[i] == 4:
        win_lens["all-right"].append(metas[i]["end"] - metas[i]["start"])
    elif 0 < k[i] < 4:
        win_lens["mixed"].append(metas[i]["end"] - metas[i]["start"])

breakdown = {
    "best_policy_used": best_pol,
    "categories": dict(cats),
    "pct_of_all_wrong": {c: round(100 * cats[c] / part["all-wrong"], 1) for c in cats},
    "overlap_ratio_all_wrong": {
        "mean": float(np.mean(overlap_allwrong)),
        "median": float(np.median(overlap_allwrong)),
        "frac_zero": float(np.mean([o <= 0 for o in overlap_allwrong])),
        "frac_ge_half": float(np.mean([o >= 0.5 for o in overlap_allwrong])),
    },
    "by_question_type": {t: dict(c) for t, c in sorted(type_cat.items())},
    "window_length_s": {g: dict(mean=round(float(np.mean(l)), 1), n=len(l))
                        for g, l in win_lens.items()},
    "proxies": {
        "block_to_time": "frame block b <-> [4b, 4b+4) s (effective 0.25 fps; verified vs max block id and n_segments=225/3600s)",
        "frontier": "streamed until 2*end s; window always within streamed range on RVS; the last n_local=15000 tokens (76 frames = 304 s) are additionally resident in the local cache at answer time and count as context-present",
        "store": "per-segment budget, no cross-segment eviction -> true zero-coverage impossible at tax0.5; 'store-coverage(logged)' = window segments absent from ALL logged store interactions (seg_scores+commit) of that video in the patch arm",
    },
}

# mixed set: who wins, by type
wins = collections.defaultdict(collections.Counter)
for i in range(n):
    if 0 < k[i] < 4:
        for p in POLS:
            if mats[p][i]:
                wins[metas[i]["qtype"]][p] += 1
mixed_by_type = {t: dict(c) for t, c in sorted(wins.items())}

# ---------------- StreamingBench ----------------
mats_x, retr_x, metas_x = load_xb()
nx = len(metas_x)
kx = partition(mats_x, nx)
part_x = collections.Counter("all-right" if kx[i] == 4 else "all-wrong" if kx[i] == 0 else "mixed"
                             for i in range(nx))
best_pol_x = max(POLS, key=lambda p: sum(mats_x[p]))
cats_x = collections.Counter()
examples_x = collections.defaultdict(list)
ovx = []
for i in range(nx):
    if kx[i] != 0:
        continue
    m = metas_x[i]
    win = max(1e-9, m["end"] - m["start"])
    blocks = retr_x[best_pol_x][i]
    ov = blocks_overlap(blocks, m["start"], m["end"], SECONDS_PER_BLOCK_XB)
    # local cache: last n_local tokens (76 frames x 2 s) before frontier=end
    local_lo = max(0.0, m["end"] - N_LOCAL_FRAMES * SECONDS_PER_BLOCK_XB)
    ov_total = min(ov + max(0.0, m["end"] - max(m["start"], local_lo)), win)
    ovx.append(ov_total / win)
    if ov_total <= 0:
        cat = "retrieval-failure"
    else:
        cat = "model-capability-failure"  # store coverage not logged on xbench
    cats_x[cat] += 1
    if len(examples_x[cat]) < 8:
        examples_x[cat].append(dict(
            video_id=m["video_id"], question=m["question"][:110], answer=m["answer"],
            window=[m["start"], m["end"]],
            retrieved_block_range=[min(blocks) if blocks else None, max(blocks) if blocks else None],
            n_retrieved=len(blocks), overlap_ratio=round(ov_total / win, 3)))

breakdown_x = {
    "best_policy_used": best_pol_x,
    "categories": dict(cats_x),
    "pct_of_all_wrong": {c: round(100 * cats_x[c] / part_x["all-wrong"], 1) for c in cats_x},
    "overlap_ratio_all_wrong": {
        "mean": float(np.mean(ovx)) if ovx else None,
        "median": float(np.median(ovx)) if ovx else None,
        "frac_zero": float(np.mean([o <= 0 for o in ovx])) if ovx else None,
    },
    "note": "store coverage not assessable: xbench commit logs carry no seg ids",
}
partition_x = {
    "n": nx, "counts": dict(part_x),
    "pct": {c: round(100 * part_x[c] / nx, 1) for c in part_x},
    "per_policy_acc": {p: round(100 * sum(mats_x[p]) / nx, 1) for p in POLS},
    "best_fixed_policy": best_pol_x,
}

json.dump({"rvs": partition_out, "xbench": partition_x},
          open(OUT + "w33_partition.json", "w"), indent=1)
json.dump({"rvs": breakdown, "xbench": breakdown_x},
          open(OUT + "w33_allwrong_breakdown.json", "w"), indent=1)
json.dump({"rvs": {c: e for c, e in examples.items()},
           "xbench": {c: e for c, e in examples_x.items()}},
          open(OUT + "w33_examples.json", "w"), indent=1)
json.dump({"rvs_mixed_wins_by_type": mixed_by_type},
          open(OUT + "w33_mixed_by_type.json", "w"), indent=1)

print("== RVS partition ==", partition_out)
print("\n== RVS all-wrong breakdown ==", json.dumps(breakdown, indent=1)[:2500])
print("\n== RVS mixed wins by type ==", json.dumps(mixed_by_type, indent=1))
print("\n== xbench partition ==", partition_x)
print("\n== xbench all-wrong ==", json.dumps(breakdown_x, indent=1))
