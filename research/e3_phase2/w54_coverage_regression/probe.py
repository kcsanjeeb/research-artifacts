#!/usr/bin/env python3
"""Quick probe: per-question coverage across the 4 W4.2 arms."""
import json, collections

W42 = "/data3/zhuotaotian2_e2/runs/20260930_0935_w42_spread/"
W41 = "/data3/zhuotaotian2_e2/runs/20260930_0935_w41_fpsfix/"
ids = json.load(open(W42 + "temporal83_ids.json"))
keys = [(r["video_id"], r["question"]) for r in ids]
occ_needed = collections.Counter(keys)
dur = {v["video_id"]: v["duration"] for v in json.load(open(W42 + "subset_anno_83.json"))}
win_of = {}
for v in json.load(open(W42 + "subset_anno_83.json")):
    for c in v["conversations"]:
        win_of[(v["video_id"], c["question"])] = (c["start_time"], c["end_time"])
N_LOCAL_S = (15000 // 196) * 2.0

def coverage_map(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        k = (r["video_id"], r["question"])
        if k not in occ_needed:
            continue
        start, end = win_of[k]
        frames = set()
        for e in r.get("pass1", {}).get("commit", []):
            seg_id, typ, fr, pa = int(e[0]), e[1], int(e[2]), int(e[3])
            frames.add(seg_id * 4 + fr)
        cov = sum(max(0.0, min((f + 1) * 2.0, end) - max(f * 2.0, start)) for f in frames)
        frontier_s = min(end, dur[r["video_id"]])
        local_lo = max(0.0, frontier_s - N_LOCAL_S)
        cov += max(0.0, min(end, frontier_s) - max(start, local_lo))
        win = max(1e-9, end - start)
        out[k] = min(cov, win) / win
    return out

arms = {"topk": coverage_map(W41 + "answers_7b_writetime_05fps/commit_log.jsonl")}
for p in ("uniform", "hybrid", "stratified"):
    arms[p] = coverage_map(W42 + f"arm_{p}/commit_log.jsonl")
for p, m in arms.items():
    assert len(m) == 83, (p, len(m))

# coverage pooled stats
allc = [v for m in arms.values() for v in m.values()]
print(f"pooled coverage: n={len(allc)} mean={sum(allc)/len(allc):.3f} min={min(allc):.3f} max={max(allc):.3f}")
# within-question variation
within_sd, between_vals = [], []
for k in keys:
    vals = [arms[p][k] for p in ("topk", "uniform", "hybrid", "stratified")]
    mu = sum(vals) / 4
    within_sd.append((sum((v - mu) ** 2 for v in vals) / 3) ** 0.5)
    between_vals.append(mu)
print(f"within-question sd of coverage: mean={sum(within_sd)/len(within_sd):.4f} "
      f"max={max(within_sd):.4f}")
bmu = sum(between_vals) / len(between_vals)
print(f"between-question sd of mean coverage: {(sum((v-bmu)**2 for v in between_vals)/82)**0.5:.4f}")
# per-question range across arms
rng = [max(arms[p][k] for p in arms) - min(arms[p][k] for p in arms) for k in keys]
print(f"per-question arm range: mean={sum(rng)/len(rng):.4f} max={max(rng):.4f}")
# pairwise correlation of coverage across arms
import itertools
names = ["topk", "uniform", "hybrid", "stratified"]
import math
def corr(a, b):
    n = len(a); ma = sum(a)/n; mb = sum(b)/n
    cov = sum((x-ma)*(y-mb) for x, y in zip(a, b))
    va = sum((x-ma)**2 for x in a); vb = sum((y-mb)**2 for y in b)
    return cov / math.sqrt(va*vb)
for x, y in itertools.combinations(names, 2):
    print(f"corr({x},{y}) = {corr([arms[x][k] for k in keys],[arms[y][k] for k in keys]):.3f}")
# coverage ranks: does any question flip ordering a lot?
flips = sum(1 for k in keys if max(arms[p][k] for p in names) - min(arms[p][k] for p in names) > 0.05)
print(f"questions with arm-range > 0.05: {flips}/83")
