#!/usr/bin/env python3
"""W5.2 stats: answer-level verification of group-scale int4 (TB6 lesson:
verify at ANSWER level, not logit level). Headline: greedy-answer
character-identical rate vs the fp16 arm on the same 120 questions (gate >=95%).
Secondary: tensor-level fidelity (TB2 mse/cosine/relerr from trapcheck), realized
bytes per arm (measured, not assumed 4x), 72B-judged accuracy delta fp16 vs each
int4 arm. Writes RESULTS_W52.md + analysis_w52_summary.json.
"""
import json, collections, hashlib, os
import pandas as pd
from scipy import stats

W52 = "/data3/zhuotaotian2_e2/runs/20261001_w52_int4_group/"
ANNO = "/data3/zhuotaotian2_e2/code/decaf/data/rvs/ego/ego4d_oe.json"
ARMS = ["fp16", "int4legacy", "int4g64", "int4g128"]
LABEL = {"fp16": "fp16 store", "int4legacy": "int4 per-channel (legacy, TB6)",
         "int4g64": "int4 group-64 asymmetric", "int4g128": "int4 group-128 asymmetric"}

anno = json.load(open(ANNO))
vid_h = {v["video_id"]: max(c["end_time"] for c in v["conversations"]) / 3600.0 for v in anno}

def answers(name):
    df = pd.read_csv(W52 + f"answers_{name}/1_0.csv")
    return df

def load_ordered(path):
    recs = json.load(open(path))["judged"]
    seen = collections.Counter(); out = collections.OrderedDict()
    for r in recs:
        k = (r["video_id"], r["question"]); i = seen[k]; seen[k] += 1
        out[(k, i)] = 1 if r["judge_pred"] == "yes" else 0
    return out

def mcnemar(a, b):
    keys = sorted(set(a) & set(b))
    ab = sum(1 for k in keys if a[k] == 1 and b[k] == 0)
    ba = sum(1 for k in keys if a[k] == 0 and b[k] == 1)
    d = ab + ba
    p_exact = stats.binomtest(ab, d, 0.5).pvalue if d else 1.0
    return dict(n_pairs=len(keys), ab=ab, ba=ba, discordant=d, p_exact=p_exact)

base = answers("fp16")
qcol, acol = "question", "pred_answer"
base_map = {(r["video_id"], r[qcol]): str(r[acol]) for _, r in base.iterrows()}

rows = []
out = ["# W5.2 results — group-scale int4 store verification (answer-level)", ""]
out.append(f"Subset: 120 questions, single video (full 1-h stream), 7B, writetime tax0.5 debias store; "
           f"fp16 vs int4 variants otherwise identical. n={len(base)} answers per arm.")
out.append("")
out.append("| arm | answer-identical vs fp16 | rate | realized GB/h | ratio vs fp16 | judged acc (72B) | McNemar vs fp16 (discordant, p) | md5 csv |")
out.append("|---|---|---|---|---|---|---|---|")

summary = {}
for name in ARMS:
    df = answers(name)
    amap = {(r["video_id"], r[qcol]): str(r[acol]) for _, r in df.iterrows()}
    keys = sorted(set(base_map) & set(amap))
    ident = sum(1 for k in keys if amap[k] == base_map[k])
    rate = ident / len(keys)
    # realized bytes from commit log (per-video max store_bytes_total)
    sb = collections.defaultdict(int)
    for line in open(W52 + f"answers_{name}/commit_log.jsonl"):
        r = json.loads(line)
        if "store_bytes" in r:
            sb[r["video_id"]] = max(sb[r["video_id"]], r["store_bytes"]["store_bytes_total"])
    hours = sum(vid_h.get(v, 0) or 1.0 for v in sb) or 1.0
    gbh = sum(sb.values()) / hours / 1e9
    md5 = hashlib.md5(open(W52 + f"answers_{name}/1_0.csv", "rb").read()).hexdigest()
    summary[name] = dict(n=len(keys), identical=ident, rate=rate, gb_h=gbh, bytes_by_video=dict(sb), md5=md5)

# judged accuracy + McNemar
j = {n: load_ordered(W52 + f"judged_{n}_72b.json") for n in ARMS}
for name in ARMS:
    acc = 100 * sum(j[name].values()) / len(j[name])
    m = mcnemar(j[name], j["fp16"]) if name != "fp16" else None
    summary[name]["judge_acc"] = acc
    if m:
        summary[name]["vs_fp16"] = m
    gbh = summary[name]["gb_h"]
    ratio = gbh / summary["fp16"]["gb_h"]
    mstr = "-" if not m else f"{m['discordant']} discordant, p={m['p_exact']:.3f}"
    out.append(f"| {LABEL[name]} | {summary[name]['identical']}/{summary[name]['n']} | "
               f"**{100*summary[name]['rate']:.1f}%** | {gbh:.2f} | {ratio:.2f}x | {acc:.1f} | {mstr} | {summary[name]['md5'][:12]} |")

out.append("")
g64 = summary["int4g64"]["rate"]; g128 = summary["int4g128"]["rate"]
best = max(g64, g128)
# gate: >=95% answer-identical AND accuracy delta within noise (McNemar n.s.)
gate_acc = all(summary[n].get("vs_fp16", {}).get("p_exact", 1.0) > 0.05 or
               summary[n]["vs_fp16"]["discordant"] <= 5 for n in ("int4g64", "int4g128"))
out.append("## Gate (work5.md: >=95% answer-identical + accuracy delta within noise)")
out.append("")
out.append(f"- int4 group-64 answer-identical: **{100*g64:.1f}%** (gate >=95%: {'PASS' if g64 >= 0.95 else 'FAIL'})")
out.append(f"- int4 group-128 answer-identical: **{100*g128:.1f}%** (gate >=95%: {'PASS' if g128 >= 0.95 else 'FAIL'})")
out.append(f"- accuracy delta vs fp16 within noise (McNemar): {'YES' if gate_acc else 'NO'} "
           f"(see table; n=120 so discordant<=5 is the 95% band edge)")
out.append("")
if best >= 0.95 and gate_acc:
    out.append("**GATE PASS -> W5.3 unlocked** (combined best-tax + int4 group scales at true 0.5 fps).")
else:
    out.append("**GATE FAIL at every group size -> int4 not available; fp16 storage floor stands; Pareto path closes per work5.md.**")
out.append("")
out.append("## Secondary fidelity (tensor level, real captures — TB2, 0.5B encode)")
out.append("")
out.append("See trapcheck outputs tc_q4g64 / tc_q4g64-128: relerr/mse/cosine of dequantized stored K/V vs raw captures.")
out.append("Headline metric is answer-level (TB6: naive per-channel int4 flipped 4/5 greedy answers at logit-near-lossless).")
out.append("")
out.append("## Storage/fidelity curve (realized, measured — NOT assumed 4x)")
out.append("")
out.append("| arm | realized ratio vs fp16 | answer-identical |")
out.append("|---|---|---|")
for name in ARMS:
    out.append(f"| {LABEL[name]} | {summary[name]['gb_h']/summary['fp16']['gb_h']:.2f}x | {100*summary[name]['rate']:.1f}% |")

open(W52 + "RESULTS_W52.md", "w").write("\n".join(out) + "\n")
json.dump(summary, open(W52 + "analysis_w52_summary.json", "w"), indent=2)
print("\n".join(out))
print("STATS_DONE")
