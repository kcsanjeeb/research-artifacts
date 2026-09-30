#!/usr/bin/env python3
"""W5.4 pooled coverage regression: per-question correctness ~ achieved window
coverage, 4 W4.2 arms x 83 temporal questions = 332 observations.

Models:
  A (primary): OLS LPM, correct ~ coverage + question FE, cluster-robust by question
  B:           OLS LPM, correct ~ coverage + C(arm) + window_len, cluster by question
  C:           OLS LPM, correct ~ coverage + question FE + arm FE, cluster by question
  D:           Logit FE (robustness; question FE), robust SE
Outputs: w54_results.json + coverage_bins (for the curve) + printed summary.
"""
import json, collections, math, os
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

W42 = "/data3/zhuotaotian2_e2/runs/20260930_0935_w42_spread/"
W41 = "/data3/zhuotaotian2_e2/runs/20260930_0935_w41_fpsfix/"
OUT = "/data3/zhuotaotian2_e2/analysis/w54_coverage_regression/"

ids = json.load(open(W42 + "temporal83_ids.json"))
keys = [(r["video_id"], r["question"]) for r in ids]
occ_needed = collections.Counter(keys)

def load_ordered(path):
    recs = json.load(open(path))["judged"]
    seen = collections.Counter()
    out = collections.OrderedDict()
    for r in recs:
        k = (r["video_id"], r["question"])
        i = seen[k]; seen[k] += 1
        if k in occ_needed and i < occ_needed[k]:
            out[k] = 1 if r["judge_pred"] == "yes" else 0
    return out

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

arm_files = {
    "topk": (W41 + "judged_7b_writetime_05fps_72b.json", W41 + "answers_7b_writetime_05fps/commit_log.jsonl"),
    "uniform": (W42 + "judged_arm_uniform_72b.json", W42 + "arm_uniform/commit_log.jsonl"),
    "hybrid": (W42 + "judged_arm_hybrid_72b.json", W42 + "arm_hybrid/commit_log.jsonl"),
    "stratified": (W42 + "judged_arm_stratified_72b.json", W42 + "arm_stratified/commit_log.jsonl"),
}

rows = []
for arm, (jpath, cpath) in arm_files.items():
    correct = load_ordered(jpath)
    cov = coverage_map(cpath)
    assert len(correct) == 83 and len(cov) == 83, (arm, len(correct), len(cov))
    for k in keys:
        s, e = win_of[k]
        rows.append(dict(qid=f"{k[0]}::{k[1]}", arm=arm,
                         correct=correct[k], coverage=cov[k],
                         winlen=e - s))
df = pd.DataFrame(rows)
assert len(df) == 332
os.makedirs(OUT, exist_ok=True)
df.to_csv(OUT + "w54_dataset.csv", index=False)

res = {"n": len(df), "n_questions": df.qid.nunique(),
       "coverage": dict(mean=float(df.coverage.mean()), min=float(df.coverage.min()),
                        max=float(df.coverage.max()),
                        within_q_sd=float(df.groupby("qid").coverage.std().mean()),
                        between_q_sd=float(df.groupby("qid").coverage.mean().std(ddof=1))),
       "accuracy_by_arm": {a: float(g.correct.mean()) for a, g in df.groupby("arm")}}

def fit(model, name, cluster=True):
    kw = dict(cov_type="cluster", cov_kwds={"groups": df.qid}) if cluster else {}
    m = model.fit(**kw)
    b = m.params["coverage"]; se = m.bse["coverage"]
    ci = m.conf_int().loc["coverage"].tolist()
    res[name] = dict(coef=float(b), se=float(se), ci_lo=float(ci[0]), ci_hi=float(ci[1]),
                     p=float(m.pvalues["coverage"]), n=int(m.nobs),
                     r2=float(m.rsquared) if hasattr(m, "rsquared") else None,
                     effect_per_0p1_pt=round(float(b) * 0.1 * 100, 2))
    return m

# A: primary — question FE
fit(smf.ols("correct ~ coverage + C(qid)", data=df), "A_lpm_qFE")
# B: arm FE + window-length control
fit(smf.ols("correct ~ coverage + C(arm) + winlen", data=df), "B_lpm_armFE_winlen")
# C: question FE + arm FE
fit(smf.ols("correct ~ coverage + C(qid) + C(arm)", data=df), "C_lpm_qFE_armFE")

# D: logit with question FE (robustness)
try:
    ml = smf.logit("correct ~ coverage + C(qid)", data=df).fit(disp=0, maxiter=200)
    b = ml.params["coverage"]; se = ml.bse["coverage"]
    ci = ml.conf_int().loc["coverage"].tolist()
    res["D_logit_qFE"] = dict(coef_or=float(b), se_or=float(se),
                              ci_or=[float(ci[0]), float(ci[1])], p=float(ml.pvalues["coverage"]),
                              n=int(ml.nobs),
                              or_per_0p1=round(float(math.exp(b * 0.1)), 3))
except Exception as ex:
    res["D_logit_qFE"] = dict(error=str(ex))

# coverage -> accuracy curve: decile-ish bins with 95% CI (normal approx)
df["bin"] = pd.qcut(df.coverage, 6, duplicates="drop")
bins = []
for iv, g in df.groupby("bin", observed=True):
    n = len(g); acc = g.correct.mean()
    se = math.sqrt(acc * (1 - acc) / n)
    bins.append(dict(bin_lo=float(iv.left), bin_hi=float(iv.right),
                     n=int(n), coverage_mean=float(g.coverage.mean()),
                     accuracy=float(acc), ci_lo=float(acc - 1.96 * se),
                     ci_hi=float(acc + 1.96 * se)))
res["curve"] = bins

json.dump(res, open(OUT + "w54_results.json", "w"), indent=2)
print(json.dumps({k: v for k, v in res.items() if k != "curve"}, indent=2))
print("\ncurve:")
for b in bins:
    print(f"  [{b['bin_lo']:.3f},{b['bin_hi']:.3f}] n={b['n']:3d} cov={b['coverage_mean']:.3f} "
          f"acc={100*b['accuracy']:.1f}% CI [{100*b['ci_lo']:.1f},{100*b['ci_hi']:.1f}]")
