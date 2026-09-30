#!/usr/bin/env python3
"""W5.1 stats: realized GB/h (per-video max store_bytes / stream-h, same sweep
methodology), total pipeline GPU-s/stream-h from answer-log timestamps (GPU-s =
wall-s, single GPU), encode-phase estimate, paired McNemar vs MuKV-faithful and
vs the W4.1 tax0.5 corrected arm, per-arm md5 integrity. Writes RESULTS_W51.md.
System python3 (scipy)."""
import json, collections, re, os, hashlib, datetime
from scipy import stats

RUN = "/data3/zhuotaotian2_e2/runs/20260930_1750_w51_lowtax_fps/"
P0 = "/data3/zhuotaotian2_e2/runs/20260927_0000_e3_phase0/"
W41 = "/data3/zhuotaotian2_e2/runs/20260930_0935_w41_fpsfix/"
ANNO = "/data3/zhuotaotian2_e2/code/decaf/data/rvs/ego/ego4d_oe.json"

ARMS = {
    "tax025": {"sd": "answers_7b_writetime_tax025_05fps", "log": "logs/tax025.answer.log",
               "desc": "writetime tax0.25 debias fp16 @ true 0.5 fps"},
    "tax0125_patch": {"sd": "answers_7b_writetime_tax0125patch_05fps", "log": "logs/tax0125_patch.answer.log",
               "desc": "writetime tax0.125 debias fp16 + patch-granular store @ true 0.5 fps"},
}

anno = json.load(open(ANO if False else ANNO))
stream_h = sum(max(c["end_time"] for c in v["conversations"]) for v in anno) / 3600.0

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
    if d > 0:
        p_exact = stats.binomtest(ab, d, 0.5).pvalue
        p_cc = stats.chi2.sf((abs(ab - ba) - 1) ** 2 / d, 1)
    else:
        p_exact = p_cc = 1.0
    return dict(n_pairs=len(keys), ab=ab, ba=ba, discordant=d,
                acc_a=round(100 * sum(a[k] for k in keys) / len(keys), 1), p_exact=p_exact, p_cc=p_cc)

def fmt_p(p):
    return "p<0.001" if p < 0.001 else f"p={p:.3f}"

def wall_s(logpath):
    ts = re.findall(r"\[I \d{6} (\d{2}):(\d{2}):(\d{2})", open(logpath).read())
    if len(ts) < 2:
        return None
    (h1, m1, s1), (h2, m2, s2) = ts[0], ts[-1]
    return (int(h2) * 3600 + int(m2) * 60 + int(s2)) - (int(h1) * 3600 + int(m1) * 60 + int(s1))

mukv = load_ordered(P0 + "judged_7b_paper_72b.json")
w41 = load_ordered(W41 + "judged_7b_writetime_05fps_72b.json")

out = ["# W5.1 results — low-tax writetime at TRUE 0.5 fps", "",
       f"stream-hours: {stream_h:.3f} (ego4d_oe.json durations, 10 videos)", ""]
summary = []
for name, a in ARMS.items():
    sd = RUN + a["sd"]
    # integrity
    md5_csv = hashlib.md5(open(sd + "/1_0.csv", "rb").read()).hexdigest()
    md5_j = hashlib.md5(open(RUN + f"judged_{name}_72b.json", "rb").read()).hexdigest()
    # realized GB/h: per-video max store_bytes_total
    sb = collections.defaultdict(int)
    n_q = 0
    for line in open(sd + "/commit_log.jsonl"):
        r = json.loads(line)
        n_q += 1
        if "store_bytes" in r:
            sb[r["video_id"]] = max(sb[r["video_id"]], r["store_bytes"]["store_bytes_total"])
    gbh = sum(sb.values()) / stream_h / 1e9
    # fps assertion
    fps = [json.loads(l) for l in open(sd + "/fps_assertion.jsonl")]
    fps_ok = all(x.get("assert_passed") for x in fps)
    # wall / GPU-s per stream-h
    ws = wall_s(RUN + a["log"])
    gpu_s_h = ws / stream_h if ws else None
    # accuracy + McNemar
    j = load_ordered(RUN + f"judged_{name}_72b.json")
    acc = 100 * sum(j.values()) / len(j)
    mm = mcnemar(j, mukv)
    mw = mcnemar(j, w41)
    summary.append(dict(arm=name, acc=acc, n=len(j), gb_h=gbh, fps_ok=fps_ok, n_fps=len(fps),
                        wall_s=ws, gpu_s_h=gpu_s_h, md5_csv=md5_csv, md5_judged=md5_j,
                        vs_mukv=mm, vs_tax05=mw, bytes_by_video=dict(sb)))
    out.append(f"## {name} — {a['desc']}")
    out.append("")
    out.append(f"- accuracy (72B judged): **{acc:.1f}** (n={len(j)})")
    out.append(f"- realized store: **{gbh:.2f} GB/h** at true 0.5 fps (per-video max store_bytes_total / {stream_h:.3f} h; "
               f"total {sum(sb.values())/1e9:.2f} GB)")
    out.append(f"- fps assertion: {'PASS' if fps_ok else 'FAIL'} ({len(fps)} videos)")
    out.append(f"- total pipeline: {ws/3600:.2f} h -> **{gpu_s_h:.0f} GPU-s/stream-h** (GPU-s = wall-s, single GPU); "
               f"encode phase ~280 GPU-s/stream-h (identical encode workload to W4.1; tax changes only store retention at assembly)")
    out.append(f"- md5: csv={md5_csv} judged={md5_j}")
    out.append("")
    out.append("| comparison | n pairs | A right B wrong | B right A wrong | discordant | McNemar exact p | cc chi2 p |")
    out.append("|---|---|---|---|---|---|---|")
    out.append(f"| {name} vs MuKV-faithful (50.9) | {mm['n_pairs']} | {mm['ab']} | {mm['ba']} | {mm['discordant']} | "
               f"{fmt_p(mm['p_exact'])} | {fmt_p(mm['p_cc'])} |")
    out.append(f"| {name} vs writetime tax0.5 @0.5fps (52.4) | {mw['n_pairs']} | {mw['ab']} | {mw['ba']} | {mw['discordant']} | "
               f"{fmt_p(mw['p_exact'])} | {fmt_p(mw['p_cc'])} |")
    out.append("")

# gate
t25 = next(s for s in summary if s["arm"] == "tax025")
gate = t25["acc"] >= 51.0 and t25["gb_h"] <= 6.0
out.append("## Gate (work5.md: tax0.25 >= ~51 at <= 6 GB/h)")
out.append("")
out.append(f"tax0.25: acc {t25['acc']:.1f}, {t25['gb_h']:.2f} GB/h -> "
           + ("**GATE PASS — proceed to W5.2 final run / W5.3**" if gate else
              "**GATE FAIL — accuracy collapses or storage exceeds gate: Pareto path dead per work5.md, record and stop**"))
out.append("")
out.append("## Corrected-vs-pre-fix context")
out.append("")
out.append("| arm | acc | GB/h @0.5fps | note |")
out.append("|---|---|---|---|")
out.append(f"| writetime tax0.25 pre-fix @0.25fps-eff | 51.8 | 2.79 | compB sweep arm (erroneous half rate) |")
out.append(f"| writetime tax0.25 corrected @0.5fps | {t25['acc']:.1f} | {t25['gb_h']:.2f} | THIS RUN |")
t125 = next(s for s in summary if s["arm"] == "tax0125_patch")
out.append(f"| writetime tax0.125 patch-store @0.5fps | {t125['acc']:.1f} | {t125['gb_h']:.2f} | sub-frame realized storage (W5.1 arm 2) |")
out.append("| writetime tax0.5 corrected @0.5fps | 52.4 | 10.38 | W4.1 |")
out.append("| MuKV published | 50.9 | 0.91-1.23 | their own 0.5 fps |")
out.append("")
out.append("Framing (work5.md, anti-drift): the achievable claim is *matches CVPR'26 SOTA accuracy at "
           "comparable storage with a different architecture* — parity, not a win; no 'beats' language "
           "outside the m=21 multiplicity family.")

open(RUN + "RESULTS_W51.md", "w").write("\n".join(out) + "\n")
json.dump(summary, open(RUN + "analysis_w51_summary.json", "w"), indent=2)
print("\n".join(out[-30:]))
print("STATS_DONE")
