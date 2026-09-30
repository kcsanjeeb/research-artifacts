import json
from collections import Counter, defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

recs = [json.loads(l) for l in open("artifacts/t1_rekv_log_all.jsonl")]
loo = [json.loads(l) for l in open("artifacts/t1_rekv_loo_all.jsonl")]

vids = {}
for r in recs:
    vids.setdefault(r["video_id"], r["n_blocks"])
freq = Counter()
for r in recs:
    for b in set(r["retrieved_blocks"]):
        freq[(r["video_id"], b)] += 1
f = np.array([freq.get((v, b), 0) for v, nb in vids.items() for b in range(nb)])

per = defaultdict(list)
for r in loo:
    per[(r["video_id"], r["block"])].append(r["drop"])
vals = np.array([np.mean(v) for v in per.values()])

def gini(x):
    x = np.sort(np.maximum(np.asarray(x, float), 0))
    n = len(x)
    if x.sum() == 0: return 0.0
    return float((2*np.arange(1,n+1) @ x)/(n*x.sum()) - (n+1)/n)

gf = gini(f)
gv = gini(vals)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
ax = axes[0]
ax.hist(np.log10(f[f > 0]), bins=30, color="tab:blue", alpha=0.8)
ax.set_xlabel("log10(retrieval frequency f_j), retrieved blocks only")
ax.set_ylabel("# blocks")
ax.set_title(f"ReKV/RVS-Ego block retrieval freq\nGini(f)={gf:.2f}; "
             f"{float(np.mean(f == 0))*100:.0f}% blocks never retrieved")
ax = axes[1]
xs = np.sort(np.maximum(vals, 0))
cum = np.concatenate([[0.0], np.cumsum(xs)])
if cum[-1] > 0: cum /= cum[-1]
ax.plot(np.arange(len(cum)) / max(1, len(cum) - 1), cum, color="tab:orange", lw=2)
ax.plot([0, 1], [0, 1], "k--", lw=0.5)
ax.set_xlabel("fraction of sampled blocks (sorted)")
ax.set_ylabel("cumulative value share")
ax.set_title(f"Lorenz, per-block marginal value (token-F1 drop)\n"
             f"Gini(v)={gv:.2f}; top-10% blocks carry "
             f"{float(np.sort(vals)[-15:].sum()/vals.sum())*100:.0f}% of value")
fig.tight_layout()
fig.savefig("value_distribution_rekv.png", dpi=150)
print("fig fixed", round(gf, 3), round(gv, 3))
