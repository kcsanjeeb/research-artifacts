import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
d = json.load(open("smb_summary.json"))
uo = d["uniform_vs_oracle"]
b = uo["budgets"]
def _get(dd, x, k):
    return dd[str(x)][k] if str(x) in dd else dd[repr(x)][k] if repr(x) in dd else dd[x][k]
ub = [_get(uo["uniform"], x, "bal") for x in b]
us = [_get(uo["uniform"], x, "bal_std") for x in b]
ob = [_get(uo["oracle"], x, "bal") for x in b]
fig, ax = plt.subplots(figsize=(6, 4.5))
ax.errorbar([x*100 for x in b], ub, yerr=us, marker="o", label="uniform retention (3 seeds)", capsize=3)
ax.plot([x*100 for x in b], ob, marker="s", label="oracle water-fill (LOO value/byte)")
ax.axhline(d["baseline_bal_acc"], color="gray", ls=":", label=f"full memory ({d['baseline_bal_acc']})")
ax.set_xlabel("retention budget (%)"); ax.set_ylabel("balanced exist+neg acc")
ax.set_title("SMB: uniform vs oracle water-filling (in-sample oracle)")
ax.legend(fontsize=9); ax.grid(alpha=0.3)
fig.tight_layout(); fig.savefig("uniform_vs_waterfill.png", dpi=150)
print("fig ok")
