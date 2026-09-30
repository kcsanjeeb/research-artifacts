"""T1 analysis + figures from t1_smb_values.json (SMB) and t1_rekv artifacts."""
import json
import os
import sys

import numpy as np

T1 = os.path.expanduser("~/e1/t1_value_heterogeneity")


def gini_nonneg(x):
    x = np.maximum(np.asarray(x, dtype=float), 0)
    x = np.sort(x)
    n = len(x)
    if x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) @ x) / (n * x.sum()) - (n + 1) / n)


def lorenz(x):
    x = np.sort(np.maximum(np.asarray(x, dtype=float), 0))
    cum = np.concatenate([[0.0], np.cumsum(x)])
    if cum[-1] > 0:
        cum = cum / cum[-1]
    return np.arange(len(cum)) / (len(cum) - 1), cum


def smb_analysis(values_path, out_prefix):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    d = json.load(open(values_path))
    v = np.array(d["v"])
    f = np.array(d["f"])
    vpb = np.array(d["v_per_byte"])
    g_f = d["gini_f"]
    g_v = gini_nonneg(v)
    g_vpb = gini_nonneg(vpb)
    print(f"SMB: gini_f={g_f:.3f} gini_v(clip0)={g_v:.3f} gini_vpb(clip0)={g_vpb:.3f} "
          f"frac_never_retrieved={d['frac_never_retrieved']:.3f} "
          f"frac_v_negative={float(np.mean(v < 0)):.3f}")

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    ax = axes[0]
    nz = vpb[vpb > 0]
    ax.hist(np.log10(nz + 1e-12), bins=40, color="tab:blue", alpha=0.8)
    ax.set_xlabel("log10(v_j / bytes_j), retrieved events only")
    ax.set_ylabel("# events")
    ax.set_title(f"Marginal value per byte (SMB)\n"
                 f"Gini(f)={g_f:.2f}, Gini(v)={g_v:.2f}; "
                 f"{d['frac_never_retrieved']*100:.0f}% never retrieved")
    ax = axes[1]
    for arr, name, col in ((f, "retrieval freq f_j", "tab:blue"),
                           (v, "marginal value v_j (clip 0)", "tab:orange")):
        lx, ly = lorenz(arr)
        ax.plot(lx, ly, label=f"{name} (Gini={gini_nonneg(arr):.2f})", color=col)
    ax.plot([0, 1], [0, 1], "k--", lw=0.5)
    ax.set_xlabel("fraction of segments (sorted)")
    ax.set_ylabel("cumulative share")
    ax.legend(fontsize=8)
    ax.set_title("Lorenz curves")
    ax = axes[2]
    uo = d["uniform_vs_oracle"]
    b = uo["budgets"]
    def _get(d, x, k):
        return d[str(x)][k] if str(x) in d else d[repr(x)][k] if repr(x) in d else d[x][k]
    ub = [_get(uo["uniform"], x, "bal") for x in b]
    ob = [_get(uo["oracle"], x, "bal") for x in b]
    ax.errorbar([x * 100 for x in b], ub,
                yerr=[_get(uo["uniform"], x, "bal_std") for x in b],
                marker="o", label="uniform retention", capsize=3)
    ax.plot([x * 100 for x in b], ob, marker="s", label="oracle water-fill (LOO value)")
    ax.axhline(d["baseline_bal_acc"], color="gray", ls=":", lw=0.8,
               label=f"full memory ({d['baseline_bal_acc']})")
    ax.set_xlabel("retention budget (%)")
    ax.set_ylabel("balanced exist+neg acc")
    ax.legend(fontsize=8)
    ax.set_title("Uniform vs oracle water-filling (SMB)")
    fig.tight_layout()
    fig.savefig(out_prefix, dpi=150)
    return {"gini_f": g_f, "gini_v_clip0": g_v, "gini_vpb_clip0": g_vpb,
            "frac_never_retrieved": d["frac_never_retrieved"],
            "frac_v_negative": float(np.mean(v < 0)),
            "uniform_vs_oracle": uo, "baseline_bal_acc": d["baseline_bal_acc"]}


if __name__ == "__main__":
    r = smb_analysis(sys.argv[1], sys.argv[2])
    json.dump(r, open(sys.argv[3], "w"), indent=1, default=str)
