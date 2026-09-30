"""E0.6 v2 -- calibrated fusion. Fixes two v1 errors, zero GPU:

1. "VLM 93.4% accuracy on escalated clips" was a base-rate artifact: a
   constant-No predictor scores 93.58% on the same pool; the VLM at threshold
   0.5 scores 93.36% -- below constant. The verdict must rest on the soft
   score's RANKING signal (within-pool AUC 0.669 vs tier-1's 0.573), not on
   thresholded accuracy at a base rate of 6.4%.

2. replace_b fused raw P(Yes) (90% of values < 0.26) into a tier-1 score
   axis where anomalous clips sit at median 0.994 -- crushing every escalated
   clip to the bottom of the global ranking. Fix: calibrate P(Yes) onto the
   detector scale before fusing. Two maps, both cross-fitted by video parity
   (fit on even videos, apply to odd, and vice versa -- no clip is scored by
   a map that saw it):
   - isotonic: non-decreasing map P(Yes) -> P(anomaly) (PAVA, pure numpy)
   - ranknorm: rank of P(Yes) within the band, scaled into (tau-margin,
     tau+margin); preserves the global ordering by construction
The oracle ceiling (escalated scores replaced by ground truth) is reported
alongside every deltaAUC, per the same rule E0.5 applied to rankers.
"""
import glob
import json
import os
import sys

import numpy as np

E0 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.join(E0, 'followup', 'results')
TAU, MARGIN = 0.5, 0.2603611499071121
SNIP, FPS = 16, 30.0

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlmvalue import auc_rank, ap_score, event_stats  # noqa: E402


# ------------------------------------------------------------------ isotonic

def pava(x, y):
    """Pool Adjacent Violators: non-decreasing isotonic fit of y on x.
    Returns (knots_x, knots_y) for step interpolation."""
    order = np.argsort(x, kind='mergesort')
    xs, ys = x[order].astype(float), y[order].astype(float)
    # aggregate duplicate x
    ux, inv = np.unique(xs, return_inverse=True)
    wy = np.bincount(inv, weights=ys)
    cnt = np.bincount(inv)
    val = wy / cnt
    w = cnt.astype(float)
    # PAVA on (ux, val, w)
    lvl = list(val)
    wt = list(w)
    xs_k = list(ux)
    i = 0
    while i < len(lvl) - 1:
        if lvl[i] > lvl[i + 1]:
            new_w = wt[i] + wt[i + 1]
            new_v = (wt[i] * lvl[i] + wt[i + 1] * lvl[i + 1]) / new_w
            lvl[i:i + 2] = [new_v]
            wt[i:i + 2] = [new_w]
            xs_k[i:i + 2] = [xs_k[i]]
            if i > 0:
                i -= 1
        else:
            i += 1
    return np.array(xs_k), np.array(lvl)


def iso_apply(knots_x, knots_y, x):
    return np.interp(x, knots_x, knots_y,
                     left=knots_y[0], right=knots_y[-1])


# ---------------------------------------------------------------------- main

def main():
    # ---- load everything ----
    vlm = {}
    for f in sorted(glob.glob(os.path.join(OUT, 'vlm_scores_shard*.jsonl'))):
        for line in open(f):
            r = json.loads(line)
            vlm[(r['vi'], r['si'])] = r
    t1 = np.load(os.path.join(E0, 'results', 'tier1_scores.npz'), allow_pickle=True)
    ids = t1['video_ids'].tolist()
    scores, lens = t1['scores'], t1['lens']
    ann = {}
    for line in open(os.path.join(E0, 'third_party', 'VadCLIP', 'list',
                                  'Temporal_Anomaly_Annotation.txt')):
        p = line.split()
        if not p:
            continue
        ann[p[0].replace('.mp4', '')] = [(int(a), int(b)) for a, b in
                                         zip(p[2::2], p[3::2])
                                         if int(a) >= 0 and int(b) >= 0]
    labels = np.zeros((len(ids), lens.max()), dtype=np.int8)
    for i, vid in enumerate(ids):
        for a, b in ann.get(vid, []):
            labels[i, max((a - 1) // SNIP, 0):min((b - 1) // SNIP, lens[i] - 1) + 1] = 1

    pairs = sorted(vlm.keys())
    pv = np.array([vlm[p]['score_b'] for p in pairs])
    gt = np.array([labels[p] for p in pairs])
    s_t1 = np.array([scores[p] for p in pairs])
    vis = np.array([p[0] for p in pairs])
    gt_frames = np.load(os.path.join(E0, 'third_party', 'VadCLIP', 'list',
                                     'gt_ucf.npy'))

    def full_frame(fused):
        concat = np.concatenate([np.repeat(fused[i, :lens[i]], SNIP)
                                 for i in range(len(ids))])
        n = min(len(concat), len(gt_frames))
        return concat[:n], gt_frames[:n]

    def fused_replace(score_map):
        out = scores.copy()
        for (vi, si), v in score_map.items():
            out[vi, si] = v
        return out

    def ev(name, fused):
        fs, g = full_frame(fused)
        return {'name': name, 'auc': auc_rank(fs, g), 'ap': ap_score(fs, g)}

    fs0, g0 = full_frame(scores)
    auc_t1, ap_t1 = auc_rank(fs0, g0), ap_score(fs0, g0)
    print(f"baseline tier-1: AUC={auc_t1:.4f} AP={ap_t1:.4f}")

    # ---- oracle ceiling ----
    oracle_map = {p: float(g) for p, g in zip(pairs, gt)}
    o = ev('oracle', fused_replace(oracle_map))
    print(f"oracle ceiling: AUC={o['auc']:.4f} (delta {o['auc']-auc_t1:+.4f}) "
          f"AP={o['ap']:.4f} (delta {o['ap']-ap_t1:+.4f})")
    ceiling = o['auc'] - auc_t1

    # ---- v1 reproduction ----
    v1 = ev('replace_b_raw', fused_replace({p: float(v) for p, v in zip(pairs, pv)}))
    print(f"v1 replace_b (raw P(Yes)): AUC={v1['auc']:.4f} "
          f"(delta {v1['auc']-auc_t1:+.4f})")

    # ---- calibration, cross-fitted by video parity ----
    cal_iso, cal_rank = {}, {}
    for parity in (0, 1):
        fit_m = (vis % 2) == parity
        app_m = ~fit_m
        # isotonic
        kx, ky = pava(pv[fit_m], gt[fit_m])
        cal_vals = iso_apply(kx, ky, pv[app_m])
        for p, v in zip(np.array(pairs, dtype=object)[app_m], cal_vals):
            cal_iso[tuple(p)] = float(v)
        # rank-normalize into the band
        r_fit = np.argsort(np.argsort(pv[fit_m]))
        # map applied values by their rank within their own set, using the
        # fit set's P(Yes) distribution as the reference grid
        grid_x = np.sort(pv[fit_m])
        grid_u = (np.arange(len(grid_x)) + 0.5) / len(grid_x)
        u_app = np.interp(pv[app_m], grid_x, grid_u)
        lo, hi = TAU - MARGIN, TAU + MARGIN
        for p, v in zip(np.array(pairs, dtype=object)[app_m], lo + u_app * (hi - lo)):
            cal_rank[tuple(p)] = float(v)
    print(f"calibrated scores: isotonic range "
          f"[{min(cal_iso.values()):.3f}, {max(cal_iso.values()):.3f}], "
          f"ranknorm range [{min(cal_rank.values()):.3f}, {max(cal_rank.values()):.3f}]")

    variants = [v1,
                ev('replace_b_isotonic', fused_replace(cal_iso)),
                ev('replace_b_ranknorm', fused_replace(cal_rank))]
    for v in variants:
        d = v['auc'] - auc_t1
        frac = d / ceiling if ceiling else float('nan')
        print(f"  {v['name']:>22}: AUC={v['auc']:.4f} (delta {d:+.4f}, "
              f"{frac:.1%} of ceiling) AP={v['ap']:.4f} (delta {v['ap']-ap_t1:+.4f})")

    # ---- VLM-only PR sweep on the escalated pool (threshold not fixed at 0.5)
    sweep = []
    for thr in np.arange(0.02, 0.99, 0.02):
        pred = pv > thr
        tp = (pred & (gt == 1)).sum()
        if pred.sum() == 0:
            continue
        prec = tp / pred.sum()
        rec = tp / max(gt.sum(), 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-9)
        sweep.append({'thr': float(thr), 'precision': float(prec),
                      'recall': float(rec), 'f1': float(f1)})
    best = max(sweep, key=lambda r: r['f1'])
    print(f"VLM-only PR sweep: best F1={best['f1']:.3f} at thr={best['thr']:.2f} "
          f"(precision={best['precision']:.3f}, recall={best['recall']:.3f})")

    # ---- ops metrics at tau=0.5 with calibrated fusion ----
    def ops(fused):
        alerts, tp = 0, 0
        recs, ttas = [], []
        for i in range(len(ids)):
            s_v = fused[i, :lens[i]]
            lab = labels[i, :lens[i]]
            a = s_v > TAU
            alerts += a.sum()
            tp += (a & (lab == 1)).sum()
            es = event_stats(s_v, lab, TAU)
            if es:
                recs.append(es[0])
                if es[1] is not None:
                    ttas.append(es[1])
        return {'precision@0.5': float(tp / max(alerts, 1)),
                'alerts_per_hour': float(alerts / (lens.sum() * SNIP / FPS) * 3600),
                'event_recall': float(np.mean(recs)),
                'tta_seconds_mean': float(np.mean(ttas) * SNIP / FPS)}

    ops_out = {'tier1': ops(scores),
               'replace_b_isotonic': ops(fused_replace(cal_iso))}
    print(f"ops: {json.dumps(ops_out, indent=2)}")

    # ---- marginal-value curve with calibrated fusion ----
    esc_s = s_t1
    order_by_s = np.argsort(-esc_s)
    Ks = [0.10, 0.25, 0.50, 1.00]
    curve = {'Random': [], 'RawTopK': []}
    for K in Ks:
        m = int(round(K * len(pairs)))
        for rule in curve:
            aucs = []
            for seed in (0, 1, 2):
                r = np.random.RandomState(seed)
                chosen = (set(r.choice(len(pairs), m, replace=False).tolist())
                          if rule == 'Random' else set(order_by_s[:m].tolist()))
                fused = scores.copy()
                for i in chosen:
                    fused[pairs[i]] = cal_iso[pairs[i]]
                fs, g = full_frame(fused)
                aucs.append(auc_rank(fs, g))
            curve[rule].append({'K': K, 'calls': m,
                                'dAUC_mean': float(np.mean(aucs) - auc_t1),
                                'dAUC_std': float(np.std(aucs))})
            print(f"[2.4v2] {rule} K={K}: dAUC={np.mean(aucs)-auc_t1:+.4f} "
                  f"+-{np.std(aucs):.4f}")

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for rule, marker in (('Random', 'o--'), ('RawTopK', 's-')):
        xs = [c['calls'] for c in curve[rule]]
        ys = [c['dAUC_mean'] for c in curve[rule]]
        es = [c['dAUC_std'] for c in curve[rule]]
        ax.errorbar(xs, ys, yerr=es, fmt=marker, label=rule, capsize=3)
    ax.axhline(0, color='k', lw=0.8)
    ax.axhline(ceiling, color='gray', ls=':', lw=1, label=f'oracle ceiling {ceiling:+.4f}')
    ax.set_xlabel('VLM calls spent (of 5,965 escalated)')
    ax.set_ylabel('delta AUC vs tier-1')
    ax.set_title('E0.6 v2 marginal value of a VLM call (calibrated fusion)')
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.join(OUT, 'figures'), exist_ok=True)
    fig.savefig(os.path.join(OUT, 'figures', 'marginal_value_calibrated.png'), dpi=150)
    plt.close(fig)

    doc = {
        'note': 'E0.6 v2: calibrated fusion. v1 errors corrected: (1) 93.4% '
                'accuracy was below the constant-No baseline 93.58%; '
                '(2) raw P(Yes) fusion crushed escalated clips below the '
                'detector score scale. Calibration cross-fitted by video parity.',
        'baseline': {'auc_tier1': auc_t1, 'ap_tier1': ap_t1},
        'oracle_ceiling': {'auc': o['auc'], 'dAUC': ceiling, 'ap': o['ap'],
                           'dAP': o['ap'] - ap_t1},
        'pool_stats': {'n': len(pairs), 'anomalous': int(gt.sum()),
                       'const_no_acc': float((gt == 0).mean()),
                       'vlm_yes_at_0.5': int((pv > 0.5).sum()),
                       'vlm_acc_at_0.5': float(((pv > 0.5).astype(int) == gt).mean()),
                       'within_pool_auc_vlm': auc_rank(pv, gt),
                       'within_pool_auc_tier1': auc_rank(s_t1, gt)},
        'variants': variants,
        'vlm_pr_sweep': {'best_f1': best, 'note': 'threshold swept 0.02..0.98; '
                                                  '0.5 is the wrong operating '
                                                  'point at 6.4% base rate'},
        'ops': ops_out,
        'marginal_value_calibrated': curve,
    }
    with open(os.path.join(OUT, 'vlmvalue_calibrated.json'), 'w') as f:
        json.dump(doc, f, indent=2)
    print(f"[saved] {OUT}/vlmvalue_calibrated.json")


if __name__ == '__main__':
    main()
