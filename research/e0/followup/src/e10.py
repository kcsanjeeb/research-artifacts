"""E0.10 -- can a cheap predictor capture the aggressive-budget headroom?
THE LAST GATE.

Sign convention (explicit, after the E0.9 sign error): degradation(s) =
per-video AUC(stride 1) - per-video AUC(stride s), so POSITIVE = loss.
Oracle 'saved' by upgrading uniform u -> s: saved = max(0, auc(s) - auc(u))
per video (recovered AUC), spent over (1/s - 1/u)*n snippets, greedy by
saved/cost under a hard budget. Per-video gain capped at zero: the oracle may
only densify; it cannot harvest the smoothing gains of sparser strides.

Design rules: pooled AUC is the endpoint; strict causality (features from
analysed snippets only); EXACT budget enforcement (token bucket, asserted);
random-at-equal-cost veto; CV by video; no tuning to gates.
"""
import json
import os
import sys

import numpy as np

E0 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(E0, 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import e09  # loaders, features, per_video_auc, held_series, pooled_auc_ap  # noqa
from vlmvalue import auc_rank, ap_score  # noqa: E402

OUT = os.path.join(E0, 'followup', 'results')
SNIP = 16
SEED = 0
STRIDES = [1, 2, 4, 8, 16, 32, 64]
BUDGETS = [1 / 8, 1 / 16, 1 / 32]
WINDOW = 32
rng = np.random.RandomState(SEED)


def ladder_for(B):
    u = int(round(1.0 / B))
    return [s for s in STRIDES if s <= u]


# ------------------------------------------------------------------ Step A

def step_a(ds, aucs, valid):
    n_vids = len(ds['lens'])
    total = int(ds['lens'].sum())
    vid_idx = np.flatnonzero(valid)
    COARSE = 64
    out = []
    for B in BUDGETS:
        u = int(round(1.0 / B))
        au_u, ap_u = e09.pooled_auc_ap(ds, np.full(n_vids, u))
        # oracle v3: everyone at the coarsest stride; spend the budget
        # upgrading the best saved-per-cost videos (sequential upgrades
        # allowed; gain capped at zero -> saved = max(0, auc(s)-auc(64)))
        options = []
        for i in vid_idx:
            n = ds['lens'][i]
            for s in STRIDES:
                if s >= COARSE:
                    continue
                saved = max(0.0, aucs[s][i] - aucs[COARSE][i])
                cost = (1.0 / s - 1.0 / COARSE) * n
                if saved > 0 and cost > 0:
                    options.append((saved / cost, i, s, cost))
        options.sort(key=lambda x: -x[0])
        budget = total * B
        sp = np.full(n_vids, COARSE, dtype=float)
        spent = total / COARSE
        for ratio, i, s, cost in options:
            if sp[i] <= s or spent + cost > budget:
                continue
            sp[i] = s
            spent += cost
        au_o, ap_o = e09.pooled_auc_ap(ds, sp)
        # random at exact cost: coarsest baseline + uniform-random upgrades
        # to the same stride distribution the oracle realized is NOT matching;
        # use two-point: fraction q at stride 1, rest at COARSE with
        # q*(1 - 1/64) = B - 1/64  ->  q = (B - 1/64)/(1 - 1/64)
        q = (B - 1.0 / COARSE) / (1.0 - 1.0 / COARSE)
        r = np.random.RandomState(SEED)
        sp_r = np.where(r.rand(n_vids) < q, 1, COARSE).astype(float)
        au_r, ap_r = e09.pooled_auc_ap(ds, sp_r)
        out.append({'budget': B, 'uniform_stride': u,
                    'uniform_auc': au_u, 'uniform_ap': ap_u,
                    'oracle_auc': au_o, 'oracle_ap': ap_o,
                    'random_auc': au_r, 'random_ap': ap_r,
                    'headroom_auc': au_o - au_u,
                    'oracle_cost_frac': spent / total})
        print(f"[A:{ds['name']}] B=1/{int(1/B):<2} uniform={au_u:.4f} "
              f"oracle={au_o:.4f} headroom={au_o - au_u:+.4f} "
              f"random={au_r:.4f} (oracle cost {spent/total:.4f})")
    return out


# ------------------------------------------------------------------ Step B

def step_b(ds, aucs, valid, B):
    """Predictor quality at budget B: target per-video LOSS at uniform u."""
    u = int(round(1.0 / B))
    n_vids = len(ds['lens'])
    vid_idx = np.flatnonzero(valid)
    # loss if kept at uniform stride u (positive = loss); cap not applied to
    # the predictor target itself -- predicting signed sensitivity is fine
    y = np.array([aucs[1][i] - aucs[u][i] for i in vid_idx])
    F = [e09.video_features(ds, i, u) for i in vid_idx]
    names = list(F[0].keys())
    X = np.array([[f[k] for k in names] for f in F])
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    corrs = {}
    for j, k in enumerate(names):
        corrs[k] = float(np.corrcoef(Xs[:, j], y)[0, 1]) if Xs[:, j].std() > 0 else 0.0
    # 5-fold CV by video, ridge
    r = np.random.RandomState(SEED)
    order = r.permutation(len(vid_idx))
    folds = np.array_split(order, 5)
    preds = np.zeros(len(vid_idx))
    for f in folds:
        tr = np.setdiff1d(order, f)
        W = e09.ridge_fit(Xs[tr], y[tr])
        preds[f] = e09.ridge_pred(W, Xs[f])
    mc = float(np.corrcoef(preds, y)[0, 1])
    qs = np.percentile(preds, [25, 75])
    q1 = float(y[preds <= qs[0]].mean())
    q4 = float(y[preds >= qs[1]].mean())
    print(f"[B:{ds['name']}] B=1/{int(1/B)} target=loss@u model_cv_corr={mc:.3f} "
          f"quartile_sep={q4 - q1:+.4f} top={sorted(corrs.items(), key=lambda kv: -abs(kv[1]))[:3]}")
    return {'budget': B, 'target': f'loss_at_stride_{u}', 'feature_corr': corrs,
            'model_cv_corr': mc, 'quartile_low': q1, 'quartile_high': q4,
            'names': names, 'mu': mu.tolist(), 'sd': sd.tolist(),
            'W': e09.ridge_fit(Xs, y).tolist()}


def predict_windows(ds, i, stride, ctx):
    """Per-window predicted loss over a video at a reference stride. Returns
    list of (w0, pred) for windows with >= 4 analysed snippets."""
    names, mu, sd, W = ctx['names'], ctx['mu'], ctx['sd'], ctx['W']
    n = ds['lens'][i]
    out = []
    for w0 in range(0, n, WINDOW):
        idx = np.arange(w0, min(w0 + WINDOW, n), stride)
        if len(idx) < 4:
            continue
        sf = e09.score_feats(ds['scores'][i, idx])
        mf = e09.motion_feats(ds['motion'][i, idx])
        fw = np.array([[sf[k] for k in names[:6]] + [mf[k] for k in names[6:]]])
        fw = (fw - np.array(mu)) / np.array(sd)
        out.append((w0, float(e09.ridge_pred(np.array(W), fw)[0])))
    return out


# ------------------------------------------------------------------ Step C

def eval_series(ds, parts):
    if ds['name'] == 'ucf':
        sc = np.concatenate([np.repeat(p, SNIP) for p in parts])
        nn = min(len(sc), len(ds['gt_concat']))
        return auc_rank(sc[:nn], ds['gt_concat'][:nn]), ap_score(sc[:nn], ds['gt_concat'][:nn])
    per_gt = e09.xd_frame_gt()
    sc = np.concatenate([np.repeat(p, SNIP) for p in parts])
    gt = np.concatenate(per_gt)
    nn = min(len(sc), len(gt))
    return auc_rank(sc[:nn], gt[:nn]), ap_score(sc[:nn], gt[:nn])


def stride_from_pred(pred, theta_map, u):
    """theta_map: list of (threshold, stride) sorted by threshold desc."""
    for th, s in theta_map:
        if pred > th:
            return s
    return u



def step_c(ds, ctx_by_B, aucs, valid):
    n_vids = len(ds['lens'])
    total = int(ds['lens'].sum())
    COARSE = 64
    vid_idx = np.flatnonzero(valid)
    out = []
    for B in BUDGETS:
        u = int(round(1.0 / B))
        ctx = ctx_by_B[B]
        au_u, ap_u = e09.pooled_auc_ap(ds, np.full(n_vids, u))

        # ---- per-video static: coarsest baseline + water-fill densification
        # by PREDICTED loss (the oracle with a predicted sensitivity)
        preds_v = {}
        for i in vid_idx:
            p = predict_windows(ds, i, u, ctx)
            preds_v[i] = float(np.mean([x[1] for x in p])) if p else 0.0
        order = sorted(vid_idx, key=lambda i: -preds_v[i])
        sp = np.full(n_vids, COARSE, dtype=float)
        spent = total / COARSE
        budget = total * B
        for i in order:
            n = ds['lens'][i]
            cur = sp[i]
            for s in STRIDES:
                if s >= cur:
                    continue
                extra = (1.0 / s - 1.0 / cur) * n
                if spent + extra <= budget:
                    sp[i] = s
                    spent += extra
        cost_pv = sum(ds['lens'][i] / sp[i] for i in range(n_vids)) / total
        au_pv, ap_pv = e09.pooled_auc_ap(ds, sp)

        # ---- per-window online: quantile ladder solved for E[cost]=B,
        # token bucket enforces the hard per-stream budget.
        # strides {1,4,16,64} at predicted-loss quantiles {p,4p,12p} with
        # p*1 + 3p/4 + 8p/16 + (1-12p)/64 = B  ->  p = (B - 1/64)/2.0625
        p = (B - 1.0 / COARSE) / 2.0625
        train_preds = []
        for i in vid_idx[:len(vid_idx) // 2]:
            train_preds += [x[1] for x in predict_windows(ds, i, 16, ctx)]
        train_preds = np.array(train_preds)
        qv = np.quantile(train_preds, [min(12 * p, 1.0), min(4 * p, 1.0), p]) \
            if len(train_preds) > 10 else np.zeros(3)
        cuts = sorted([(qv[2], 1), (qv[1], 4), (qv[0], 16)], key=lambda x: -x[0])
        bucket_cap = B * 4 * WINDOW
        bucket = 0.0
        cost_pw = 0.0
        pw_parts = []
        for i in range(n_vids):
            n = ds['lens'][i]
            parts = []
            stride_prev = u
            w = 0
            while w < n:
                L = min(WINDOW, n - w)
                bucket = min(bucket + B * L, bucket_cap)
                if w == 0:
                    s_use = u
                else:
                    pw = None
                    idx = np.arange(w - WINDOW, w, stride_prev)
                    if len(idx) >= 4:
                        sf = e09.score_feats(ds['scores'][i, idx])
                        mf = e09.motion_feats(ds['motion'][i, idx])
                        fw = np.array([[sf[k] for k in ctx['names'][:6]]
                                       + [mf[k] for k in ctx['names'][6:]]])
                        fw = (fw - np.array(ctx['mu'])) / np.array(ctx['sd'])
                        pw = float(e09.ridge_pred(np.array(ctx['W']), fw)[0])
                    want = stride_from_pred(pw if pw is not None else -1e9,
                                            cuts, COARSE)
                    s_use = COARSE
                    for s in STRIDES:
                        if s <= want and L / s <= bucket + 1e-9:
                            s_use = s
                            break
                bucket -= L / s_use
                cost_pw += L / s_use
                idx = np.arange(w, w + L, s_use)
                parts.append(np.repeat(ds['scores'][i, idx], s_use)[:L])
                stride_prev = s_use
                w += WINDOW
            pw_parts.append(np.concatenate(parts))
        cost_pw /= total
        assert cost_pw <= B + 1e-9, f"budget violated: {cost_pw} > {B}"
        au_pw, ap_pw = eval_series(ds, pw_parts)

        # ---- random per-window floor: identical mapping, random ranks
        r3 = np.random.RandomState(SEED + 2)
        cost_rw = 0.0
        rw_parts = []
        for i in range(n_vids):
            n = ds['lens'][i]
            parts = []
            w = 0
            while w < n:
                L = min(WINDOW, n - w)
                rk = r3.rand()
                want = stride_from_pred(rk, [(1 - p, 1), (1 - 4 * p, 4),
                                             (1 - 12 * p, 16)], COARSE)
                s_use = want
                idx = np.arange(w, w + L, s_use)
                parts.append(np.repeat(ds['scores'][i, idx], s_use)[:L])
                cost_rw += len(idx)
                w += WINDOW
            rw_parts.append(np.concatenate(parts))
        cost_rw /= total
        au_rw, ap_rw = eval_series(ds, rw_parts)

        # ---- oracle (coarsest baseline + true sensitivity)
        options = []
        for i in vid_idx:
            n = ds['lens'][i]
            for s in STRIDES:
                if s >= COARSE:
                    continue
                saved = max(0.0, aucs[s][i] - aucs[COARSE][i])
                cost = (1.0 / s - 1.0 / COARSE) * n
                if saved > 0 and cost > 0:
                    options.append((saved / cost, i, s, cost))
        options.sort(key=lambda x: -x[0])
        sp_o = np.full(n_vids, COARSE, dtype=float)
        spent_o = total / COARSE
        for ratio, i, s, cost in options:
            if sp_o[i] <= s or spent_o + total * B < spent_o + cost:
                continue
            if spent_o + cost > total * B:
                continue
            sp_o[i] = s
            spent_o += cost
        au_o, ap_o = e09.pooled_auc_ap(ds, sp_o)
        head = au_o - au_u
        capture = (au_pw - au_u) / head if head > 1e-9 else 0.0
        beats_random = au_pw - au_rw
        print(f"[C:{ds['name']}] B=1/{int(1/B):<2} uniform={au_u:.4f} "
              f"adapt_pv={au_pv:.4f}(cost {cost_pv:.4f}) "
              f"adapt_pw={au_pw:.4f}(cost {cost_pw:.4f}) "
              f"rand_pw={au_rw:.4f}(cost {cost_rw:.4f}) "
              f"oracle={au_o:.4f} headroom={head:+.4f} capture={capture:+.1%} "
              f"beats_random={beats_random:+.4f}")
        out.append({'budget': B, 'uniform_auc': au_u, 'uniform_ap': ap_u,
                    'adaptive_pervideo_auc': au_pv, 'adaptive_pervideo_ap': ap_pv,
                    'adaptive_pervideo_cost': float(cost_pv),
                    'adaptive_perwindow_auc': au_pw, 'adaptive_perwindow_ap': ap_pw,
                    'adaptive_perwindow_cost': float(cost_pw),
                    'random_perwindow_auc': au_rw, 'random_perwindow_ap': ap_rw,
                    'random_perwindow_cost': float(cost_rw),
                    'oracle_auc': au_o, 'oracle_ap': ap_o,
                    'headroom_auc': float(head), 'capture': float(capture),
                    'beats_random_auc': float(beats_random),
                    'cost_check_ok': bool(cost_pw <= B + 1e-9 and cost_pv <= B + 1e-9),
                    'cameras_per_4gpu_server': 14400.0 / (359.273 * B)})
    return out

def figures(oracle, ctrl, preds):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir = os.path.join(OUT, 'figures')
    os.makedirs(figdir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for name in ('ucf', 'xd'):
        xs = [1.0 / r['budget'] for r in ctrl[name]]
        ys = [r['capture'] for r in ctrl[name]]
        ax.plot(xs, ys, 'o-', label=f'{name} per-window capture')
        yo = [r['headroom_auc'] * 100 for r in oracle[name]]
        ax.plot(xs, yo, 's--', alpha=0.5, label=f'{name} oracle headroom (x100)')
    ax.set_xscale('log', base=2)
    ax.set_xlabel('subsampling factor (1/B)')
    ax.set_ylabel('capture fraction / headroom')
    ax.set_title('E0.10: capture vs budget')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, 'e10_capture.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    styles = [('uniform', 'o-', 'k'), ('random_perwindow', 'v--', 'gray'),
              ('adaptive_pervideo', 's-', 'tab:blue'),
              ('adaptive_perwindow', '^-', 'tab:red'),
              ('oracle', 'x', 'tab:green')]
    for name in ('ucf', 'xd'):
        for key, mk, col in styles:
            xs = [c['cameras_per_4gpu_server'] for c in ctrl[name]]
            ys = [c[f'{key}_auc'] for c in ctrl[name]]
            ax.plot(xs, ys, mk, color=col,
                    label=f'{name}:{key}' if name == 'ucf' else None,
                    alpha=0.9 if name == 'ucf' else 0.45)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('cameras per 4-GPU server (equal cost)')
    ax.set_ylabel('pooled AUC')
    ax.set_title('E0.10 frontier: pooled AUC vs cameras served')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, 'e10_frontier.png'), dpi=150)
    plt.close(fig)
    print('[figures] e10_capture.png, e10_frontier.png')


# ------------------------------------------------------------------- main

def main():
    STRIDES7 = [1, 2, 4, 8, 16, 32, 64]
    oracle, preds, ctrl = {}, {}, {}
    for loader in (e09.load_ucf, e09.load_xd):
        ds = loader()
        print(f'=== {ds["name"]} ===')
        aucs = {s: e09.per_video_auc(ds, s) for s in STRIDES7}
        valid = ~np.isnan(aucs[1])
        # degradation convention: loss = auc1 - aucs (positive = loss)
        for s in (8, 32):
            loss = aucs[1][valid] - aucs[s][valid]
            print(f"[A:{ds['name']}] loss at stride {s}: mean={loss.mean():+.4f} "
                  f"lose>0.02={(loss > 0.02).mean():.2f} improve={(loss < -0.005).mean():.2f}")
        oracle[ds['name']] = step_a(ds, aucs, valid)
        ctx_by_B = {}
        for B in BUDGETS:
            c = step_b(ds, aucs, valid, B)
            preds.setdefault(ds['name'], []).append(
                {k: v for k, v in c.items() if k not in ('names', 'mu', 'sd', 'W')})
            ctx_by_B[B] = c
        ctrl[ds['name']] = step_c(ds, ctx_by_B, aucs, valid)
    figures(oracle, ctrl, preds)
    cap = {n: next(r['capture'] for r in ctrl[n] if abs(r['budget'] - 1 / 32) < 1e-9)
           for n in ctrl}
    br = {n: next(r['beats_random_auc'] for r in ctrl[n] if abs(r['budget'] - 1 / 32) < 1e-9)
          for n in ctrl}
    def gatev(c, b):
        if c >= 0.40 and b > 0.001:
            return 'SYSTEM'
        if c >= 0.20 and b > 0.001:
            return 'MARGINAL'
        return 'NO SYSTEM'
    gate = {n: gatev(cap[n], br[n]) for n in cap}
    verdict = ('SYSTEMS PAPER' if all(g == 'SYSTEM' for g in gate.values())
               else 'MEASUREMENT PAPER')
    doc = {'config': {'budgets': BUDGETS, 'window_snippets': WINDOW,
                      'strides': STRIDES7, 'cv': 'by_video', 'seed': SEED,
                      'degradation_convention': 'loss = auc(1) - auc(s), positive = loss',
                      'sign_fix_note': 'E0.9 oracle used saved = max(0, deg_u - deg_s) '
                                       'which is negated; corrected here to '
                                       'saved = max(0, auc(s) - auc(u)).'},
           'oracle': oracle, 'predictors': preds, 'controller': ctrl,
           'gate_capture_at_1_32': cap,
           'gate_beats_random_at_1_32': br,
           'gate': gate, 'verdict': verdict}
    with open(os.path.join(OUT, 'e10.json'), 'w') as f:
        json.dump(doc, f, indent=2)
    print(f"\nGATE capture@1/32: {cap}")
    print(f"GATE beats_random@1/32: {br}")
    print(f"GATE: {gate}  VERDICT: {verdict}")
    print(f"[saved] {os.path.join(OUT, 'e10.json')}")


if __name__ == '__main__':
    main()
