"""E0.9 -- is there a system here? Adaptive per-stream sampling (final go/no-go).

All CPU, all artifacts on disk. Design rules enforced throughout:
- control decision per-stream; evaluation on POOLED AUC/AP only
- features computed from ANALYSED snippets only (causality, E0.9 6.1)
- cross-validation by video
- per-video gain capped at zero in the oracle (do not harvest the 29%
  that "improve" under subsampling -- smoothing noise)
- exact cost accounting: cost = snippets actually analysed
- random-at-equal-cost floor reported next to every adaptive number
"""
import json
import os
import sys

import numpy as np

E0 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(E0, 'src'))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlmvalue import auc_rank, ap_score  # noqa: E402

OUT = os.path.join(E0, 'followup', 'results')
SNIP = 16
SEED = 0
STRIDES = [1, 2, 4, 8, 16]
BUDGETS = [0.5, 0.25, 0.125, 0.0625]
WINDOW = 32          # snippets (~17s at UCF, ~21s at XD)
CAP = 0.0            # per-video gain cap (no harvesting improvements)
rng = np.random.RandomState(SEED)


# ------------------------------------------------------------------ loading

def load_ucf():
    t1 = np.load(os.path.join(E0, 'results', 'tier1_scores.npz'),
                 allow_pickle=True)
    vids = t1['video_ids'].tolist()
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
    labels = np.zeros((len(vids), lens.max()), dtype=np.int8)
    for i, v in enumerate(vids):
        for a, b in ann.get(v, []):
            labels[i, max((a - 1) // SNIP, 0):min((b - 1) // SNIP, lens[i] - 1) + 1] = 1
    motion = np.load(os.path.join(E0, 'results', 'motion_features.npz'))['motion']
    gt_frames = np.load(os.path.join(E0, 'third_party', 'VadCLIP', 'list',
                                     'gt_ucf.npy'))
    return {'name': 'ucf', 'scores': scores, 'lens': lens, 'labels': labels,
            'motion': motion, 'gt_concat': gt_frames}


def xd_test_vids():
    """XD test video ids in csv order (csv module; avoids the pandas dep)."""
    import csv
    vids = []
    with open(os.path.join(E0, 'third_party', 'VadCLIP', 'list',
                           'xd_CLIP_rgbtest.csv')) as f:
        for row in csv.DictReader(f):
            vids.append(os.path.basename(row['path']).replace('__0.npy', ''))
    return vids


def xd_frame_gt():
    """Frame-level XD GT, mirroring VadCLIP's make_gt_xd.py:
    (n_snip+1)*16 frames per video, anomaly segments from annotations.txt,
    then drop the last 16 -> n_snip*16. Native implementation (no pandas)."""
    ann = {}
    for line in open(os.path.join(E0, 'third_party', 'VadCLIP', 'list',
                                  'annotations.txt')):
        p = line.strip().split()
        if not p:
            continue
        name = p[0].replace('.mp4', '')
        ann[name] = [(int(p[i]), int(p[i + 1]))
                     for i in range(1, len(p) - 1, 2)]
    t1 = np.load(os.path.join(OUT, 'xd_tier1_scores.npz'), allow_pickle=True)
    vids = t1['video_ids'].tolist()
    lens = t1['lens']
    per_video = []
    for i, v in enumerate(vids):
        n = int(lens[i])
        gt = np.zeros((n + 1) * SNIP, dtype=np.float32)
        if 'label_A' not in v:
            for a, b in ann.get(v, []):
                gt[a:b] = 1.0
        per_video.append(gt[:-SNIP].astype(np.int8))
    return per_video


def load_xd():
    t1 = np.load(os.path.join(OUT, 'xd_tier1_scores.npz'), allow_pickle=True)
    vids = t1['video_ids'].tolist()
    scores, lens = t1['scores'], t1['lens']
    per_video = xd_frame_gt()
    labels = np.zeros((len(vids), lens.max()), dtype=np.int8)
    gt_concat = np.concatenate(per_video)
    for i in range(len(vids)):
        n = lens[i]
        labels[i, :n] = per_video[i][:n * SNIP].reshape(n, SNIP).max(axis=1)
    motion = np.load(os.path.join(OUT, 'xd_motion_features.npz'))['motion']
    return {'name': 'xd', 'scores': scores, 'lens': lens, 'labels': labels,
            'motion': motion, 'gt_concat': gt_concat}


# ------------------------------------------------------------- series utils

def held_series(s, stride, length):
    return np.repeat(s[::stride], stride)[:length]


def pooled_auc_ap(ds, stride_per_video):
    parts = []
    if ds['name'] == 'ucf':
        for i in range(len(ds['lens'])):
            n = ds['lens'][i]
            sc = held_series(ds['scores'][i, :n], int(stride_per_video[i]), n)
            parts.append(np.repeat(sc, SNIP))
        scc = np.concatenate(parts)
        n = min(len(scc), len(ds['gt_concat']))
        return auc_rank(scc[:n], ds['gt_concat'][:n]), ap_score(scc[:n], ds['gt_concat'][:n])
    per_video = xd_frame_gt()
    gts = []
    for i in range(len(ds['lens'])):
        n = ds['lens'][i]
        sc = held_series(ds['scores'][i, :n], int(stride_per_video[i]), n)
        parts.append(np.repeat(sc, SNIP))
        gts.append(per_video[i])
    scc = np.concatenate(parts)
    gt = np.concatenate(gts)
    n = min(len(scc), len(gt))
    return auc_rank(scc[:n], gt[:n]), ap_score(scc[:n], gt[:n])


def per_video_auc(ds, stride):
    """AUC per video (videos with both classes only) at a stride."""
    out = np.full(len(ds['lens']), np.nan)
    for i in range(len(ds['lens'])):
        n = ds['lens'][i]
        lab = ds['labels'][i, :n]
        if lab.min() == lab.max():
            continue
        out[i] = auc_rank(held_series(ds['scores'][i, :n], stride, n), lab)
    return out


# ------------------------------------------------------------------ Step A

def step_a(ds):
    aucs = {s: per_video_auc(ds, s) for s in STRIDES}
    valid = ~np.isnan(aucs[1])
    # downsampled-definition AUCs (analysed snippets only, no zero-order hold)
    aucs_ds = {}
    for s in STRIDES:
        a = np.full(len(ds['lens']), np.nan)
        for i in range(len(ds['lens'])):
            n = ds['lens'][i]
            idx = np.arange(0, n, s)
            lab = ds['labels'][i, idx]
            if lab.min() == lab.max():
                continue
            a[i] = auc_rank(ds['scores'][i, idx], lab)
        aucs_ds[s] = a
    valid_ds = ~np.isnan(aucs_ds[1]) & ~np.isnan(aucs_ds[8])
    d8 = aucs_ds[8][valid_ds] - aucs_ds[1][valid_ds]
    downsampled = {'mean': float(d8.mean()), 'std': float(d8.std()),
                   'p10': float(np.percentile(d8, 10)),
                   'p90': float(np.percentile(d8, 90)),
                   'frac_lose_gt_0.02': float((d8 > 0.02).mean()),
                   'frac_improve': float((d8 < -0.005).mean()),
                   'n': int(valid_ds.sum())}
    rows = {}
    for s in STRIDES:
        d = aucs[s][valid] - aucs[1][valid]
        rows[s] = {
            'n_videos': int(valid.sum()),
            'mean': float(d.mean()), 'std': float(d.std()),
            'p10': float(np.percentile(d, 10)), 'p25': float(np.percentile(d, 25)),
            'p50': float(np.percentile(d, 50)), 'p75': float(np.percentile(d, 75)),
            'p90': float(np.percentile(d, 90)), 'max': float(d.max()),
            'frac_lose_gt_0.02': float((d > 0.02).mean()),
            'frac_flat': float((np.abs(d) <= 0.02).mean()),
            'frac_improve': float((d < -0.005).mean()),
        }
    # best stride at tolerance 0.01 AUC
    best = np.ones(valid.sum(), dtype=int)
    tol = 0.01
    vid_idx = np.flatnonzero(valid)
    for k, i in enumerate(vid_idx):
        d = {s: aucs[s][i] - aucs[1][i] for s in STRIDES}
        b = 1
        for s in STRIDES:
            if d[s] < tol:
                b = s
        best[k] = b
    hist = {str(s): float((best == s).mean()) for s in STRIDES}
    print(f"[A:{ds['name']}] stride8 ZOH: mean={rows[8]['mean']:+.4f} "
          f"std={rows[8]['std']:.4f} lose>0.02={rows[8]['frac_lose_gt_0.02']:.2f} "
          f"improve={rows[8]['frac_improve']:.2f} best-stride hist={hist}")
    print(f"[A:{ds['name']}] stride8 downsampled: mean={downsampled['mean']:+.4f} "
          f"std={downsampled['std']:.4f} lose>0.02={downsampled['frac_lose_gt_0.02']:.2f} "
          f"improve={downsampled['frac_improve']:.2f} n={downsampled['n']}")
    return {'per_stride': {str(s): rows[s] for s in STRIDES},
            'best_stride_hist': hist, 'aucs': aucs, 'valid': valid,
            'downsampled_stride8': downsampled}


# ------------------------------------------------------------------ Step B

def uniform_stride_for(B):
    return int(round(1.0 / B))


def event_recall_of(ds, sp):
    from vlmvalue import event_stats
    recs = []
    for i in range(len(ds['lens'])):
        n = ds['lens'][i]
        sc = held_series(ds['scores'][i, :n], int(sp[i]), n)
        es = event_stats(sc, ds['labels'][i, :n], 0.5)
        if es:
            recs.append(es[0])
    return float(np.mean(recs))


def step_b(ds, aucs, valid):
    n_vids = len(ds['lens'])
    total_snips = int(ds['lens'].sum())
    out = []
    for B in BUDGETS:
        u = uniform_stride_for(B)
        cost_u = total_snips / u
        # uniform
        sp_u = np.full(n_vids, u)
        auc_u, ap_u = pooled_auc_ap(ds, sp_u)
        # oracle: greedy on (capped) degradation saved per snippet spent
        vid_idx = np.flatnonzero(valid)
        base_d = {i: aucs[u][i] - aucs[1][i] for i in vid_idx}   # uniform deg
        coarse = 16
        d16 = {i: aucs[coarse][i] - aucs[1][i] for i in vid_idx}
        options = []
        for i in vid_idx:
            n = ds['lens'][i]
            for s in STRIDES:
                if s >= u:
                    continue
                saved = max(CAP, base_d[i] - (aucs[s][i] - aucs[1][i]))
                cost = (1.0 / s - 1.0 / u) * n
                if saved > 0 and cost > 0:
                    options.append((saved / cost, i, s, cost))
        options.sort(key=lambda x: -x[0])
        budget = total_snips * B
        sp_o = np.full(n_vids, u, dtype=float)
        spent = cost_u
        used = set()
        for ratio, i, s, cost in options:
            if i in used:
                continue
            if spent + cost <= budget:
                sp_o[i] = s
                spent += cost
                used.add(i)
        auc_o, ap_o = pooled_auc_ap(ds, sp_o)
        # random at equal cost: two-point mixture hitting cost B exactly
        # dense stride d, sparse u, q dense: q/d + (1-q)/u = B
        # choose d < u: q = (1/u - B) / (1/u - 1/d)
        d_dense = max(1, u // 2)
        q = (1.0 / u - B) / (1.0 / u - 1.0 / d_dense)
        r = np.random.RandomState(SEED)
        pick = r.rand(n_vids) < q
        sp_r = np.where(pick, d_dense, u).astype(float)
        auc_r, ap_r = pooled_auc_ap(ds, sp_r)
        out.append({'budget': B, 'uniform_stride': u, 'dense_stride': d_dense,
                    'q_dense': float(q),
                    'uniform_auc': auc_u, 'uniform_ap': ap_u,
                    'oracle_auc': auc_o, 'oracle_ap': ap_o,
                    'random_auc': auc_r, 'random_ap': ap_r,
                    'headroom_auc': auc_o - auc_u,
                    'uniform_event_recall': event_recall_of(ds, sp_u),
                    'oracle_event_recall': event_recall_of(ds, sp_o),
                    'random_event_recall': event_recall_of(ds, sp_r),
                    'oracle_cost_frac': spent / total_snips})
        print(f"[B:{ds['name']}] B={B}: uniform={auc_u:.4f} "
              f"oracle={auc_o:.4f} headroom={auc_o - auc_u:+.4f} "
              f"random={auc_r:.4f}")
    return out


# ------------------------------------------------------------------ Step C

def motion_feats(mgrids):
    """Features from a set of analysed snippet motion grids (n,48), causal."""
    if len(mgrids) == 0:
        return {k: 0.0 for k in ('mot_mean', 'mot_var', 'grid_entropy',
                                 'entropy_change', 'scene_change_rate')}
    x = mgrids.astype(np.float64)
    density = np.linalg.norm(x, axis=1)
    # activity distribution over cells, averaged
    prof = np.abs(x).mean(axis=0)
    prof = prof / max(prof.sum(), 1e-9)
    ent = float(-(prof * np.log(prof + 1e-12)).sum())
    # per-snippet entropy of the grid itself
    eps = 1e-9
    p = np.abs(x) / np.maximum(np.abs(x).sum(axis=1, keepdims=True), eps)
    ents = -(p * np.log(p + eps)).sum(axis=1)
    if len(x) > 1:
        change = float(np.abs(np.diff(ents)).mean())
        cos = (x[:-1] * x[1:]).sum(axis=1) / np.maximum(
            np.linalg.norm(x[:-1], axis=1) * np.linalg.norm(x[1:], axis=1), eps)
        scr = float((cos < 0.6).mean())
    else:
        change, scr = 0.0, 0.0
    return {'mot_mean': float(density.mean()), 'mot_var': float(density.var()),
            'grid_entropy': ent, 'entropy_change': change,
            'scene_change_rate': scr}


def score_feats(s_sub):
    s = np.asarray(s_sub, dtype=np.float64)
    if len(s) < 2:
        return {k: 0.0 for k in ('roughness', 'std', 'mx', 'rng',
                                 'lag1', 'frac_hi')}
    d = np.diff(s)
    lag1 = 0.0
    if s.std() > 0:
        c = np.corrcoef(s[:-1], s[1:])[0, 1]
        lag1 = float(c) if np.isfinite(c) else 0.0
    return {'roughness': float(np.abs(d).mean()), 'std': float(s.std()),
            'mx': float(s.max()), 'rng': float(s.max() - s.min()),
            'lag1': lag1, 'frac_hi': float((s > 0.5).mean())}


def video_features(ds, i, stride):
    n = ds['lens'][i]
    idx = np.arange(0, n, stride)
    sf = score_feats(ds['scores'][i, idx])
    mf = motion_feats(ds['motion'][i, idx])
    f = {**sf, **mf}
    return f


def ridge_fit(X, y, lam=1.0):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    A = Xb.T @ Xb + lam * np.eye(Xb.shape[1])
    return np.linalg.solve(A, Xb.T @ y)


def ridge_pred(W, X):
    Xb = np.hstack([X, np.ones((len(X), 1))])
    return Xb @ W


def step_c(ds, aucs, valid, ref_stride=8):
    vid_idx = np.flatnonzero(valid)
    # UNCAPPED degradation as the prediction target (the cap governs oracle
    # harvesting only; a predictor may legitimately target signed sensitivity)
    y = np.array([aucs[ref_stride][i] - aucs[1][i] for i in vid_idx])
    F = [video_features(ds, i, ref_stride) for i in vid_idx]
    names = list(F[0].keys())
    X = np.array([[f[k] for k in names] for f in F])
    # standardise
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Xs = (X - mu) / sd
    # correlations
    corrs = {}
    for j, k in enumerate(names):
        if Xs[:, j].std() > 0:
            corrs[k] = float(np.corrcoef(Xs[:, j], y)[0, 1])
        else:
            corrs[k] = 0.0
    # CV by video (5-fold), ridge on top-|corr| features (all, small)
    r = np.random.RandomState(SEED)
    order = r.permutation(len(vid_idx))
    folds = np.array_split(order, 5)
    preds = np.zeros(len(vid_idx))
    for f in folds:
        tr = np.setdiff1d(order, f)
        W = ridge_fit(Xs[tr], y[tr])
        preds[f] = ridge_pred(W, Xs[f])
    model_corr = float(np.corrcoef(preds, y)[0, 1])
    # quartile separation
    qs = np.percentile(preds, [25, 75])
    m1 = preds <= qs[0]
    m4 = preds >= qs[1]
    q1 = float(y[m1].mean()) if m1.any() else float('nan')
    q4 = float(y[m4].mean()) if m4.any() else float('nan')
    # per-window threshold: window-level predicted-sensitivity median from CV
    window_preds = []
    for i in vid_idx:
        n = ds['lens'][i]
        for w0 in range(0, n, WINDOW):
            idx = np.arange(w0, min(w0 + WINDOW, n), ref_stride)
            if len(idx) < 4:
                continue
            sf = score_feats(ds['scores'][i, idx])
            mf = motion_feats(ds['motion'][i, idx])
            fw = np.array([[sf[k] for k in names[:6]] + [mf[k] for k in names[6:]]])
            fw = (fw - mu) / sd
            window_preds.append(float(ridge_pred(W_all if False else ridge_fit(Xs, y), fw)[0]))
    theta = float(np.median(window_preds))
    print(f"[C:{ds['name']}] model corr (5-fold by video)={model_corr:.3f} "
          f"quartile sep={q4 - q1:+.4f} (q1={q1:.4f} q4={q4:.4f}) "
          f"top feats={sorted(corrs.items(), key=lambda kv: -abs(kv[1]))[:3]}")
    return {'target_stride': ref_stride, 'feature_corr': corrs,
            'model_cv_corr': model_corr,
            'quartile_low': float(q1), 'quartile_high': float(q4),
            'window_theta': theta, 'names': names, 'mu': mu.tolist(),
            'sd': sd.tolist(), 'W': ridge_fit(Xs, y).tolist()}


# ------------------------------------------------------------------ Step D

def predict_window(ds, i, w0, stride, ctx):
    names, mu, sd, W = ctx['names'], ctx['mu'], ctx['sd'], ctx['W']
    n = ds['lens'][i]
    idx = np.arange(w0, min(w0 + WINDOW, n), stride)
    if len(idx) < 4:
        return None
    sf = score_feats(ds['scores'][i, idx])
    mf = motion_feats(ds['motion'][i, idx])
    fw = np.array([[sf[k] for k in names[:6]] + [mf[k] for k in names[6:]]])
    fw = (fw - np.array(mu)) / np.array(sd)
    return float(ridge_pred(np.array(W), fw)[0])


def step_d(ds, ctx_c, aucs, valid):
    n_vids = len(ds['lens'])
    total_snips = int(ds['lens'].sum())
    theta = ctx_c['window_theta']
    out = []
    for B in BUDGETS:
        u = uniform_stride_for(B)
        d_dense = max(1, u // 2)
        q = (1.0 / u - B) / (1.0 / u - 1.0 / d_dense) if u > 1 else 0.0
        sp_u = np.full(n_vids, u)
        auc_u, ap_u = pooled_auc_ap(ds, sp_u)
        # oracle (recompute quickly from step_b results passed in? recompute)
        vid_idx = np.flatnonzero(valid)
        base_d = {i: aucs[u][i] - aucs[1][i] for i in vid_idx}
        options = []
        for i in vid_idx:
            n = ds['lens'][i]
            for s in STRIDES:
                if s >= u:
                    continue
                saved = max(CAP, base_d[i] - (aucs[s][i] - aucs[1][i]))
                cost = (1.0 / s - 1.0 / u) * n
                if saved > 0 and cost > 0:
                    options.append((saved / cost, i, s, cost))
        options.sort(key=lambda x: -x[0])
        budget = total_snips * B
        sp_o = np.full(n_vids, u, dtype=float)
        spent = total_snips / u
        used = set()
        for ratio, i, s, cost in options:
            if i in used or spent + cost > budget:
                continue
            sp_o[i] = s
            spent += cost
            used.add(i)
        auc_o, ap_o = pooled_auc_ap(ds, sp_o)
        # random static
        r = np.random.RandomState(SEED)
        pick = r.rand(n_vids) < q
        auc_rs, ap_rs = pooled_auc_ap(ds, np.where(pick, d_dense, u).astype(float))
        # adaptive per-video static: top-q fraction by predicted sensitivity
        sens = np.zeros(n_vids)
        for i in vid_idx:
            n = ds['lens'][i]
            idx = np.arange(0, n, u)
            sf = score_feats(ds['scores'][i, idx])
            mf = motion_feats(ds['motion'][i, idx])
            fw = np.array([[sf[k] for k in ctx_c['names'][:6]]
                           + [mf[k] for k in ctx_c['names'][6:]]])
            fw = (fw - np.array(ctx_c['mu'])) / np.array(ctx_c['sd'])
            sens[i] = float(ridge_pred(np.array(ctx_c['W']), fw)[0])
        thr = np.quantile(sens[vid_idx], 1 - q) if q > 0 else np.inf
        sp_pv = np.where(sens >= thr, d_dense, u).astype(float)
        cost_pv = sum(ds['lens'][i] / sp_pv[i] for i in range(n_vids)) / total_snips
        auc_pv, ap_pv = pooled_auc_ap(ds, sp_pv)
        # adaptive per-window online (previous-window features only)
        per_video_gt = xd_frame_gt() if ds['name'] == 'xd' else None
        sp_pw_series = np.zeros(n_vids)  # for cost accounting
        cost_pw = 0.0
        pw_parts = []
        for i in range(n_vids):
            n = ds['lens'][i]
            w = 0
            stride_prev = u
            series_parts = []
            cost_i = 0.0
            while w < n:
                # choose stride for THIS window from the PREVIOUS window's
                # analysed snippets only (strictly causal)
                if w == 0:
                    s_use = u
                else:
                    pw = predict_window(ds, i, w - WINDOW, stride_prev, ctx_c)
                    s_use = d_dense if (pw is not None and pw > theta) else u
                stride_prev = s_use
                idx = np.arange(w, min(w + WINDOW, n), s_use)
                held = np.repeat(ds['scores'][i, idx], s_use)[:min(WINDOW, n - w)]
                series_parts.append(held)
                cost_i += len(idx)
                w += WINDOW
            sp_pw_series[i] = cost_i / n
            cost_pw += cost_i
            pw_parts.append(np.concatenate(series_parts))
        cost_pw /= total_snips
        if ds['name'] == 'ucf':
            sc = np.concatenate([np.repeat(p, SNIP) for p in pw_parts])
            nn = min(len(sc), len(ds['gt_concat']))
            auc_pw, ap_pw = auc_rank(sc[:nn], ds['gt_concat'][:nn]), ap_score(sc[:nn], ds['gt_concat'][:nn])
        else:
            sc = np.concatenate([np.repeat(p, SNIP) for p in pw_parts])
            gt = np.concatenate(per_video_gt)
            nn = min(len(sc), len(gt))
            auc_pw, ap_pw = auc_rank(sc[:nn], gt[:nn]), ap_score(sc[:nn], gt[:nn])
        # random per-window at matched realized cost: prob p of dense per window
        # match E[cost] = cost_pw: p*(1/d_dense)+(1-p)*(1/u) = cost_pw
        p_dense = (cost_pw - 1.0 / u) / (1.0 / d_dense - 1.0 / u) \
            if d_dense != u else 0.0
        p_dense = min(max(p_dense, 0.0), 1.0)
        r3 = np.random.RandomState(SEED + 2)
        rw_parts = []
        for i in range(n_vids):
            n = ds['lens'][i]
            series_parts = []
            w = 0
            while w < n:
                s_use = d_dense if r3.rand() < p_dense else u
                idx = np.arange(w, min(w + WINDOW, n), s_use)
                held = np.repeat(ds['scores'][i, idx], s_use)[:min(WINDOW, n - w)]
                series_parts.append(held)
                w += WINDOW
            rw_parts.append(np.concatenate(series_parts))
        if ds['name'] == 'ucf':
            sc = np.concatenate([np.repeat(p, SNIP) for p in rw_parts])
            nn = min(len(sc), len(ds['gt_concat']))
            auc_rw, ap_rw = auc_rank(sc[:nn], ds['gt_concat'][:nn]), ap_score(sc[:nn], ds['gt_concat'][:nn])
        else:
            sc = np.concatenate([np.repeat(p, SNIP) for p in rw_parts])
            gt = np.concatenate(per_video_gt)
            nn = min(len(sc), len(gt))
            auc_rw, ap_rw = auc_rank(sc[:nn], gt[:nn]), ap_score(sc[:nn], gt[:nn])
        head = auc_o - auc_u
        capture = (auc_pw - auc_u) / head if abs(head) > 1e-9 else 0.0
        print(f"[D:{ds['name']}] B={B}: uniform={auc_u:.4f} rand_static={auc_rs:.4f} "
              f"adapt_pv={auc_pv:.4f}(cost {cost_pv:.3f}) "
              f"adapt_pw={auc_pw:.4f}(cost {cost_pw:.3f}) "
              f"rand_pw={auc_rw:.4f} oracle={auc_o:.4f} capture={capture:+.1%}")
        out.append({'budget': B, 'uniform_auc': auc_u, 'uniform_ap': ap_u,
                    'random_static_auc': auc_rs, 'random_static_ap': ap_rs,
                    'adaptive_pervideo_auc': auc_pv, 'adaptive_pervideo_ap': ap_pv,
                    'adaptive_pervideo_cost': float(cost_pv),
                    'adaptive_perwindow_auc': auc_pw, 'adaptive_perwindow_ap': ap_pw,
                    'adaptive_perwindow_cost': float(cost_pw),
                    'random_perwindow_auc': auc_rw, 'random_perwindow_ap': ap_rw,
                    'oracle_auc': auc_o, 'oracle_ap': ap_o,
                    'headroom_auc': float(head),
                    'capture_perwindow': float(capture),
                    'cameras_per_4gpu_server': 14400.0 / (359.273 * B)})
    return out


# ------------------------------------------------------------------ figures

def figures(het, ctrl):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    figdir = os.path.join(OUT, 'figures')
    os.makedirs(figdir, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, (name, h) in zip(axes, het.items()):
        data = [[ (h['aucs'][s][h['valid']] - h['aucs'][1][h['valid']]) ]
                for s in STRIDES]
        ax.boxplot([d[0] for d in data], labels=[str(s) for s in STRIDES],
                   showfliers=False)
        ax.axhline(0, color='k', lw=0.8)
        ax.set_xlabel('stride')
        ax.set_ylabel('per-video AUC degradation vs stride 1')
        ax.set_title(f'E0.9 heterogeneity ({name})')
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, 'e09_heterogeneity.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    styles = [('uniform', 'o-', 'k'), ('random_static', 'v--', 'gray'),
              ('adaptive_pervideo', 's-', 'tab:blue'),
              ('adaptive_perwindow', '^-', 'tab:red'),
              ('oracle', 'x', 'tab:green')]
    for name, h in het.items():
        for key, mk, col in styles:
            xs = [c['cameras_per_4gpu_server'] for c in ctrl[name]]
            ys = [c[f'{key}_auc'] for c in ctrl[name]]
            ax.plot(xs, ys, mk, color=col,
                    label=f'{name}:{key}' if name == 'ucf' else None,
                    alpha=0.9 if name == 'ucf' else 0.45)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('cameras per 4-GPU server (equal cost)')
    ax.set_ylabel('pooled AUC')
    ax.set_title('E0.9 frontier: pooled AUC vs cameras served')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(figdir, 'e09_frontier.png'), dpi=150)
    plt.close(fig)
    print('[figures] saved e09_heterogeneity.png, e09_frontier.png')


# ------------------------------------------------------------------- main

def main():
    het, oracle, preds, ctrl = {}, {}, {}, {}
    for loader in (load_ucf, load_xd):
        ds = loader()
        print(f'=== {ds["name"]} ===')
        h = step_a(ds)
        het[ds['name']] = h
        oracle[ds['name']] = step_b(ds, h['aucs'], h['valid'])
        c = step_c(ds, h['aucs'], h['valid'])
        preds[ds['name']] = {k: v for k, v in c.items()
                             if k not in ('names', 'mu', 'sd', 'W')}
        ctrl[ds['name']] = step_d(ds, c, h['aucs'], h['valid'])
    figures(het, ctrl)
    # gates
    g1 = {n: next(r['headroom_auc'] for r in oracle[n] if r['budget'] == 0.25)
          for n in oracle}
    g2 = {n: next(r['capture_perwindow'] for r in ctrl[n] if r['budget'] == 0.25)
          for n in ctrl}
    def gate1v(v):
        return 'REAL HEADROOM' if v >= 0.02 else 'MARGINAL' if v >= 0.005 else 'NO SYSTEM'
    def gate2v(v):
        return 'SYSTEM' if v >= 0.30 else 'MARGINAL' if v >= 0.15 else 'NO SYSTEM'
    verdict = ('SYSTEMS PAPER'
               if all(gate1v(g1[n]) == 'REAL HEADROOM' for n in g1)
               and all(gate2v(g2[n]) == 'SYSTEM' for n in g2)
               else 'MEASUREMENT PAPER')
    doc = {'config': {'strides': STRIDES, 'budgets': BUDGETS,
                      'window_snippets': WINDOW, 'cv': 'by_video', 'seed': SEED,
                      'per_video_gain_cap': CAP},
           'heterogeneity': {n: {'per_stride': het[n]['per_stride'],
                                 'best_stride_hist': het[n]['best_stride_hist']}
                             for n in het},
           'oracle': oracle, 'predictors': preds, 'controller': ctrl,
           'gate1_headroom_at_quarter': g1,
           'gate2_capture_at_quarter': g2,
           'gate1': {n: gate1v(g1[n]) for n in g1},
           'gate2': {n: gate2v(g2[n]) for n in g2},
           'verdict': verdict}
    with open(os.path.join(OUT, 'e09.json'), 'w') as f:
        json.dump(doc, f, indent=2)
    print(f"\nGATE1 headroom@1/4: {g1} -> {[gate1v(v) for v in g1.values()]}")
    print(f"GATE2 capture@1/4: {g2} -> {[gate2v(v) for v in g2.values()]}")
    print(f"VERDICT: {verdict}")
    print(f"[saved] {os.path.join(OUT, 'e09.json')}")


if __name__ == '__main__':
    main()
