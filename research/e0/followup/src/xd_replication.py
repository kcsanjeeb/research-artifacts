"""XD-Violence replication of E0 findings 1 (contention), 2 (cache dead),
4 (VLM doesn't pay) + E0.8 budget/oversampling, following VadCLIP's own
XD protocol exactly (xd_test.py / make_gt_xd.py / xd_option.py).

Phases:
  --score        tier-1 CLIPVAD scoring (GPU, minutes); AP sanity vs 84.51
  --demand       finding 1: N-stream escalation sim, crossover vs CAPACITY
  --redundancy   finding 2: causal NN over XD features (scene+motion)
  --stride       E0.8: stride sweep (ZOH) on AP/AUC
  --vlm-freeze   finding 4: freeze escalated sets (band 8.6% + top-K)
  --vlm-extract  extract clips from XD videos (12 frames, 448x448)
  analysis of VLM cells reuses vlm_calibrated machinery at --vlm-analyze
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

E0 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(E0, 'src'))
import common  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlmvalue import auc_rank, ap_score, event_stats  # noqa: E402
from vlm_calibrated import pava, iso_apply  # noqa: E402

OUT = os.path.join(E0, 'followup', 'results')
XD_FEAT = os.environ.get('XD_FEAT_DIR',
                         os.path.join(E0, 'data', 'XDTestClipFeatures'))
XD_VIDS = os.path.join(E0, 'data', 'xd_test_videos')
XD_CKPT = os.path.join(E0, 'data', 'model_xd.pth')
VADCLIP = os.path.join(E0, 'third_party', 'VadCLIP')
SNIP = 16
XD_FPS = 24.0
TAU = 0.5
CAPACITY = 3.043970158442251  # E0 capacity.json, 4x V100
SEED = 0

XD_LABEL_MAP = {'A': 'normal', 'B1': 'fighting', 'B2': 'shooting',
                'B4': 'riot', 'B5': 'abuse', 'B6': 'car accident',
                'G': 'explosion'}


# ------------------------------------------------------------------ metadata

def xd_rows():
    """Test rows in csv order: (video_id, feature_path)."""
    import pandas as pd
    df = pd.read_csv(os.path.join(VADCLIP, 'list', 'xd_CLIP_rgbtest.csv'))
    rows = []
    for _, r in df.iterrows():
        base = os.path.basename(r['path'])
        vid = base.replace('__0.npy', '')
        rows.append({'video_id': vid,
                     'feature_path': os.path.join(XD_FEAT, vid + '.npy')})
    return rows


def xd_gt_segments():
    """annotations.txt -> {video_id: [(start_frame, end_frame), ...]}."""
    ann = {}
    for line in open(os.path.join(VADCLIP, 'list', 'annotations.txt')):
        p = line.strip().split()
        if not p:
            continue
        name = p[0].replace('.mp4', '')
        segs = [(int(p[i]), int(p[i + 1])) for i in range(1, len(p) - 1, 2)]
        ann[name] = segs
    return ann


def build_xd_labels(rows):
    """Per-video snippet labels, mirroring make_gt_xd.py:
    frame GT over (n_snip+1)*16 frames, then drop the last 16 -> n_snip*16."""
    ann = xd_gt_segments()
    per_video = []
    for r in rows:
        n = np.load(r['feature_path']).shape[0]
        lens = (n + 1) * SNIP
        gt = np.zeros(lens, dtype=np.float32)
        if 'label_A' not in r['video_id']:
            for a, b in ann.get(r['video_id'], []):
                gt[a:b] = 1.0
        per_video.append(gt[:-SNIP].astype(np.int8))
    return per_video


# ------------------------------------------------------------------- scoring

def phase_score():
    import torch
    sys.path.insert(0, common.VADCLIP_SRC)
    from demand import _patch_clip_load
    from model import CLIPVAD
    from utils.tools import get_batch_mask, get_prompt_text

    state = _patch_clip_load(XD_CKPT)
    device = 'cuda'
    maxlen = 256
    model = CLIPVAD(7, 512, maxlen, 512, 1, 1, 64, 10, 10, device)
    model.load_state_dict(state)
    model.to(device).eval()
    prompt_text = get_prompt_text(XD_LABEL_MAP)

    rows = xd_rows()
    per_video = build_xd_labels(rows)
    gt_concat = np.concatenate(per_video)

    s1_all, s2_all = {}, {}
    with torch.no_grad():
        for i, r in enumerate(rows):
            f = np.load(r['feature_path'])
            length = len(f)
            pad = (-length) % maxlen
            if pad:
                f = np.concatenate([f, np.zeros((pad, f.shape[1]), dtype=f.dtype)])
            v = torch.tensor(f).float().to(device).reshape(-1, maxlen, 512)
            n_seg = (length + maxlen - 1) // maxlen
            lengths = torch.tensor([min(maxlen, length - j * maxlen)
                                    for j in range(n_seg)], dtype=torch.int)
            padding_mask = get_batch_mask(lengths, maxlen).to(device)
            _, logits1, logits2 = model(v, padding_mask, prompt_text, lengths)
            logits1 = logits1.reshape(-1, logits1.shape[2])
            logits2 = logits2.reshape(-1, logits2.shape[2])
            s1_all[r['video_id']] = torch.sigmoid(
                logits1[0:length].squeeze(-1)).cpu().numpy()
            s2_all[r['video_id']] = (1 - logits2[0:length].softmax(dim=-1)[:, 0]
                                     ).cpu().numpy()

    def frame_eval(s_all):
        concat = np.concatenate([np.repeat(s_all[r['video_id']], SNIP)
                                 for r in rows])
        n = min(len(concat), len(gt_concat))
        return (auc_rank(concat[:n], gt_concat[:n]),
                ap_score(concat[:n], gt_concat[:n]))

    auc1, ap1 = frame_eval(s1_all)
    auc2, ap2 = frame_eval(s2_all)
    print(f"[score] XD: AUC1={auc1:.4f} AP1={ap1:.4f} | AUC2={auc2:.4f} "
          f"AP2={ap2:.4f}  (published AP 84.51)")
    used, auc_used, ap_used = (s1_all, auc1, ap1)  # visual branch (as UCF)
    vids = [r['video_id'] for r in rows]
    lens = np.array([len(used[v]) for v in vids])
    mx = lens.max()
    scores = np.zeros((len(vids), mx), dtype=np.float32)
    for i, v in enumerate(vids):
        scores[i, :lens[i]] = used[v]
    np.savez(os.path.join(OUT, 'xd_tier1_scores.npz'),
             video_ids=np.array(vids), scores=scores, lens=lens,
             auc1=auc1, ap1=ap1, auc2=auc2, ap2=ap2)
    print(f"[score] saved {len(vids)} videos, {lens.sum()} snippets")


def load_xd():
    d = np.load(os.path.join(OUT, 'xd_tier1_scores.npz'), allow_pickle=True)
    vids = d['video_ids'].tolist()
    rows = xd_rows()
    per_video = build_xd_labels(rows)
    labels = np.zeros((len(vids), d['lens'].max()), dtype=np.int8)
    for i, v in enumerate(vids):
        n = d['lens'][i]
        frame_gt = per_video[i][:n * SNIP]          # frame-level, n*16 entries
        labels[i, :n] = frame_gt.reshape(n, SNIP).max(axis=1)  # snippet label
    return vids, d['scores'], d['lens'], labels


# ---------------------------------------------------------------- finding 1

def phase_demand():
    vids, scores, lens, labels = load_xd()
    flat = np.concatenate([scores[i, :lens[i]] for i in range(len(vids))])
    # margin at 8.6% on XD (same stated rule as UCF)
    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        if (np.abs(flat - TAU) < mid).mean() < 0.086:
            lo = mid
        else:
            hi = mid
    margin = (lo + hi) / 2
    print(f"[demand] XD margin@8.6%={margin:.4f}")

    spf = SNIP / XD_FPS
    curve = []
    for period in (0.5, 1.0, 2.0):
        for N in (1, 2, 4, 8, 16, 32, 64, 128, 256):
            cams = [[] for _ in range(N)]
            for i in range(len(vids)):
                cams[i % N].append(i)
            cam_tl, cam_dur = [], []
            for cam in cams:
                tl = [(vi, si) for vi in cam for si in range(lens[vi])]
                cam_tl.append(tl)
                cam_dur.append(len(tl) * spf)
            horizon = max(max(cam_dur), 900.0)
            bins = np.zeros(int(horizon) + 1)
            for tl, dur in zip(cam_tl, cam_dur):
                if not tl:
                    continue
                t = 0.0
                while t < horizon:
                    pos = t % dur
                    vi, si = tl[min(int(pos / spf), len(tl) - 1)]
                    if abs(scores[vi, si] - TAU) < margin:
                        bins[int(t)] += 1
                    t += period
            rec = {'N': N, 'clip_period_s': period,
                   'mean_demand': float(bins.mean()),
                   'p95_demand': float(np.percentile(bins, 95)),
                   'p99_demand': float(np.percentile(bins, 99))}
            rec['rho_mean'] = rec['mean_demand'] / CAPACITY
            rec['rho_p95'] = rec['p95_demand'] / CAPACITY
            curve.append(rec)
    for period in (0.5, 1.0, 2.0):
        cs = [c for c in curve if c['clip_period_s'] == period]
        cm = next((c['N'] for c in cs if c['mean_demand'] > CAPACITY), None)
        c95 = next((c['N'] for c in cs if c['p95_demand'] > CAPACITY), None)
        print(f"  P={period}s: N_crossover_mean={cm} N_crossover_p95={c95}")
    _save('xd_demand.json', {'capacity': CAPACITY, 'tau': TAU,
                             'margin_at_8.6pct': float(margin),
                             'curve': curve})


# ---------------------------------------------------------------- finding 2

def phase_redundancy():
    vids, scores, lens, labels = load_xd_labels_only() if False else load_xd()
    # scene features
    feats = []
    for v in vids:
        feats.append(np.load(os.path.join(XD_FEAT, v + '.npy')).astype(np.float32))
    mx = max(len(f) for f in feats)
    e_scene = np.zeros((len(vids), mx, 512), dtype=np.float32)
    for i, f in enumerate(feats):
        e_scene[i, :len(f)] = f
    e_scene /= np.maximum(np.linalg.norm(e_scene, axis=-1, keepdims=True), 1e-9)

    # motion features from videos (cached)
    mcache = os.path.join(OUT, 'xd_motion_features.npz')
    if os.path.exists(mcache):
        dm = np.load(mcache)
        e_motion = dm['motion']
    else:
        e_motion = _xd_motion(vids, lens)
        np.savez(mcache, motion=e_motion)
    e_motion = e_motion / np.maximum(np.linalg.norm(e_motion, axis=-1,
                                                    keepdims=True), 1e-9)

    # timeline: 32 virtual cameras, single pass (same as UCF)
    N = 32
    spf = SNIP / XD_FPS
    cams = [[] for _ in range(N)]
    for i in range(len(vids)):
        cams[i % N].append(i)
    events = []
    for c, cam in enumerate(cams):
        t = 0.0
        for vi in cam:
            for si in range(lens[vi]):
                events.append((t, c, vi, si))
                t += spf
    events.sort(key=lambda e: (e[0], e[1]))
    ev_vi = np.array([e[2] for e in events])
    ev_si = np.array([e[3] for e in events])
    ev_lab = labels[ev_vi, ev_si]
    print(f"[redundancy] {len(events)} clips, anomalous={ev_lab.mean():.4f}")

    # kmeans scene clusters (k=20), video-level
    from sklearn.cluster import KMeans
    means = np.stack([e_scene[i, :lens[i]].mean(axis=0) for i in range(len(vids))])
    means /= np.maximum(np.linalg.norm(means, axis=1, keepdims=True), 1e-9)
    clus = KMeans(n_clusters=20, random_state=SEED, n_init=10).fit(means).labels_
    ev_clus = clus[ev_vi]

    import faiss
    thetas = np.round(np.arange(0.80, 0.991, 0.01), 2)
    out = {}
    for alpha in (0.0, 0.5, 1.0):
        keys = np.concatenate([alpha * e_scene[ev_vi, ev_si],
                               (1 - alpha) * e_motion[ev_vi, ev_si]], axis=1)
        keys = (keys / np.maximum(np.linalg.norm(keys, axis=1, keepdims=True),
                                  1e-9)).astype(np.float32)
        index = faiss.IndexFlatIP(keys.shape[1])
        top_sim = np.full(len(events), -1.0, dtype=np.float32)
        nn_pos = np.full(len(events), -1, dtype=np.int64)
        for pos in range(len(events)):
            k = keys[pos:pos + 1]
            if index.ntotal > 0:
                sim, j = index.search(k, 1)
                top_sim[pos] = sim[0, 0]
                nn_pos[pos] = j[0, 0]
            index.add(k)
        nn_vi = np.where(nn_pos >= 0, ev_vi[np.clip(nn_pos, 0, None)], -1)
        nn_clus = np.where(nn_pos >= 0, ev_clus[np.clip(nn_pos, 0, None)], -1)
        types = np.where(nn_vi == ev_vi, 'T1',
                         np.where(nn_clus == ev_clus, 'T2', 'T3'))
        nn_lab = np.where(nn_pos >= 0, ev_lab[np.clip(nn_pos, 0, None)], -1)
        rows = []
        for th in thetas:
            hit = (nn_pos >= 0) & (top_sim >= th)
            if hit.sum() == 0:
                continue
            agree = float((ev_lab[hit] == nn_lab[hit]).mean())
            # base-rate-corrected: const-majority agreement + suppressed anomaly frac
            const = float(max(ev_lab.mean(), 1 - ev_lab.mean()))
            n_anom = max((ev_lab == 1).sum(), 1)
            suppressed = float((hit & (ev_lab == 1) & (nn_lab == 0)).sum() / n_anom)
            rows.append({'theta': float(th), 'hit_rate': float(hit.mean()),
                         'label_agreement': agree,
                         'const_majority_agreement': const,
                         'suppressed_anomaly_frac': suppressed,
                         'by_type': {t: float(((types == t) & hit).sum() / len(hit))
                                     for t in ('T1', 'T2', 'T3')}})
        # operating point: max hit rate with suppressed anomaly <= 1%
        ok = [r for r in rows if r['suppressed_anomaly_frac'] <= 0.01]
        op = max(ok, key=lambda r: r['hit_rate']) if ok else None
        out[str(alpha)] = {'curve': rows, 'operating_point': op}
        if op:
            print(f"  alpha={alpha}: theta*={op['theta']} hit={op['hit_rate']:.3f} "
                  f"agree={op['label_agreement']:.3f} (const {op['const_majority_agreement']:.3f}) "
                  f"T1={op['by_type']['T1']:.3f} T2={op['by_type']['T2']:.4f} "
                  f"T3={op['by_type']['T3']:.4f} suppressed={op['suppressed_anomaly_frac']:.4f}")
    _save('xd_redundancy.json', {'n_clips': len(events), 'alphas': out,
                                 'note': 'XD has no shared-site cameras; T2 is a '
                                         'k-means scene-cluster proxy as on UCF'})


def _xd_motion(vids, lens):
    """6x8 motion grid per snippet from the XD videos (48x64 diffs)."""
    import cv2
    from decord import VideoReader, cpu
    mx = lens.max()
    out = np.zeros((len(vids), mx, 48), dtype=np.float32)
    for i, v in enumerate(vids):
        vf = os.path.join(XD_VIDS, v + '.mp4')
        if not os.path.exists(vf):
            continue
        vr = VideoReader(vf, ctx=cpu(0), num_threads=2)
        n_snips = min(lens[i], len(vr) // SNIP)
        prev = None
        acc = np.zeros((48, 64), dtype=np.float64)
        cnt = 0
        snip = 0
        for fi in range(len(vr)):
            if snip >= n_snips:
                break
            fr = vr[fi].asnumpy()
            g = cv2.cvtColor(fr, cv2.COLOR_RGB2GRAY)
            g = cv2.resize(g, (64, 48), interpolation=cv2.INTER_AREA)
            if prev is not None:
                acc += np.abs(g.astype(np.float64) - prev.astype(np.float64))
                cnt += 1
            prev = g
            if (fi + 1) % SNIP == 0:
                d = acc / max(cnt, 1)
                cell = d.reshape(6, 8, 8, 8)
                out[i, snip] = cell.mean(axis=(1, 3)).astype(np.float32).ravel()
                acc[:] = 0
                cnt = 0
                snip += 1
        if i % 100 == 0:
            print(f"  motion {i}/{len(vids)}", flush=True)
    return out


# --------------------------------------------------------------------- stride

def phase_stride():
    vids, scores, lens, labels = load_xd()
    rows = xd_rows()
    per_video = build_xd_labels(rows)
    gt_concat = np.concatenate(per_video)  # true frame-level GT

    def full_frame(fused):
        concat = np.concatenate([np.repeat(fused[i, :lens[i]], SNIP)
                                 for i in range(len(vids))])
        n = min(len(concat), len(gt_concat))
        return concat[:n], gt_concat[:n]

    out = []
    for stride in (1, 2, 4, 8, 16):
        held = scores.copy()
        for i in range(len(vids)):
            s = scores[i, :lens[i]]
            held[i, :lens[i]] = np.repeat(s[::stride], stride)[:lens[i]]
        fs, g = full_frame(held)
        out.append({'stride': stride,
                    'auc_held': auc_rank(fs, g), 'ap_held': ap_score(fs, g)})
        print(f"[stride {stride:>2}] auc={out[-1]['auc_held']:.4f} "
              f"ap={out[-1]['ap_held']:.4f}")
    _save('xd_stride.json', {'sweep': out})


# ---------------------------------------------------------------- finding 4

def phase_vlm_freeze():
    vids, scores, lens, labels = load_xd()
    flat = np.concatenate([scores[i, :lens[i]] for i in range(len(vids))])
    lo, hi = 0.0, 0.5
    for _ in range(60):
        mid = (lo + hi) / 2
        if (np.abs(flat - TAU) < mid).mean() < 0.086:
            lo = mid
        else:
            hi = mid
    margin = (lo + hi) / 2
    vis, sis = [], []
    for vi in range(len(vids)):
        esc = np.abs(scores[vi, :lens[vi]] - TAU) < margin
        for si in np.flatnonzero(esc):
            vis.append(vi)
            sis.append(int(si))
    np.savez(os.path.join(OUT, 'xd_escalated_band.npz'),
             video_ids=np.array(vids), vi=np.array(vis), si=np.array(sis),
             margin=margin)
    print(f"[vlm-freeze] band: {len(vis)} clips (margin={margin:.4f})")
    # top-K with same budget
    flat_idx = [(scores[vi, si], vi, si) for vi in range(len(vids))
                for si in range(lens[vi])]
    flat_idx.sort(key=lambda x: -x[0])
    K = len(vis)
    top = flat_idx[:K]
    np.savez(os.path.join(OUT, 'xd_escalated_topk.npz'),
             video_ids=np.array(vids),
             vi=np.array([t[1] for t in top]), si=np.array([t[2] for t in top]))
    anom = np.mean([labels[t[1], t[2]] for t in top])
    print(f"[vlm-freeze] topk: {K} clips, anomalous={anom:.3f}")


def _xd_clip_extract(args):
    vi, items, vids, cdir = args
    from decord import VideoReader, cpu
    from PIL import Image
    sys.path.insert(0, common.HOLMES_DIR)
    from holmesvau.internvl_utils import get_index
    vf = os.path.join(XD_VIDS, vids[vi] + '.mp4')
    if not os.path.exists(vf):
        return 0
    vr = VideoReader(vf, ctx=cpu(0), num_threads=1)
    for k, si in items:
        f0 = si * SNIP
        idx = get_index(bound=None, fps=XD_FPS, max_frame=f0 + SNIP - 1,
                        first_idx=f0, num_segments=12)
        idx = [min(int(i), len(vr) - 1) for i in idx]
        for j, fi in enumerate(idx):
            img = Image.fromarray(vr[fi].asnumpy()).convert('RGB')
            img = img.resize((448, 448), Image.BICUBIC)
            img.save(os.path.join(cdir, f"{k:05d}_f{j:02d}.jpg"), quality=90)
    return len(items)


def phase_vlm_extract(gate):
    from concurrent.futures import ProcessPoolExecutor
    d = np.load(os.path.join(OUT, f'xd_escalated_{gate}.npz'), allow_pickle=True)
    vids = d['video_ids'].tolist()
    vis, sis = d['vi'], d['si']
    cdir = os.path.join(E0, 'data', f'xd_clip_cache_{gate}')
    os.makedirs(cdir, exist_ok=True)
    by_video = {}
    for k, (vi, si) in enumerate(zip(vis, sis)):
        by_video.setdefault(vi, []).append((k, si))
    todo = []
    for vi, items in by_video.items():
        need = [(k, si) for k, si in items
                if not os.path.exists(os.path.join(cdir, f"{k:05d}_f11.jpg"))]
        if need:
            todo.append((vi, need, vids, cdir))
    print(f"[vlm-extract {gate}] {sum(len(t[1]) for t in todo)} clips")
    done = 0
    with ProcessPoolExecutor(max_workers=16) as ex:
        for n in ex.map(_xd_clip_extract, todo):
            done += n
    print(f"[vlm-extract {gate}] done {done}")


def phase_vlm_analyze():
    """Finding 4 on XD: HolmesVAU on the escalated sets, isotonic crossfit
    fusion, per-set oracle ceilings, capture fractions. XD-native metric (AP)
    reported alongside AUC."""
    vids, scores, lens, labels = load_xd()
    rows = xd_rows()
    per_video = build_xd_labels(rows)
    gt_concat = np.concatenate(per_video)

    def full_frame(fused):
        concat = np.concatenate([np.repeat(fused[i, :lens[i]], SNIP)
                                 for i in range(len(vids))])
        n = min(len(concat), len(gt_concat))
        return concat[:n], gt_concat[:n]

    fs0, g0 = full_frame(scores)
    auc_t1, ap_t1 = auc_rank(fs0, g0), ap_score(fs0, g0)
    print(f"[vlm-analyze] tier-1 baseline: AUC={auc_t1:.4f} AP={ap_t1:.4f}")

    cells = []
    for gate in ('band', 'topk'):
        d = np.load(os.path.join(OUT, f'xd_escalated_{gate}.npz'),
                    allow_pickle=True)
        vis, sis = d['vi'], d['si']
        vlm = {}
        for f in sorted(glob.glob(os.path.join(OUT,
                                               f'xd_vlm_scores_{gate}_shard*.jsonl'))):
            for line in open(f):
                r = json.loads(line)
                vlm[(r['vi'], r['si'])] = r
        print(f"[{gate}] vlm verdicts: {len(vlm)}/{len(vis)}")
        assert len(vlm) == len(vis)
        gt_pool = labels[vis, sis]
        s_pool = scores[vis, sis]
        # ceilings
        fused = scores.copy()
        fused[vis, sis] = labels[vis, sis]
        fs, g = full_frame(fused)
        ceil_auc = auc_rank(fs, g) - auc_t1
        ceil_ap = ap_score(fs, g) - ap_t1
        pv = np.array([vlm[(vi, si)]['score_b'] for vi, si in zip(vis, sis)],
                      dtype=float)
        # isotonic crossfit by video parity
        cal = {}
        for parity in (0, 1):
            fit_m = (vis % 2) == parity
            app_m = ~fit_m
            kx, ky = pava(pv[fit_m], gt_pool[fit_m])
            vals = iso_apply(kx, ky, pv[app_m])
            for idx, v in zip(np.flatnonzero(app_m), vals):
                cal[idx] = float(v)
        fused = scores.copy()
        for idx, (vi, si) in enumerate(zip(vis, sis)):
            fused[vi, si] = cal[idx]
        fs, g = full_frame(fused)
        dauc = auc_rank(fs, g) - auc_t1
        dap = ap_score(fs, g) - ap_t1
        cell = {'gate': gate, 'span_snippets': 0, 'model': 'HolmesVAU-2B',
                'n_clips': len(vis),
                'anomalous_in_pool': float(gt_pool.mean()),
                'oracle_ceiling_dAUC': float(ceil_auc),
                'oracle_ceiling_dAP': float(ceil_ap),
                'dAUC': float(dauc), 'dAP': float(dap),
                'capture_fraction_auc': float(dauc / ceil_auc),
                'capture_fraction_ap': float(dap / ceil_ap),
                'within_pool_auc_vlm': auc_rank(pv, gt_pool),
                'within_pool_auc_tier1': auc_rank(s_pool, gt_pool),
                'vlm_yes_rate': float((pv > 0.5).mean()),
                'vlm_acc': float(((pv > 0.5).astype(int) == gt_pool).mean()),
                'const_majority_acc': float(max(gt_pool.mean(),
                                                1 - gt_pool.mean()))}
        cells.append(cell)
        print(f"  {gate}: dAUC={dauc:+.4f}/{ceil_auc:+.4f} "
              f"capture={dauc/ceil_auc:+.1%} | dAP={dap:+.4f}/{ceil_ap:+.4f} "
              f"capture={dap/ceil_ap:+.1%} | poolAUC vlm={cell['within_pool_auc_vlm']:.3f} "
              f"t1={cell['within_pool_auc_tier1']:.3f} "
              f"acc={cell['vlm_acc']:.3f}/const={cell['const_majority_acc']:.3f}")
    _save('xd_vlm.json', {'baseline': {'auc_tier1': auc_t1, 'ap_tier1': ap_t1},
                          'cells': cells,
                          'config': {'model': 'HolmesVAU-2B', 'prompt': PROMPT_XD,
                                     'fusion': 'isotonic_crossfit_by_video_parity'}})


PROMPT_XD = ("Does this video contain an anomaly? Answer Yes or No first, "
             "then explain briefly.")


def _save(name, doc):
    with open(os.path.join(OUT, name), 'w') as f:
        json.dump(doc, f, indent=2)
    print(f"[saved] {os.path.join(OUT, name)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--score', action='store_true')
    ap.add_argument('--demand', action='store_true')
    ap.add_argument('--redundancy', action='store_true')
    ap.add_argument('--stride', action='store_true')
    ap.add_argument('--vlm-freeze', action='store_true')
    ap.add_argument('--vlm-extract', default=None)
    ap.add_argument('--vlm-analyze', action='store_true')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.score:
        phase_score()
    if args.demand:
        phase_demand()
    if args.redundancy:
        phase_redundancy()
    if args.stride:
        phase_stride()
    if args.vlm_freeze:
        phase_vlm_freeze()
    if args.vlm_extract:
        phase_vlm_extract(args.vlm_extract)
    if args.vlm_analyze:
        phase_vlm_analyze()


if __name__ == '__main__':
    main()
