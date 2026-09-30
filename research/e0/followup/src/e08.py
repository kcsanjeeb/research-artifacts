"""E0.8 -- where does the GPU budget actually go, and what does each stage buy?

Step 1 (GPU, server): cost per stage, capacity.py protocol (10 warmup + 100
timed, synchronize, p50/p95/p99, VRAM torch+smi, NVML util).
  --verify-spec   (done interactively; recorded in e08.json:tier1_spec)
  --decode        CPU decode (1 and 16 processes) + NVDEC via decord gpu ctx
  --extract       CLIP ViT-B/16 fp16 batch sweep {1,8,16,32,64,128}
  --head          CLIPVAD classifier over precomputed features
  --e2e           decode -> extract -> head end-to-end composition check
Step 2 (CPU): stride sweep on tier-1 scores with zero-order hold.
  --stride
  --analyze       budget model + headline table + iso-budget + e08.json + figs

tier1_spec (verified empirically against the released features):
  CLIP ViT-B/16 image encoder, raw (unnormalized) encode_image output,
  mean-pooled per 16-frame snippet, input 224x224; released features are
  10-crop (VadCLIP uses crop index 5); deployment = 1 crop = 1 forward/frame.
  Live reproduction with the standard CLIP transform gives cosine 0.73-0.92 vs
  the released features (our mp4s are a re-encode; the released features
  reproduce the published AUC exactly, so scoring uses them, throughput is
  measured with the same model+resolution).
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

E0 = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(E0, 'src'))
import common  # noqa: E402
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from vlmvalue import auc_rank, ap_score, event_stats  # noqa: E402

OUT = os.path.join(E0, 'followup', 'results')
SNIP, FPS = 16, 30.0
TAU, MARGIN = 0.5, 0.2603611499071121
SEED = 0
THR = os.path.join(OUT, 'e08_throughput.json')


# ------------------------------------------------------------------- decode

def _decode_one(video_path):
    import cv2
    cap = cv2.VideoCapture(video_path)
    n = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    return n, time.time() - t0


def phase_decode():
    rows = common.load_test_list()
    rng = np.random.RandomState(SEED)
    picks = rng.choice(len(rows), 24, replace=False)
    vids = [common.video_file_for(rows[i]['video_id'], rows[i]['category'])
            for i in picks]

    # single process
    tot_n, tot_t = 0, 0.0
    for v in vids:
        n, t = _decode_one(v)
        tot_n += n
        tot_t += t
    single_fps = tot_n / tot_t
    print(f"[decode] CPU 1 proc: {single_fps:.0f} frames/s "
          f"({single_fps/SNIP:.1f} clips/s)")

    # 16 processes on disjoint videos
    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=16) as ex:
        res = list(ex.map(_decode_one, vids * 4))
    wall = time.time() - t0
    par_fps = sum(r[0] for r in res) / wall
    print(f"[decode] CPU 16 proc: {par_fps:.0f} frames/s "
          f"({par_fps/SNIP:.1f} clips/s)")

    # NVDEC via decord GPU context
    nvdec = {'available': False}
    try:
        from decord import VideoReader, gpu
        vr = VideoReader(vids[0], ctx=gpu(0))
        n = len(vr)
        for _ in range(2):  # warmup
            _ = vr[:min(512, n)].asnumpy()
        t0 = time.time()
        cnt = 0
        step = 512
        for i in range(0, n, step):
            _ = vr[i:i + step].asnumpy()
            cnt += len(vr[i:i + step])
        dt = time.time() - t0
        nvdec = {'available': True, 'frames_per_s': cnt / dt,
                 'clips_per_s': cnt / dt / SNIP}
        print(f"[decode] NVDEC: {cnt/dt:.0f} frames/s ({cnt/dt/SNIP:.1f} clips/s)")
    except Exception as e:
        print(f"[decode] NVDEC unavailable: {type(e).__name__}: {e}")

    res = {'decode_cpu_1proc': {'frames_per_s': single_fps,
                                'clips_per_s': single_fps / SNIP},
           'decode_cpu_16proc': {'frames_per_s': par_fps,
                                 'clips_per_s': par_fps / SNIP, 'cores': 16},
           'decode_nvdec': nvdec}
    _save_thr(res)


# ------------------------------------------------------------------ extract

def _load_clip_model():
    import torch
    sys.path.insert(0, common.VADCLIP_SRC)
    from clip.clip import build_model
    state = torch.load(common.VADCLIP_CKPT, map_location='cpu')
    clip_state = {k[len('clipmodel.'):]: v for k, v in state.items()
                  if k.startswith('clipmodel.')}
    return build_model(clip_state).cuda().eval()


def _frame_pool(n_frames=2048):
    """Pre-decode a pool of real frames at 224x224 (extraction-only timing)."""
    import cv2
    rows = common.load_test_list()
    rng = np.random.RandomState(SEED)
    picks = rng.choice(len(rows), 8, replace=False)
    frames = []
    for i in picks:
        vf = common.video_file_for(rows[i]['video_id'], rows[i]['category'])
        cap = cv2.VideoCapture(vf)
        step = 7
        idx = 0
        while len(frames) < n_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, fr = cap.read()
            if not ok:
                break
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            fr = cv2.resize(fr, (224, 224), interpolation=cv2.INTER_CUBIC)
            frames.append(fr.astype(np.float32) / 255.0)
            idx += step
        cap.release()
        if len(frames) >= n_frames:
            break
    MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    x = np.stack(frames[:n_frames])
    x = (x - MEAN) / STD
    return x.transpose(0, 3, 1, 2).astype(np.float32)


def phase_extract():
    import torch
    model = _load_clip_model()
    pool = _frame_pool()
    results = []
    for b in [1, 8, 16, 32, 64, 128]:
        try:
            lat = []
            sampler = common.GpuUtilSampler(0)
            sampler.start()
            torch.cuda.reset_peak_memory_stats()
            n_iters = 10 + 100
            for it in range(n_iters):
                sel = np.random.RandomState(it).randint(0, len(pool) - b)
                x = torch.tensor(pool[sel:sel + b]).cuda()
                if next(model.parameters()).dtype == torch.float16:
                    x = x.half()
                torch.cuda.synchronize()
                t0 = time.time()
                with torch.no_grad():
                    model.encode_image(x)
                torch.cuda.synchronize()
                if it >= 10:
                    lat.append((time.time() - t0) * 1000)
            util = sampler.stop()
            total = sum(lat) / 1000
            rec = {'batch': b, 'frames_per_s': b * len(lat) / total,
                   'clips_per_s': b * len(lat) / total / SNIP,
                   **common.percentiles(lat),
                   'vram_torch_mb': torch.cuda.max_memory_allocated() / 1e6,
                   'vram_smi_mb': common.nvidia_smi_mem_mb(0), **util}
            results.append(rec)
            print(f"[extract] b={b}: {rec['frames_per_s']:.0f} frames/s "
                  f"({rec['clips_per_s']:.1f} clips/s) "
                  f"vram={rec['vram_torch_mb']:.0f}MB")
        except torch.cuda.OutOfMemoryError:
            results.append({'batch': b, 'oom': True})
            torch.cuda.empty_cache()
            break
    valid = [r for r in results if not r.get('oom')]
    best = max(valid, key=lambda r: r['frames_per_s'])
    _save_thr({'tier1_extract': {'batch_sweep': results, 'best_batch': best['batch'],
                                 'frames_per_s': best['frames_per_s'],
                                 'clips_per_s': best['clips_per_s'],
                                 'vram_gb': best['vram_torch_mb'] / 1000,
                                 'p50_ms': best['p50_ms'],
                                 'p95_ms': best['p95_ms']}})


# --------------------------------------------------------------------- head

def phase_head():
    import torch
    from demand import _patch_clip_load
    from model import CLIPVAD
    sys.path.insert(0, common.VADCLIP_SRC)
    from utils.tools import get_prompt_text

    state = _patch_clip_load(common.VADCLIP_CKPT)
    device = 'cuda'
    maxlen = 256
    model = CLIPVAD(14, 512, maxlen, 512, 1, 2, 8, 10, 10, device)
    model.load_state_dict(state)
    model.to(device).eval()
    prompt_text = get_prompt_text(common.UCF_LABEL_MAP)

    rows = common.load_test_list()
    rng = np.random.RandomState(SEED)
    picks = rng.choice(len(rows), 20, replace=False)
    feats = [np.load(rows[i]['feature_path']) for i in picks]
    total_snips = sum(len(f) for f in feats)
    from utils.tools import get_batch_mask

    def run_video(f):
        length = len(f)
        pad = (-length) % maxlen
        if pad:
            f = np.concatenate([f, np.zeros((pad, f.shape[1]), dtype=f.dtype)])
        v = torch.tensor(f).float().to(device).reshape(-1, maxlen, f.shape[1])
        if length < maxlen:
            pass  # already (1, 256, 512) after reshape+pad
        lengths = torch.zeros(int(length / maxlen) + 1)
        rem = length
        for j in range(int(length / maxlen) + 1):
            if j == 0 and length < maxlen:
                lengths[j] = length
            elif j == 0 and length > maxlen:
                lengths[j] = maxlen
                rem -= maxlen
            elif rem > maxlen:
                lengths[j] = maxlen
                rem -= maxlen
            else:
                lengths[j] = rem
        lengths = lengths.to(int)
        padding_mask = get_batch_mask(lengths, maxlen).to(device)
        model(v, padding_mask, prompt_text, lengths)

    with torch.no_grad():
        for f in feats[:3]:  # warmup
            run_video(f)
        torch.cuda.synchronize()
        t0 = time.time()
        for f in feats:
            run_video(f)
        torch.cuda.synchronize()
        dt = time.time() - t0
    print(f"[head] {total_snips/dt:.0f} snippets/s "
          f"({total_snips/dt:.0f} clips/s over {len(feats)} videos)")
    _save_thr({'tier1_head': {'clips_per_s': total_snips / dt}})


# ---------------------------------------------------------------------- e2e

def phase_e2e():
    import cv2
    import torch
    model = _load_clip_model()
    rows = common.load_test_list()
    rng = np.random.RandomState(SEED + 1)
    picks = rng.choice(len(rows), 6, replace=False)
    MEAN = np.array([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    STD = np.array([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)
    tot_frames, tot_t = 0, 0.0
    for i in picks:
        vf = common.video_file_for(rows[i]['video_id'], rows[i]['category'])
        cap = cv2.VideoCapture(vf)
        buf = []
        t0 = time.time()
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            tot_frames += 1
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
            buf.append(cv2.resize(fr, (224, 224), interpolation=cv2.INTER_CUBIC)
                       .astype(np.float32) / 255.0)
            if len(buf) == 64:
                x = torch.tensor((np.stack(buf) - MEAN) / STD)
                x = x.permute(0, 3, 1, 2).cuda().half()
                with torch.no_grad():
                    model.encode_image(x)
                buf = []
        cap.release()
        tot_t += time.time() - t0
    print(f"[e2e] decode+extract: {tot_frames/tot_t:.0f} frames/s "
          f"({tot_frames/tot_t/SNIP:.1f} clips/s)")
    _save_thr({'end_to_end': {'frames_per_s': tot_frames / tot_t,
                              'clips_per_s': tot_frames / tot_t / SNIP,
                              'note': 'decode(resize)+extract, batch 64, '
                                      'single process, 1 GPU'}})


def _save_thr(d):
    old = {}
    if os.path.exists(THR):
        old = json.load(open(THR))
    old.update(d)
    with open(THR, 'w') as f:
        json.dump(old, f, indent=2)


# ------------------------------------------------------------------- stride

def phase_stride():
    ids, scores, lens, labels = _load_scores_labels()
    gt_frames = common.load_frame_gt()

    def full_frame(fused):
        concat = np.concatenate([np.repeat(fused[i, :lens[i]], SNIP)
                                 for i in range(len(ids))])
        n = min(len(concat), len(gt_frames))
        return concat[:n], gt_frames[:n]

    fs0, g0 = full_frame(scores)
    print(f"[stride] baseline auc={auc_rank(fs0, g0):.4f} ap={ap_score(fs0, g0):.4f}")

    out = []
    for stride in (1, 2, 4, 8, 16):
        held = scores.copy()
        for i in range(len(ids)):
            s = scores[i, :lens[i]]
            z = np.repeat(s[::stride], stride)[:lens[i]]
            held[i, :lens[i]] = z
        fs, g = full_frame(held)
        auc_h, ap_h = auc_rank(fs, g), ap_score(fs, g)
        # analysed-only variant: metrics on the analysed snippets only
        sa, ga = [], []
        for i in range(len(ids)):
            z = held[i, :lens[i]]
            sa.append(np.repeat(z[::stride], SNIP))
            ga.append(np.repeat(labels[i, :lens[i]][::stride], SNIP))
        sa, ga = np.concatenate(sa), np.concatenate(ga)
        auc_a = auc_rank(sa, ga)
        # event metrics on held
        recs, ttas = [], []
        for i in range(len(ids)):
            es = event_stats(held[i, :lens[i]], labels[i, :lens[i]], TAU)
            if es:
                recs.append(es[0])
                if es[1] is not None:
                    ttas.append(es[1])
        rec = {'stride': stride, 'auc_held': auc_h, 'ap_held': ap_h,
               'auc_analysed_only': auc_a,
               'event_recall': float(np.mean(recs)),
               'tta_seconds_mean': float(np.mean(ttas) * SNIP / FPS)}
        out.append(rec)
        print(f"[stride {stride:>2}] auc_held={auc_h:.4f} ap_held={ap_h:.4f} "
              f"auc_analysed={auc_a:.4f} recall={rec['event_recall']:.3f} "
              f"tta={rec['tta_seconds_mean']:.2f}s")
    _save_thr({'stride_sweep': out})


def _load_scores_labels():
    t1 = np.load(os.path.join(common.RESULTS_DIR, 'tier1_scores.npz'),
                 allow_pickle=True)
    ids = t1['video_ids'].tolist()
    scores, lens = t1['scores'], t1['lens']
    ann = common.load_temporal_annotations()
    labels = np.zeros((len(ids), lens.max()), dtype=np.int8)
    for i, vid in enumerate(ids):
        for a, b in ann.get(vid, []):
            labels[i, max((a - 1) // SNIP, 0):min((b - 1) // SNIP, lens[i] - 1) + 1] = 1
    return ids, scores, lens, labels


# ------------------------------------------------------------------ analyze

def phase_analyze():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    thr = json.load(open(THR))
    stride_sweep = thr['stride_sweep']

    # --- fleet model at CLIP_PERIOD=1.0s ---
    FRAMES_PER_CAMHOUR = FPS * 3600            # 108,000 (30fps continuous)
    SNIPS_PER_CAMHOUR = FRAMES_PER_CAMHOUR / SNIP  # 6,750
    CLIPS_PER_CAMHOUR = 3600                   # 1 per second
    ESC_RATE = 0.086

    dec = thr['decode_cpu_16proc']['frames_per_s']
    ext = thr['tier1_extract']['frames_per_s']
    head = thr['tier1_head']['clips_per_s']
    vlm = 0.757  # E0 capacity best single-GPU (b4, 1 replica), clips/s

    dec_s = FRAMES_PER_CAMHOUR / dec
    ext_s = FRAMES_PER_CAMHOUR / ext
    head_s = SNIPS_PER_CAMHOUR / head
    vlm_s = CLIPS_PER_CAMHOUR * ESC_RATE / vlm
    total = dec_s + ext_s + head_s + vlm_s
    split = {'decode_gpu_s': dec_s, 'tier1_extract_gpu_s': ext_s,
             'tier1_head_gpu_s': head_s, 'vlm_gpu_s_at_8.6pct': vlm_s,
             'total_gpu_s': total, 'vlm_share_of_total': vlm_s / total}
    print(f"[budget] decode={dec_s:.3f} extract={ext_s:.3f} head={head_s:.4f} "
          f"vlm={vlm_s:.3f} GPU-s/cam-hour; VLM share={vlm_s/total:.1%}")

    cams_with = 4 * 3600 / total
    cams_without = 4 * 3600 / (dec_s + ext_s + head_s)
    print(f"[budget] cameras per 4-GPU server: with VLM={cams_with:.0f}, "
          f"without={cams_without:.0f}")

    # --- headline table: AUC per GPU-s ---
    auc_t1 = stride_sweep[0]['auc_held']
    vlm_dauc = -0.0006  # E0.6 v2 isotonic-calibrated fusion, band s0
    table = []
    for rec in stride_sweep:
        cost = (dec_s + ext_s) / rec['stride'] + head_s / rec['stride']
        table.append({'allocation': f"tier1 stride {rec['stride']}, no VLM",
                      'gpu_s_per_camera_hour': cost, 'auc': rec['auc_held'],
                      'dAUC_vs_base': rec['auc_held'] - auc_t1,
                      'auc_per_gpu_s': rec['auc_held'] / cost})
    for stride_rec in stride_sweep[:2]:
        cost = ((dec_s + ext_s + head_s) / stride_rec['stride']) + vlm_s
        table.append({'allocation': f"tier1 stride {stride_rec['stride']} + VLM @8.6%",
                      'gpu_s_per_camera_hour': cost,
                      'auc': stride_rec['auc_held'] + vlm_dauc,
                      'dAUC_vs_base': vlm_dauc,
                      'auc_per_gpu_s': (stride_rec['auc_held'] + vlm_dauc) / cost})
    for t in table:
        print(f"  {t['allocation']:>34}: {t['gpu_s_per_camera_hour']:.3f} GPU-s "
              f"AUC={t['auc']:.4f} dAUC={t['dAUC_vs_base']:+.4f} "
              f"auc/gpu_s={t['auc_per_gpu_s']:.4f}")

    # --- iso-budget curves ---
    budgets = np.array([2, 4, 8, 16]) * 3600.0  # GPU-s per hour wall clock
    iso = []
    for rec in stride_sweep:
        cost = (dec_s + ext_s + head_s) / rec['stride']
        for B in budgets:
            iso.append({'budget_gpu_s_per_h': float(B),
                        'strategy': f"all_tier1_stride{rec['stride']}",
                        'cameras': B / cost, 'auc': rec['auc_held']})
    cost_v = dec_s + ext_s + head_s + vlm_s
    for B in budgets:
        iso.append({'budget_gpu_s_per_h': float(B),
                    'strategy': 'tier1_stride1+VLM@8.6%',
                    'cameras': B / cost_v, 'auc': auc_t1 + vlm_dauc})

    # figures
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    stages = ['decode', 'tier1_extract', 'tier1_head', 'vlm@8.6%']
    vals = [dec_s, ext_s, head_s, vlm_s]
    daucs = ['0', '0.8802 (base)', '0', '-0.0006']
    bars = ax.bar(stages, vals)
    for b, d in zip(bars, daucs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height(), d,
                ha='center', va='bottom', fontsize=9)
    ax.set_ylabel('GPU-s per camera-hour')
    ax.set_title('E0.8 cost per stage, with each stage\'s AUC contribution')
    fig.tight_layout()
    os.makedirs(os.path.join(OUT, 'figures'), exist_ok=True)
    fig.savefig(os.path.join(OUT, 'figures', 'e08_cost_benefit.png'), dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for strategy, marker in (('all_tier1_stride1', 'o-'), ('all_tier1_stride2', 's-'),
                             ('all_tier1_stride4', '^-'), ('all_tier1_stride8', 'd-'),
                             ('tier1_stride1+VLM@8.6%', 'x')):
        pts = [r for r in iso if r['strategy'] == strategy]
        pts.sort(key=lambda r: r['cameras'])
        if pts:
            ax.plot([p['cameras'] for p in pts], [p['auc'] for p in pts],
                    marker, label=strategy, markersize=8 if marker == 'x' else 4)
    ax.set_xscale('log', base=2)
    ax.set_xlabel('cameras served')
    ax.set_ylabel('fleet AUC')
    ax.set_title('E0.8 iso-budget: AUC vs cameras at fixed GPU budget')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'figures', 'e08_rate_auc.png'), dpi=150)
    plt.close(fig)

    # gates
    share = split['vlm_share_of_total']
    gate1 = ('STRONG' if share >= 0.20 else 'WORKABLE' if share >= 0.05
             else 'NO PROBLEM')
    s4 = [r for r in stride_sweep if r['stride'] == 4][0]
    degrades = auc_t1 - s4['auc_held'] > 0.01
    rec_drop = stride_sweep[0]['event_recall'] - s4['event_recall']
    gate2 = ('RATE IS A TARGET' if degrades else
             'TEMPORAL COST HIDDEN BY AUC' if rec_drop > 0.05 else 'FLAT')
    verdict = ('SYSTEMS PAPER' if gate1 in ('STRONG', 'WORKABLE') and gate2 != 'FLAT'
               else 'MORE-CAMERAS PAPER' if gate1 in ('STRONG', 'WORKABLE')
               else 'MEASUREMENT PAPER')

    doc = {
        'tier1_spec': {
            'model': 'CLIP ViT-B/16 (weights rebuilt from model_ucf.pth '
                     'clipmodel.* keys; openaipublic unreachable)',
            'resolution': '224x224', 'frames_per_snippet': 16,
            'pooling': 'mean of raw (unnormalized) per-frame encode_image '
                       'outputs over each 16-frame snippet',
            'crops': 'released features are 10-crop; VadCLIP uses crop 5; '
                     'deployment measured at 1 crop = 1 forward/frame',
            'verification': 'live extraction with standard CLIP transform '
                            'matches released features at cosine 0.73-0.92 '
                            '(our mp4s are a re-encode; released features '
                            'reproduce AUC 0.8802 exactly)'},
        'throughput': {**thr, 'vlm_holmesvau2b': {'clips_per_s': vlm,
                                                  'source': 'E0 capacity.json '
                                                            'best single-GPU (b4)'}},
        'fleet_model': {'frames_per_camera_hour': FRAMES_PER_CAMHOUR,
                        'snippets_per_camera_hour': SNIPS_PER_CAMHOUR,
                        'vlm_clips_per_camera_hour': CLIPS_PER_CAMHOUR * ESC_RATE,
                        'clip_period_s': 1.0, 'escalation_rate': ESC_RATE},
        'budget_split_per_camera_hour': split,
        'cameras_per_4gpu_server': {'with_vlm': cams_with,
                                    'without_vlm': cams_without},
        'headline_table': table,
        'stride_sweep': stride_sweep,
        'iso_budget': iso,
        'gate1_verdict': gate1,
        'gate2_verdict': gate2,
        'verdict': verdict,
    }
    with open(os.path.join(OUT, 'e08.json'), 'w') as f:
        json.dump(doc, f, indent=2)
    print(f"gate1={gate1} gate2={gate2} -> {verdict}")
    print(f"[saved] {OUT}/e08.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--decode', action='store_true')
    ap.add_argument('--extract', action='store_true')
    ap.add_argument('--head', action='store_true')
    ap.add_argument('--e2e', action='store_true')
    ap.add_argument('--stride', action='store_true')
    ap.add_argument('--analyze', action='store_true')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.decode:
        phase_decode()
    if args.extract:
        phase_extract()
    if args.head:
        phase_head()
    if args.e2e:
        phase_e2e()
    if args.stride:
        phase_stride()
    if args.analyze:
        phase_analyze()


if __name__ == '__main__':
    main()
