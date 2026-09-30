"""E0.7 -- is the negative result a property of the VLM, or of how we asked it?

2x3 grid: {uncertainty band, top-K by score} x {+-0, +-8, +-32 snippets}.
Frame count held at 12 in every cell (widen the span, not the frame budget).
Same prompt, same fp16/eager config, same isotonic-crossfit-by-video-parity
fusion in every cell. Headline metric: CAPTURE FRACTION = dAUC / per-cell
oracle ceiling -- the only quantity comparable across cells.

Subcommands:
  --freeze-topk                      freeze escalated_topk.npz (K=5965 global by score)
  --extract --gate G --span S        extract one cell's clips to JPEG cache
  --worker --gate G --span S --shard K --gpu 0     score one shard (4 procs)
  --analyze                          all 6 cells -> e07.json + e07_grid.png
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
from vlmvalue import auc_rank, ap_score, event_stats, PROMPT  # noqa: E402
from vlm_calibrated import pava, iso_apply  # noqa: E402

OUT = os.path.join(E0, 'followup', 'results')
TAU, MARGIN = 0.5, 0.2603611499071121
SNIP, FPS = 16, 30.0
K_BUDGET = 5965
SEED = 0

GATES = ('band', 'topk')
SPANS = (0, 8, 32)


# ------------------------------------------------------------ list handling

def topk_path():
    return os.path.join(OUT, 'escalated_topk.npz')


def band_path():
    return os.path.join(OUT, 'escalated_clips.npz')


def load_scores_labels():
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


def freeze_topk():
    ids, scores, lens, labels = load_scores_labels()
    flat = []
    for vi in range(len(ids)):
        for si in range(lens[vi]):
            flat.append((scores[vi, si], vi, si))
    flat.sort(key=lambda x: -x[0])
    top = flat[:K_BUDGET]
    vis = np.array([t[1] for t in top])
    sis = np.array([t[2] for t in top])
    anom = np.mean([labels[vi, si] for vi, si in zip(vis, sis)])
    np.savez(topk_path(), video_ids=np.array(ids), vi=vis, si=sis)
    print(f"[freeze-topk] {len(vis)} clips, anomalous={anom:.3f}, "
          f"score range [{scores[vis, sis].min():.4f}, {scores[vis, sis].max():.4f}]")


def load_list(gate):
    d = np.load(band_path() if gate == 'band' else topk_path(), allow_pickle=True)
    return d['video_ids'].tolist(), d['vi'], d['si']


def cache_dir(gate, span):
    if gate == 'band' and span == 0:
        return os.path.join(E0, 'data', 'clip_cache')  # E0.6 cache
    return os.path.join(E0, 'data', f'clip_cache_{gate}_s{span}')


def shard_pattern(gate, span):
    if gate == 'band' and span == 0:
        return os.path.join(OUT, 'vlm_scores_shard*.jsonl')
    return os.path.join(OUT, f'vlm_scores_{gate}_s{span}_shard*.jsonl')


# --------------------------------------------------------------- extraction

def _extract_video_e07(args):
    vi, items, span, rows, cdir = args
    from decord import VideoReader, cpu
    from PIL import Image
    sys.path.insert(0, common.HOLMES_DIR)
    from holmesvau.internvl_utils import get_index
    r = rows[vi]
    vr = VideoReader(common.video_file_for(r['video_id'], r['category']),
                     ctx=cpu(0), num_threads=1)
    n_frames = len(vr)
    clamped = 0
    for k, si in items:
        f_lo = (si - span) * SNIP
        f_hi = (si + span) * SNIP + SNIP - 1
        if f_lo < 0 or f_hi > n_frames - 1:
            clamped += 1
        f_lo = max(f_lo, 0)
        f_hi = min(f_hi, n_frames - 1)
        idx = get_index(bound=None, fps=FPS, max_frame=f_hi,
                        first_idx=f_lo, num_segments=12)
        idx = [min(max(int(i), 0), n_frames - 1) for i in idx]
        for j, fi in enumerate(idx):
            img = Image.fromarray(vr[fi].asnumpy()).convert('RGB')
            img = img.resize((448, 448), Image.BICUBIC)
            img.save(os.path.join(cdir, f"{k:05d}_f{j:02d}.jpg"), quality=90)
    return len(items), clamped


def extract(gate, span):
    from concurrent.futures import ProcessPoolExecutor
    ids, vis, sis = load_list(gate)
    cdir = cache_dir(gate, span)
    os.makedirs(cdir, exist_ok=True)
    rows = common.load_test_list()
    by_video = {}
    for k, (vi, si) in enumerate(zip(vis, sis)):
        by_video.setdefault(vi, []).append((k, si))
    todo = []
    for vi, items in by_video.items():
        need = [(k, si) for k, si in items
                if not os.path.exists(os.path.join(cdir, f"{k:05d}_f11.jpg"))]
        if need:
            todo.append((vi, need, span, rows, cdir))
    print(f"[extract {gate}/s{span}] {sum(len(t[1]) for t in todo)} clips")
    done, clamped = 0, 0
    with ProcessPoolExecutor(max_workers=16) as ex:
        for n, c in ex.map(_extract_video_e07, todo):
            done += n
            clamped += c
    # clamped fraction over ALL clips in the cell (incl. previously cached)
    stats = {'gate': gate, 'span': span, 'clips': len(vis),
             'clamped': int(clamped), 'clamped_fraction': clamped / len(vis)}
    with open(os.path.join(OUT, f'clamp_{gate}_s{span}.json'), 'w') as f:
        json.dump(stats, f)
    print(f"[extract {gate}/s{span}] done, clamped={clamped} "
          f"({clamped/len(vis):.3f})")


# ------------------------------------------------------------------- worker

def load_clip_pixels_cell(cdir, k, transform):
    import torch
    from PIL import Image
    return torch.stack([transform(Image.open(
        os.path.join(cdir, f"{k:05d}_f{j:02d}.jpg")).convert('RGB'))
        for j in range(12)])


def run_worker(gate, span, shard, n_shards, batch):
    import torch
    sys.path.insert(0, os.path.join(E0, 'src'))
    from capacity import load_holmes, CLIP_SPEC
    sys.path.insert(0, common.HOLMES_DIR)
    from holmesvau.internvl_utils import build_transform
    from holmesvau.ATS.anomaly_scorer import URDMU

    common.set_seed(SEED)
    device = torch.device("cuda:0")
    model, tok, _ = load_holmes(common.HOLMES_MODEL_PATH, device)
    scorer = URDMU().to(device)
    scorer.load_state_dict(torch.load(
        os.path.join(common.HOLMES_DIR, 'holmesvau', 'ATS', 'anomaly_scorer.pth'),
        map_location=device))
    scorer.eval()
    transform = build_transform(input_size=448)
    yn = {w: tok.encode(w, add_special_tokens=False)[0]
          for w in ("Yes", "No", "yes", "no")}
    sys.path.insert(0, common.HOLMES_MODEL_PATH)
    from conversation import get_conv_template

    ids, vis, sis = load_list(gate)
    cdir = cache_dir(gate, span)
    mine = list(range(shard, len(vis), n_shards))
    print(f"[{gate}/s{span} shard{shard}] {len(mine)} clips", flush=True)

    out_path = os.path.join(OUT, f"vlm_scores_{gate}_s{span}_shard{shard}.jsonl")
    with open(out_path, 'w') as fout:
        for b0 in range(0, len(mine), batch):
            ks = mine[b0:b0 + batch]
            pv = torch.cat([load_clip_pixels_cell(cdir, k, transform) for k in ks])
            pv = pv.to(torch.float16).to(device)
            with torch.no_grad():
                vit = model.vision_model(pixel_values=pv, output_hidden_states=False,
                                         return_dict=True).last_hidden_state
                cls = vit[:, 0, :].to(torch.float32).unsqueeze(0)
                ats = scorer(cls)['anomaly_scores'][0]
                ats = ats.detach().cpu().numpy().reshape(len(ks), 12).mean(axis=1)
            video_prefix = ''.join([f'Frame{i+1}: <image>\n'
                                    for i in range(CLIP_SPEC["frames"])])
            questions = [video_prefix + PROMPT for _ in ks]
            queries, pos = [], 0
            IMG_START, IMG_END, IMG_CTX = '<img>', '</img>', '<IMG_CONTEXT>'
            model.img_context_token_id = tok.convert_tokens_to_ids(IMG_CTX)
            npl = [1] * pv.shape[0]
            for q in questions:
                template = get_conv_template(model.template)
                template.system_message = model.system_message
                template.append_message(template.roles[0], q)
                template.append_message(template.roles[1], None)
                query = template.get_prompt()
                for _ in range(query.count('<image>')):
                    image_tokens = (IMG_START + IMG_CTX * model.num_image_token
                                    * npl[pos] + IMG_END)
                    pos += 1
                    query = query.replace('<image>', image_tokens, 1)
                queries.append(query)
            tok.padding_side = 'left'
            inputs = tok(queries, return_tensors='pt', padding=True)
            gen_out = model.generate(
                pixel_values=pv,
                input_ids=inputs['input_ids'].to(device),
                attention_mask=inputs['attention_mask'].to(device),
                max_new_tokens=64, do_sample=False,
                eos_token_id=tok.convert_tokens_to_ids(template.sep),
                output_scores=True, return_dict_in_generate=True)
            first_logits = gen_out.scores[0]
            texts = tok.batch_decode(gen_out.sequences, skip_special_tokens=True)
            for i, k in enumerate(ks):
                logits = first_logits[i].float()
                yes = torch.logsumexp(logits[[yn["Yes"], yn["yes"]]], dim=0)
                no = torch.logsumexp(logits[[yn["No"], yn["no"]]], dim=0)
                p_yes = torch.softmax(torch.stack([no, yes]), dim=0)[1].item()
                text = texts[i].split(template.sep)[0].strip()
                fw = text.split()[0].strip('.,!').lower() if text.split() else ''
                fout.write(json.dumps({
                    'k': int(k), 'vi': int(vis[k]), 'si': int(sis[k]),
                    'score_b': p_yes, 'score_c': 1 if fw == 'yes' else 0,
                    'score_a': float(ats[i]), 'text': text[:400]}) + '\n')
            fout.flush()
            if (b0 + len(ks)) % 200 < batch:
                print(f"[{gate}/s{span} shard{shard}] {b0+len(ks)}/{len(mine)}",
                      flush=True)
    print(f"[{gate}/s{span} shard{shard}] DONE", flush=True)


# ----------------------------------------------------------------- analysis

def load_cell_scores(gate, span):
    vlm = {}
    for f in sorted(glob.glob(shard_pattern(gate, span))):
        for line in open(f):
            r = json.loads(line)
            vlm[(r['vi'], r['si'])] = r
    return vlm


def analyze():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ids, scores, lens, labels = load_scores_labels()
    gt_frames = common.load_frame_gt()

    def full_frame(fused):
        concat = np.concatenate([np.repeat(fused[i, :lens[i]], SNIP)
                                 for i in range(len(ids))])
        n = min(len(concat), len(gt_frames))
        return concat[:n], gt_frames[:n]

    fs0, g0 = full_frame(scores)
    auc_t1, ap_t1 = auc_rank(fs0, g0), ap_score(fs0, g0)

    def ceiling_of(vis, sis):
        fused = scores.copy()
        fused[vis, sis] = labels[vis, sis]
        fs, g = full_frame(fused)
        return auc_rank(fs, g) - auc_t1

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

    cells = []
    for gate in GATES:
        g_ids, vis, sis = load_list(gate)
        gt_pool = labels[vis, sis]
        s_pool = scores[vis, sis]
        ceiling = ceiling_of(vis, sis)
        print(f"[{gate}] pool={len(vis)} anomalous={gt_pool.mean():.3f} "
              f"ceiling={ceiling:+.4f}")
        for span in SPANS:
            vlm = load_cell_scores(gate, span)
            assert len(vlm) == len(vis), f"{gate}/s{span}: {len(vlm)} != {len(vis)}"
            pv = np.array([vlm[(vi, si)]['score_b'] for vi, si in zip(vis, sis)])
            # isotonic cross-fit by video parity
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
            auc_f, ap_f = auc_rank(fs, g), ap_score(fs, g)
            dauc, dap = auc_f - auc_t1, ap_f - ap_t1
            # vlm-only stats
            acc = float(((pv > 0.5).astype(int) == gt_pool).mean())
            const_acc = float(max(gt_pool.mean(), 1 - gt_pool.mean()))
            best = max(({'thr': float(t),
                         'f1': float(2 * ((pv > t) & (gt_pool == 1)).sum() /
                                 max((pv > t).sum() + (gt_pool == 1).sum(), 1))}
                        for t in np.arange(0.02, 0.99, 0.02)),
                       key=lambda r: r['f1'])
            clamp_f = 0.0
            cp = os.path.join(OUT, f'clamp_{gate}_s{span}.json')
            if os.path.exists(cp):
                clamp_f = json.load(open(cp))['clamped_fraction']
            cell = {
                'gate': gate, 'span_snippets': span,
                'dAUC': float(dauc), 'dAP': float(dap),
                'oracle_ceiling_dAUC': float(ceiling),
                'capture_fraction': float(dauc / ceiling),
                'within_pool_auc_vlm': auc_rank(pv, gt_pool),
                'within_pool_auc_tier1': auc_rank(s_pool, gt_pool),
                'vlm_acc': acc, 'const_majority_acc': const_acc,
                'vlm_yes_rate': float((pv > 0.5).mean()),
                'best_f1_thr': best['thr'], 'best_f1': best['f1'],
                'ops': ops(fused),
                'clamped_fraction': clamp_f,
            }
            cells.append(cell)
            print(f"  s{span:>2}: dAUC={dauc:+.4f} capture={dauc/ceiling:+.1%} "
                  f"poolAUC vlm={cell['within_pool_auc_vlm']:.3f} "
                  f"t1={cell['within_pool_auc_tier1']:.3f} "
                  f"acc={acc:.3f}/const={const_acc:.3f} yes={cell['vlm_yes_rate']:.3f}")

    # attribution on capture fractions (2x3 additive decomposition)
    cf = np.array([[c['capture_fraction'] for c in cells if c['gate'] == g]
                   for g in GATES])  # (2 gates, 3 spans)
    grand = cf.mean()
    gate_eff = cf[1].mean() - cf[0].mean()
    span_means = cf.mean(axis=0)
    context_eff = span_means[1:].mean() - span_means[0]
    additive = (grand + (cf.mean(axis=1, keepdims=True) - grand)
                + (cf.mean(axis=0, keepdims=True) - grand))
    interaction = float(np.abs(cf - additive).mean())
    attribution = {'gate_effect_topk_minus_band': float(gate_eff),
                   'context_effect_span8_32_minus_span0': float(context_eff),
                   'interaction_mean_abs_residual': interaction,
                   'capture_grid': cf.tolist()}
    max_capture = float(np.abs(cf).max())
    max_signed = float(cf.max())
    print(f"\nattribution: gate={gate_eff:+.4f} context={context_eff:+.4f} "
          f"interaction={interaction:.4f}")
    print(f"max capture fraction: {max_signed:+.1%}")

    verdict = ('POSITIVE' if max_signed >= 0.30 else
               'MARGINAL' if max_signed >= 0.10 else 'NEGATIVE CONFIRMED')
    print(f"VERDICT: {verdict}")

    # figure: 2x3 capture heatmap
    fig, ax = plt.subplots(figsize=(7, 3.2))
    im = ax.imshow(cf, cmap='RdYlGn', vmin=-0.35, vmax=0.35, aspect='auto')
    ax.set_xticks(range(3), ['±0 (0.53s)', '±8 (~8.5s)', '±32 (~34s)'])
    ax.set_yticks(range(2), ['uncertainty band', 'top-K by score'])
    for i in range(2):
        for j in range(3):
            ax.text(j, i, f"{cf[i, j]:+.1%}", ha='center', va='center',
                    fontsize=11)
    ceilings = {g: [c['oracle_ceiling_dAUC'] for c in cells if c['gate'] == g][0]
                for g in GATES}
    ax.set_title(f"E0.7 capture fraction (dAUC / ceiling); ceilings: "
                 f"band {ceilings['band']:+.4f}, topK {ceilings['topk']:+.4f}")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    os.makedirs(os.path.join(OUT, 'figures'), exist_ok=True)
    fig.savefig(os.path.join(OUT, 'figures', 'e07_grid.png'), dpi=150)
    plt.close(fig)

    doc = {
        'config': {'frames': 12, 'resolution': '448x448', 'K': K_BUDGET,
                   'prompt': PROMPT,
                   'fusion': 'isotonic_crossfit_by_video_parity'},
        'baseline': {'auc_tier1': auc_t1, 'ap_tier1': ap_t1},
        'cells': cells,
        'attribution': attribution,
        'max_capture_fraction': max_signed,
        'verdict': verdict,
    }
    with open(os.path.join(OUT, 'e07.json'), 'w') as f:
        json.dump(doc, f, indent=2)
    print(f"[saved] {OUT}/e07.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze-topk', action='store_true')
    ap.add_argument('--extract', action='store_true')
    ap.add_argument('--worker', action='store_true')
    ap.add_argument('--analyze', action='store_true')
    ap.add_argument('--gate', default='band')
    ap.add_argument('--span', type=int, default=0)
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--n-shards', type=int, default=4)
    ap.add_argument('--batch', type=int, default=4)
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.freeze_topk:
        freeze_topk()
    if args.extract:
        extract(args.gate, args.span)
    if args.worker:
        run_worker(args.gate, args.span, args.shard, args.n_shards, args.batch)
    if args.analyze:
        analyze()


if __name__ == '__main__':
    main()
