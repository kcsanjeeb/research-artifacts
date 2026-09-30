"""E0.6 -- is a VLM call worth making? (Docs/NEXT_STEPS.md section 2)

Runs Holmes-VAU-2B over the 5,965 escalated clips (PerStreamGate, tau=0.5,
margin=0.2604 -- frozen list), derives three scalar VLM scores, fuses with
tier-1, and measures the full-test-set delta.

Score extraction (the methodological choice, NEXT_STEPS 2.2.4):
  PRIMARY (b): P(Yes) -- softmax over {Yes,No,yes,no} at the first generated
      token for the prompt "...Does this video contain an anomaly? Answer Yes
      or No first, then explain briefly." This is the MLLM's own scalar
      verdict, and it comes from ONE 64-token generation per clip.
  (c): the hard parse of the same answer (first word Yes/No). Coarsest.
  (a): Holmes-VAU's ATS scorer (UR-DMU) over the clip's 12 ViT CLS tokens,
      mean-pooled. Recorded for completeness, NOT used as primary: UR-DMU is a
      separately-trained WSVAD network (trained on UCF-Crime train split) that
      Holmes-VAU uses for frame sampling -- it is not the MLLM's verdict, so it
      cannot answer "is the VLM call worth making". The repo's own code casts
      pixels to bf16 here; on sm_70 we run the same computation in fp16
      (the model itself is fp16 in our config), UR-DMU stays fp32 exactly as
      upstream.

Phases:
  --freeze                 build the frozen escalated list (once)
  --extract                extract clips to JPEG cache (CPU, parallel)
  --worker --gpu G --shard K/4  score one shard (one process per GPU)
  --analyze                fusion + metrics + marginal-value curve + JSON/figs
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

OUT = os.path.join(E0, 'followup', 'results')
CACHE = os.path.join(E0, 'data', 'clip_cache')
FROZEN = os.path.join(OUT, 'escalated_clips.npz')
TAU, MARGIN = 0.5, 0.2603611499071121
SNIP, FPS = 16, 30.0
SEED = 0

PROMPT = ("Does this video contain an anomaly? Answer Yes or No first, "
          "then explain briefly.")


# ---------------------------------------------------------------- freeze

def freeze():
    t1 = np.load(os.path.join(common.RESULTS_DIR, 'tier1_scores.npz'),
                 allow_pickle=True)
    ids = t1['video_ids'].tolist()
    scores, lens = t1['scores'], t1['lens']
    vis, sis = [], []
    for vi in range(len(ids)):
        s = scores[vi, :lens[vi]]
        esc = np.abs(s - TAU) < MARGIN
        for si in np.flatnonzero(esc):
            vis.append(vi)
            sis.append(int(si))
    np.savez(FROZEN, video_ids=np.array(ids), vi=np.array(vis), si=np.array(sis),
             tau=TAU, margin=MARGIN)
    print(f"[freeze] {len(vis)} escalated clips -> {FROZEN}")


def load_frozen():
    d = np.load(FROZEN, allow_pickle=True)
    return d['video_ids'].tolist(), d['vi'], d['si']


# ---------------------------------------------------------------- extract

def _extract_video(args):
    vi, items, rows = args
    from decord import VideoReader, cpu
    from PIL import Image
    sys.path.insert(0, common.HOLMES_DIR)
    from holmesvau.internvl_utils import get_index
    r = rows[vi]
    vr = VideoReader(common.video_file_for(r['video_id'], r['category']),
                     ctx=cpu(0), num_threads=1)
    for k, si in items:
        f0 = si * SNIP
        idx = get_index(bound=None, fps=FPS, max_frame=f0 + SNIP - 1,
                        first_idx=f0, num_segments=12)
        idx = [min(int(i), len(vr) - 1) for i in idx]
        for j, fi in enumerate(idx):
            img = Image.fromarray(vr[fi].asnumpy()).convert('RGB')
            img = img.resize((448, 448), Image.BICUBIC)
            img.save(clip_path(k, j), quality=90)
    return len(items)


def extract():
    """12 frames uniformly sampled from the 16-frame snippet window, resized
    to 448x448, saved as JPEG. Grouped by video so each mp4 is opened once."""
    from concurrent.futures import ProcessPoolExecutor

    ids, vis, sis = load_frozen()
    os.makedirs(CACHE, exist_ok=True)
    rows = common.load_test_list()
    by_video = {}
    for k, (vi, si) in enumerate(zip(vis, sis)):
        by_video.setdefault(vi, []).append((k, si))

    todo = []
    for vi, items in by_video.items():
        need = [(k, si) for k, si in items
                if not os.path.exists(clip_path(k, 11))]
        if need:
            todo.append((vi, need, rows))
    print(f"[extract] {sum(len(t[1]) for t in todo)} clips to extract "
          f"across {len(todo)} videos")

    done = 0
    with ProcessPoolExecutor(max_workers=16) as ex:
        for n in ex.map(_extract_video, todo):
            done += n
            if done % 500 < 100:
                print(f"[extract] ~{done}", flush=True)
    print("[extract] done")


def clip_path(k, j):
    return os.path.join(CACHE, f"{k:05d}_f{j:02d}.jpg")


def load_clip_pixels(k, transform):
    import torch
    from PIL import Image
    pvs = []
    for j in range(12):
        img = Image.open(clip_path(k, j)).convert('RGB')
        pvs.append(transform(img))
    return torch.stack(pvs)


# ---------------------------------------------------------------- worker

YESNO_WORDS = ["Yes", "No", "yes", "no"]


def run_worker(gpu, shard, n_shards, batch):
    import torch
    sys.path.insert(0, os.path.join(E0, 'src'))
    from capacity import load_holmes, CLIP_SPEC
    sys.path.insert(0, common.HOLMES_DIR)
    from holmesvau.internvl_utils import build_transform
    from holmesvau.ATS.anomaly_scorer import URDMU

    common.set_seed(SEED)
    device = torch.device(f"cuda:{gpu}")
    model, tok, load_s = load_holmes(common.HOLMES_MODEL_PATH, device)
    scorer = URDMU().to(device)
    scorer.load_state_dict(torch.load(
        os.path.join(common.HOLMES_DIR, 'holmesvau', 'ATS', 'anomaly_scorer.pth'),
        map_location=device))
    scorer.eval()
    transform = build_transform(input_size=448)

    # yes/no token ids
    yn = {w: tok.encode(w, add_special_tokens=False) for w in YESNO_WORDS}
    yn = {w: ids[0] for w, ids in yn.items()}
    print(f"[worker {gpu}] yes/no ids: {yn}", flush=True)

    ids, vis, sis = load_frozen()
    mine = list(range(shard, len(vis), n_shards))
    print(f"[worker {gpu}] {len(mine)} clips", flush=True)

    sys.path.insert(0, common.HOLMES_MODEL_PATH)
    from conversation import get_conv_template

    out_path = os.path.join(OUT, f"vlm_scores_shard{shard}.jsonl")
    with open(out_path, 'w') as fout:
        for b0 in range(0, len(mine), batch):
            ks = mine[b0:b0 + batch]
            pv = torch.cat([load_clip_pixels(k, transform) for k in ks])
            pv = pv.to(torch.float16).to(device)
            npl = [1] * pv.shape[0]
            # ---- ATS score (fp16 pixels; UR-DMU fp32 -- see module docstring)
            with torch.no_grad():
                vit = model.vision_model(pixel_values=pv, output_hidden_states=False,
                                         return_dict=True).last_hidden_state
                cls = vit[:, 0, :].to(torch.float32).unsqueeze(0)
                ats = scorer(cls)['anomaly_scores'][0]
                ats = ats.detach().cpu().numpy().reshape(len(ks), 12).mean(axis=1)
            # ---- generation with first-token scores
            video_prefix = ''.join([f'Frame{i+1}: <image>\n'
                                    for i in range(CLIP_SPEC["frames"])])
            questions = [video_prefix + PROMPT for _ in ks]
            queries = []
            pos = 0
            IMG_START, IMG_END, IMG_CTX = '<img>', '</img>', '<IMG_CONTEXT>'
            model.img_context_token_id = tok.convert_tokens_to_ids(IMG_CTX)
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
            first_logits = gen_out.scores[0]  # (B, vocab)
            texts = tok.batch_decode(gen_out.sequences, skip_special_tokens=True)
            for i, k in enumerate(ks):
                logits = first_logits[i].float()
                yes = torch.logsumexp(logits[[yn["Yes"], yn["yes"]]], dim=0)
                no = torch.logsumexp(logits[[yn["No"], yn["no"]]], dim=0)
                p_yes = torch.softmax(torch.stack([no, yes]), dim=0)[1].item()
                text = texts[i].split(template.sep)[0].strip()
                first_word = text.split()[0].strip('.,!').lower() if text.split() else ''
                rec = {'k': int(k), 'vi': int(vis[k]), 'si': int(sis[k]),
                       'score_b': p_yes,
                       'score_c': 1 if first_word == 'yes' else 0,
                       'score_a': float(ats[i]),
                       'text': text[:400]}
                fout.write(json.dumps(rec) + '\n')
            fout.flush()
            done = b0 + len(ks)
            if done % 100 < batch:
                print(f"[worker {gpu}] {done}/{len(mine)}", flush=True)
    print(f"[worker {gpu}] DONE -> {out_path}", flush=True)


# ---------------------------------------------------------------- analyze

def auc_rank(score, label):
    order = np.argsort(score, kind='mergesort')
    ranks = np.empty(len(score), dtype=np.float64)
    ranks[order] = np.arange(1, len(score) + 1)
    ss = score[order]
    i = 0
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    pos = label == 1
    np_, nn_ = pos.sum(), (~pos).sum()
    if np_ == 0 or nn_ == 0:
        return None
    return float((ranks[pos].sum() - np_ * (np_ + 1) / 2) / (np_ * nn_))


def ap_score(score, label):
    order = np.argsort(-score, kind='mergesort')
    lab = label[order]
    tp = np.cumsum(lab)
    prec = tp / np.arange(1, len(lab) + 1)
    return float((prec * lab).sum() / max(lab.sum(), 1))


def event_stats(frame_scores, labels_v, tau):
    """Event-level recall + time-to-alert (snippets) for one video."""
    gt = labels_v
    events, t0 = [], None
    for t, g in enumerate(gt):
        if g == 1 and t0 is None:
            t0 = t
        if (g == 0 or t == len(gt) - 1) and t0 is not None:
            events.append((t0, t if g == 0 else t + 1))
            t0 = None
    if not events:
        return None
    rec, ttas = 0, []
    for a, b in events:
        above = [t for t in range(a, b) if frame_scores[t] > tau]
        if above:
            rec += 1
            ttas.append(above[0] - a)
    return rec / len(events), (np.mean(ttas) if ttas else None)


def analyze():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    ids, vis, sis = load_frozen()
    t1 = np.load(os.path.join(common.RESULTS_DIR, 'tier1_scores.npz'),
                 allow_pickle=True)
    scores, lens = t1['scores'], t1['lens']
    ann = common.load_temporal_annotations()
    labels = np.zeros((len(ids), lens.max()), dtype=np.int8)
    for i, vid in enumerate(ids):
        for a, b in ann.get(vid, []):
            labels[i, max((a - 1) // SNIP, 0):min((b - 1) // SNIP, lens[i] - 1) + 1] = 1

    # collect VLM scores
    vlm = {}
    for f in sorted(glob.glob(os.path.join(OUT, 'vlm_scores_shard*.jsonl'))):
        for line in open(f):
            r = json.loads(line)
            vlm[(r['vi'], r['si'])] = r
    print(f"[analyze] collected {len(vlm)} / {len(vis)} VLM verdicts")
    assert len(vlm) == len(vis), "missing VLM verdicts -- run all shards first"

    gt_frames = common.load_frame_gt()

    def full_frame_scores(fused_snippet):
        concat = np.concatenate([np.repeat(fused_snippet[i, :lens[i]], SNIP)
                                 for i in range(len(ids))])
        n = min(len(concat), len(gt_frames))
        return concat[:n], gt_frames[:n]

    def fused_with(score_key, subset=None, blend=None):
        out = scores.copy()
        for (vi, si), r in vlm.items():
            if subset is not None and (vi, si) not in subset:
                continue
            v = r[score_key]
            out[vi, si] = v if blend is None else blend * scores[vi, si] + (1 - blend) * v
        return out

    def eval_variant(name, fused):
        fs, gt = full_frame_scores(fused)
        return {'name': name, 'auc': auc_rank(fs, gt), 'ap': ap_score(fs, gt)}

    # baseline
    fs_t1, gt_f = full_frame_scores(scores)
    auc_t1, ap_t1 = auc_rank(fs_t1, gt_f), ap_score(fs_t1, gt_f)
    print(f"[analyze] tier-1 baseline: AUC={auc_t1:.4f} AP={ap_t1:.4f}")

    variants = [eval_variant('tier1', scores),
                eval_variant('replace_b', fused_with('score_b')),
                eval_variant('replace_a', fused_with('score_a')),
                eval_variant('replace_c', fused_with('score_c')),
                eval_variant('blend_0.5_b', fused_with('score_b', blend=0.5))]
    for v in variants:
        print(f"    {v['name']:>14}: AUC={v['auc']:.4f} (delta {v['auc']-auc_t1:+.4f}) "
              f"AP={v['ap']:.4f} (delta {v['ap']-ap_t1:+.4f})")

    # alert / event metrics on tier1 vs replace_b
    def ops_metrics(fused):
        alerts, tp = 0, 0
        recs, ttas = [], []
        for i in range(len(ids)):
            s_v = np.repeat(fused[i, :lens[i]], 1)  # snippet-level
            lab = labels[i, :lens[i]]
            a = s_v > TAU
            alerts += a.sum()
            tp += (a & (lab == 1)).sum()
            es = event_stats(s_v, lab, TAU)
            if es:
                recs.append(es[0])
                if es[1] is not None:
                    ttas.append(es[1])
        prec = tp / max(alerts, 1)
        alerts_per_hour = alerts / (lens.sum() * SNIP / FPS) * 3600
        return {'precision@0.5': float(prec),
                'alerts_per_hour': float(alerts_per_hour),
                'event_recall': float(np.mean(recs)),
                'tta_snippets_mean': float(np.mean(ttas)),
                'tta_seconds_mean': float(np.mean(ttas) * SNIP / FPS)}

    ops = {'tier1': ops_metrics(scores), 'replace_b': ops_metrics(fused_with('score_b'))}
    print(f"[analyze] ops: {json.dumps(ops, indent=2)}")

    # VLM accuracy on escalated clips
    esc_gt = np.array([labels[vi, si] for vi, si in zip(vis, sis)])
    esc_t1 = np.array([(scores[vi, si] > TAU) != bool(g)
                       for (vi, si), g in zip(zip(vis, sis), esc_gt)])
    acc = {}
    for key, thr in (('score_b', 0.5), ('score_a', None), ('score_c', 0.5)):
        v = np.array([vlm[(vi, si)][key] for vi, si in zip(vis, sis)])
        if thr is None:
            thr = float(np.median(v))
        acc[key] = {'thr': thr, 'acc': float(((v > thr).astype(int) == esc_gt).mean())}
    acc['tier1_acc'] = float(1 - esc_t1.mean())
    print(f"[analyze] escalated-clip accuracy: {json.dumps(acc, indent=2)}")

    # ---- 2.4 marginal-value curve ----
    rng = np.random.RandomState(SEED)
    esc_pairs = list(zip(vis.tolist(), sis.tolist()))
    esc_s = np.array([scores[vi, si] for vi, si in esc_pairs])
    Ks = [0.10, 0.25, 0.50, 1.00]
    curve = {'Random': [], 'RawTopK': []}
    order_by_s = np.argsort(-esc_s)
    for K in Ks:
        m = int(round(K * len(esc_pairs)))
        for rule in curve:
            aucs = []
            for seed in (0, 1, 2):
                r = np.random.RandomState(seed)
                if rule == 'Random':
                    chosen = set(r.choice(len(esc_pairs), m, replace=False).tolist())
                else:
                    chosen = set(order_by_s[:m].tolist())
                subset = {esc_pairs[i] for i in chosen}
                fused = fused_with('score_b', subset=subset)
                fs, gt = full_frame_scores(fused)
                aucs.append(auc_rank(fs, gt))
            curve[rule].append({'K': K, 'calls': m,
                                'dAUC_mean': float(np.mean(aucs) - auc_t1),
                                'dAUC_std': float(np.std(aucs))})
            print(f"[2.4] {rule} K={K}: dAUC={np.mean(aucs)-auc_t1:+.4f} "
                  f"+-{np.std(aucs):.4f}")

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for rule, marker in (('Random', 'o--'), ('RawTopK', 's-')):
        xs = [c['calls'] for c in curve[rule]]
        ys = [c['dAUC_mean'] for c in curve[rule]]
        es = [c['dAUC_std'] for c in curve[rule]]
        ax.errorbar(xs, ys, yerr=es, fmt=marker, label=rule, capsize=3)
    ax.axhline(0, color='k', lw=0.8)
    ax.set_xlabel('VLM calls spent (of 5,965 escalated)')
    ax.set_ylabel('delta AUC vs tier-1')
    ax.set_title('E0.6 marginal value of a VLM call')
    ax.legend()
    fig.tight_layout()
    os.makedirs(os.path.join(OUT, 'figures'), exist_ok=True)
    fig.savefig(os.path.join(OUT, 'figures', 'marginal_value.png'), dpi=150)
    plt.close(fig)

    doc = {
        'config': {'tau': TAU, 'margin': MARGIN, 'seed': SEED,
                   'prompt': PROMPT, 'fusion_primary': 'replace_b',
                   'score_extraction': {
                       'primary_b': 'P(Yes) at first generated token, softmax '
                                    'over {Yes,No,yes,no}',
                       'c': 'hard parse of first word of same answer',
                       'a': 'ATS (UR-DMU) mean over 12 frame scores; NOT the '
                            'MLLM verdict (separately trained sampler); fp16 '
                            'pixels instead of repo bf16 (sm_70)'}},
        'population': {'escalated_clips': len(vis),
                       'vlm_verdicts': len(vlm)},
        'baseline': {'auc_tier1': auc_t1, 'ap_tier1': ap_t1},
        'variants': variants,
        'ops': ops,
        'escalated_accuracy': acc,
        'marginal_value': curve,
    }
    with open(os.path.join(OUT, 'vlmvalue.json'), 'w') as f:
        json.dump(doc, f, indent=2)
    print(f"[saved] {OUT}/vlmvalue.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--freeze', action='store_true')
    ap.add_argument('--extract', action='store_true')
    ap.add_argument('--worker', action='store_true')
    ap.add_argument('--gpu', type=int, default=0)
    ap.add_argument('--shard', type=int, default=0)
    ap.add_argument('--n-shards', type=int, default=4)
    ap.add_argument('--batch', type=int, default=4)
    ap.add_argument('--analyze', action='store_true')
    args = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    if args.freeze:
        freeze()
    if args.extract:
        extract()
    if args.worker:
        run_worker(args.gpu, args.shard, args.n_shards, args.batch)
    if args.analyze:
        analyze()


if __name__ == '__main__':
    main()
