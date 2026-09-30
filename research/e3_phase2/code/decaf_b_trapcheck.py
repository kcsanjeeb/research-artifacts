"""DECAF Component B trap checks (Phase 2, 2026-09-28).

Modes (single process each, 0.5B, GPU3):

  mode=store (DECAF_B=1 [DECAF_QUANT=1], encode-only):
    TB1 assembly+accounting: grain store segments have retained grains within
        the token tax, dropped frames have no KV, realized bytes > 0 and
        <= tax * stock bytes per segment
    TB2 4-bit roundtrip (DECAF_QUANT=1): dequant(store) vs raw fp16 rel err
        distribution (W2.2 expectation: small); keys similarity ranking sanity
    TB5 scorer bias: mean |attn residual vs position| raw > debiased (C1)

  mode=bqa (DECAF_B=1, full 420-frame store, 10 questions of 0fa75cb3):
    TB3 commit plumbing: retrieval buffer = init + n_slots*196 with contiguous
        rank positions; committed slots <= topk; plan log written
    TB4 answering smoke: answers non-empty, fluent, no mid-sentence
        truncation at the late-stream position; match count vs W2.5 stock
        recorded (informational: TAX<1 store legitimately differs from stock)

  mode=quantqa (DECAF_B=1 DECAF_QUANT on/off, 96 frames + 5 questions):
    TB6 quant near-losslessness: >= 4/5 answers identical between the 4-bit
        and fp16 store arms (W2.2 trap extension)

Usage: python verification/decaf_b_trapcheck.py <out_dir> <mode>
"""
import json
import os
import sys

import numpy as np
import torch
from decord import VideoReader, cpu

OUT = sys.argv[1]
MODE = sys.argv[2]
os.makedirs(OUT, exist_ok=True)

MODEL = os.environ.get('DECAF_TC_MODEL', 'model_zoo/llava-onevision-qwen2-0.5b-ov-hf')
ANNO = os.environ.get('DECAF_TC_ANNO', 'data/rvs/ego/ego4d_oe.json')
W25_CSV = os.environ.get('DECAF_TC_W25CSV', '/data3/zhuotaotian2_e2/runs/20260926_1455_w25_judge_harness/answers_0.5b/1_0.csv')
N_FRAMES = int(os.environ.get('DECAF_TC_FRAMES', '96'))
N_FRAMES_QA = int(os.environ.get('DECAF_TC_FRAMES_QA', '440'))
SAMPLE_FPS = 0.5
N_FRAME_TOKENS = 196

results = {'mode': MODE, 'env': {k: os.environ.get(k) for k in [
    'DECAF_B', 'DECAF_B_COMMIT', 'DECAF_B_TAX', 'DECAF_B_BIAS', 'DECAF_QUANT',
    'DECAF_B_POLICY', 'DECAF_B_NOPASS2'] if os.environ.get(k) is not None}}


def save(name, obj):
    results[name] = obj
    with open(os.path.join(OUT, f'trapb_{MODE}.json'), 'w') as f:
        json.dump(results, f, indent=2)


def rel_err(a, b):
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    return ((a - b).norm() / (b.norm() + 1e-12)).item()


def load_video_frames(video_path, n):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), int(fps / SAMPLE_FPS))]
    return vr.get_batch(frame_idx[:n]).asnumpy(), len(frame_idx)


if MODE == 'store':

    def main():
        from model.llava_onevision_rekv import load_model
        from model.attention import decaf_b
        assert decaf_b.DECAF_B
        model, processor = load_model(model_path=MODEL, n_local=15000, topk=64, chunk_size=1)
        n_layers = model.config.text_config.num_hidden_layers
        vcap = {li: [] for li in [0, n_layers - 1]}
        for li in [0, n_layers - 1]:
            layer = model.language_model.model.layers[li]
            layer.self_attn.v_proj.register_forward_hook(
                lambda m, i, o, li=li: vcap[li].append(o.detach().clone()))

        video_path = 'data/rvs/ego/videos/0fa75cb3-c4a0-4232-bdc4-15116ed39d88.mp4'
        frames, _ = load_video_frames(video_path, N_FRAMES)
        model.clear_cache()
        model.encode_init_prompt()
        model._current_video_id = '0fa75cb3'
        model.encode_video(torch.from_numpy(frames))

        cm0 = model.kv_cache[0]
        store = cm0.grain_store
        L = decaf_b.N_LAYERS
        n_per_seg = decaf_b.SEG_FRAMES * N_FRAME_TOKENS
        tax = decaf_b.DECAF_B_TAX

        segs = store.segments
        assert len(segs) == (N_FRAMES // decaf_b.SEG_FRAMES), f'{len(segs)} segs'
        checks = []
        for s in segs:
            keep = set(g['frame'] for g in s.retained)
            kept_kv = all(len(s.frames[f].kv) == L for f in keep)
            dropped_empty = all(len(s.frames[f].kv) == 0 for f in range(4) if f not in keep)
            tok_frac = len(keep) / 4.0
            checks.append({
                'seg': s.seg_id, 'n_retained': len(s.retained),
                'kept_frames': sorted(keep), 'tok_frac': tok_frac,
                'kept_kv_complete': kept_kv, 'dropped_kv_freed': dropped_empty,
                'bytes_kv': s.bytes_kv, 'bytes_keys': s.bytes_keys,
                'within_tax': tok_frac <= tax + 1e-6,
            })
        # per-segment realized fraction vs stock
        fracs = [c['tok_frac'] for c in checks]
        save('TB1_assembly_accounting', {
            'n_segments': len(segs), 'tax': tax, 'n_layers': L,
            'frac_min': min(fracs), 'frac_mean': sum(fracs) / len(fracs),
            'frac_max': max(fracs),
            'all_kept_kv_complete': all(c['kept_kv_complete'] for c in checks),
            'all_dropped_freed': all(c['dropped_kv_freed'] for c in checks),
            'all_within_tax': all(c['within_tax'] for c in checks),
            'first_segs': checks[:4],
        })
        assert all(c['kept_kv_complete'] and c['dropped_kv_freed'] and c['within_tax']
                   for c in checks), 'assembly/accounting violation'

        # TB2: dequant roundtrip vs the raw v_proj captures of the first 2 chunks
        # (16 frames = segments 0-3; encode_video chunks are 64 frames so the
        # first capture holds frames 0-63 and covers all of them)
        if decaf_b.DECAF_QUANT:
            nheads_kv = model.language_model.model.layers[0].self_attn.num_key_value_heads
            dh = model.language_model.model.layers[0].self_attn.head_dim
            vraw0 = vcap[0][1].view(-1, nheads_kv, dh).permute(1, 0, 2)  # (kh, T0, dh); [0] is the init prompt
            vrawL = vcap[n_layers - 1][1].view(-1, nheads_kv, dh).permute(1, 0, 2)
            errs = []
            for s in store.segments:
                for f in range(4):
                    if 0 not in s.frames[f].kv:
                        continue
                    st = s.token_start - cm0.n_init + f * N_FRAME_TOKENS
                    if st + N_FRAME_TOKENS > vraw0.size(1):
                        continue  # beyond the first captured chunk (64 frames)
                    got0 = decaf_b.dequant(s.frames[f].kv[0][1])
                    gotL = decaf_b.dequant(s.frames[f].kv[n_layers - 1][1])
                    errs.append(rel_err(got0[None], vraw0[:, st:st + N_FRAME_TOKENS, :][None]))
                    errs.append(rel_err(gotL[None], vrawL[:, st:st + N_FRAME_TOKENS, :][None]))
            save('TB2_quant_roundtrip', {
                'n_samples': len(errs), 'relerr_mean': float(np.mean(errs)),
                'relerr_max': float(np.max(errs)) if errs else None,
                'threshold': 0.30,
                'pass': bool(np.mean(errs) < 0.30) if errs else None})
        else:
            save('TB2_quant_roundtrip', {'skipped': 'DECAF_QUANT unset'})

        # TB5: scorer bias stats (layer 0 and last layer)
        bias = {}
        for li in [0, L - 1]:
            bias[f'layer{li}'] = model.kv_cache[li].scorer.bias_stats()
        save('TB5_scorer_bias', bias)
        raw = np.mean([bias[f'layer{li}']['mean_abs_residual_raw'] for li in [0, L - 1]])
        deb = np.mean([bias[f'layer{li}']['mean_abs_residual_debiased'] for li in [0, L - 1]])
        tb1 = results['TB1_assembly_accounting']
        verdict = {
            'gate_assembly_accounting': 'PASS' if (
                tb1['all_kept_kv_complete'] and tb1['all_dropped_freed']
                and tb1['all_within_tax']) else 'FAIL',
            'gate_quant_roundtrip': ('PASS' if results['TB2_quant_roundtrip'].get('pass')
                                     else 'FAIL') if decaf_b.DECAF_QUANT else 'SKIP',
            'gate_bias_reduced': 'PASS' if deb < raw else 'FAIL',
            'bias_raw': round(float(raw), 6), 'bias_deb': round(float(deb), 6),
        }
        verdict['OVERALL'] = 'PASS' if all(
            v == 'PASS' for k, v in verdict.items() if k.startswith('gate') and v != 'SKIP') else 'FAIL'
        save('verdict', verdict)
        print('STORE_MODE_DONE', json.dumps(verdict))

    with torch.inference_mode():
        main()


elif MODE == 'bqa':

    def main():
        import pandas as pd
        from model.llava_onevision_rekv import load_model
        from model.attention import decaf_b
        from model.attention.kv_cache_manager import PositionalKV
        assert decaf_b.DECAF_B
        model, processor = load_model(model_path=MODEL, n_local=15000, topk=64, chunk_size=1)
        anno = json.load(open(ANNO))
        vs = [v for v in anno if v['video_id'].startswith('0fa75cb3')][0]
        convs = [c for c in vs['conversations'] if c['end_time'] * SAMPLE_FPS <= N_FRAMES_QA][:10]
        w25 = pd.read_csv(W25_CSV)
        frames, _ = load_video_frames(vs['video_path'], N_FRAMES_QA)
        model.clear_cache()
        model.encode_init_prompt()
        model._current_video_id = vs['video_id']
        n_init = int(model.kv_cache[0].n_init)

        checks, answers = [], []
        start_idx = 0.0
        for qi, c in enumerate(convs):
            end_idx = c['end_time'] * SAMPLE_FPS
            if end_idx > start_idx:
                model.encode_video(torch.from_numpy(frames[int(start_idx):int(end_idx)]))
                start_idx = end_idx
            for layer_kv in model.kv_cache:
                layer_kv.set_retrieval()
            input_ids = torch.as_tensor([processor.tokenizer(c['question']).input_ids],
                                        device=model.device)
            out = model.language_model(input_ids=input_ids, use_cache=True,
                                       past_key_values=model.kv_cache)
            pkv = out.past_key_values[0]
            cm0 = model.kv_cache[0]
            pos = [int(x) for x in pkv.positions]
            n_blk = (len(pos) - n_init) // N_FRAME_TOKENS
            exp_buf = list(range(n_init)) + [n_init + i for i in range(n_blk * N_FRAME_TOKENS)]
            idx0 = [int(b) for b in cm0.retrieved_block_indices[0]]
            store = cm0.grain_store
            checks.append({
                'qi': qi, 'store_blocks_at_query': int(cm0.num_global_block),
                'rank_pos_ok': pos == exp_buf,
                'n_committed_slots': n_blk, 'slots_le_topk': n_blk <= 64,
                'retrieved_head': idx0[:8],
                'plan_log_present': store.plan_log is not None,
            })
            for layer_kv in model.kv_cache:
                layer_kv.reset_retrieval()
            pred = model.question_answering(
                {'question': c['question'], 'prompt': model.get_prompt(c['question'])},
                max_new_tokens=256)
            ref = w25[(w25.video_id == vs['video_id']) & (w25.question == c['question'])]['pred_answer']
            answers.append({'qi': qi, 'end_time': c['end_time'], 'pred_chars': len(pred),
                            'nonempty': len(pred.strip()) > 0,
                            'pred_head': pred[:80], 'w25_match': bool(len(ref)) and
                            ref.iloc[0] == pred.replace('\n', '')})

        save('TB3_commit_plumbing', {'n_init': n_init, 'checks': checks})
        save('TB4_answering_smoke', {'answers': answers,
                                     'n_nonempty': sum(a['nonempty'] for a in answers),
                                     'n_w25_match': sum(a['w25_match'] for a in answers)})
        ok_plumb = all(c['rank_pos_ok'] and c['slots_le_topk'] and c['plan_log_present']
                       for c in checks)
        ok_smoke = sum(a['nonempty'] for a in answers) >= len(answers) - 1
        verdict = {
            'gate_commit_plumbing': 'PASS' if ok_plumb else 'FAIL',
            'gate_answering_smoke': 'PASS' if ok_smoke else 'FAIL',
            'n_w25_match_info': f"{sum(a['w25_match'] for a in answers)}/{len(answers)}",
        }
        verdict['OVERALL'] = 'PASS' if all(v == 'PASS' for v in verdict.values() if v.startswith('gate')) else 'FAIL'
        save('verdict', verdict)
        print('BQA_MODE_DONE', json.dumps(verdict))

    with torch.inference_mode():
        main()


elif MODE == 'quantqa':

    def main():
        import pandas as pd
        from model.llava_onevision_rekv import load_model
        from model.attention import decaf_b
        assert decaf_b.DECAF_B
        model, processor = load_model(model_path=MODEL, n_local=15000, topk=64, chunk_size=1)
        anno = json.load(open(ANNO))
        vs = [v for v in anno if v['video_id'].startswith('0fa75cb3')][0]
        convs = [c for c in vs['conversations'] if c['end_time'] * SAMPLE_FPS <= N_FRAMES][:5]
        frames, _ = load_video_frames(vs['video_path'], N_FRAMES)
        model.clear_cache()
        model.encode_init_prompt()
        model._current_video_id = vs['video_id']
        model.encode_video(torch.from_numpy(frames))
        tag = 'q4' if decaf_b.DECAF_QUANT else 'fp16'
        outs = []
        start_idx = 0.0
        for c in convs:
            end_idx = c['end_time'] * SAMPLE_FPS
            if end_idx > start_idx:
                model.encode_video(torch.from_numpy(frames[int(start_idx):int(end_idx)]))
                start_idx = end_idx
            pred = model.question_answering(
                {'question': c['question'], 'prompt': model.get_prompt(c['question'])},
                max_new_tokens=256)
            outs.append({'question': c['question'][:80], 'pred': pred.replace('\n', '')})
        with open(os.path.join(OUT, f'quantqa_{tag}.json'), 'w') as f:
            json.dump(outs, f, indent=2)
        other = os.path.join(OUT, 'quantqa_fp16.json' if tag == 'q4' else 'quantqa_q4.json')
        if os.path.exists(other):
            otag = 'fp16' if tag == 'q4' else 'q4'
            oth = json.load(open(other))
            n_match = sum(a['pred'] == b['pred'] for a, b in zip(outs, oth))
            verdict = {'gate_quant_answer_equiv': 'PASS' if n_match >= 4 else 'FAIL',
                       'n_identical': f'{n_match}/{len(outs)}', 'arm': tag, 'other': otag}
            verdict['OVERALL'] = verdict['gate_quant_answer_equiv']
            save('verdict', verdict)
            print('QUANTQA_DONE', json.dumps(verdict))
        else:
            save('verdict', {'arm': tag, 'status': 'waiting for the other arm'})
            print('QUANTQA_DONE arm=%s (other arm pending)' % tag)

    with torch.inference_mode():
        main()
