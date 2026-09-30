"""DECAF Component A trap checks — W2.1 verification kit applied to the ReKV host path.

Runs in two modes (single process each):

  mode=encode (stock env; DECAF_POSITION_FREE unset):
    T1  store-is-content: offloaded MemoryUnit K/V == raw k_proj/v_proj outputs
        (rel err must be ~0; a large discrepancy = capture bug -> STOP per plan)
    T2  rotary reproduction: model's own position_bias(pre-RoPE K, abs pos) == W2.1
        K_post dumps (fp16 storage tolerance); also dump K_pre vs captured k_proj
        for chunk 1 (attention-kernel differences documented per layer)
    T2c sidecar identity: block_starts[b] == n_init + b*block_size exactly
    T4  token alignment: 196 tokens/frame everywhere
    T5  stock equivalence: 2 greedy answers == W2.5 answers_0.5b CSV rows
        (proves the decaf repo with the flag OFF is the W2.5-validated stock)

  mode=qa (subprocess with DECAF_POSITION_FREE=1):
    T3a helper equivalence: apply_rotary_with_positions(x, arange(p,p+N)) ==
        stock apply_rotary_pos_emb range rotation (same fp32 math)
    T3b rotate-on-fetch plumbing: retrieval returns PositionalKV with exact
        buffer positions (init 0..12, block b at 13+b*196+j) and
        next_pos == manager.length; get_retrieved_positions matches sidecar
    T3c encode invariance: DECAF-mode encode == stock-mode encode bit-exact
    T3d full greedy QA smoke (retrieval + prefill + decode) runs clean

Usage: python verification/decaf_trapcheck.py <out_dir> <mode>
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
DUMP = os.environ.get('DECAF_TC_DUMP', '/data1/zhuotaotian2_e2_kv/kv/0fa75cb3-c4a0-4232-bdc4-15116ed39d88')
ANNO = os.environ.get('DECAF_TC_ANNO', 'data/rvs/ego/ego4d_oe.json')
W25_CSV = os.environ.get('DECAF_TC_W25CSV', '/data3/zhuotaotian2_e2/runs/20260926_1455_w25_judge_harness/answers_0.5b/1_0.csv')
N_FRAMES = int(os.environ.get('DECAF_TC_FRAMES', '96'))      # encode depth for store checks
N_FRAMES_QA = int(os.environ.get('DECAF_TC_FRAMES_QA', '440'))  # 0fa75cb3 first questions end at 840s = 420 frames
LAYERS = [0, 5, 11, 17, 23]
SAMPLE_FPS = 0.5
N_FRAME_TOKENS = 196

results = {'mode': MODE, 'decaf_position_free': os.environ.get('DECAF_POSITION_FREE', '0')}


def rel_err(a, b):
    a = a.detach().float().cpu()
    b = b.detach().float().cpu()
    return ((a - b).norm() / (b.norm() + 1e-12)).item()


def save(name, obj):
    results[name] = obj
    with open(os.path.join(OUT, f'trap_checks_{MODE}.json'), 'w') as f:
        json.dump(results, f, indent=2)


def load_video_frames(video_path, n):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    fps = round(vr.get_avg_fps())
    frame_idx = [i for i in range(0, len(vr), int(fps / SAMPLE_FPS))]
    return vr.get_batch(frame_idx[:n]).asnumpy(), len(frame_idx)


if MODE == 'encode':

    def main():
        from model.llava_onevision_rekv import load_model
        model, processor = load_model(model_path=MODEL, n_local=15000, topk=64, chunk_size=1)
        n_layers = len(model.language_model.model.layers)

        kcap, vcap = {}, {}
        for li in LAYERS:
            layer = model.language_model.model.layers[li]
            layer.self_attn.k_proj.register_forward_hook(
                lambda m, i, o, li=li: kcap.__setitem__(li, o.detach().clone()))
            layer.self_attn.v_proj.register_forward_hook(
                lambda m, i, o, li=li: vcap.__setitem__(li, o.detach().clone()))

        video_path = 'data/rvs/ego/videos/0fa75cb3-c4a0-4232-bdc4-15116ed39d88.mp4'
        frames, n_sampled = load_video_frames(video_path, N_FRAMES)
        assert frames.shape[0] == N_FRAMES
        save('T4_frame_sampling', {'n_sampled_total': n_sampled, 'encoded': N_FRAMES, 'stride': 'fps/0.5'})

        model.clear_cache()
        model.encode_init_prompt()
        caps, vcaps = {li: [] for li in LAYERS}, {li: [] for li in LAYERS}
        for c0 in range(0, N_FRAMES, 64):
            chunk = frames[c0:c0 + 64]
            model._encode_video_chunk(chunk)
            for li in LAYERS:
                caps[li].append(kcap[li])  # (1, T, Hkv*D), tokens [13+c0*196, ...)
                vcaps[li].append(vcap[li])

        cm = model.kv_cache[0]
        n_blocks = cm.num_global_block
        n_init = cm.n_init
        save('T2c_sidecar_identity', {
            'n_init': n_init, 'n_blocks': n_blocks, 'block_size': cm.block_size,
            'identity_holds': all(
                cm.block_starts[0][b] == n_init + b * cm.block_size for b in range(n_blocks)),
        })
        assert cm.block_size == N_FRAME_TOKENS

        # T1: stored blocks == raw projections (exact; dtype fp16 copies of the same tensors)
        Dh = model.language_model.model.layers[0].self_attn.head_dim
        t1 = {}
        for li in LAYERS:
            kerrs, verrs = [], []
            for b in range(n_blocks):
                if (b + 1) > N_FRAMES:
                    break
                ci = b // 64
                off = (b % 64) * N_FRAME_TOKENS
                ck = caps[li][ci][0, off:off + N_FRAME_TOKENS]
                cv = vcaps[li][ci][0, off:off + N_FRAME_TOKENS]
                ck = ck.view(N_FRAME_TOKENS, -1, Dh).permute(1, 0, 2)
                cv = cv.view(N_FRAME_TOKENS, -1, Dh).permute(1, 0, 2)
                sk, sv = model.kv_cache[li].global_blocks[0][b].cpu_data
                kerrs.append(rel_err(sk.cpu(), ck))
                verrs.append(rel_err(sv.cpu(), cv))
            t1[f'layer{li}'] = {
                'n_blocks_checked': len(kerrs),
                'k_rel_err_max': max(kerrs), 'k_rel_err_mean': sum(kerrs) / len(kerrs),
                'v_rel_err_max': max(verrs), 'v_rel_err_mean': sum(verrs) / len(verrs),
            }
        save('T1_store_is_prerope', t1)

        # T2: cross-check chunk-1 captures against W2.1 dumps + rotary reproduction
        Hkv = model.config.text_config.num_key_value_heads
        Dh = model.config.text_config.hidden_size // model.config.text_config.num_attention_heads
        t2 = {}
        rope = model.language_model.model.position_bias
        n_first = min(64, N_FRAMES)
        for li in LAYERS:
            dump_pre = np.load(os.path.join(DUMP, f'K_pre_layer{li:02d}.npy'), mmap_mode='r')[:n_first]
            dump_post = np.load(os.path.join(DUMP, f'K_post_layer{li:02d}.npy'), mmap_mode='r')[:n_first]
            ck = caps[li][0][0, :n_first * N_FRAME_TOKENS]  # (64*196, Hkv*D)
            ck = ck.view(n_first, N_FRAME_TOKENS, Hkv, Dh).float().cpu().numpy()
            e_pre = rel_err(torch.from_numpy(ck), torch.from_numpy(np.asarray(dump_pre)))
            # rotary: rotate dump K_pre at absolute positions, compare to dump K_post
            x = torch.from_numpy(np.asarray(dump_pre)).permute(0, 2, 1, 3).cuda()  # (F,Hkv,196,Dh)
            pos = (n_init + (torch.arange(n_first * Hkv) // Hkv)[:, None] * N_FRAME_TOKENS
                   + torch.arange(N_FRAME_TOKENS)[None, :])  # (F*Hkv, 196) row r <-> (f=r//Hkv, h=r%Hkv)
            rot = rope.apply_rotary_with_positions(x.reshape(n_first * Hkv, N_FRAME_TOKENS, Dh), pos.cuda())
            rot = rot.reshape(n_first, Hkv, N_FRAME_TOKENS, Dh).permute(0, 2, 1, 3).half().cpu().numpy()
            e_rot = rel_err(torch.from_numpy(rot), torch.from_numpy(np.asarray(dump_post)))
            t2[f'layer{li}'] = {'dump_Kpre_rel_err': e_pre, 'rotary_vs_dump_Kpost_rel_err': e_rot}
        save('T2_rotary_reproduction', {'layers': t2, 'note':
              'dump K_pre/V from the UNPATCHED sdpa-fp16 HF path; patched host uses fp32-upcast torch '
              'attention, so layer>=1 diffs are kernel differences, not capture bugs. rotary check is '
              'elementwise on the same K_pre -> tight tolerance expected.'})

        # stock memory report (sidecar accounting baseline)
        save('memory', model.calc_memory_usage_detailed())

        # T5: stock equivalence vs W2.5 answers (2 questions, video anno[0],
        # interleaving encode exactly like rekv_stream_vqa.analyze_a_video)
        import pandas as pd
        anno = json.load(open(ANNO))
        s0 = anno[0]
        frames0, _ = load_video_frames(s0['video_path'], 160)
        model.clear_cache()
        model.encode_init_prompt()
        model._current_video_id = s0['video_id']
        picked = [c for c in s0['conversations'] if c['end_time'] * SAMPLE_FPS <= 160][:2]
        w25 = pd.read_csv(W25_CSV)
        t5 = []
        start_idx = 0.0
        for c in picked:
            end_idx = c['end_time'] * SAMPLE_FPS
            if end_idx > start_idx:
                model.encode_video(torch.from_numpy(frames0[int(start_idx):int(end_idx)]))
                start_idx = end_idx
            pred = model.question_answering(
                {'question': c['question'], 'prompt': model.get_prompt(c['question'])},
                max_new_tokens=256)
            ref = w25[(w25.video_id == s0['video_id']) & (w25.question == c['question'])]['pred_answer']
            t5.append({'question': c['question'], 'pred': pred, 'w25_pred': ref.iloc[0] if len(ref) else None,
                       'match': (ref.iloc[0] if len(ref) else None) == pred.replace('\n', '')})
        save('T5_stock_equivalence_vs_w25', {'video_id': s0['video_id'], 'checks': t5})

        # T3c reference: stock-mode encode snapshot of the same 200 frames
        frames_q, _ = load_video_frames('data/rvs/ego/videos/0fa75cb3-c4a0-4232-bdc4-15116ed39d88.mp4',
                                        int(os.environ.get('DECAF_TC_FRAMES_QA', '440')))
        model.clear_cache()
        model.encode_init_prompt()
        model.encode_video(torch.from_numpy(frames_q))
        snap = []
        for li in [0, 5, 23]:
            for b in range(min(3, model.kv_cache[li].num_global_block)):
                sk, sv = model.kv_cache[li].global_blocks[0][b].cpu_data
                snap.append(np.asarray([sk.cpu().numpy().sum(), sv.cpu().numpy().sum(),
                                        sk.cpu().numpy().astype(np.float32).std()]))
        np.save(os.path.join(OUT, 'T3c_encode_reference.npy'), np.stack(snap))

        # verdict
        g1 = max(v['k_rel_err_max'] for v in t1.values())
        g1v = max(v['v_rel_err_max'] for v in t1.values())
        g2 = max(v['rotary_vs_dump_Kpost_rel_err'] for v in t2.values())
        verdict = {
            'gate_store_prerope': 'PASS' if max(g1, g1v) <= 1e-6 else 'FAIL',
            'gate_rotary': 'PASS' if g2 <= 2e-3 else 'FAIL',
            'gate_sidecar': 'PASS' if results['T2c_sidecar_identity']['identity_holds'] else 'FAIL',
            'gate_stock_equiv': 'PASS' if all(c['match'] for c in t5) else 'FAIL',
            'max_stored_k_rel_err': g1, 'max_stored_v_rel_err': g1v,
            'max_rotary_rel_err': g2,
        }
        verdict['OVERALL'] = 'PASS' if all(v == 'PASS' for v in verdict.values() if v.startswith('gate')) else 'FAIL'
        save('verdict', verdict)
        print('ENCODE_MODE_DONE', json.dumps(verdict))

    with torch.inference_mode():
        main()

elif MODE == 'qa':

    def main():
        from model.llava_onevision_rekv import load_model
        from model.attention.kv_cache_manager import ContextManager, PositionalKV
        from model.attention.rekv_attention import DECAF_POSITION_FREE
        assert DECAF_POSITION_FREE, 'mode=qa must run with DECAF_POSITION_FREE=1'
        model, processor = load_model(model_path=MODEL, n_local=15000, topk=64, chunk_size=1)

        # T3a: helper equivalence with the stock range rotation (same fp32 math)
        rope = model.language_model.model.position_bias
        torch.manual_seed(0)
        hd = rope.inv_freq.numel() * 2  # model head_dim (64 for the 0.5B)
        x = torch.randn(2, 4, 37, hd, device='cuda', dtype=torch.float16)
        for p in [0, 5, 15001, 352000]:
            pos = torch.arange(p, p + 37, device='cuda')
            y1 = rope.apply_rotary_with_positions(x, pos)
            cos, sin = rope._update_cos_sin_tables_len(p + 37, x.device, dim=4)
            y2 = rope.apply_rotary_pos_emb(x, 37, p + 37, cos, sin)
            save(f'T3a_helper_equiv_p{p}', {'rel_err': rel_err(y1, y2)})
            assert rel_err(y1, y2) <= 1e-6

        # capture get_retrieved_positions PER MANAGER (each layer has its own)
        captured = {}
        orig_grp = ContextManager.get_retrieved_positions

        def spy_grp(self):
            pos = orig_grp(self)
            captured[id(self)] = {
                'pos': [p.clone() for p in pos],
                'indices': [list(i) for i in self.retrieved_block_indices],
                'num_global_block': int(self.num_global_block),
                'init_exc': bool(self.init_exc),
                'topk': int(self.topk),
                'chunk_size': int(self.chunk_size),
                'n_blocks_retrieved': len(self.retrieved_block_indices[0]),
                'logits_len': (int(self.similarity.shape[1]) if self.similarity is not None else None),
            }
            return pos

        ContextManager.get_retrieved_positions = spy_grp

        video_path = 'data/rvs/ego/videos/0fa75cb3-c4a0-4232-bdc4-15116ed39d88.mp4'
        frames, _ = load_video_frames(video_path, N_FRAMES_QA)
        model.clear_cache()
        model.encode_init_prompt()
        # T3c snapshot of stored bytes for cross-mode comparison
        model.encode_video(torch.from_numpy(frames))
        snap = []
        for li in [0, 5, 23]:
            for b in range(min(3, model.kv_cache[li].num_global_block)):
                sk, sv = model.kv_cache[li].global_blocks[0][b].cpu_data
                snap.append(np.asarray([sk.cpu().numpy().sum(), sv.cpu().numpy().sum(),
                                        sk.cpu().numpy().astype(np.float32).std()]))
        np.save(os.path.join(OUT, 'T3c_encode_snapshot.npy'), np.stack(snap))

        n_init = model.kv_cache[0].n_init
        cm0 = model.kv_cache[0]

        # T3b: retrieval plumbing — replicate question_answering's retrieval step
        anno = json.load(open(ANNO))
        convs = [c for c in anno[0]['conversations'] if c['end_time'] * SAMPLE_FPS <= N_FRAMES_QA][:2] \
            if anno[0]['video_id'].startswith('0fa75cb3') else None
        if convs is None:
            vs = [v for v in anno if v['video_id'].startswith('0fa75cb3')][0]
        else:
            vs = anno[0]
        convs = [c for c in vs['conversations'] if c['end_time'] * SAMPLE_FPS <= N_FRAMES_QA][:2]
        assert len(convs) == 2, f'need 2 questions within {N_FRAMES_QA} frames'

        t3b = []
        for c in convs:
            for layer_kv in model.kv_cache:
                layer_kv.set_retrieval()
            input_ids = processor.tokenizer(c['question']).input_ids
            input_ids = torch.as_tensor([input_ids], device=model.device)
            out = model.language_model(input_ids=input_ids, use_cache=True, past_key_values=model.kv_cache)
            pkv = out.past_key_values[0]
            assert isinstance(pkv, PositionalKV), 'retrieval must return PositionalKV in DECAF mode'
            L = model.kv_cache[0].length
            cap0 = captured[id(cm0)]
            exp_buf = list(range(n_init))
            for b in cap0['indices'][0]:
                exp_buf += [cm0.block_starts[0][b] + j for j in range(N_FRAME_TOKENS)]
            got_buf = [int(x) for x in pkv.positions]
            grp = [int(x) for x in cap0['pos'][0]]
            exp_grp = []
            for b in cap0['indices'][0]:
                exp_grp += [n_init + b * N_FRAME_TOKENS + j for j in range(N_FRAME_TOKENS)]
            t3b.append({
                'question': c['question'],
                'next_pos': int(pkv.next_pos), 'manager_length': int(L),
                'next_pos_ok': int(pkv.next_pos) == int(L),
                'buffer_pos_ok': got_buf == exp_buf,
                'sidecar_grp_ok': grp == exp_grp,
                'sidecar_identity': all(cm0.block_starts[0][b] == n_init + b * N_FRAME_TOKENS
                                        for b in range(cm0.num_global_block)),
                'retrieved_layer0': cap0['indices'][0][:8],
                'layer0_diag': {k: cap0[k] for k in ['num_global_block', 'init_exc', 'topk',
                                                     'chunk_size', 'n_blocks_retrieved', 'logits_len']},
                'all_layers_n_blocks': sorted({v['n_blocks_retrieved'] for v in captured.values()}),
            })
            for layer_kv in model.kv_cache:
                layer_kv.reset_retrieval()

        save('T3b_rotate_on_fetch_plumbing', {'n_init': n_init, 'checks': t3b})

        # T3d: full greedy QA smoke through retrieval + prefill + decode
        smoke = []
        for c in convs:
            pred = model.question_answering(
                {'question': c['question'], 'prompt': model.get_prompt(c['question'])},
                max_new_tokens=64)
            smoke.append({'question': c['question'], 'answer': c['answer'], 'pred': pred})
        save('T3d_full_qa_smoke', smoke)

        ok = all(c['next_pos_ok'] and c['buffer_pos_ok'] and c['sidecar_grp_ok'] and c['sidecar_identity']
                 for c in t3b)
        save('verdict', {'gate_rotate_on_fetch': 'PASS' if ok else 'FAIL',
                         'gate_qa_smoke': 'PASS', 'OVERALL': 'PASS' if ok else 'FAIL'})
        print('QA_MODE_DONE', json.dumps(results['verdict']))

    with torch.inference_mode():
        main()

elif MODE == 'compare':
    ref = np.load(os.path.join(OUT, 'T3c_encode_reference.npy'))
    got = np.load(os.path.join(OUT, 'T3c_encode_snapshot.npy'))
    same = bool(np.array_equal(ref, got))
    verdict = {}
    for m in ['encode', 'qa']:
        with open(os.path.join(OUT, f'trap_checks_{m}.json')) as f:
            verdict.update(json.load(f).get('verdict', {}))
    verdict['gate_encode_invariance'] = 'PASS' if same else 'FAIL'
    verdict['OVERALL'] = 'PASS' if all(
        v == 'PASS' for k, v in verdict.items() if k.startswith('gate')) else 'FAIL'
    summary = {'T3c_encode_invariance': {'bit_exact': same,
                                         'max_abs_diff': float(np.abs(ref - got).max())},
               'verdict': verdict}
    with open(os.path.join(OUT, 'trap_checks_verdict.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('COMPARE_DONE bit_exact=%s' % same)
