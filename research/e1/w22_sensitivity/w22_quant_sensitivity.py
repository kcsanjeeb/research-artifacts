#!/usr/bin/env python3
"""W2.2 — Does TTKV's sensitivity finding transfer to video KV?

Extends the T1 machinery (t1_rekv_shard.py): re-encodes RVS-Ego videos with
ReKV-0.5B (KV was not persisted), then for sampled retrieved blocks quantizes
ONE block at a time at 2/4/8-bit (symmetric uniform quant-dequant simulating
storage at b bits; K per-(head,channel) scales over tokens, V per-(head,token)
scales over channels, fp32 scales, KIVI-style grouping) while holding the query,
the selection set (logged top-64 from T1) and all other blocks fixed.

Distortion (primary, parallel to T1's LOO drop):
    drop = f1_ext_baseline - f1_gold(pred_with_block_j_quantized)
Secondary: out_f1 = token-F1(pred_q, pred_ext_baseline)  (direct output change)

Baseline path: external retrieval with the logged selection set, unquantized
(the external decode path is deterministic; restore after each quantization is
bit-exact to this baseline — verified in w22_probe.py). The retrieval index
(block representative keys, block_k) is left at full precision: only the stored
KV payload of block j is quantized, matching "all other blocks fixed".

Block frequency / quintiles: counted from
~/e1/runs/20260925_2237_t1_rekv_sharded/artifacts/t1_rekv_log.jsonl (120
queries). Quintiles are taken over USED blocks (f>0) per video — 96% of blocks
are never selected, so all-block quintiles are degenerate. Q1=Cold .. Q5=Hot by
frequency rank. We sample --per_quintile blocks per quintile per video.

Usage (env e1rekv): python w22_quant_sensitivity.py --gpu 1 --shard 0 --nshards 3 --run_dir R
"""
import argparse
import json
import os
import re
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch  # module-level: helper functions below use it; CUDA_VISIBLE_DEVICES
              # is set in main() before any CUDA work, which is what matters.

REKV = os.path.expanduser("~/e1/ReKV")
sys.path.insert(0, REKV)
sys.path.insert(0, os.path.expanduser("~/e1/baselines"))

T1_LOG = os.path.expanduser(
    "~/e1/runs/20260925_2237_t1_rekv_sharded/artifacts/t1_rekv_log.jsonl")
SEED = 20260926
BITS = (2, 4, 8)


def tok_f1(a, b):
    A = re.findall(r"[a-z0-9]+", a.lower())
    B = re.findall(r"[a-z0-9]+", b.lower())
    if not A or not B:
        return 0.0
    ca, cb = Counter(A), Counter(B)
    inter = sum((ca & cb).values())
    p, r = inter / len(A), inter / len(B)
    return 2 * p * r / (p + r) if p + r else 0.0


def quantize_tensor(t, bits, is_key):
    x = t.float()
    qmax = 2 ** (bits - 1) - 1
    if is_key:  # per (head, channel) scale over tokens
        scale = x.abs().amax(dim=1, keepdim=True) / qmax
    else:       # per (head, token) scale over channels
        scale = x.abs().amax(dim=2, keepdim=True) / qmax
    scale = scale.clamp_min(1e-8)
    q = torch.round(x / scale).clamp(-qmax - 1, qmax) * scale
    return q.to(t.dtype)


def quantize_block(model, j, bits):
    backup = []
    with torch.inference_mode():
        for cm in model.kv_cache:
            mu = cm.global_blocks[0][j]
            if mu.gpu_data is not None:
                mu.offload()
            k0, v0 = mu.cpu_data
            backup.append((k0.clone(), v0.clone()))
            k0.copy_(quantize_tensor(k0, bits, True))
            v0.copy_(quantize_tensor(v0, bits, False))
    return backup


def restore_block(model, j, backup):
    with torch.inference_mode():
        for cm, (k0, v0) in zip(model.kv_cache, backup):
            mu = cm.global_blocks[0][j]
            if mu.gpu_data is not None:
                mu.offload()
            mu.cpu_data[0].copy_(k0)
            mu.cpu_data[1].copy_(v0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, required=True)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--nshards", type=int, default=3)
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--per_quintile", type=int, default=5)
    ap.add_argument("--skip", default="",
                    help="comma-separated video_id prefixes to skip (resume)")
    args = ap.parse_args()
    skip = set(x for x in args.skip.split(",") if x)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    import torch
    from decord import VideoReader, cpu
    from model import llava_onevision_rekv

    # T1 log: selection sets + frequencies (per video, over its 12 queries)
    t1 = [json.loads(l) for l in open(T1_LOG)]
    by_vid = defaultdict(list)
    for r in t1:
        by_vid[r["video_id"]].append(r)
    vids = sorted(by_vid.items())
    vids = [v for i, v in enumerate(vids) if i % args.nshards == args.shard]
    print(f"shard {args.shard}/{args.nshards}: {len(vids)} videos", flush=True)

    os.makedirs(os.path.join(args.run_dir, "artifacts"), exist_ok=True)
    out_p = os.path.join(args.run_dir, "artifacts",
                         f"w22_quant_shard{args.shard}.jsonl")

    print("loading ReKV 0.5B ...", flush=True)
    model, _proc = llava_onevision_rekv.load_model(
        model_path=os.path.join(REKV,
                                "model_zoo/llava-onevision-qwen2-0.5b-ov-hf"),
        n_local=15000, topk=64, chunk_size=1)

    rng = np.random.RandomState(SEED + args.shard)
    with open(out_p, "a") as fo:
        for vi, (vid, qrows) in enumerate(vids):
            if vid[:8] in skip:
                print(f"[{vi}] {vid[:8]} skipped (already done)", flush=True)
                continue
            vp = qrows[0].get("video_path")
            if vp is None:
                # t1 log rows carry no video_path; resolve via questions loader
                from rvs_experiment import load_questions as _lq
                qmap = {q["video_id"]: q["video_path"] for q in _lq()}
                vp = qmap[vid]
            t0 = time.time()
            vr = VideoReader(vp, ctx=cpu(0), num_threads=1)
            fps = round(vr.get_avg_fps())
            nfr = len(vr)
            frame_idx = list(range(0, nfr, max(1, int(fps / 0.5))))
            video_t = torch.from_numpy(vr.get_batch(frame_idx).asnumpy())
            model.clear_cache()
            model.encode_init_prompt()
            model.encode_video(video_t)
            print(f"[{vi}] {vid[:8]} encoded {len(frame_idx)} blocks in "
                  f"{time.time()-t0:.0f}s", flush=True)

            # --- external unquantized baselines for this video's queries ---
            base = {}
            for qi, r in enumerate(qrows):
                qq = r["question"]
                it = {"question": qq, "prompt": model.get_prompt(qq)}
                sel = sorted(set(int(b) for b in r["retrieved_blocks"]))
                pred = model.question_answering(it, max_new_tokens=64,
                                                retrieved_indices=[sel])
                base[qi] = {"pred": pred, "sel": sel,
                            "f1_gold": tok_f1(pred, r["answer"])}
            print(f"[{vi}] baselines done", flush=True)

            # --- quintiles from T1 frequency over used blocks ---
            freq = Counter()
            for r in qrows:
                freq.update(set(int(b) for b in r["retrieved_blocks"]))
            used = sorted(freq)
            order = sorted(used, key=lambda b: (freq[b], b))
            qn = len(order)
            quint = {}
            for rank, b in enumerate(order):
                quint[b] = min(4, rank * 5 // qn)  # Q1..Q5 -> 0..4
            byq = defaultdict(list)
            for b, q_ in quint.items():
                byq[q_].append(b)
            picked = []
            for q_ in range(5):
                blk = byq[q_]
                k = min(args.per_quintile, len(blk))
                picked += [(b, q_) for b in rng.choice(blk, size=k,
                                                       replace=False)]
            print(f"[{vi}] used={qn}, sampled {len(picked)} blocks", flush=True)

            n_ev = 0
            for j, q_ in picked:
                qrows_j = [qi for qi, r in enumerate(qrows)
                           if j in base[qi]["sel"]]
                if not qrows_j:
                    continue
                for bits in BITS:
                    backup = quantize_block(model, j, bits)
                    for qi in qrows_j:
                        r = qrows[qi]
                        it = {"question": r["question"],
                              "prompt": model.get_prompt(r["question"])}
                        pred_q = model.question_answering(
                            it, max_new_tokens=64,
                            retrieved_indices=[base[qi]["sel"]])
                        f1_q = tok_f1(pred_q, r["answer"])
                        fo.write(json.dumps({
                            "video_id": vid, "question": r["question"],
                            "q_idx": qi, "block": int(j), "quintile": q_,
                            "freq": int(freq[j]), "bits": bits,
                            "f1_gold_base": round(base[qi]["f1_gold"], 4),
                            "f1_gold_q": round(f1_q, 4),
                            "drop": round(base[qi]["f1_gold"] - f1_q, 4),
                            "out_f1": round(tok_f1(pred_q, base[qi]["pred"]), 4),
                        }) + "\n")
                        fo.flush()
                        n_ev += 1
                    restore_block(model, j, backup)
                # keep GPU KV cache bounded across blocks
            print(f"[{vi}] {n_ev} quantization evals", flush=True)

    print("W22 QUANT SHARD DONE", flush=True)


if __name__ == "__main__":
    main()
