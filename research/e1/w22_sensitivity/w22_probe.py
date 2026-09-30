#!/usr/bin/env python3
"""W2.2 probe: verify ReKV KV storage layout + quantize/restore round-trip.

1. Encode first 200 frames of video 1; print ContextManager layout stats.
2. Verify quantize/restore of one block's cpu_data changes output and restores.
"""
import json
import os
import sys

REKV = os.path.expanduser("~/e1/ReKV")
sys.path.insert(0, REKV)
sys.path.insert(0, os.path.expanduser("~/e1/baselines"))

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import torch
from decord import VideoReader, cpu
from model import llava_onevision_rekv
from rvs_experiment import load_questions


def quantize_tensor(t, bits, is_key):
    """Symmetric uniform quant-dequant. t: (H, T, D) fp16 cpu.
    K: per-(head, channel) scale over tokens; V: per-(head, token) scale over channels."""
    x = t.float()
    qmax = 2 ** (bits - 1) - 1
    if is_key:
        scale = x.abs().amax(dim=1, keepdim=True) / qmax  # (H,1,D)
    else:
        scale = x.abs().amax(dim=2, keepdim=True) / qmax  # (H,T,1)
    scale = scale.clamp_min(1e-8)
    q = torch.round(x / scale).clamp(-qmax - 1, qmax) * scale
    return q.to(t.dtype)


def quantize_block(model, j, bits):
    """Quantize block j in every layer's ContextManager. Returns backup."""
    backup = []
    with torch.inference_mode():
        for cm in model.kv_cache:
            mu = cm.global_blocks[0][j]
            if mu.gpu_data is not None:
                mu.offload()  # drop possibly-stale GPU copy; reload from CPU later
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
    qs = load_questions()
    vid = sorted({q["video_id"] for q in qs})[0]
    vq = [q for q in qs if q["video_id"] == vid]
    vp = vq[0]["video_path"]
    vr = VideoReader(vp, ctx=cpu(0), num_threads=1)
    fps = round(vr.get_avg_fps())
    nfr = len(vr)
    frame_idx = list(range(0, nfr, max(1, int(fps / 0.5))))[:200]
    video_t = torch.from_numpy(vr.get_batch(frame_idx).asnumpy())
    print(f"video {vid[:8]}: {nfr} frames, encoding {len(frame_idx)} blocks", flush=True)

    model, _proc = llava_onevision_rekv.load_model(
        model_path=os.path.join(REKV, "model_zoo/llava-onevision-qwen2-0.5b-ov-hf"),
        n_local=15000, topk=64, chunk_size=1)
    model.clear_cache()
    model.encode_init_prompt()
    model.encode_video(video_t)

    cm0 = model.kv_cache[0]
    print(f"n layers: {len(model.kv_cache)}")
    print(f"cm0: num_global_block={cm0.num_global_block}, "
          f"len(global_blocks[0])={len(cm0.global_blocks[0])}, "
          f"local_k={tuple(cm0.local_k.shape)}, "
          f"remainder={tuple(cm0.global_remainder[0].shape)}, "
          f"init_exc={cm0.init_exc}")
    mu = cm0.global_blocks[0][5]
    print(f"block5 cpu_data shapes: {[tuple(t.shape) for t in mu.cpu_data]}, "
          f"pinned={mu.cpu_data[0].is_pinned()}, gpu_data is None: {mu.gpu_data is None}")

    q = vq[0]["question"]
    input_text = {"question": q, "prompt": model.get_prompt(q)}
    pred_full = model.question_answering(input_text, max_new_tokens=64)
    print(f"internal retrieval, max block idx = {max(model.last_retrieved_blocks)}")
    print(f"full pred (internal): {pred_full!r}", flush=True)

    # external retrieval with logged-style indices (W2.2 baseline path)
    blocks = list(model.last_retrieved_blocks)
    pred_ext = model.question_answering(input_text, max_new_tokens=64,
                                        retrieved_indices=[blocks])
    print(f"external baseline pred: {pred_ext!r}")
    print(f"external == internal pred: {pred_ext == pred_full}")

    # quantize one mid block at 2 bits, check output changes
    j = blocks[len(blocks) // 2]
    backup = quantize_block(model, j, 2)
    pred_q = model.question_answering(input_text, max_new_tokens=64,
                                      retrieved_indices=[blocks])
    print(f"2-bit block {j}: pred changed vs ext baseline: {pred_q != pred_ext}; pred: {pred_q!r}", flush=True)
    restore_block(model, j, backup)
    pred_r = model.question_answering(input_text, max_new_tokens=64,
                                      retrieved_indices=[blocks])
    print(f"restored: pred == ext baseline: {pred_r == pred_ext}")

    # 8-bit sanity: should be (near-)identical
    backup = quantize_block(model, j, 8)
    pred_8 = model.question_answering(input_text, max_new_tokens=64,
                                      retrieved_indices=[blocks])
    print(f"8-bit block {j}: pred == ext baseline: {pred_8 == pred_ext}")
    restore_block(model, j, backup)
    print("PROBE DONE", flush=True)


if __name__ == "__main__":
    main()
