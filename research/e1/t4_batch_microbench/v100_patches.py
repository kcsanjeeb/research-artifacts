"""Patches for Qwen2.5-VL on sm_70 (V100) to make batching feasible.

1. Vision SDPA attention: transformers 4.49 builds a dense [1,N,N] bool mask
   over the concatenated image batch (N = total patches of ALL images in the
   batch). With any attn_mask on sm_70, torch sdpa falls back to the math
   backend and materializes heads*N^2 scores -> OOM at batch>=2 (16 img x 768
   patches: one 9 GiB alloc). The mask is block-diagonal over cu_seqlens
   segments (per image for full-attn layers, per 64-token window otherwise),
   so per-segment sdpa without a mask is mathematically identical and O(N).

2. lm_head last-token-only: the model forward computes full-sequence fp32
   logits (bs*seq*152k*4 bytes; 2 GiB at bs=2 prefill, ~24 GiB at bs=16).
   generate() only ever uses logits[:, -1, :]. Slicing before lm_head keeps
   generation identical and drops this to O(bs).

Both are memory-only fixes; the math is unchanged. Documented in run config.
"""
from collections import defaultdict

import torch
import torch.nn.functional as F


def patched_vision_sdpa_forward(self, hidden_states, cu_seqlens,
                                rotary_pos_emb=None, position_embeddings=None):
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        apply_rotary_pos_emb_vision,
    )
    seq_length = hidden_states.shape[0]
    q, k, v = self.qkv(hidden_states).reshape(
        seq_length, 3, self.num_heads, -1).permute(1, 0, 2, 3).unbind(0)
    if position_embeddings is None:
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        cos, sin = emb.cos(), emb.sin()
    else:
        cos, sin = position_embeddings
    q, k = apply_rotary_pos_emb_vision(q, k, cos, sin)
    q = q.transpose(0, 1)  # [H, N, D]
    k = k.transpose(0, 1)
    v = v.transpose(0, 1)
    cu = cu_seqlens.tolist()
    segs = [(cu[i - 1], cu[i]) for i in range(1, len(cu))]
    groups = defaultdict(list)
    for idx, (a, b) in enumerate(segs):
        groups[b - a].append(idx)
    out = torch.empty_like(q)
    for L, idxs in groups.items():
        pos = torch.cat([torch.arange(segs[i][0], segs[i][1], device=q.device)
                         for i in idxs])
        n = len(idxs)
        qa = q.index_select(1, pos).reshape(self.num_heads, n, L, -1)
        ka = k.index_select(1, pos).reshape(self.num_heads, n, L, -1)
        va = v.index_select(1, pos).reshape(self.num_heads, n, L, -1)
        o = F.scaled_dot_product_attention(qa, ka, va, dropout_p=0.0)
        out.index_copy_(1, pos, o.reshape(self.num_heads, n * L, -1))
    attn_output = out.transpose(0, 1).reshape(seq_length, -1)
    return self.proj(attn_output)


def apply_patches(model):
    from transformers.models.qwen2_5_vl import modeling_qwen2_5_vl as m
    m.Qwen2_5_VLVisionSdpaAttention.forward = patched_vision_sdpa_forward

    orig_lm_head_forward = model.lm_head.forward
    def lm_head_last_token(x):
        return orig_lm_head_forward(x[:, -1:, :])
    model.lm_head.forward = lm_head_last_token
    return model
