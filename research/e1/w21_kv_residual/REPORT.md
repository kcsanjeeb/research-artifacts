# W2.1 — Is video KV differentially compressible? (THE GATE)

**Date:** 2026-09-26 · **Server:** ubuntu-SYS-420GP-TNR, RTX 6000 Ada ×8 (used 0–2) · **Run dir:** `/data3/zhuotaotian2_e2/runs/20260926_1430_w21_encode/`

## Setup (measurement only; nothing built)

KV was not persisted on the old server, so all 9 videos were **re-encoded** on Ada (sanctioned by work2.md). Path mirrors ReKV's `llava_onevision_rekv.py` exactly: same init prompt (13 tokens), same video-feature pipeline (SigLIP vision tower → `[:,1:]` select → projector → `apply_pooling` → **196 tokens/frame**), chunked growing-cache prefill, `sample_fps=0.5` (UCF: decord stride-60 of 30fps sources; RVS-Ego: ffmpeg `fps=0.5` frame extraction — decord cannot decode VP9; uniform ~0.5fps sampling either way). Per-frame per-layer KV saved for all 24 layers: pre-RoPE K (k_proj forward hook), post-RoPE K and V (cache slice). 33 GB total.

**Numerical-equivalence note (Ada vs V100):** encoded in fp16 with SDPA attention — identical dtype/attention config to ReKV's validated setup (`torch_dtype=float16`, `fattn=False` → torch kernel). FlashAttention-2/bf16 were available but deliberately not used, to stay bit-compatible with the validated path. External consistency check: our measured raw KV rate for the 0.5B at 0.5 fps is **4.34 GB/h**, matching ReKV's published ≈4.0 GB/h for this model.

## Trap checks (mandatory)

**(a) RoPE rotation — validated, and it matters.** Applying the model's *own* `rotary_emb` to the captured pre-RoPE K reproduces the cached post-RoPE K with relative error **0.000000** at layers 0, 5, 11, 17, 23 (probe run) — the pre-RoPE capture and the position layout (frame f at positions 13+196f …) are exactly right. Consequence, measured on static surveillance:

| layer | ‖ΔK_post‖/‖K‖ (rotation-contaminated) | ‖ΔK_pre‖/‖K‖ (clean) | inflation |
|---|---|---|---|
| 0 | 0.334 | 0.0046 | **73×** |
| 12 | 0.836 | 0.266 | 3.1× |
| 23 | 0.098 | 0.044 | 2.2× |

All headline numbers below use pre-RoPE K. V carries no RoPE and is reported separately as the control. The large K/V discrepancies (e.g. static L0: V residual 0.39 vs K_pre 0.0046, 86×) are **not** a rotation artifact (K is pre-RoPE) — they are real: layer-0 keys are nearly frozen across frames while values are not.

**(b) Token alignment — stable layout, no finding.** Every frame of every video yields exactly 196 tokens after pooling (729 raw vision tokens → 2×2 spatial pool → 14×14 grid, asserted at encode); no per-frame merging/pruning exists in this path. Spatial ordering is fixed (row-major patch grid). Diagonal-dominance check on V (token i of frame t vs tokens of frame t−1): aligned cosine 0.86–0.92 vs off-diagonal 0.22–0.47 on surveillance (dominance ratio 2.1–3.9); ego lower (1.3–1.5) as expected with genuine ego motion. Differencing token-by-token is sound.

## Headline numbers (mean over 3 videos per regime; per-layer detail in `analysis_raw_*.json`)

Adjacent-frame residual energy ratio ‖X_t − X_{t−1}‖_F / ‖X_t‖_F:

| regime | K (pre-RoPE), layer mean | K best layer | K worst layer | V, layer mean | V range |
|---|---|---|---|---|---|
| static (UCF normal) | **0.249** | 0.005 (L0) | 0.458 (L9) | **0.537** | 0.34–0.64 |
| dynamic (UCF anomaly) | **0.284** | 0.006 (L0) | 0.531 (L9) | **0.619** | 0.41–0.73 |
| ego (RVS-Ego) | **0.423** | 0.010 (L0) | 0.736 | **0.944** | 0.86–0.99 |

Per-layer K curves: L0–L1 ≈ 0.005–0.02 (static & dynamic alike), L2 ≈ 0.09–0.11, then 0.15–0.53 through L22 (peak mid-layers), L23 drops to 0.04–0.06. Consecutive-block cosine mirrors this (L0–L1 ≈ 1.000, mid-layers 0.85–0.95 static, 0.72–0.92 ego). **The static/dynamic contrast exists only in layers 0–2; everywhere else static scenes are as non-smooth as dynamic ones.**

Quantized entropy (bits/symbol; per-frame-per-channel min-max uniform quantizer):

| quantity | 2-bit | 4-bit raw | 4-bit residual | 8-bit residual |
|---|---|---|---|---|
| K, static | 1.19 | 3.65–3.98 | **3.01** | 7.05 |
| K, ego | 1.27 | 3.74–3.87 | **3.30** | 7.35 |
| V, static | — | 3.35–3.59 | **3.01** | — |
| V, ego | — | 3.43–3.63 | **3.30** | — |

## Achievable compression at fixed reconstruction error

Scheme: keyframe at 8 bits/elem, other frames as residuals at b bits/elem, reconstruction error MSE-matched to 4-bit raw quantization of the block (per-frame-per-channel quantizer, idealized residual chain using true previous frame).

- Minimal b (K): L0–L1 = **2 bits** (2× gain), L23 = 3 bits, L2–L22 = 4 bits (**no gain**). Minimal b (V): ≥4 bits everywhere (**no gain**).
- Estimated storage (keyframe 8-bit + residual at mean b_min): **static 1.62 GB/h · dynamic 1.65 GB/h · ego 1.72 GB/h**.
- Reference points: ReKV raw fp16 = 4.34 GB/h; MuKV = 0.91 GB/h; gate threshold = 0.91/3 = **0.30 GB/h**.
- → compression vs MuKV = **0.56× / 0.55× / 0.53×** — i.e. ~2–6× *worse* than MuKV, and only ~1.0× better than the trivial baseline of 4-bit quantizing every frame independently.

## Gate verdict: **FAIL**

- Criterion 1 (static residual < 0.3): passes only for K in layers 0–2 (0.005–0.09) and weakly layer 23 (0.04–0.06). Fails for V at every layer (0.34–0.64 static) and for mid-layer K (0.15–0.53).
- Criterion 2 (≥3× over MuKV's 0.91 GB/h at equal reconstruction error): **0.56×** — off by a factor of ~5. FAIL decisively.

**The mechanism is dead as proposed: video KV is not differentially compressible in any layer that holds most of the bytes.** Even when pixels are static, mid/late-layer K and all-layer V carry 30–100% residual energy per frame. Root cause, stated plainly: stored KV is a function of the *entire prefix*, not of frame content alone — RoPE-position-dependent attention makes the hidden states (hence V, and mid-layer K) drift as the sequence grows, even under pixel-identical input. Differential coding against the previous frame encodes that drift at ≈ the cost of encoding the block.

The one real effect found (reported for honesty, not as a rescue): **input-layer keys are genuinely frozen** (residual 0.005–0.02 across all regimes, 2-bit sufficient, entropy 1.2 bits). That is a property of early vision features, not of temporal redundancy, and covers a negligible fraction of storage.

**Consequence per work2.md:** the inside-the-model differential-coding direction closes; return to the systems/fleet paper. (W2.2/W2.4 test retrieval-side claims and are unaffected by this gate's logic; W2.3 below also fails.)

Artifacts: `residual_energy_by_layer.png` · `compression_ratio_by_layer.png` · `compression_estimate.json` · `analysis_raw_{static,dynamic,ego}.json` · `trap_checks.json`. KV dumps: `/data3/zhuotaotian2_e2/runs/20260926_1430_w21_encode/kv/` (33 GB, keep for W2.2/W2.4).
