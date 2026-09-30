# T4 — Batch microbenchmark (Qwen2.5-VL-7B narration, Tesla V100)

Run dir: `~/e1/runs/20260926_1248_t4_batch_microbench/` — config.json, env.json,
code_state.md5, gpus.txt, metrics_gpu{1,2,3}_*.jsonl, cost.json, logs.
Date: 2026-09-26. GPUs 1–3 (exclusive), 3 parallel single-tenant processes.

## Setup

- Qwen2.5-VL-7B-Instruct, fp16, `attn_implementation="sdpa"`, transformers
  4.49.0, torch 2.6.0+cu124, HF offline cache.
- Job = schema narration from `~/e1/narration_spike/narrate.py` (same prompt,
  same sampling T=0.2 top_p=0.9, max_new_tokens=512), frames pre-resized to
  448×336 (multiples of 28), cached spike frames/events (200 UCF events).
- Fixed batch sizes {1,2,4,8,16}; steady state: 2 warmup batches (1 at b16),
  ≥30 jobs per point; GPU-s/job = wall time on an exclusive GPU (util ~97–100%
  during generation).
- **Two memory-only patches were required for any batching at all**
  (`v100_patches.py`, equivalence-verified at bs=1 — greedy outputs identical,
  max logit diff 0.05 = fp16 kernel-order noise):
  1. transformers 4.49 vision SDPA builds a dense `[1,N,N]` bool mask over the
     concatenated image batch (N = all images × 768 patches). With any mask on
     sm_70, torch sdpa falls back to the math backend and materializes
     heads×N² scores → a 9 GiB alloc at **bs=2** → OOM. Patched to per-segment
     sdpa over `cu_seqlens` (the mask is block-diagonal; identical math, O(N)).
  2. The model forward computes full-sequence fp32 logits (2 GiB at bs=2,
     ~24 GiB at bs=16) although `generate` only uses `logits[:, -1, :]`.
     Patched `lm_head` to last-token-only.
  Without these patches the stock spike stack OOMs at batch 2. This is itself a
  deployment finding: the deadline-aware batcher (if built) must include them
  (or a newer transformers) — stock transformers 4.49 cannot batch on V100.

## Headline: GPU-s/job (mean; p50/p95 in metrics)

| batch | 8 frames GPU-s/job | 8f speedup | 4 frames GPU-s/job | 4f speedup |
|---|---|---|---|---|
| 1 | 5.831 (p50 5.63, p95 7.93) | 1.00× | 5.037 (p50 4.93, p95 6.83) | 1.00× |
| 2 | 3.978 | 1.47× | 3.098 | 1.63× |
| 4 | 2.935 | 1.99× | 2.252 | 2.24× |
| 8 | **2.620** | **2.23×** | **1.880** | **2.68×** |
| 16 | 3.414 | 1.71× | 1.229 | 4.10× |

Throughput (jobs/min): 8f: 10.3 / 15.1 / 20.4 / 22.9 / 17.6 (b1..b16).
4f: 11.9 / 19.4 / 26.6 / 31.9 / 48.8.

## Peak VRAM (torch.cuda.max_memory_allocated)

| batch | 8f | 4f | mixed |
|---|---|---|---|
| 1 | 15.9 GiB | 15.7 GiB | – |
| 2 | 16.3 GiB | 15.9 GiB | – |
| 4 | 17.0 GiB | 16.3 GiB | – |
| 8 | 18.5 GiB | 17.1 GiB | 18.4 GiB |
| 16 | 21.4 GiB | 18.7 GiB | 21.3 GiB |

(Fits comfortably in 32 GiB *after* the patches; stock stack OOMs at bs≥2.)

## Batch-16 regression at 8 frames = straggler effect, not compute loss

Batched generate pads every row to the batch’s longest output. Mean gen
tokens/job: 8f b8=280, b16=512 (the cap). Per-token cost barely worsens
(GPU-s/1k padded tokens: 1.319 at b8 → 1.539 at b16, +17%) while tokens/job
nearly doubles → the b16 GPU-s/job regression is the max-of-16 output-length
straggler. 4f outputs are shorter (b16 mean 234 < 512 cap) so 4f keeps
improving at b16. Diagnostic at b16 confirmed clean single-EOS outputs.

## Mixed batches vs homogeneous (same total job count)

| point | GPU-s/job | GPU-s/1k padded tok |
|---|---|---|
| homogeneous 8f b8 / b16 | 2.620 / 3.414 | 1.319 / 1.539 |
| homogeneous 4f b8 / b16 | 1.880 / 1.229 | 1.569 / 1.056 |
| perfect bucketing (avg of the two) b8 / b16 | 2.250 / 2.322 | – |
| **mixed 4f+8f b8 / b16** | **2.088 / 1.877** | **1.088 / 0.962** |

Padding/bucketing waste is **negative**: mixed batches beat perfect bucketing
by 7.2% (b8) and 19.2% (b16) in GPU-s/job. Padded prefill of the short jobs is
cheap; strict bucketing loses more to the max-length straggler per batch. On
this hardware, mixing job types in one fixed-size batch is preferable to
length bucketing.

## Convexity (M3 “degradation as batching enabler”)

Throughput gain from degrading 8f→4f grows with batch size:
Δjobs/min = +1.6 (b1) → +9.0 (b8) → +31.2 (b16). Batch-16 is *unreachable* at
8 frames (regression) but is the *best* point at 4 frames. The joint
(batch × fidelity) surface shows a positive interaction consistent with the
convexity M3 needs, and nothing contradicts it. Caveat: only two fidelity
levels were measured, so strict convexity (curvature) cannot be established —
only that the interaction is positive and monotone over this range.

## Verdict

**PASS.** Batch-8 GPU-s/job improvement = **2.23×** (5.831 → 2.620) ≥ 2× gate,
even under the honest fixed-batch-size steady-state protocol with sampling.
→ Build the deadline-aware batcher (P4-c), with the two V100 memory patches
as a hard prerequisite.

Notes:
- Our b1 baseline is 5.83 GPU-s/job, not the 12.4 cited in planning docs
  (that figure likely included the freeform variant / frame extraction
  overhead; here only the schema variant is timed, generation only).
- The RTX 6000 Ada reassessment box is **not accessible** — could not repeat
  there. Expected batching gains there are likely *weaker* per job but with
  far more headroom (48 GB, flash-attn, bf16).
- The dominant limiter at scale on V100 is not compute but (a) the vision
  tower’s O(N²) dense mask (patched) and (b) decode straggler length at large
  fixed batches. A length-aware batch composer (cap max-output spread) would
  push 8f b16 above its b8 value.

## Cost

0.312 GPU-h total across 3 GPUs (1124 GPU-s of measured generation) + ~2 min
model load per process. See cost.json.
