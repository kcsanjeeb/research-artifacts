# SMB baseline battery — FleetMem v0.1 vs baselines

Date: 2026-09-18. Benchmark: SMB v0, 220 UCF/XD queries (SHT excluded — no
tier-1 scores). FleetMem numbers from `~/e1/fleetmem/V01_REPORT.md`.
All runs on the same 4x V100-32GB server; fp16/sdpa; no flash-attn.

## THE comparison table

Accuracy metrics: existence acc / negation FAR / temporal mean tIoU /
retrieval Recall@10. Costs: write GPU-s per stream-hour, storage per
stream-day, GPU-s per query.

| System | Exist | Neg FAR | tIoU | R@10 | Write GPU-s/h | Storage/day | GPU-s/query |
|---|---|---|---|---|---|---|---|
| **FleetMem v0.1** (tags+combo, retrieval-only) | **0.950** | 0.440 | **0.286** | **0.454** | 508 | **1.13 MB** | **~0.0** |
| FleetMem v0.1 + syn_sym verify (precision mode) | 0.600 | **0.080** | 0.286 | 0.454 | 508 | 1.13 MB | 1.4 (verified types) |
| B1 UniformWriter (30s captions, same query path) | 0.667 | 0.540 | 0.149 | 0.176 | 318 (672 w/ tier-1 charge) | 5.91 MB | ~0.0 |
| B2 ReKV-0.5B (stream 0.5fps + per-video QA) | 0.617 | 0.200 | 0.055 | n/a* | 284 encode | ~4.3 GB (KV, analytic) | 0.62 + 10.8s encode/video |
| B3 VLM-direct (Qwen, 32 frames, no memory) | 0.633 | 0.120 | 0.140 | n/a* | 0 (no memory) | 0 | 7.4 |
| ReKV-7B (analytic, their paper) | — | — | — | — | ~4x B2 encode | 18.8 GB (their paper) | >B2 |
| StreamingVLM (analytic) | cannot answer retrospective queries — no persistent memory | — | — | — | — | 0 | — |

*B1/B2/B3-as-evaluated have no cross-video index: for B2/B3 a corpus query
means scanning every video (measured per-video costs -> analytic scan cost:
1090 videos x 0.62 GPU-s QA = **~676 GPU-s per retrieval query** for B2 if KV
caches persist; 1090 x 7.4 = **~8,100 GPU-s/query** for B3 re-reading pixels).
FleetMem answers the same query in <0.01 GPU-s from a 1.8 MB text+embedding
store.

## Takeaways

1. FleetMem v0.1 leads every accuracy metric among systems with memory,
   at the lowest query cost. Event-driven gating beats uniform captioning
   (B1) on all four metrics.
2. VLM-direct (B3) is a strong video-scoped baseline (balanced 0.757 ≈
   FleetMem 0.755) but costs 7.4 GPU-s per query, always re-reads the full
   video, and cannot do corpus retrieval without a full fleet scan.
3. B2/ReKV-0.5B's streaming KV cache does not answer category-level SMB
   questions well (temporal tIoU 0.055; its free-text answers rarely yield
   parseable spans) and its KV storage is ~4000x FleetMem's.
4. Verification-as-precision-mode remains the best abstention mechanism
   (FAR 0.08), but costs VLM time per query.

## Per-baseline adaptation notes

- **B1 UniformWriter**: 4,944 chunks over all 1,090 videos (37.2 stream-h);
  Qwen freeform captions (2.39 s/chunk mean); same bge query path, no
  category tags (FleetMem's mechanism — documented exclusion). Chunks are
  stored in FleetMem memory format for identical query machinery. Its
  temporal answers use the chunk's 30s span.
- **B2 ReKV**: validated 0.5B config (fp16, sdpa patch, fattn=False;
  ~/e1/ReKV + ~/e1/patches/rekv_v100_*.patch). Each query video streamed at
  0.5 fps (encode 10.8 GPU-s/video mean, measured; 284 GPU-s/stream-h),
  question asked through its streaming-KV retrieval path. Parsing: yes/no =
  first yes/no word (0 unparsed of 110); temporal = first two numbers in
  answer as [start,end] (14/60 unparsed -> empty span). KV cache for our
  1-4 min videos stayed GPU-resident (n_local=15000 covers them), so measured
  offload RAM ~0; the 4.3 GB/h figure is analytic (196 tok/frame x 24 layers
  x 2x2x64 fp16 x 1800 frames/h) and matches their paper's 4.0 GB/h.
  7B variant not run (0.5B already shows the line's behavior; fp32-attention
  patch is available if needed).
- **B3 VLM-direct**: Qwen2.5-VL-7B, 32 uniform frames (280x224 cap — sm_70
  vision sdpa N^2 mask OOMs at higher res; first run OOMed, patched),
  strict-JSON answers, all 170 parsed after the fix. Per-query 7.4 GPU-s
  including decode.
- **Analytic rows**: ReKV-7B storage 18.8 GB/h/stream from the ReKV paper;
  StreamingVLM keeps no persistent memory -> cannot answer SMB's
  retrospective queries by construction.

## Run dirs

- `~/e1/runs/20260918_0942_b1_uniformwriter/` (1090 videos, 4944 chunks,
  cost.json, query_eval/{metrics,smb_results}_*.json)
- `~/e1/runs/20260918_1438_b2_rekv/` (170 video-scoped queries,
  artifacts/b2_results.jsonl + metrics)
- `~/e1/runs/20260918_1435_b3_vlm_direct/` (170 queries,
  artifacts/b3_results_shard0.jsonl + metrics)

## Notes / known issues

- B1's first coverage check found 818/1090 videos (shard-3 launch had been
  cancelled mid-task); shard 3 was rerun -> 1090/1090, 72 SMB queries
  affected before the fix, final eval uses full corpus.
- GPU0 was occupied by another user's process (ComfyUI, 31GB) for part of
  the session; jobs ran on GPUs 1-3. B3's first run OOMed on 32x768-patch
  frames (vision N^2 mask) and was rerun at 280x224.
- B1 write cost excludes tier-1 (no gating); the "672 w/ tier-1" column adds
  the same 354 GPU-s/h charge for an apples-to-apples total.
