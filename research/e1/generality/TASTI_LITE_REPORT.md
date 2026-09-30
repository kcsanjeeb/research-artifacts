# TASTI-lite baseline on SMB (claim C4)

Date: 2026-09-19. Run: `~/e1/runs/20260919_1028_c4_tasti_lite/`.

## Design

Policy-level reimplementation of TASTI (SIGMOD'22): a pure embedding index —
embed every 30-s chunk of the corpus (4,944 chunks over 1,090 UCF/XD videos,
identical chunking to B1 UniformWriter), CLIP ViT-B/16 image embedding of the
chunk's center frame, no narration, no text. Queries: CLIP text encoder on the
question (77-token truncation noted), cosine over chunk embeddings, τ
calibrated on the same frozen 20% existence+negation split (seed 0),
identical metrics code (`querypath.evaluate`) as FleetMem/B1.

Not their code: TASTI's repo is CUDA-10.1-era and its night-street data is
GDrive-blocked (documented in E1_BASELINE_AUDIT / week1 reports). The
reimplementation is the policy, not their engine.

## Results (220 UCF/XD SMB queries)

| System | Existence | Negation FAR | Balanced | Temporal tIoU | Retrieval R@10 | AP |
|---|---|---|---|---|---|---|
| **FleetMem v0.1** (text+tag memory) | **0.950** | 0.440 | **0.755** | **0.286** | **0.454** | **0.353** |
| TASTI-lite (CLIP appearance index) | 0.583 | 0.200 | 0.692 | 0.167 | 0.269 | 0.157 |
| B1 UniformWriter (VLM captions) | 0.667 | 0.540 | 0.564 | 0.149 | 0.176 | 0.093 |

τ for TASTI-lite calibrated to 0.20-0.35 band on the frozen split.

## Does the E0 appearance-signature finding hold in-system? YES.

Appearance embeddings without language fail exactly where predicted:
- **Retrieval**: R@10 0.269 vs FleetMem 0.454 — "find all videos with a
  robbery" needs category semantics that appearance embeddings lack; CLIP
  text-image alignment recovers part of it but stays far below text memory
  with tags.
- **Existence**: 0.583 vs 0.950 — chunk appearance alone can't confirm
  category-level questions.
- **Temporal**: 0.167 vs 0.286 — the best-matching chunk is often the wrong
  part of the event.
- Negation FAR 0.20 is its best number (fewer spurious matches than VLM
  captions, which over-trigger) — but that trades against existence.

FleetMem's text+tag memory beats the pure embedding index on every metric
while writing only 749 narrations vs 4,944 chunk embeddings (6.6× fewer
writes; write cost 508 vs ~50 GPU-s/stream-h for CLIP embedding — TASTI-lite
is cheaper to build but much less answerable).

## Cost

CLIP embedding: wall ~35 min on one V100 for 4,944 chunks (~0.4 s/chunk
incl. decode), ≈ 50 GPU-s/stream-hour (no narration, no tier-1 charge;
center-frame decode 73 CPU-s/stream-h equivalent).

## Files

- Code: `~/e1/baselines/tasti_lite.py`
- Run: `~/e1/runs/20260919_1028_c4_tasti_lite/` — `artifacts/chunk_embs.npy`
  (4944×512), `chunk_meta.json`, `smb_results.jsonl`, `metrics.jsonl`,
  `embed_cost.json`.
