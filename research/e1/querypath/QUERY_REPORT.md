# FleetMem query path v0 — first SMB evaluation

Date: 2026-09-18. Runs: ingestion `~/e1/runs/20260918_0111_writepath_fullcorpus/`,
query eval in `<run>/query_eval/`. Benchmark: SMB v0 (`~/e1/smb/`), UCF+XD
queries only (220 of 300; SHT excluded — no tier-1 scores, documented SMB v0
limitation).

## Ingestion scale (Step 1)

Full UCF+XD test corpus: **1,090 videos, 37.2 stream-hours**, 4 GPU shards,
~45 min wall. **749 union-gate events narrated** (5,734 GPU-s narration total
≈ **1.6 GPU-hours**, well under the 3 GPU-h cap). Write cost: 508
GPU-s/stream-hour total (354 tier-1 fixed + 154 narration + 0.16 embed);
10.4 CPU-s/h decode; storage 1.13 MB/stream-day. Normal videos correctly yield
empty memory. Memory coverage: 749 events over 37.2 h ≈ 20 events/stream-hour.

## Query path

Retrieval: bge-small-en-v1.5 query embedding vs two per-event channels —
summary embedding (from write path) + full narration-text embedding (added
after v0a showed summary-only is too sparse; score = max of channels).
Answering modes: (a) retrieval-only with threshold τ=0.48 calibrated on a
frozen 20% held-out split of existence+negation queries; (b) + Qwen2.5-VL
verification of top-3 candidates (8 frames, temp 0.1, strict JSON match).

## Results (SMB v0, UCF+XD, n=220)

| Type | Mode a (retrieval-only) | Mode b (+ VLM verify) | Δ |
|---|---|---|---|
| Existence accuracy (n=60) | **0.950** | 0.500 | −0.45 |
| Negation accuracy (n=50) | 0.560 | **0.900** | +0.34 |
| Negation false-alarm rate | 0.440 | **0.100** | −0.34 |
| Balanced exist+neg acc | **0.755** | 0.700 | −0.055 |
| Temporal mean tIoU (n=60) | 0.201 | 0.201 | 0 |
| Temporal tIoU≥0.3 | 0.217 | 0.217 | 0 |
| Retrieval Recall@10 (n=50) | 0.254 | 0.254 | 0 |
| Retrieval AP | 0.141 | 0.141 | 0 |
| Extra cost | 0 GPU-s | 435.7 GPU-s total (≈2.0 GPU-s/query over verified types) | |

Retrieval and temporal are identical across modes (verification only applied
to existence/negation/temporal top-3; temporal's verified pick never differed
from top-1 here).

## Verification value-for-cost verdict

**Verification is worth it for abstention, not (yet) for confirmation.**
It cuts negation false alarms 4.4× (0.44→0.10) for ~2 GPU-s/query, but its
strict category-level check over 8 sparse frames false-rejects true events:
27 existence queries flipped right→wrong, and in 26/27 the write path HAD
covered the GT event — the retrieved event was rejected because its narration
is generic ("a man in a blue shirt walks away from a metal gate" for a GT
Abuse span) and the frames don't literally show the category word. A hybrid
(veto only on confident mismatch, or verify-with-category-synonyms prompt) is
the obvious fix; v0 reports both modes raw.

## Failure analysis (narrations inspected)

1. **Verification false-rejects (26/27 existence mode-b losses)**: write-path
   covered, retrieval delivered the right event, verifier rejected because
   category-level semantics ("abuse", "shooting") aren't visually explicit in
   8 frames of a long coarse event.
2. **Temporal span dilution (31/47 temporal failures)**: formed events swallow
   the whole tier-1 plateau (pred [0,45.9s] vs GT [5.3,8.5s]) — recall is
   fine but tIoU is low. Event granularity, not retrieval, is the bottleneck.
   Needs sub-event localization (e.g., per-snippet peak refinement).
3. **Write-path misses (16/47 temporal failures)**: GT event never crossed the
   loose gate → nothing in memory to find.
4. **Retrieval semantic gap**: narrations describe visible content, not GT
   category names (groundedness: category consistency 40-62%), so
   "find all robberies" relies on synonym proximity in embedding space —
   Recall@10 0.254 reflects this.
5. **Negation residual (5 false alarms in mode b)**: XD multi-label normal-ish
   clips where generic motion ("fights?") got a yes from the verifier on
   ambiguous frames.

## Honest limitations

- SMB v0 is query-after-ingest; no streaming queries. τ calibrated on 22
  held-out queries — small.
- Retrieval embeddings use text only; no visual re-ranking in v0.
- Temporal metric penalizes long formed events; an "event contains GT"
  (point-wise recall) variant would look much better (see write-path frontier:
  0.95-0.97 recall by any-overlap).
- SHT excluded; corpus = UCF+XD test splits only.

## Files

- `query_eval/smb_results_{retr,vlmverify}.jsonl`, `metrics_{retr,vlmverify}.json`,
  `threshold.json`, `text_embs.npy`, `modeb.out` under the run dir.
- Code: `~/e1/fleetmem/querypath.py`, `add_text_embs.py`, `analyze_fails.py`.
