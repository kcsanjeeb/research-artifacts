# T3 — M1 sketch surrogate validity (decides P2-V3)

Date: 2026-09-25. Output dir: `~/e1/t3_sketch_validity/` (code:
`~/e1/t3_sketch_validity.py`). No GPU.

**Premise tested.** P2-V3 gates writes on marginal gain against memory using
a per-camera product-quantized count sketch over tier-1 pooled CLIP features
(+ time-of-day bucket). This only works if the sketch count predicts actual
redundancy.

## Setup

- 749 events from `runs/20260918_0111_writepath_fullcorpus`, narrations =
  L1.summary, narration embeddings = existing bge-small CLS vectors (384-d).
- **Redundancy label** (per work1.md): max embedding cosine of an event's
  narration to *prior same-camera* narrations; secondary label = tag-match
  rate vs priors. Camera = one video/stream; strictly per-camera, never
  cross-camera.
- **Sketch count**: pooled tier-1 CLIP feature per event (mean of 16-frame
  snippet features over the event window; UCF = 10-crop features @30fps,
  XD = snippet features @24fps, from `~/FleetVAD/research/e0/data/`),
  PQ-encoded (M=8 subspaces × 64 dims, k=64 centroids, KMeans seed 0),
  counted against prior same-camera events. Time-of-day bucket =
  start_sec // 300 s (no wall-clock timestamps exist in UCF/XD; documented
  proxy).
- **Eligible events: only 116 of 749** have ≥1 prior same-camera event —
  the corpus is 559/633 cameras with exactly 1 event. The correlation is
  computed on those 116; this sparsity is itself a finding (a marginal-gain
  gate would read count=0 for 85% of writes on this corpus).

## Measured correlations (Spearman ρ vs redundancy, n=116)

| predictor | ρ vs narration-cosine label | p |
|---|---|---|
| PQ sketch count, strict full-code | +0.306 | 8.2e-04 |
| PQ sketch count, strict + time-of-day | +0.316 | 5.5e-04 |
| PQ sketch count, subspace-mean | **+0.383** | 2.3e-05 |
| oracle: raw pooled-CLIP cosine (diagnostic) | +0.416 | 3.3e-06 |
| **tag fallback:** tag-any-match | −0.019 | 0.84 |
| **tag fallback:** tag-match-rate | −0.081 | 0.39 |

(The tag predictors' ρ against the *tag-rate label* — 0.86 / 1.00 — is
tautological, not evidence; the redundancy label that matters is narration
similarity.)

Scatter: `sketch_vs_redundancy.png`. Per-event data:
`artifacts/per_event.jsonl`.

## Verdict: **FAIL — and the fallback fails too.**

1. **PQ sketch: FAIL.** Best variant ρ = 0.383 < 0.4 gate. The signal is
   real (p = 2e-5) but sub-threshold, and quantization is not the main loss:
   the unquantized oracle only reaches ρ = 0.416 — pooled CLIP appearance
   similarity itself is only marginally predictive of narration redundancy
   on this corpus.
2. **Tag fallback (P2-V7): FAIL.** Closed-taxonomy tags carry no signal
   against the narration-redundancy label (ρ ≈ 0).
3. Per the task spec, with both surrogates failing, **the marginal-gain
   mechanism (M1 / P2-V3 / P2-V7) is dead on this corpus** — said plainly.

Caveats worth recording: (a) n=116 is small and the corpus has almost no
per-camera repetition (mean 1.18 events/camera) — the mechanism may be
irrelevant here rather than impossible in a true long-running per-camera
fleet regime; (b) the time-of-day bucket is a within-video proxy, not real
wall-clock time.
