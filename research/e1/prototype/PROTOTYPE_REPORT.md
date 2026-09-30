# FleetMem write-path prototype v0 — report

Date: 2026-09-18. Run dirs: `~/e1/runs/20260918_0047_writepath_pilot/`,
`~/e1/runs/20260918_0048_writepath_full/`. Code: `~/e1/fleetmem/`.

## Pipeline

Causal single-stream replay of 50 test videos (30 UCF + 20 XD; 33 anomalous /
17 normal; 1.82 stream-hours). Per stream: simulated-live tier-1 (E0's causal
VadCLIP snippet scores, charged at the E0.8-measured 354 GPU-s/stream-hour),
hysteresis event formation (3 gates), union-narrated Qwen2.5-VL-7B schema
narrations (8 frames, 448x336, fp16/sdpa), bge-small-en-v1.5 summary
embeddings, per-stream memory.jsonl + embedding npy.

## Per-tier costs (full run, per stream-hour)

| Tier | Cost | Notes |
|---|---|---|
| Tier-1 salience | 354 GPU-s/h (fixed charge) | E0.8 stride-1 measurement; 89.8 at stride 4 |
| Decode (frame extraction for events) | 10.4 CPU-s/h | 8 frames per event, decord |
| Narration (Qwen2.5-VL-7B) | 175.7 GPU-s/h | 40 events, mean 8.0s/event, peak 21.6GB |
| Embedding (bge-small) | 0.23 GPU-s/h | negligible |
| **Total (union/loose)** | **~530 GPU-s/h** | ≈ 15% of one V100 per always-on stream |

Storage: ~1.2 MB per stream-day (memory JSONL + embeddings) for the union
narrations on this data mix.

## Frontier (event recall vs write cost)

GT = SMB inventory spans (59 GT events on subset videos). Recall = GT event
overlapped by ≥1 formed event passing the gate.

| gate | events narrated | recall | GPU-s/stream-h | KB/stream-day |
|---|---|---|---|---|
| loose (0.50/0.20) | 40 | 0.966 (57/59) | 529.7 | 1257 |
| calibrated (0.65/0.35) | 37 | 0.949 | 516.4 | 1163 |
| strict (0.80/0.50) | 35 | 0.949 | 509.2 | 1100 |
| time-driven 30s chunks (analytic) | 244 | 1.000 (by construction) | 1441.6 | ~7671 |

Figure: `artifacts/frontier.png` in the full run dir.

**Headline:** gated event-driven writing matches time-driven recall within
3-5 points at **2.7x lower GPU cost** (510-530 vs 1442 GPU-s/stream-hour).
The gate sweep barely separates on this subset: VadCLIP scores are bimodal
(anomaly videos score high throughout; normal videos low), so 35/40 union
events pass even the strict gate. Misses: 2 GT events never crossed 0.50
(weak-signal events — consistent with the narration spike's borderline
findings).

## Honest limitations

- Tier-1 is simulated-live: E0 scores were computed per-snippet causally but
  offline; the 354 GPU-s/h charge is a stride-1 measurement, not re-measured
  in this pipeline.
- 50 videos / 1.82 stream-hours; normal-video false-positive events exist
  (e.g., Normal_Videos_887 produced 2 strict-gate events) but are rare.
- Recall uses any-overlap with GT spans; no tIoU threshold (spans are long
  and hysteresis events tend to cover them whole or merge adjacent GT
  fragments — union events 40 < GT 59 partly due to merging).
- Time-driven reference is analytic (30s grid x measured 8.12s/narration),
  not executed; its recall is 1.0 by construction (coverage, not detection).
- Narration quality per se is covered by the groundedness analysis, not
  re-scored here.
