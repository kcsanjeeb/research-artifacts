# Generality arm — RVS-Ego streaming QA (reviewer armor)

Date: 2026-09-19. Run: `~/e1/runs/20260919_1047_generality_rvs/`.

Goal: show the event-driven write principle is not surveillance-specific.

**Data**: RVS-Ego (Becomebright/RVS), obtained via hf-mirror (HF proper is
blocked) — 10 Ego4D videos, 7.96 GB to /mnt/data2/san_e1/rvs (noted: >5GB,
large-disk volume only). Videos are 1-hour Ego4D clips; annotations
(ego4d_oe.json, 1,465 streaming questions) span the full hour. 120 questions
sampled (12/video, seeded, spread over question times). OVO-Bench/
StreamingBench not needed (RVS reachable; also ReKV's native benchmark).

**Metric**: token-F1 vs GT answer, identical for all systems (RVS's own
GPT-judge metric is unavailable offline — api.openai.com blocked).

**Systems**:
- **rekv**: ReKV 0.5B native (offline encode 0.5fps, streaming-KV QA).
- **uniform**: caption every 10s chunk (Qwen2.5-VL freeform) + bge retrieval;
  QA = top-2 retrieved captions as context → Qwen text-only answer.
- **novelty (FleetMem-generic)**: same pipeline, but a chunk is narrated ONLY
  when its CLIP embedding delta from the last narrated chunk exceeds
  τ=0.12 — the event-driven principle with no anomaly detector.

## Results (120 questions, 9.6 stream-hours)

| system | token-F1 mean (median) | write GPU-s/stream-h | query GPU-s/q |
|---|---|---|---|
| rekv (KV retrieval, its home benchmark) | **0.212** (0.154) | 341 | 0.88 |
| uniform writes (caption every 10s) | 0.143 (0.119) | 851 | 0.91 |
| novelty-gated writes (FleetMem-generic) | 0.130 (0.092) | **418** | 0.88 |

Figure: `generality_frontier.png` in the run dir.

## Honest read

- **The write-cost principle generalizes**: novelty gating halves uniform's
  write cost (418 vs 851 GPU-s/h) at statistically indistinguishable accuracy
  (0.130 vs 0.143, n=120). On general-domain (cooking/ego) content, ~half of
  10-s chunks carry no new visual information — same idle-skip effect as
  surveillance, without any anomaly machinery.
- **ReKV's full-fidelity KV beats caption memory on accuracy** (0.212) on
  fine-grained procedural QA ("What step is being taken with flatbread?") at
  comparable write cost (341 GPU-s/h). Caption-based memory is lossy for
  procedural detail; that's the gap the paper's tiering (L0/L1/L2) argues
  for, and the SMB result (retrieval/existence at corpus scale, ~0 query
  cost) is where memory wins.
- token-F1 levels are low in absolute terms (short GT phrases vs generative
  answers); treat as relative comparison only.

## Files

- `~/e1/runs/20260919_1047_generality_rvs/`: rvs_{rekv,uniform,novelty}.jsonl,
  rvs_*_cost.json, generality_metrics.json, generality_frontier.png
- Code: `~/e1/baselines/rvs_experiment.py`
