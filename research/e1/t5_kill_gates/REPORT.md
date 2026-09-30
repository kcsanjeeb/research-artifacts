# T5 — Cheap kill gates: REPORT

Date: 2026-09-25. Server: san@10.249.185.176, conda env `e1rekv` (numpy 2.2.6,
faiss 1.15.0 CPU). Scripts: `~/e1/t5_kill_gates/scripts/`. One run dir per
gate under `~/e1/runs/` (config/env/code_state/metrics + REPORT.md each).
Two early draft run dirs (20260925_1925_t5_1, 20260925_1939_t5_3) are kept
per the no-delete rule; they were superseded by the refined reruns listed
below — the numbers below come from the listed (latest) dirs only.

| Gate | Question | Measured | Verdict | Consequence |
|---|---|---|---|---|
| T5.1 | Do SHT events co-occur across cameras? | raw proxy 51.9% = null 50.5%; excess +1.4pt; strong (IoU≥0.5) 10.3% vs null 8.8%; no shared clock exists | **KILL** | cross-camera dedup **P4-f dies** |
| T5.2 | Does λ oscillate under dual ascent? | sustained limit cycle at all η ∈ {0.05..1.0}, both cells: p2p/mean 0.42–2.82, 47–74 reversals/100 windows | **PASS** | build hysteresis **M2** |
| T5.3 | Does flat index search break before 1M? | p99 @1M: 110.6 ms at 72 thr (oversubscribed), **36.5 ms at 32 thr**; batch-32: 2.79 ms/query, 358 QPS | **flat holds → KILL P5-f** | index re-architecture **P5-f dies** at ≤1M scale |
| T5.4 | Does fleet-quantile beat raw score on SHT? | input missing | **BLOCKED** | M7/P2-V2 undecidable; no SHT tier-1 scores exist and computing them is new work |

---

## T5.1 — SHT cross-camera co-occurrence

Run dir: `~/e1/runs/20260925_1929_t5_1_sht_cooccurrence/`

Input: `~/e1/smb/inventory_sht.jsonl` (194 GT events, 12 cameras),
`~/e1/smb/videometa_sht.jsonl`. SHT has **no shared clock across videos**
(SMB PROTOCOL.md), so true co-occurrence is unidentifiable; all numbers are
proxies over normalized per-clip windows (generous upper bound).

Measured (16,319 cross-camera event pairs):
- raw normalized-window overlap: **0.5187** — but the null (random placement,
  same window lengths) is **0.5050**: excess over chance **+0.0137**.
- strong overlap (IoU ≥ 0.5, closer to "same event twice"): **0.1026**,
  null **0.0884** — at chance and below the 15% gate in absolute terms.
- mean normalized window length 0.247 of a clip → the raw proxy is saturated
  by construction.

**Verdict: KILL P4-f.** Read literally on the raw proxy the 15% gate is not
triggered, but the raw number is chance-level noise; the decision-relevant
readings (excess +1.4pt, strong overlap 10.3%) are far below 15%, and the
strict quantity cannot be measured at all on SHT. Cross-camera dedup has no
evidence base on the only true multi-camera dataset we have.

## T5.2 — λ oscillation under dual ascent

Run dir: `~/e1/runs/20260925_1931_t5_2_lambda_replay/`

Logged fleet-v1 arrival streams replayed verbatim from
`~/e1/runs/20260919_0100_fleet_v1_sweep` (`fcfs_N16`: 338/341 arrivals,
complete to vt 3520 s; `fcfs_N24`: 358 arrivals, wall-truncated at vt 2726 s,
replay restricted to logged horizon). Minimal simulated controller:
admit iff `salience ≥ λ`; `λ ← max(0, λ + η·(spend_W − B_W)/B_W)` per
60 vt-s window; B_W = 0.9 × (3 lanes / 2.4 accel) × 60 = 67.5 GPU-s.

Measured (post-transient second half; criteria: p2p/mean ≥ 10% AND ≥ 6
reversals/100 windows):
- fcfs_N16 (demand 1.19 GPU-s/vt-s vs capacity 1.25): oscillates at every
  η ∈ {0.05, 0.2, 0.5, 1.0}; p2p/mean 0.66 → 2.82, 47–57 reversals/100 w.
- fcfs_N24 (demand 1.64): oscillates at every η; p2p/mean 0.42 → 2.18,
  48–74 reversals/100 w.
- Trace plot: `artifacts/lambda_traces.png` — visible sustained limit
  cycles, λ repeatedly spiking and collapsing to the 0 clamp, no convergence.

**Verdict: PASS** — λ does not converge smoothly; sustained oscillation is
the default behaviour at every step size tried. **Build hysteresis M2.**

## T5.3 — Flat index vs 1M events

Run dir: `~/e1/runs/20260925_1953_t5_3_flat_index_bench/` (REPORT.md +
ADDENDUM.md — the addendum carries the corrected verdict; earlier draft run
20260925_1939 kept per no-delete rule).

faiss IndexFlatIP, CPU only (GPUs untouched per instructions), 384-dim
unit-norm float32 — dimension/dtype/normalization validated against the real
bge-small embeddings in run `20260918_0111_writepath_fullcorpus`
(633 files, (1,384) float32, L2 norm 1.0). 1000 single queries, k=10;
self-recall@1 = 1.0 sanity. Threshold documented in config: p99 < 50 ms
single-query latency at 1M (interactive retrieval feeding a 60 s-SLO path).

Measured:
- 10K: p99 0.61 ms · 100K: p99 67.3 ms · 1M: p99 110.6 ms — all at the
  default 72 OpenMP threads on this shared 72-core box.
- The 10K→100K cliff is the L3 boundary (15 MB → 154 MB working set);
  100K→1M grows only ~1.6× (memory-bandwidth bound, 1.5 GB/query).
- Thread sweep at 1M: **32 threads → p99 36.5 ms (p50 29.3 ms)**; 8 threads
  → p99 88.8 ms; 72 threads is oversubscribed.
- Batched serving at 1M (32 queries/batch): **2.79 ms/query, 358 QPS**.

**Verdict: flat HOLDS at 1M → KILL P5-f (index re-architecture).** The gate's
kill condition ("flat holds at 1M with acceptable latency") is met under the
documented 50 ms p99 bar once threads are configured sanely — a config knob,
not a re-architecture — and met 18× over in batched serving. Caveats: the
72-thread default does break the bar on this shared machine; P5-f may be
revisited beyond 1M records or under heavy CPU contention, neither of which
is the current regime.

## T5.4 — Fleet-quantile vs raw score on SHT

Run dir: `~/e1/runs/20260925_1943_t5_4_fleetquantile_input_audit/`

**BLOCKED.** No per-clip tier-1 (VadCLIP) anomaly scores exist for
ShanghaiTech test videos:
- `~/FleetVAD/research/e0/results/tier1_scores.npz` → UCF-Crime (290 videos;
  e0 REPORT.md: VadCLIP sanity AUC1 = 0.8802 = published UCF number).
- `~/FleetVAD/research/e0/followup/results/xd_tier1_scores.npz` → XD-Violence
  (800 videos).
- `~/FleetVAD/research/e0/results/sht_features.npz` → CLIP scene/motion
  features for 107 SHT videos (redundancy study), NOT anomaly scores.
- Glob audit for `*sht*score*` / `*tier1*sht*` under `~/FleetVAD/research/e0/`
  and `~/e1/`: 0 hits.

Per the standing rule, no substitute was used and no scores were computed
(that is new inference work outside Work 1). The gate becomes runnable the
moment SHT VadCLIP per-clip scores exist — SHT GT frame masks are already
local (`~/FleetVAD/research/e0/data/shanghaitech/.../test_pixel_mask/`).
**M7/P2-V2 undecidable, not killed, not validated.**
