# W4 — fps fix (W4.1) + coverage-spread decision experiment (W4.2)

Status: LAUNCHED 2026-09-30 09:40 CST. Results pending (see RESULTS_W41.md /
RESULTS_W42.md in the Ada run dirs when the watcher completes).

## Ada run dirs
- W4.1: `/data3/zhuotaotian2_e2/runs/20260930_0935_w41_fpsfix/`
- W4.2: `/data3/zhuotaotian2_e2/runs/20260930_0935_w42_spread/`

## W4.1 fps fix (foundation repair)
- Bug: `video_qa/rekv_stream_vqa.py` npy branch re-resampled the 0.5-fps
  `/dev/shm/e3_mukv_npy` arrays at `sample_fps` → **0.25 fps effective**, while
  the MuKV anchor (no resample) and stock-ReKV anchor (mp4 path) ran at true
  0.5 fps.
- npy rate verified 0.5 fps (1800–1806 frames / 3600 s; decode stride
  `fps/0.5`) → no re-decode; fix is to consume at full rate.
- Fix (commit `eeb7f60`): `NPY_FPS=0.5` class constant; npy consumed at full
  rate (resample only if config fps < 0.5); **standing assertion**
  `_assert_effective_fps` at encode start — effective fps = n_frames/duration
  must equal config fps (tol 0.05), logged per video to
  `save_dir/fps_assertion.jsonl`, aborts the run on mismatch.
- Loader verified offline: 0.5→1800 frames (eff 0.5000), 0.25→900, assert
  fires for fps>0.5. All 4 live arms logged
  `{"effective_fps": 0.5, "n_frames": 1800, "assert_passed": true}`.
- Rerun: writetime tax0.5 debias, full RVS-Ego 1465q, GPU0 detached
  (identical store config to compB except the fps fix; judge: 72B TP=4).

## W4.2 coverage-spread (the decision gate)
- Question set: the **83** all-wrong temporal-before/after questions from the
  W3.3 ceiling analysis, recomputed 1:1 with `w33_ceiling.py` logic (86
  temporal all-wrong = 83 model-capability + 3 store-coverage; the 83 are
  used). Subset anno spans 2 videos; window ends non-decreasing per video ⇒
  per-question store state identical to a full run.
- Same store config, same 64-slot budget; only retrieval allocation changes
  (code `adb5f78`, slot-guard fix `91fa66b`):
  - A0 top-k (relevance) — from the W4.1 run, no extra GPU
  - A1 uniform — best unit per 64 equal temporal bins over [0, frontier)
  - A2 hybrid — 32 top-k + 32 uniform (PRIMARY arm)
  - A3 stratified — slots ∝ softmax(bin max similarity)
- 🔴 Oracle constraint: bins span streamed history up to the store frontier
  only (derived from segment token_starts); the gold window is not an input
  to `plan()` — asserted in code, `gold_window_used=false` logged per
  question. (First launch had a slot-guard bug counting grain entries —
  patches pack 4/slot — underfilling 64→16 slots; caught ~12 min in at
  first-commit-log inspection, fixed, arms relaunched; W4.1 arm untouched.)

## Pre-declared statistics (written to W4.2 config.json BEFORE any number)
- Primary: hybrid vs top-k, paired McNemar (exact binomial, two-sided) on the
  83 occurrence-indexed questions. Secondary: uniform, stratified vs top-k.
  Multiplicity: Holm across the 3 arm comparisons.
- **MDE statement (n=83, observed base acc ≈14%):** exact binomial McNemar at
  two-sided α=0.05 reaches 80% power only for discordant imbalances
  |b−c| ≥ ~13–17 given the expected d≈20–30 discordant pairs — i.e. a
  **minimum detectable accuracy difference of ≈16–21 points** (exact binomial
  computation: d=20→15.7 pts, d=25→18.1, d=30→20.5). A +10-pt lift over the
  14% base yields ~26 expected discordants, at the edge of detectability.
  The experiment is powered only for very large effects; this caveat is
  reported regardless of outcome.
- Decision mapping: coverage↑ AND hybrid acc +≥3pts → diagnosis actionable,
  Phase-4 redirection justified; coverage↑ acc flat → genuinely model
  capability, direction closed, diagnosis paper stronger; coverage flat →
  allocator bug, not a finding.

## What runs with what ETA (start 09:40 CST)
- GPU0: W4.1 writetime@0.5fps full 1465q (~2.5h, encode-dominated)
- GPU1-3: W4.2 uniform/hybrid/stratified on 83q × 2 videos (~1-1.5h)
- Then: orchestrator watcher → 72B TP=4 judge ×4 → stats (3 McNemar for W4.1,
  3 McNemar + Holm + coverage table for W4.2) → RESULTS_W41.md / RESULTS_W42.md

---

# W4 CLOSE-OUT — 2026-09-30 (four senior directives)

## W4.1 verdict (fps fix) — CLOSED
True-0.5-fps writetime tax0.5-debias arm completed and judged: **52.4 (n=1465)**.
FPS_ASSERT passed on all 10 videos (effective 0.5000–0.5002, 1800–1806 frames /
3600 s; `answers_7b_writetime_05fps/fps_assertion.jsonl`). Paired McNemar
(n=1465): vs MuKV paper-config 50.9 → **p=0.203**; vs stock ReKV 54.1 →
**p=0.139**; vs old 0.25fps-effective writetime 51.1 → **p=0.093**.
Full detail: `runs/20260930_0935_w41_fpsfix/RESULTS_W41.md` (mirrored to Mac
`research/e3_phase2/w41_fpsfix/`).

## Directive 2 — Pareto statement at corrected fps (bytes and encode compute SEPARATE)

**(a) Did bytes change with the fps fix? YES — bytes doubled. The
"bytes-unchanged" hypothesis is refuted.** Segments are 4-frame token segments
(784 tokens), and the writetime per-segment budget is tax·seg_tokens/2 = 196
tokens (1 frame) per signal — budget is per *segment*, not per wall-clock
second. Doubling fps doubles segments per video (n_segments 225 → 450 on
full-length videos), so the realized store doubles:

| arm | store GB/h | total bytes (10 videos, 9.543 stream-h) |
|---|---|---|
| writetime tax0.5 @0.5 fps (W4.1, corrected) | **10.38** | 99.04 GB |
| writetime tax0.5 @0.25 fps effective (compB sweep) | 5.44 | 51.93 GB |
| MuKV published (their own 0.5 fps measurement) | 0.91–1.23 | — |

Same methodology as the sweep: per-video max `store_bytes_total` from
commit_log.jsonl, summed over 10 videos, divided by 9.543 stream-hours
(ego4d_oe.json durations). Per-video bytes are NOT identical between arms
(max 8.66–10.39 GB/video new vs 5.19 GB/video full-length old).

**Consequence for the MuKV ratio:** the old 4.4–6.0× storage ratio
(5.44 / 0.91–1.23) compared our half-rate store against MuKV's full-rate
number — NOT like-for-like. The corrected like-for-like ratio at MuKV's own
0.5 fps is **8.4–11.4×** (10.38 / 1.23 → 10.38 / 0.91). This is the first
time the comparison is apples-to-apples on fps; the ratio worsens and must be
quoted as 8.4–11.4×, not 4.4–6.0×.

**(b) Encode compute** (wall-clock, single GPU — GPU-s ≈ wall-s):
total pipeline 2h55m50s = 10 550 s → **1105 GPU-s/stream-hour** at 0.5 fps,
vs 2h17m01s = 8221 s → **861 GPU-s/stream-hour** for the old 0.25fps-effective
arm (+28.4%). Steady-state per-video (excl. first-video warmup) 920.7 s vs
786.9 s; the +134 s/video increment is the encode phase (frames doubled
900→1800/video; answer count and judge identical), i.e. **encode-phase
compute roughly doubles** (~140 → ~280 GPU-s/stream-hour, decomposition
estimate). What the doubled encode + doubled bytes bought: +1.3 pt over the
half-rate arm (52.4 vs 51.1, p=0.093, n.s.) and like-for-like fps against
the MuKV anchor for the first time (52.4 vs 50.9, +1.6 pt, p=0.203, n.s.).

## Directive 1 — W4.2 closes as UNDERPOWERED-NULL (dissection STOPPED by senior)

The deciding test already ran: hybrid vs top-k paired McNemar on the 83
temporal-before/after questions = **p=0.180** (2/7/9 discordant; Holm 0.539
across the 3 arms), observed **+6.0 pt** (12.0 vs 6.0) against a pre-declared
MDE of ~16–21 pt at n=83 (80% power; d=20→15.7 pt, d=25→18.1, d=30→20.5).
**Verdict: underpowered-null.** The coverage-spread experiment cannot decide
the causal question at n=83; no effect is established to explain, and per
senior directive the flat-coverage dissection is STOPPED. The accuracy
differences across arms (6.0 → 12.0%, i.e. 0–6 correct out of 83) are ~1.7σ
events — a third of the MDE — the same shape as the learnability probe's
split-selection noise (seed-42 +1.90 pt vs 200-split mean −1.02 ± 1.75 pt).
Scaling caveat (not a queued task): deciding the causal question needs a
substantially larger temporal-question set.

**Superseded material:** the dissection agent's early work in
`runs/20260930_0935_w42_spread/` — `debug_w42.py` + `debug_w42.json`
(retrieval-set Jaccard between arms: top-k∩uniform 0.177, top-k∩hybrid 0.270,
top-k∩stratified 0.287 — the arms genuinely retrieve different sets, coverage
still flat) and `stats_output.txt` (a crashed partial stats run) — is
**superseded-by-directive** and must not be developed further. The
pre-registered decision mapping's "coverage flat → allocator bug suspected"
branch is likewise closed without an allocator-bug finding: at n=83 the
accuracy arm of the mapping has no power, so neither branch of the mapping is
actionable.

## Directive 4 — corrected comparisons through the W3.1 family

`analysis/w41_family_correction/` (script + family_A/B jsons + correction
json + matrix csv): the W3.1 methodology re-run with the corrected
writetime@0.5fps arm replacing the old 0.25fps-effective writetime entries in
family A (m=21, all pairs among the 7 full-run arms) and in the family-B
writetime filter-in (oracle 2-video subset, m=15).

- **writetime@0.5fps vs MuKV paper-config:** raw p=0.203 → **Holm 1.0,
  BH 0.388** (family A, m=21). Nothing survives.
- **writetime@0.5fps vs stock ReKV:** raw p=0.139 → **Holm 1.0, BH 0.291**.
- Family B (m=15): nothing survives (min Holm 0.301). Union (m=36): corrected
  arm Holm 1.0 / BH 0.312–0.430.
- Pairs that DO survive are pre-existing structural ones not involving the
  corrected arm: stock-ReKV-above-tax075 (Holm 0.020), MuKV-below-stock
  (BH 0.037), and writetime@0.5fps-above-tax075 (raw 0.0053, BH 0.037, Holm
  0.101 — the corrected successor of the W3.1 primary Pareto pair, surviving
  BH only). No new accuracy claim arises from the corrected arm.

## Status
W4.1 CLOSED (foundation repaired, standing fps assertion live).
W4.2 CLOSED as underpowered-null; causal coverage question needs a larger
temporal-question set (scaling caveat, no queued task).
W4.3 stands as the correlational synthesis; its causal gate (W4.2) returned
underpowered-null, so W4.3 remains suggestive, not causal.
