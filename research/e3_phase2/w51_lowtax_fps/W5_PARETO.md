# W5 Pareto attempt — status (2026-09-30, in-session snapshot)

Scope: W5.1 (low-tax writetime at true 0.5 fps — the gate) and W5.2 (int4
group scales) of `ongoing-work/work5.md`. Framing is binding: **parity, not a
win**; no "beats" outside the m=21 multiplicity family; the Pareto attempt
optimizes an existing operating point, it invents no mechanism.

## W5.1 — low-tax writetime at true 0.5 fps

Run dir (Ada): `runs/20260930_1750_w51_lowtax_fps/` · mirror:
`research/e3_phase2/w51_lowtax_fps/`

**Explicit low-tax validity verification (required by work5.md).** Verified in
code (`model/attention/decaf_b.py::_assemble_segment`): the writetime budget is
`tax * 784` tokens per segment but the realized storage granularity is the
**whole frame (196 tokens)**. Consequences:
- tax in [0.25, 0.5): exactly 1 frame/segment, byte-identical at any tax in the
  range — the only sub-0.5 writetime operating point at frame granularity.
- tax < 0.25 (incl. 0.125): budget < 196 → keep-list empty → **degenerate
  (empty store)**. Frame-granular writetime at tax0.125 is INVALID.
- The W3 dead zone (per-signal frame quantization in [0.5, 1.0)) is a
  *deferred-arm* constraint and does not apply to writetime — confirmed.

**Arm 2 therefore runs tax0.125 with env-gated patch-granular realized storage**
(`DECAF_B_PATCHSTORE=1`, fp16, writetime only): the ranked keep-list charges
49-token patch slices, stores only those slices, and the query path's existing
patch-slice materialization serves them. This is the "lowest valid writetime
tax" the formulation allows once sub-frame realized storage is enabled — it is
the same ranked keep-list mechanism, honest byte accounting, no new mechanism.

### Launch state (2026-09-30 ~17:53 CST)

| arm | GPU | config | status | ETA (CST) |
|---|---|---|---|---|
| tax025 | 0 | writetime tax0.25 debias fp16, corrected loader (commit 91fa66b), full RVS-Ego 1,465q | RUNNING (fps assert 0.5000 on video 1 at 17:53) | answers ~20:50; judged ~21:05 |
| tax0125_patch | 1 | writetime tax0.125 debias fp16 + DECAF_B_PATCHSTORE=1 (commit 9413784+), same data | launching after trap gates (all PASS) | answers ~1h after launch; judged +~15 min |

Watcher (`watcher.sh`, setsid nohup, abort-without-verdict): waits for both
arms' EXIT 0 → verifies fps_assertion.jsonl (10 videos, assert_passed) → row
counts → flock-serialized 72B TP=4 judge → `stats_w51.py` → RESULTS_W51.md
(gate evaluation: tax0.25 ≥ ~51 at ≤ 6 GB/h).

Code gates passed before launch: TB1 assembly/accounting (patchstore realized
token fraction exactly 0.125/segment; honest per-entry bytes incl. patch
slices), TB4 answering smoke 10/10 non-empty on patch-sliced store, TB2
quant roundtrip (q4g), bias debias — `OVERALL PASS` (TB3 rank-position
bookkeeping is SKIP in this transformers build: `past_key_values` positions
unavailable — pre-existing introspection gap, not a store defect).

### Gate prediction

- tax0.25 accuracy: the pre-fix arm scored 51.8 at half rate; W4.1 showed the
  fps correction moved tax0.5 by +1.3 (51.1→52.4). Prediction: **52–53 at
  ~5.6 GB/h** → gate (≥~51, ≤6 GB/h) likely PASSES.
- tax0.125_patch: first sub-frame operating point; accuracy unknown — if it
  holds ≥50 at ~2.8 GB/h it becomes the W5.3 candidate point; if it collapses,
  tax0.25+int4 (1.40 GB/h arithmetic) remains the Pareto candidate.

## W5.2 — int4 group scales

Implementation: `model/attention/decaf_b.py` @ 9413784, env-gated, default off:
- `DECAF_QUANT=1 DECAF_QUANT_GROUP=64|128`: asymmetric int4 over token groups
  (ragged last group — a 196-token block stays 196 tokens; zero-padding would
  have cost ~20% realized bytes), per-(head, group, channel) fp16 scale + uint8
  zero-point, K and V quantized by separate calls. Legacy per-channel path
  unchanged (GROUP=0).
- Honest per-entry byte accounting (`entry_nbytes`) feeding the existing
  store_bytes GB/h methodology — realized ratios measured, not assumed.

Unit verification (random N(0,2) tensors): G=64 cos 0.9958 / G=128 cos 0.9953
vs fp16; realized 3.56×/3.77× (ragged). TB2 on real 0.5B captures, G=64:
relerr mean 0.106 / max 0.171 (PASS < 0.30), mse/cosine instrumentation added.

Verification suite `runs/20261001_w52_int4_group/` (GPU2, detached):
120 questions from one full-length video (7B, writetime tax0.5 debias store),
arms fp16 / int4-legacy / int4-g64 / int4-g128 → 72B judge per arm →
`stats_w52.py`: headline **greedy-answer character-identical rate vs fp16**
(gate ≥95%, TB6 lesson — answer level, not logit level), realized-bytes curve,
judged accuracy delta + McNemar vs fp16. Answer ETA ~4×25 min sequential;
judge phase flock-serialized with the W5.1 watcher (72B TP=4 needs all of
GPUs 0–3).

Gate prediction: group asymmetry fixes the exact per-channel failure mode TB6
isolated (outlier channels collapsing group-uniform quantization); expect
g128 ≥ g64 ≥ legacy. If ≥95% answer-identical holds at g64/g128 with accuracy
delta within noise → W5.3 unlocked; else fp16 is the storage floor.

## Issues

- 2026-09-30 ~18:20: full network outage to Ada (100% packet loss, beyond the
  known sshd flakiness). All runs are detached (setsid nohup) and proceed on
  the box; pushes (watcher launch, arm-2 launch, suite launch, this file)
  resumed when connectivity returned. If the box rebooted, /dev/shm npy cache
  is volatile — arms would abort at the next video decode and the watcher
  would record ABORT (no verdict fabricated).
