# W2.2 — Does TTKV's sensitivity finding transfer to video KV? (FINAL REPORT)

**Task:** work2.md §W2.2 / inside-the-model.md §6 secondary check. TTKV found frequently-selected
LLM KV blocks are 2.22× more sensitive to 2-bit quantization than rarely-selected ones — the
justification for retrieval-adaptive precision (component (c)). Gate: **PASS** if Hot 2-bit
distortion ≥ 1.5× Cold; **FAIL** if flat across quintiles → drop component (c).

## Method

- **Data:** selection sets + frequencies from run `20260925_2237_t1_rekv_sharded`
  (`artifacts/t1_rekv_log.jsonl`; ReKV-0.5B, RVS-Ego, 10 videos × 12 seeded queries, 120 queries,
  top-64 per query). KV was not persisted → re-encoded with the T1 machinery
  (`~/e1/baselines/t1_rekv_shard.py` extended as `w22_quant_sensitivity.py`; same model/config).
- **Quintiles:** per video over USED blocks (f>0; 96% of blocks are never selected, so all-block
  quintiles are degenerate). Q1=Cold .. Q5=Hot by frequency rank. 5 blocks sampled per quintile
  per video (seeded): **250 blocks, 8,367 quantization evals** (every query whose logged top-64
  contains the block, × 3 bitwidths).
- **Quantization:** one block at a time at 2/4/8-bit — symmetric uniform quant-dequant simulating
  b-bit storage; K per-(head,channel) scales over tokens, V per-(head,token) scales over channels;
  applied to the stored KV of the single block across all 24 layers; GPU-cached copy offloaded
  before mutation. Query, selection set (logged top-64), and all other blocks held fixed; the
  retrieval index (block representative keys) left at full precision.
- **Baseline:** external retrieval path, unquantized, per (video, query). Path verified
  deterministic and restore-after-quantization bit-exact (`w22_probe.py`); 8-bit output-identical.
- **Distortion metrics** (GPT-judge harness does not exist; token-F1 is the established T1 proxy):
  - primary `drop` = token-F1(gold | baseline) − token-F1(gold | quantized), parallel to T1 LOO drops;
  - secondary `out_f1` = token-F1(pred_quantized, pred_baseline); output distortion = 1 − out_f1.

## Results — distortion by quintile (block-level means, 50 blocks per quintile)

| Quintile | 2-bit drop (gold-F1) | 4-bit | 8-bit | 1 − out_f1 @ 2-bit |
|---|---|---|---|---|
| Q1 Cold | 0.0456 | −0.0029 | 0.0004 | 0.411 |
| Q2      | 0.0320 | −0.0026 | −0.0006 | 0.378 |
| Q3      | 0.0391 | −0.0001 | −0.0010 | 0.351 |
| Q4      | 0.0431 | +0.0000 | −0.0009 | 0.407 |
| Q5 Hot  | 0.0451 | −0.0022 | −0.0007 | 0.476 |

- 2→4-bit distortion reduction: 0.039 / 0.035 / 0.039 / 0.043 / 0.047 (Cold..Hot) — flat.
- **Hot/Cold 2-bit ratio: 0.99** (gold-F1 drop; Welch t = −0.03, p = 0.975) — no difference.
- Output-distortion ratio Hot/Cold: **1.16** (p = 0.054, marginal; non-monotone across quintiles,
  0.411 / 0.378 / 0.351 / 0.407 / 0.476 — only the Hot endpoint elevated).
- Spearman(selection frequency, 2-bit drop) over all 250 blocks: −0.05 (p = 0.43);
  vs output distortion: −0.06 (p = 0.32). **No frequency-sensitivity relationship.**
- Sanity: 8-bit output-identical in the probe and ~0 drop at scale; 4/8-bit gold-F1 drops ≈ 0
  (quantization invisible through the noisy gold lens even though 4-bit still rewrites outputs,
  out_f1 ≈ 0.95–0.97).
- Noise check: only 34–44% of 2-bit evals have positive gold-F1 drop (same weak-instrument caveat
  as T1's 45.7% negative LOO deltas). Single-block 2-bit quantization *rewrites* the output
  (out_f1 ≈ 0.52–0.65) but the rewrite is as likely to help as hurt against gold token-F1.

## Drift test (log-only, `w22_drift.py`)

Split each video's 12 queries (log order) into disjoint early (q1–6) / late (q7–12) halves;
blocks assigned to Cold/Warm/Hot terciles by frequency rank independently per half.

P(late | early), rows = early, cols = late:

| | Cold | Warm | Hot |
|---|---|---|---|
| Cold | 0.966 | 0.026 | **0.0086** |
| Warm | 0.031 | 0.965 | 0.004 |
| Hot  | 0.005 | 0.009 | **0.987** |

- cold→hot = **0.83%** (TTKV reference: 7–8%). Spearman ρ (early vs late frequency) = **0.949**
  per-video mean (TTKV: 0.64–0.78). Selection-frequency groups are *more* stable here than in
  TTKV's LLM session. The proposal's "importance drift is the dominant regime" claim finds no
  support at this scale; the multi-day regime it cites is untested by this data.

## Verdict: **FAIL — drop component (c)** (retrieval-adaptive precision)

Hot blocks are NOT ≥ 1.5× more quantization-sensitive than Cold blocks on the gate metric
(0.99×; output-distortion alternative 1.16×, also below gate). Per work2.md: drop component (c);
components (a) GOP coding and (b) nested planes stand on their own, and this result does not
affect W2.1.

**Caveats.** (1) The gate was measured through the token-F1 proxy on a 0.5B model — the same weak
instrument T1 flagged (noise swamps per-block signal; a logit-KL measure or a real judge might
resolve a small true gradient, and out_f1 hints Hot blocks distort somewhat more). "FAIL" here means
*no detectable sensitivity gradient with this instrument*, not proof of perfect flatness.
(2) Quintiles are within-video over used blocks; 12 queries give coarse frequencies (1–12, many ties).
(3) Coverage: 10/10 videos, 250/250 planned blocks, 8,367 evals — complete, no missing input.

## Operational history (failures retained, nothing overwritten)

- shard0/1 first launch (15:20): died at first quantize call — `torch` imported only inside main()
  (helpers need module level). Fixed; relaunched 15:37 as shard0b/1b; shard2b (fixed code) ran clean.
- Network/sshd outage #1 16:20–17:35; shard2b died silently between videos (no traceback, same
  pattern as T1 shard2). shard0b/1b survived (detached).
- shard2c (17:37, `--skip 2bb31b69`) finished shard2.
- Outage #2 18:14–22:31. shard1b died silently before its 3rd video (af400f98); shard0b was killed
  by a coordinating session at 22:50 (jsonl backed up to `.bak_preresume`, truncated to completed
  videos); shard0d (`--skip 0fa75cb3`) and shard1c (`--skip 20bc995f,879dd163`) completed the
  remainder. shard0d re-generated 6baa36c1 (768 evals, identical count to the killed run — deterministic).
- GPU0 (another tenant) untouched throughout.

## Artifacts

- `w22_quant_sensitivity.py` (runner; extends t1_rekv_shard.py), `w22_probe.py` (machinery
  verification), `w22_drift.py` + `w22_drift.json` (drift test), `w22_analyze.py` (aggregation),
  `w22_sensitivity.json` + `w22_sensitivity.png` (gate results + figure), `w22_block_level.json`
  (per-block means), `REPORT.md`.
- Run dir: `~/e1/runs/20260926_1445_w22_quant_sensitivity/` (config/env/code_state, shard*.out,
  artifacts/w22_quant_shard{0,1,2}.jsonl, artifacts/w22_quant_shard0.jsonl.bak_preresume).
