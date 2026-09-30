# INTEGRITY NOTE — deferred@tax0.75 arm (byte-identical to deferred@tax0.5)
Date: 2026-09-28 (investigation, CPU-only; no relaunch)
Run dir: /data3/zhuotaotian2_e2/runs/20260928_0315_e3_phase2_compB
Verdict: NO data fabrication / NO back-fill copy. The arm RAN for real (17:31–19:52, GPU3) but
DECAF_B_TAX=0.75 is a structural NO-OP for the deferred policy at the current storage granularity,
so its outputs are identical to tax0.5 BY CONSTRUCTION. Retract/relabel the row; a rerun at 0.75
would deterministically reproduce the identical files.

## 1. Did the arm run? YES — a genuine, distinct process
- sweep.status: WAVE1_LAUNCHED 17:31:04; WAVE1_ARMS_DONE 19:56:05. The tax075 arm was launched
  on GPU3 with `DECAF_B_COMMIT=deferred DECAF_B_BIAS=debias DECAF_B_TAX=0.75 DECAF_QUANT=0`
  (sweep_chain.sh launch_arm call, verbatim in the script).
- logs/answers_7b_b_deferred_tax075.answer.log: real run, start 17:31:08, model load 17:31:08-14,
  10/10 videos, 2h21m38s total. md5 (de727f94...) differs from the tax0.5 log (10b63333...);
  per-video timings differ (e.g. video 1: 1413.94 s/it vs 1277.02 s/it for the 12:49 tax0.5 run).
- answers_7b_b_deferred_tax075/{1_0.csv,commit_log.jsonl} written 19:52 (fresh mtime, not a copy
  preserving the 15:13 tax0.5 timestamps).
- Config evidence: no per-arm config.json is dumped by the launcher (gap — see follow-ups), but
  the env var IS exported in launch_arm and IS consumed by the code
  (model/attention/decaf_b.py:25 `DECAF_B_TAX = float(os.environ.get(...))`); code @ ac79712 as
  recorded in run config.json.

## 2. How the identical files came to be — config no-op, not a copy
- No back-fill: grep of sweep_chain.sh and run_judge_phase2.sh finds NO cp/rsync of answer or
  judged files (the only "cp" in sweep_chain.sh is a Python variable name, line 106).
  judge_arm re-runs the 72B judge on $ARM/1_0.csv; the judged json is identical because its INPUT
  csv is byte-identical and the judge is deterministic (temp 0).
- Mechanism (root cause): model/attention/decaf_b.py
  - L271: `budget = DECAF_B_TAX * self.seg_tokens`, seg_tokens = 4 frames x 196 = 784.
  - Deferred branch (L281+): budget is split per signal, `budget/2` each, and a frame costs
    block_size=196 tokens if not already kept (whole-frame storage granularity).
  - Per-signal budget vs frame cost 196:
      tax 0.25 ->  98  -> 0 frames/signal  (nothing retained)
      tax 0.50 -> 196  -> exactly 1 frame/signal
      tax 0.75 -> 294  -> still 1 frame/signal (2nd frame needs 392 > 294)
      tax 1.00 -> 392  -> 2 frames/signal
  => For deferred, TAX only changes retention at TAX >= k/2 (k=1,2,...). The value 0.75 falls in
     the dead zone [0.5, 1.0) and is mathematically guaranteed to reproduce the tax0.5 run.
- Commit-log stats confirm: identical store totals for tax0.5/tax0.75
  (both 6278.067 GB summed store_bytes, 272002 segments, 158/1465 uncertain), while tax0.25 is
  radically different (163.776 GB, 493/1465 uncertain). Identical retention -> identical greedy
  answers (same seed 2024) -> identical 1_0.csv -> identical judged json -> identical GB/h
  (both 5.44). The equal GB/h is corroboration, not coincidence.

## 3. Impact
- The deferred@tax0.75 row in the sweep table is a DUPLICATE of deferred@tax0.5, not an
  independent measurement. It must not be cited as a tax-sweep datapoint.
- The equal-tax Gate-2 pair at tax0.75 (deferred vs writetime) still has a valid writetime arm
  (judged_answers_7b_b_writetime_tax075_72b.json differs from writetime tax0.5: 1079510 vs
  1076849 bytes), but its deferred counterpart contributes no new information beyond tax0.5.

## 4. Recommendation (not executed)
- Do NOT rerun deferred@tax0.75 as-is: with current granularity it will deterministically produce
  byte-identical output again (4h GPU wasted).
- Retract/relabel the row in the sweep table as "degenerate: identical to tax0.5 by construction
  (per-signal budget 294 < 2-frame cost 392)". The deferred-vs-writetime Gate-2 decision at tax0.5
  stands on the valid arms (McNemar already decided it).
- If an above-0.5 deferred point is scientifically wanted, run deferred@tax1.0 (per-signal budget
  392 = 2 frames/signal; ~4h GPU, one arm) and label the current 0.75 row as retracted.
- Process fix (cheap, do regardless): launch_arm should append the exported DECAF_B_* env to
  $RUN/$ARM/config.json (or `env | grep DECAF` dump) at arm start so future arms carry their own
  config evidence; and the sweep table generator should assert md5(commit_log) uniqueness across
  arms before publishing rows.
