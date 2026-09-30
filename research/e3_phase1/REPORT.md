# E3 Phase 1 — Component A: position-disentangled store (Gate 1)

Date: 2026-09-27 · Server: Ada (RTX 6000 Ada x8, GPUs 0-3) · Code: code/decaf @5babc2f
Plan: draft-plan/beat-mukv.md §2 Component A, §5 Phase 1.

## What was built (env-gated DECAF_POSITION_FREE=1; unset = bit-identical stock ReKV)
- Position sidecar: per-block absolute start position (block_starts); identity
  block_starts[b] == n_init + b*196 trap-checked exactly.
- Rotate-on-fetch: retrieved blocks rotated to EXACT original positions from the
  sidecar before attention; question + decoding tokens placed at TRUE streaming
  positions (PositionalKV.next_pos = store length at query time). fp32 rotation
  math identical to the stock RotaryEmbeddingESM (helper-equivalence checked).
- Retrieval scoring UNCHANGED (content keys both sides).
- Instrumentation: DECAF_RETRIEVAL_LOG per-question jsonl (per-layer retrieved
  blocks + layer-0 similarities), DECAF_RECALL_ONLY mode, detailed memory report.

## Audit finding (changes the paper's framing, verify in T2 dump cross-check)
Stock ReKV's CPU-offloaded store ALREADY holds pre-RoPE keys (T1: stored K/V ==
raw k_proj/v_proj outputs with rel err exactly 0.0), and its internal retrieval
ALREADY scores pre-RoPE content keys. ReKV's position entanglement is at FETCH
TIME: retrieved blocks are rotated to buffer-RANK positions (contiguous in
retrieval-score order), and the question is placed at a buffer-relative offset.
DECAF-A's delta = exact-position rotate-on-fetch via an explicit sidecar, making
positions first-class store data (required for Phase-2 per-token grains).

## Trap checks (verification/decaf_trapcheck.py; results in trapcheck/)
- T1 store-is-content: rel err 0.0 (K and V) at layers 0/5/11/17/23, 96 blocks. PASS.
- T2c sidecar identity: exact. PASS.
- T3a helper == stock range rotation: PASS. T3b rotate-on-fetch plumbing (exact
  buffer positions, next_pos == store length, per-layer 64-block selection over a
  440-block store): PASS. T3d 2-question greedy smoke: fluent. PASS.
- T2 rotary-vs-W2.1-dump cross-check, T5 stock-equivalence vs W2.5 answers,
  T3c encode-invariance: third run (run_trapcheck_encode3.sh) — see trapcheck.status.
- 196-token/frame alignment: block_size=196 asserted; frame sampling stride-60 @30fps.

## Runs (detached)
- answers_7b_decaf (GPU1, started ~13:12, ETA ~17:30), answers_05b_decaf (GPU0,
  restarted 13:24, ETA ~15:30), recall_entangled_7b (GPU2), recall_entangled_05b (GPU3).
- Watcher: run_gate1_watcher.sh -> judges DECAF CSVs (Qwen2.5-72B TP=4, gpu_mem
  0.80) when GPUs free, computes temporal-oracle recall for all four retrieval
  logs, writes gate1_verdict.txt. Status: gate1_watcher.status.

## Recall oracle definition
Per-question window [start_time, end_time] (ego4d_oe.json) at 0.5 fps: relevant
blocks = {frame b : start*0.5 <= b < end*0.5} restricted to blocks encoded at
query time; micro + macro recall on layer-0 retrieved blocks; relaxed variant
+-30 s. Script: verification/compute_recall.py.

## Equal memory
Store bytes identical to entangled (same fp16 K/V path, MemoryUnit untouched);
sidecar = n_blocks x 8 B x 24 layer-managers (e.g. ~0.3 MB/video at 1729 blocks),
reported per run by calc_memory_usage_detailed (in answer logs).

## Gate 1 (pending)
recall +>=5 pts OR RVS-Ego Acc +>=2 pts (72B-judged) vs entangled (W2.5:
0.5B 45.5/2.36, 7B 54.1/2.82).

## Known issues / notes
- Runs MUST set PYTHONPATH=code/decaf (rekv env editable-installs code/ReKV).
- longva editable install was missing from /data4/rekv; reinstalled from
  code/decaf/model/longva (without it every stream-VQA run dies at import).
- Watchdog caveat: answering CSVs are written only at the end of a run.

---

## INCIDENT + RECOVERY (2026-09-27 ~18:40 CST)

### Incident report ("all four jobs silently died") — FORENSICS CONCLUSION: they did not die.
- All four measurement runs RAN TO COMPLETION with EXIT 0:
  answers_7b_decaf 13:04-16:53, answers_05b_decaf 13:20-15:17,
  recall_entangled_7b 13:12-15:36, recall_entangled_05b 13:18-14:58.
  Each 1_0.csv has 1465/1465 rows; answer.log progress bars show 10/10 videos.
- Root cause of the "dead jobs" appearance: the launcher passed RELATIVE status
  paths; run_answer_decaf.sh cds into code/decaf first, so the EXIT lines were


---

## INCIDENT + RECOVERY (2026-09-27 ~18:40 CST)

### Incident report ("all four jobs silently died") — FORENSICS CONCLUSION: they did not die.
- All four measurement runs RAN TO COMPLETION with EXIT 0:
  answers_7b_decaf 13:04-16:53, answers_05b_decaf 13:20-15:17,
  recall_entangled_7b 13:12-15:36, recall_entangled_05b 13:18-14:58.
  Each 1_0.csv has 1465/1465 rows; answer.log progress bars show 10/10 videos.
- Root cause of the "dead jobs" appearance: the launcher passed RELATIVE status
  paths; run_answer_decaf.sh cd's into code/decaf first, so the EXIT lines were
  written to /data3/zhuotaotian2_e2/code/decaf/*.status (4 files, all ending in
  EXIT 0) instead of the run dir. Driver logs are 0 bytes because the runner
  redirects all python output into <save_dir>/answer.log.
- The gate1 watcher (PID 1979329) waited on run-dir status files that never
  appeared, looped forever, and was then itself killed during session cleanup.
  It never judged anything.
- Validity checks of the completed runs (why we do NOT relaunch):
  (a) retrieval logs exist for all four arms (36/35 MB);
  (b) decaf answers differ from W2.5 stock at 91.5%/95.5% of rows (0.5B/7B) —
      with the flag OFF the repo is bit-stock (T5), so a low match rate proves
      the position-free path was genuinely active;
  (c) config matches W2.5 (fps 0.5, n_local 15000, retrieve 64, greedy,
      seed 2024) per answer.log headers; GPUs were correctly assigned.
- Known caveat: 100/1465 (0.5B) and 18/1465 (7B) empty pred_answer (W2.5 had 0);
  concentrated in video cbfeb6c8 (71). Treated as measured model behavior, not a
  script bug; costs accuracy in the judge. Flagged for the analysis payload.

### ENCODE4 trap-check failure — ROOT CAUSE: verdict-aggregation bug, mechanism sound
- ENCODE4 exited 1 because decaf_trapcheck.py:215 filtered verdict.values() with
  v.startswith("gate") while the dict also contains FLOAT entries
  (max_stored_*_rel_err) -> AttributeError AFTER all checks had run and passed.
  Fixed: filter on keys (k.startswith("gate")).
- Rerun (run_trapcheck_encode4.sh, GPU3, 18:41-18:46): ENCODE4_EXIT 0,
  COMPARE2_EXIT 0. T3c encode-invariance bit_exact=True (max_abs_diff 0.0) —
  the reference regenerated on GPU3 matches the qa-mode snapshot byte-for-byte.
- FINAL TRAP-CHECK STATE: all 7 gates PASS, OVERALL PASS
  (trapcheck/trap_checks_verdict.json): store_is_prerope (0.0), rotary (3.8e-4),
  sidecar identity, rotate-on-fetch, qa smoke, stock-equivalence (T5 both
  questions match W2.5), encode-invariance (bit-exact). Mechanism verified;
  prior agent's "script-side bug, not mechanism bug" claim CONFIRMED.
- Patch applied to verification/decaf_trapcheck.py (uncommitted on top of
  5babc2f; encode/verdict jsons from the crashed run preserved as *.bak_enc3).

### Recovery actions
- Killed stale watcher PID 1979329 (verified dead); no other stale phase-1 procs.
- Moved the 4 status files from code/decaf into the run dir.
- Relaunched gate1 watcher as run_gate1_judge.sh (setsid nohup, PID 2215640,
  kill-proof, status+EXIT lines in gate1_watcher.status, driver log
  gate1_judge.driver.log). Flow: verify 4x EXIT 0 -> rows 1465/1465 -> wait
  GPUs 0-3 <10GB -> judge both decaf CSVs (Qwen2.5-72B TP=4 gpu_mem 0.80)
  -> temporal-oracle recall (strict + +-30s) for all four retrieval jsonls
  -> gate1_verdict.py -> gate1_verdict.txt/.json.
