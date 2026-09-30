# E3 Phase 1 — Component A CORRECTED (rank-remapped fetch positions), Gate 1 re-confirmation

Date: 2026-09-27 ~21:45 CST · Server: Ada (GPUs 0-3) · Code: code/decaf @0277a19
Context: draft-plan/beat-mukv.md §5 Gate-1 post-mortem; prior run
runs/20260927_1200_e3_phase1_compA (exact-position fetch failed 22.3 vs 54.1).

## Design change (corrected Component A)
Same position-free store (pre-RoPE K/V + position sidecar) and content-key
retrieval; fetch-time position assignment changed from EXACT streaming
positions to RANK-REMAPPED buffer positions: the block loaded at buffer slot
cnt gets position n_init + cnt*196 + j (contiguous, retrieval-score/load
order — the stock ReKV convention); question/decoding tokens continue right
after the buffer (next_pos left unset, decoding continues from carried rank
positions). Gate: DECAF_FETCH_RANK=1 (requires DECAF_POSITION_FREE=1 for the
store/sidecar; rank branch takes precedence at fetch). Exact path kept for
the record. Rationale: exact positions (~86k tokens on a full RVS-Ego
stream) exceed the model in-distribution RoPE range and collapsed late-stream
generation (measured 22.3 vs 54.1 stock).

Commits: fe7af47 (rank fetch impl), 8ffe916 + 0277a19 (trapcheck mode=rankqa).

## Trap checks (verification/decaf_trapcheck.py mode=rankqa; 0.5B, GPU3)
- TR1 rank plumbing: retrieval prefill on the full store carries contiguous
  buffer-rank positions; next_pos unset. 10/10 questions PASS.
- TR2 late-stream stock equivalence: all 10 questions of 0fa75cb3 answered
  with the FULL 420-frame store (82,333 tokens at query time — the exact-mode
  failure position) under rank mode: 10/10 answers character-identical to
  W2.5 stock (answers_0.5b CSV; T5 proves the flag-off repo == W2.5 stock in
  this harness). Answers fluent, mid/long form; no mid-sentence truncation.
- Verdict: gate_rank_plumbing PASS, gate_late_stream_stock_equiv PASS,
  OVERALL PASS (trap_checks_rankqa.json, RANKQA_EXIT 0 21:59:40).

## Measurement run (detached)
- answers_7b_rank: 7B llava_ov_7b, GPU1, W2.5 config (fps 0.5, n_local
  15000, retrieve 64, greedy, seed 2024), DECAF_POSITION_FREE=1 +
  DECAF_FETCH_RANK=1, PYTHONPATH=/data3/zhuotaotian2_e2/code/decaf,
  HF_HUB_OFFLINE=1. Started 21:50; prior exact-mode 7B run took 3:49 ->
  ETA ~01:40 2026-09-28. Retrieval log: retrieval_rank_7b.jsonl.
- Judge watcher (run_rank_judge.sh, setsid PID logged in rank_judge.status):
  waits for EXIT 0 + 1465 rows, waits GPUs 0-3 free, judges with
  Qwen2.5-72B TP=4 gpu_mem 0.80 (env /data4/e2judge) ->
  judged_7b_rank_72b.json -> rank_verdict.txt vs W2.5 stock 54.1/2.82.
- GATE (corrected): judged Acc within +/-1.5 pts of 54.1 -> Phase 2 base set.

## Known issues / notes
- First answer launch used a wrong model key (llava_ov_qwen2_7b ->
  KeyError); fixed to llava_ov_7b and relaunched. The stale EXIT line made
  the first judge-watcher launch abort; status file cleared, watcher
  relaunched (verifies EXIT + row count before judging).
- /dev/shm/e3_mukv_npy pre-decoded videos intact (10 npy).
