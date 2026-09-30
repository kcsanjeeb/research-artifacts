# Phase 2 compB recovery — crash diagnosis (2026-09-28 ~12:30 CST)

## Symptom
All 8 arms (3 answering + analyze_bias + 4 oracle) EXIT 1 within ~4-15 min of
the 04:20 launch. Judge watcher then processed the empty arms into a
MISSING/PENDING verdict (phase2_verdict.txt), and the oracle chain likewise
produced oracle_verdict.txt with no policy scores.

## Root cause (single code bug, hit by every arm)
Every arm's answer.log ends in the same traceback:

    File ".../model/attention/decaf_b.py", line 196, in add_frame_layer
        fr.attn[:, layer_idx] = attn_vec.cpu()
    IndexError: index 28 is out of bounds for dimension 1 with size 28

Mechanism:
- Component B (commit 4a837fe) assigns each ContextManager a `layer_idx` from a
  module-global creation-order counter `_LAYER_CTR` (kv_cache_manager.py:35,260).
  The comment assumes managers are constructed "layer 0..L-1 at first forward
  (encode_init_prompt)" — true only for the FIRST video in the process.
- `ReKVStreamVQA.analyze_a_video` calls `clear_cache()` + `encode_init_prompt()`
  per video. clear_cache drops the managers, so video N's init prompt constructs
  L fresh managers and the counter keeps climbing: video 1 gets layer_idx 0..27,
  video 2 gets 28..55 (7B: N_LAYERS=28).
- `MultiGrainStore.add_frame_layer` allocates `fr.attn` with N_LAYERS columns,
  so video 2's first offloaded block crashes with layer_idx=28.

Side effect of the same lifecycle gap (also fixed): the grain store singleton
(`decaf_b._STORE`) is never reset between videos, so video N>1 would inherit
video N-1's leftover frames (token_start key collisions) and segments (stale
segments eligible for retrieval commits in the new video).

## Why the trap checks did not catch it
`verification/decaf_b_trapcheck.py` runs ONE video per process (store / bqa /
quantqa modes, 0.5B), so the counter never exceeds N_LAYERS and the store never
carries across videos. Phase 1 (compA) never hit it because DECAF_B was off
(the layer_idx/store path is Component-B-only).

## Fix (code/decaf, committed)
- kv_cache_manager.py: `reset_layer_ctr()`; comment documents the per-video
  invariant.
- decaf_b.py: `reset_store()` (drops the shared store singleton).
- abstract_rekv.py `encode_init_prompt()`: when DECAF_B, reset both — this is
  the per-video stream entry point (matches the documented invariant).

## Verdict-logic fix (launchers, not committed to code repo)
- run_judge_phase2.sh v2: aborts without writing phase2_verdict.txt on any
  arm/judge failure (phase-1 rank_judge pattern — no MISSING verdicts); writes
  ARMS_JUDGED marker; writes phase2_verdict.txt ONLY when all 3 arm judged
  jsons AND all 4 oracle policy judged jsons exist (Gate 2 part 2 needs oracle).
- oracle_chain.sh v2: waits for ARMS_JUDGED (waiting for WATCHER_DONE would
  deadlock, since the verdict now waits for the oracle jsons); aborts without
  oracle_verdict.txt on any oracle arm/judge failure; writes the verdict only
  when all four policies are judged.

## Rejected suspects
- (a) group-kill of freshly launched arms: ruled out — arms ran 10-22 min and
  crashed with an in-process Python traceback, not a signal.
- (b) /data3 full: ruled out as crash cause (the IndexError is deterministic),
  but noted: /data3 is at 100% (44GB headroom). Inputs are on /dev/shm
  (all 10 npy present, 922GB free) — unaffected.
- (c) env/PYTHONPATH: launchers match the phase-1 known-good pattern.

## Verification
- 2-video 7B smoke (DECAF_B=1 deferred, subset of this run's anno): video 2
  encodes + answers past the point that previously crashed (see recovery agent
  handoff / EXPERIMENT_LOG).

## Store precision note (resolved)
launch_arms.sh runs the answering arms with DECAF_QUANT=0 (fp16 store) --
DELIBERATE: TB6 quant answer-equivalence FAILED (1/5 identical at 0.5B,
~12% tensor rel err), so per the standing log the main arms run fp16 and 4-bit
stays an option. config.json's "DECAF_QUANT": 1 arm entries are STALE
(pre-TB6); the launchers are authoritative.
