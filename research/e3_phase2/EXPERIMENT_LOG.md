# EXPERIMENT LOG — /data3/zhuotaotian2_e2 (Ada box, GPUs 0-3 only)

## 2026-09-26 — W2.1 (THE GATE) + W2.3
- Re-encoded 9 videos (3 UCF static / 3 UCF dynamic / 3 RVS-Ego) with LLaVA-OneVision-0.5B (ReKV path) @0.5fps, fp16+SDPA (matches ReKV validated config; measured raw rate 4.34 GB/h ≈ paper 4.0). KV dumps: runs/20260926_1430_w21_encode/kv/ (33GB, 24 layers x {K_pre,K_post,V} per frame).
- Trap (a) RoPE: pre-RoPE K capture validated exactly (model rotary_emb(K_pre) == cached K_post, rel err 0.0); post-RoPE residuals inflated up to 73x. Trap (b): 196 tok/frame stable everywhere, diagonal dominance 1.3-3.9.
- RESULT W2.1: FAIL. Static residual energy ratio K=0.249 (L0 0.005, mid-layers 0.15-0.53), V=0.537. Compression at MSE-matched error: static 1.62 GB/h vs MuKV 0.91 -> 0.56x (need 3x / 0.30 GB/h). Video KV not differentially compressible; root cause: stored KV is a function of the whole prefix (RoPE position drift), not frame content. Only L0-L2 K is frozen (0.005-0.09). Direction dead per work2.md.
- RESULT W2.3: FAIL. GOP static/dynamic mean ratio 0.74-1.11 across theta sweep 780-72.7k; no content adaptivity. Event-boundary bonus: enrichment ~1.0-1.3, no alignment at meaningful GOP lengths. UCF GT spans pulled from V100 server -> data/ucf_event_spans.json.
- Deliverables: deliverables/w21_kv_residual/, deliverables/w23_gop/. GPUs 0-3 free after 15:45 CST.
- NOTE for W2.5 agent: Qwen2.5-72B hf download failed 2026-09-26 ~14:33 (401 from cas-server.xethub.hf.co); see logs/w25_download_72b.log.

## 2026-09-26 — W2.5 judge harness (RUN: runs/20260926_1455_w25_judge_harness)
- Judge: Qwen2.5-72B-Instruct bf16, vLLM 0.7.3 TP=4, exact ReKV eval_open_ended.py prompt. Answers: ReKV-0.5B and ReKV-7B, validated ported V100 config, full RVS-Ego 1465Q.
- RESULT: 0.5B 45.5 Acc / 2.36 Score (published 54.7/3.9, d=-9.2/-1.54). 7B 54.1/2.82 (published 63.7/4.0, d=-9.6/-1.18). Ranking + model gap reproduced (8.6 vs 9.0).
- Inter-judge agreement (Qwen2.5-72B vs Qwen1.5-14B-Chat, 50Q fixed prefix): 94.0% pred agreement, score MAE 0.60.
- VERDICT: UNVALIDATED vs published (offset ~9-10 Acc pts, systematic judge-leniency difference vs GPT). Harness usable for internal A/B; absolute numbers not citable as field-comparable.
- Blockers: V100 spot-check (no v100 key); Qwen3-14B unsupported by vllm 0.7.3 (used Qwen1.5-14B-Chat).
- Deliverables: deliverables/w25_judge_harness/ (REPORT.md, code, judged jsons, metrics.json).

## 2026-09-26 — W2.5 (judge harness) — COMPLETE, judge UNVALIDATED vs published
- Qwen2.5-72B-Instruct bf16 download: 3 failed attempts (xethub 401; hf.co unreachable; hf-mirror ReadTimeout at 93GB) -> attempt 4 via hf-mirror loop script (dl72b_loop.sh, HF_HUB_DOWNLOAD_TIMEOUT=60) COMPLETED 20:00 CST; 37/37 shards, 145.4GB. 32B fallback NOT needed.
- Answering: ReKV-0.5B (GPU0, 17:11-19:11) and ReKV-7B (GPU1, 17:11-20:29) on all 10 RVS-Ego videos, 1465/1465 questions each, 0 empty preds, exit 0.
- Grading: Qwen2.5-72B vLLM TP=4, verbatim eval_open_ended.py prompt, temp 0. 0.5B: Acc 45.5 / Score 2.36. 7B: first attempt OOM-killed (exit 137, cudagraph capture, other tenant 5GB on GPUs 2/3, gpu_mem 0.85); killed our orphaned TP workers, reran at 0.80 -> Acc 54.1 / Score 2.82.
- VALIDATION vs published (54.7/3.9, 63.7/4.0): FAILED the +/-2 pt band — delta -9.2 Acc / -1.54 Score (0.5B), -9.6 / -1.18 (7B). Systematic offset; ordering reproduces (+8.6 vs +9.0). Judge substitution (gpt-3.5-turbo -> Qwen2.5-72B) unvalidated for absolute claims.
- Inter-judge (50-Q prefix, 0.5B): 72B vs Qwen1.5-14B-Chat agree 94.0%, score MAE 0.60 (72B 48.0/2.32, 14B 50.0/2.80).
- Verdict: absolute numbers NOT citable vs published; relative within-harness comparisons citable (same judge, full coverage, 0 parse failures). Report: deliverables/w25_judge_harness/REPORT.md. GPUs 0-3 FREE as of ~21:10 (14B judge and 72B judge exited cleanly).

## 2026-09-27 — E3 Phase 0 (DECAF infrastructure; RUN: runs/20260927_0000_e3_phase0)
- Gate 0 targets locked from MuKV paper main table (GPT-judged): MuKV-0.5B RVS-Ego 57.9 / RVS-Movie 45.2; MuKV-7B 59.5 / 48.5. Offset-adjusted bands (published -9 +/-3, W2.5): 0.5B [45.9,51.9], 7B [47.5,53.5].
- V100 server DOWN since ~01:00 CST (ssh connect timeout from Ada and Mac; Mac lacks v100_key). Detached watcher pid 1492907 (code/scripts_e3/pull_v100_mukv.sh) retries every 60s; on success rsyncs ~/e1/MuKV -> code/MuKV (excl. model_zoo/*.safetensors) + records git provenance + refreshes code/patches/.
- Datasets (detached pid 1494856, phase1; pid 1505006 phase2): StreamingBench repo tarball DONE (codeload, /data2/.../streamingbench); OVO-Bench src_videos 46.3GB downloading to /data1 (chunked_videos 156GB DEFERRED - disk); MLVU_Test 75.3GB queued -> /data2; Video-MME metadata only now, videos gated on free disk. Sizes: OVO 200GB total, Video-MME 101GB, MLVU_Test 75.3GB.
- Env: e3dekv = clone of e2judge (vllm 0.7.3, torch 2.5.1, transformers 4.49.0) at /data4/e3dekv. Answering stack needs transformers==4.46.3 which CONFLICTS with vllm 0.7.3 (>=4.48.2) -> MuKV answering will run in existing rekv env (torch 2.6.0, tf 4.46.3, W2.5-validated); e3dekv reserved for judging/DECAF dev.
- DECAF skeleton: code/decaf/ = fresh git repo, fork of Ada ReKV @1fd9a3d+patches (host code state: docs/rekv_host_code_state.patch) + W2.1 verification kit imported (verification/encode_kv.py pre-RoPE capture + trap checks, w21_analyze.py). README states Component A/B/C design as implementation TODOs. Commits 57ae130, 98a971a. No mechanism code.
- GPU: 0-3 FREE (4-7 other tenant vLLM untouched). GPUs reserved for MuKV answering once code lands.
- 02:00 CST SESSION HANDOFF: V100 still down (1h). OVO src videos 42/46GB (5/6 files). Skeleton+envs+runbook+comparator staged. Detached and running: v100 watcher (pid 1492907), dataset phase1 (1494858), phase2 (1505008). Next agent: follow code/scripts_e3/MUKV_RUNBOOK.md once watcher reports DONE.


## 2026-09-27 — E3 Gate 0 execution (MuKV on Ada, direct-from-GitHub)
- **Code:** V100 never came back; MuKV obtained on Ada directly from GitHub: code/MuKV @ 2127b875ea1c2d91e4349324edbd779168a3da9e (PROVENANCE.txt, zip download). Old V100 watcher already dead.
- **Env:** answering in /data4/rekv (torch 2.6.0+cu124, tf 4.46.3) — MuKV native env (torch 2.8.0 + tf@66bc4de + flash-attn 2.8.3) NOT built; sdpa/torch fallback per runbook. Patches: mukv_v100_sdpa.patch + fp32-upcast torch_impl.py (copied from patched ReKV; files identical pre-patch) + decord threads edit. Full provenance: runs/20260927_0000_e3_phase0/code_state/MUKV_CODE_STATE.txt.
- **Decord blocker + fix:** decord threaded decoder broken on this box (threads>=4: run_.load() assert; threads 0/2 work but NO speedup — 13 min per 1080p VP9 video; ffmpeg same speed). Fix: pre-decoded all 10 RVS-Ego videos to /dev/shm/e3_mukv_npy/*.npy (10 parallel procs, 13 min total; exact mp4-path frame indices, byte-identical) + patched npy branch to identity-load. NOTE: shm contents vanish on reboot — re-decode with runs/20260927_0000_e3_phase0/decode_to_shm.sh if needed.
- **Smoke:** 0.5B 2-question end-to-end PASS (04:55): model load, streaming encode, retrieval, generation; CSV schema OK; memory_stats OK (peak 5.4GB, kv 513MB).
- **Answering (detached, README paper config):** sample_fps 0.5, n_local 15000, retrieve 64, grains 49/196/784 (default topks), keep_ratio 0.70/0.50/0.30, importance attention_fft_weighted, fft diff, rerank on (a0.5 b0.6 n5). 0.5B GPU0 started 05:46 (ETA ~07:15); 7B GPU1 (ETA ~08:00). CSVs -> runs/20260927_0000_e3_phase0/answers_{05b,7b}/results.csv.
- **Judge watcher (detached):** runs/20260927_0000_e3_phase0/run_judge_when_ready.sh — waits for both EXITs + 1465-row check, waits for GPUs 0-3 free, then Qwen2.5-72B TP=4 gpu_mem 0.80 (w25_judge.py, verbatim ReKV eval prompt) x2, then gate0_verdict.py. Status: judge.status; verdict: gate0_verdict.txt. Expected ~08:45 CST.
- **GPU state:** 0,1 BUSY (MuKV answering); 2,3 FREE; 4-7 other tenant untouched.

| 2026-09-27 09:15 | E3 Gate 0 diag | ROOT CAUSE FOUND: config fidelity, not code path. Original Gate-0 run used run_mukv_rvs_ego.py DEFAULTS (keep 0.70/0.50/0.30 per grain 49/196/784, no --granularity_topks -> weight-proportional split, rerank a0.5/b0.6, fft_method=diff). Paper RVS-Ego config (4.1 + suppl Tab 10) is retention rho=(0.1,0.1,0.8) patch/frame/SEGMENT (segment-dominant), per-grain topk (20,32,12)=64, rerank lambda=(0.3,0.3,0) top_n=5, alpha=(0.5,0.7,0.8); repo scripts/sh/run_mukv_rvs_ego.sh matches the paper. Our keep profile was near-inverse of paper (Tab 10: rho=(0.8,0.1,0.1) is the WORST setting, 54.7). Answer-style check: MuKV-7B answers coherent, same length as ReKV (25.0 vs 25.8 words), not degenerate -> not a code bug. RE-RUNS (detached): GPU0 7B paper config (pid 1803790), GPU1 7B keep 0.9/0.9/0.9 code-path sanity (pid 1803791), GPU2 0.5B paper config (pid 1803792); judge watcher pid 1804886 (72B TP=4 after answering). | 0,1,2 | 3 FREE; 4-7 other tenant |

| 2026-09-27 11:35 | E3 Gate 0 diag | RE-RUNS COMPLETE. Judged (72B harness, RVS-Ego, n=1465, 0 parse failures all): 7B paper config = 50.9 Acc / 2.67 Score (band [47.5,53.5] PASS, expected ~50.5); 0.5B paper config = 48.0 / 2.45 (band [45.9,51.9] PASS); 7B keep9 sanity (0.9/0.9/0.9 minimal pruning) = 51.2 / 2.67 -> ReKV-level, sdpa/fp32-upcast code path SOUND, native flash-attn env not needed. Original runner-default runs (43.4/46.0) kept as superseded. Gate 0 REVISED: PASS. Faithful baselines: 0.5B 48.0, 7B 50.9. DIAGNOSIS.md + REPORT.md updated, mirrored to Mac. Future MuKV runs MUST use run_mukv_answer_v2.sh (paper config). | none | all 0-3 FREE; 4-7 other tenant |

## 2026-09-27 — E3 Phase 1 (Component A: position-disentangled store; RUN: runs/20260927_1200_e3_phase1_compA)
- **Code:** code/decaf commit 5babc2f. Env-gated DECAF_POSITION_FREE=1: position sidecar (per-block absolute start, asserted == n_init + b*196), rotate-on-fetch to EXACT positions (retrieved blocks from sidecar; question/decode at true streaming positions via PositionalKV), retrieval scoring unchanged (stock already uses pre-RoPE content keys — audit finding, T1-verified), per-question retrieval jsonl logging, DECAF_RECALL_ONLY mode. CRITICAL: PYTHONPATH=code/decaf required (rekv env editable-installs code/ReKV); longva editable install was MISSING from /data4/rekv (reinstalled from code/decaf/model/longva) — without it ANY stream-VQA run dies on import.
- **Trap checks (verification/decaf_trapcheck.py):** T1 store-is-content: stored K/V == raw k_proj/v_proj outputs, rel err **0.0 at layers 0/5/11/17/23, 96 blocks** (stock ReKV CPU store is pre-RoPE — the 'post-RoPE store' premise of the briefing does not hold for ReKV; its entanglement is rank-based position reassignment at fetch). T2c sidecar identity exact. T3 rotate-on-fetch plumbing PASS (buffer positions exact, next_pos == store length, per-layer 64-block retrieval over 440-block store). T3d 2-question greedy smoke fluent. T2 rotary-vs-W2.1-dump cross-check rerunning (encode2). T5 stock-equivalence + T3c encode-invariance pending in same rerun.
- **Runs launched (detached) ~13:12-13:24:** 7B position-free answering (GPU1), 0.5B position-free answering (GPU0), 7B entangled RECALL-ONLY (GPU2), 0.5B entangled RECALL-ONLY (GPU3). Same config as W2.5 (fps 0.5, n_local 15000, retrieve 64, greedy, seed 2024). Entangled ACC baseline = W2.5 judged jsons (0.5B 45.5/2.36, 7B 54.1/2.82).
- **Watcher:** run_gate1_watcher.sh (detached) — waits for 4 EXITs, judges DECAF CSVs with Qwen2.5-72B TP=4 (gpu_mem 0.80) when GPUs free, computes temporal-oracle recall (ego4d_oe [start,end] windows at 0.5 fps, strict + +-30s variants) via verification/compute_recall.py, writes gate1_verdict.txt. Status: gate1_watcher.status.
- Gate 1 criteria: recall +>=5 pts OR Acc +>=2 pts (72B-judged, equal memory) vs entangled.

| 2026-09-27 18:50 | E3 Phase 1 recovery | "4 dead jobs" forensics: FALSE ALARM — all four measurement runs completed cleanly (EXIT 0, 1465/1465 rows each, 10/10 videos); EXIT lines landed in code/decaf/*.status because the launcher passed relative status paths after the runner cd'd; driver logs empty by design (output -> answer.log). Old gate1 watcher PID 1979329 was looping on never-appearing run-dir status files, killed. Validity verified instead of relaunching: decaf-vs-W2.5 answer match only 4.5%/8.0% (0.5B/7B) proves position-free path was active (flag-off is bit-stock per T5); config == W2.5. Caveat: 100 (0.5B) / 18 (7B) empty preds, concentrated in video cbfeb6c8. ENCODE4 exit-1 root cause = verdict-aggregation crash in decaf_trapcheck.py:215 (float values vs startswith), NOT a mechanism bug — fixed (key-filter), rerun ENCODE4_EXIT 0 / COMPARE2_EXIT 0, T3c encode-invariance bit-exact, ALL 7 TRAP GATES PASS / OVERALL PASS. Fresh gate1 judge watcher launched (run_gate1_judge.sh, setsid, PID 2215640): 72B TP=4 judging of both decaf CSVs + temporal-oracle recall x4 + gate1_verdict.txt/.json. | none (judging 0-3) | 4-7 other tenant |
| 2026-09-27 22:05 | E3 Phase 1 Gate-1 corrected | Implemented DECAF_FETCH_RANK=1 (rank-remapped buffer fetch positions, stock ReKV convention; store/content-retrieval/sidecar unchanged; exact path kept). Commits fe7af47 + 8ffe916 + 0277a19 on top of ca7b9df. Trapcheck mode=rankqa: TR1 rank plumbing PASS, TR2 LATE-stream stock equivalence 10/10 answers CHARACTER-IDENTICAL to W2.5 stock with the full 420-frame store (82,333 tokens, the exact-mode failure position) — the 22.3-collapse failure mode is fixed at trap scale. OVERALL PASS (RANKQA_EXIT 0 21:59:40, run dir runs/20260927_2200_e3_phase1_compA_rankfetch). 7B answering DETACHED on GPU1 (started 21:50, ETA ~01:40), judge watcher (72B TP=4 0.80) armed -> rank_verdict.txt vs stock 54.1/2.82, PASS band +/-1.5 pts. | 1 (answering), 3 (trapcheck done) | 0,2,3 FREE; 4-7 other tenant |

## 2026-09-28 E3 Phase 2 — Component B (multi-grain deferred commitment) STARTED
- Run dir: runs/20260928_0315_e3_phase2_compB; code/decaf @5932ee0.
- Built on corrected-A (rank fetch, Gate1 PASS 54.1/2.82).
- Impl (model/attention/decaf_b.py + kv_cache_manager + llava_onevision_rekv):
  multi-grain store (patch49/frame196/segment784 per 8s segment, MuKV set),
  MuKV-style dual-signal write-time scoring (received attention + FFT high-freq
  of content keys) with online per-position detrend (debias arm = C1 evidence),
  hedged superset retention (union of per-signal keep-lists, realized frame-byte
  tax) vs MuKV-style write-time prune at the same tax, cross-grain consistency
  rerank (gamma=0.5), uncertainty-triggered coarse->fine re-commitment
  (entropy>1.8 nats or degenerate), 4-bit per-channel grain storage (DECAF_QUANT).
- Traps (verification/decaf_b_trapcheck.py, 0.5B GPU3):
  TB1 assembly/accounting PASS (after fix: token_start-keyed frame assembly —
  layers offload block-major; realized frame-byte tax — patch-spread inflation).
  TB2 4-bit roundtrip PASS (per-channel int4 rel err ~0.12; threshold is a
  bug-gate 0.30; metric-level near-losslessness is TB6 per W2.2).
  TB5 bias reduced PASS: mean |attn residual vs position| raw 0.412 -> debiased
  0.330 (0.5B, 96 blocks; 7B full-stream numbers from analyze arm).
  TB3 commit plumbing PASS (contiguous rank positions, slots<=topk, plan log).
  TB4 answering smoke PASS (10/10 fluent on full 420-frame store; 3/10 == stock
  answers - expected, tax-0.5 store differs from stock; mechanism works).
  TB6 quant answer equivalence: PENDING.
- Arms (7B): GPU0 deferred(debias,tax0.5,q4,pass2 on), GPU1 writetime(debias,
  tax0.5,q4,single-pass), GPU2 writetime(raw bias, tax0.5,q4) [revised Gate 1
  pair]; GPU3 analyze (encode-only bias curves + GB/h bytes) then free.
- Judge watcher: run_judge_phase2.sh (72B TP=4 after all arms; Gate 2 part 1 +
  revised Gate 1 verdict). Oracle grain-policy runs (clairvoyant bound):
  PENDING (scripts to be added by next session).
- TB6 quant answer equivalence: FAIL (1/5 identical 0.5B, per-channel int4 K+V,
  ~12% tensor rel err) -> main arms run fp16 store (DECAF_QUANT=0); 4-bit kept
  as option, needs group/token-scale quant (KIVI-style) before C3 claims. Logged
  as a result per standing rules; contradicts naive reading of W2.2 (which
  measured metric-level effects in the W2.2 setup, not answer-invariance here).
- PERF BUG + FIX: per-block .cpu() syncs made DECAF-B encode CPU-bound (42k
  syncs/128 frames 0.5B; 7B stuck >9 min on first chunk). Batched once-per-
  forward offload (4 bulk copies/layer/forward): 7B 128 frames 31s. First arm
  launch (03:30) killed; relaunched 03:48 with fp16 store.
- INFRA BUG (not method): /data3 100% full + pathological seek pattern in the
  2.2GB 9198b9a4 mp4 -> decord re-read 640GB/video under cache pressure; all
  four 7B arms stuck in video loading (GPU 0%). Standalone repro >120s. Fix:
  switched arms to the Phase-0 pre-decoded npy in /dev/shm/e3_mukv_npy (all 10
  present; anno copy ego4d_oe_npy.json in the run dir). Caveat logged: npy
  linspace subsample != mp4 stride sampling -> slight input shift vs the mp4-
  based 54.1 stock anchor; within-arm ablations (Gate 2) unaffected.
- Arms relaunched 04:11 on npy; writetime 27 commits / deferred 9 / rawbias 9
  in the first minutes (expected: writetime single-pass is fastest; deferred
  pays pass-2 on uncertain questions). Judge + oracle watchers armed.
- Watchers restarted 04:17 (first judge watcher aborted on stale EXIT 137 left
  by the killed first launch - the run_arm.sh wrapper survives pkill -f
  rekv_stream_vqa; do NOT pkill arm pythons, kill process groups instead).
  Both watchers (judge + oracle chain) confirmed alive. Subset anno repointed
  to npy. Steady state ~04:20: writetime ~6.6 q/min -> ETA ~3.7h/arm; deferred
  slower (pass-2); analyze finished video 1 (420 blocks logged).

## 2026-09-28 12:50 CST — E3 Phase 2 compB: arm crash recovery (agent-34)
- CRASH ROOT CAUSE (all 8 arms EXIT 1 on video 2): Component B layer indexing
  bug. `_LAYER_CTR` (creation-order layer id, kv_cache_manager.py) is only valid
  for the first video of a process; clear_cache()+encode_init_prompt() rebuild
  per-layer managers every video, so video 2's managers got layer_idx 28.. (7B,
  N_LAYERS=28) -> IndexError in decaf_b.add_frame_layer (fr.attn[:, 28], size 28).
  Trap checks missed it because verification/decaf_b_trapcheck.py runs ONE video
  per process; Phase 1 never exercised it (DECAF_B off). Side-effect fixed too:
  grain store singleton now reset per video (stale segments no longer leak into
  the next video's commits; token_start key collisions avoided).
- FIX: code/decaf @ac79712 — reset_layer_ctr() + decaf_b.reset_store() called
  from Abstract_ReKV.encode_init_prompt (per-video stream entry point, DECAF_B
  only). Verified with a 2-video 7B DECAF_B=1 smoke on the run's own anno subset:
  both videos complete with sane answers+retrieval (previously: crash on video 2
  block 0). Full diagnosis: runs/20260928_0315_e3_phase2_compB/DIAGNOSIS.md;
  crashed artifacts archived under crash_20260928_0442/.
- VERDICT LOGIC FIX: run_judge_phase2.sh v2 + oracle_chain.sh v2 — no more
  MISSING/PENDING verdicts. Watcher aborts (no verdict file) on any arm/judge
  failure; writes ARMS_JUDGED marker; writes phase2_verdict.txt ONLY when all 3
  arm + 4 oracle judged jsons exist (Gate 2 part 2 = clairvoyant gap closure
  needs oracle). Oracle chain waits on ARMS_JUDGED (waiting on WATCHER_DONE
  would deadlock). Crashed-run MISSING verdicts deleted with the archive.
- Note: DECAF_QUANT=0 in launchers is DELIBERATE (TB6 failed 1/5; see earlier
  entry); config.json arm entries saying q4 are stale.
- RELAUNCHED 12:49 CST, all detached (setsid nohup): deferred GPU0, writetime
  GPU1, rawbias GPU2, analyze GPU3; judge watcher pid 2765831; oracle chain pid
  2765832. Prior steady-state rate ~6.6 q/min -> arms ETA ~16:30-17:30 CST,
  then 72B TP=4 judging of 3 arms, then 4 oracle policy arms (2-video subset,
  316 q) + their judging; verdicts follow automatically.

## 2026-09-28 16:45 CST — E3 Gate 2 reassessment (agent, this session)
- GATE-2 PART 1 RESULT (1465q, 72B judge, 0 unparsed): deferred 51.8/2.745 vs
  writetime 51.1/2.717 -> +0.7 pts. FAIL (needs >=+2). writetime_rawbias 52.1/2.740
  -> debias has NO accuracy effect at this config (revised Gate 1 debias leg null).
- ORACLE CRASH ROOT CAUSE: plan() applied the slot-budget break only for
  policy==default; oracle seg policy (TAX=1.0) committed EVERY frame of EVERY
  segment (hundreds of slots vs topk=64), so st=init_len+s*196 ran past the
  materialization buffer (topk*196+n_init tokens) -> empty dest slice ->
  RuntimeError size a(0) vs b(196) in get_retrieved_kv. Oracle clairvoyance is
  in the ranking, not the budget.
- FIX: code/decaf @93014b3 — (1) decaf_b.plan(): budget break applies to ALL
  policies; (2) get_retrieved_kv(): skip grains whose slot falls past the
  buffer + bookkeeping .get guard (defensive). Empty commit (policy selects
  zero grains) now degrades to init-only context instead of crashing.
- PASS-2 DIAGNOSTIC (deferred commit_log, 1465q): entropy trigger fired on only
  157/1465 = 10.7% of questions; on those, deferred 62.4% == writetime 62.4%
  (score 3.153 vs 3.172) — ZERO gain where pass-2 fires. The +0.7 overall comes
  from single-pass questions (+0.6, noise level). Uncertainty-triggered
  refinement is not paying at threshold 1.8 nats.
- RELAUNCHED 16:42 CST: oracle chain v2 (4 policy arms on GPUs 0-3, seg/frame/
  patch/default, 316q 2-video subset, TAX=1.0, then sequential 72B judging ->
  oracle_verdict.txt). Sweep chain armed behind it (sweep_chain.sh, waits for
  ORACLE_CHAIN_DONE): wave1 = writetime+deferred at tax 0.25/0.75 (equal-tax
  Gate-2 pairs, GPUs 0-3); wave2 = deferred tax0.5 NOPASS2 (single-pass) +
  deferred tax0.5 ENTROPY=1.2 (threshold sensitivity); each wave 72B-judged.
  All detached setsid nohup; verdicts: oracle_verdict.txt / sweep_verdict.txt.
- ORACLE RESULTS (2-video subset, 316q, all judged 0 unparsed; oracle arms
  TAX=1.0 = full store, same 64-slot budget, clairvoyance in ranking only):
  seg 41.1/2.23 | frame 43.4/2.30 | patch 45.6/2.40 | default 41.1/2.23.
  SAME-subset actuals: deferred 41.1/2.212 | writetime 42.7/2.323 | rawbias 43.4/2.297.
  -> best-policy clairvoyant gap vs deferred = +4.4, vs writetime = +2.8;
     closure = -56% (deferred is FURTHER from clairvoyant than writetime).
  -> oracle default policy - deferred actual = 0.0: with the FULL store the
     planner does not beat the actual deferred system at all -- write-time
     retention (tax) is not the binding constraint; the query-time planner
     (similarity/rerank/commit) adds nothing even with every grain available.
  -> oracle best (patch) - writetime = +2.9: total query-aware headroom on this
     store/questions is ~3 pts at n=316 (se ~2.8) -- there is no >=2-pt
     clairvoyant margin for deferred to harvest on RVS-Ego.
  Realized store (fp16, tax 0.5): 5.44 GB/h vs MuKV class 0.91-1.23 GB/h
  (4.4-6x); tax sweep arms will map the accuracy-vs-tax curve for the Pareto.

## 2026-09-28 21:00 CST — E3 Gate 2 reassessment COMPLETE (this session)
- SWEEP (1465q, 72B judged, 0 unparsed): wt_tax025 51.8/2.79GB/h | def_tax025 35.9
  (INVALID: per-signal budget tax*392 < 196-token frame cost -> structurally empty store;
  deferred needs tax>=0.5 to keep one frame per signal) | wt 51.1 | def 51.8 (+0.8 FAIL)
  | wt_tax075 50.2/8.09 | def_tax075 51.8/5.44 (+1.6 FAIL). No config reaches +2.
- PASS-2: fires 10.8% (33.7% under starvation), zero gain where it fires (62.4==62.4).
- ORACLE (316q subset, full store, 64 slots): seg 41.1 | frame 43.4 | patch 45.6 |
  default 41.1 | PER-QUESTION-MAX 51.6. Same-subset actuals: deferred 41.1, writetime 42.7.
  Fixed-policy closure -56% (FAIL); per-query policy-selection headroom +8.9 over
  writetime / +10.5 over deferred (subset n=314, se~2.8, selection inflation ~+6 vs best
  single policy) -- headroom lives in per-query POLICY choice, which the default planner
  cannot access (oracle default == deferred actual = 0.0).
- VERDICT: Gate 2 FAIL at every valid config. Recommendation ADJUST: (1) no Component C
  on this evidence; (2) learned commitment scorer (C5) becomes core Phase-3 experiment;
  (3) same oracle protocol on StreamingBench/OVO (stop thesis if clairvoyant <2 there);
  (4) fix per-signal budget quantization before Phase 3.
- Deliverable: deliverables/e3_phase2/GATE2_ASSESSMENT.md. Wave-2 arms (nopass2, ent12)
  launched 20:39 CST, auto-judging to sweep_verdict.txt.
- INFRA NOTE: Ada network partitioned 18:35-20:09 and ~20:25-20:53 CST (host up 80d
  throughout; detached chains unaffected). sshd drops standalone connections -- use the
  ControlMaster mux.

## 2026-09-28 21:15 CST — E3 Phase 2 senior-review statistics (McNemar + learnability probe)
- McNemar (1465q paired, 72B judge pred, 0 excluded): deferred vs MuKV-paper
  +0.96pt, 149/135 discordant, p=0.44 -> "beats MuKV" NOT significant; use "matches".
  deferred vs writetime p=0.31 (Gate-2 indistinguishable CONFIRMED). deferred_tax075
  vs writetime_tax075 +1.64pt p=0.042 (only significant pair; favors deferred).
  deferred vs stock-ReKV-W2.5 -2.32pt p=0.056 (marginal, deficit real-ish but not
  ironclad). writetime-debias vs rawbias -1.02pt p=0.08 (debias null NOT rejected).
- DATA INTEGRITY FLAG: judged_answers_7b_b_deferred_tax075_72b.json and its
  commit_log.jsonl are byte-identical (md5) to the plain deferred run, despite
  sweep_chain.sh launching a real TAX=0.75 deferred arm. Either deferred commit is
  provably TAX-insensitive or the dir was back-filled by copy. Pair (c) therefore
  tested deferred vs writetime_tax075.
- Learnability probe (oracle 2-video subset, 316 records/314 uniq q): label balance
  patch 45.6/frame 43.4/seg=default 41.1; 48% questions all-policies-wrong; no policy
  ever sole winner. Held-out seed-42 split: LR +1.90pt (CI [0.00,4.43]) BUT 200x
  resplit mean -1.02+-1.75pt (26% positive) -> seed split is noise. Decision rule
  verdict: FAIL / ~0 -- per-query policy headroom NOT learnable from question type +
  length + store stats (entropy, committed slots/grains, bytes, pass1 policy).
  In-sample memorization ceiling +6.3pt over patch (absolute per-question cap);
  LR in-sample only +1.9pt. C5 scorer needs richer (retrieval-content) features.
- Deliverable: deliverables/e3_phase2/STATS.md (+ mcnemar_results.json,
  learnability_results.json, both analysis scripts). Mirrored to Mac
  research/e3_phase2/.

## 2026-09-28 21:11 — E3 Phase 2 cross-benchmark clairvoyant oracle (StreamingBench)
- Question: does the RVS-Ego clairvoyant margin (+8.9) replicate on StreamingBench? <2pts kills generality.
- Subset: questions_real_stream.json (real-time streaming MC split); 150/498 videos available offline (zips 1-50,101-150,201-250 unzipped to videos_rt/). Took 36 longest videos = 180 questions (5-6 q/video is the SB structure; deviation from RVS 2-long-video shape documented). Answers balanced A/B/C/D ~625 each in full split.
- Adaptations: MC options appended to question text, gold = letter; audio ignored (video-only pipeline); deterministic letter-match judge (xb_judge.py --mode letter) as PRIMARY (mirrors SB official eval, gives per-question correctness for true clairvoyant); 72B LLM MC-adapted judge as cross-check when GPUs 0-3 free.
- Arms: answers_7b_writetime (GPU2, writetime tax0.5 debias nopass2 = reference config); oracle_{seg,frame,patch,default} (deferred tax1.0, policy env; seg->frame GPU3 chain, patch->default GPU2 chain after system).
- code/decaf @93014b3 (oracle seg slot-budget fix). Launch bug fixed at 21:15: run_oracle_arm.sh self-backgrounded -> two arms collided on GPU3; frame killed, chains rewritten to foreground with skip-if-done guards.
- Watcher: xb_watcher.sh -> letter judge all 5 -> optional 72B xcheck -> xb_verdict.py (per-policy acc, system acc, per-question clairvoyant, margins) -> xbench_verdict.txt + clairvoyant_detail.json.
- 2026-09-28 INTEGRITY: deferred@tax0.75 arm files byte-identical to tax0.5 NOT due to copy — arm ran 17:31-19:52 GPU3 but TAX=0.75 is a no-op for deferred (per-signal budget 294 < 2-frame cost 392 @ decaf_b.py:271); row degenerate/duplicate, retract from sweep table; see runs/20260928_0315_e3_phase2_compB/INTEGRITY_tax075.md
## 2026-09-30 — W3.2 protocol reconciliation (RVS / SB / OVO oracle-gap nulls)
- Senior-reviewer flag: OVO printed p=0.9736 looked like a 97.4th-percentile tail flip vs RVS 1.06th /
  SB 12.6th. RESOLVED: no flip — all three are the same within-question exchangeability null
  (per-question permutation of 4 policy correctness labels; gap = oracle - best-fixed; 10k iters,
  seed 2024; one-sided upper tail). OVO's p=0.9736 means the observed gap sits at the 2.6th percentile
  (LOW tail), same direction as RVS/SB.
- Minor real deviations in OVO implementation, now removed: (1) correctness from regex letter
  extraction on raw CSVs instead of the judged_*_letter.json files; (2) no +1 correction on p;
  (3) RNG stream consumed by the random-selector sim first (exact-draw difference only).
- Unified recomputation from judged jsons (w32_oracle_null/w32_unified_null.py, deterministic, CPU):
  RVS  obs +6.33pt at 1.07th pct, p=0.9893 (z=-2.48); SB obs +5.00pt at 12.29th pct, p=0.8771
  (z=-0.79); OVO obs +4.44pt at 2.47th pct, p=0.9753 (z=-1.82). OVO judged-json numbers match the
  regex path (clairvoyant 45.6 both).
- Multiplicity over the 3 tests: Holm adj p = 1.0 for all; BH adj p = 0.9893 for all; NO rejection
  at alpha 0.05 (none survives uncorrected either, all raw p > 0.87).
- Verdict: no consistent harvestable policy-selection edge on any benchmark; oracle remains a
  bounding device. Writeup: analysis/w32_oracle_null/PROTOCOL_RECONCILIATION.md (mirrored to Mac
  research/e3_phase2/analysis/w32_oracle_null/).

## 2026-09-30 — W4.1 fps fix closed (true 0.5 fps) + W4.2 closed as underpowered-null
- W4.1: npy double-resample bug fixed (eeb7f60: NPY_FPS=0.5, consume at full rate, standing
  FPS_ASSERT logged per video to fps_assertion.jsonl — all 10 videos 0.5000-0.5002 eff fps).
  writetime tax0.5 debias rerun at TRUE 0.5 fps: 52.4 (n=1465). McNemar: vs MuKV paper 50.9
  p=0.203; vs stock ReKV 54.1 p=0.139; vs old 0.25fps-eff writetime 51.1 p=0.093.
  Run dir runs/20260930_0935_w41_fpsfix/ (mirrored: research/e3_phase2/w41_fpsfix/).
- Directive 2 (Pareto at corrected fps, bytes vs encode compute separate): store bytes DOUBLED
  with the fix — 10.38 GB/h vs 5.44 GB/h (same sweep methodology: per-video max
  store_bytes_total / 9.543 stream-h). Segments are 4-frame token segments; per-segment budget
  tax*784/2=196 tokens fixed -> segment count doubles with fps (225->450/video) -> store doubles.
  Corrected like-for-like ratio vs MuKV published 0.91-1.23 GB/h (their 0.5 fps): 8.4-11.4x
  (old 4.4-6.0x was our half-rate vs their full-rate, not like-for-like). Encode compute:
  861 -> 1105 GPU-s/stream-h total wall (+28%); encode-phase increment ~doubles (~140 -> ~280
  GPU-s/stream-h estimate), bought +1.3 pt (p=0.093, n.s.) and fps-matched MuKV anchor.
- W4.2 (coverage-spread, 83 temporal q, pre-registered MDE ~16-21 pt at n=83): hybrid vs top-k
  +6.0 pt (12.0 vs 6.0), p=0.180 (2/7/9 discordant), Holm 0.539 across arms. CLOSED as
  UNDERPOWERED-NULL per senior directive; dissection STOPPED (no effect established to explain;
  ~1.7sigma diffs, same shape as learnability split-selection noise). Scaling caveat logged:
  causal coverage question needs a substantially larger temporal-question set (no queued task).
  Dissection agent early material (debug_w42.py/json retrieval-set Jaccard; crashed stats_output.txt)
  recorded as superseded-by-directive in W4_FPS_AND_SPREAD.md.
- Directive 4 (corrected arm through W3.1 family): analysis/w41_family_correction/ — writetime@0.5fps
  replacing old writetime in family A (m=21) + family B filter-in (m=15). vs MuKV: raw 0.203 ->
  Holm 1.0 / BH 0.388. vs stock ReKV: raw 0.139 -> Holm 1.0 / BH 0.291. Nothing involving the
  corrected arm survives in any family/union; family B clean (min Holm 0.301). Survivors are
  pre-existing structural pairs only (stock>tax075 Holm 0.020; MuKV<stock BH 0.037;
  writetime05fps>tax075 BH 0.037 — W3.1 primary pair's corrected successor, BH only).
- Deliverables updated: deliverables/e3_phase2/W4_FPS_AND_SPREAD.md (close-out appended);
  analysis/w41_family_correction/ + .py. All mirrored to research/e3_phase2/.

## 2026-09-30 — W5.4 pooled coverage regression: temporal diagnosis REFUTED ON EVIDENCE, direction closed
- Pooled all 4 W4.2 arms (top-k + uniform/hybrid/stratified) x 83 temporal questions = 332 obs;
  regressed per-question correctness (72B judge) on achieved window coverage (W3.3 def, per-question
  per-arm from commit logs). CPU only; statsmodels installed into /data4/rekv. Analysis:
  analysis/w54_coverage_regression/ (w54_coverage_regression.py, dataset csv, results json).
- Key diagnostic first: arms spread retrieval CONTENT (Jaccard 0.18-0.29) but NOT coverage —
  within-question coverage sd 0.019 vs between-question 0.070; cross-arm coverage corr 0.94-0.99;
  only 27/83 questions have arm range >0.05. Pooled coverage band 0.18-0.49 (per-arm means 0.29-0.32).
- Results (LPM, cluster-robust by question, 83 clusters): A question-FE coef -0.33 CI [-1.99,+1.33]
  p=0.69; B arm-FE + window-length +0.56 CI [-0.54,+1.66] p=0.32; C qFE+armFE +0.40 CI [-1.33,+2.13]
  p=0.65; logit FE fails to converge (prevalence artifact). Binned curve NON-monotone (worst bin at
  cov 0.31-0.33, acc 1.8%). No spec shows a positive gradient; MDE this data supports ~11-17pt/0.1.
- Hybrid (+6.0pt vs top-k in W4.2) had coverage slightly LOWER than top-k — the one real accuracy
  movement ran against the diagnosed direction.
- VERDICT per pre-declared read: diagnosis REFUTED ON EVIDENCE (not on power — 4x obs + real
  within-question variation). Temporal direction CLOSED honestly. Caveat: conditional on the narrow
  achieved coverage band; no allocator in this family reaches high coverage at this store budget.
- Deliverable: deliverables/e3_phase2/W54_COVERAGE_REGRESSION.md (mirrored:
  research/e3_phase2/w54_coverage_regression/).
