# E3 Phase 0 — Infrastructure report (DECAF)

**Date:** 2026-09-27 · **Run dir:** `/data3/zhuotaotian2_e2/runs/20260927_0000_e3_phase0/`
**Plan:** `draft-plan/beat-mukv.md` §5 Phase 0 · **Gate 0:** MuKV reproduces published RVS-Ego behavior within harness tolerance.
**Session status at handoff (02:00 CST):** P0-3 and P0-4 complete/in-flight; **P0-1/P0-2 BLOCKED — V100 server down since ~01:00 CST** (only source of the MuKV repo).

## Gate 0 (MuKV on Ada) — REVISED VERDICT: PASS (2026-09-27, post-diagnosis)

Full diagnosis: `deliverables/e3_phase0/DIAGNOSIS.md`. Summary:

- **Root cause of the original FAIL (46.0/43.4): config fidelity.** The original run used
  the argparse defaults of `scripts/run_mukv_rvs_ego.py` (keep 0.70/0.50/0.30 per grain,
  weight-proportional retrieval split, rerank λ 0.5/0.6, fft=diff) — not the paper config.
  The paper's RVS-Ego config (§4.1, suppl. Tab 10/11; mirrored by
  `scripts/sh/run_mukv_rvs_ego.sh`) is retention **0.1/0.1/0.8** (segment-dominant),
  per-grain topk **20/32/12**, rerank λ **0.3/0.3**, fft=**fft**. Our defaults were nearly
  the inverse keep profile; the paper's own sensitivity table shows the patch-heavy end is
  their worst setting.
- **Re-runs (paper-faithful config, same code/env/judge):**

  | Run | RVS-Ego Acc | Score | Parse fail | Band | Result |
  |---|---|---|---|---|---|
  | MuKV-7B, runner defaults (original) | 46.0 | 2.49 | 0 | [47.5, 53.5] | FAIL (superseded) |
  | MuKV-0.5B, runner defaults (original) | 43.4 | 2.26 | 0 | [45.9, 51.9] | FAIL (superseded) |
  | **MuKV-7B, paper config** | **50.9** | **2.67** | 0 | [47.5, 53.5] | **PASS** |
  | **MuKV-0.5B, paper config** | **48.0** | **2.45** | 0 | [45.9, 51.9] | **PASS** |
  | MuKV-7B, keep 0.9/0.9/0.9 (code-path sanity) | 51.2 | 2.67 | 0 | — | see below |

- **Code path sound:** with near-full retention (0.9/0.9/0.9) MuKV-7B scores 51.2 —
  ReKV-with-offload-level (54.1 in-harness) and ≈ the paper-config run (50.9). The
  sdpa/torch-fallback + fp32-upcast path does NOT break MuKV's DCP scoring; native
  flash-attn env not required for Gate 0.
- **Answer style ruled out:** MuKV-7B answers vs ReKV-7B on 1429 shared questions —
  coherent, 25.0 vs 25.8 mean words, not degenerate. The original deficit was store
  content, not generation style or judge interaction.
- **In-harness ordering preserved:** MuKV-7B 50.9 < ReKV-7B 54.1, matching published
  ordering (MuKV 59.5 < ReKV 63.7; MuKV's edge is memory, not raw RVS accuracy).
- **Faithful MuKV baselines going forward (C2/C3):** RVS-Ego, Qwen2.5-72B harness —
  **0.5B = 48.0 / 2.45, 7B = 50.9 / 2.67** (n=1465, 0 parse failures).
- **Action item:** use the paper config (now encoded in
  `runs/20260927_0000_e3_phase0/run_mukv_answer_v2.sh`) for ALL future MuKV runs; the
  runner-script defaults in `scripts/run_mukv_rvs_ego.py` are a decoy and must not be
  used for published-number comparisons.

### Superseded: original Gate-0 FAIL record (2026-09-27 morning, runner-default config)

- **Code landed from GitHub directly** (V100 never returned): `code/MuKV` @
  `2127b875ea1c2d91e4349324edbd779168a3da9e` (PROVENANCE.txt).
- **Env/patches:** answering in `/data4/rekv` (torch 2.6.0+cu124, tf 4.46.3) with
  `mukv_v100_sdpa.patch` (fattn=False, attn_implementation=sdpa) + fp32-upcast
  `torch_impl.py` (copied from patched ReKV — identical pre-patch) + npy identity-load
  branch. MuKV native env (torch 2.8.0/tf@66bc4de/flash-attn) NOT built. Provenance:
  `runs/20260927_0000_e3_phase0/code_state/MUKV_CODE_STATE.txt`.
- **Deviation (documented):** decord threaded decoder broken on Ada (threads>=4 crash;
  <=2 no speedup; 13 min/video 1080p VP9). All 10 videos pre-decoded to
  `/dev/shm/e3_mukv_npy/*.npy` at the exact mp4-path frame indices (byte-identical;
  re-decode via `runs/20260927_0000_e3_phase0/decode_to_shm.sh` after any reboot).
- **Smoke (0.5B, 2 Q):** PASS 04:55 — load/stream/retrieve/answer/CSV/memory stats all OK.
- **Running (detached):** 0.5B on GPU0 (ETA ~07:15), 7B on GPU1 (ETA ~08:00), README
  paper config; watcher auto-judges with 72B TP=4 (gpu_mem 0.80) then runs
  `gate0_verdict.py` (~08:45). Status: `runs/20260927_0000_e3_phase0/judge.status`,
  verdict -> `gate0_verdict.txt`.
- **Verdict at the time: FAIL** (0.5B 43.4 / 7B 46.0, judged 07:28/07:39 CST;
  `gate0_verdict.txt`). **Superseded** by the revised PASS above — root cause was the
  runner-default config, not the code or judge (see DIAGNOSIS.md). Bands: 0.5B Ego
  [45.9, 51.9], 7B Ego [47.5, 53.5], 0 parse failures required.

### Superseded: blocked-on-V100 staging notes (kept for provenance)



- **Published targets** (MuKV paper, CVPR 2026, main table, GPT-judged):
  | model | RVS-Ego Acc | RVS-Movie Acc |
  |---|---|---|
  | MuKV 0.5B (LLaVA-OV-0.5B) | 57.9 | 45.2 |
  | MuKV 7B (LLaVA-OV-7B) | 59.5 | 48.5 |
- **Pass bands** (W2.5 offset −9 Acc ±3; 0 parse failures required):
  0.5B Ego **[45.9, 51.9]** · 7B Ego **[47.5, 53.5]**
- **Verdict: not yet runnable — blocked on code transfer.**
- **Watcher:** `code/scripts_e3/pull_v100_mukv.sh` (pid 1492907, detached) polls V100
  every 60 s; on recovery it rsyncs `~/e1/MuKV` → `code/MuKV` (excl. weights) and records
  git provenance to `logs/mukv_provenance.txt`. Check `logs/v100_pull.status` for `DONE`.
- **Runbook for the moment it lands:** `code/scripts_e3/MUKV_RUNBOOK.md` (mirrored to Mac) —
  patch steps (mukv sdpa + fp32-upcast hunk), env choice, smoke test, detached answering
  (0.5B GPU0 / 7B GPU1, ETA ~2h / ~3.3h from W2.5 ReKV timings), judging, Gate-0 comparator.
- **Pre-verified inputs:** judge snapshot complete (37/37 shards, 145.4GB); both LLaVA-OV
  models in `code/ReKV/model_zoo/`; RVS-Ego 10 videos + ego4d_oe.json at
  `code/ReKV/data/rvs/ego/`; answering stack = `rekv` env (torch 2.6.0, tf 4.46.3);
  comparator `code/scripts_e3/gate0_verdict.py`.

## Datasets — manifest: DATA_MANIFEST.md (status at 02:00 CST)

| Dataset | Size | Location | Status |
|---|---|---|---|
| StreamingBench code+QA | 26 MB | /data2/zhuotaotian2_e2_data/streamingbench/ | DONE (extracted) |
| StreamingBench videos (HF mjuicem/StreamingBench) | TBD | /data2/.../streamingbench/videos | QUEUED (phase 2, auto after phase 1) |
| OVO-Bench src videos (46.3 GB of 199.6 GB) | 42/46 GB | /data1/zhuotaotian2_e2_data/ovo_bench/ | DOWNLOADING (5/6 files, last part in .incomplete) |
| OVO-Bench chunked videos (156 GB) | — | — | DEFERRED (disk; can chunk locally from src) |
| MLVU test (75.3 GB) | — | /data2/zhuotaotian2_e2_data/mlvu/ | QUEUED (phase 1 step 3) |
| Video-MME | meta ~10 MB / 101 GB | /data2/zhuotaotian2_e2_data/videomme/ | PARTIAL; videos auto-gated on free disk in phase 2 |

Downloaders detached: phase 1 pid 1494858, phase 2 pid 1505008; status files in
`runs/20260927_0000_e3_phase0/logs/datasets{,_p2}.status`. Disk at 02:00: /data1 30G
free (shrinking — other tenants), /data2 120G free.

## DECAF skeleton — DONE (`code/decaf/`)

- Fresh git repo, commits `57ae130`, `98a971a`. Host = fork of ReKV @1fd9a3d + Ada patch
  set (exact state `docs/rekv_host_code_state.patch`).
- W2.1 verification kit imported (`verification/encode_kv.py` pre-RoPE capture + rotary
  reproduction / token-alignment trap checks; `w21_analyze.py`).
- README: Component A (position-free store + sidecar + rotate-on-fetch), B (deferred
  commitment), C (evidence stratum) as implementation TODOs with gate criteria.
  **No mechanism code**, per plan.

## Environments

- `e3dekv` created (clone of e2judge: vllm 0.7.3, torch 2.5.1, transformers 4.49.0) —
  reserved for judging / DECAF dev.
- **Deviation:** MuKV answering will use the existing `rekv` env (torch 2.6.0,
  transformers 4.46.3): the answering code path needs transformers 4.46.3, which
  conflicts with vllm 0.7.3 (>=4.48.2) in e3dekv.

## Deviations / blockers

1. **V100 down since ~01:00 CST** (ssh connect-timeout from both Ada and Mac; ping
   100% loss) → P0-1/P0-2 blocked; watcher + runbook in place.
2. e3dekv cannot host both judging and answering (vllm↔transformers pin conflict);
   answering stays on the validated rekv env.
3. OVO-Bench chunked videos deferred (156 GB vs ~150 GB total free at session start).
4. Briefing quoted MuKV 0.5B as "≈56.5/3.9-ish"; paper main table = **57.9 Ego /
   45.2 Movie** (56.5/46.0 is the granularity-ablation row). Bands computed from
   main-table numbers.

## Next actions (any agent, in order)

1. `tail -2 runs/20260927_0000_e3_phase0/logs/v100_pull.status` — when `DONE`, follow
   `code/scripts_e3/MUKV_RUNBOOK.md` (kill watcher first: pid 1492907).
2. When OVO finishes, phase 1 continues to MLVU automatically; phase 2 then pulls
   StreamingBench videos and Video-MME videos if disk allows.
3. ~~After MuKV answering + judging: gate0_verdict, update this report~~ **DONE —
   Gate 0 PASS (revised)**, see the Gate-0 section and DIAGNOSIS.md above. Phase 0
   remaining: dataset downloads (items 1–2) and RVS-Movie / StreamingBench MuKV
   baselines using `run_mukv_answer_v2.sh` (paper config).
