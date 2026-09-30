# E3 Phase 0 — Infrastructure report (DECAF)

**Date:** 2026-09-27 · **Run dir:** `/data3/zhuotaotian2_e2/runs/20260927_0000_e3_phase0/`
**Plan:** `draft-plan/beat-mukv.md` §5 Phase 0 · **Gate 0:** MuKV reproduces published RVS-Ego behavior within harness tolerance.
**Session status at handoff (02:00 CST):** P0-3 and P0-4 complete/in-flight; **P0-1/P0-2 BLOCKED — V100 server down since ~01:00 CST** (only source of the MuKV repo).

## Gate 0 (MuKV on Ada) — BLOCKED (V100 down), fully staged

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
3. After MuKV answering + judging: `python code/scripts_e3/gate0_verdict.py judged_*.json`,
   update this report + EXPERIMENT_LOG.md, re-mirror to Mac.
