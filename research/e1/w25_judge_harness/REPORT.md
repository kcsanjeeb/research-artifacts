# W2.5 — RVS Judge Harness: Validation Report

**Date:** 2026-09-26 · **Run dir:** /data3/zhuotaotian2_e2/runs/20260926_1455_w25_judge_harness

## Verdict

**JUDGE NOT VALIDATED against published numbers (±2 pt criterion FAILED).** Both ReKV models
land ~9–10 Accuracy points below their published GPT-judged values under our judge — a
systematic offset, not noise. The harness is **internally consistent** (two local judges agree
at 94% on a 50-Q prefix; 0/1465 parse failures per grading; model ordering reproduces), so
**relative comparisons within this harness are usable; absolute numbers are NOT comparable to
published RVS numbers.** See "Citable?" below.

## Setup

- Judge actually used: **Qwen2.5-72B-Instruct bf16** (145.4 GB, 37/37 safetensors shards verified
  against model.safetensors.index.json; downloaded via hf-mirror.com, XET disabled, after 3
  failed attempts — see GPU_COORDINATION.md), vLLM 0.7.3, TP=4 on GPUs 0–3,
  gpu_memory_utilization 0.80–0.85, max_model_len 2048, greedy (temperature 0), max 64 new tokens.
  (Original 72B plan kept; 32B fallback NOT needed.)
- Second judge (agreement): Qwen1.5-14B-Chat (local, /data1/LLM_models), same prompt/protocol,
  50-question fixed prefix, TP=1 on GPU 2.
- Grading prompt: exact copy of ReKV repo `video_qa/eval/eval_open_ended.py` (re-read and
  diff-checked against w25_judge.py on 2026-09-26; system+user strings verbatim). Note the
  ReKV repo grades with `gpt-3.5-turbo-0613`, temperature 0, max_tokens 300.
- Metrics per protocol: Accuracy = yes/(yes+no); Score = mean integer 0–5.
- Answering: ReKV repo @ 1fd9a3d + V100 port patches (fp32-upcast attention, sdpa, fattn=False;
  rope shim for transformers>=4.45). `rekv_stream_vqa`, sample_fps 0.5, n_local 15000,
  retrieve_size 64, chunk_size 1, internal retrieval, seed 2024. transformers 4.46.3 + torch 2.6.0.
- Data: ego4d_oe.json, **full coverage: 1465/1465 questions, 10/10 videos for BOTH models**
  (answers_{0.5b,7b}/1_0.csv, 0 empty predictions). Exit 0 both.

## Reproduction vs published

| Model | Published Acc / Score (GPT-judged) | This harness Acc / Score (Qwen2.5-72B judge) | ΔAcc | ΔScore |
|---|---|---|---|---|
| ReKV-0.5B | 54.7 / 3.9 | **45.5 / 2.36** (yes=667, no=798, unparsed=0/1465) | **−9.2** | −1.54 |
| ReKV-7B   | 63.7 / 4.0 | **54.1 / 2.82** (yes=793, no=672, unparsed=0/1465) | **−9.6** | −1.18 |

- The offset is systematic (−9.2 and −9.6 Acc) — both models shifted down together.
- Model ordering reproduces: ours 7B−0.5B = **+8.6 Acc** (published +9.0).
- Deviation exceeds the ±2 pt validation band by ~4–5× on Accuracy, ~0.8–1.5 on Score.
  **VALIDATION: FAILED / UNVALIDATED** as a GPT-equivalent judge for absolute claims.
- Cannot apportion the offset between judge strictness (GPT-3.5 vs Qwen2.5-72B) and
  answering-side port deviations (fp32-upcast attention, transformers 4.46.3) without GPT access;
  the tight systematicity across two model sizes suggests a large shared component (judging
  behavior and/or pipeline offset common to both runs).

## Inter-judge agreement (50-question fixed prefix, 0.5B answers)

| Judge | Accuracy | Score |
|---|---|---|
| Qwen2.5-72B-Instruct | 48.0 (yes=24, no=26) | 2.32 |
| Qwen1.5-14B-Chat | 50.0 (yes=25, no=25) | 2.80 |

- Pred agreement **94.0%** (47/50; 23 both-yes, 24 both-no, 3 discordant).
- Score MAE **0.60** (n=50, both scored, 0 unparsed).
- Two local judges of very different size largely agree with each other — the disagreement is
  with the published GPT-judged numbers, i.e. a judge-substitution / pipeline offset.

## Known deviations / risks

- Published numbers were GPT-judged (repo default gpt-3.5-turbo-0613). Our judge is
  Qwen2.5-72B — a judge substitution; this validation was designed to bound it and it failed
  the ±2 pt band.
- V100 numerical-equivalence spot-check BLOCKED (no V100 ssh key); answering config reproduces
  the ported V100 config exactly on the hardware-independent torch/sdpa path. Ada-vs-V100
  equivalence NOT independently spot-checked.
- transformers 4.46.3 instead of pinned git commit; 2-line rope shim (same values).
- Judge serving colocated with another tenant ~5 GB training jobs on GPUs 2/3: first 7B grading
  died (exit 137, CUDA OOM during cudagraph capture at gpu_mem 0.85); succeeded at 0.80 after
  killing our own orphaned TP workers. 0.5B grading unaffected.
- Grading determinism: temperature 0; vLLM batching nondeterminism may flip borderline judgments.

## Citable?

- **Absolute RVS Accuracy/Score from this harness vs published numbers: NO.** Deviation
  −9.2/−9.6 Acc (±2 band failed by ~5×), judge substitution (GPT-3.5 → Qwen2.5-72B) unvalidated.
- **Relative comparisons within this harness: YES** — same judge, both ReKV sizes, full 1465-Q /
  10-video coverage, 0 parse failures, 94% inter-judge agreement, reproduced model ordering
  (+8.6 vs published +9.0). FleetVAD vs ReKV head-to-head numbers graded by THIS harness are
  citable as internally consistent; any claim quoting published RVS magnitudes is not.

## Artifacts (run dir)

- answers_0.5b/1_0.csv, answers_7b/1_0.csv (1465 rows each, full coverage)
- judged_0.5b_72b.json, judged_7b_72b.json (full judged outputs)
- judged_0.5b_qwen15_14b_prefix50.json (14B agreement grading)
- w25_judge.py (+ .bak72b), w25_agreement.py, run_judge72b.sh, run_judge72b_7b.sh,
  logs_judge72b.log, logs_judge72b_7b.log, answers_{0.5b,7b}.log
