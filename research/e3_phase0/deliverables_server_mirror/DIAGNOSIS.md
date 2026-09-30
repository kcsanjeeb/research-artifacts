# Gate 0 failure — diagnosis (2026-09-27)

**Verdict up front: root cause = config fidelity, NOT code path, NOT judge style.**
Original FAIL was an artifact of running MuKV with the runner-script defaults instead of
the paper RVS-Ego config. With the paper-faithful config, MuKV-7B scores **50.9** in our
72B harness (band [47.5, 53.5], expected ≈ 50.5) → **Gate 0 REVISED: PASS**.

Run dir: `/data3/zhuotaotian2_e2/runs/20260927_0000_e3_phase0/`
Code: `code/MuKV` @ `2127b875` + sdpa/fp32-upcast patches, env `/data4/rekv`.
Judge: Qwen2.5-72B vLLM TP=4 (W2.5-calibrated, −9 Acc systematic offset vs published,
ordering preserved). All numbers below are in-harness Acc on RVS-Ego (n=1465, 0 parse
failures in every run).

## 1. Root cause: keep-ratio / retrieval config mismatch

The original run used the argparse defaults of `scripts/run_mukv_rvs_ego.py` (what the
README example implies), which are **not** the paper configuration. The repo
`scripts/sh/run_mukv_rvs_ego.sh` matches the paper. Side by side (grain order
49=patch, 196=frame, 784=segment, per `model/mukv_rerank.py:253` and paper §4.1):

| Parameter | What we ran (runner defaults) | Paper RVS-Ego config (§4.1 + suppl. Tab 10/11) |
|---|---|---|
| Retention 49 (patch)   ρ_p | **0.70** | **0.10** |
| Retention 196 (frame)  ρ_f | **0.50** | **0.10** |
| Retention 784 (segment) ρ_s | **0.30** | **0.80** |
| Per-grain retrieval topk | not passed → weight-proportional split of 64 by α ({0.5,0.7,0.8}) ≈ 17/22/24 | **20 / 32 / 12** (k_g=64 total) |
| Rerank λ (α/β) | 0.5 / 0.6 | **0.3 / 0.3** (suppl. Tab 11 best for Ego), top_n=5 |
| FFT method | `diff` | `fft` (Fourier-domain frequency, §4.1) |
| Attention weights α | 0.5 / 0.7 / 0.8 | 0.5 / 0.7 / 0.8 (same) |
| fps / n_local / retrieve | 0.5 / 15000 / 64 | 0.5 / (15000) / 64 (same) |
| importance method | attention_fft_weighted | attention_fft_weighted (same) |

Our retention profile (0.7/0.5/0.3, patch-heavy) is nearly the **inverse** of the paper
(segment-dominant 0.1/0.1/0.8). The paper sensitivity table (suppl. Tab 10) shows
exactly this axis costs points: ρ=(0.8,0.1,0.1) — the patch-heavy end — is their
**worst** setting (54.7 vs 57.9 best). Segment-level KV carries RVS-Ego (Tab 2:
segment-only 54.9 beats patch-only 51.6 / frame-only 53.1). Pruning 70% of it, as our
default config did, removes the dominant evidence source. This fully accounts for the
gap; nothing about the code or judge needed to be wrong.

## 2. Measured numbers (all Qwen2.5-72B-judged, RVS-Ego, n=1465)

| Run | Config | Acc | Score | Parse failures |
|---|---|---|---|---|
| MuKV-7B (original, runner defaults) | keep 0.70/0.50/0.30, auto topks, rerank 0.5/0.6, fft=diff | 46.0 | 2.49 | 0 |
| MuKV-0.5B (original, runner defaults) | same | 43.4 | 2.26 | 0 |
| **MuKV-7B paper config** | keep 0.1/0.1/0.8, topks 20/32/12, rerank 0.3/0.3, fft=fft | **50.9** | **2.67** | 0 |
| **MuKV-0.5B paper config** | same | **48.0** | **2.45** | 0 |
| MuKV-7B keep9 (code-path sanity) | keep 0.9/0.9/0.9, topks 20/32/12, rerank 0.3/0.3 | 51.2 | 2.67 | 0 |
| Reference: ReKV-7B (W2.5, same harness) | — | 54.1 | — | 0 |

Bands (published − 9 ± 3): 7B [47.5, 53.5] → 50.9 PASS; 0.5B [45.9, 51.9] → 48.0 PASS.
Expected in-harness 7B = 50.5; measured 50.9 (+0.4 vs expectation, −8.6 vs published
59.5, exactly the calibrated offset). In-harness ordering also matches published
ordering: MuKV-7B (50.9) < ReKV-7B (54.1), as published MuKV 59.5 < ReKV 63.7.

## 3. Task 2 — code-path sanity (sdpa/torch fallback + fp32-upcast)

Near-full retention (0.9/0.9/0.9, minimal pruning, paper topks/rerank) gives **51.2** —
the same level as the paper-config run (50.9) and within ~3 pts of ReKV-with-offload
(54.1 in-harness). If the fallback attention path broke MuKV DCP scoring or retrieval,
minimal-pruning accuracy would have collapsed far below this. Conclusion: the
sdpa/torch-fallback + fp32-upcast path is **sound**; building MuKV native flash-attn
env is unnecessary for Gate 0 (optional hardening for Phase 4 final numbers).
Note: paper-config (50.9) ≈ keep9 (51.2) — DCP at the right ratios is near-lossless,
consistent with paper Tab 3 where DCP(67%) *improves* over no compression.

## 4. Task 3 — answer-style check

Compared all 1429 unique shared questions between MuKV-7B (original run) and ReKV-7B
(W2.5 answers): mean answer length 25.0 vs 25.8 words, coherent fluent sentences,
same phrasing style, near-empty answers 24 vs 4. MuKV answers are **not degenerate**
— the original 46.0 deficit was content (wrong store content after inverted pruning),
not style. No evidence of a judge-style interaction; the W2.5 offset applies uniformly.

## 5. Revised Gate-0 verdict

**PASS.** MuKV reproduces its published RVS-Ego behavior within harness tolerance on
our judge: 7B 50.9 ∈ [47.5, 53.5], 0.5B 48.0 ∈ [45.9, 51.9], 0 parse failures.
Faithful MuKV baselines for C2/C3 going forward: **0.5B = 48.0/2.45, 7B = 50.9/2.67**
(RVS-Ego, Qwen2.5-72B harness). The original FAIL record is kept as superseded
history in REPORT.md per no-delete rules.

Artifacts: `answers_{7b_paper,7b_keep9,05b_paper}/`, `judged_{7b_paper,7b_keep9,05b_paper}_72b.json`,
`gate0_verdict_v2.txt`, `judge_v2.status`, `run_mukv_answer_v2.sh`, `run_judge_v2.sh`
in the run dir above. Judge JSONs: `judged_7b_keep9_72b.json` (51.2) has no gate0 row
by design (no published keep9 reference).
