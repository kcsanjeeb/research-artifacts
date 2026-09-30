# E3 Phase 2 — Senior-review statistics (2026-09-28)

Analyses: (1) McNemar paired significance on per-question judged outcomes (n=1465,
RVS-Ego, Qwen2.5-72B judge, 0 unparsed); (2) clairvoyant learnability probe on the
2-video oracle subset (n=316 records / 314 unique questions). CPU-only, python3
(system scipy 1.15.3 / numpy 2.2.6 / pandas 2.3.0; no sklearn — selectors implemented
in numpy). Scripts: `mcnemar_task.py`, `learnability_task.py` (this dir);
machine-readable: `mcnemar_results.json`, `learnability_results.json`.

---

## Task 1 — McNemar paired tests

Outcome = judge `pred` (yes/no), paired on (video_id, question, occurrence-index).
0 questions excluded (all arms cover identical 1465 keys; all arms 1465 records,
1465 unique keys — no duplicates in the main-run files).

Per-arm accuracy (records): deferred 51.81 | writetime 51.06 | writetime_rawbias
52.08 | deferred_tax075 51.81 | writetime_tax075 50.17 | MuKV-paper 50.85 |
stock-ReKV-W2.5 54.13.

| # | Pair (a vs b) | n | both ok | both wrong | disc a-only | disc b-only | chi2 (cc) | p (cc) | p (exact) | diff (pt) | verdict |
|---|---------------|---|---------|------------|-------------|-------------|-----------|--------|-----------|-----------|---------|
| (a) | deferred vs MuKV-paper | 1465 | 610 | 571 | 149 | 135 | 0.595 | 0.4405 | 0.4405 | +0.96 | **NOT significant** |
| (b) | deferred vs writetime | 1465 | 705 | 663 | 54 | 43 | 1.031 | 0.3099 | 0.3099 | +0.75 | ns (Gate-2: indistinguishable) |
| (c) | deferred_tax075 vs writetime_tax075 | 1465 | 683 | 654 | 76 | 52 | 4.133 | 0.0421 | 0.0416 | +1.64 | **p<0.05, favors deferred** |
| (d) | deferred vs stock-ReKV W2.5 | 1465 | 627 | 540 | 132 | 166 | 3.654 | 0.0559 | 0.0557 | −2.32 | marginal (p≈0.056), deficit NOT significant at 5% |
| (e) | writetime-debias vs writetime-rawbias | 1465 | 724 | 678 | 24 | 39 | 3.111 | 0.0778 | 0.0769 | −1.02 | ns — debias null NOT rejected (rawbias numerically higher) |

Caveats:
- Judge parses: 0 unparsed `pred` in every arm; judge_prompt identical harness
  (72B vLLM TP=4, temp 0). Judge `pred` used as the binary outcome per senior spec.
- **Data-integrity note on pair (c):** `judged_answers_7b_b_deferred_tax075_72b.json`
  is **byte-identical (md5 80b8161e…) to `judged_answers_7b_b_deferred_72b.json`**, and
  its `answers_7b_b_deferred_tax075/commit_log.jsonl` is byte-identical (md5
  2aff2765…) to the plain deferred commit log — despite `sweep_chain.sh` launching a
  real `DECAF_B_TAX=0.75` deferred arm at 19:52. Either the deferred commit policy is
  provably TAX-insensitive (plausible: TAX throttles writetime eviction, not deferred
  commits), or the tax075 dir was back-filled by copying the deferred run. Pair (c)
  therefore effectively tests **deferred vs writetime_tax075**: the tax075 writetime
  arm loses 1.6 pt to deferred with p≈0.042 (the only tax arm that is statistically
  separable from its deferred counterpart — consistent with the Gate-2 Pareto
  flattening as tax cuts the writetime store).
- Multiple comparisons: 5 pairs; with Bonferroni α=0.01 only (c) survives.
- Absolute accuracies remain judge-relative (W2.5: 72B harness offset −9 to −10 pt
  vs published GPT-judged numbers); all conclusions are within-harness paired, which
  cancels the offset.

### Bottom line on the headline claim
**We may NOT say "DECAF beats MuKV" in the statistical sense.** The +0.9 pt deferred
edge over MuKV-paper (149 vs 135 discordants) has p≈0.44 — a coin flip. Approved
wording: "deferred matches MuKV paper-config (+0.9 pt, p=0.44, n=1465 paired)".
Conversely the −2.3 pt deficit vs stock ReKV W2.5 is marginal (p≈0.056) — real but
not ironclad; and Gate-2's FAIL verdict is strengthened: deferred vs writetime is
firmly null (p=0.31).

---

## Task 2 — Clairvoyant learnability probe (Phase-3 insurance policy)

### Data & labels
- Oracle subset: 2 videos (`9198b9a4…`, `0fa75cb3…`), 316 judged records =
  **314 unique questions** (1 question, "What is the setting of the video clip?",
  appears 3× identically in every policy file — benign triple-entry, all judged
  yes with same pred; occurrence-index pairing used).
- Label = correct-policy set per question from `judged_oracle_{seg,frame,patch,default}_72b.json`
  (judge pred yes/no; ties allowed).
- **Label balance:** patch 45.6% (144/316) > frame 43.4% (137) > seg 41.1% (130)
  = default 41.1% (130). Correct-set sizes: all-4-correct 110 | 3 15 | 2 17 | 1 22
  | **all-4-wrong 152 (48%)**. Unique-correct: patch 12, frame 10, seg 0, default 0
  — no policy is ever the *sole* winner, and 48% of questions are unwinnable by any
  fixed policy. Selection is inherently hard: any selector is capped by construction.

### Features (available BEFORE answering)
- From `ego4d_oe.json` annotations: query length (words, chars); question type via
  heuristic taxonomy over the annotation questions (11 buckets: temporal-firstlast,
  temporal-beforeafter, duration, counting, task, step, overview, location,
  reason-why, action-doing, other-what). NOTE: `ego4d_oe.json` carries **no explicit
  type/category field** — type is derived from question text regexes.
- From deferred run `answers_7b_b_deferred/commit_log.jsonl` (joined on
  video_id+question, occurrence-indexed, 316/316 joined): pass-1 mean entropy,
  n_tokens, committed_slots, n_committed_grains, store_bytes_total, store_bytes_kv,
  n_segments, uncertain flag, pass-1 policy/level.

### Protocol
Split 316 in half (seed 42, shuffled); selectors fit on train, evaluated on held-out
half. Reward of a pick = 1 if picked policy ∈ correct-set(q). Captured gain =
selector held-out reward − always-patch held-out reward. Bootstrap CI (2000
resamples of held-out half). Robustness: 200 additional seeded 50/50 splits.
Models: (i) decision stump (all continuous thresholds + categorical-equality
stumps); (ii) multinomial logistic regression (own numpy GD, standardized inputs,
question-type one-hots, 3000 epochs, l2=1e-3); (iii) question-type-conditioned
policy table (patch-fallback).

### Results

Held-out, seed-42 split (patch baseline on that half: 46.8%):

| selector | held-out | gain over patch | bootstrap 95% CI |
|----------|----------|-----------------|------------------|
| decision stump (entropy ≤ 1.27) | 46.8% | +0.00 pt | [−2.53, +2.53] |
| logistic regression | 48.7% | **+1.90 pt** | [+0.00, +4.43] |
| type-conditioned | 46.8% | +0.00 pt | [+0.00, +0.00] |

Robustness — 200× seeded 50/50 splits, mean held-out gain over patch:

| selector | mean gain | sd | fraction of splits > 0 |
|----------|-----------|----|------------------------|
| logistic regression | **−1.02 pt** | 1.75 pt | 0.26 |
| type-conditioned | −1.24 pt | 2.05 pt | 0.17 |
| decision stump | −1.41 pt | 1.79 pt | 0.09 |

Oracle-selector upper bounds (fit on all 316, in-sample, same feature access):
memorization lookup 51.9% (+6.3 pt over patch) — this is the absolute per-question
ceiling (mean correct-set size / 316), NOT a feature achievement; LR in-sample 47.5%
(+1.9 pt); stump in-sample 46.5% (+0.9 pt). Even perfect memorization buys only
+6.3 pt, and 48% of questions contribute zero to any selector.

### Decision-rule verdict
Rule: ≥ +1.0 pt held-out gain ⇒ learnable, LoRA-scale Phase 3 justified.

**Verdict: FAIL / ~0 — headroom is NOT learnable from the available pre-answer
features.** The single seed-42 split shows LR at +1.90 pt, but the 200-split
distribution (mean −1.02 ± 1.75 pt, only 26% of splits positive) shows that number
is split-selection noise; the CI lower bound is exactly 0.00. The in-sample
memorization ceiling of +6.3 pt (itself capped by 48% unwinnable questions) is never
approached by any feature-based model even in-sample (LR in-sample +1.9 pt),
indicating the features carry almost no signal about which policy wins.
**Flag for senior: per-query policy choice headroom (+8.9/+10.5 pt on this subset)
is not predictable from question metadata + store statistics; a learned C5
commitment scorer would need features beyond these (e.g., retrieval-content
features) or it will not beat always-patch.**

---

## Reproduce
```
python3 mcnemar_task.py        # ~5 s
python3 learnability_task.py   # ~90 s (LR GD + 200 resplits + 2000 bootstraps)
```
