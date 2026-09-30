# W5.4 — Pooled coverage regression (temporal-grounding diagnosis, powered re-test)

**Date:** 2026-09-30 · **Compute:** CPU only (Ada, /data4/rekv python: scipy 1.17.1, statsmodels 0.15.0)
**Analysis dir (Ada):** `analysis/w54_coverage_regression/` · **Code:** `w54_coverage_regression.py` (deterministic; probe.py = pre-reg diagnostics)

## Why this exists

W4.2 (coverage-spread decision experiment, 83 temporal before/after questions, 4 arms) returned an
underpowered null: MDE 16–21 pt at n=83, so the temporal-grounding diagnosis (correctness should
rise with achieved gold-window coverage) was **untested, not refuted**. W5.4 pools all per-question
observations across the four arms — top-k (W4.1 writetime@tax0.5, 0.5 fps) + uniform / hybrid /
stratified spread arms — into one regression of per-question correctness on achieved window
coverage: 4 × 83 = **332 observations**.

## Data

- **Correctness:** judged files, 72B judge — `runs/20260930_0935_w42_spread/judged_arm_{uniform,hybrid,stratified}_72b.json`
  and `runs/20260930_0935_w41_fpsfix/judged_7b_writetime_05fps_72b.json`, restricted to the 83 ids in
  `temporal83_ids.json`. One binary observation per (question, arm); all 83 keys unique, occ=0.
- **Coverage:** per question per arm, same W3.3 definition as the W4.2 stats (`stats_w4.py`):
  committed 2-s frame blocks (0.5 fps) overlapping the gold [start,end] window + local-cache
  addressable-window term, divided by window length; computed from each arm's `commit_log.jsonl`.
- The arms genuinely retrieved different sets (W4.2 debug: pairwise retrieval Jaccard 0.18–0.29),
  so coverage **varies within question** across arms — this is what identifies the fixed-effects model.

## Coverage actually observed

| quantity | value |
|---|---|
| pooled coverage mean / min / max | 0.310 / 0.181 / 0.491 |
| within-question sd of coverage (across the 4 arms) | 0.019 (max per-question arm range 0.102) |
| between-question sd of question-mean coverage | 0.070 |
| per-arm accuracy on the 83 | top-k 6.0%, uniform 9.6%, hybrid 12.0%, stratified 8.4% |
| pairwise coverage correlation across arms | 0.94 – 0.99 |

≈90% of coverage variance is **between** questions; the four arms' coverages move together within a
question (correlation 0.94–0.99). Only 27/83 questions have an arm-to-arm coverage range > 0.05.
The arms spread retrieval content but barely spread achieved *coverage* — the manipulation was
weaker in coverage space than the Jaccard suggests.

## Regression models

All OLS linear probability models, cluster-robust standard errors clustered by question (83
clusters). Question-type control is vacuous (all 83 questions are temporal before/after); window
length enters model B. Dataset: `w54_dataset.csv`.

| model | spec | coverage coef | 95% CI | p | n | implied effect per +0.1 coverage |
|---|---|---|---|---|---|---|
| **A (primary)** | correct ~ coverage + question FE | −0.334 | [−1.99, +1.33] | 0.694 | 332 | −3.3 pt |
| B | correct ~ coverage + arm FE + window length | +0.559 | [−0.54, +1.66] | 0.320 | 332 | +5.6 pt |
| C | correct ~ coverage + question FE + arm FE | +0.403 | [−1.33, +2.13] | 0.647 | 332 | +4.0 pt |
| D (robustness) | logit, coverage + question FE | −12.3 (logit failed to converge; OR unreliable) | [−47.5, +22.9] | 0.493 | 332 | OR 0.29 per 0.1 — not interpretable |

(Coefficients are in probability units per 1.0 of coverage. Logit with question FE on a 6–12%
prevalence outcome does not converge — expected; the LPM is the reliable read.)

**Coefficient: not significantly different from zero in any convergent specification.** Even the
largest point estimate (B, +5.6 pt per 0.1 coverage) has a CI spanning −5.4 to +16.6 pt.

## Coverage-vs-accuracy curve (6 quantile bins)

| coverage bin | n | mean cov | accuracy | 95% CI |
|---|---|---|---|---|
| 0.180–0.236 | 57 | 0.209 | 5.3% | [−0.5, 11.1] |
| 0.236–0.275 | 54 | 0.260 | 13.0% | [4.0, 21.9] |
| 0.275–0.305 | 64 | 0.292 | 14.1% | [5.5, 22.6] |
| 0.305–0.333 | 57 | 0.323 | 1.8% | [−1.7, 5.2] |
| 0.333–0.389 | 45 | 0.360 | 6.7% | [−0.6, 14.0] |
| 0.389–0.491 | 55 | 0.430 | 12.7% | [3.9, 21.5] |

Non-monotone: accuracy does not rise with coverage (the 0.305–0.333 bin is the *worst*). The
curve is flat noise around ~9%.

## Additional internal check

The one arm with a real accuracy gain in W4.2, hybrid (+6.0 pt vs top-k, p=0.180), had coverage
*slightly lower* than top-k (median 0.284 vs 0.305). The largest observed accuracy movement in the
experiment moved against the diagnosed direction.

## Verdict

**The diagnosis is REFUTED ON EVIDENCE, and the temporal direction closes honestly.** Per the
pre-declared read: the pooled coefficient of correctness on achieved window coverage is flat and
insignificant across the observed coverage range (pooled 0.18–0.49; per-arm means 0.29–0.32), in
the question-fixed-effects specification that absorbs question difficulty (coef −0.33, 95% CI
[−1.99, +1.33], n=332), in the arm-FE + window-length specification (+0.56, CI [−0.54, +1.66]), and
in the binned curve (non-monotone). Unlike W4.2, this null is not a power artifact of an 83-point
two-arm comparison: the pooled design has 4× the observations and real within-question coverage
variation to identify on. A coverage→correctness gradient large enough to matter would have had to
exceed ≈±11–17 pt per 0.1 coverage to be visible; nothing in that band (nor anything smaller) is
present. This refutes the *continuous coverage relationship* as the explanation for the temporal
questions' low accuracy, complementing the W4.2 decision mapping (coverage flat vs top-k) — the
temporal weakness is not explained by achieved gold-window coverage under this allocator family.

## Caveats

1. **Narrow coverage range.** The arms spread retrieval *content* (Jaccard 0.18–0.29) but not
   coverage: within-question sd is 0.019 and 90% of coverage variance is between questions. The
   refutation is conditional on the achieved band (~0.18–0.49, with the bulk at 0.24–0.40). A
   design that actually drove coverage to, e.g., 0.6–0.9 could in principle find a relationship —
   but no DECAF allocator in the tested family gets there at this store budget, so for this
   system the direction is closed.
2. **FE power.** The question-FE model is identified only on within-question variation (mean sd
   0.019); its CI (±1.66 per unit coverage) still cannot exclude effects below ~±17 pt per 0.1
   coverage. The between-question-informed model B is tighter (±0.56 per 0.1 → MDE ≈ 11 pt per 0.1
   at 95%) and also null. The refutation is strongest read jointly: no specification, at any
   plausible power level this data can support, shows a positive gradient.
3. **Logit instability** (model D) is a prevalence artifact, not evidence; LPM preferred per spec.
4. Scope: refutes the coverage↔correctness relationship for the 72B-judged RVS-Ego temporal-83 on
   this model family; it does not re-open any killed mechanism and does not speak to coverage
   effects outside the observed band.

## Decision-table consequence

| Task | Question | Result | Consequence |
|---|---|---|---|
| W5.4 | Is correctness significantly increasing in coverage? | **No — refuted on evidence** (pooled regression, n=332; coef n.s. everywhere; curve non-monotone) | **temporal direction CLOSED on evidence** |

No follow-on experiment is justified. Consistent with W4.2's decision mapping (coverage FLAT vs
top-k → not a coverage lever).
