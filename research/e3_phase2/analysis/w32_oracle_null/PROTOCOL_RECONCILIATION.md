# W3.2 Oracle-Gap Permutation Null — Protocol Reconciliation (RVS / StreamingBench / OVO-Bench)

Date: 2026-09-30. Trigger: senior-reviewer flag that OVO's printed `p=0.9736` ("97.4th percentile")
implied a different null protocol than RVS (1.06th pct) and StreamingBench (12.6th pct).

**Conclusion up front: no protocol flip.** All three nulls are the same within-question
exchangeability test; the apparent tail flip is a misreading of OVO's printout — `p = 0.9736` is an
upper-tail p-value, which places the observed gap at the **2.6th percentile** of the null (low tail),
the same direction as RVS (1.06th) and SB (12.6th). There were, however, three minor protocol
deviations in the OVO implementation (correctness source, +1 correction, RNG stream position),
which are removed by the unified recomputation below.

## 1. Protocol table — as originally computed (before)

| Aspect | RVS (w32_oracle_null.py, null2) | StreamingBench (same script, null2) | OVO (runs/20260929_1215_e3_phase2_ovo/ovo_verdict.py) |
|---|---|---|---|
| Correctness source | judged_oracle_{seg,frame,patch,default}_72b.json (72B judge, yes/no) | judged_oracle_*_letter.json (letter-match judge) | **regex letter extraction from raw `1_0.csv` preds** — judged_*_letter.json NOT used for headline |
| Null construction | per-question permutation of the 4 policy correctness values across policy labels (`rng.permutation(4)` per row; per-question multiset fixed) | same | per-row permutation via `argsort(rng.random((n,4)))` — distributionally identical |
| Gap under null | oracle acc (per-question max) − best-fixed acc (max column mean); oracle invariant under null | same | same ("clairvoyant" − best-fixed) |
| Iterations / seed | 10,000 / 2024 — but null2 reuses the rng AFTER null1 (40k column perms), so exact draws differ | same | 10,000 / 2024 — but rng first consumed by the random-selector simulation (10k×n integers) |
| p / tail | one-sided upper tail, **with +1 correction** `(#{null≥obs}+1)/(N+1)`; also prints obs percentile in null | same | one-sided upper tail, **no +1 correction** (`mean(null≥obs)`); prints p and z, no percentile |
| Printed result | obs gap +6.33 pt at **1.06th pct**, p = 0.9904 | obs gap +5.00 pt at **12.6th pct**, p = 0.8712 | obs gap +4.44 pt → **p = 0.9736**, z = −1.80 (i.e. 2.64th pct, NOT 97.4th) |

Null 1 (independent column shuffle across questions) in the w32 script is a separate diagnostic
(observed oracle far BELOW the independence null, p = 1.0 — policies strongly correlated); it is not
the oracle-gap test under review and is unchanged.

## 2. Unified recomputation (after) — one identical protocol

Script: `w32_unified_null.py` (this directory). From the per-question judged jsons for all three
benchmarks (RVS: judged_oracle_*_72b.json; SB: judged_oracle_*_letter.json; OVO:
judged_oracle_*_letter.json). Protocol: per-question permutation of the 4 policy correctness labels;
gap = per-question-max accuracy − best-fixed-policy accuracy; 10,000 iterations; seed 2024 (fresh
stream per benchmark); one-sided upper-tail p with +1 correction for all three.

| Benchmark | n | oracle | best-fixed | system | obs gap (pt) | null mean ± sd (pt) | obs pct in null | p (one-sided) | z |
|---|---|---|---|---|---|---|---|---|---|
| RVS subset | 316 | 51.9 | patch 45.6 | 51.1 | +6.33 | 7.84 ± 0.61 | 1.07th | **0.9893** | −2.48 |
| StreamingBench RT | 180 | 76.7 | frame 71.7 | 73.9 | +5.00 | 5.57 ± 0.73 | 12.29th | **0.8771** | −0.79 |
| OVO-Bench | 180 | 45.6 | frame 41.1 | 39.4 | +4.44 | 5.69 ± 0.69 | 2.47th | **0.9753** | −1.82 |

OVO under judged-json correctness (frame 41.1 / patch 40.6) is essentially unchanged vs the regex
path (41.1 / 40.6; clairvoyant 45.6 both ways): p moves 0.9736 → 0.9753. RVS/SB reproduce the
original null2 results up to RNG-stream position (0.9904→0.9893, 0.8712→0.8771).

## 3. Multiplicity correction across the three benchmarks

Family = the three W3.2 oracle-gap permutation tests. α = 0.05.

| Benchmark | raw p | Holm adj p | Holm verdict | BH adj p | BH verdict |
|---|---|---|---|---|---|
| RVS subset | 0.9893 | 1.0000 | not rejected | 0.9893 | not rejected |
| StreamingBench RT | 0.8771 | 1.0000 | not rejected | 0.9893 | not rejected |
| OVO-Bench | 0.9753 | 1.0000 | not rejected | 0.9893 | not rejected |

No benchmark survives even uncorrected (all raw p > 0.87); corrections change nothing.

## 4. Is there a consistent harvestable policy-selection edge?

**No.** Under the unified within-question exchangeability null, the oracle-minus-best-fixed gap is
not larger than chance in any of the three benchmarks — observed gaps of +6.3 (RVS), +5.0 (SB) and
+4.4 (OVO) pt sit at the 1.1th, 12.3th and 2.5th percentiles of their null distributions
(p = 0.989 / 0.877 / 0.975; Holm- and BH-adjusted, none remotely significant). The clairvoyant
oracle stays a bounding device, not an attainable target: its margin over the best single policy is
fully explained by max-over-exchangeable-policies selection, and the one genuinely positive
system-level fact is unrelated to grain selection — the deployed system already matches or beats the
random-selector floor everywhere. Per-question grain selection as implemented carries no
statistically detectable harvestable edge on any benchmark; the earlier OVO printout did not
contradict this, as its p = 0.974 is the same non-significant low-tail outcome in the same test.

## Artifacts

- `w32_unified_null.py` — unified recomputation script (CPU, deterministic)
- `w32_unified_null.json` — unified per-benchmark stats + Holm/BH
- `w32_results.json` — original RVS/SB analysis (null1 + null2 + random floor), unchanged
- OVO original: `runs/20260929_1215_e3_phase2_ovo/ovo_verdict.{py,txt}`, `clairvoyant_detail.json`
