# W4.3 — window coverage vs planner gain (2026-09-30, CPU-only)

**Claim welded:** the benchmark-dependent planner gain (+4.3 StreamingBench / +0.9 OVO /
−0.1 RVS-Ego over the random-selector floor) tracks mean window coverage almost perfectly.
**Coverage is the prerequisite; grain selection is a second-order term — and the field is
optimizing the second-order term.** This is CORRELATIONAL (3 benchmarks; 6 markers), not
causal. The causal test is W4.2 (coverage-spread retrieval), in flight.

## 4-column table

| benchmark | system acc | planner gain vs random floor (pt) | mean window coverage (all q) | n | mean cov (all-wrong) | n_wrong | mean window (s) |
|---|---|---|---|---|---|---|---|
| RVS-Ego (oracle subset) | 42.7 | −0.1 | 0.214 (median 0.133) | 316 | 0.209 (median 0.126) | 152 | 470 |
| OVO-Bench | 39.4 | +0.9 | 0.587 (median 0.500) | 180 | 0.576 (median 0.500) | 98 | 592 |
| StreamingBench | 73.9 | +4.3 | 1.000 | 180 | 1.000 | 42 | 39 |

Monotone in both coverage populations (all-questions and all-wrong-subset means order
identically: 0.21 < 0.59 < 1.00 against gains −0.1 < +0.9 < +4.3).

## Method (same definition as W3.3, `analysis/w33_ceiling.py`)

- Coverage = |∪ retrieved frame-blocks ∩ [start,end]| + local-cache addendum (last
  15000/196 = 76 tokens = 76 frames resident at answer time), capped at 1, over the
  streamed history up to the question. Measurement against the gold window only —
  allocation (W4.2) must never use it.
- Block→time: RVS 4 s/block (0.25 fps effective, double-decimated npy path); SB/OVO
  2 s/block (0.5 fps mp4, `--sample_fps 0.5`; max retrieved block ≈ duration/2 verified).
- Frontier: RVS min(2·end, duration) (2×-end encode quirk); SB/OVO = end (cumulative).
- Sources: system writetime arms `answers_7b_writetime/1_0.csv` (SB, OVO) and
  `answers_7b_b_writetime/1_0.csv` filtered to the 2-video W3.3 subset (RVS); all-wrong
  subset uses oracle best-policy retrieval (RVS patch, SB frame, OVO frame), W3.3's
  convention. Row order verified against judged files and anno (question-text asserts).
- **W3.3 reproduction:** RVS all-wrong overlap mean 0.20903 / median 0.12649 — matches
  `w33_ceiling/w33_allwrong_breakdown.json` to full float precision (both populations).

## Numbers provenance (all pre-existing, nothing re-judged)

- System acc / gain: RVS + SB from `w32_oracle_null/w32_results.json`
  (`gap_system_minus_floor_pt` is already in points: −0.079 → −0.1; +4.306 → +4.3);
  OVO from `ORACLE_OVO.md` (system 39.4, random floor 38.5 ± 1.15, 10k repeats).
- Coverage: computed here; SB/OVO commit logs carry no seg ids (W3.3 note) but
  retrieved-block coverage needs only `1_0.csv` — no gap.

## Verdict

**Suggestive monotone, exactly the thesis direction — with the honest caveat: 3 benchmarks,
n = 3 independent settings, one point per benchmark.** No statistic is computed on the
correlation itself (3 points cannot carry one). Figure: `gain_vs_coverage.png`.
W4.2 (coverage-spread arms A1–A3 on RVS) is the causal gate: if coverage is the
prerequisite, forcing coverage ↑ must move temporal accuracy; if coverage ↑ but accuracy
is flat, the correlational story here is confounded and the direction closes.

## Reproduce

```
python3 deliverables/e3_phase2/analysis/w43_coverage.py   # ~5 s, CPU only
```

Outputs: `w43_coverage.json`, `gain_vs_coverage.png`, this file.
