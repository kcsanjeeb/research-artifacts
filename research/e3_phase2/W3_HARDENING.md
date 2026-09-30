# W3 — Claim Hardening (analysis half): multiple-comparisons correction, permutation nulls, ceiling analysis, equal-tax closure

**Date:** 2026-09-29 · **Runs:** `20260928_0315_e3_phase2_compB` (RVS), `20260928_2105_e3_phase2_xbench` (StreamingBench RT)
**Scripts/outputs:** `deliverables/e3_phase2/analysis/{w31_multiple_comparisons,w32_oracle_null,w33_ceiling}/` (mirrored to `research/e3_phase2/analysis/` on Mac)
**Everything CPU/server-side. No mechanisms built, no GPU used.**

---

## Verdict up front (W3.1)

**The pre-declared primary comparison does NOT survive correction.**
Primary: deferred@tax0.5 vs writetime@tax0.75, raw p=0.042 (n=1465, exact McNemar).

| Family | m | Holm adj. | BH adj. |
|---|---|---|---|
| A: all pairs among 7 full-run RVS arms | 21 | **0.67** | **0.15** |
| A∪B: every comparison in the project | 36 | **1.00** | **0.17** |

(raw p-exact 0.0416 / p-cc 0.0421; both corrections on both p variants agree: not significant.)

**Consequence, stated plainly:** the paper's quantitative claim is **"matches CVPR'26 SOTA (MuKV faithful config) at −2.3 pt vs full-fidelity ReKV (n.s., p=0.056), with no accuracy claim surviving multiplicity correction."**
The Pareto pair p=0.042 may appear only as a *descriptive, uncorrected* observation, explicitly labelled as not surviving Holm (0.67) / BH (0.15). Do not restate it as significant anywhere.

---

## W3.1 — Full multiple-comparisons accounting

### Family A — full-run RVS-Ego arms (n=1465, 72B judge), all 21 pairs

Arms: deferred (tax0.5, debias) 51.81 · writetime (tax0.5, debias) 51.06 · writetime_rawbias (tax0.5) 52.08 · writetime_tax025 51.81 · writetime_tax075 50.17 · mukv_paper 50.85 · stock_rekv_w25 54.13.
`deferred_tax075` is byte-identical to `deferred` (INTEGRITY_tax075.md) — not an independent arm; `deferred_tax025` degenerate (excluded per Gate-2 report). Full matrix (b, c, p) in `w31_family_A.json` / `w31_matrix.csv`. Sorted by p:

| Pair | Δ pt | b | c | p (exact) |
|---|---|---|---|---|
| stock_rekv vs writetime_tax075 | +3.96 | 179 | 121 | 0.0010 |
| mukv_paper vs stock_rekv | −3.28 | 111 | 159 | 0.0041 |
| stock_rekv vs writetime | +3.07 | 155 | 110 | 0.0068 |
| writetime_rawbias vs writetime_tax075 | +1.91 | 81 | 53 | 0.0193 |
| stock_rekv vs writetime_tax025 | +2.32 | 139 | 105 | 0.0344 |
| **deferred vs writetime_tax075 (PRIMARY)** | **+1.64** | **76** | **52** | **0.0416** |
| deferred vs stock_rekv | −2.32 | 132 | 166 | 0.0557 |
| writetime vs writetime_rawbias | −1.02 | 24 | 39 | 0.0769 |
| writetime_tax025 vs writetime_tax075 | +1.64 | 97 | 73 | 0.0774 |
| stock_rekv vs writetime_rawbias | +2.05 | 154 | 124 | 0.0818 |
| writetime vs writetime_tax075 | +0.89 | 63 | 50 | 0.2589 |
| deferred vs writetime | +0.75 | 54 | 43 | 0.3099 |
| mukv_paper vs writetime_rawbias | −1.23 | 139 | 157 | 0.3231 |
| writetime vs writetime_tax025 | −0.75 | 62 | 73 | 0.3895 |
| deferred vs mukv_paper | +0.96 | 149 | 135 | 0.4405 |
| mukv_paper vs writetime_tax025 | −0.96 | 138 | 152 | 0.4453 |
| mukv_paper vs writetime_tax075 | +0.68 | 165 | 155 | 0.6150 |
| deferred vs writetime_rawbias | −0.27 | 37 | 41 | 0.7343 |
| writetime_rawbias vs writetime_tax025 | +0.27 | 75 | 71 | 0.8040 |
| mukv_paper vs writetime | −0.20 | 145 | 148 | 0.9070 |
| deferred vs writetime_tax025 | 0.00 | 71 | 71 | 1.0000 |

What survives Holm over m=21: only the three stock-ReKV anchor pairs (p≤0.007) — i.e., the *known* "full-fidelity ReKV sits above the compressed arms" ordering. Nothing about the DECAF system vs MuKV, vs writetime, or the Pareto pair survives.

### Family B — oracle-subset arms (n=316), all 15 pairs

Full matrix in `w31_family_B.json`. Strongest: patch beats seg/default (p≈0.024), deferred-vs-patch p=0.020 — none of these were ever claimed; included for completeness. The union family (m=36) contains the primary with Holm 1.00 / BH 0.17.

**m used for the paper's claims:** any reviewer recomputing over the sweep will land on m≥15 and the primary at rank 6 of 21 → Holm 0.67. There is no defensible family under which p=0.042 survives.

---

## W3.2 — Random floor and permutation nulls (both benchmarks)

### (a) Random-per-question selector baseline — the honest floor, reported next to the oracle

| Benchmark | oracle | best-fixed | **system actual** | **random floor** (analytic; sim mean ± sd, 95% CI) | system − floor |
|---|---|---|---|---|---|
| RVS subset (n=316) | 51.9 | patch 45.6 | writetime 42.7 | 42.80; 42.79 ± 1.07, [40.8, 44.9] | **−0.1 pt — AT the floor** |
| StreamingBench RT (n=180) | 76.7 | frame 71.7 | writetime 73.9 | 69.58; 69.58 ± 1.28, [67.2, 72.2] | +4.3 pt |

Analytic floor = mean_q(k_q/4) where k_q = number of policies correct on question q; simulation = 10,000 seeded draws (seed 2024). RVS floor ≈ 42.8 because all-right questions (110/316) reward any pick.

**What "system at the random floor" means, stated plainly:** on the RVS subset, per-question random selection among the four grain policies would match the actual system's accuracy (42.8 vs 42.7). The system's fixed planner is not beating chance-level per-question selection on that subset. Combined with (b) below, the oracle-gap "headroom" must be reported as a *bounding/characterization* result only (see W3.6 framing in work3.md), never as harvestable.

### (b) Permutation nulls for the oracle gap (10,000 perms, seed 2024)

**Null 1 (primary) — shuffle each policy's correctness column across questions** (preserves per-policy marginals, destroys per-question correlation):

| Benchmark | observed oracle | null oracle mean ± sd | null q95 / max | gap oracle−system: obs, p | gap oracle−best-fixed: obs, p |
|---|---|---|---|---|---|
| RVS | 51.9 | 89.3 ± 1.4 | 91.5 / 94.0 | +9.2, **p=1.000** | +6.3, **p=1.000** |
| StreamingBench | 76.7 | 99.2 ± 0.7 | 100 / 100 | +2.8, **p=1.000** | +5.0, **p=1.000** |

Interpretation: policy correctness is *strongly positively correlated* across questions. Under an independence null the oracle max would sit at 89–99%, far above the observed 52/77. The observed gap is therefore **not** an inflation artifact of the independence kind — but it is also **not** evidence of exploitable structure, as Null 2 shows.

**Null 2 (tightest) — within each question, permute which policy gets which correctness value** (per-question correct-count k_q fixed; oracle invariant by construction):

| Benchmark | observed gap oracle−best-fixed | null gap mean ± sd | [q05, q95] | observed percentile | P(null ≥ obs) |
|---|---|---|---|---|---|
| RVS | +6.3 pt | 7.8 ± 0.6 pt | [6.6, 8.5] | **1.1th** | 0.990 |
| StreamingBench | +5.0 pt | 5.6 ± 0.7 pt | [4.4, 6.7] | **12.6th** | 0.871 |

**Interpretation (must go in the paper):** given only how many policies get each question right (k_q), a *random* per-question assignment of correctness to policy labels already produces an oracle−best-fixed gap of ~7.8 pt (RVS) / ~5.6 pt (StreamingBench) on average. The observed gaps (+6.3 / +5.0) sit at or below the low end of those nulls. **The data contain no per-question policy-assignment structure beyond the k_q distribution; the oracle gap is statistically indistinguishable from a max-over-exchangeable-policies artifact.** This closes the loop with the learnability probe (features don't predict winners either): on both benchmarks the honest statement is the W3.6 framing — "per-question grain selection carries ~5–9 pt that no fixed policy can access, and we show it is not predictable from question metadata, store statistics, or chance structure alone."

---

## W3.3 — Ceiling analysis: partition and the all-wrong autopsy

### Partition (4 policies per question)

| Set | RVS subset (n=316) | StreamingBench (n=180) |
|---|---|---|
| all-right | 110 (34.8%) | 113 (62.8%) |
| mixed (1–3 correct) | 54 (17.1%) | 25 (13.9%) |
| **all-wrong** | **152 (48.1%)** | **42 (23.3%)** |

The mixed set is the entire addressable headroom for any selection mechanism: 54 questions (RVS) / 25 (StreamingBench). The all-wrong set bounds *any* mechanism on this data: no grain policy, learned or hand-crafted, can touch 48% of RVS questions.

### All-wrong autopsy — three-way breakdown with documented proxies

Documented proxies (all verifiable from run artifacts):
- **Block→time:** frame block b ↔ [4b, 4b+4) s on RVS (effective 0.25 fps: npy decoded at 0.5 fps, then `load_video` linspace-resamples ×0.5 again — verified against max retrieved block ≈ duration/4 and store `n_segments=225` for 3600 s). On StreamingBench: [2b, 2b+2) s (0.5 fps mp4 path).
- **Stream frontier:** the harness encodes until `end_time × sample_fps` sampled frames → frontier = **2×end** on RVS (real seconds; a documented units quirk — the window is always fully streamed), **end** on StreamingBench. The last `n_local=15000` tokens (76 frames = 304 s RVS / 152 s StreamingBench) are additionally resident in the local cache at answer time and count as context-present.
- **Store retention:** per-segment budget (TAX×784 tokens), **no cross-segment eviction** (`decaf_b.py`) → at tax0.5 every streamed segment retains ≥1 grain; true zero-coverage of a streamed window is structurally impossible. Logged-coverage proxy: window segment ids vs the union of seg_ids in the arm's `commit_log` (top-32 `seg_scores` + `commit` lists) across that video.
- Chance-level retrieved-window overlap (64 random blocks): RVS 0.071, StreamingBench 0.153.

**RVS all-wrong (152), best-policy (patch) arm:**

| Category | n | % of all-wrong |
|---|---|---|
| (i) retrieval failure — no context in retrieval ∪ local cache | 0 | 0% |
| (ii) store-coverage (logged) failure — context present, but the window's segments never appear in any logged store interaction of that video | 14 | 9.2% |
| (iii) model-capability failure — window segments in logged store interactions, all 4 policies still wrong | 138 | 90.8% |

**StreamingBench all-wrong (42):** 100% model-capability, 0 retrieval failures — **by construction**: every RT evidence window is ≤60 s and ends exactly at the stream frontier, and the local cache holds the last 152 s, so the gold window is *always fully visible without any retrieval*. On StreamingBench the store/retrieval machinery is never the binding constraint for these questions. (Store coverage not assessable on xbench: commit logs carry no seg ids.)

**The uncomfortable texture inside (iii):** window coverage for RVS all-wrong questions is low — mean 0.21, median 0.13 of the window seconds covered (retrieval ∪ local), only 11.8% ≥ half-covered — barely above the 0.071 chance level of retrieved overlap. "Model-capability failure" therefore splits in practice into *fine-grained temporal questions where a little overlapping context was present but insufficient* (dominant: 83/138 are temporal-before/after order-judging questions, mean window 509 s, only ~1/8 covered) and *questions where the model genuinely fails with the window in hand*. The 14 logged-coverage failures are concentrated in `other-what` scene-summary questions (11/14) — see examples.

**Per-category example questions** (full lists: `w33_ceiling/w33_examples.json`):

- **(ii) store-coverage (logged), RVS:** "What ingredients are being used?" (ans. "Dough and mixtures in bowls.", window [3240, 3540] of 3600 s video; retrieved blocks confined to [54, 770] = 216–3080 s; window segments 202–221 never logged in any store interaction). Same window: "What equipment is being utilized?", "What is the main activity occurring?".
- **(iii) model-capability, RVS:** "What kitchen fixture is being utilized?" (ans. "Opening a cabinet.", window [0, 300], overlap 0.39, all 4 policies wrong); "Is the flatbread being seasoned with sauce?" (Y/N, window [0, 300], overlap 0.39, all wrong).
- **StreamingBench (all model-capability by construction):** e.g. sb_137 order-judging and scene-description MC questions with full local-window coverage, all 4 policies wrong (see `w33_examples.json`).

**Mixed-set structure (who wins when someone wins):** no type-level separation. On RVS the 54 mixed questions split patch 14 / frame 13 / seg 8 / default 8 (`other-what`) and patch 19 / frame 13 / seg 10 / default 10 (`temporal-beforeafter`) — patch/frame lead but no category is winnable by a single policy. This is the type-level version of the learnability FAIL.

**What follow-on work must attack (the sentence for the executive summary):** the binding constraint on RVS is not grain selection but (1) long-window temporal grounding — 90% of all-wrong questions already have their window segments in the store and the model still fails with ~10–20% window coverage, so retrieval that actually covers 500-s windows, and models that reason over thin coverage, are the levers; and (2) a hard ~48% floor of questions no policy answers, dominated by temporal order-judging. Grain-policy selection addresses none of this — consistent with four failed mechanisms and the learnability FAIL.

---

## W3.4 — Equal-tax integrity loop, closed (no GPU)

### (a) Equal-storage comparison, computed from existing data

The writetime storage–accuracy frontier (realized GB/h, MuKV methodology): tax0.25 → 2.79 GB/h @ 51.8 · tax0.50 → **5.44 GB/h @ 51.1** · tax0.75 → 8.09 GB/h @ 50.2. deferred@tax0.5 realizes **exactly 5.44 GB/h** → the equal-storage comparator is not an interpolation but the existing arm **writetime@tax0.5**:
**equal-storage Δ = 51.81 − 51.06 = +0.75 pt, McNemar p = 0.31 (n=1465) — not significant.** The Gate-2 FAIL ("deferred does not beat writetime at any valid config") is confirmed statistically at equal storage.

Two further hardening facts that must appear in the paper:
1. **writetime@tax0.25 (51.81 @ 2.79 GB/h) Pareto-dominates deferred@tax0.5 (51.81 @ 5.44 GB/h)** — identical accuracy at 49% less storage; deferred vs writetime_tax025 McNemar p=1.0000. The system's flagship arm is dominated by a simpler write-time arm at lower tax.
2. The only nominally significant Pareto pair (deferred@0.5 vs writetime@0.75, +1.64 pt at −33% storage, p=0.042) does not survive multiplicity correction (Holm 0.67, BH 0.15 — W3.1). As a *descriptive* Pareto statement it remains true that deferred@0.5 sits on the writetime frontier's accuracy axis at lower storage than tax0.75.

### (b) Framing paragraph (for the paper)

> Accuracy-vs-storage Pareto claims are cross-configuration **by construction** — each point is a distinct (policy, tax) operating point, and no claim is made about matched-tax superiority beyond the two non-degenerate points we can stand behind. The matched-tax deferred@tax0.75 arm is degenerate by quantization: the per-signal hedged budget ⌊TAX·784/2/196⌋ retains exactly one frame for every TAX in the dead zone [0.5, 1.0), so deferred@0.75 is byte-identical to deferred@0.5 (md5-verified, INTEGRITY_tax075.md) and is not an independent data point. The only non-degenerate equal-tax evidence is therefore tax0.5: deferred 51.8 vs writetime 51.1 (+0.75 pt, p=0.31, n=1465 paired) — statistically indistinguishable, i.e., **deferred commitment matches write-time commitment at equal storage; it does not beat it**. deferred@tax1.0 (the only other non-degenerate deferred operating point) is not yet run and is listed as an open item. Per-arm env dumps + md5-uniqueness assertions on commit logs are now a standing rule for any published sweep table.

### (c) The honest MuKV storage statement

MuKV's published storage is **0.91–1.23 GB/h**; our arms run **fp16 at 5.44 GB/h** (tax0.5). At matched-in-harness accuracy (DECAF 51.8 vs MuKV faithful-config 50.9, p=0.44) **our store is ~4.4–6.0× larger than MuKV's published operating point**. This must be stated plainly; the mitigations are (i) the structural comparison vs ReKV's unbounded offload (ReKV-full 54.1 with no fixed budget — the real Pareto partner), and (ii) the queued int4-with-group-scales fix (TB6: naive per-channel int4 flips 4/5 greedy answers; group scales are follow-up GPU work), which the byte accounting projects to ~4× storage reduction, i.e., rough parity with MuKV's range. Until TB6 lands, "matches SOTA accuracy at ~4.4–6× SOTA storage (fp16), with int4 group-scales queued" is the honest position.

---

## Reproduce

```
python3 deliverables/e3_phase2/analysis/w31_multiple_comparisons.py   # ~10 s
python3 deliverables/e3_phase2/analysis/w32_oracle_null.py            # ~5 min (20k perms x2 benchmarks)
python3 deliverables/e3_phase2/analysis/w33_ceiling.py                # ~30 s
```

Machine-readable: `w31_multiple_comparisons/w31_{family_A,family_B,correction}.json`, `w31_matrix.csv`; `w32_oracle_null/w32_results.json`; `w33_ceiling/w33_{partition,allwrong_breakdown,examples,mixed_by_type}.json`.
