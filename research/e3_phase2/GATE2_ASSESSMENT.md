# DECAF Phase 2 — Gate 2 reassessment (final)

**Date:** 2026-09-28 · **Run:** `runs/20260928_0315_e3_phase2_compB` · **Code:** `code/decaf @93014b3`
**Harness:** LLaVA-OneVision-Qwen2-7B answering; every number judged by the same in-harness
Qwen2.5-72B TP=4 judge; 0 unparsed in all arms. Base: corrected Component A (Gate 1 PASS 54.1/2.82).

## 1. Sweep table (full RVS-Ego, 1465q)

| Arm (debias, fp16 store, 64-slot commit) | Acc | Score | Store GB/h | deferred−writetime @ equal tax |
|---|---|---|---|---|
| stock ReKV 7B (W2.5 anchor) | 54.1 | 2.82 | — | — |
| writetime tax 0.25 | 51.8 | 2.745 | 2.79 | — |
| **deferred tax 0.25** | **35.9** | 1.889 | **0.14** | **−15.9 (INVALID — see §3)** |
| writetime tax 0.50 | 51.1 | 2.717 | 5.44 | — |
| **deferred tax 0.50** | **51.8** | **2.745** | 5.44 | **+0.8 (Gate 2 part 1: FAIL, needs ≥+2)** |
| writetime tax 0.75 | 50.2 | 2.674 | 8.09 | — |
| **deferred tax 0.75** | **51.8** | **2.745** | **5.44** | **+1.6 (FAIL)** |
| writetime RAW bias tax 0.50 (revised Gate 1 leg) | 52.1 | 2.740 | 5.44 | debias−raw = −0.3 → **null** |
| MuKV paper-config (context) | 50.9 | 2.67 | 0.91–1.23 | — |

**No valid config reaches deferred − writetime ≥ +2.** Best observed: +1.6 at tax 0.75
(deferred stores *less* than writetime there — 5.44 vs 8.09 GB/h — and still wins only
1.6 pts). Accuracy is flat (~51–52) across the entire valid tax range for both commit
modes; the 54.1 stock anchor is not recovered by any Component-B configuration.

## 2. Pass-2 (entropy-triggered refinement) — inert

| Arm | pass-2 fired | effect |
|---|---|---|
| deferred tax 0.50 | 158/1465 = 10.8% | on those 157–158q: deferred 62.4% == writetime 62.4% (score 3.153 vs 3.172) — **zero gain where it fires** |
| deferred tax 0.75 | 158/1465 = 10.8% | same commits as tax 0.5 (identical store) — no effect |
| deferred tax 0.25 | 493/1465 = 33.7% | fires 3× more under store starvation; accuracy collapses anyway (starved store, §3) |

The +0.8 at tax 0.5 comes from single-pass questions (+0.6, noise; 94 judge flips total,
both directions). Escalation re-ranks with the same similarity signal and reranker, so it
re-derives the same commits; the trigger detects *model* uncertainty, not retrievable-
evidence gaps.

## 3. The tax-0.25 arm is degenerate by construction (not a mechanism result)

deferred retention = union of two per-signal keep-lists, each budgeted `tax·seg_tokens/2`
(decaf_b.py:287). Segments are 4×196 = 784 tokens, a frame costs 196 tokens, so each
signal needs `tax·392 ≥ 196` → **tax ≥ 0.5 to retain a single frame**. At tax 0.25 the
per-signal budget (98 tokens) cannot afford one frame: the store is structurally empty
(0.14 GB/h = keys only, 0 grains committed). Store sizes confirm the quantization:
deferred keeps 1 frame/signal at both tax 0.5 and 0.75 (identical 5.19 GB/video), vs
writetime's clean scaling (2.66/5.19/8.09). **Findings:** (a) the −15.9 is an
implementation-design defect, excluded from the Gate-2 verdict; (b) the hedged superset
as implemented only exists at tax ≥ 0.5 — the deferred mechanism has no operating point
below half the segment budget, a real design limitation for the C3 storage Pareto claim.

## 4. Clairvoyant oracle (2-video subset, 316q; TAX=1.0 full store, same 64-slot budget)

| Policy (clairvoyance in ranking only) | Acc |
|---|---|
| seg | 41.1 |
| frame | 43.4 |
| patch | 45.6 |
| default | 41.1 |
| **per-question max over policies (true clairvoyant)** | **51.6** (162/314; 2q missing across policy jsons) |

Same-subset actuals: deferred 41.1 · writetime 42.7 · rawbias 43.4.

- **Fixed-policy headroom is small:** best single policy − writetime = +2.9 (patch, 45.6);
  gap vs deferred = +4.4 → gap closure −56% (FAIL, needs ≥+50). Oracle default − deferred
  actual = 0.0: given the *full store*, the planner's own policy class ties the actual
  system — storage is not the binding constraint.
- **Per-question clairvoyant headroom is large:** 51.6 − writetime 42.7 = **+8.9**;
  51.6 − deferred 41.1 = **+10.5**. Subset caveat: n=314, se≈2.8; per-question max across
  four 72B-judged policies carries selection inflation (~+6 over best single policy) — but
  it is the honest upper bound computable from existing artifacts, and it isolates WHERE
  the headroom lives: **per-query grain-policy selection**, not storage, not escalation.
- A clean per-question-argmax over commitment plans is NOT available from existing logs
  (commit logs record one plan per pass; per-question policy judged jsons exist only for
  the 4 fixed policies). The 51.6 number is the clairvoyant figure to report.

## 5. Verdict and recommendation

**Gate 2: FAIL at every valid configuration** (part 1: +0.8 / +1.6, needs ≥+2; part 2 vs
fixed-policy oracle: −56% closure). **Recommendation: ADJUST.**

The deferred thesis is not dead — the hand-crafted planner is. Evidence: the clairvoyant
per-question margin on this store is ~+9–10 pts, an order of magnitude above what any
fixed policy or the default gamma-rerank planner realizes; the mechanism's value was
supposed to be query-time commitment, and an oracle that chooses the grain policy per
query confirms the value is there, but the similarity→rerank→commit pipeline cannot
access it (its own policy class ties actual at 41.1 even with the full store; pass-2
escalation changes nothing). Concretely:

1. **Do not build Component C on this evidence** (Gate 3's stratum would attack a
   constraint that is not binding on RVS-Ego).
2. **Promote the learned commitment scorer (C5, LoRA-scale) from ablation to the core
   Phase-3 experiment:** it is the only plausible way to harvest per-question policy
   selection; Gate: learned scorer ≥ +2 over writetime at equal memory on the subset.
3. **In parallel, test the same oracle+deferred/writetime protocol on StreamingBench /
   OVO-Bench** (counting/text; MuKV CT 39.4) where fine-grained evidence is known to
   gate answers — if clairvoyant headroom there is <2 pts, STOP the deferred thesis and
   narrow the paper to the position-free store + retrieval-precision contribution.
4. Engineering debt to fix before Phase 3: per-signal budget quantization (§3) — charge
   the union budget globally, not per signal, so deferred has valid operating points
   below tax 0.5.

## Appendix — status of in-flight arms

Wave-2 sweep arms (`answers_7b_b_deferred_nopass2` = single-pass tax 0.5;
`answers_7b_b_deferred_ent12` = entropy threshold 1.2) launched 20:39 CST on GPUs 0–1,
~3.5 h ETA, auto-judged by `sweep_chain.sh` → `sweep_verdict.txt`. Expected: single-pass
≈ deferred (pass-2 inert); ent12 = more firing, same null. Oracle crash fixed at
`@93014b3` (slot budget now enforced for all policies; all 4 oracle arms ran 316/316).
Artifacts: `oracle_verdict.txt`, `phase2_verdict.txt`, `sweep_verdict.txt` (pending wave 2),
`analysis/`, per-arm `commit_log.jsonl`. Mac mirror: `research/e3_phase2/`.
