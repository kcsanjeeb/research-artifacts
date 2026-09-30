# T1 — Is memory value heavy-tailed? (FINAL REPORT)

**Task:** work1.md §T1 — the deciding measurement for the water-filling
fidelity-allocation thesis. Gate: **PASS** if Gini ≥ 0.4 OR oracle water-filling
beats uniform by ≥ 2 accuracy points at equal bytes; **FAIL** if near-uniform /
< 1 point headroom; 1–2 points inconclusive (SMB side decides).

**Bottom line:** **PASS on the SMB corpus, decisive on both halves of the gate**
(Gini(v/byte) = 0.947; oracle headroom +21 to +33 points). **Mixed on
ReKV/RVS-Ego**: retrieval frequency is extremely concentrated (Gini(f) = 0.96,
96% of blocks never retrieved) and the sampled value Gini passes the gate
(0.57 ≥ 0.4), but the direct uniform-vs-oracle headroom measurement is ~0 points
(< 1 → that half-gate fails there). Per the spec, the SMB side decides
inconclusive cases → **overall T1 verdict: PASS** (build the fidelity ladder),
with the material caveat that the RVS-Ego value signal is noisy under the
token-F1 proxy (36% of LOO deltas negative) and the headroom null result on
their benchmark should be revisited once a GPT-judge harness exists.

---

## Data source 1 — SMB corpus (FleetMem 749 events × 220 SMB queries)

Method (run `20260925_1935_t1_smb_loo`, `t1_smb.py`): retrieval frequency f_j =
top-3 appearances over 220 queries in the v0.1 query config; marginal value v_j
= full leave-one-out balanced-accuracy drop (existence+negation) for every
retrieved event; bytes_j = memory JSONL record + 1536 B embedding. Uniform-vs-
oracle: retention budgets 90/75/50/25/10%, uniform = mean of 3 random subsets,
oracle = keep highest v_j/bytes_j (in-sample oracle — v_j measured on the same
queries used for evaluation, so headroom is an upper bound).

**Numbers** (`smb_summary.json`):

| Quantity | Value |
|---|---|
| Gini(retrieval frequency f_j) | **0.794** |
| Gini(marginal value v_j, clipped at 0) | **0.945** |
| Gini(v_j / bytes_j) | **0.947** |
| Events never retrieved | 72.0% |
| Events with negative LOO value | 2.4% |
| Baseline balanced acc (full memory) | 0.755 |

Uniform vs oracle balanced accuracy (equal bytes):

| Budget | Uniform (3 seeds) | Oracle water-fill | Headroom (pts) |
|---|---|---|---|
| 90% | 0.705 ± 0.014 | 0.919 | **+21.4** |
| 75% | 0.690 ± 0.005 | 0.919 | **+22.9** |
| 50% | 0.632 ± 0.028 | 0.860 | **+22.8** |
| 25% | 0.583 ± 0.009 | 0.853 | **+27.0** |
| 10% | 0.531 ± 0.022 | 0.863 | **+33.2** |

Gate: Gini 0.947 ≥ 0.4 **and** headroom ≥ 2 pts at every budget → **PASS, both
criteria**.

## Data source 2 — ReKV-0.5B on RVS-Ego (competitors' benchmark)

Method (run `20260925_2237_t1_rekv_sharded`, fixing the crashed
`20260925_1932_t1_rekv_logged`): 10 RVS-Ego videos × 12 seeded questions,
sharded 3 ways (4/3/3 videos); offline ReKV encoding at 0.5 fps (block = 196
tokens = 1 frame); per-query retrieved-block logging (topk=64); sampled LOO (30
retrieved blocks/video, seeded per shard) with token-F1 delta. All blocks are
the same size (2,408,448 B of fp16 KV), so Gini(v/byte) = Gini(v).

**Coverage:** 8 of 10 videos, 96 queries, 2,710 LOO evals. Shard2's process
exited silently after its first video (2026-09-25 23:06, no traceback in
`shard2.out`); per instructions no duplicate shard job was launched, so its
remaining 2 videos are missing — documented partial data. Shard0's second video
encode took 43,788 s (~12.2 h, vs ~1,900 s typical) due to a storage outage;
shard1's second video likewise 43,908 s.

**Numbers** (`rekv_summary.json`, `rekv_perblock_summary.json`):

| Quantity | Value |
|---|---|
| Mean token-F1, full memory | 0.235 |
| Gini(retrieval frequency f_j), pooled | **0.964** |
| Gini(f) null model (uniform-random 64-of-~1803 × 12 queries) | 0.701 |
| Blocks never retrieved | **96.2%** |
| Gini(v) sampled blocks only (= Gini(v/byte)) | **0.567** (pooled) / 0.563 (per-block) |
| Gini(v) per-video mean (incl. unsampled zeros) | 0.866 |
| Sampled LOO deltas negative | **36.3%** |
| Mean sampled LOO drop (token-F1) | +0.020 |
| Per-query value concentration (mean top-1 block share) | 0.071 |

Uniform vs oracle (additive approximation over LOO-sampled blocks, drops clipped
at 0; `rekv_summary.json:uniform_vs_oracle_additive`):

| Budget (within sampled blocks) | Uniform (10 seeds) | Oracle water-fill | Headroom (pts) |
|---|---|---|---|
| 75% | 0.1290 | 0.1286 | **−0.04** |
| 50% | 0.1260 | 0.1255 | **−0.05** |
| 25% | 0.1244 | 0.1244 | **0.00** |
| 10% | 0.1237 | 0.1237 | **0.00** |

Gate: Gini(v/byte) = 0.567 ≥ 0.4 → **passes the Gini criterion**; headroom ≈ 0
pts < 1 → **fails the headroom criterion**. Mixed.

Interpretation notes:

- The f-skew is real, not just structural: measured Gini(f) = 0.964 vs 0.701 for
  a null model that retrieves 64 uniformly random blocks per query. 96% of
  blocks are never retrieved by any of the 12 queries.
- But per-query marginal value is *spread thin*, not concentrated: the single
  most valuable sampled block carries only 7.1% of a query's total clipped
  sampled value. High f-skew (few blocks ever retrieved) coexists with low
  per-query value concentration (within a query, many blocks each contribute a
  little). The first fact passes the Gini gate; the second is why oracle
  water-filling shows no headroom.
- 36% negative LOO deltas means the token-F1 LOO signal is noisy at 0.5B
  (removing a block often *changes* rather than degrades the answer), so both
  the Gini(v) and the headroom numbers on this side are measured through a weak
  instrument.

## Caveats (read before citing)

1. **Accuracy proxy is token-F1** on the ReKV side — the GPT-judge harness is
   unreachable from this network, so these numbers are **not comparable to
   published RVS Accuracy/Score** (flagged per work1.md). token-F1 mean is 0.235
   with high variance across videos (0.107–0.460).
2. **ReKV uniform-vs-oracle is an additive approximation**: query F1 with a
   removed set = max(0, F1_full − Σ clipped per-block drops). Marginal drops are
   not additive for redundant blocks (the sum overshoots — note uniform F1 at
   75% budget, 0.129, sits well below full-memory 0.235), and the oracle ranks
   blocks per-video, not per-query. It is a coarse instrument; ~0 measured
   headroom means "no headroom detectable with this instrument", not a proof of
   absence.
3. **SMB oracle is in-sample** (v_j measured on the evaluation queries) →
   headroom is an upper bound.
4. **Shard2 partial** (1/3 videos) and the 12 h encode outage on both other
   shards are documented above; no data was fabricated for the missing videos.
5. Analysis-script fix during this session: `t1_rekv_analyze.py` built its value
   array with `np.zeros_like(f)` (int dtype), truncating all mean drops to 0 →
   reported Gini(v) = 0.0. Fixed to `dtype=float`; corrected value 0.563 agrees
   with the independent pooled script (0.567). The pooled script is the
   authoritative source.

## Verdict

| Source | Gini(v/byte) | Oracle headroom | Gate |
|---|---|---|---|
| SMB corpus (749 events, 220 queries) | **0.947** | **+21.4 to +33.2 pts** | **PASS (both criteria)** |
| ReKV-0.5B / RVS-Ego (8 videos, 96 queries) | **0.567** (sampled) | **~0 pts** | Gini PASS / headroom FAIL → mixed |

**Overall: PASS.** Value heterogeneity is real and heavy-tailed on our own
corpus by a wide margin, and the SMB side is the designated tie-breaker. On the
competitors' benchmark, retrieval *frequency* is heavily skewed and the sampled
value Gini clears 0.4, but no oracle headroom was measurable through the
token-F1/additive instrument — treat the RVS-Ego headroom question as
**unresolved, not failed**, and re-measure with a real judge harness before
claiming the ladder wins on their turf.

## Artifacts

- Server run dir: `~/e1/runs/20260925_2237_t1_rekv_sharded/` (config/env/
  code_state/metrics, `artifacts/t1_rekv_{log,loo}{,_shard*}.jsonl`,
  `shard{0,1,2}.out`)
- Analysis (server `~/e1/t1_value_heterogeneity/`, mirrored to Mac
  `research/e1/t1_value_heterogeneity/`): `smb_summary.json`,
  `rekv_summary.json`, `rekv_perblock_summary.json`, `value_distribution.png`,
  `uniform_vs_waterfill.png` (SMB), `rekv_pooled.png`,
  `rekv_value_distribution.png` (ReKV), scripts `t1_smb.py`, `t1_analyze.py`,
  `make_uw_fig.py`, `t1_rekv_analyze.py` (dtype-fixed),
  `t1_rekv_pooled_analysis.py`


---

## ADDENDUM (2026-09-26, FINAL — full 10/10 video coverage)

The report above was written at 8/10 videos. The dead shard2 process was
subsequently recovered via `~/e1/baselines/t1_rekv_resume.py` (skip-logged-videos
resume; this was recovery of a dead process, not a duplicate job) run
2026-09-26 13:53–14:19 server time, completing its 2 missing videos. Final
numbers over **10/10 videos, 120 queries, 3,401 LOO evals, 300 sampled blocks**
(`rekv_summary.json`, regenerated 14:19 by `t1_rekv_pooled_analysis.py`):

| Quantity | 8-video (above) | **Final 10-video** |
|---|---|---|
| Mean token-F1, full memory | 0.235 | **0.219** |
| Gini(f), pooled | 0.964 | **0.964** |
| Gini(f) null model | 0.701 | **0.699** |
| Blocks never retrieved | 96.2% | **96.2%** |
| Gini(v) sampled (= Gini(v/byte)) | 0.567 | **0.647** |
| Gini(v) per-video mean (incl. zeros) | 0.866 | **0.792** |
| Sampled LOO deltas negative | 36.3% | **45.7%** |
| Mean sampled LOO drop (token-F1) | +0.020 | **+0.009** |
| Per-query top-1 block value share | 0.071 | **0.089** |

Uniform vs oracle (additive, clip0), final:

| Budget | Uniform | Oracle | Headroom (pts) |
|---|---|---|---|
| 75% | 0.1218 | 0.1227 | **+0.09** |
| 50% | 0.1192 | 0.1202 | **+0.10** |
| 25% | 0.1178 | 0.1180 | **+0.02** |
| 10% | 0.1171 | 0.1174 | **+0.03** |

**Verdict unchanged.** Gini(v/byte) = 0.647 ≥ 0.4 (Gini criterion passes);
oracle-vs-uniform headroom ≤ 0.1 token-F1 point at every budget (headroom
criterion fails, < 1 pt). Combined with the decisive SMB side (Gini(vpb) =
0.947, oracle +21.4 to +33.2 pts), the overall T1 verdict remains **PASS**,
with the RVS-Ego headroom question **unresolved** (token-F1 proxy is noisy:
45.7% of LOO deltas negative; additive model is coarse; no GPT-judge harness
on this network).
