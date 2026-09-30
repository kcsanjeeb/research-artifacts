# T2 — integrity fixes: T2.4 honest metric reporting

Date: 2026-09-25. No GPU, no new inference. Sources:
`~/e1/runs/20260919_1019_eviction_c3/` (eviction sweep),
`~/e1/runs/20260925_1924_t23_tombstone_matrix/` (full policy×tombstone
matrix, used as cross-check — 0 mismatches on existence accuracy),
`~/e1/smb/queries.jsonl`, `~/e1/runs/20260918_0111_writepath_fullcorpus/`,
`~/FleetVAD/research/e0/results/`.

## T2.4(a) — existence accuracy ALONE per eviction budget

Balanced accuracy (existence ⊕ negation) hides the existence collapse
because fewer retained events mechanically lower the negation false-alarm
rate — the "balanced" number can hold ~0.51–0.53 while existence falls to
~0.03. Measured existence accuracy (n=60 queries per cell; random = mean of
3 seeds):

| policy | 100% | 50% | 25% | 10% | 5% | 2% |
|---|---|---|---|---|---|---|
| none | **0.950** | — | — | — | — | — |
| recency | | 0.317 | 0.183 | 0.117 | 0.050 | 0.033 |
| random (3 seeds) | | 0.556 | 0.350 | 0.161 | 0.083 | 0.050 |
| salience | | 0.600 | 0.283 | 0.183 | 0.133 | 0.033 |
| coverage | | 0.683 | **0.333** | 0.150 | 0.117 | 0.083 |

The queue's claim is verified: existence falls **0.950 → 0.333 at 25%
retention** (none vs coverage, the policy the headline used). Against
recency it is 0.950 → 0.183. At 2% every policy is ≤0.083.

Negation false-alarm rate (the confound) for the same cells — it falls
*with* eviction, flattering balanced accuracy:

| policy | 100% | 50% | 25% | 10% | 5% | 2% |
|---|---|---|---|---|---|---|
| none | 0.44 | — | — | — | — | — |
| recency | | 0.14 | 0.06 | 0.04 | 0.00 | 0.00 |
| random | | 0.27 | 0.15 | 0.09 | 0.03 | 0.00 |
| salience | | 0.24 | 0.14 | 0.08 | 0.04 | 0.00 |
| coverage | | 0.26 | 0.12 | 0.06 | 0.06 | 0.06 |

**Required paper fix (Tables 2, 7):** report existence accuracy and negation
FAR as separate columns per budget; do not report balanced accuracy alone.

## T2.4(b) — cross-camera accuracy: **BLOCKED (input missing)**

SMB declares 30 `sht/cross_camera` queries (80 SHT queries total in
`queries.jsonl`). They were **never evaluated**, and cannot be with the
existing artifacts:

- 0 `memory_sht_*` files among the 1,090 memory files of the full-corpus
  write-path run — SHT was never ingested (UCF+XD only).
- 0 `cross_camera` lines in any of the 19 `smb_results_*.jsonl` files across
  all runs (`20260918_0111_writepath_fullcorpus/query_eval`,
  `20260918_0802_v01_tagging/query_eval`); the eviction harness explicitly
  filters to `dataset ∈ {ucf, xd}`.
- No SHT tier-1 scores: `~/FleetVAD/research/e0/results/tier1_scores.npz`
  covers 290 UCF test videos only. `sht_features.npz` (107 videos) holds E0
  scene/motion *redundancy* features, not tier-1 CLIP anomaly scores, and no
  SHT CLIP snippet features exist in the E0 data dir for ingestion.

Per standing rules this is reported as BLOCKED, not substituted. Unblocking
requires: tier-1 scoring of the 107 SHT videos (CLIP features + VadCLIP
head) and ingestion into a memory, then evaluating the 80 SHT queries
(30 cross_camera among them).
