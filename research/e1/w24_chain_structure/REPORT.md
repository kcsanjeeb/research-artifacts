# W2.4 — Is there chain structure in retrieval to exploit? (FINAL REPORT)

**Task:** work2.md §W2.4 — under GOP coding, fetching one block makes its GOP-mates
nearly free, so top-*r* selection becomes coverage over chains rather than ranking
over independent items. That only matters if retrieved blocks actually cluster
temporally.

**Method.** From `~/e1/runs/20260925_2237_t1_rekv_sharded/artifacts/t1_rekv_log.jsonl`
(ReKV-0.5B, RVS-Ego, 10 videos × 12 seeded queries = 120 queries, top-64 retrieved
blocks per query, block = 1 frame at 0.5 fps, ~1,806 blocks/video): for each query,
sorted unique retrieved indices were decomposed into maximal runs of consecutive
integers (gap = 1). Metrics: number of contiguous segments, mean run length
(= 64 / n_segments), max run, and fraction of retrieved blocks sitting in runs of
length ≥ 2. Null model: 500 draws per query of the same count (64) sampled uniformly
without replacement from the same video's block range `[0, n_blocks)`.

## Results (pooled over 120 queries)

| Metric | Actual | Null (mean) | Ratio |
|---|---|---|---|
| Mean run length (blocks) | **10.97** | 1.04 | **×10.6** |
| Contiguous segments per query | **31.9** | 61.8 | ×0.52 |
| Max run per query | **19.5** | 1.98 | ×9.8 |
| Fraction of blocks in runs ≥ 2 | **0.637** | 0.165 | ×3.9 |

- **100% of queries (120/120)** have mean run length above the null mean + 1.96σ;
  0% below the null − 1.96σ. Mean z = 409.
- Under the null, 64 uniform draws from ~1,806 blocks almost never produce a single
  adjacency (mean run length 1.04 = 4% of runs reach length 2). Actual retrieval
  produces ~32 contiguous segments of average length ~11 — nearly two-thirds of all
  retrieved blocks are inside multi-block contiguous runs.

## Verdict: **PASS — chain structure is unambiguously real.**

Retrieved blocks cluster temporally at ~10× the null's run length, with the entire
query set individually significant. Chain-aware retrieval (GOP-mate amortization,
submodular coverage over chains) has a genuine empirical basis on this workload.
Caveat: clustering of *retrieval* does not by itself prove GOP *coding* wins — W2.1
still has to show adjacent-block KV is smooth enough to code differentially; and the
runs here (~11 blocks ≈ 22 s at 0.5 fps) are longer than any plausible GOP, so the
coverage structure exists independently of W2.1's outcome.

## Artifacts

- `w24_chain_analysis.py` — analysis script (seed 20260926, 500 null draws/query)
- `chain_structure.json` — per-query stats + null moments + summary
- `chain_structure.png` — actual-vs-null scatter (run length, segments) + z histogram
- Server run dir: `~/e1/runs/20260926_1438_w24_chain_structure/`
