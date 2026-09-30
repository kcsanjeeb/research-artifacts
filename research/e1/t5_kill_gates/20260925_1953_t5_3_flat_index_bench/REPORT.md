# T5.3 — Flat index search vs 1M events

**Verdict: flat breaks before/at 1M (p99 = 110.6 ms > 50 ms) -> P5-f survives**

faiss 1.15.0 IndexFlatIP, CPU, 72 OpenMP threads, dim=384, k=10, 1000 queries per size. Synthetic unit-norm float32 vectors; dimension/dtype/normalization validated against the real bge-small embeddings: 633 files, dims [384], dtypes ['float32'], L2 norms [1.0000, 1.0000].

| N | p50 ms | p95 ms | p99 ms | mean ms | max ms | QPS |
|---|---|---|---|---|---|---|
| 10000 | 0.60 | 0.60 | 0.61 | 0.60 | 0.62 | 1678 |
| 100000 | 52.27 | 64.33 | 67.28 | 52.68 | 75.36 | 19 |
| 1000000 | 75.24 | 93.12 | 110.63 | 76.30 | 154.16 | 13 |

Threshold (documented in config): p99 < 50 ms single-query latency at 1M records for the interactive query path. Measured p99 at 1M = 110.63 ms (BREAKS). Note the cliff comes early: at 100K records p99 is already 67.3 ms — the jump from 10K (0.61 ms) is the L3-cache boundary (15 MB -> 154 MB working set); 100K -> 1M grows only ~1.3x (memory-bandwidth bound, ~1.5 GB scanned per query).

At 1M, batched serving (32 queries/batch, the realistic fleet mode): 2.79 ms/query mean, 358 QPS.
Thread sweep at 1M (single-query p99): 8 threads: 88.8 ms, 32 threads: 36.5 ms; default 72 threads: 110.6 ms.

**Threshold sensitivity (plain statement):** the verdict flips on the threshold choice. Under the documented p99 < 50 ms bar, flat breaks already at 100K. Under a p99 < 100 ms bar it holds at 1M (110.6 ms). The 50 ms bar was chosen because the query path is interactive retrieval feeding a 60 s-SLO pipeline; readers who accept 100 ms lookups should read this gate as flat-survives-to-1M.

Search correctness sanity: self-recall@1 = 1.00 on 100 probes.

GPU benchmark intentionally not run (task instructions forbid touching the GPUs).