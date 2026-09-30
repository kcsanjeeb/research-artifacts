# T5.3 — Flat index search vs 1M events

**Verdict: flat breaks before 1M -> P5-f survives**

faiss 1.15.0 IndexFlatIP, CPU, 72 OpenMP threads, dim=384, k=10, 1000 queries per size. Synthetic unit-norm float32 vectors; dimension/dtype/normalization validated against the real bge-small embeddings: 633 files, dims [384], dtypes ['float32'], L2 norms [1.0000, 1.0000].

| N | p50 ms | p95 ms | p99 ms | mean ms | max ms | QPS |
|---|---|---|---|---|---|---|
| 10000 | 0.43 | 0.45 | 0.48 | 0.44 | 0.54 | 2286 |
| 100000 | 52.41 | 64.93 | 72.74 | 53.48 | 94.95 | 19 |
| 1000000 | 75.95 | 88.84 | 95.90 | 76.87 | 133.25 | 13 |

Threshold (documented in config): p99 < 50 ms at 1M records for the interactive query path. Measured p99 at 1M = 95.90 ms.

Search correctness sanity: self-recall@1 = 1.00 on 100 probes.

GPU benchmark intentionally not run (faiss-gpu availability not confirmed and task instructions forbid touching the GPUs).