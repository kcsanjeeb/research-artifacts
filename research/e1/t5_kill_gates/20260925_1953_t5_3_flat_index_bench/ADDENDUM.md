# ADDENDUM (2026-09-25, same day) — corrected verdict

The auto-verdict in REPORT.md/metrics.json judged only the DEFAULT thread
configuration (faiss max = 72 OpenMP threads on this 72-core shared box).
The thread sweep measured in the same run shows that configuration is
oversubscribed: at 1M records, single-query p99 = 110.6 ms at 72 threads but
**36.5 ms at 32 threads** (p50 29.3 ms), and 2.79 ms/query (358 QPS) in
batch-32 serving.

Corrected gate verdict: flat inner-product search **holds at 1M records**
under the documented p99 < 50 ms bar once threads are set sanely — a config
knob, not a re-architecture. **KILL P5-f (index re-architecture)** at the
≤1M-event scale. Caveats: the 72-thread default DOES break the bar on this
shared machine (110.6 ms), and 100K at 72 threads was 67.3 ms p99 (the
L3-boundary cliff, 15 MB → 154 MB working set); P5-f may revisit beyond 1M
or under heavy CPU contention, but neither is the current regime.
