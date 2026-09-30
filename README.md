# FleetVAD — Research Artifacts

Research code and experiment artifacts for a program on **efficiently serving
vision-language-model (VLM) based video understanding at fleet scale** — many camera
streams sharing one GPU server. The program started from a planned serving system for
VLM-based video anomaly detection (FleetVAD), pivoted on measurement evidence to a
*where-should-the-GPU-budget-go* analysis of that field, and then extended to budgeted,
queryable semantic memory for always-on video fleets (working name FleetMem). A third
line (DECAF) tested inside-the-model streaming video-LLM KV memory against MuKV/ReKV
and closed as a set of rigorous negative results plus a measurement stack. All
experiments run on public datasets (UCF-Crime, XD-Violence, ShanghaiTech, RVS-Ego,
StreamingBench, OVO-Bench) on one 4x Tesla V100 server (E0/E1 lines) and an 8x RTX
6000 Ada box (E3 line). Work was conducted September 2026.

## Repository structure

- `research/e0/` — the multi-stream VAD feasibility measurement line. `src/`
  contains the capacity benchmark (`capacity.py`), demand/contention simulator
  (`demand.py`), cache-redundancy analysis (`redundancy.py`, `redundancy_sht.py`),
  figure generation, and environment collection; `configs/` holds the YAML configs
  for each measurement; `results/` holds the measured JSON outputs (throughput,
  demand load, redundancy curves); `followup/` holds the deeper experiments
  (VLM marginal value, value-function ranking, cross-camera analysis, VERA
  replication, XD-Violence replication). `REPORT.md` is the E0 feasibility report.
- `research/e1/` — the FleetMem semantic-memory line: the Surveillance Memory Bench
  (`smb/`: queries, inventories, protocol), write-path and query-path prototypes
  (`prototype/`, `querypath/`, `querypath_v01/`), the baseline battery
  (`baselines/`: uniform writer, ReKV, VLM-direct, TASTI-lite), fleet-contention
  simulator (`fleet/`), eviction-under-budget studies (`eviction/`), generality arm
  (`generality/`), narration quality spike (`narration_spike/`), kill-gate and
  integrity measurements (`t1_*`–`t5_*`, `w21_*`–`w25_*`), and `EXPERIMENT_LOG.md`
  indexing every run with its config, headline result, and status.
- `research/e3/`, `research/e3_phase0/`, `research/e3_phase1/`, `research/e3_phase2/`
  — the DECAF line. Phase 0: MuKV reproduction and config-fidelity diagnosis.
  Phase 1: the position-disentangled store (Component A) with trap checks and
  corrected rank-remapped fetch. Phase 2: the multi-grain store and deferred
  commitment mechanism (Component B), judged answer sets, oracle arms, and scripted
  statistical analysis (McNemar, multiple-comparison corrections, permutation nulls,
  ceiling decomposition). `research/e3/writing/` holds the paper draft.

## Getting started / reproduction

- Experiment code lives next to its results: each `research/` subdirectory has its
  own scripts and configs; `research/e1/EXPERIMENT_LOG.md` and the per-directory
  `REPORT.md` / `PROTOCOL.md` files document what each run did and with what
  configuration.
- Hardware assumed: NVIDIA V100s (fp16 + eager attention; V100-specific patches for
  the batching path are included in the `t4_batch_microbench/` work) or RTX 6000 Ada
  for the E3 line.
- **Large artifacts are not committed** to this repository: datasets, model
  checkpoints, extracted features/scores (`.npz`, `.pth`, `.pt`), extracted video
  clips (`.mp4`), and third-party codebases (HolmesVAU, VadCLIP, ReKV, MuKV, etc.,
  cloned per their upstream repos). These are reproducible from the scripts here or
  available on request.
- Target datasets are public: UCF-Crime, XD-Violence, ShanghaiTech, RVS-Ego,
  StreamingBench, and OVO-Bench.

## Key findings (as measured, September 2026)

- **Multi-stream contention is real**: a 4x V100 pool sustains ~3 VLM verdicts/s;
  escalation demand crosses capacity at ~16 streams on both UCF-Crime and
  XD-Violence, with demand burstiness ~2.7x (`research/e0/REPORT.md`).
- **The VLM escalation stage is not worth its cost**: called on only ~8.6% of clips,
  it consumes ~53% of the GPU budget while adding only 3–8% of the available accuracy
  headroom — across two datasets and two VLM families.
- **Tier-1 scoring is 4–8x oversampled**: accuracy is flat when analysing every
  4th–8th snippet; dropping the VLM stage and subsampling the cheap stage projects to
  ~8x more cameras at equal accuracy (currently arithmetic; measurement pending).
- **Verdict reuse is within-stream and sub-minute**: the cross-camera shared cache's
  core premise is not supported on either dataset at safe thresholds.
- **FleetMem v0.1** (event-driven VLM narration into tiered text memory) matches
  time-driven recall at ~2.7x lower GPU cost and ~4000x lower storage than KV-state
  baselines on SMB; baseline battery and eviction frontiers are in `research/e1/`.
- **Admission control, not ordering, is the effective lever under saturation**: in
  fleet simulations, salience-weighted load shedding holds freshness SLOs where FCFS
  and priority ordering collapse (`research/e1/fleet/`).
- **DECAF (inside-the-model KV memory) is a negative result**: video KV is not
  differentially compressible; no accuracy claim survives multiplicity correction;
  the binding constraint on retrieval QA is long-window temporal grounding.

## Future plans

1. Build and run the measured multi-stream serving experiment (E0 line): convert the
   cameras-per-server headline from arithmetic to measurement.
2. Work 1 survivors: value-per-byte fidelity ladder, price-domain hysteresis with
   admission lock, deadline-aware batching (with the documented V100 patches).
3. Work 5: storage-parity attempt for the DECAF stack (int4 group scales at true
   0.5 fps) — parity with MuKV-level storage, not an accuracy win.
4. ShanghaiTech ingestion (tier-1 scoring) to unblock cross-camera queries and the
   fleet-quantile gate.
5. Temporal-grounding mechanism direction for streaming memory, per the DECAF
   ceiling diagnosis (Work 4).
6. Paper writing: the E0 measurement paper and the FleetMem systems paper.

## License

No license has been applied. For research and academic review.
