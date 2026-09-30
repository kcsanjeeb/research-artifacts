# FleetMem — Fleet Contention Experiment v0 (C2) — Honest Report

**Date:** 2026-09-19 (sweep executed 2026-09-18 20:11–22:02 CST)
**Runs:** `~/e1/runs/20260918_1804_fleet_pilot`, `~/e1/runs/20260918_1804_fleet_sweep` (13 cells)
**Data:** `~/e1/fleetmem/fleetsim/fleet_summary.json`, `artifacts_fleet_frontier.png` (mirrored to Mac `research/e1/fleet/`)

## TL;DR

The v0 fleet simulator has a **virtual-time coupling flaw** that makes the N=24/32 cells incomparable to N=4–16, and the experiment shows **no arbiter advantage over FCFS on unweighted metrics** — which is the correct queueing-theoretic outcome (work-conservation) and tells us the arbiter must be evaluated on salience-weighted metrics and load-shedding, not ordering. C2 is **not yet demonstrated**. Two valid findings survive: the end-to-end narration throughput (~12.4 GPU-s/job) and the SLO behavior at low load (99%+ within 60 s at N=4).

## Setup

Discrete-event simulator (`~/e1/fleetmem/fleetsim/sim.py`): N virtual streams replay corpus event streams (union-gate events from run `20260918_0111`), virtual time accelerated, narration jobs processed by real GPU workers (Qwen2.5-VL-7B, `e1qwen`, 3 GPUs — GPU0 squatted by another user). Policies: `fcfs`, `arbiter` (priority = peak score + aging), `arbiter_deg` (arbiter + degrade to 4 frames/job under overload). Synchronized burst injection at N=4/8 (4 bursts). Cells: N ∈ {4,8,16,24,32} × policies, ~510–525 wall-s each.

## Results (per cell; freshness = virtual event→write-completion time)

| Cell | arrivals | done | backlog | virt. horizon (s) | realized accel | fresh p50/p95 (s) | SLO-60 | recall@SLO60 |
|---|---|---|---|---|---|---|---|---|
| fcfs N=4 | 190 | 119 | 71 | 4,934 | 9.6× | 12.5 / 18.9 | 0.992 | 0.621 |
| arbiter N=4 | 190 | 113 | 77 | 4,711 | 9.2× | 12.8 / 18.9 | 0.991 | 0.589 |
| fcfs N=8 | 338 | 115 | 221 | 6,316 | 12.2× | 14.0 / 5,418 | 0.696 | 0.237 |
| arbiter N=8 | 338 | 121 | 216 | 7,156 | 14.0× | 13.8 / 6,270 | 0.686 | 0.246 |
| fcfs N=16 | 704 | 126 | 577 | 5,818 | 11.2× | 14.4 / 5,215 | 0.683 | 0.122 |
| arbiter N=16 | 704 | 122 | 582 | 5,694 | 11.0× | 16.1 / 5,100 | 0.664 | 0.115 |
| fcfs N=24 | 999 | 113 | 885 | 910 | 1.8× | 14.0 / 52.8 | 1.000 | 0.113 |
| arbiter N=24 | 999 | 116 | 881 | 938 | 1.8× | 13.8 / 43.8 | 0.991 | 0.115 |
| arbiter_deg N=24 | 999 | 127 | 871 | 1,006 | 2.0× | 12.1 / 31.6 | 0.992 | 0.126 |
| fcfs N=32 | 1,376 | 120 | 1,256 | 602 | 1.2× | 18.8 / 71.7 | 0.833 | 0.073 |
| arbiter N=32 | 1,376 | 118 | 1,258 | 595 | 1.2× | 21.1 / 81.1 | 0.847 | 0.073 |
| arbiter_deg N=32 | 1,376 | 137 | 1,239 | 713 | 1.4× | 13.2 / 51.2 | 0.985 | 0.098 |

## Analysis

**Flaw 1 — virtual-time coupling (invalidates N=24/32 freshness).** Virtual time advances only as GPU workers drain jobs, so when the GPU saturates, the virtual clock crawls: realized acceleration drops from ~11–14× (N=8/16) to 1.2–2.0× (N=24/32), and the virtual horizon collapses from ~5,700–7,200 s to ~600–1,000 s. The "good" freshness at N=24/32 (p95 ≈ 44–81 s) is an artifact of measuring a queue that never had time to form. Cells are also run at *uncontrolled* accelerations (9.2–14.0× across N=4–16), so even those loads aren't strictly matched.

**Flaw 2 — unweighted metrics hide the arbiter by construction.** At N=8 and N=16 (the comparable cells), arbiter ≈ FCFS on every unweighted metric (SLO-60: 0.686 vs 0.696; 0.664 vs 0.683). Under saturation, all work-conserving policies process the same jobs; priority only reorders *who* waits. Since jobs carry a salience value (peak tier-1 score), the arbiter's benefit must appear in **salience-weighted recall/freshness** — high-value events written promptly at the expense of low-value ones. v0's aggregate metrics cannot see this.

**Finding 1 — real throughput number.** ~12.4 GPU-s per narration end-to-end (8 s inference + overhead; ~120 jobs per 510 wall-s on 3 GPUs). This anchors the fleet capacity math: at the measured write cost (508 GPU-s/stream-h), 3 V100s carry ~21 always-on streams at 1× — consistent with backlog formation appearing at N=8 under 12× acceleration (≈ 96 stream-equivalents of arrival rate).

**Finding 2 — SLO behavior at low load.** At N=4 both policies hold 99%+ of writes within the 60-s virtual SLO (p95 ≈ 19 s); the burst phase barely perturbs this (burst p50 13–15 s, max 45 s). The failure mode between N=4 and N=8 is backlog formation (backlog 71→221), not latency inflation on processed jobs (p50 stays ~14 s everywhere).

**Finding 3 — degradation helps modestly.** `arbiter_deg` (4 frames/job) reduces the freshness tail (N=32: p95 51.2 vs 81.1 s; SLO-60 0.985 vs 0.847) — direction supports fidelity-degradation as an overload mechanism, but the effect is small because job count, not per-job cost, dominates the backlog.

## What C2 needs (v1 design)

1. **Decoupled virtual clock**: fixed virtual horizon per cell (e.g., 3,600 virtual-s), arrivals generated from it, GPU workers processing in wall time; backlog = the honest overload signal.
2. **Controlled acceleration** identical across cells.
3. **Salience-weighted metrics**: recall/freshness within SLO computed per salience band (top-quartile score vs rest); the arbiter should win the top band under saturation even while ties on unweighted.
4. **Admission control / load-shedding**: the mechanism that actually holds SLOs under overload — shed or defer lowest-salience jobs when backlog > threshold; compare fcfs vs priority vs priority+shedding. FCFS cannot shed; this is where scheduling creates capacity.
5. **Query-side SLO**: issue SMB existence queries against the fleet memory mid-run; measure answer staleness as a function of N.

## Operational notes

- rc=134 on some cells is a cosmetic teardown crash (`terminate called without an active exception` — unjoined decord/ffmpeg reader thread); all metrics print before it. Not fixed; harmless.
- The collection agent timed out twice on ssh flakiness; this report was written from `fleet_summary.json` + `sweep.out` pulled to the Mac.


---

# v1 — fixed simulator (2026-09-19)

v0's flaws (virtual clock coupled to lane time, budget-truncated windows,
unweighted metrics, no load shedding) are fixed in `fleetsim/sim_v1.py`:
fixed 3600 virtual-s horizon per cell, virtual clock = accel × wall
(accel=2.4 frozen, see calibration), real-time arrival release, backlog(t)
logged, metrics over ALL arrivals (backlog/drops count as SLO misses),
salience bands (top-quartile tier-1 peak vs rest), new `arbiter_shed`
admission control (drop sub-75th-percentile salience when backlog > B;
high-salience always admitted). Teardown crash bypassed via os._exit(0).

**Calibration** (pilot cells, N=16 arbiter_shed): measured service 12.5
wall-s/job/lane. accel=10 -> rho~4 (0% SLO); accel=4 -> job = 50 virtual-s,
zero-wait alone nearly breaks SLO-60, topQ completion 98% vs 50% under shed;
accel=2.4 frozen -> job = 30 virtual-s, N=8 rho~0.75 (under), N=16 rho~1.6,
N=24+ clearly over.

## v1 sweep results (15 cells + 2 shed-B=4 variants; horizon 3600 virtual-s)

SLO-60 recall over all arrivals / freshness p50 (virtual-s), per policy:

| policy | band | N=4 | N=8 | N=16 | N=24 | N=32 |
|---|---|---|---|---|---|---|
| fcfs | all | .99/20 | .96/24 | .47/64 | .01/434 | .01/886 |
| fcfs | topQ | 1.00/19 | 1.00/23 | .49/60 | .02/479 | .02/795 |
| arbiter | all | .99/21 | .93/24 | .52/56 | .01/501 | .01/802 |
| arbiter | topQ | 1.00/20 | 1.00/24 | .59/51 | .03/512 | .03/703 |
| shed B=12 | all | .99/21 | .93/23 | .56/52 | .01/148 | .01/150 |
| shed B=12 | topQ | 1.00/21 | 1.00/22 | .64/47 | .04/145 | .03/148 |
| shed B=4 | all | — | — | — | .27/67 | .18/69 |
| shed B=4 | topQ | — | — | — | .40/68 | .32/69 |

(Full per-band table incl. rest-band and drops: `fleet_v1_summary.json`.)

## Headline answers (v1)

- **Does arbiter_shed hold SLO-60 for top-quartile at N=16-32 where FCFS
  collapses?** At N=16 (rho~1.6): yes-ish — shed topQ 0.64 vs FCFS 0.49
  (arbiter 0.59). At N=24/32 (rho>=1.9): nobody holds 60s for a majority —
  but shed B=4 lifts topQ SLO from 0.02 (FCFS) / 0.03 (arbiter) to **0.40
  (N=24) / 0.32 (N=32)** and bounds freshness p50 at ~67 virtual-s vs
  434-908 unbounded. Shed converts queue collapse into bounded delay +
  prioritized completion (N=24 B=4: 320 done / 148 dropped; topQ 127/129
  done at B=12).
- **Plain arbiter vs FCFS**: ordering alone gives a small topQ edge under
  mild overload (0.59 vs 0.49 at N=16) and nothing at rho>=2 — admission
  control is the lever that matters, not ordering.
- **Anomaly**: tier-1 peak saliences pile up at ~1.0, so a percentile rule
  on raw peaks sheds weakly (B=12 cells admitted ~69-72% of arrivals, not
  25%); the shed threshold should use rank-normalized salience (v1.1).

## Backlog dynamics (backlog.jsonl)

N=16: backlog oscillates ~5-25 jobs and drains by horizon end for all
policies. N=32: fcfs/arbiter grow ~linearly to ~300 jobs (never drain);
shed pins backlog at ~B for the whole horizon. Figure:
`fleetsim/fleet_v1_curves.png`.

## v1 limitations

- Salience pile-up at peak~1.0 weakens percentile-based shedding (see
  anomaly above). Startup pileup (all streams begin at vt=0) dominates
  early freshness; a staggered-start variant is future work.
- Tier-1 still analytic (354/89.8 GPU-s/stream-h charge, not scheduled in
  lanes); narration is the only real GPU work. accel=2.4 is a modeling
  choice — absolute freshness numbers scale with it; the policy ordering
  (shed > arbiter > fcfs) is robust to it.
- 3 usable GPUs; the v0 transient lane-stall artifact did not recur in v1
  (real-time release removed the coupling that amplified it).
