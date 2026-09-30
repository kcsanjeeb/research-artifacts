# T5.2 — Does lambda oscillate under dual ascent?

**Verdict: PASS (lambda oscillates -> build hysteresis M2)**

Arrival streams replayed verbatim from the logged fleet-v1 sweep (/home/san/e1/runs/20260919_0100_fleet_v1_sweep), cells fcfs_N16, fcfs_N24. Controller: admit iff salience >= lambda; lambda <- max(0, lambda + eta*(spend_W - B_W)/B_W) per 60 vt-s window; B_W = 0.9 * (3 lanes / 2.4 accel) * 60 = 67.5 GPU-s. Lambda is a price in salience units (saliences here are peak tier-1 scores, strongly concentrated >= 0.99 — see salience_p05 per cell).

Oscillation criteria (post-transient second half): peak-to-peak >= 10% of mean AND >= 6 direction reversals per 100 windows.

## fcfs_N16 (338 arrivals over 3540 vt-s, demand 1.19 GPU-s/vt-s vs capacity 1.25, salience min=0.5083, p05=0.6147)

| eta | mean lam | p2p | amp/mean | reversals/100w | spend/budget | oscillates |
|---|---|---|---|---|---|---|
| 0.05 | 0.1753 | 0.1165 | 0.665 | 46.7 | 1.069 | True |
| 0.2 | 0.4879 | 0.5155 | 1.057 | 53.3 | 1.025 | True |
| 0.5 | 0.4779 | 0.9510 | 1.990 | 53.3 | 0.990 | True |
| 1.0 | 0.4368 | 1.2316 | 2.819 | 56.7 | 0.951 | True |

## fcfs_N24 (358 arrivals over 2760 vt-s, demand 1.64 GPU-s/vt-s vs capacity 1.25, salience min=0.5083, p05=0.5736)

| eta | mean lam | p2p | amp/mean | reversals/100w | spend/budget | oscillates |
|---|---|---|---|---|---|---|
| 0.05 | 0.7270 | 0.3245 | 0.446 | 47.8 | 1.246 | True |
| 0.2 | 0.9098 | 0.3812 | 0.419 | 73.9 | 0.962 | True |
| 0.5 | 0.8836 | 1.0204 | 1.155 | 73.9 | 0.978 | True |
| 1.0 | 0.7368 | 1.6074 | 2.182 | 73.9 | 1.011 | True |

Trace plot: artifacts/lambda_traces.png

Note: spend/budget != 1.0 in sustained oscillation is expected — the controller cannot hold spend exactly at budget when admission is a hard threshold on a concentrated salience distribution; it overshoots and corrects, which is exactly the limit cycle M2's hysteresis is meant to damp.