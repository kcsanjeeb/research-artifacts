# Fleet sim v1 — README addendum

`sim_v1.py` — redesigned after v0 (`FLEET_REPORT.md` analysis):

1. **Decoupled virtual clock**: fixed horizon 3600 virtual-s per cell; the
   virtual clock runs at a FIXED accel (2.4x, frozen after calibration) vs wall
   time; arrivals release by the global virtual clock; GPU workers drain the
   admitted queue in wall time. `artifacts/backlog.jsonl` logs backlog(t) —
   the honest overload signal. Metrics cover ALL arrivals (backlogged/dropped
   jobs count as SLO misses).
2. **Controlled acceleration**: accel identical for every cell. Calibration:
   measured service 12.5 wall-s/job/lane; accel=4 makes a zero-wait job take
   50 virtual-s > 60-s SLO (infeasible); accel=2.4 gives 30 virtual-s and puts
   N=8 under capacity (rho~0.75), N=16 ~1.6x, N=24+ clearly over.
3. **Salience weighting**: salience = peak tier-1; bands top-quartile vs rest
   (per cell). Per-band freshness p50/p95 + recall-within-60s-SLO.
4. **arbiter_shed**: admission control — if backlog > B (=12) and salience <
   75th percentile of the cell's saliences, drop at admission; high-salience
   always admitted. Pilot (N=16, accel 4): 341 arrivals, 218 done, 112
   dropped, backlog_end 11, topQ completion 98% vs rest 50%.
5. Teardown crash worked around with `os._exit(0)` after writing results.

Policies: fcfs, arbiter (salience + 0.005/s aging), arbiter_shed.
Sweep: `run_cells_v1.sh` -> `~/e1/runs/20260919_0100_fleet_v1_sweep/<policy>_N<N>/`,
log `sweep_v1.out`.
