# Fleet sim v1 — run status

- Pilot: `~/e1/runs/20260919_0100_fleet_v1_pilot/` (N=16, arbiter_shed; two
  cells: accel=10 (miscalibrated, discarded) then accel=4.0). COMPLETE.
- Sweep: `~/e1/runs/20260919_0100_fleet_v1_sweep/<policy>_N<N>/`, 15 cells,
  accel=2.4 frozen. Launched ~01:35 via `fleetsim/run_cells_v1.sh`,
  log `fleetsim/sweep_v1.out`. ETA ~6.5-7 h (15 cells x ~27 min).

## Check status

```bash
ssh -i ~/.ssh/24SF51025 san@10.249.185.176
tail -3 ~/e1/fleetmem/fleetsim/sweep_v1.out
find ~/e1/runs/20260919_0100_fleet_v1_sweep -name cell_summary.json | wc -l
```

15 summaries = done. Cells write metrics.jsonl + artifacts/{dropped,backlog}.jsonl
+ cell_summary.json; exit is via os._exit(0) after CELL DONE (rc may be
non-zero from teardown; CELL DONE is the truth marker).
