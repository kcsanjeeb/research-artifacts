
## 2026-09-30 — W5.1 launch (low-tax writetime @ true 0.5 fps) + W5.2 int4 group scales

- **W5.1 run dir**: `runs/20260930_1750_w51_lowtax_fps/`. Arm tax025 (GPU0,
  writetime tax0.25 debias fp16, corrected loader @91fa66b) launched 17:53 CST,
  ETA ~20:50 answers / ~21:05 judged. Arm tax0125_patch (GPU1, writetime
  tax0.125 + DECAF_B_PATCHSTORE=1 patch-granular realized storage) launched
  after code gates.
- **Low-tax validity (explicit, per work5.md)**: frame-granular writetime is
  valid only at tax >= 0.25 (budget < 196 tokens = empty keep-list);
  tax0.125 requires sub-frame realized storage → env-gated
  DECAF_B_PATCHSTORE=1 (49-token slices, honest per-entry byte accounting,
  query-path patch materialization reused). Dead zone is deferred-arm-only:
  confirmed not applicable to writetime.
- **W5.2 implementation** @9413784 (+trapcheck fix): DECAF_QUANT_GROUP
  {64,128} asymmetric int4, ragged last group, per-(head,group,channel) scale
  + zero-point, K/V separate; entry_nbytes honest accounting. Both env-gated,
  default off. Unit: g64 cos 0.9958 (3.56x realized), g128 cos 0.9953 (3.77x);
  TB2 real-capture relerr g64 0.106 mean (PASS<0.30).
- **Verification suite**: `runs/20261001_w52_int4_group/` (GPU2, detached):
  120q single-video, arms fp16/int4legacy/int4g64/int4g128, 72B judge each,
  answer-identity headline (gate >=95%), realized-bytes curve, McNemar vs fp16.
- Watchers: W5.1 `watcher.sh` (abort-without-verdict, fps-assert anchor,
  flock-serialized judge); W5.2 suite self-contained. Judge lock:
  `runs/judge_gpu.lock` (flock) — W5.1 and W5.2 judges cannot collide on
  GPUs 0-3.
- **Ada network outage ~18:20 CST** (100% packet loss; sshd flakiness known,
  this was worse). All processes detached on the box; session-end pushes
  deferred to reconnect. Risk recorded: /dev/shm npy cache is volatile — if
  the box rebooted, arms abort at next decode and watchers record ABORT.
- Gate predictions: tax0.25 ~52-53 acc at ~5.6 GB/h (gate pass expected);
  W5.2 g64/g128 answer-identity >=95% expected (asymmetric groups fix TB6's
  per-channel failure mode).
