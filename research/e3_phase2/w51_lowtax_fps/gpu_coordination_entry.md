
## 2026-09-30 17:53 CST — W5.1/W5.2 launch

- GPU0: W5.1 arm tax025 (encode+answer, ~3h, until ~20:50)
- GPU1: W5.1 arm tax0125_patch (after trap gates; ~3h from launch)
- GPU2: W5.2 verification suite (4 sequential 7B arms, ~25 min each, then
  blocked on flock + GPUs-free for the 72B judge phase)
- GPU3: trap checks only (done 18:15); free for other tenants' work — reserved
  for nothing; W5.1 watcher will grab GPUs 0-3 (flock) when arms finish for the
  judge phase.
- GPUs 4-7: other tenant (unchanged).
- Post-arm judge phases (W5.1 watcher, W5.2 suite) are flock-serialized via
  runs/judge_gpu.lock — no 72B TP=4 collision possible.
