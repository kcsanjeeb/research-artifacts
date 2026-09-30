# W2.3 — Would GOP structure be content-adaptive?

**Date:** 2026-09-26 · **Inputs:** W2.1 per-frame KV residual energies (pre-RoPE K + V, all 24 layers combined: e_t = √(Σ_l ‖ΔK_l‖² + ‖ΔV_l‖²)). No new GPU compute.

## Method

Threshold rule: frame 0 is a keyframe; a new GOP opens at frame t when the cumulative residual energy since the current keyframe exceeds θ. θ swept over 30 points from 780 to 72,693 (log grid spanning p5 → 50×p99.9 of pooled frame energies; mean frame energy: static ≈ 560–740, dynamic ≈ 600–870, ego ≈ 1110–1120). GOP-length distributions (mean/median/p10/p90) per regime per θ in `gop_simulation.json`; figure `gop_length_by_theta.png`.

Reconstruction is *not* the binding constraint anywhere on the grid: keyframe@8-bit + 4-bit-quantized-residual chain gives relative MSE 5e-6 (θ=780) to 6e-3 (θ=53k) vs block energy — far below the 4-bit-raw quantization error. So any θ is "acceptable-reconstruction" and the PASS question reduces to whether regimes separate.

## Result

Mean GOP length (frames @ 0.5 fps; p90 in parentheses):

| θ | static | dynamic | ego | static/dynamic |
|---|---|---|---|---|
| 780 | 1.7 (2) | 1.5 (2) | 1.0 | 1.11 |
| 3,186 | 5.3 (6) | 4.8 (7) | 3.3 | 1.10 |
| 8,142 | 10.5 (15) | 10.9 (15) | 7.8 | 0.96 |
| 13,015 | 14.5 (23) | 16.4 (22) | 12.1 | 0.88 |
| 33,260 | 29.0 (48) | 39.4 (54) | 30.2 | 0.74 |
| 53,170 | 38.7 (58) | 49.2 (79) | 47.4 | 0.79 |

**Static/dynamic mean-GOP ratio ranges 0.74–1.11 across the entire sweep — never remotely near 3×; at large θ dynamic videos get *longer* GOPs than static ones** (anomaly clips contain long static prefixes; the event occupies a minority of the duration, consistent with the 7.8%-anomalous workload measurement).

## Verdict: **FAIL**

KV-residual-driven GOP structure is not content-adaptive; it degenerates to fixed-stride keyframing. Per work2.md, drop the "content-adaptive by construction" claim. This is consistent with the W2.1 root-cause finding: per-frame KV residual energy is dominated by prefix-length/position drift, which accumulates at nearly the same rate in static and dynamic scenes.

## Bonus: GOP boundaries vs ground-truth event boundaries (UCF anomaly)

GT event spans pulled from the V100 server (`san@10.249.185.176:~/e1/smb/inventory_ucf.jsonl`, the only source of UCF temporal annotations; saved locally at `/data3/zhuotaotian2_e2/data/ucf_event_spans.json`): Arrest001 39.5–49.6 s; Arson007 74.7–190.4 s; Fighting003 60.3–102.9 s.

- **Boundary-density enrichment** (GOP-boundary rate inside event ÷ outside): 1.29 / 0.99 / 1.13 (θ ≈ 28–39k, mid-late grid) — ≈1: **no concentration of boundaries at events**.
- At low θ (≈1,160; GOP ≈ 2 frames, a boundary every ~4 s) a boundary falls within ±2 s of the GT event start in all three videos — but at that boundary density the expected chance hit is high (events are 10–116 s long), so this is not evidence of alignment. At θ giving meaningful GOPs (≥10 frames) there is **no boundary within ±2 s of any event start**, and max GOP inside the event equals the global GOP scale (events do not force early GOP closure).
- **Verdict on the bonus: not supported.** KV residual energy yields no usable event-segmentation signal in these clips.

Artifacts: `gop_simulation.json` (full sweep incl. per-regime p10/median/p90) · `gop_length_by_theta.png` · `REPORT.md`.
