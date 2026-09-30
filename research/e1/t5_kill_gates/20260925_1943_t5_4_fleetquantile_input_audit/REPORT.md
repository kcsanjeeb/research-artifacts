# T5.4 — Fleet-quantile vs raw score on SHT: **BLOCKED**

BLOCKED — no per-clip tier-1 (VadCLIP) anomaly scores exist for ShanghaiTech test videos. tier1_scores.npz is UCF (290 videos, VadCLIP sanity AUC 0.8802 per e0 REPORT.md); xd_tier1_scores.npz is XD-Violence (800 videos); sht_features.npz holds CLIP features for 107 SHT videos (redundancy study), NOT anomaly scores. Computing SHT tier-1 scores is new inference work outside Work 1; not substituted.

## What was checked
- `~/FleetVAD/research/e0/results/tier1_scores.npz`:
  290 videos, sample ids
  ['Abuse028_x264', 'Abuse030_x264', 'Arrest001_x264', 'Arrest007_x264', 'Arrest024_x264'] — UCF-Crime test set
  (290 videos; e0 REPORT.md: VadCLIP sanity AUC1=0.8802 = published UCF number).
- `~/FleetVAD/research/e0/followup/results/xd_tier1_scores.npz`:
  800 videos — XD-Violence.
- `~/FleetVAD/research/e0/results/sht_features.npz`: keys
  ['scene', 'motion', 'labels', 'meta'] — CLIP scene/motion features for
  107 SHT videos
  from the cross-camera redundancy study; no anomaly scores.
- Glob search for any `*sht*score*` / `*tier1*sht*` artifact under
  `~/FleetVAD/research/e0/` and `~/e1/`: 0 hits.

## Consequence
The T5.4 gate (kill M7/P2-V2 if fleet-quantile gain < 0.02 AUC on SHT)
CANNOT be evaluated with existing artifacts. SHT ground-truth frame masks DO
exist (`~/FleetVAD/research/e0/data/shanghaitech/.../test_pixel_mask/`), so
the gate becomes runnable the moment SHT VadCLIP per-clip scores are computed
— that computation is new work and was deliberately not done here.
Status: BLOCKED, no substitution made.
