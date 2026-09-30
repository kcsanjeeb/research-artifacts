"""T5.4 — input audit: do per-clip tier-1 (VadCLIP) anomaly scores exist for
ShanghaiTech test videos?

The gate (AUC of raw vs fleet-quantile-normalized score on SHT) requires
per-clip VadCLIP scores for SHT test videos. This script only AUDITS the
known candidate locations and records what exists. If no SHT tier-1 scores
are found, T5.4 is BLOCKED — computing them is new work outside Work 1 and
is not done here.
"""
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/e1/fleetmem"))
from make_run_dir import make_run_dir

CANDIDATES = [
    "/home/san/FleetVAD/research/e0/results/tier1_scores.npz",
    "/home/san/FleetVAD/research/e0/followup/results/xd_tier1_scores.npz",
    "/home/san/FleetVAD/research/e0/results/sht_features.npz",
]


def probe(path):
    if not os.path.exists(path):
        return {"exists": False}
    d = np.load(path, allow_pickle=True)
    info = {"exists": True, "keys": d.files}
    if "video_ids" in d.files:
        ids = d["video_ids"]
        info["n_videos"] = int(len(ids))
        info["sample_ids"] = [str(x) for x in ids[:5]]
    for k in d.files:
        a = d[k]
        if hasattr(a, "shape"):
            info.setdefault("shapes", {})[k] = list(a.shape)
    return info


def main():
    audit = {p: probe(p) for p in CANDIDATES}
    # broad search for any SHT score artifact anywhere under e0 and ~/e1
    hits = []
    for pat in ["/home/san/FleetVAD/research/e0/**/*sht*score*",
                "/home/san/FleetVAD/research/e0/**/*score*sht*",
                "/home/san/FleetVAD/research/e0/**/sht_tier1*",
                "/home/san/e1/**/*sht*score*",
                "/home/san/e1/**/tier1*sht*"]:
        hits.extend(glob.glob(pat, recursive=True))
    audit["sht_score_file_search_hits"] = sorted(set(hits))
    sht_scores_exist = len(hits) > 0

    rd = make_run_dir("t5_4_fleetquantile_input_audit", {
        "task": "T5.4", "candidates": CANDIDATES,
        "gate": "AUC raw vs fleet-quantile on SHT; kill M7/P2-V2 if gain < 0.02"})
    verdict = ("RUNNABLE" if sht_scores_exist else
               "BLOCKED — no per-clip tier-1 (VadCLIP) anomaly scores exist "
               "for ShanghaiTech test videos. tier1_scores.npz is UCF "
               "(290 videos, VadCLIP sanity AUC 0.8802 per e0 REPORT.md); "
               "xd_tier1_scores.npz is XD-Violence (800 videos); "
               "sht_features.npz holds CLIP features for 107 SHT videos "
               "(redundancy study), NOT anomaly scores. Computing SHT tier-1 "
               "scores is new inference work outside Work 1; not substituted.")
    audit["verdict"] = verdict
    with open(os.path.join(rd, "metrics.json"), "w") as f:
        json.dump(audit, f, indent=2, default=str)
    with open(os.path.join(rd, "REPORT.md"), "w") as f:
        f.write(f"""# T5.4 — Fleet-quantile vs raw score on SHT: **BLOCKED**

{verdict}

## What was checked
- `~/FleetVAD/research/e0/results/tier1_scores.npz`:
  {audit[CANDIDATES[0]].get('n_videos')} videos, sample ids
  {audit[CANDIDATES[0]].get('sample_ids')} — UCF-Crime test set
  (290 videos; e0 REPORT.md: VadCLIP sanity AUC1=0.8802 = published UCF number).
- `~/FleetVAD/research/e0/followup/results/xd_tier1_scores.npz`:
  {audit[CANDIDATES[1]].get('n_videos')} videos — XD-Violence.
- `~/FleetVAD/research/e0/results/sht_features.npz`: keys
  {audit[CANDIDATES[2]].get('keys')} — CLIP scene/motion features for
  {audit[CANDIDATES[2]].get('shapes', {}).get('scene', ['?'])[0]} SHT videos
  from the cross-camera redundancy study; no anomaly scores.
- Glob search for any `*sht*score*` / `*tier1*sht*` artifact under
  `~/FleetVAD/research/e0/` and `~/e1/`: {len(hits)} hits.

## Consequence
The T5.4 gate (kill M7/P2-V2 if fleet-quantile gain < 0.02 AUC on SHT)
CANNOT be evaluated with existing artifacts. SHT ground-truth frame masks DO
exist (`~/FleetVAD/research/e0/data/shanghaitech/.../test_pixel_mask/`), so
the gate becomes runnable the moment SHT VadCLIP per-clip scores are computed
— that computation is new work and was deliberately not done here.
Status: BLOCKED, no substitution made.
""")
    print(rd)
    print(verdict)


if __name__ == "__main__":
    main()
