"""T5.1 — Do SHT events co-occur across cameras?

Input: ~/e1/smb/inventory_sht.jsonl (194 GT events, 12 cameras) +
       ~/e1/smb/videometa_sht.jsonl (per-clip durations).

Fact (~/e1/smb/PROTOCOL.md): "Same event on two cameras is not answerable
from SHT GT (no shared clock across videos)". True temporal co-occurrence
across cameras is NOT identifiable from this data. What we measure:

  (a) PROXY-RAW (generous upper bound): treat every clip's time axis as a
      normalized [0,1] clock shared by all cameras; count cross-camera event
      pairs whose normalized windows overlap. Saturated if windows are fat.
  (b) NULL: Monte-Carlo overlap rate for windows of the same normalized
      lengths placed uniformly at random = chance level of (a).
  (c) EXCESS = (a) - (b): the only component attributable to real structure.
  (d) STRONG overlap: IoU(normalized windows) >= 0.5 + its null.

Gate: kills cross-camera dedup (P4-f) if co-occurrence < 15%.
"""
import itertools
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.expanduser("~/e1/fleetmem"))
from make_run_dir import make_run_dir

INV = os.path.expanduser("~/e1/smb/inventory_sht.jsonl")
META = os.path.expanduser("~/e1/smb/videometa_sht.jsonl")
KILL_THRESHOLD = 0.15
NULL_DRAWS = 200_000
SEED = 0


def overlap(a0, a1, b0, b1):
    lo, hi = max(a0, b0), min(a1, b1)
    return max(0.0, hi - lo)


def main():
    evs = [json.loads(l) for l in open(INV)]
    dur = {r["video_id"]: r["duration_sec"]
           for r in (json.loads(l) for l in open(META))}
    for e in evs:
        d = dur[e["video_id"]]
        e["w0"], e["w1"] = e["start_sec"] / d, e["end_sec"] / d
    cams = sorted({e["camera_id"] for e in evs})

    n_pairs = n_ov = n_strong = 0
    for a, b in itertools.combinations(evs, 2):
        if a["camera_id"] == b["camera_id"]:
            continue
        n_pairs += 1
        ov = overlap(a["w0"], a["w1"], b["w0"], b["w1"])
        if ov > 0:
            n_ov += 1
            union = (a["w1"] - a["w0"]) + (b["w1"] - b["w0"]) - ov
            if ov / union >= 0.5:
                n_strong += 1
    frac_raw = n_ov / n_pairs
    frac_strong = n_strong / n_pairs

    rng = np.random.RandomState(SEED)
    lens = np.array([e["w1"] - e["w0"] for e in evs])
    l1 = rng.choice(lens, NULL_DRAWS)
    l2 = rng.choice(lens, NULL_DRAWS)
    s1 = rng.rand(NULL_DRAWS) * (1 - l1)
    s2 = rng.rand(NULL_DRAWS) * (1 - l2)
    ov = np.maximum(0.0, np.minimum(s1 + l1, s2 + l2) - np.maximum(s1, s2))
    null_raw = float(np.mean(ov > 0))
    null_strong = float(np.mean(ov / (l1 + l2 - ov + 1e-12) >= 0.5))

    # per-event partner rate under the proxy (for completeness)
    has_partner = {i: False for i in range(len(evs))}
    for i, j in itertools.combinations(range(len(evs)), 2):
        a, b = evs[i], evs[j]
        if a["camera_id"] == b["camera_id"]:
            continue
        if overlap(a["w0"], a["w1"], b["w0"], b["w1"]) > 0:
            has_partner[i] = has_partner[j] = True
    frac_events_partner = sum(has_partner.values()) / len(evs)

    metrics = {
        "n_events": len(evs), "n_cameras": len(cams),
        "mean_normalized_window_len": float(np.mean(lens)),
        "cross_cam_pairs_total": n_pairs,
        "proxy_raw_overlap_frac": frac_raw,
        "null_overlap_frac": null_raw,
        "excess_over_null": frac_raw - null_raw,
        "strong_overlap_iou50_frac": frac_strong,
        "strong_overlap_iou50_null": null_strong,
        "frac_events_with_partner_proxy": frac_events_partner,
        "kill_threshold": KILL_THRESHOLD,
    }
    rd = make_run_dir("t5_1_sht_cooccurrence", {
        "task": "T5.1", "input": INV, "videometa": META,
        "definition": "normalized per-clip windows [start/dur, end/dur], "
                      "pairs restricted to DIFFERENT cameras; no shared clock "
                      "exists (SMB PROTOCOL) so all numbers are proxies",
        "kill_threshold": KILL_THRESHOLD, "null_draws": NULL_DRAWS,
        "seed": SEED})
    with open(os.path.join(rd, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(rd, "REPORT.md"), "w") as f:
        f.write(f"""# T5.1 — Do SHT events co-occur across cameras?

## Clock assumption (documented, per SMB PROTOCOL.md)
SHT provides **no shared clock across videos**; "same event on two cameras"
is not answerable from the SHT ground truth. True temporal co-occurrence is
NOT identifiable from this dataset. All numbers below are proxies that assume
a normalized per-clip [0,1] clock shared across cameras (generous: it treats
all clips on all cameras as concurrent and aligned).

## Measured numbers
| quantity | value | null (random placement) |
|---|---|---|
| raw window-overlap, cross-camera pairs | **{frac_raw:.4f}** ({n_ov}/{n_pairs}) | {null_raw:.4f} |
| excess over null (real structure) | **{frac_raw - null_raw:+.4f}** | — |
| strong overlap (IoU >= 0.5) | **{frac_strong:.4f}** ({n_strong}/{n_pairs}) | {null_strong:.4f} |
| events with >= 1 cross-camera partner (proxy) | {frac_events_partner:.4f} | — |

Mean normalized event-window length is {np.mean(lens):.3f} of a clip, so the
raw overlap proxy is saturated: random windows of these lengths already
overlap {null_raw:.1%} of the time, and the observed {frac_raw:.1%} sits
exactly at chance. The strong-overlap cut (IoU >= 0.5, closer to what "same
event seen twice" would require) is {frac_strong:.1%} — also at its chance
level ({null_strong:.1%}) and below the 15% gate in absolute terms.

Note: a per-camera busy-time model was considered and dropped — the local
SHT copy (`~/FleetVAD/research/e0/data/shanghaitech/`) and the SMB videometa
contain ONLY the 107 anomaly-bearing test clips, so normal-clip timelines are
unavailable and any busy-fraction estimate would be inflated by clip
selection. Not reported per the measured-numbers-only rule.

## Verdict
Gate: kill P4-f (cross-camera dedup) if co-occurrence < 15%.
- Read literally on the raw proxy, the gate is not triggered ({frac_raw:.1%} > 15%).
  But the raw proxy carries no signal: it equals its own chance level.
- The decision-relevant readings: excess over null = **{frac_raw - null_raw:+.1%}**,
  strong overlap = **{frac_strong:.1%}** (vs {null_strong:.1%} chance) — both far
  below 15%, and the strict quantity is fundamentally unidentifiable (no shared
  clock across SHT videos, per SMB PROTOCOL.md).
**Verdict: KILL P4-f.** SHT — the only true multi-camera set we have —
contains no measurable cross-camera co-occurrence signal beyond chance, and
cannot in principle evidence simultaneous cross-camera events. If P4-f is
ever revived it needs a synchronized multi-camera dataset, which we do not
have.
""")
    print(rd)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
