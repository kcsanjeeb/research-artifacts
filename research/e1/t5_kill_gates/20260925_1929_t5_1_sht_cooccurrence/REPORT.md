# T5.1 — Do SHT events co-occur across cameras?

## Clock assumption (documented, per SMB PROTOCOL.md)
SHT provides **no shared clock across videos**; "same event on two cameras"
is not answerable from the SHT ground truth. True temporal co-occurrence is
NOT identifiable from this dataset. All numbers below are proxies that assume
a normalized per-clip [0,1] clock shared across cameras (generous: it treats
all clips on all cameras as concurrent and aligned).

## Measured numbers
| quantity | value | null (random placement) |
|---|---|---|
| raw window-overlap, cross-camera pairs | **0.5187** (8464/16319) | 0.5050 |
| excess over null (real structure) | **+0.0137** | — |
| strong overlap (IoU >= 0.5) | **0.1026** (1675/16319) | 0.0884 |
| events with >= 1 cross-camera partner (proxy) | 1.0000 | — |

Mean normalized event-window length is 0.247 of a clip, so the
raw overlap proxy is saturated: random windows of these lengths already
overlap 50.5% of the time, and the observed 51.9% sits
exactly at chance. The strong-overlap cut (IoU >= 0.5, closer to what "same
event seen twice" would require) is 10.3% — also at its chance
level (8.8%) and below the 15% gate in absolute terms.

Note: a per-camera busy-time model was considered and dropped — the local
SHT copy (`~/FleetVAD/research/e0/data/shanghaitech/`) and the SMB videometa
contain ONLY the 107 anomaly-bearing test clips, so normal-clip timelines are
unavailable and any busy-fraction estimate would be inflated by clip
selection. Not reported per the measured-numbers-only rule.

## Verdict
Gate: kill P4-f (cross-camera dedup) if co-occurrence < 15%.
- Read literally on the raw proxy, the gate is not triggered (51.9% > 15%).
  But the raw proxy carries no signal: it equals its own chance level.
- The decision-relevant readings: excess over null = **+1.4%**,
  strong overlap = **10.3%** (vs 8.8% chance) — both far
  below 15%, and the strict quantity is fundamentally unidentifiable (no shared
  clock across SHT videos, per SMB PROTOCOL.md).
**Verdict: KILL P4-f.** SHT — the only true multi-camera set we have —
contains no measurable cross-camera co-occurrence signal beyond chance, and
cannot in principle evidence simultaneous cross-camera events. If P4-f is
ever revived it needs a synchronized multi-camera dataset, which we do not
have.
