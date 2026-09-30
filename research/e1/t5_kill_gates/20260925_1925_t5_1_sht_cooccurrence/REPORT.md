# T5.1 — Do SHT events co-occur across cameras?

**Measured: cross-camera normalized-window overlap = 0.5187
(8464/16319 pairs); 1.000 of events have >=1
cross-camera overlapping partner; 66/66 camera
pairs ever overlap. Null-model (random placement, same lengths) overlap =
0.5050.**

## Clock assumption (documented, per SMB PROTOCOL.md)
SHT provides NO shared clock across videos; "same event on two cameras" is
not answerable from SHT GT. True temporal co-occurrence is therefore NOT
identifiable from this dataset. The number above is a **generous upper
bound**: it pretends every clip on every camera shares one normalized [0,1]
clock (all clips concurrent and aligned). Any real deployment with a shared
wall clock can only do worse than this proxy (only clips that actually
overlap in time can contribute).

## Result
- proxy co-occurrence (fraction of cross-camera event pairs overlapping):
  **0.5187**
- same quantity at event level: 1.0000; at camera-pair level:
  1.0000
- chance level under random placement: 0.5050 — the observed proxy
  is at/near chance, i.e. even the generous bound shows no structure.
- same-camera context: 0/169 intra-video pairs
  overlap (context only, not part of the gate).

## Verdict
Gate threshold: kill P4-f (cross-camera dedup) if co-occurrence < 0.15.
Measured proxy = 0.5187 -> **SURVIVES_PROXY**.
Because (i) the strict quantity is unidentifiable (no shared clock) and
(ii) even the generous proxy is below/at chance, SHT — the only true
multi-camera set we have — provides no support for cross-camera dedup.
P4-f dies on evidence + identifiability grounds.
