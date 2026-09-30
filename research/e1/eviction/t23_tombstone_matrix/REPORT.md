# T2.3 — tombstones ON/OFF × every eviction policy

Date: 2026-09-25. Run: `~/e1/runs/20260925_1924_t23_tombstone_matrix/`. No GPU
inference. Same memory (749 events, `20260918_0111_writepath_fullcorpus` +
tags from `20260918_0802_v01_tagging`), same 220 UCF/XD SMB queries, same
v0.1 final query config as C3 (`20260919_1019_eviction_c3`). Selection rules
reused unchanged from `fleetmem/eviction/eviction_eval.py`
(`t23_tombstone_matrix.py` only changes the cell grid). All C3
tombstones-OFF cells reproduce exactly (0 mismatches on existence accuracy).

Grid: {recency, random(3 seeds, mean), salience, coverage} × {tombstones OFF,
ON} × retention {50, 25, 10, 5, 2}%. 61 cells.

## What tombstones touch

Tombstones only append tag-matched evicted videos to the **retrieval**
ranking, after kept-event videos. They do not change existence, negation,
temporal, or R@10 (verified: identical OFF vs ON in every cell). So the
tombstone question is entirely about retrieval **AP**.

## Retrieval AP: OFF → ON per policy

| retention | recency | random (3 seeds) | salience | coverage |
|---|---|---|---|---|
| 50% | .254 → .249 | .258 → .266 | .287 → .284 | .126 → .193 |
| 25% | .163 → .178 | .147 → .182 | .177 → .200 | .091 → .162 |
| 10% | .077 → .125 | .066 → .116 | .074 → .118 | .053 → .122 |
| 5%  | .060 → .136 | .049 → .120 | .053 → .118 | .035 → .118 |
| 2%  | .047 → .184 | .044 → .176 | .052 → .184 | .027 → .173 |

(Reference, no eviction @100%: AP .353.)

## Does the 6.4× headline survive? **No — not as stated.**

1. The 6.4× was coverage+tombstone vs coverage at 2% (0.173 vs 0.027).
   Coverage-OFF is the **weakest** no-tombstone cell at 2% — the comparison
   was against our own weakest baseline, exactly the criticism T2.3 was
   tasked to check.
2. Against the **strongest** selection policy the tombstone gain at 2% is
   0.052 → 0.184 = **3.5×** (salience). Still large, but not 6.4×.
3. With tombstones ON, **all four policies converge** at 2%: AP 0.173–0.184.
   Coverage+tombstones (0.173) is the *worst* of the four;
   salience+tombstones and recency+tombstones (0.184) beat it. The selection
   policy stops mattering once tombstones carry the retrieval signal.
4. At higher budgets the gain shrinks: 5% ≈ 2.2–3.4×, 10% ≈ 1.6–2.3×,
   25% ≈ 1.1–1.8×, 50% ≈ 0 (salience/recency see no gain at all).
5. Tombstones never fix the recall ceiling: R@10 is unchanged in every cell
   (tombstone videos are appended after kept-event videos, so they rarely
   enter the top-10). The AP gain comes from re-ranking the tail, not from
   recovering top-rank hits.

## Corrected headline

> At 2% retention, per-video tombstones lift retrieval AP from 0.052 to 0.184
> (**3.5×**) against the strongest selection policy (salience), and collapse
> the difference between selection policies (0.173–0.184 across all four).
> The previously reported 6.4× compared against the weakest policy
> (coverage, AP 0.027) and does not survive a full-policy sweep.

## Balanced accuracy (existence+negation), for completeness

Tombstones-independent; matches C3 exactly. Best per budget in **bold**:

| policy | 50% | 25% | 10% | 5% | 2% |
|---|---|---|---|---|---|
| none @100% | .755 | — | — | — | — |
| recency | .589 | .561 | .538 | .525 | .516 |
| random | .644 | .598 | .537 | .528 | .525 |
| salience | .680 | .572 | .551 | **.546** | .516 |
| coverage | **.712** | **.607** | **.545** | .528 | .511 |

Coverage is the best selection policy for *balanced accuracy* at 50–10%,
salience at 5%; differences at 2% are noise (all ≈ .51–.53, i.e. chance
given the negation-side false-alarm confound — see t2_integrity REPORT).
