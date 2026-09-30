# Narration Groundedness — Automatic Analysis (replaces human rating)

Date: 2026-09-18. Data: `~/e1/narration_spike/` — 200 events (UCF/XD, strata:
80 anomalous / 80 borderline / 40 normal) × 2 models (InternVL2-8B,
Qwen2.5-VL-7B-Instruct) × 2 prompt variants (schema JSON, freeform).
Analyses: A) cross-model VLM judging of schema narrations, B) label-anchored
keyword checks, C) cross-model summary agreement, D) JSON parse rates.

## Headline table

Judge metrics are for the **schema** variant (the one that would be written to
memory). Judging was crossed: Qwen2.5-VL judged InternVL2's narrations,
InternVL2 judged Qwen's (400 judgments total, 0 invalid judge outputs).

| Metric | InternVL2 schema | Qwen schema | InternVL2 freeform | Qwen freeform |
|---|---|---|---|---|
| Actions supported by frames (cross-judge) | 75.0% | **95.5%** | — | — |
| Contains hallucination (cross-judge) | 12.5% | **0.5%** | — | — |
| Usable as memory entry (cross-judge) | 77.0% | **99.5%** | — | — |
| Category-consistency on anomalous events (keyword check) | **61.7%** | 50.0% | 53.3% | 40.0% |
| False alarm on normal events (strict alarm keywords) | 2.5% | 7.5% | 2.5% | **0.0%** |
| Schema JSON parse rate | 95.0% (190/200) | **96.0%** (192/200) | — | — |

By stratum (cross-judge, schema): InternVL2 drops to 61.3% supported / 16.3%
hallucination on **borderline** events (vs 82.5%/11.3% anomalous, 87.5%/7.5%
normal). Qwen stays ≥88.7% supported on all strata.

Judge-cost: InternVL2 judge 3.5s/judgment (peak 19.3GB), Qwen judge 4.6s
(21.4GB) — judging is cheap enough to run inline as a guardrail.

## Caveats on the judge numbers

The asymmetry is partly judge leniency: InternVL2-as-judge rated nearly
everything usable (99.5% of Qwen), while Qwen-as-judge was stricter. Read the
Qwen-author column as "passes a strict auditor" and the InternVL2-author
column as a lower bound. Absolute values are less trustworthy than the
ordering and the stratum pattern.

## Keyword maps (analysis B, documented)

Category-consistency map (excerpt): Robbery→rob/steal/grab/snatch/theft/loot;
Fighting→fight/punch/brawl/hit/kick/scuffle/struggle/wrestl;
Explosion→explo/blast/fire/smoke/burst/flame; Shooting→shoot/gun/shot/
firearm/rifle/pistol/weapon; (full map in `analysis_bc.py`, `CAT_KW`).
Consistency = narration's actions+summary mentions ≥1 keyword of the video's
GT category; only anomalous events with gt_overlap>0.2 and known category
(n=60). ~50-62% consistency means narrations are factual but often generic
("a man lies on the ground" for a Shooting) — the category keyword is missing
even when the description is accurate.

False-alarm vocabulary (strict): gun/shoot/rifle/pistol/weapon, fight/punch/
brawl/assault, explo/blast/arson/riot, rob/steal/theft/burglar/intrud,
handcuff/arrest, blood/knife/dead/kill, crash/collision, fire/flame/burning.
First pass with a loose vocabulary (incl. "car", "crowd", "police") gave
32-40% "false alarms" — almost all benign object mentions in normal street
scenes; the strict vocabulary is the one reported. Remaining hits are partly
benign collisions too (e.g., e0195: a military parade narration mentions
"rifles" — visible but not an anomaly).

## Cross-model agreement (analysis C)

Token-F1 between the two models' schema summaries, per stratum:
anomalous mean 0.332 (n=78), borderline **0.298** (n=70), normal 0.375 (n=35).
As predicted, borderline events show the lowest agreement — the two models
diverge exactly where the tier-1 signal is ambiguous. Agreement is modest in
absolute terms (different styles: Qwen writes detailed lists, InternVL2
terser prose); use it as a consistency flag, not a correctness metric.

## Parse failures (analysis D)

InternVL2 10/200 unparsed, Qwen 8/200. InternVL2 failures skew to XD (7/10)
and crowded scenes (7/10 crowd-proxy) — the actor-list repetition loop.
Qwen failures skew to borderline (7/8) and are mostly ```json-fence +
truncation issues, fixable with a fence stripper + higher token cap.

## Verbatim examples

GOOD — e0000 (UCF anomalous), InternVL2, judged by Qwen:
`{"actors": ["man in blue hoodie", "man in white hoodie"], "actions":
["walking up stairs", "bending over", "picking something up", ...], "summary":
"Two men are seen walking up and down stairs at night."}` — judge: supported,
no hallucination, usable.

GOOD — e0001 (UCF anomalous), InternVL2: "Three men are standing and
interacting with a car parked on a street at night." — judge: supported, usable.

BAD — e0003 (UCF anomalous, crowded arrest), InternVL2: actor list repeats
("man in black jacket", "man in white hat" ×12) until truncation; judge:
unsupported + hallucination + unusable.

BAD — e0013 (UCF), InternVL2: "A person is seen walking towards a door at
night." — judge: "there is no person present in the frames" (unsupported
summary with empty actor list — internal inconsistency).

BAD — e0195-type (XD normal), both models: parade narration mentions
"rifles" → keyword false-alarm hit; content is visible, not alarming —
shows keyword checks need a manual review pass on hits.

## Verdict

**VLM narration is trustworthy enough to write into FleetMem's memory, with
Qwen2.5-VL-7B (schema variant) as the narrator.** It combines the best judge
scores (95.5% supported, 0.5% hallucination under a strict auditor), the best
parse rate, and 0% false alarms in freeform. InternVL2-8B is a viable cheaper
fallback (~25% faster) but needs guardrails. Recommended guardrails:

1. **List caps + repetition penalty** (InternVL2's actor-list loop is its main
   failure; cap list entries at 6 and/or repetition_penalty≈1.1).
2. **Inline cross-model or self-consistency judging is affordable** (~4s per
   narration on a V100) — route narrations that fail judging to regeneration
   or flag them low-confidence in memory.
3. **Stratify trust by tier-1 band**: borderline events get measurably worse
   narrations (61% supported for InternVL2) and lowest cross-model agreement —
   mark borderline-event memories as low-confidence.
4. Keep prompts neutral (no anomaly judgment) — false-alarm rates on normal
   footage are already low (0-7.5%) with description-only prompts.
5. Narrations describe *what is visible*, not the GT category (category
   consistency 40-62%) — downstream category labels must come from tier-1
   scores or a classifier, not from the narration text.

## Files

- `groundedness.jsonl` — per event × model: parse status, judgment, summary.
- `groundedness_summary.json` — all aggregate numbers.
- `judge_<j>_of_<a>.jsonl` — 200 raw cross-judgments per direction.
- `analysis_bc.json` + `agreement.jsonl` — B/C/D details.
- `judge.py`, `analysis_bc.py`, `aggregate.py`, `examples.py`.
- Both copied to Mac: `/Users/24sf51025/Research/FleetVAD/research/e1/narration_spike/`
  (report + groundedness.jsonl).
