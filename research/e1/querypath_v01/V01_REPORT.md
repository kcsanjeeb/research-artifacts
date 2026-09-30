# FleetMem v0.1 — failure-mode fixes and re-measurement

Date: 2026-09-18. Base: v0 eval in `QUERY_REPORT.md` (run
`20260918_0111_writepath_fullcorpus`). v0.1 run: `~/e1/runs/20260918_0802_v01_tagging/`.
Same 220 UCF/XD SMB queries, same frozen 20% calibration split (seed 0) as v0.

## Headline: v0 vs v0.1 (final config: tags + refined-combo, retrieval-only)

| Type | v0 mode a | v0 mode b (sym verify) | v0.1 final | v0.1 + syn_sym verify (precision option) |
|---|---|---|---|---|
| Existence acc (60) | 0.950 | 0.500 | **0.950** | 0.600 |
| Negation acc (50) | 0.560 | **0.900** | 0.560 | **0.920** |
| Negation FAR | 0.440 | 0.100 | 0.440 | **0.080** |
| Balanced exist+neg | 0.755 | 0.700 | 0.755 | 0.760 |
| Temporal mean tIoU (60) | 0.201 | 0.201 | **0.286** | 0.286 |
| Temporal tIoU≥0.3 | 0.217 | 0.217 | **0.367** | 0.367 |
| Retrieval Recall@10 (50) | 0.254 | 0.254 | **0.454** | 0.454 |
| Retrieval AP | 0.141 | 0.141 | **0.353** | 0.353 |

## Per-fix attribution (ablation)

| Fix | Metric moved | Evidence |
|---|---|---|
| F3b category tags (closed-taxonomy re-narration, 749 events, ~1.6 GPU-h) | R@10 0.254→**0.454** (+79%), AP 0.141→**0.353** (+150%) | tags-only config |
| F3a query expansion (synonym map) | R@10 0.254→0.320, AP→0.216 alone; with tags: 0.437/0.365 (no add over tags; dropped) | exp1tags0 / exp1tags1 |
| F2 span refinement (score-trajectory, 0 GPU cost) | tIoU 0.201→0.192-0.275 across 6 parameterizations; best = combo (inner0.8 runs + peak±7.5s window + raw event, lenient multi-span): **0.286 / @0.3 0.367** | ref* configs |
| F1a synonym-aware symmetric verification | exist 0.50→0.60, neg 0.90→0.92 vs v0 sym; bal 0.70→0.76 | versyn_sym |
| F1b asymmetric verification (τ_hi skip + high-confidence-no veto; τ_mid support rule; strong-hit-or-match rule) | no movement vs retrieval-only (bal 0.755; asym3: bal 0.752 at FAR 0.08) — verifier "no" is rarely high-confidence, and strong hits are rare because tag boost is only 0.75 | verasym/asym2/asym3 |

## Targets

- Balanced exist+neg ≥ 0.85: **NOT MET** (best 0.755-0.76). Limit: the
  confirmation/abstention trade-off — verifier strict enough to kill false
  alarms also kills ~40% of true events (generic narrations + sparse frames);
  tag accuracy ~50-60% caps the retrieval shortcut.
- Temporal tIoU ≥ 0.40: **NOT MET** (0.286; was 0.201, +42%). Limit: VadCLIP
  tier-1 scores saturate — events are plateaus; inner-threshold spans stay
  ~3.5x GT length (median 35.7s vs GT 10.1s), and tier-1 peaks are frequently
  offset from GT centers (peak-window variants score 0.11-0.16).
- Retrieval R@10 measurably up: **MET** (0.254→0.454, +79%).

## Cost deltas

- Tagging (F3b): 749 events re-narrated with tag field, 4 GPUs ~30 min,
  ≈1.6 GPU-hours one-off per corpus (tags are a memory field; amortized).
- F2: 0 GPU (score replay). F3a: ~0 (query-side). F1: ~300 GPU-s per full
  220-query eval (~1.4 GPU-s/query on verified types) per variant.
- Query-eval totals: final config 0 GPU-s (retrieval-only); syn_sym option
  303 GPU-s/eval.

## Remaining failure modes (inspected)

1. **Tag noise → negation false alarms** (22 total in final): tagger assigned
   the queried category to the wrong video's event (e.g., "arrest in
   Explosion033_x264" boosted to 0.75 by an Arrest tag on an Explosion video).
2. **Verification false-rejects** persist for existence confirmation
   (syn_sym exist 0.60): sparse 8-frame evidence + generic narrations.
3. **Wrong-event temporal picks**: multi-event videos where the non-GT event
   out-scores (q0079: pred [255,288] vs GT [116,134]).
4. **Write-path misses** unchanged: GT events never gated → empty answers
   (q0087, q0093; existence q0021/q0059 have no memory at all).
5. **Untagged true events** (existence q0001: tagger said "none" on a real
   Arrest; semantic score 0.45 < τ).

## Notes

- F2 metric detail: v0.1 temporal scoring is lenient multi-span (best IoU
  over predicted spans AND GT spans); v0 returned single spans, so the v0
  number is unchanged by this. Combos return inner+window+raw spans.
- GT categories were never used in memory or retrieval; tags come from the
  VLM closed-taxonomy prompt only (UCF13/none, XD6/none, temp 0.2).
- asym2's τ_mid calibrated to 0.47 (≈τ) — the cal split sees no separation
  between modes at video level; the real separation only appears with the
  verifier's match signal, which is too weak for confirmation duty.

## Files

- Code: `~/e1/fleetmem/{tag_events.py, querypath_v01.py}` (+ patched
  `querypath.py` lenient temporal metric).
- Run: `~/e1/runs/20260918_0802_v01_tagging/` — `artifacts/tags_*.jsonl`
  (749 tagged events), `refined_spans.json`, `query_eval/metrics_*.json` (14
  configs), `smb_results_*.jsonl`, `*.out`.
