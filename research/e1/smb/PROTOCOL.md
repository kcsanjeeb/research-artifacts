# SMB v0 — Surveillance Memory Bench — Evaluation Protocol

SMB v0 evaluates **memory systems** for always-on surveillance video: after
ingesting streams, can the system answer natural-language questions about what
happened — under a compute/memory budget? v0 is **offline/query-after-ingest**:
all streams are presented (causally) before any query is issued.

## Files

- `inventory_{ucf,xd,sht}.jsonl` — one line per anomaly EVENT: `dataset,
  video_id, category/categories, start_sec, end_sec, duration_sec,
  peak_tier1, mean_tier1, camera_id` (SHT only).
- `videometa_{ucf,xd,sht}.jsonl` — one line per VIDEO: `duration_sec,
  categories_present, categories_absent, is_anomaly_video, camera_id`.
- `queries.jsonl` — one line per QUERY:
  `query_id, dataset, type, text, text_template, scope
  (video|corpus|fleet), answer, evidence_spans, relevant_videos, difficulty,
  label_source, paraphrase_candidates`.
  `text` is the (usually LLM-paraphrased) query to show the system;
  `text_template` is the deterministic slot-filled original.

## Datasets and label sources

| Dataset | Split | Videos | Label source | Taxonomy |
|---|---|---|---|---|
| UCF-Crime | test (VadCLIP list order) | 290 | `Temporal_Anomaly_Annotation.txt` (frame spans, 30 fps) + csv categories | 13 categories |
| XD-Violence | test | 800 | `annotations.txt` (frame spans, 24 fps) + filename `label_*` codes | 6 categories (video-level) |
| ShanghaiTech | testing (anomalous subset only — the 210 normal test videos were never downloaded to this server) | 107 | `test_frame_mask/*.npy` (frame masks, 25 fps) | generic "Anomaly", 12 cameras |

XD caveat: filename labels are video-level and multi-class; span→category
attribution is only unambiguous for single-category videos, so XD existence /
temporal queries are restricted to those. Retrieval uses video-level labels.

SHT caveat: no tier-1 scores exist for SHT in E0 (it was used for the
redundancy experiment only), so `peak_tier1` is null there and SHT queries use
the generic "anomalous event" category.

## Query types and metrics

1. **existence** (75) — scope=video. `answer.value` ∈ {yes}. Metric:
   **accuracy**; reported jointly with negation accuracy (balanced accuracy).
2. **temporal** (75) — scope=video. `answer.spans` = [[start_s, end_s], ...].
   Metric: **tIoU** between predicted span and GT span (union if multiple);
   a prediction is correct if tIoU ≥ 0.3 (report mean tIoU too).
3. **retrieval** (60) — scope=corpus (whole dataset test split) or camera
   subset (SHT). `answer.relevant_videos`. Metric: **Recall@10** and **AP**
   over the system's ranked video list.
4. **cross_camera** (30, SHT only) — scope=fleet. Sub-forms: camera set
   ("which cameras…", metric: set F1), top camera (exact match), pairwise
   comparison (accuracy), camera count (exact match).
5. **negation** (60) — scope=video; queried category is absent from the video
   (UCF/XD: different category present or normal video; SHT: category outside
   the SHT taxonomy). `answer.value` = "no". Metric: **false-alarm rate**
   (fraction answered "yes") and abstention accuracy.

## How a memory system consumes SMB v0

1. Present each dataset's test videos to the system in a fixed order
   (simulated streams; causal within a video). The system may write whatever
   memory representation it wants, subject to its budget.
2. After full ingest, issue `queries.jsonl` (field `text`), scoped per
   `dataset`; the corpus for a query is that dataset's test split.
3. System answers in a JSON-ish form: yes/no for existence/negation,
   [start,end] spans for temporal, ranked video list for retrieval, camera
   set/id/count for cross_camera.
4. Score per type with the metrics above; report per-difficulty breakdown
   (`difficulty` field: short/long × high/low peak tier-1 score).

## Difficulty strata

`short` = event < 10 s, `long` ≥ 10 s; `_lowpeak` = max tier-1 score < 0.8
(harder — weak anomaly signal), `_highpeak` ≥ 0.8. `corpus`/`fleet` = not
event-scoped. `absent_cat` = negation.

## Known limitations of v0

- Query-after-ingest only; no streaming/online queries, no concept of
  query-time latency budget (planned for v1).
- SHT normal test videos absent on this server → SHT negation relies on
  taxonomy mismatch rather than truly normal footage; SHT has no category
  taxonomy (generic "Anomaly").
- XD span→category attribution is video-level; multi-category videos excluded
  from existence/temporal.
- "Same event on two cameras" is not answerable from SHT GT (no shared clock
  across videos), so cross-camera queries are count/set-based.
- Event boundaries follow the official annotations; tier-1 scores are only
  used for difficulty stratification, not for defining GT.
- Narrations/VLM captions are NOT part of v0 ground truth; SMB measures
  retrieval/localization memory, not caption quality (covered by the
  narration spike + rating sheet).
