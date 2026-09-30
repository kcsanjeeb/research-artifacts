# E0 Feasibility Report

> Executed 2026-09-13 on the target server (4x NVIDIA Tesla V100-PCIE-32GB, driver
> 530.30.02, torch 2.6.0+cu124, transformers 4.37.2). All artifacts in
> `research/e0/results/`: `environment.json`, `capacity.json`, `capacity_joint.json`,
> `demand.json`, `redundancy.json`, `redundancy_sht.json`, `figures/`.

## Verdict

**GO — with a rescope of the cache contribution.** Contention exists at a very
plausible scale (p95 escalation demand exceeds the 4-GPU pool at **N=16 streams**),
so the allocation problem is real; but verdict redundancy is overwhelmingly
*within-camera and sub-minute*, and cross-camera reuse — the shared cache's
distinctive bet — is ≈0 at safe thresholds on **both** UCF-Crime and ShanghaiTech.
The arbiter should carry the paper; the cache is a within-stream/temporal
mechanism unless a smarter cross-camera signature can be demonstrated.

## 1. Capacity

- Clip spec (fixed from the repo's own inference path, `inference.py` +
  `holmesvau_utils.py`): 12 frames, uniform sampling (`get_index`), 448x448
  single tile (`max_num=1`), prompt "Could you specify the anomaly events present
  in the video?", greedy decoding. `max_new_tokens=64` (see Surprises/deviations).
- Best single-GPU config: **batch 4, 1 replica** → 0.764 clips/s
  (p50 5.30 s, p95 5.51 s, VRAM 15.3 GB torch / 21.5 GB nvidia-smi).
  Batch sweep: b1 0.526, b2 0.669, b4 0.757, b8 0.749 clips/s; **b16 OOM**.
  Batching plateaus at b4 — the GPU is compute-bound at b≥4 (util ≥96%).
- **CAPACITY (4x V100, measured, not extrapolated): 3.04 clips/s**
  (`pool_throughput_clips_per_s_4gpu = 3.044`).
- V100-specific issues encountered (exactly as E0 3.1 predicted):
  - repo defaults to `torch.bfloat16` → forced `float16` (config patch);
  - `llm_config.attn_implementation` shipped as `flash_attention_2` → forced
    `eager` (FA2 needs sm_80);
  - sentencepiece must be ≤0.1.99 (0.2.x rejects InternLM2's tokenizer.model).
- **Joint batch×replica packing (added beyond the work order):** at b4 a single
  replica already uses ~21.5 GB (nvidia-smi), so 2 replicas OOM. At smaller
  batches replicas *fit* but *thrash*: b1×3 replicas = 0.612 clips/s, b2×2 =
  0.643 clips/s — both **below** b4×1 (0.764). Replicas do not help this model
  on V100; the knee is at 1 replica. Data-parallel scaling across GPUs is
  effectively linear (4 × 0.761 measured = 3.04).

## 2. Contention

- Tier-1 sanity: VadCLIP on official precomputed CLIP features reproduces the
  published number **exactly: AUC = 0.8802** (visual branch; the alignment
  branch gives 0.8569). Scores for all 290 test videos saved
  (`results/tier1_scores.npz`).
- Gate calibration (`tau=0.5`): margin for the MemoVAD-matching 8.6% escalation
  rate is **0.2604** — VadCLIP scores are strongly bimodal, so an 8.6% rate
  needs a wide band. Alternative calibration (see deviations): margin=0.49,
  F1=0.539 at targeting tier-1 errors on the validation half.
- Crossover (`CLIP_PERIOD=1.0s`):
  **N_crossover_p95 = 16, N_crossover_mean = 32.**
  At N=16: mean demand 1.48, p95 4.0 clips/s vs CAPACITY 3.04 → ρ_p95 = 1.31.
  At N=32: mean 3.09, p95 6.0 → ρ_mean 1.02, ρ_p95 1.97.
- Burstiness (p95/mean at P=1.0s): 11.6 at N=1, 2.77 at N=8, 2.71 at N=16,
  1.94 at N=32, decaying to 1.22 at N=256 — demand is bursty exactly in the
  regime where systems get built. Full curve incl. P ∈ {0.5, 2.0}s in
  `demand.json`; figure `figures/demand_load_factor.png`.
- **Gate B verdict: STRONG (N_crossover_p95 = 16 ≤ 64).** Contention appears at
  16 cameras on a 4-GPU server — an entirely plausible deployment. GO.

## 3. Redundancy

UCF-Crime (69,368 causal clips, 32 virtual cameras, k-means k=20 scene proxy,
per-snippet GT labels):

| alpha | theta* | safe hit rate | agreement | T1 | T2 | T3 |
|---|---|---|---|---|---|---|
| 0.0 (motion only) | 0.96 | 0.474 | 0.954 | 0.365 | 0.014 | 0.094 |
| 0.25 | 0.88 | 0.927 | 0.950 | 0.774 | 0.028 | 0.125 |
| 0.5 | 0.80 | 0.995 | 0.974 | 0.963 | 0.014 | 0.017 |
| 0.75 | 0.80 | 0.999 | 0.986 | 0.992 | 0.004 | 0.003 |
| 1.0 (scene only) | 0.80 | 0.999 | 0.986 | 0.992 | 0.004 | 0.003 |

- Dangerous-hit rate (normal neighbour, anomalous clip): 0.71% of all clips at
  the alpha=1.0 operating point.
- Decay of T1 hits: 94.4% within <1 min, 2.8% 1-5 min, 2.2% 5-30 min, 0.6% >30 min.
- Acausal upper bound: 1.000 (labelled; not used for any verdict).
- **T2 = 0.41% < 5%** → per Gate C, Experiment 3 was re-run on ShanghaiTech
  (107 test videos, 12 cameras, one campus; 2,514 clips, 47.5% anomalous):

| alpha | max agreement (theta) | hit rate there | T1 | T1b | T2 |
|---|---|---|---|---|---|
| 0.5 | 0.9307 (0.98) | 0.161 | 0.138 | 0.022 | 0.000 |
| 0.75 | 0.9317 (0.99) | 0.146 | 0.131 | 0.015 | 0.000 |
| 1.0 | 0.8972 (0.99) | 0.449 | 0.422 | 0.027 | 0.000 |

  **No (alpha, theta) reaches the 95% agreement bar on ShanghaiTech**, and T2
  (different camera, same campus — exact here) is 0.0 at every near-safe point
  (≤13% even at the loosest theta=0.80, where agreement is only 75-88%).

- **Gate C verdict: MIXED.** On the rubric's blended UCF number (≥40% safe hit
  rate) the cache looks "strong" — but the decomposition shows ≥96% of that is
  same-video, sub-minute temporal redundancy, which a *per-stream* cache (or any
  temporal gate) already captures. The *cross-camera* premise — the shared
  cache's reason to exist — is not supported by either dataset at safe
  thresholds. Additionally, the 95% safety bar itself is unreachable on
  anomaly-dense data (SHT), meaning `SAFE_HIT_RATE` is base-rate sensitive.

## 4. Surprises (the most valuable section)

1. **The redundancy result cuts against the paper's cache design, not for it.**
   UCF's 99.9% safe hit rate looks like a headline and is nearly meaningless:
   it is dominated by "the previous snippet of the same video" (sim >0.99 for
   adjacent snippets). Any serving stack with per-stream state gets those hits
   for free. If FleetVAD's cache is to be a contribution, it must win *beyond*
   per-stream reuse — and measured T2/T1b says it currently would not.
2. **ShanghaiTech breaks the 95% safety bar entirely.** With 47.5% anomaly
   density, even adjacent-snippet reuse tops out at ~93% label agreement.
   Safety guarantees for verdict reuse are base-rate-dependent; the cache's
   safety law (never suppress a rising edge) is not optional decoration, it is
   load-bearing — and E7 (cache correctness) must be run on anomaly-dense
   workloads, not only UCF-Crime.
3. **Replicas lose to batching on V100.** 2B fp16 at b4 leaves only ~10 GB
   headroom, and 2-3 small-batch replicas thrash below single-replica
   throughput (0.61-0.64 vs 0.76 clips/s). The serving design should plan on
   one worker per GPU with internal batching, not per-request replicas.
4. **Contention arrives earlier than the master plan assumed.** Crossover at
   N=16 (p95) means E4's N sweep {1,...,64} comfortably brackets the
   interesting regime; the `rho >= 1` fallback is unnecessary.
5. **LAVIDA is not runnable** (commit 56b3058): code released, but no
   checkpoints and no data-prep/usage instructions ("Wait for further
   updates"). E0 ran single-detector. For E6 (generality), either re-check
   LAVIDA later or substitute another released VLM-VAD (e.g. VERA/InternVL2-8B,
   listed in the master plan).
6. **VadCLIP scores are bimodal**: matching MemoVAD's 8.6% escalation rate
   needs margin 0.26 (half the score range). Per-stream gates on such scores
   escalate almost exclusively on genuinely ambiguous content — the
   misallocation story (E2.3) must therefore lean on *burst correlation*, not
   on gates firing at random.
7. **Environment friction (documented for reproducibility):** huggingface.co,
   OneDrive API, Dropbox, crcv.ucf.edu, Google Drive, Kaggle and
   openaipublic (CLIP weights) are all unreachable from the server. Worked
   around via hf-mirror.com (datasets + HolmesVAU weights), SharePoint
   `download.aspx` share links (VadCLIP checkpoint), and rebuilding CLIP
   ViT-B/16 from `clipmodel.*` keys inside `model_ucf.pth`. UCF-Crime test
   videos came from `backseollgi/UCF-Crime_TEST_SET` (hf); the CLIP features
   zip is byte-identical in size to the official OneDrive one.

## 5. Deviations from the work order (rule 8)

- `max_new_tokens=64` for the capacity clip spec (repo demo uses 1024). A
  single anomaly *judgement* is short (observed: ~50 tokens); capacity at 1024
  would measure rambling, not serving. Throughput scales ~linearly with output
  length; if a different bound is chosen later, re-run E0.1.
- Gate calibration (b): "margin that maximises tier-1 AUC" is degenerate
  (margin does not change tier-1 scores), so it was implemented as the margin
  maximising F1 of *escalate ↔ tier-1 error* on the even/odd validation half.
- UCF e_scene: VadCLIP's per-snippet CLIP features stand in for "CLIP embedding
  of the clip's middle frame" (E0 3.3 mandates reuse; approximation).
- UCF scene clusters: video-level k-means (k=20) on mean e_scene; k=50 not run
  (T2 is far below 5% at k=20 and cannot cross the threshold at any plausible k;
  noted as a limitation).
- Demand simulation: cameras loop their playlists; horizon =
  max(longest playlist, 900 s). UCF-Crime treated as 30 fps.
- ShanghaiTech: frames used directly (25 fps assumed); added a T1b bucket
  (same camera, different video) since SHT makes the T1/T2 boundary ambiguous;
  CLIP features self-extracted (none exist in VadCLIP format for SHT);
  12 of 13 cameras appear in the test split.

## 6. Recommendation

1. **GO to P1 (reproduction) immediately** — Gate B is strong and the system
   premise (shared budget, per-stream gates misallocate) is intact.
2. **Reframe the cache contribution as within-stream temporal reuse + safety
   law**, and demote "cross-camera reuse" from headline to an open question.
   The arbiter (global budget allocation under bursty contention, N=16
   crossover, burstiness 2.7x) is the primary contribution.
3. Before E4, run one targeted follow-up: test whether a *richer* cross-camera
   signature (e.g. activity-level features rather than CLIP scene + motion
   grid) finds any safe T2 on ShanghaiTech — one day of work, decides whether
   "shared" buys anything over "per-stream".
4. E7 (cache correctness) must include an anomaly-dense workload; the 95%
   agreement bar needs to be restated relative to base rate.
5. For E6 generality, line up VERA (InternVL2-8B) now as the second detector;
   monitor LAVIDA's repo for checkpoints.
