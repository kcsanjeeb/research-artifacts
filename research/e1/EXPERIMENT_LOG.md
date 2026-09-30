# E1 Experiment Log

Convention (from 2026-09-18): every run gets `~/e1/runs/YYYYMMDD_HHMM_<name>/`
with config.json, env.json, code_state.json, data_manifest.json, logs,
metrics.jsonl, cost.json, artifacts/; every run gets an append-only entry here:
run-id | date | purpose / config+env+code refs / headline result / status.

Entries before 2026-09-18 are retroactive: no run dirs were created at the
time; artifacts live in place ("pre-convention").

---

run-id: pre-convention/week1-validation | date: 2026-09-17 | purpose: week-1
critical-path validation — env survey, HF-mirror download test, ReKV smoke on
V100/sm_70, LiveVLM/MuKV/TASTI/VideoRAG ports & assessments | code+env:
~/e1/ReKV @1fd9a3d, ~/e1/LiveVLM @8b84be8, ~/e1/MuKV (main tarball 2026-05-24),
~/e1/tasti (master tarball), ~/e1/VideoRAG (main tarball); envs e1rekv,
e1livevlm, e1tasti, e1videorag; patches ~/e1/patches/*.patch | headline: ReKV
0.5B+7B run on V100 (7B needed fp32-softmax fix, rekv_v100_fp32_attn.patch);
LiveVLM+MuKV smoke PASS; TASTI core PASS / night-street BLOCKED (GDrive);
VideoRAG assessment PASS | artifacts: ~/e1/week1_critical_path.md,
~/e1/week1_part2.md | status: complete (pre-convention, artifacts in place)

run-id: pre-convention/narration-spike | date: 2026-09-17 | purpose: VLM
narration-quality spike — 200 events (UCF/XD, tier-1-gated strata) × 2 models
(InternVL2-8B, Qwen2.5-VL-7B) × 2 prompts (schema/freeform), 800 narrations |
code+env: ~/e1/narration_spike/narrate.py; envs fleetvad-vlm (InternVL2),
e1qwen (Qwen, transformers 4.49.0, fp16/sdpa, frames pre-resized 448x336) |
headline: Qwen schema 8.1s/event p50 7.7s, 21.4GB; InternVL2 6.0s, 18.4GB;
parse 192/200 vs 190/200 | artifacts: ~/e1/narration_spike/{events.jsonl,
results_*.jsonl, spike_report.md, frames/} | status: complete (pre-convention)

run-id: pre-convention/groundedness | date: 2026-09-17/18 | purpose: automatic
groundedness analysis replacing human rating — cross-model judging (400
judgments), label-anchored keyword checks, cross-model agreement | code+env:
~/e1/narration_spike/{judge.py, analysis_bc.py, aggregate.py}; same envs |
headline: Qwen schema 95.5% supported / 0.5% hallucination (strict cross
judge); InternVL2 75.0% / 12.5%; false-alarm on normal 0-7.5%; verdict:
Qwen2.5-VL-7B schema as FleetMem narrator + guardrails | artifacts:
~/e1/narration_spike/{groundedness_report.md, groundedness.jsonl,
groundedness_summary.json} | status: complete (pre-convention)

run-id: pre-convention/smb-v0 | date: 2026-09-17 | purpose: build SMB v0
query/memory benchmark — event inventories (UCF 156 / XD 1238 / SHT 194
events) + 300 queries across 5 types with GT, LLM-paraphrased (238/300) |
code+env: ~/e1/smb/{build_inventory.py, generate_queries.py,
paraphrase_queries.py, qc_check.py}; e1rekv + e1qwen | headline: 300/300
queries QC-verified against raw label files, 0 errors | artifacts:
~/e1/smb/{queries.jsonl, inventory_*.jsonl, videometa_*.jsonl, PROTOCOL.md,
STATS.md} | status: complete (pre-convention)

run-id: 20260918_0047_writepath_pilot | date: 2026-09-18 | purpose: FleetMem
write-path v0 pilot — 5 videos end-to-end (tier-1 replay, gate formation,
Qwen2.5-VL narration, bge embeddings, cost accounting) | config+env+code:
~/e1/runs/20260918_0047_writepath_pilot/{config,env,code_state,data_manifest}.json;
code ~/e1/fleetmem/writepath.py; env e1qwen | headline: 5/5 videos OK, 1 event
each, narrate 7.4s mean, peak 21.6GB | status: complete

run-id: 20260918_0048_writepath_full | date: 2026-09-18 | purpose: FleetMem
write-path full subset (50 videos, 1.82 stream-h) + cost-vs-recall frontier |
config+env+code: ~/e1/runs/20260918_0048_writepath_full/{config,env,
code_state,data_manifest}.json; 3 GPU shards | headline: 40 union events; recall
0.966/0.949/0.949 for loose/calibrated/strict at 530/516/509 GPU-s/stream-h vs
time-driven 1.0 @ 1442; storage ~1.2MB/stream-day | artifacts: frontier.csv,
artifacts/frontier.png, frontier_summary.json, memory_*.jsonl + emb_*.npy |
status: complete

run-id: 20260918_0111_writepath_fullcorpus | date: 2026-09-18 | purpose:
FleetMem write-path scaled to full UCF+XD test corpus (1090 videos, 37.2
stream-h, 4 GPU shards) | config+env+code: run dir config/env/code_state; code
~/e1/fleetmem/writepath.py; env e1qwen | headline: 749 union events narrated,
508 GPU-s/stream-h total write cost, storage 1.13MB/stream-day, ~1.6 GPU-h
narration | status: complete

run-id: 20260918_0111_writepath_fullcorpus/query_eval | date: 2026-09-18 |
purpose: first SMB v0 evaluation of FleetMem query path (220 UCF+XD queries),
mode a retrieval-only vs mode b +VLM verification | code:
~/e1/fleetmem/querypath.py; threshold tau=0.48 (frozen 20% calibration) |
headline: existence 0.95(a)/0.50(b), negation FAR 0.44(a)/0.10(b), temporal
tIoU 0.20, retrieval R@10 0.254/AP 0.141; verification helps abstention,
hurts confirmation | artifacts: query_eval/{smb_results_*.jsonl,
metrics_*.json, threshold.json}; report ~/e1/fleetmem/QUERY_REPORT.md |
status: complete

run-id: 20260918_0802_v01_tagging | date: 2026-09-18 | purpose: FleetMem v0.1
failure-mode fixes: F1 synonym-aware/asymmetric verification, F2 sub-event
span refinement, F3 query expansion + closed-taxonomy category tags (749
events re-narrated, ~1.6 GPU-h) | code: ~/e1/fleetmem/{tag_events.py,
querypath_v01.py}; same 220 SMB queries + frozen split as v0 | headline: R@10
0.254->0.454 (tags), tIoU 0.201->0.286 (combo refinement), bal acc unchanged
~0.755 (syn_sym option: 0.760 at FAR 0.08); targets: retrieval MET, tIoU
>=0.4 NOT met (score plateau), bal>=0.85 NOT met | artifacts:
~/e1/runs/20260918_0802_v01_tagging/ + ~/e1/fleetmem/V01_REPORT.md | status:
complete

run-id: 20260918_0942_b1_uniformwriter | date: 2026-09-18 | purpose: baseline
B1 UniformWriter — caption every 30s chunk of all 1090 videos (4944 chunks,
Qwen freeform, bge embeddings, FleetMem memory format), then SMB eval via same
query path | code: ~/e1/baselines/uniform_writer.py; env e1qwen | headline:
write 318 GPU-s/stream-h (672 w/ tier-1 charge), 5.9MB/day storage; SMB:
exist 0.667, neg FAR 0.54, tIoU 0.149, R@10 0.176 — worse than FleetMem v0.1
everywhere | status: complete

run-id: 20260918_1438_b2_rekv | date: 2026-09-18 | purpose: baseline B2 ReKV
0.5B on 170 video-scoped SMB queries (stream 0.5fps + streaming-KV QA) | code:
~/e1/baselines/rekv_baseline.py; env e1rekv + rekv patches | headline: exist
0.617, neg FAR 0.20, tIoU 0.055; encode 284 GPU-s/stream-h, QA 0.62
GPU-s/query, KV ~4.3GB/h analytic; corpus retrieval ~676 GPU-s/query analytic
scan | status: complete

run-id: 20260918_1435_b3_vlm_direct | date: 2026-09-18 | purpose: baseline B3
VLM-direct (Qwen 32-frame, no memory) on 170 video-scoped queries | code:
~/e1/baselines/vlm_direct.py; env e1qwen | headline: exist 0.633, neg FAR
0.12, tIoU 0.140, 7.4 GPU-s/query; cannot do corpus retrieval | status:
complete

run-id: 20260918_1804_fleet_sweep | date: 2026-09-18 | purpose: fleet contention
v0 (C2): N in {4,8,16,24,32} streams x {fcfs, arbiter, arbiter_deg}, virtual-time
sim + real Qwen narration on GPUs 1-3 | code: ~/e1/fleetmem/fleetsim/sim.py |
headline: VALID: ~12.4 GPU-s/narration throughput; N=4 SLO-60 0.99 both
policies; arbiter==FCFS on unweighted metrics at N=8/16 (work-conservation);
INVALID: N=24/32 cells (virtual-time coupling collapses horizon to ~600-1000s);
deg variant cuts p95 tail ~1.6x | verdict: C2 NOT demonstrated; v1 needs
decoupled clock, controlled accel, salience-weighted metrics, load-shedding |
report: ~/e1/fleetmem/FLEET_REPORT.md | status: complete (negative/mixed,
v1 required)

run-id: 20260919_0100_fleet_v1_pilot | date: 2026-09-19 | purpose: fleet sim
v1 pilot (decoupled virtual clock, salience bands, shed policy) + accel
calibration (10 -> 4 -> 2.4 frozen) | code: ~/e1/fleetmem/fleetsim/sim_v1.py;
env e1qwen, GPUs 1-3 | headline: accel-10 and accel-4 pilots both overload
N=16 (rho 4x / 1.6x); shed works (112/341 dropped at pctl 0.75, topQ
completion 98% vs 50% rest); accel 2.4 frozen so SLO-60 is attainable |
status: complete; sweep running at ~/e1/runs/20260919_0100_fleet_v1_sweep/

run-id: 20260919_0100_fleet_v1_sweep | date: 2026-09-19 | purpose: fleet sim
v1 full sweep — 15 cells (N=4..32 x fcfs/arbiter/arbiter_shed) + 2 shed B=4
cells, decoupled virtual clock accel=2.4, horizon 3600vs, salience bands,
backlog(t) logged | code: ~/e1/fleetmem/fleetsim/sim_v1.py; env e1qwen, GPUs
1-3 | headline: under capacity (N<=8) all policies equal (SLO .93-1.0); N=16
shed topQ 0.64 vs fcfs 0.49; N=24/32 all collapse on SLO-60 but shed B=4
lifts topQ 0.02->0.40/0.32 and caps freshness ~67-69 vs 434-908 virtual-s;
arbiter ~= fcfs (ordering alone weak) | artifacts: fleet_v1_summary.json,
fleet_v1_curves.png, FLEET_REPORT.md v1 section | status: complete

run-id: 20260919_1019_eviction_c3 | date: 2026-09-19 | purpose: C3
eviction-under-budget frontier over the 749-event full-corpus memory: 6
policies x 6 budgets (36 cells), re-eval on 220 UCF/XD SMB queries (v0.1
query config, frozen tau), no GPU | code:
~/e1/fleetmem/eviction/eviction_eval.py | headline: C3 target NOT met as
stated (coverage@25% = 0.607 bal = 80% of v0.1); separation is cleanest in
retrieval (salience/coverage > random > recency) and GT-coverage of rare
events; tombstones give 6.4x AP recovery at 2% retention; recency/FIFO
collapses as predicted | artifacts: eviction_summary.json, metrics.jsonl,
artifacts/eviction_frontier.png, ~/e1/fleetmem/EVICTION_REPORT.md | status:
complete

run-id: 20260919_1028_c4_tasti_lite | date: 2026-09-19 | purpose: C4 —
TASTI-lite (pure CLIP embedding index, 4944 30s chunks, center-frame ViT-B/16,
text-encoder queries, frozen-split tau=0.27) on the same 220 SMB queries |
code: ~/e1/baselines/tasti_lite.py; env e1rekv | headline: exist 0.583, neg
FAR 0.20, tIoU 0.167, R@10 0.269/AP 0.157 — underperforms FleetMem v0.1
(0.95/0.44/0.286/0.454/0.353) everywhere except negation FAR; E0
appearance-finding holds in-system | status: complete

run-id: 20260919_1047_generality_rvs | date: 2026-09-19 | purpose:
generality arm — RVS-Ego (10 videos, 9.6 stream-h, 120 questions): rekv vs
uniform-writes vs novelty-gated writes (CLIP-delta tau=0.12) | code:
~/e1/baselines/rvs_experiment.py | headline: token-F1 0.212 (rekv) / 0.143
(uniform, 851 GPU-s/h) / 0.130 (novelty, 418 GPU-s/h) — write-cost principle
generalizes (half cost, equal accuracy); KV retrieval more accurate on
procedural QA | status: complete

run-id: 20260925_1924_t23_tombstone_matrix | date: 2026-09-25 | purpose:
T2.3 integrity — tombstones ON/OFF x EVERY eviction policy (recency,
random x3, salience, coverage) x retention {50,25,10,5,2}%, 61 cells, same
749-event memory + 220 UCF/XD SMB queries as C3, selection-only no GPU |
code: ~/e1/fleetmem/eviction/t23_tombstone_matrix.py (reuses
eviction_eval.py; C3 tombstones-OFF cells reproduce exactly, 0 mismatches) |
headline: 6.4x tombstone claim does NOT survive the full sweep — it compared
against the weakest OFF cell (coverage@2% AP 0.027); vs the strongest
policy (salience) the gain is 0.052->0.184 = 3.5x; with tombstones ON all
policies converge (AP 0.173-0.184 @2%, coverage+tombstone is worst);
tombstones touch retrieval AP only, R@10/existence/temporal unchanged |
artifacts: metrics.jsonl, matrix_summary.json, REPORT.md | status: complete

task: T2.4 honest metric reporting | date: 2026-09-25 | purpose: report
existence accuracy alone per eviction budget + cross-camera query status; no
new inference | code: /tmp/t24_integrity.py -> ~/e1/t2_integrity/ |
headline: existence falls 0.950 (none) -> 0.333 (coverage@25%), 0.183
(recency@25%), <=0.083 for all policies @2% — balanced acc hid this via the
negation-FAR confound (FAR 0.44 -> ~0 under eviction); cross-camera: BLOCKED
— 30 SHT cross_camera queries never evaluated (0 cross_camera lines in any
of 19 smb_results files), SHT never ingested (0/1090 memory files), no SHT
tier-1 scores exist (tier1_scores.npz = 290 UCF videos only) |
artifacts: ~/e1/t2_integrity/{REPORT.md,metrics.json,config.json,env.json,
code_state.json} | status: complete (cross-camera arm BLOCKED, input missing)

task: T3 M1 sketch surrogate validity | date: 2026-09-25 | purpose: does a
per-camera PQ count sketch over tier-1 pooled CLIP features (+time-of-day
bucket) predict narration redundancy? decides P2-V3 | code:
~/e1/t3_sketch_validity.py; env e1qwen, no GPU | headline: FAIL — best PQ
sketch variant Spearman rho=0.383 (<0.4 gate, n=116 events with >=1 prior
same-camera event; 559/633 cameras have 1 event); oracle unquantized
pooled-CLIP cosine rho=0.416 (marginal); tag fallback rho=-0.02/-0.08 (FAIL)
— marginal-gain mechanism (M1/P2-V3/P2-V7) dead on this corpus per the task
gate | artifacts: ~/e1/t3_sketch_validity/{REPORT.md,metrics.json,
sketch_vs_redundancy.png,artifacts/per_event.jsonl} | status: complete

run-id: 20260925_1929_t5_1_sht_cooccurrence | date: 2026-09-25 | purpose: T5.1
kill gate — do SHT GT events co-occur across cameras (P4-f cross-camera dedup
premise) | code+env: ~/e1/t5_kill_gates/scripts/t5_1_sht_cooccurrence.py, env
e1rekv; input ~/e1/smb/inventory_sht.jsonl (194 events, 12 cams) | headline:
raw normalized-window overlap 0.519 = null 0.505 (excess +1.4pt); strong
IoU>=0.5 overlap 0.103 vs null 0.088; no shared clock exists so strict
co-occurrence unidentifiable -> KILL P4-f | artifacts: run dir +
~/e1/t5_kill_gates/REPORT.md | status: complete (draft run 20260925_1925
superseded, kept)

run-id: 20260925_1931_t5_2_lambda_replay | date: 2026-09-25 | purpose: T5.2
kill gate — does lambda oscillate under dual ascent (M2 hysteresis premise) |
code+env: ~/e1/t5_kill_gates/scripts/t5_2_lambda_replay.py, env e1rekv;
logged arrival streams from 20260919_0100_fleet_v1_sweep fcfs_N16/N24 |
headline: sustained limit cycle at ALL eta in {0.05,0.2,0.5,1.0}, both cells
(p2p/mean 0.42-2.82, 47-74 reversals/100 windows) -> PASS, build hysteresis
M2 | artifacts: run dir incl. artifacts/lambda_traces.png | status: complete

run-id: 20260925_1953_t5_3_flat_index_bench | date: 2026-09-25 | purpose: T5.3
kill gate — does flat IP index break before 1M events (P5-f premise) |
code+env: ~/e1/t5_kill_gates/scripts/t5_3_flat_index_bench.py, env e1rekv
(faiss 1.15.0 CPU), synthetic 384-d unit-norm float32 validated vs real
bge-small embs | headline: p99@1M 110.6ms @72thr (oversubscribed) but 36.5ms
@32thr and 2.79ms/query batch-32 (358 QPS) -> flat holds at 1M under p99<50ms
bar -> KILL P5-f; verdict corrected in run-dir ADDENDUM.md | artifacts: run
dir | status: complete (draft run 20260925_1939 superseded, kept)

run-id: 20260925_1943_t5_4_fleetquantile_input_audit | date: 2026-09-25 |
purpose: T5.4 kill gate — fleet-quantile vs raw score AUC on SHT | code+env:
~/e1/t5_kill_gates/scripts/t5_4_input_audit.py, env e1rekv | headline:
BLOCKED — no per-clip VadCLIP scores exist for SHT test videos
(tier1_scores.npz=UCF 290 vids, xd_tier1_scores.npz=XD 800 vids,
sht_features.npz=CLIP features not scores); M7/P2-V2 undecidable; computing
scores is new work, not done | artifacts: run dir | status: blocked

run-id: 20260925_1935_t1_smb_loo | date: 2026-09-25 | purpose: T1 complement —
per-event retrieval frequency + full LOO marginal value over 749-event FleetMem
memory x 220 SMB queries (no GPU; v0.1-final retrieval config, tau=0.48) |
code: ~/e1/t1_value_heterogeneity/t1_smb.py | headline: Gini(f)=0.794,
Gini(v)=0.945, 72% never retrieved; oracle water-fill beats uniform retention
by +21..+33 bal-acc points (10-90% budgets, in-sample) -> gate PASS |
artifacts: ~/e1/t1_value_heterogeneity/{value_distribution.png,
uniform_vs_waterfill.png, smb_summary.json} | status: complete

run-id: 20260925_1932_t1_rekv_logged + 20260925_2237_t1_rekv_sharded | date:
2026-09-25/26 | purpose: T1 on their turf — ReKV 0.5B on RVS-Ego with
per-query KV-block retrieval logging + sampled leave-one-out (token-F1 drops)
| code: ~/e1/baselines/t1_rekv.py (crashed on index format), sharded fixed
version by parallel session; patch ~/e1/patches/rekv_t1_logging.patch |
headline (5/10 videos, 72 q, 150 sampled blocks, 1725 LOO evals):
Gini(f)=0.964 (96.2% blocks never retrieved), Gini(per-block drop)=0.592,
top-10% blocks carry 41.2% of value | status: analysis complete on partial
coverage; encode continues server-side

run-id: 20260925_2237_t1_rekv_sharded | date: 2026-09-25/26 | purpose: T1
deciding measurement — is retrieval value-per-byte heavy-tailed on ReKV/RVS-Ego
(per-query retrieval logging + sampled LOO, 3 shards, fixes crashed
20260925_1932_t1_rekv_logged) | code+env: ~/e1/baselines/t1_rekv_shard.py, env
e1rekv, ReKV-0.5B topk=64, 30 sampled blocks/video; analysis
~/e1/t1_value_heterogeneity/t1_rekv_pooled_analysis.py + t1_rekv_analyze.py
(int-dtype bug fixed) | headline: 8/10 vids, 96 queries, 2710 LOO evals
(shard2 died after 1/3 vids, no relaunch per instructions); Gini(f)=0.964 vs
null 0.701, 96.2% blocks never retrieved, Gini(v/byte)=0.567 sampled BUT
uniform-vs-oracle additive headroom ~0 pts (36% LOO deltas negative, token-F1
proxy, no GPT-judge harness); combined with SMB side (Gini(vpb)=0.947, oracle
+21..+33 pts) -> overall T1 PASS, RVS headroom unresolved | artifacts: run dir
+ ~/e1/t1_value_heterogeneity/{REPORT.md,rekv_summary.json,
rekv_perblock_summary.json,rekv_pooled.png,rekv_value_distribution.png},
mirrored to Mac research/e1/t1_value_heterogeneity/ | status: complete

| 2026-09-27 18:50 | E3 Phase 1 recovery | "4 dead jobs" forensics: FALSE ALARM — all four measurement runs completed cleanly (EXIT 0, 1465/1465 rows each, 10/10 videos); EXIT lines landed in code/decaf/*.status because the launcher passed relative status paths after the runner cd'd; driver logs empty by design (output -> answer.log). Old gate1 watcher PID 1979329 was looping on never-appearing run-dir status files, killed. Validity verified instead of relaunching: decaf-vs-W2.5 answer match only 4.5%/8.0% (0.5B/7B) proves position-free path was active (flag-off is bit-stock per T5); config == W2.5. Caveat: 100 (0.5B) / 18 (7B) empty preds, concentrated in video cbfeb6c8. ENCODE4 exit-1 root cause = verdict-aggregation crash in decaf_trapcheck.py:215 (float values vs startswith), NOT a mechanism bug — fixed (key-filter), rerun ENCODE4_EXIT 0 / COMPARE2_EXIT 0, T3c encode-invariance bit-exact, ALL 7 TRAP GATES PASS / OVERALL PASS. Fresh gate1 judge watcher launched (run_gate1_judge.sh, setsid, PID 2215640): 72B TP=4 judging of both decaf CSVs + temporal-oracle recall x4 + gate1_verdict.txt/.json. | none (judging 0-3) | 4-7 other tenant |
