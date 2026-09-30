# MuKV-on-Ada runbook (P0-1/P0-2) — execute in order once V100 is back

0. **Confirm the pull landed:** `cat runs/20260927_0000_e3_phase0/logs/v100_pull.status` shows
   `DONE`, `code/MuKV/` exists, `logs/mukv_provenance.txt` records the git HEAD of ~/e1/MuKV.
   Pin that commit in the run config. Kill the watcher if still looping.
1. **Verify GPUs:** `nvidia-smi` — GPUs 0-3 must be free (4-7 = other tenant, never touch).
2. **Patch for Ada (torch fallback, no flash-attn in rekv env):**
   - `cd code/MuKV && patch -p0 < ../patches/mukv_v100_sdpa.patch` (fattn=False + attn_implementation="sdpa").
     If already applied on the V100 copy (check `git status`/`grep fattn`), skip.
   - Apply the fp32-upcast hunk to `model/attention/dot_production_attention/torch_impl.py`
     (same file as ReKV; the two hunks: `torch.matmul(tmp.to(v.dtype), v)` and
     `torch.matmul(q.float(), k.transpose(-1, -2).float())`). If MuKV's file is identical
     to ReKV's patched one, copy it over and record in code_state.
3. **Env:** `conda activate rekv` (torch 2.6.0+cu124, transformers 4.46.3 — the W2.5-validated
   answering stack). e3dekv cannot answer (vllm 0.7.3 needs transformers>=4.48.2).
4. **Smoke test (0.5B, 1 video):** find the stream-VQA entry in `video_qa/` (MuKV fork of
   rekv_stream_vqa.py), run with the paper config: sample_fps 0.5, mem-token budget ~59K/300frames,
   topk/retrieve per MuKV README. Output CSV must have columns
   video_id,question,answer,pred_answer (same as ReKV) for w25_judge.py.
5. **Full answering, detached:** 0.5B on GPU0 and 7B on GPU1 in parallel
   (`setsid nohup ... > logs/mukv_answers_{05b,7b}.log 2>&1 &`). ETA from W2.5 ReKV timings:
   0.5B ~2h, 7B ~3.3h for 10 videos x 1465 questions. Save CSVs under
   `runs/20260927_0000_e3_phase0/answers_{05b,7b}/`.
6. **Judge (after BOTH finish, needs all 4 GPUs):** Qwen2.5-72B vLLM TP=4, gpu_mem 0.80
   (0.85 OOM'd once in W2.5), `w25_judge.py --pred_csv <csv> --out judged_<m>_72b.json`.
7. **Gate 0:** `python code/scripts_e3/gate0_verdict.py judged_05b_72b.json judged_7b_72b.json`.
   Bands: 0.5B Ego [45.9, 51.9], 7B Ego [47.5, 53.5] (published 57.9 / 59.5, offset −9 ±3).
   0 parse failures required. Update REPORT.md + EXPERIMENT_LOG.md, mirror to Mac.
8. RVS-Ego data (VERIFIED 2026-09-27): 10 videos + ego4d_oe.json at code/ReKV/data/rvs/ego/ (videos in videos/ subdir); use --anno_path data/rvs/ego/ego4d_oe.json from code/MuKV (mirror the same layout or symlink to ../ReKV/data).
