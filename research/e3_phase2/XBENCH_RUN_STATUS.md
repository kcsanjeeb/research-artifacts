# XBench (StreamingBench) clairvoyant-oracle — run status 2026-09-28 21:25 CST

RUN DIR (Ada): /data3/zhuotaotian2_e2/runs/20260928_2105_e3_phase2_xbench/
DELIVERABLE (Ada): /data3/zhuotaotian2_e2/deliverables/e3_phase2/ORACLE_XBENCH.md
(code @93014b3, llava-ov-7b, fps 0.5, n_local 15000, retrieve 64)

## Subset
StreamingBench real-time streaming MC split (questions_real_stream.json).
150/498 videos available offline (videos_rt/ from zips 1-50,101-150,201-250).
Selected 36 longest = 180 questions (SB gives only 5-6 q/video; deviation
from RVS 2-video shape documented). Anno: xbench_anno.json, manifest:
subset_manifest.json.

## Arms (launched 21:17 CST, healthy at 21:24, ~354 s/video -> ~3.5 h/arm)
- GPU2 chain: answers_7b_writetime (system, writetime tax0.5 debias nopass2)
  -> oracle_patch -> oracle_default   (ETA ~08:00 CST)
- GPU3 chain: oracle_seg -> oracle_frame  (ETA ~04:30 CST)
- xb_watcher.sh: waits all 5 EXIT 0 -> letter-match judge (xb_judge.py,
  deterministic, PRIMARY, gives per-question clairvoyant) -> optional 72B
  LLM MC-adapted cross-check when GPUs 0-3 free (6h window) -> xb_verdict.py
- meta_finalize.sh: on WATCHER_DONE runs finalize_xbench_md.py which
  regenerates ORACLE_XBENCH.md with the numbers.

## Follow-up checklist
1. ssh Ada; cat $RUN/watcher.status  (expect WATCHER_DONE)
2. cat $RUN/xbench_verdict.txt  (5 accuracies, clairvoyant, margins, verdict)
3. cat $RUN/xbench_verdict.txt -> margins; re-mirror ORACLE_XBENCH.md to here.
4. If LLM_XCHECK_SKIPPED: fine, letter-match is primary. If judged_*_llm.json
   exist, cross-check numbers are in the regenerated deliverable.

## Incidents
- 21:11 launch bug: run_oracle_arm.sh self-backgrounded -> seg+frame collided
  on GPU3; killed frame; chain scripts rewritten (foreground + skip-if-done).
- 21:15 OOM: duplicate arms from old+new chains shared GPU2/3 -> full clean
  restart at 21:17. Status files/logs/dirs were wiped before relaunch.

## Blockers / not done
- OVO-Bench: NOT present on Ada (/data2/zhuotaotian2_e2_data has only mlvu,
  streamingbench, videomme). Would require a download pass.
- 72B LLM cross-check may be skipped if the parallel compB analysis holds
  GPUs 0-3 (6h wait window in the watcher).
