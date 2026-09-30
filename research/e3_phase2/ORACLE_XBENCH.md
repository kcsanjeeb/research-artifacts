# ORACLE_XBENCH — StreamingBench clairvoyant-oracle (STATUS: RUNNING)

Launched 2026-09-28 21:17 CST. Run dir: runs/20260928_2105_e3_phase2_xbench/.

Subset: StreamingBench real-time streaming MC split (questions_real_stream.json),
36 longest available videos (of 150 available offline) = 180 questions.
Arms: DECAF system writetime (tax0.5 debias) + oracle grain policies
seg/frame/patch/default, all llava-ov-7b fps0.5, code @93014b3.
Judge: deterministic letter-match (primary) + optional 72B LLM MC-adapted cross-check.

This file is REGENERATED with full numbers by finalize_xbench_md.py once
xb_watcher.sh completes (see watcher.status / xbench_verdict.txt in the run dir).
