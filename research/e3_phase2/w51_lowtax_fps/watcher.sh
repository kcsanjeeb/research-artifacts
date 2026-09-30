#!/bin/bash
# W5.1 orchestrator watcher (kill-proof: setsid nohup). Waits for the two W5.1
# arms (tax025 GPU0, tax0125_patch GPU1), verifies the standing fps assertion
# in both, judges both with the 72B TP=4 when GPUs 0-3 are free, then stats +
# RESULTS. Abort-without-verdict: any failure -> ABORT line + exit, no numbers
# are produced from a broken run.
RUN=/data3/zhuotaotian2_e2/runs/20260930_1750_w51_lowtax_fps
M=/data3/zhuotaotian2_e2/models/hf/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31
J=/data3/zhuotaotian2_e2/deliverables/w25_judge_harness/w25_judge.py
PY=/data4/rekv/bin/python
ST=$RUN/watcher.status
echo "WATCHER_START $(date)" > $ST
abort() { echo "ABORT: $1 $(date)" >> $ST; echo "WATCHER_DONE $(date)" >> $ST; exit 1; }

# 0. both arms must exit 0
while true; do
  DONE=1
  for A in $RUN/answers_7b_writetime_tax025_05fps $RUN/answers_7b_writetime_tax0125patch_05fps; do
    RC=$(grep "^EXIT" $A.status 2>/dev/null | tail -1 | awk '{print $2}')
    if [ -z "$RC" ]; then DONE=0; elif [ "$RC" != "0" ]; then abort "arm $A exited $RC"; fi
  done
  [ "$DONE" = "1" ] && break
  sleep 300
done
echo "ALL_ARMS_DONE $(date)" >> $ST

# 1. fps assertion evidence (standing anchor check)
for F in $RUN/answers_7b_writetime_tax025_05fps/fps_assertion.jsonl \
         $RUN/answers_7b_writetime_tax0125patch_05fps/fps_assertion.jsonl; do
  [ -f "$F" ] || abort "missing fps_assertion.jsonl: $F"
  grep -q '"assert_passed": true' "$F" || abort "fps assertion FAILED in $F"
  NP=$(grep -c . "$F")
  [ "$NP" -ge 10 ] || abort "fps assertion only $NP lines in $F (expect 10)"
  echo "FPS_ASSERT_OK $F lines=$NP $(date)" >> $ST
done

# 2. row counts
for N in tax025 tax0125_patch; do
  R=$($PY -c "import pandas as pd; print(len(pd.read_csv('$RUN/answers_7b_writetime_' + ('tax025_05fps' if '$N'=='tax025' else 'tax0125patch_05fps') + '/1_0.csv')))" 2>/dev/null || echo 0)
  echo "ROWS $N=$R $(date)" >> $ST
  [ "$R" -lt 1460 ] && abort "$N: only $R rows"
done

# 3. GPUs free, then judge both (serialized with any other judge user via flock)
exec 9>>/data3/zhuotaotian2_e2/runs/judge_gpu.lock
flock -w 21600 9 || abort "judge lock timeout"
while true; do
  N=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk 'NR<=4 && $2>10000 {c++} END {print c+0}')
  [ "$N" = "0" ] && break; sleep 120
done
echo "GPUS_FREE $(date)" >> $ST
cd $RUN
export CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1
/data4/e2judge/bin/python $J --pred_csv answers_7b_writetime_tax025_05fps/1_0.csv \
  --out judged_tax025_72b.json --model $M --tp 4 --gpu_mem 0.80 --max_len 2048 \
  > logs_judge_tax025.log 2>&1
[ $? != "0" ] && abort "judge tax025 exit $?"
[ -f $RUN/judged_tax025_72b.json ] || abort "judge tax025 no json"
echo "JUDGE_tax025_EXIT 0 $(date)" >> $ST
/data4/e2judge/bin/python $J --pred_csv answers_7b_writetime_tax0125patch_05fps/1_0.csv \
  --out judged_tax0125_patch_72b.json --model $M --tp 4 --gpu_mem 0.80 --max_len 2048 \
  > logs_judge_tax0125_patch.log 2>&1
[ $? != "0" ] && abort "judge tax0125_patch exit $?"
[ -f $RUN/judged_tax0125_patch_72b.json ] || abort "judge tax0125_patch no json"
echo "JUDGE_tax0125_patch_EXIT 0 $(date)" >> $ST

# 4. stats + results
python3 $RUN/stats_w51.py > $RUN/stats_output.txt 2>&1 \
  || abort "stats failed (see $RUN/stats_output.txt)"
echo "STATS_DONE $(date)" >> $ST
echo "WATCHER_DONE $(date)" >> $ST
