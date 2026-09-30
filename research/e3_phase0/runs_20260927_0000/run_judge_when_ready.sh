#!/bin/bash
RUN=/data3/zhuotaotian2_e2/runs/20260927_0000_e3_phase0
M=/data3/zhuotaotian2_e2/models/hf/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31
J=/data3/zhuotaotian2_e2/deliverables/w25_judge_harness/w25_judge.py
echo "WATCHER_START $(date)" > $RUN/judge.status
rows() { /data4/rekv/bin/python -c "import pandas as pd; print(len(pd.read_csv('$1')))" 2>/dev/null || echo 0; }
while true; do
  E5=$(grep -c "^EXIT" $RUN/answers_05b.status 2>/dev/null || echo 0)
  E7=$(grep -c "^EXIT" $RUN/answers_7b.status 2>/dev/null || echo 0)
  if [ "$E5" -ge 1 ] && [ "$E7" -ge 1 ]; then break; fi
  sleep 120
done
echo "ANSWERING_DONE $(date)" >> $RUN/judge.status
R5=$(rows $RUN/answers_05b/results.csv); R7=$(rows $RUN/answers_7b/results.csv)
echo "ROWS 05b=$R5 7b=$R7" >> $RUN/judge.status
if [ "$R5" -lt 1465 ] || [ "$R7" -lt 1465 ]; then
  echo "JUDGE_SKIPPED incomplete CSVs (expect 1465 each)" >> $RUN/judge.status
  exit 1
fi
while true; do
  N=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk "NR<=4 && \$2>10000 {c++} END {print c+0}")
  [ "$N" = "0" ] && break
  sleep 60
done
echo "GPUS_FREE $(date)" >> $RUN/judge.status
cd $RUN
export CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1
/data4/e2judge/bin/python $J --pred_csv answers_05b/results.csv --out judged_05b_72b.json --model $M --tp 4 --gpu_mem 0.80 --max_len 2048 > logs_judge_05b.log 2>&1
echo "JUDGE_05B_EXIT $? $(date)" >> $RUN/judge.status
/data4/e2judge/bin/python $J --pred_csv answers_7b/results.csv --out judged_7b_72b.json --model $M --tp 4 --gpu_mem 0.80 --max_len 2048 > logs_judge_7b.log 2>&1
echo "JUDGE_7B_EXIT $? $(date)" >> $RUN/judge.status
/data4/rekv/bin/python /data3/zhuotaotian2_e2/code/scripts_e3/gate0_verdict.py judged_05b_72b.json judged_7b_72b.json > gate0_verdict.txt 2>&1
echo "VERDICT_EXIT $? $(date)" >> $RUN/judge.status
echo "WATCHER_DONE $(date)" >> $RUN/judge.status
