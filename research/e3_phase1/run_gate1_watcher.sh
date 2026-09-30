#!/bin/bash
# Gate 1 watcher: waits for answering+recall runs, judges DECAF answers with the
# W2.5 72B harness, computes retrieval recall, writes gate1_verdict.txt.
RUN=/data3/zhuotaotian2_e2/runs/20260927_1200_e3_phase1_compA
W25=/data3/zhuotaotian2_e2/runs/20260926_1455_w25_judge_harness
M=/data3/zhuotaotian2_e2/models/hf/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31
J=/data3/zhuotaotian2_e2/deliverables/w25_judge_harness/w25_judge.py
PY=/data4/rekv/bin/python
echo "WATCHER_START $(date)" > $RUN/gate1_watcher.status
wait_exit() { while true; do E=$(grep -c '^EXIT' $1 2>/dev/null || echo 0); [ "$E" -ge 1 ] && break; sleep 120; done; }
wait_exit $RUN/answers_7b_decaf.status
wait_exit $RUN/answers_05b_decaf.status
wait_exit $RUN/recall_entangled_7b.status
wait_exit $RUN/recall_entangled_05b.status
echo "ALL_RUNS_DONE $(date)" >> $RUN/gate1_watcher.status
rows() { $PY -c "import pandas as pd; print(len(pd.read_csv('$1')))" 2>/dev/null || echo 0; }
R7=$(rows $RUN/answers_7b_decaf/1_0.csv); R5=$(rows $RUN/answers_05b_decaf/1_0.csv)
echo "ROWS decaf_7b=$R7 decaf_05b=$R5" >> $RUN/gate1_watcher.status
while true; do
  N=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk 'NR<=4 && $2>10000 {c++} END {print c+0}')
  [ "$N" = '0' ] && break; sleep 60
done
echo "GPUS_FREE $(date)" >> $RUN/gate1_watcher.status
cd $RUN
export CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1
if [ "$R7" -ge 1400 ]; then
  /data4/e2judge/bin/python $J --pred_csv answers_7b_decaf/1_0.csv --out judged_7b_decaf_72b.json --model $M --tp 4 --gpu_mem 0.80 --max_len 2048 > logs_judge_7b_decaf.log 2>&1
  echo "JUDGE_7B_EXIT $? $(date)" >> $RUN/gate1_watcher.status
else echo "JUDGE_7B_SKIPPED rows=$R7" >> $RUN/gate1_watcher.status; fi
if [ "$R5" -ge 1400 ]; then
  /data4/e2judge/bin/python $J --pred_csv answers_05b_decaf/1_0.csv --out judged_05b_decaf_72b.json --model $M --tp 4 --gpu_mem 0.80 --max_len 2048 > logs_judge_05b_decaf.log 2>&1
  echo "JUDGE_05B_EXIT $? $(date)" >> $RUN/gate1_watcher.status
else echo "JUDGE_05B_SKIPPED rows=$R5" >> $RUN/gate1_watcher.status; fi
# retrieval recall (CPU)
for TAG in 7b 05b; do
  $PY /data3/zhuotaotian2_e2/code/decaf/verification/compute_recall.py /data3/zhuotaotian2_e2/code/decaf/data/rvs/ego/ego4d_oe.json $RUN/retrieval_decaf_$TAG.jsonl $RUN/recall_decaf_$TAG.json > /dev/null 2>&1
  $PY /data3/zhuotaotian2_e2/code/decaf/verification/compute_recall.py /data3/zhuotaotian2_e2/code/decaf/data/rvs/ego/ego4d_oe.json $RUN/retrieval_entangled_$TAG.jsonl $RUN/recall_entangled_$TAG.json > /dev/null 2>&1
  echo "RECALL_$TAG_DONE $(date)" >> $RUN/gate1_watcher.status
done
$PY /data3/zhuotaotian2_e2/code/decaf/verification/gate1_verdict.py   $RUN/judged_7b_decaf_72b.json $RUN/judged_05b_decaf_72b.json   $W25/judged_7b_72b.json $W25/judged_0.5b_72b.json   $RUN/recall_decaf_7b.json $RUN/recall_decaf_05b.json   $RUN/recall_entangled_7b.json $RUN/recall_entangled_05b.json   $RUN/gate1_verdict > $RUN/gate1_verdict.out 2>&1
echo "VERDICT_EXIT $? $(date)" >> $RUN/gate1_watcher.status
echo "WATCHER_DONE $(date)" >> $RUN/gate1_watcher.status
