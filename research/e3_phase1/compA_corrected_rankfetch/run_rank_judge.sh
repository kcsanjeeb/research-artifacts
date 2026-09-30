#!/bin/bash
# Corrected Gate 1 judge watcher. Waits for the rank-fetch 7B answering run,
# then judges with Qwen2.5-72B TP=4 (gpu_mem 0.80, env /data4/e2judge) once
# GPUs 0-3 are free, and writes rank_verdict.txt against the W2.5 stock
# baseline (7B 54.1 Acc / 2.82 Score; PASS = within +/-1.5 pts Acc).
# Kill-proof: launch with setsid nohup ... < /dev/null > driver.log 2>&1 &
RUN=/data3/zhuotaotian2_e2/runs/20260927_2200_e3_phase1_compA_rankfetch
W25=/data3/zhuotaotian2_e2/runs/20260926_1455_w25_judge_harness
M=/data3/zhuotaotian2_e2/models/hf/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31
J=/data3/zhuotaotian2_e2/deliverables/w25_judge_harness/w25_judge.py
PY=/data4/rekv/bin/python
echo "WATCHER_START $(date)" > $RUN/rank_judge.status
# 1. wait for answering to finish cleanly
while true; do
  RC=$(grep "^EXIT" $RUN/answers_7b_rank.status 2>/dev/null | tail -1 | awk '{print $2}')
  if [ -n "$RC" ]; then break; fi
  sleep 300
done
echo "ANSWER_RC $RC $(date)" >> $RUN/rank_judge.status
if [ "$RC" != "0" ]; then echo "ABORT: answering exited $RC" >> $RUN/rank_judge.status; echo "WATCHER_DONE $(date)" >> $RUN/rank_judge.status; exit 1; fi
R7=$($PY -c "import pandas as pd; print(len(pd.read_csv('$RUN/answers_7b_rank/1_0.csv')))" 2>/dev/null || echo 0)
echo "ROWS rank_7b=$R7 $(date)" >> $RUN/rank_judge.status
if [ "$R7" -lt 1460 ]; then echo "ABORT: only $R7 rows" >> $RUN/rank_judge.status; echo "WATCHER_DONE $(date)" >> $RUN/rank_judge.status; exit 1; fi
# 2. wait for GPUs 0-3 to be free (<10GB used on each)
while true; do
  N=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk 'NR<=4 && $2>10000 {c++} END {print c+0}')
  [ "$N" = "0" ] && break; sleep 120
done
echo "GPUS_FREE $(date)" >> $RUN/rank_judge.status
# 3. judge
cd $RUN
export CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1
/data4/e2judge/bin/python $J --pred_csv answers_7b_rank/1_0.csv --out judged_7b_rank_72b.json \
  --model $M --tp 4 --gpu_mem 0.80 --max_len 2048 > logs_judge_7b_rank.log 2>&1
echo "JUDGE_7B_EXIT $? $(date)" >> $RUN/rank_judge.status
# 4. verdict vs W2.5 stock baseline
$PY - > $RUN/rank_verdict.txt <<'EOF'
import json
rank = json.load(open('/data3/zhuotaotian2_e2/runs/20260927_2200_e3_phase1_compA_rankfetch/judged_7b_rank_72b.json'))
stock = json.load(open('/data3/zhuotaotian2_e2/runs/20260926_1455_w25_judge_harness/judged_7b_72b.json'))
d_acc = (rank['accuracy'] - stock['accuracy']) * 100
d_score = rank['average_score'] - stock['average_score']
verdict = 'PASS' if abs(d_acc) <= 1.5 else 'FAIL'
print(f"corrected-A (rank fetch) 7B: Acc {rank['accuracy']*100:.1f} / Score {rank['average_score']:.2f} (n={rank['n_questions']})")
print(f"W2.5 stock 7B baseline:    Acc {stock['accuracy']*100:.1f} / Score {stock['average_score']:.2f}")
print(f"delta: {d_acc:+.1f} pts Acc / {d_score:+.2f} Score")
print(f"Gate 1 (corrected) criterion: |delta Acc| <= 1.5 -> {verdict}")
EOF
echo "VERDICT_EXIT $? $(date)" >> $RUN/rank_judge.status
echo "WATCHER_DONE $(date)" >> $RUN/rank_judge.status
