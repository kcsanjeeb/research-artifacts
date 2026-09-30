#!/bin/bash
# Chained oracle launcher v2 (recovery 2026-09-28): waits for the judge
# watcher's ARMS_JUDGED marker (not WATCHER_DONE, which now comes only after
# the oracle jsons exist -- waiting on it would deadlock), then runs the 4
# grain policies in parallel on GPUs 0-3, judges them (sequentially, 72B
# TP=4), and writes oracle_verdict.txt only when all four policies judged.
# Kill-proof: setsid nohup ... < /dev/null > driver.log 2>&1 &
RUN=/data3/zhuotaotian2_e2/runs/20260928_0315_e3_phase2_compB
M=/data3/zhuotaotian2_e2/models/hf/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31
J=/data3/zhuotaotian2_e2/deliverables/w25_judge_harness/w25_judge.py
PY=/data4/rekv/bin/python
abort() { echo "ORACLE_ABORT: $1 $(date)" >> $RUN/oracle.status; exit 1; }
echo "ORACLE_CHAIN_START $(date)" > $RUN/oracle.status
$RUN/make_subset.sh >> $RUN/oracle.status 2>&1
# 1. wait for the judge watcher to finish judging the answering arms
while true; do
  grep -q "^ARMS_JUDGED" $RUN/judge.status 2>/dev/null && break
  grep -q "^WATCHER_DONE" $RUN/judge.status 2>/dev/null && abort "watcher done without ARMS_JUDGED"
  sleep 300
done
echo "JUDGE_DONE $(date)" >> $RUN/oracle.status
# 2. wait for GPUs 0-3 free (judge watcher is finished at ARMS_JUDGED)
while true; do
  N=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk 'NR<=4 && $2>10000 {c++} END {print c+0}')
  [ "$N" = "0" ] && break; sleep 120
done
echo "GPUS_FREE $(date)" >> $RUN/oracle.status
for P in seg frame patch default; do
  setsid nohup $RUN/run_oracle_arm.sh $P $([ "$P" = seg ] && echo 0 || ([ "$P" = frame ] && echo 1 || ([ "$P" = patch ] && echo 2 || echo 3))) \
    < /dev/null > $RUN/logs/launch_oracle_$P.log 2>&1 &
done
# 3. wait for the four oracle arms; any nonzero exit aborts
while true; do
  DONE=1
  for P in seg frame patch default; do
    RC=$(grep "^EXIT" $RUN/oracle_$P.status 2>/dev/null | tail -1 | awk '{print $2}')
    if [ -z "$RC" ]; then DONE=0; elif [ "$RC" != "0" ]; then abort "oracle arm $P exited $RC"; fi
  done
  [ "$DONE" = "1" ] && break
  sleep 300
done
echo "ORACLE_ARMS_DONE $(date)" >> $RUN/oracle.status
while true; do
  N=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk 'NR<=4 && $2>10000 {c++} END {print c+0}')
  [ "$N" = "0" ] && break; sleep 120
done
for P in seg frame patch default; do
  cd $RUN
  export CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1
  /data4/e2judge/bin/python $J --pred_csv oracle_$P/1_0.csv --out judged_oracle_${P}_72b.json \
    --model $M --tp 4 --gpu_mem 0.80 --max_len 2048 > logs_judge_oracle_${P}.log 2>&1
  RC=$?
  [ "$RC" != "0" ] && abort "judge oracle $P exit $RC"
  [ -f $RUN/judged_oracle_${P}_72b.json ] || abort "judge oracle $P produced no json"
  echo "JUDGE_ORACLE_${P}_EXIT 0 $(date)" >> $RUN/oracle.status
done
# 4. oracle verdict: reached only when all four policies judged
$PY - > $RUN/oracle_verdict.txt <<'PYEOF'
import json, os
RUN = '/data3/zhuotaotian2_e2/runs/20260928_0315_e3_phase2_compB'
judged = {p: json.load(open(f'{RUN}/judged_oracle_{p}_72b.json')) for p in ['seg', 'frame', 'patch', 'default']}
lines = ["oracle policy judged scores: " + json.dumps({p: [round(v['accuracy']*100, 1), round(v['average_score'], 2)] for p, v in judged.items()})]
best = max(judged.values(), key=lambda j: j['accuracy'])
lines.append(f"clairvoyant (best policy): Acc {best['accuracy']*100:.1f} / Score {best['average_score']:.2f}")
lines.append("(per-question clairvoyant max unavailable: judge json has no per-question scores)")
print('\n'.join(lines))
PYEOF
echo "ORACLE_CHAIN_DONE $(date)" >> $RUN/oracle.status
