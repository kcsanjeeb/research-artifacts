#!/bin/bash
# Phase 2 judge watcher v2 (recovery 2026-09-28): waits for the three 7B
# answering arms, judges each with Qwen2.5-72B TP=4 (gpu_mem 0.80, env
# /data4/e2judge) once GPUs 0-3 are free, then waits for the oracle chain's
# four policy judged jsons and ONLY THEN writes phase2_verdict.txt (Gate 2 +
# revised Gate 1 + clairvoyant gap closure). Any arm/judge failure ABORTs
# without writing a verdict (phase-1 rank_judge pattern) -- no MISSING verdicts.
# Kill-proof: setsid nohup ... < /dev/null > driver.log 2>&1 &
RUN=/data3/zhuotaotian2_e2/runs/20260928_0315_e3_phase2_compB
M=/data3/zhuotaotian2_e2/models/hf/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31
J=/data3/zhuotaotian2_e2/deliverables/w25_judge_harness/w25_judge.py
PY=/data4/rekv/bin/python
echo "WATCHER_START $(date)" > $RUN/judge.status
ARMS="answers_7b_b_deferred answers_7b_b_writetime answers_7b_b_writetime_rawbias"
POLICIES="seg frame patch default"
abort() { echo "ABORT: $1 $(date)" >> $RUN/judge.status; echo "WATCHER_DONE $(date)" >> $RUN/judge.status; exit 1; }
# 1. wait for all three arms; any nonzero exit aborts (no verdict written)
while true; do
  DONE=1
  for A in $ARMS; do
    RC=$(grep "^EXIT" $RUN/$A.status 2>/dev/null | tail -1 | awk '{print $2}')
    if [ -z "$RC" ]; then DONE=0; elif [ "$RC" != "0" ]; then abort "arm $A exited $RC"; fi
  done
  [ "$DONE" = "1" ] && break
  sleep 300
done
echo "ALL_ARMS_DONE $(date)" >> $RUN/judge.status
# 2. wait for GPUs 0-3 free
while true; do
  N=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk 'NR<=4 && $2>10000 {c++} END {print c+0}')
  [ "$N" = "0" ] && break; sleep 120
done
echo "GPUS_FREE $(date)" >> $RUN/judge.status
# 3. judge each arm sequentially
for A in $ARMS; do
  R=$($PY -c "import pandas as pd; print(len(pd.read_csv('$RUN/$A/1_0.csv')))" 2>/dev/null || echo 0)
  echo "ROWS $A=$R $(date)" >> $RUN/judge.status
  [ "$R" -lt 1460 ] && abort "$A: only $R rows"
  cd $RUN
  export CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1
  /data4/e2judge/bin/python $J --pred_csv $A/1_0.csv --out judged_${A}_72b.json \
    --model $M --tp 4 --gpu_mem 0.80 --max_len 2048 > logs_judge_${A}.log 2>&1
  RC=$?
  [ "$RC" != "0" ] && abort "judge $A exit $RC"
  [ -f $RUN/judged_${A}_72b.json ] || abort "judge $A produced no json"
  echo "JUDGE_${A}_EXIT 0 $(date)" >> $RUN/judge.status
done
echo "ARMS_JUDGED $(date)" >> $RUN/judge.status
# 4. wait for the oracle chain to judge all four policies; only then write verdict
while true; do
  grep -q "^ORACLE_ABORT" $RUN/oracle.status 2>/dev/null && abort "oracle chain aborted"
  MISS=0
  for P in $POLICIES; do [ -f $RUN/judged_oracle_${P}_72b.json ] || MISS=1; done
  [ "$MISS" = "0" ] && break
  sleep 300
done
echo "ALL_JUDGED $(date)" >> $RUN/judge.status
# 5. verdict: reached only when all 3 arm + 4 oracle judged jsons exist
$PY - > $RUN/phase2_verdict.txt <<'PYEOF'
import json, os
RUN = '/data3/zhuotaotian2_e2/runs/20260928_0315_e3_phase2_compB'
W25 = '/data3/zhuotaotian2_e2/runs/20260926_1455_w25_judge_harness'
def load(p):
    return json.load(open(p))
ARMS = ['answers_7b_b_deferred', 'answers_7b_b_writetime', 'answers_7b_b_writetime_rawbias']
POLICIES = ['seg', 'frame', 'patch', 'default']
arms = {a: load(f'{RUN}/judged_{a}_72b.json') for a in ARMS}
oracle = {p: load(f'{RUN}/judged_oracle_{p}_72b.json') for p in POLICIES}
stock = load(f'{W25}/judged_7b_72b.json')
out = []
def fmt(name, j):
    return f"{name}: Acc {j['accuracy']*100:.1f} / Score {j['average_score']:.2f} (n={j['n_questions']})"
out.append(fmt('deferred (B, debias, tax0.5)', arms['answers_7b_b_deferred']))
out.append(fmt('writetime (MuKV-style, debias, tax0.5)', arms['answers_7b_b_writetime']))
out.append(fmt('writetime RAW bias (equal memory)', arms['answers_7b_b_writetime_rawbias']))
out.append(fmt('stock ReKV 7B (W2.5)', stock))
d, w, r = (arms[a] for a in ARMS)
delta = (d['accuracy'] - w['accuracy']) * 100
out.append(f"Gate 2 (part 1): deferred - writetime = {delta:+.1f} pts Acc (need >= +2)")
delta1 = (d['accuracy'] - r['accuracy']) * 100
out.append(f"Revised Gate 1: debias - raw = {delta1:+.1f} pts Acc at equal memory")
out.append(f"deferred vs stock 54.1: {(d['accuracy']-stock['accuracy'])*100:+.1f} pts")
for p in POLICIES:
    out.append(fmt(f'oracle policy {p} (tax1.0)', oracle[p]))
best = max(oracle.values(), key=lambda j: j['accuracy'])
out.append(fmt('oracle clairvoyant (best policy)', best))
gap = (best['accuracy'] - w['accuracy'])
if gap > 1e-9:
    closure = (d['accuracy'] - w['accuracy']) / gap
    out.append(f"Gate 2 (part 2): clairvoyant gap closure = {closure*100:.0f}% "
               f"(deferred-writetime {delta:+.1f} pts of oracle-writetime {gap*100:+.1f} pts; need >= 50%)")
else:
    out.append("Gate 2 (part 2): oracle-writetime gap ~ 0; closure undefined")
print('\n'.join(out))
PYEOF
echo "VERDICT_EXIT $? $(date)" >> $RUN/judge.status
echo "WATCHER_DONE $(date)" >> $RUN/judge.status
