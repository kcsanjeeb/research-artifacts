#!/bin/bash
# Gate 2 reassessment sweep chain (2026-09-28): deferred-machanism config sweep.
# Wave 1 (GPUs 0-3): writetime/deferred at tax 0.25/0.75 (equal-tax Gate-2 pairs).
# Wave 2 (GPUs 0-1): deferred tax0.5 single-pass (pass-2 off) + deferred tax0.5
#                    entropy threshold 1.2 (pass-2 fires more often).
# Each wave judged sequentially with the 72B TP=4 harness after arms finish.
# Kill-proof: setsid nohup ... < /dev/null > driver.log 2>&1 &
RUN=/data3/zhuotaotian2_e2/runs/20260928_0315_e3_phase2_compB
M=/data3/zhuotaotian2_e2/models/hf/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31
J=/data3/zhuotaotian2_e2/deliverables/w25_judge_harness/w25_judge.py
PY=/data4/rekv/bin/python
PJ=/data4/e2judge/bin/python

abort() { echo "SWEEP_ABORT: $1 $(date)" >> $RUN/sweep.status; exit 1; }
gpus_free() {
  N=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk 'NR<=4 && $2>10000 {c++} END {print c+0}')
  [ "$N" = "0" ]
}
launch_arm() { # name gpu env...
  local ARM=$1 GPU=$2; shift 2
  setsid nohup bash -c "
    cd /data3/zhuotaotian2_e2/code/decaf
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate rekv
    export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 PYTHONPATH=/data3/zhuotaotian2_e2/code/decaf
    export DECAF_POSITION_FREE=1 DECAF_FETCH_RANK=1 DECAF_B=1
    export $*
    export DECAF_B_LOG=$RUN/$ARM/commit_log.jsonl
    python video_qa/rekv_stream_vqa.py --model llava_ov_7b --sample_fps 0.5 \
      --n_local 15000 --retrieve_size 64 --save_dir $RUN/$ARM \
      --anno_path $RUN/ego4d_oe_npy.json --debug false --num_chunks 1 --chunk_idx 0 \
      > $RUN/logs/${ARM}.answer.log 2>&1
    echo \"EXIT $? $(date)\" >> $RUN/${ARM}.status
  " < /dev/null > $RUN/logs/launch_${ARM}.log 2>&1 &
}
wait_arms() { # arm...
  while true; do
    DONE=1
    for A in "$@"; do
      RC=$(grep "^EXIT" $RUN/$A.status 2>/dev/null | tail -1 | awk '{print $2}')
      if [ -z "$RC" ]; then DONE=0; elif [ "$RC" != "0" ]; then abort "arm $A exited $RC"; fi
    done
    [ "$DONE" = "1" ] && break
    sleep 300
  done
  while ! gpus_free; do sleep 120; done
}
judge_arm() { # arm
  local A=$1
  cd $RUN
  export CUDA_VISIBLE_DEVICES=0,1,2,3 HF_HUB_OFFLINE=1
  $PJ $J --pred_csv $A/1_0.csv --out judged_${A}_72b.json \
    --model $M --tp 4 --gpu_mem 0.80 --max_len 2048 > logs_judge_${A}.log 2>&1
  RC=$?
  [ "$RC" != "0" ] && abort "judge $A exit $RC"
  [ -f $RUN/judged_${A}_72b.json ] || abort "judge $A produced no json"
  echo "JUDGE_${A}_EXIT 0 $(date)" >> $RUN/sweep.status
}

echo "SWEEP_CHAIN_START $(date)" > $RUN/sweep.status
# wait for the oracle chain to fully finish (arms + judging + verdict)
while true; do
  grep -q "^ORACLE_CHAIN_DONE" $RUN/oracle.status 2>/dev/null && break
  grep -q "^ORACLE_ABORT" $RUN/oracle.status 2>/dev/null && abort "oracle chain aborted"
  sleep 300
done
echo "ORACLE_DONE $(date)" >> $RUN/sweep.status
while ! gpus_free; do sleep 120; done

# ---- wave 1: equal-tax Gate-2 pairs at tax 0.25 / 0.75
W1="answers_7b_b_writetime_tax025 answers_7b_b_deferred_tax025 answers_7b_b_writetime_tax075 answers_7b_b_deferred_tax075"
launch_arm answers_7b_b_writetime_tax025 0 DECAF_B_COMMIT=writetime DECAF_B_BIAS=debias DECAF_B_TAX=0.25 DECAF_QUANT=0 DECAF_B_NOPASS2=1
launch_arm answers_7b_b_deferred_tax025 1 DECAF_B_COMMIT=deferred DECAF_B_BIAS=debias DECAF_B_TAX=0.25 DECAF_QUANT=0
launch_arm answers_7b_b_writetime_tax075 2 DECAF_B_COMMIT=writetime DECAF_B_BIAS=debias DECAF_B_TAX=0.75 DECAF_QUANT=0 DECAF_B_NOPASS2=1
launch_arm answers_7b_b_deferred_tax075 3 DECAF_B_COMMIT=deferred DECAF_B_BIAS=debias DECAF_B_TAX=0.75 DECAF_QUANT=0
echo "WAVE1_LAUNCHED $(date)" >> $RUN/sweep.status
wait_arms $W1
echo "WAVE1_ARMS_DONE $(date)" >> $RUN/sweep.status
for A in $W1; do judge_arm $A; done

# ---- wave 2: single-pass (pass-2 off) + entropy threshold 1.2, both deferred tax0.5
W2="answers_7b_b_deferred_nopass2 answers_7b_b_deferred_ent12"
while ! gpus_free; do sleep 120; done
launch_arm answers_7b_b_deferred_nopass2 0 DECAF_B_COMMIT=deferred DECAF_B_BIAS=debias DECAF_B_TAX=0.5 DECAF_QUANT=0 DECAF_B_NOPASS2=1
launch_arm answers_7b_b_deferred_ent12 1 DECAF_B_COMMIT=deferred DECAF_B_BIAS=debias DECAF_B_TAX=0.5 DECAF_QUANT=0 DECAF_B_ENTROPY=1.2
echo "WAVE2_LAUNCHED $(date)" >> $RUN/sweep.status
wait_arms $W2
echo "WAVE2_ARMS_DONE $(date)" >> $RUN/sweep.status
for A in $W2; do judge_arm $A; done

# ---- verdict
$PY - > $RUN/sweep_verdict.txt <<'PYEOF'
import json, os
RUN = '/data3/zhuotaotian2_e2/runs/20260928_0315_e3_phase2_compB'
arms = ['answers_7b_b_deferred', 'answers_7b_b_writetime',
        'answers_7b_b_writetime_tax025', 'answers_7b_b_deferred_tax025',
        'answers_7b_b_writetime_tax075', 'answers_7b_b_deferred_tax075',
        'answers_7b_b_deferred_nopass2', 'answers_7b_b_deferred_ent12']
rows = []
for a in arms:
    p = f'{RUN}/judged_{a}_72b.json'
    if not os.path.exists(p):
        continue
    j = json.load(open(p))
    gbh = None
    cp = f'{RUN}/{a}/commit_log.jsonl'
    if os.path.exists(cp):
        import subprocess
    rows.append((a, round(j['accuracy'] * 100, 1), round(j['average_score'], 3), j['n_yes'], j['n_unparsed_pred']))
print('arm judged Acc/Score/n_yes/n_unparsed:')
for r in rows:
    print('  %-38s %5.1f  %.3f  %4d  %d' % r)
PYEOF
echo "SWEEP_CHAIN_DONE $(date)" >> $RUN/sweep.status
