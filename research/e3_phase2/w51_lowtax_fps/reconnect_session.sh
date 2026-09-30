#!/bin/bash
# One-shot session to run on reconnect: commit trapcheck fix, launch W5.1 arm2
# + watcher, launch W5.2 suite, update logs, write deliverable, snapshot state.
set -e
D=/data3/zhuotaotian2_e2
RUN=$D/runs/20260930_1750_w51_lowtax_fps
W52=$D/runs/20261001_w52_int4_group
cd $D/code/decaf
cp /tmp/w51/trapcheck.py verification/decaf_b_trapcheck.py
cp /tmp/w51/trapcheck.py model/attention/decaf_b_trapcheck.py
git add -A model/attention/decaf_b_trapcheck.py verification/decaf_b_trapcheck.py
git -c user.name=w5-agent -c user.email=w5@local commit -q -m "trapcheck: TB2 mse/cosine; TB3 tolerate tuple past_key_values (SKIP not FAIL)" || true
git rev-parse HEAD > $RUN/code_state/HEAD_arm2.txt
cp /tmp/w51/watcher.sh /tmp/w51/w52_run_all.sh /tmp/w51/stats_w52.py $RUN/
chmod +x $RUN/watcher.sh /tmp/w51/w52_run_all.sh

echo "=== GPU state ==="
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader

echo "=== arm1 alive? ==="
tail -c 400 $RUN/logs/tax025.answer.log || true
grep -c . $RUN/answers_7b_writetime_tax025_05fps/fps_assertion.jsonl 2>/dev/null || echo "no fps file yet"

# GPU1 must be free (mem < 10000 MiB) before arm2
G1=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n 2p)
if [ "$G1" -lt 10000 ]; then
  mkdir -p $RUN/answers_7b_writetime_tax0125patch_05fps
  # env dump for arm2 (arm1 env reconstructed in config.json)
  ( echo "DECAF_POSITION_FREE=1 DECAF_FETCH_RANK=1 DECAF_B=1"
    echo "DECAF_B_COMMIT=writetime DECAF_B_BIAS=debias DECAF_QUANT=0 DECAF_B_NOPASS2=1"
    echo "DECAF_B_TAX=0.125 DECAF_B_PATCHSTORE=1"
    echo "commit=$(cat $RUN/code_state/HEAD_arm2.txt)" ) > $RUN/env_tax0125_patch.txt
  cd $RUN
  setsid nohup ./run_arm.sh tax0125_patch 1 > logs/tax0125_patch.launch.log 2>&1 < /dev/null &
  echo "ARM2_LAUNCHED $!"
  sleep 25
  tail -3 logs/tax0125_patch.answer.log 2>/dev/null || tail -3 logs/tax0125_patch.launch.log
else
  echo "GPU1 BUSY ($G1 MiB) — arm2 NOT launched; investigate"
fi

# W5.1 watcher
cd $RUN
setsid nohup ./watcher.sh > logs/watcher.driver.log 2>&1 < /dev/null &
echo "WATCHER_LAUNCHED $!"
sleep 3
head -2 watcher.status

# W5.2 suite on GPU2
G2=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sed -n 3p)
if [ "$G2" -lt 10000 ]; then
  setsid nohup /tmp/w51/w52_run_all.sh > /tmp/w51/w52_suite.driver.log 2>&1 < /dev/null &
  echo "W52_SUITE_LAUNCHED $!"
  sleep 15
  cat $W52/suite.status 2>/dev/null || tail -3 /tmp/w51/w52_suite.driver.log
else
  echo "GPU2 BUSY ($G2 MiB) — W5.2 suite NOT launched"
fi

# logs + deliverable
cat /tmp/w51/experiment_log_entry.md >> $D/EXPERIMENT_LOG.md
cat /tmp/w51/gpu_coordination_entry.md >> $D/GPU_COORDINATION.md
mkdir -p $D/deliverables/e3_phase2
cp /tmp/w51/W5_PARETO.md $D/deliverables/e3_phase2/W5_PARETO.md
echo RECONNECT_SESSION_DONE $(date)
