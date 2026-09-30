#!/bin/bash
# Launch 8 XD self-extract workers (2 per GPU), thread-capped
cd /home/san/FleetVAD/research/e0
PY=/home/san/miniconda/envs/fleetvad-vlm/bin/python
for g in $(seq 0 7); do
  gpu=$((g % 4))
  OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 CUDA_VISIBLE_DEVICES=$gpu nohup $PY -u followup/src/xd_self_extract.py --shard $g --n-shards 8 > logs/xd_xtract8_$g.log 2>&1 &
done
echo "launched 8 thread-capped workers"
