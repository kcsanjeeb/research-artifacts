#!/bin/bash
# Launch one XD self-extract worker per GPU
cd /home/san/FleetVAD/research/e0
source /home/san/miniconda/etc/profile.d/conda.sh
conda activate fleetvad-vlm
for g in 0 1 2 3; do
  CUDA_VISIBLE_DEVICES=$g nohup python -u followup/src/xd_self_extract.py --shard $g --n-shards 4 > logs/xd_xtract_$g.log 2>&1 &
done
echo "launched 4 extract workers"
