#!/bin/bash
cd /data3/zhuotaotian2_e2/code/decaf
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rekv
RUN=/data3/zhuotaotian2_e2/runs/20260928_0315_e3_phase2_compB
export CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 PYTHONPATH=/data3/zhuotaotian2_e2/code/decaf
export DECAF_POSITION_FREE=1 DECAF_FETCH_RANK=1 DECAF_B=1 DECAF_B_COMMIT=deferred DECAF_B_TAX=0.5 DECAF_B_BIAS=debias DECAF_QUANT=1
export DECAF_B_LOG=$RUN/trapb/commit_bqa.jsonl
python verification/decaf_b_trapcheck.py $RUN/trapb bqa > $RUN/logs/trapbqa.log 2>&1
echo "EXIT $? $(date)" >> $RUN/trapb/bqa.status
