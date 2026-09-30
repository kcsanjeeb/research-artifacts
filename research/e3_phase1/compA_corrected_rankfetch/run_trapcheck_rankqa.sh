#!/bin/bash
# Corrected Component A trap check: TR1 rank plumbing + TR2 late-stream stock
# equivalence (full store, the exact-mode failure mode), 0.5B on GPU3.
set -x
RUN=/data3/zhuotaotian2_e2/runs/20260927_2200_e3_phase1_compA_rankfetch
TC=/data3/zhuotaotian2_e2/code/decaf/verification/decaf_trapcheck.py
cd /data3/zhuotaotian2_e2/code/decaf
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rekv
export CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 PYTHONPATH=/data3/zhuotaotian2_e2/code/decaf
export DECAF_POSITION_FREE=1 DECAF_FETCH_RANK=1
/data4/rekv/bin/python $TC $RUN/trapcheck rankqa > $RUN/logs/trapcheck_rankqa.log 2>&1
echo "RANKQA_EXIT $? $(date)" >> $RUN/trapcheck.status
