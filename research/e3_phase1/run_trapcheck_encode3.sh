#!/bin/bash
# Re-run of the T2 rotary cross-check (encode mode) + compare, sharing GPU3 with the 0.5B recall run.
RUN=/data3/zhuotaotian2_e2/runs/20260927_1200_e3_phase1_compA
TC=/data3/zhuotaotian2_e2/code/decaf/verification/decaf_trapcheck.py
cd /data3/zhuotaotian2_e2/code/decaf
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rekv
export CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 PYTHONPATH=/data3/zhuotaotian2_e2/code/decaf
/data4/rekv/bin/python $TC $RUN/trapcheck encode > $RUN/logs/trapcheck_encode3.log 2>&1
echo "ENCODE3_EXIT $?" >> $RUN/trapcheck.status
/data4/rekv/bin/python $TC $RUN/trapcheck compare >> $RUN/logs/trapcheck_encode3.log 2>&1
echo "COMPARE2_EXIT $?" >> $RUN/trapcheck.status
