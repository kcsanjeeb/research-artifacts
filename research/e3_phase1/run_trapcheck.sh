#!/bin/bash
# DECAF Component A trap checks: encode mode (stock), qa mode (DECAF_POSITION_FREE=1), compare.
set -x
RUN=/data3/zhuotaotian2_e2/runs/20260927_1200_e3_phase1_compA
TC=/data3/zhuotaotian2_e2/code/decaf/verification/decaf_trapcheck.py
cd /data3/zhuotaotian2_e2/code/decaf
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rekv
export CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 PYTHONPATH=/data3/zhuotaotian2_e2/code/decaf
/data4/rekv/bin/python $TC $RUN/trapcheck encode > $RUN/logs/trapcheck_encode.log 2>&1
echo "ENCODE_EXIT $?" >> $RUN/trapcheck.status
DECAF_POSITION_FREE=1 PYTHONPATH=/data3/zhuotaotian2_e2/code/decaf /data4/rekv/bin/python $TC $RUN/trapcheck qa > $RUN/logs/trapcheck_qa.log 2>&1
echo "QA_EXIT $?" >> $RUN/trapcheck.status
/data4/rekv/bin/python $TC $RUN/trapcheck compare >> $RUN/logs/trapcheck_encode.log 2>&1
echo "COMPARE_EXIT $?" >> $RUN/trapcheck.status
