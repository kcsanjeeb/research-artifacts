#!/bin/bash
# Phase 2 arm launcher (call after trap checks PASS). Each arm: setsid detached.
RUN=/data3/zhuotaotian2_e2/runs/20260928_0315_e3_phase2_compB
setsid nohup $RUN/run_arm.sh answers_7b_b_deferred 0 \
  DECAF_B_COMMIT=deferred DECAF_B_BIAS=debias DECAF_B_TAX=0.5 DECAF_QUANT=0 \
  < /dev/null > $RUN/logs/launch_deferred.log 2>&1 &
setsid nohup $RUN/run_arm.sh answers_7b_b_writetime 1 \
  DECAF_B_COMMIT=writetime DECAF_B_BIAS=debias DECAF_B_TAX=0.5 DECAF_QUANT=0 DECAF_B_NOPASS2=1 \
  < /dev/null > $RUN/logs/launch_writetime.log 2>&1 &
setsid nohup $RUN/run_arm.sh answers_7b_b_writetime_rawbias 2 \
  DECAF_B_COMMIT=writetime DECAF_B_BIAS=raw DECAF_B_TAX=0.5 DECAF_QUANT=0 DECAF_B_NOPASS2=1 \
  < /dev/null > $RUN/logs/launch_rawbias.log 2>&1 &
echo "arms launched $(date)" > $RUN/arms.status
