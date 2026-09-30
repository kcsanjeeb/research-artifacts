#!/bin/bash
# DECAF Phase 2 answering arm launcher. Usage: run_arm.sh <arm_name> <gpu> <extra env assignments...>
ARM=$1; GPU=$2; shift 2
RUN=/data3/zhuotaotian2_e2/runs/20260928_0315_e3_phase2_compB
cd /data3/zhuotaotian2_e2/code/decaf
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rekv
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 PYTHONPATH=/data3/zhuotaotian2_e2/code/decaf
export DECAF_POSITION_FREE=1 DECAF_FETCH_RANK=1 DECAF_B=1
export "$@"
export DECAF_B_LOG=$RUN/${ARM}/commit_log.jsonl
python video_qa/rekv_stream_vqa.py --model llava_ov_7b --sample_fps 0.5 \
  --n_local 15000 --retrieve_size 64 --save_dir $RUN/$ARM \
  --anno_path $RUN/ego4d_oe_npy.json --debug false --num_chunks 1 --chunk_idx 0 \
  > $RUN/logs/${ARM}.answer.log 2>&1
echo "EXIT $? $(date)" >> $RUN/${ARM}.status
