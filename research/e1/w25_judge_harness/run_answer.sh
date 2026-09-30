#!/bin/bash
# W2.5 answering phase: ReKV stream VQA on RVS-Ego, validated V100 config ported.
# Usage: run_answer.sh <model_key> <gpu_id> <save_dir>
set -e
MODEL=$1; GPU=$2; SAVE=$3
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rekv
cd /data3/zhuotaotian2_e2/code/ReKV
export CUDA_VISIBLE_DEVICES=$GPU
export HF_HUB_OFFLINE=0
python video_qa/rekv_stream_vqa.py \
    --model $MODEL \
    --sample_fps 0.5 \
    --n_local 15000 \
    --retrieve_size 64 \
    --save_dir $SAVE \
    --anno_path data/rvs/ego/ego4d_oe.json \
    --debug false \
    --num_chunks 1 --chunk_idx 0
