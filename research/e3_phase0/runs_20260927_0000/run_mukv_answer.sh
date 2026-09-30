#!/bin/bash
# MuKV Gate-0 answering. Usage: run_mukv_answer.sh <05b|7b> <gpu>
set -e
M=$1; GPU=$2
RUN=/data3/zhuotaotian2_e2/runs/20260927_0000_e3_phase0
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || true
cd /data3/zhuotaotian2_e2/code/MuKV
export CUDA_VISIBLE_DEVICES=$GPU
export PYTHONPATH=/data3/zhuotaotian2_e2/code/MuKV:$PYTHONPATH
export HF_HUB_OFFLINE=1 HF_ENDPOINT=https://hf-mirror.com TOKENIZERS_PARALLELISM=false
MODEL=/data3/zhuotaotian2_e2/code/ReKV/model_zoo/llava-onevision-qwen2-0.5b-ov-hf
[ "$M" = "7b" ] && MODEL=/data3/zhuotaotian2_e2/code/ReKV/model_zoo/llava-onevision-qwen2-7b-ov-hf
echo "START $(date)" > $RUN/answers_$M.status
/data4/rekv/bin/python scripts/run_mukv_rvs_ego.py   --model_path $MODEL   --anno_path $RUN/npy_anno.json --video_format npy   --sample_fps 0.5 --n_local 15000 --retrieve_size 64 --retrieve_chunk_size 1   --save_dir $RUN/answers_$M   --num_chunks 1 --chunk_idx 0   --enable_compression true --enable_rerank true
rc=$?
echo "EXIT $rc $(date)" >> $RUN/answers_$M.status
