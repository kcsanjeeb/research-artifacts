#!/bin/bash
# DECAF Component A answering / recall-only runner.
# Usage: run_answer_decaf.sh <model_key> <gpu> <save_dir> <status_file> <arm>
#   arm = decaf | entangled_recall
RUN=/data3/zhuotaotian2_e2/runs/20260927_1200_e3_phase1_compA
MODEL=$1; GPU=$2; SAVE=$3; STATUS=$4; ARM=$5
cd /data3/zhuotaotian2_e2/code/decaf
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rekv
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 PYTHONPATH=/data3/zhuotaotian2_e2/code/decaf
TAG=$(echo $MODEL | sed 's/llava_ov_//')
if [ "$ARM" = 'decaf' ]; then
  export DECAF_POSITION_FREE=1
  export DECAF_RETRIEVAL_LOG=$RUN/retrieval_decaf_$TAG.jsonl
else
  export DECAF_RECALL_ONLY=1
  export DECAF_RETRIEVAL_LOG=$RUN/retrieval_entangled_$TAG.jsonl
fi
SAVE=$RUN/$SAVE
python video_qa/rekv_stream_vqa.py --model $MODEL --sample_fps 0.5 --n_local 15000   --retrieve_size 64 --save_dir $SAVE --anno_path data/rvs/ego/ego4d_oe.json   --debug false --num_chunks 1 --chunk_idx 0 > $SAVE/answer.log 2>&1
echo "EXIT $? $(date)" >> $STATUS
