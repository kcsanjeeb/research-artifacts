#!/bin/bash
# Corrected Component A answering: DECAF_FETCH_RANK=1 (rank-remapped fetch
# positions), 7B on GPU1, W2.5 config (fps 0.5, n_local 15000, retrieve 64,
# greedy, seed 2024). PYTHONPATH MUST point at code/decaf or the stock
# editable-installed ReKV code silently runs.
RUN=/data3/zhuotaotian2_e2/runs/20260927_2200_e3_phase1_compA_rankfetch
cd /data3/zhuotaotian2_e2/code/decaf
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rekv
export CUDA_VISIBLE_DEVICES=1 HF_HUB_OFFLINE=1 PYTHONPATH=/data3/zhuotaotian2_e2/code/decaf
export DECAF_POSITION_FREE=1 DECAF_FETCH_RANK=1
export DECAF_RETRIEVAL_LOG=$RUN/retrieval_rank_7b.jsonl
python video_qa/rekv_stream_vqa.py --model llava_ov_7b --sample_fps 0.5 \
  --n_local 15000 --retrieve_size 64 --save_dir $RUN/answers_7b_rank \
  --anno_path data/rvs/ego/ego4d_oe.json --debug false --num_chunks 1 --chunk_idx 0 \
  > $RUN/answers_7b_rank.answer.log 2>&1
echo "EXIT $? $(date)" >> $RUN/answers_7b_rank.status
