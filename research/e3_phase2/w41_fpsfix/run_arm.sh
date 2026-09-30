#!/bin/bash
# W4.1: DECAF writetime (tax0.5, debias) on full RVS-Ego at TRUE 0.5 fps (fps-fix run).
# GPU0, detached. Same store config as compB answers_7b_b_writetime, corrected loader.
RUN=/data3/zhuotaotian2_e2/runs/20260930_0935_w41_fpsfix
cd /data3/zhuotaotian2_e2/code/decaf
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rekv
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1 PYTHONPATH=/data3/zhuotaotian2_e2/code/decaf
export DECAF_POSITION_FREE=1 DECAF_FETCH_RANK=1 DECAF_B=1
export DECAF_B_COMMIT=writetime DECAF_B_BIAS=debias DECAF_B_TAX=0.5 DECAF_QUANT=0 DECAF_B_NOPASS2=1
export DECAF_B_LOG=$RUN/answers_7b_writetime_05fps/commit_log.jsonl
echo "START $(date)" > $RUN/answers_7b_writetime_05fps.status
python video_qa/rekv_stream_vqa.py --model llava_ov_7b --sample_fps 0.5 \
  --n_local 15000 --retrieve_size 64 --save_dir $RUN/answers_7b_writetime_05fps \
  --anno_path $RUN/ego4d_oe_npy.json --debug false --num_chunks 1 --chunk_idx 0 \
  > $RUN/logs/answers_7b_writetime_05fps.answer.log 2>&1
echo "EXIT $? $(date)" >> $RUN/answers_7b_writetime_05fps.status
