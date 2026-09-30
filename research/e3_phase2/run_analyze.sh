#!/bin/bash
# Encode-only analyze arm (7B, GPU3): full-stream bias curves + cumulative store
# bytes for the GB/h accounting. Retrieval runs per question (cheap), no generation.
RUN=/data3/zhuotaotian2_e2/runs/20260928_0315_e3_phase2_compB
cd /data3/zhuotaotian2_e2/code/decaf
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rekv
export CUDA_VISIBLE_DEVICES=3 HF_HUB_OFFLINE=1 PYTHONPATH=/data3/zhuotaotian2_e2/code/decaf
export DECAF_POSITION_FREE=1 DECAF_FETCH_RANK=1 DECAF_B=1 DECAF_B_COMMIT=deferred \
       DECAF_B_TAX=0.5 DECAF_B_BIAS=debias DECAF_QUANT=0 DECAF_RECALL_ONLY=1 \
       DECAF_B_ANALYZE=$RUN/analyze_bias.jsonl
python video_qa/rekv_stream_vqa.py --model llava_ov_7b --sample_fps 0.5 \
  --n_local 15000 --retrieve_size 64 --save_dir $RUN/analyze_bias \
  --anno_path $RUN/ego4d_oe_npy.json --debug false --num_chunks 1 --chunk_idx 0 \
  > $RUN/logs/analyze_bias.log 2>&1
echo "EXIT $? $(date)" >> $RUN/analyze_bias.status
