#!/bin/bash
# W5.1 arm launcher. Usage: run_arm.sh <arm> <gpu>
#   arm = tax025 | tax0125_patch
set -e
A=$1; GPU=$2
RUN=/data3/zhuotaotian2_e2/runs/20260930_1750_w51_lowtax_fps
cd /data3/zhuotaotian2_e2/code/decaf
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rekv
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 PYTHONPATH=/data3/zhuotaotian2_e2/code/decaf
export DECAF_POSITION_FREE=1 DECAF_FETCH_RANK=1 DECAF_B=1
export DECAF_B_COMMIT=writetime DECAF_B_BIAS=debias DECAF_QUANT=0 DECAF_B_NOPASS2=1
if [ "$A" = "tax025" ]; then
  export DECAF_B_TAX=0.25
  SD=$RUN/answers_7b_writetime_tax025_05fps
elif [ "$A" = "tax0125_patch" ]; then
  export DECAF_B_TAX=0.125 DECAF_B_PATCHSTORE=1
  SD=$RUN/answers_7b_writetime_tax0125patch_05fps
else
  echo "unknown arm $A"; exit 2
fi
export DECAF_B_LOG=$SD/commit_log.jsonl
echo "START $(date)" > $SD.status
python video_qa/rekv_stream_vqa.py --model llava_ov_7b --sample_fps 0.5 \
  --n_local 15000 --retrieve_size 64 --save_dir $SD \
  --anno_path $RUN/ego4d_oe_npy.json --debug false --num_chunks 1 --chunk_idx 0 \
  > $RUN/logs/$A.answer.log 2>&1
echo "EXIT $? $(date)" >> $SD.status
