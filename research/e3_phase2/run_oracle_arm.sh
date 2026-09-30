#!/bin/bash
# Oracle grain-policy arm: DECAF_B_POLICY in {seg,frame,patch,default} at TAX=1.0
# (full segment retained so every grain is available), single-pass, 2-video subset.
# Usage: run_oracle_arm.sh <policy> <gpu>
P=$1; GPU=$2
RUN=/data3/zhuotaotian2_e2/runs/20260928_0315_e3_phase2_compB
cd /data3/zhuotaotian2_e2/code/decaf
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rekv
export CUDA_VISIBLE_DEVICES=$GPU HF_HUB_OFFLINE=1 PYTHONPATH=/data3/zhuotaotian2_e2/code/decaf
export DECAF_POSITION_FREE=1 DECAF_FETCH_RANK=1 DECAF_B=1 DECAF_B_COMMIT=deferred \
       DECAF_B_BIAS=debias DECAF_B_TAX=1.0 DECAF_QUANT=0 DECAF_B_NOPASS2=1 \
       DECAF_B_POLICY=$P DECAF_B_LOG=$RUN/oracle_$P/commit_log.jsonl
python video_qa/rekv_stream_vqa.py --model llava_ov_7b --sample_fps 0.5 \
  --n_local 15000 --retrieve_size 64 --save_dir $RUN/oracle_$P \
  --anno_path $RUN/oracle_subset_anno.json --debug false --num_chunks 1 --chunk_idx 0 \
  > $RUN/logs/oracle_$P.answer.log 2>&1
echo "EXIT $? $(date)" >> $RUN/oracle_$P.status
