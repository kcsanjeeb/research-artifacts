#!/bin/bash
# W5.2 verification suite (detached). GPU2. 120 questions from ONE video
# (0fa75cb3, ~146q) so each arm costs ~20-25 min instead of a full 10-video
# pass. Arms: fp16 / int4-legacy-per-channel / int4-group64 / int4-group128,
# identical writetime tax0.5 debias store config otherwise. Then 72B judge on
# every arm's answers + answer-identity + realized-bytes stats.
set -e
RUN=/data3/zhuotaotian2_e2/runs/20260930_1750_w51_lowtax_fps
W52=/data3/zhuotaotian2_e2/runs/20261001_w52_int4_group
mkdir -p $W52/logs
cd /data3/zhuotaotian2_e2/code/decaf
source ~/miniconda3/etc/profile.d/conda.sh
conda activate rekv
export CUDA_VISIBLE_DEVICES=2 HF_HUB_OFFLINE=1 PYTHONPATH=/data3/zhuotaotian2_e2/code/decaf
export DECAF_POSITION_FREE=1 DECAF_FETCH_RANK=1 DECAF_B=1
export DECAF_B_COMMIT=writetime DECAF_B_TAX=0.5 DECAF_B_BIAS=debias DECAF_B_NOPASS2=1

# subset anno: first 120 questions of video 0fa75cb3 (full temporal stream)
/data4/rekv/bin/python - <<'PY'
import json
anno = json.load(open('/data3/zhuotaotian2_e2/runs/20260930_1750_w51_lowtax_fps/ego4d_oe_npy.json'))
v = [x for x in anno if x['video_id'].startswith('0fa75cb3')][0]
v = dict(v); v['conversations'] = v['conversations'][:120]
json.dump([v], open('/data3/zhuotaotian2_e2/runs/20261001_w52_int4_group/subset_anno_120.json','w'))
print('subset questions:', len(v['conversations']))
PY

run_arm() {  # run_arm <name> <extra exports...>
  N=$1; shift
  SD=$W52/answers_$N
  mkdir -p $SD
  echo "START $(date)" > $SD.status
  ( export "$@"; export DECAF_B_LOG=$SD/commit_log.jsonl
    /data4/rekv/bin/python video_qa/rekv_stream_vqa.py --model llava_ov_7b --sample_fps 0.5 \
      --n_local 15000 --retrieve_size 64 --save_dir $SD \
      --anno_path $W52/subset_anno_120.json --debug false --num_chunks 1 --chunk_idx 0 \
      > $W52/logs/answers_$N.log 2>&1
    echo "EXIT $? $(date)" >> $SD.status )
  RC=$(grep "^EXIT" $SD.status | tail -1 | awk '{print $2}')
  [ "$RC" != "0" ] && { echo "ARM $N FAILED rc=$RC" >> $W52/suite.status; exit 1; }
  grep -q '"assert_passed": true' $SD/fps_assertion.jsonl || { echo "ARM $N FPS FAIL" >> $W52/suite.status; exit 1; }
  echo "ARM $N OK $(date)" >> $W52/suite.status
}

echo "SUITE_START $(date)" > $W52/suite.status
run_arm fp16        DECAF_QUANT=0
run_arm int4legacy  DECAF_QUANT=1 DECAF_QUANT_GROUP=0
run_arm int4g64     DECAF_QUANT=1 DECAF_QUANT_GROUP=64
run_arm int4g128    DECAF_QUANT=1 DECAF_QUANT_GROUP=128

# 72B judge on all four answer sets (TP=4 needs GPUs 0-3 free; the W5.1 arms
# may still be running, so poll)
M=/data3/zhuotaotian2_e2/models/hf/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31
J=/data3/zhuotaotian2_e2/deliverables/w25_judge_harness/w25_judge.py
exec 9>>/data3/zhuotaotian2_e2/runs/judge_gpu.lock
flock -w 21600 9 || { echo "JUDGE LOCK TIMEOUT" >> $W52/suite.status; exit 1; }
while true; do
  N=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk 'NR<=4 && $2>10000 {c++} END {print c+0}')
  [ "$N" = "0" ] && break; sleep 300
done
export CUDA_VISIBLE_DEVICES=0,1,2,3
cd $W52
for N in fp16 int4legacy int4g64 int4g128; do
  /data4/e2judge/bin/python $J --pred_csv answers_$N/1_0.csv --out judged_${N}_72b.json \
    --model $M --tp 4 --gpu_mem 0.80 --max_len 2048 > logs_judge_$N.log 2>&1 \
    || { echo "JUDGE $N FAIL" >> $W52/suite.status; exit 1; }
  echo "JUDGE $N OK $(date)" >> $W52/suite.status
done

# stats: answer identity + realized bytes + judged accuracy delta
python3 /data3/zhuotaotian2_e2/runs/20260930_1750_w51_lowtax_fps/stats_w52.py \
  > $W52/stats_output.txt 2>&1 || { echo "STATS FAIL" >> $W52/suite.status; exit 1; }
echo "SUITE_DONE $(date)" >> $W52/suite.status
