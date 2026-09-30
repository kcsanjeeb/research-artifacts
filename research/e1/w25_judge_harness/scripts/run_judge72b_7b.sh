#!/bin/bash
cd /data3/zhuotaotian2_e2/runs/20260926_1455_w25_judge_harness
export CUDA_VISIBLE_DEVICES=0,1,2,3
M=/data3/zhuotaotian2_e2/models/hf/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31
/data4/e2judge/bin/python w25_judge.py --pred_csv answers_7b/1_0.csv --out judged_7b_72b.json --model $M --tp 4 --gpu_mem 0.80 --max_len 2048
echo "GRADE_7B_EXIT=$?"
