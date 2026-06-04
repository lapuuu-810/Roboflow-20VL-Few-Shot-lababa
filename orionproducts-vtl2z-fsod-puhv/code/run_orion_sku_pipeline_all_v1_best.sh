#!/usr/bin/env bash
set -euo pipefail

: "${DASHSCOPE_API_KEY:?set DASHSCOPE_API_KEY first}"

cd /data/LPP/cvpr/few_shot/sam3-main/best_sam3/orionproducts-vtl2z-fsod-puhv

conda run -n qwen python qwen_orion_sku_pipeline.py \
  --image-ids all \
  --topk-per-image 40 \
  --min-area 80 \
  --max-area 200000 \
  --sam-min-score 0.0 \
  --candidate-nms 0.45 \
  --crop-max-side 768 \
  --stage3-score-mode qconf \
  --final-score-thr 0.05 \
  --final-nms 0.25 \
  --max-per-image 120 \
  --out qwen_orion_sku_pipeline_all_v1_best.json \
  --raw-dir qwen_orion_sku_pipeline_all_v1_best_raw \
  --resume \
  --evaluate
