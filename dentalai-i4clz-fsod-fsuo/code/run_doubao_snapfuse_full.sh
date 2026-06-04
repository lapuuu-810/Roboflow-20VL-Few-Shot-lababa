#!/usr/bin/env bash
set -euo pipefail
cd /data/LPP/cvpr/few_shot/sam3-main/best_sam3/dentalai-i4clz-fsod-fsuo

: "${ARK_API_KEY:?Set ARK_API_KEY before running}"

python doubao_dental_sam3_candidate_api.py \
  --image-ids all \
  --topk-per-image 35 \
  --min-area 80 \
  --max-area 5000 \
  --min-score 0.0 \
  --candidate-nms 0.65 \
  --context 1.15 \
  --max-side 720 \
  --coord-mode norm1000 \
  --accept-shift 0.85 \
  --accept-iou 0.03 \
  --box-source hybrid \
  --sam3-score-weight 0.25 \
  --score-scale 1.0 \
  --final-nms 0.45 \
  --out doubao_dental_sam3crop_all_top35_rawapi_v1.json \
  --raw-dir doubao_dental_sam3crop_all_top35_rawapi_v1_raw \
  --resume \
  --evaluate

python snap_fuse_qwen_sam3.py \
  --api doubao_dental_sam3crop_all_top35_rawapi_v1.json \
  --image-ids all \
  --out doubao_dental_snapfuse_all_v1.json
