#!/usr/bin/env bash
set -euo pipefail

ROOT=/data/LPP/cvpr/few_shot/sam3-main/best_sam3/soda-bottles-fsod-haga
PY=/opt/conda/envs/sam3/bin/python

"$PY" "$ROOT/soda_sam3_color_prompt.py" \
  --gpus "${GPUS:-0,1,2,3}" \
  --batch-size "${BATCH_SIZE:-4}" \
  --threshold "${SAM3_THRESHOLD:-0.35}" \
  --score-thr "${SCORE_THR:-0.0}" \
  --nms-thr "${NMS_THR:-0.6}" \
  --topk-image "${TOPK_IMAGE:-40}" \
  --scale-x "${SCALE_X:-1.0}" \
  --scale-y "${SCALE_Y:-1.0}" \
  --min-w-frac "${MIN_W_FRAC:-0.0}" \
  --min-h-frac "${MIN_H_FRAC:-0.0}" \
  --min-area-frac "${MIN_AREA_FRAC:-0.0}" \
  --out-dir "${OUT_DIR:-$ROOT/sam3_color_prompt}" \
  --vis \
  "$@"
