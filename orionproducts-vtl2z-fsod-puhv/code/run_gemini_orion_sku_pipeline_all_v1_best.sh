#!/usr/bin/env bash
set -euo pipefail

: "${GEMINI_API_KEY:?set GEMINI_API_KEY first}"

python /data/LPP/few_shot/final/orionproducts-vtl2z-fsod-puhv/code/gemini_orion_sku_pipeline.py   --image-ids all   --topk-per-image 40   --min-area 80   --max-area 200000   --sam-min-score 0.0   --candidate-nms 0.45   --use-gemini-direct   --direct-coord-mode auto   --direct-topk-per-image 25   --crop-max-side 768   --stage3-score-mode qconf   --final-score-thr 0.05   --final-nms 0.25   --max-per-image 120   --out /data/LPP/few_shot/final/orionproducts-vtl2z-fsod-puhv/code/gemini_orion_sku_pipeline_all_v1_best.json   --raw-dir /data/LPP/few_shot/final/orionproducts-vtl2z-fsod-puhv/code/gemini_orion_sku_pipeline_all_v1_best_raw   --resume   --evaluate
