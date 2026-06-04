#!/usr/bin/env bash
set -euo pipefail

: "${GEMINI_API_KEY:?set GEMINI_API_KEY first}"

python /data/LPP/few_shot/final/orionproducts-vtl2z-fsod-puhv/code/gemini_orion_direct_api.py   --image-ids all   --coord-mode pixel   --min-score 0.03   --nms 0.35   --max-per-image 80   --out /data/LPP/few_shot/final/orionproducts-vtl2z-fsod-puhv/code/gemini_orion_direct_all_v1.json   --raw-dir /data/LPP/few_shot/final/orionproducts-vtl2z-fsod-puhv/code/gemini_orion_direct_all_v1_raw   --resume   --evaluate
