#!/usr/bin/env bash
set -euo pipefail

# Full-data launch for the current best strategy:
# 1) CLAHE/SAM3 finger-bone filtered fixed 17 boxes with the 0-29 best params.
# 2) 17-slot anatomical distribution-prior calibration over the selected image split.
# 3) Copy final outputs to stable current-best filenames.

PYTHON_BIN="${PYTHON_BIN:-/opt/conda/bin/python}"
IDS="${IDS:-all}"
TRAIN_IDS="${TRAIN_IDS:-$IDS}"
EVAL_IDS="${EVAL_IDS:-$IDS}"

SAM_OUT="sam3_finger_bone_filter17_full_current"
SLOT_OUT="xray_slot_prior_v1_sam3filter17_train${TRAIN_IDS//[^A-Za-z0-9]/_}_eval${EVAL_IDS//[^A-Za-z0-9]/_}"

"$PYTHON_BIN" xray_make_sam3_filter17_fixed_params.py \
  --image-ids "$IDS" \
  --out-dir "$SAM_OUT"

"$PYTHON_BIN" xray_slot_prior_calibrate_v1.py \
  --base "$SAM_OUT/sam3_finger_bone_filter17_best.json" \
  --train-ids "$TRAIN_IDS" \
  --eval-ids "$EVAL_IDS" \
  --out-dir "$SLOT_OUT" \
  --out "$SLOT_OUT/best.json" \
  --passes 2

cp "$SLOT_OUT/best.json" xray_map_best_current_full.json
cp "$SLOT_OUT/best_eval.json" xray_map_best_current_full_eval.json
cp "$SLOT_OUT/montage_percat.jpg" montage_best_current_full.jpg

printf '\nDONE\n'
printf 'PRED: %s\n' "xray_map_best_current_full.json"
printf 'EVAL: %s\n' "xray_map_best_current_full_eval.json"
printf 'VIS : %s\n' "montage_best_current_full.jpg"
