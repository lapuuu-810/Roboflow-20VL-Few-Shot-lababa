# Code

This folder contains the SAM3 initial pipeline scripts for `trail-camera-fsod-egos` plus a short conversion note.

Raw candidates:
`/data/LPP/cvpr/few_shot/sam3-main/best_sam3/trail-camera-fsod-egos/predictions_gpu*.jsonl`

Final conversion: SAM3 `xyxy` boxes -> COCO `xywh`, class-wise NMS, per-image top-k cap, and `server_id = local_id - 1` for the dummy Roboflow category export. Parameters are stored in `../result/best_eval.json`.
