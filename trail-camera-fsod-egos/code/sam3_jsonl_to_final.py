#!/usr/bin/env python3
"""Convert saved SAM3 initial jsonl predictions into final RF20-VL format.

Steps: read predictions_gpu*.jsonl, map filenames to COCO image_id, convert SAM3
xyxy boxes to COCO xywh, apply the score threshold/NMS/max-per-image parameters
recorded in ../result/best_eval.json, then export JSON and pickle. When the local
Roboflow export contains a dataset placeholder at category_id=0, submission ids are
local ids minus one.
"""
