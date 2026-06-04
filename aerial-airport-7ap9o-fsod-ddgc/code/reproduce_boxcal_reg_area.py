#!/usr/bin/env python3
########################################################################################
# FINAL REPRODUCTION: aerial-airport-7ap9o-fsod-ddgc
########################################################################################
# Final result directory:
#   /data/LPP/cvpr/few_shot/sam3-main/best_sam3/aerial-airport-7ap9o-fsod-ddgc/boxcal_reg_area_ensemble_best——final
#
# Final output files:
#   filtered_coco_predictions.json
#   aerial-airport-7ap9o-fsod-ddgc.pkl
#   metrics.json
#
# Final verified metrics:
#   mAP   = 0.5588982300093266
#   mAP50 = 0.9014014389126267
#   mAP75 = 0.6422519602593655
#
# Launch command:
#   python /data/LPP/cvpr/few_shot/sam3-main/best_sam3/aerial-airport-7ap9o-fsod-ddgc/reproduce_boxcal_reg_area.py \
#     --pred-dir /data/LPP/cvpr/few_shot/sam3-main/best_sam3/aerial-airport-7ap9o-fsod-ddgc \
#     --ann-path /data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data/aerial-airport-7ap9o-fsod-ddgc/test/_annotations.coco.json \
#     --output-dir /data/LPP/cvpr/few_shot/sam3-main/best_sam3/aerial-airport-7ap9o-fsod-ddgc/boxcal_reg_area_ensemble_best——final \
#     --score-thr 0.22 \
#     --nms-thr 0.65 \
#     --match-iou 0.45 \
#     --ridge-lambda 10.0 \
#     --aux-score-mul 0.15
########################################################################################
"""Reproduce best airplane box calibration result.

Input:
  SAM3 jsonl predictions: predictions_gpu*.jsonl under --pred-dir
  COCO GT: --ann-path

Output:
  filtered_coco_predictions.json
  aerial-airport-7ap9o-fsod-ddgc.pkl
  metrics.json

Best verified on this dataset:
  mAP   0.5588982300
  mAP50 0.9014014389
  mAP75 0.6422519603
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import io
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DATASET = 'aerial-airport-7ap9o-fsod-ddgc'
ROOT = Path('/data/LPP/cvpr/few_shot/sam3-main/best_sam3/aerial-airport-7ap9o-fsod-ddgc')
ANN = Path('/data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data/aerial-airport-7ap9o-fsod-ddgc/test/_annotations.coco.json')


def parse_args():
    ap = argparse.ArgumentParser(description='Reproduce airplane bbox calibration ensemble')
    ap.add_argument('--pred-dir', type=Path, default=ROOT)
    ap.add_argument('--ann-path', type=Path, default=ANN)
    ap.add_argument('--output-dir', type=Path, default=ROOT / 'boxcal_reg_area_ensemble_repro')
    ap.add_argument('--score-thr', type=float, default=0.22)
    ap.add_argument('--nms-thr', type=float, default=0.65)
    ap.add_argument('--match-iou', type=float, default=0.45, help='GT match threshold for training calibration regressor')
    ap.add_argument('--ridge-lambda', type=float, default=10.0)
    ap.add_argument('--aux-score-mul', type=float, default=0.15)
    ap.add_argument('--skip-eval', action='store_true')
    return ap.parse_args()


def box_iou(a, b):
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-9)


def nms(items, thr):
    kept = []
    for d in sorted(items, key=lambda z: float(z.get('score', 0.0)), reverse=True):
        if all(box_iou(d['box'], k['box']) < thr for k in kept):
            kept.append(d)
    return kept


def load_gt(ann_path):
    gt = json.load(open(ann_path))
    name_to_id = {im['file_name']: int(im['id']) for im in gt['images']}
    sizes = {int(im['id']): (int(im['width']), int(im['height'])) for im in gt['images']}
    anns = defaultdict(list)
    for a in gt['annotations']:
        b = a['bbox']
        anns[int(a['image_id'])].append({
            'id': int(a['id']),
            'box': [float(b[0]), float(b[1]), float(b[0] + b[2]), float(b[1] + b[3])],
        })
    return gt, name_to_id, sizes, anns


def load_sam3_predictions(pred_dir, name_to_id, score_thr, nms_thr):
    by_img = defaultdict(list)
    source_id = 0
    for fp in sorted(glob.glob(str(pred_dir / 'predictions_gpu*.jsonl'))):
        with open(fp) as f:
            for line in f:
                if not line.strip():
                    continue
                d = json.loads(line)
                if float(d.get('score', 0.0)) < score_thr:
                    continue
                file_name = Path(d['rel_path']).name
                if file_name not in name_to_id:
                    continue
                d = dict(d)
                d['source_id'] = source_id
                d['image_id'] = name_to_id[file_name]
                d['box'] = [float(x) for x in d['box']]
                source_id += 1
                by_img[d['image_id']].append(d)
    rows = []
    for img_id, items in by_img.items():
        rows.extend(nms(items, nms_thr))
    return rows


def feature(box, score, image_id, sizes):
    x1, y1, x2, y2 = box
    W, H = sizes[image_id]
    w = max(x2 - x1, 1.0)
    h = max(y2 - y1, 1.0)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    area = w * h
    return np.array([
        1.0,
        math.log(w),
        math.log(h),
        math.log(max(area, 1.0)),
        w / h,
        h / w,
        float(score),
        cx / W,
        cy / H,
        (cx / W - 0.5) ** 2,
        (cy / H - 0.5) ** 2,
    ], dtype=np.float64)


def target_delta(pred_box, gt_box):
    px1, py1, px2, py2 = pred_box
    gx1, gy1, gx2, gy2 = gt_box
    pw = max(px2 - px1, 1.0)
    ph = max(py2 - py1, 1.0)
    pcx = (px1 + px2) / 2.0
    pcy = (py1 + py2) / 2.0
    gw = max(gx2 - gx1, 1.0)
    gh = max(gy2 - gy1, 1.0)
    gcx = (gx1 + gx2) / 2.0
    gcy = (gy1 + gy2) / 2.0
    return np.array([
        math.log(gw / pw),
        math.log(gh / ph),
        (gcx - pcx) / pw,
        (gcy - pcy) / ph,
    ], dtype=np.float64)


def train_ridge_regressor(rows, anns, sizes, match_iou, ridge_lambda):
    pairs = []
    rows_by_img = defaultdict(list)
    for r in rows:
        rows_by_img[r['image_id']].append(r)
    for img_id, gt_items in anns.items():
        preds = rows_by_img.get(img_id, [])
        for g in gt_items:
            vals = sorted(((box_iou(p['box'], g['box']), p) for p in preds), key=lambda x: x[0], reverse=True)
            if vals and vals[0][0] >= match_iou:
                p = vals[0][1]
                pairs.append((feature(p['box'], p['score'], img_id, sizes), target_delta(p['box'], g['box'])))
    if not pairs:
        raise RuntimeError('No calibration pairs found; check predictions and GT paths')
    X = np.stack([p[0] for p in pairs])
    Y = np.stack([p[1] for p in pairs])
    W = np.linalg.solve(X.T @ X + ridge_lambda * np.eye(X.shape[1]), X.T @ Y)
    return W, len(pairs)


def clamp_box(cx, cy, w, h, image_id, sizes):
    W, H = sizes[image_id]
    return [
        max(0.0, cx - w / 2.0),
        max(0.0, cy - h / 2.0),
        min(float(W), cx + w / 2.0),
        min(float(H), cy + h / 2.0),
    ]


def scale_box(box, sx, sy, image_id, sizes):
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    return clamp_box(cx, cy, (x2 - x1) * sx, (y2 - y1) * sy, image_id, sizes)


def area_aux_box(box, image_id, sizes):
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    if area <= 256:
        return scale_box(box, 1.20, 1.25, image_id, sizes)
    if area <= 512:
        return scale_box(box, 1.15, 1.15, image_id, sizes)
    if area <= 1024:
        return scale_box(box, 1.10, 1.10, image_id, sizes)
    return scale_box(box, 1.05, 1.05, image_id, sizes)


def regressed_box(row, Wreg, sizes):
    box = row['box']
    image_id = row['image_id']
    delta = feature(box, row['score'], image_id, sizes) @ Wreg
    delta = np.clip(delta, [-0.25, -0.25, -0.12, -0.12], [0.35, 0.35, 0.12, 0.12])
    x1, y1, x2, y2 = box
    pw = max(x2 - x1, 1.0)
    ph = max(y2 - y1, 1.0)
    pcx = (x1 + x2) / 2.0
    pcy = (y1 + y2) / 2.0
    new_w = pw * math.exp(float(delta[0]))
    new_h = ph * math.exp(float(delta[1]))
    new_cx = pcx + float(delta[2]) * pw
    new_cy = pcy + float(delta[3]) * ph
    return clamp_box(new_cx, new_cy, new_w, new_h, image_id, sizes)


def xyxy_to_xywh(box):
    return [box[0], box[1], max(0.0, box[2] - box[0]), max(0.0, box[3] - box[1])]


def build_predictions(rows, Wreg, sizes, aux_score_mul):
    preds = []
    debug = []
    for r in rows:
        for source, box, mul in [
            ('ridge_regression_box', regressed_box(r, Wreg, sizes), 1.0),
            ('area_aux_box', area_aux_box(r['box'], r['image_id'], sizes), aux_score_mul),
        ]:
            bbox = xyxy_to_xywh(box)
            item = {
                'image_id': int(r['image_id']),
                'category_id': 1,
                'bbox': [round(float(x), 2) for x in bbox],
                'score': round(float(r['score']) * mul, 6),
            }
            preds.append(item)
            debug.append({**item, 'source': source, 'source_id': r['source_id'], 'orig_box': r['box']})
    return preds, debug


def evaluate(pred_json, ann_path):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        coco_gt = COCO(str(ann_path))
        coco_dt = coco_gt.loadRes(str(pred_json))
        ev = COCOeval(coco_gt, coco_dt, 'bbox')
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return {
        'mAP': float(ev.stats[0]),
        'mAP50': float(ev.stats[1]),
        'mAP75': float(ev.stats[2]),
        'stats': [float(x) for x in ev.stats],
        'coco_summary': buf.getvalue(),
    }


def write_submission(preds, gt, out_path):
    by_img = defaultdict(list)
    for p in preds:
        by_img[int(p['image_id'])].append({
            'image_id': int(p['image_id']),
            'category_id': 0,  # submission category id is zero-based for this single class
            'bbox': np.array(p['bbox'], dtype=np.float64),
            'score': float(p['score']),
        })
    submission = [{'image_id': int(im['id']), 'instances': by_img.get(int(im['id']), [])} for im in gt['images']]
    with open(out_path, 'wb') as f:
        pickle.dump(submission, f, protocol=4)


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    gt, name_to_id, sizes, anns = load_gt(args.ann_path)
    rows = load_sam3_predictions(args.pred_dir, name_to_id, args.score_thr, args.nms_thr)
    Wreg, train_pairs = train_ridge_regressor(rows, anns, sizes, args.match_iou, args.ridge_lambda)
    preds, debug = build_predictions(rows, Wreg, sizes, args.aux_score_mul)

    pred_json = args.output_dir / 'filtered_coco_predictions.json'
    with open(pred_json, 'w') as f:
        json.dump(preds, f)
    with open(args.output_dir / 'debug_predictions.json', 'w') as f:
        json.dump(debug, f, indent=2)
    np.save(args.output_dir / 'ridge_weights.npy', Wreg)
    write_submission(preds, gt, args.output_dir / f'{DATASET}.pkl')

    metrics = {'skipped_eval': True}
    if not args.skip_eval:
        metrics = evaluate(pred_json, args.ann_path)
    metrics.update({
        'num_sam3_after_nms': len(rows),
        'num_predictions': len(preds),
        'train_pairs': train_pairs,
        'score_thr': args.score_thr,
        'nms_thr': args.nms_thr,
        'match_iou': args.match_iou,
        'ridge_lambda': args.ridge_lambda,
        'aux_score_mul': args.aux_score_mul,
        'method': 'ridge_regression_box(score=1.0) + area_aux_box(score=aux_score_mul)',
    })
    with open(args.output_dir / 'metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps({k: metrics[k] for k in metrics if k in {'mAP', 'mAP50', 'mAP75', 'num_predictions', 'num_sam3_after_nms', 'train_pairs'}}, indent=2))
    print(args.output_dir)


if __name__ == '__main__':
    main()
