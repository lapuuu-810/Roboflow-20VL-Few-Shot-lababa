#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DATASET = 'wb-prova-stqnm-fsod-rbvg'
ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path('/data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data') / DATASET
GT_PATH = DATA_ROOT / 'test' / '_annotations.coco.json'

METRIC_KEYS = [
    'mAP', 'mAP50', 'mAP75', 'mAP_small', 'mAP_medium', 'mAP_large',
    'AR1', 'AR10', 'AR100', 'AR_small', 'AR_medium', 'AR_large',
]


def load_jsonl_predictions():
    gt = json.load(open(GT_PATH))
    file_to_img = {im['file_name']: im for im in gt['images']}
    rows = []
    for fp in sorted(ROOT.glob('predictions_gpu*.jsonl')):
        with open(fp) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                im = file_to_img[os.path.basename(r['image_path'])]
                x1, y1, x2, y2 = [float(v) for v in r['box']]
                x1 = max(0.0, min(float(im['width']) - 1.0, x1))
                y1 = max(0.0, min(float(im['height']) - 1.0, y1))
                x2 = max(0.0, min(float(im['width']), x2))
                y2 = max(0.0, min(float(im['height']), y2))
                if x2 <= x1 or y2 <= y1:
                    continue
                rows.append({
                    'image_id': int(im['id']),
                    'category_id': int(r['category_id']),
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'score': float(r.get('score', 1.0)),
                    'prompt': r.get('prompt', ''),
                })
    return gt, rows


def build_predictions(rows, image_ids, topk):
    by_img = defaultdict(list)
    for r in rows:
        by_img[int(r['image_id'])].append(r)
    preds = []
    for iid in image_ids:
        ps = sorted(by_img.get(iid, []), key=lambda p: p['score'], reverse=True)[:topk]
        for p in ps:
            preds.append({
                'image_id': iid,
                'category_id': int(p['category_id']),
                'bbox': [round(float(v), 2) for v in p['bbox']],
                'score': round(float(p['score']), 6),
            })
    return preds


def evaluate(preds, image_ids):
    if not preds:
        return {k: 0.0 for k in METRIC_KEYS}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        coco = COCO(str(GT_PATH))
        dt = coco.loadRes(preds)
        ev = COCOeval(coco, dt, 'bbox')
        ev.params.imgIds = image_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {k: float(v) for k, v in zip(METRIC_KEYS, ev.stats)}


def write_submission(preds, gt, out_path):
    by_img = defaultdict(list)
    for p in preds:
        by_img[int(p['image_id'])].append({
            'image_id': int(p['image_id']),
            'category_id': int(p['category_id']) - 1,
            'bbox': np.array([float(v) for v in p['bbox']], dtype=np.float32),
            'score': float(p['score']),
        })
    submission = [
        {'image_id': int(im['id']), 'instances': by_img.get(int(im['id']), [])}
        for im in gt['images']
    ]
    with open(out_path, 'wb') as f:
        pickle.dump(submission, f, protocol=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--topk', type=int, default=8)
    ap.add_argument('--out-dir', type=Path, default=ROOT / 'wb_sam3_topk_postprocess_v1')
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gt, rows = load_jsonl_predictions()
    image_ids = [int(im['id']) for im in gt['images']]
    preds = build_predictions(rows, image_ids, args.topk)
    metrics = evaluate(preds, image_ids)
    metrics.update({
        'topk_per_image': args.topk,
        'raw_predictions': len(rows),
        'kept_predictions': len(preds),
        'images': len(image_ids),
        'method': 'sam3_score_topk_no_nms',
    })

    pred_path = args.out_dir / 'wb_sam3_topk_postprocess_v1.json'
    eval_path = args.out_dir / 'wb_sam3_topk_postprocess_v1_eval.json'
    pkl_path = args.out_dir / f'{DATASET}.pkl'
    json.dump(preds, open(pred_path, 'w'), indent=2)
    json.dump(metrics, open(eval_path, 'w'), indent=2)
    write_submission(preds, gt, pkl_path)

    print('FINAL', pred_path)
    print('EVAL', eval_path)
    print('PKL', pkl_path)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
