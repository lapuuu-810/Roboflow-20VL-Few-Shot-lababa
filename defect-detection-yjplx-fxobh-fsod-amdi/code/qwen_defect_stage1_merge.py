#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import Counter

import qwen_defect_detection_api_pipeline as base


def iou_box(a, b):
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    inter = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0.0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def parse_ids(value):
    if not value or value.strip().lower() == 'all':
        return None
    return {int(x) for x in value.split(',') if x.strip()}


def load_preds(path, image_ids):
    preds = base.load_json(path)
    if image_ids is None:
        return preds
    return [p for p in preds if int(p['image_id']) in image_ids]


def merge(primary, recall, iou_thr, recall_score_scale, class_aware=True):
    out = [dict(p) for p in primary]
    for pred in recall:
        duplicate = False
        for old in out:
            if pred['image_id'] != old['image_id']:
                continue
            if class_aware and int(pred['category_id']) != int(old['category_id']):
                continue
            if iou_box(pred['bbox'], old['bbox']) >= iou_thr:
                duplicate = True
                break
        if duplicate:
            continue
        item = dict(pred)
        item['score'] = round(max(0.001, min(0.999, float(item.get('score', 0.7)) * recall_score_scale)), 6)
        item['source'] = item.get('source', 'recall_supplement')
        out.append(item)
    return out


def subset_gt(gt_path, image_ids, out_path):
    gt = base.load_json(gt_path)
    if image_ids is None:
        return gt_path
    sub = dict(gt)
    sub['images'] = [im for im in gt['images'] if im['id'] in image_ids]
    sub['annotations'] = [ann for ann in gt['annotations'] if ann['image_id'] in image_ids]
    base.save_json(sub, out_path)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--primary', required=True, help='high-precision structured first-stage predictions')
    ap.add_argument('--recall', required=True, help='looser recall-oriented first-stage predictions')
    ap.add_argument('--out', required=True)
    ap.add_argument('--image-ids', default='all')
    ap.add_argument('--iou-thr', type=float, default=0.35)
    ap.add_argument('--recall-score-scale', type=float, default=0.55)
    ap.add_argument('--class-agnostic', action='store_true')
    ap.add_argument('--gt', default=str(base.GT))
    ap.add_argument('--evaluate', action='store_true')
    args = ap.parse_args()

    image_ids = parse_ids(args.image_ids)
    primary = load_preds(args.primary, image_ids)
    recall = load_preds(args.recall, image_ids)
    merged = merge(primary, recall, args.iou_thr, args.recall_score_scale, not args.class_agnostic)
    base.save_json(merged, args.out)
    print(f'wrote {len(merged)} merged predictions to {args.out}')
    print('counts', dict(Counter(int(p['category_id']) for p in merged)))

    if args.evaluate:
        gt_path = args.gt
        if image_ids is not None:
            gt_path = str(Path(args.out).with_suffix('')) + '_gt_subset.json'
            subset_gt(args.gt, image_ids, gt_path)
        metrics = base.evaluate(gt_path, args.out)
        eval_path = str(Path(args.out).with_suffix('')) + '_eval.json'
        base.save_json(metrics, eval_path)
        print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
