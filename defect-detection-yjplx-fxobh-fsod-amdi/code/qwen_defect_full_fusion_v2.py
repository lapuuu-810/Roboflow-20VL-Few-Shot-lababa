#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
GT = Path('/data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data/defect-detection-yjplx-fxobh-fsod-amdi/test/_annotations.coco.json')


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)


def iou(a, b):
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    inter = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0.0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def grouped_nms(preds, thr):
    out = []
    groups = defaultdict(list)
    for pred in preds:
        groups[(pred['image_id'], pred['category_id'])].append(pred)
    for items in groups.values():
        keep = []
        for pred in sorted(items, key=lambda x: x['score'], reverse=True):
            if all(iou(pred['bbox'], old['bbox']) < thr for old in keep):
                keep.append(pred)
        out.extend(keep)
    return out


def load_sam3_jsonl(paths, gt):
    image_id_by_name = {im['file_name']: im['id'] for im in gt['images']}
    preds = []
    for path in paths:
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                image_id = image_id_by_name.get(os.path.basename(row['image_path']))
                if image_id is None:
                    continue
                x1, y1, x2, y2 = [float(v) for v in row['box']]
                if x2 <= x1 or y2 <= y1:
                    continue
                preds.append({
                    'image_id': image_id,
                    'category_id': int(row['category_id']),
                    'bbox': [x1, y1, x2 - x1, y2 - y1],
                    'score': float(row.get('score', 0.5)),
                })
    return preds


def filter_by_image(preds, category_id, min_score, min_area, max_area, topk, nms_thr, score=None, scale=1.0):
    groups = defaultdict(list)
    for pred in preds:
        if pred['category_id'] != category_id or pred['score'] < min_score:
            continue
        x, y, w, h = [float(v) for v in pred['bbox']]
        area = w * h
        if area < min_area or area > max_area:
            continue
        if scale != 1.0:
            cx, cy = x + w / 2.0, y + h / 2.0
            w *= scale
            h *= scale
            x = max(0.0, cx - w / 2.0)
            y = max(0.0, cy - h / 2.0)
        groups[pred['image_id']].append({
            'image_id': pred['image_id'],
            'category_id': category_id,
            'bbox': [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
            'score': float(score if score is not None else pred['score']),
        })
    out = []
    for items in groups.values():
        out.extend(grouped_nms(items, nms_thr)[:topk])
    return out


def make_missing_from_api_fasteners(api_preds, score=0.85, scale=1.2, topk=100):
    groups = defaultdict(list)
    for pred in api_preds:
        if pred['category_id'] != 2:
            continue
        x, y, w, h = [float(v) for v in pred['bbox']]
        area = w * h
        if area < 10000 or area > 160000:
            continue
        cx, cy = x + w / 2.0, y + h / 2.0
        nw, nh = w * scale, h * scale
        groups[pred['image_id']].append({
            'image_id': pred['image_id'],
            'category_id': 3,
            'bbox': [round(max(0.0, cx - nw / 2.0), 2), round(max(0.0, cy - nh / 2.0), 2), round(nw, 2), round(nh, 2)],
            'score': float(score),
        })
    out = []
    for items in groups.values():
        out.extend(grouped_nms(items[:topk], 0.45))
    return out


def evaluate(gt_path, pred_path):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco = COCO(str(gt_path))
    dt = coco.loadRes(str(pred_path))
    ev = COCOeval(coco, dt, 'bbox')
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return {'mAP': float(ev.stats[0]), 'mAP50': float(ev.stats[1]), 'mAP75': float(ev.stats[2]), 'stats': [float(x) for x in ev.stats]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--api-pred', default=str(ROOT / 'qwen_defect_full_api.json'))
    ap.add_argument('--gt', default=str(GT))
    ap.add_argument('--out', default=str(ROOT / 'qwen_defect_full_fusion_v2.json'))
    ap.add_argument('--sam3-jsonl', nargs='*', default=[str(p) for p in sorted(ROOT.glob('predictions_gpu*.jsonl'))])
    ap.add_argument('--evaluate', action='store_true')
    args = ap.parse_args()

    gt = load_json(args.gt)
    api = load_json(args.api_pred)
    sam3 = load_sam3_jsonl([Path(p) for p in args.sam3_jsonl], gt)

    preds = []
    preds.extend(filter_by_image(sam3, 1, min_score=0.0, min_area=300000, max_area=2500000, topk=2, nms_thr=0.45))
    preds.extend(filter_by_image(api, 2, min_score=0.88, min_area=0, max_area=10**12, topk=100, nms_thr=0.60))
    preds.extend(make_missing_from_api_fasteners(api, score=0.85, scale=1.2, topk=100))
    preds.extend(filter_by_image(api, 4, min_score=0.95, min_area=200000, max_area=2000000, topk=4, nms_thr=0.60))
    preds = grouped_nms(preds, 0.45)
    save_json(preds, args.out)
    print(f'wrote {len(preds)} predictions to {args.out}')
    if args.evaluate:
        metrics = evaluate(Path(args.gt), Path(args.out))
        eval_path = str(Path(args.out).with_suffix('')) + '_eval.json'
        save_json(metrics, eval_path)
        print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
