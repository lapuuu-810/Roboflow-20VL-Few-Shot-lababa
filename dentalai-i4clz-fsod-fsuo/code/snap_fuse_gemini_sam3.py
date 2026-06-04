#!/usr/bin/env python3
from __future__ import annotations

import argparse, glob, math
from collections import defaultdict
from pathlib import Path

from gemini_dental_sam3_candidate_api import GT, evaluate, iou_box, load_json, load_sam3_candidates, nms, save_json


def center(b):
    return (b[0] + b[2] / 2, b[1] + b[3] / 2)


def snap(api_preds, cands, max_dist=38.0):
    by = defaultdict(list)
    for c in cands:
        by[c['image_id']].append(c)
    out = []
    for p in api_preds:
        pc = center(p['bbox'])
        best = None
        best_score = -1
        for c in by[p['image_id']]:
            cc = center(c['bbox'])
            dist = math.hypot(pc[0] - cc[0], pc[1] - cc[1])
            score = -dist + 18 * iou_box(p['bbox'], c['bbox']) + 8 * float(c.get('score', 0.5))
            if dist <= max_dist and score > best_score:
                best_score = score
                best = c
        if best:
            out.append({
                'image_id': p['image_id'],
                'category_id': p['category_id'],
                'bbox': [float(v) for v in best['bbox']],
                'score': round(float(p.get('score', 0.8)) * 0.85 + float(best.get('score', 0.5)) * 0.15, 6),
            })
    return out


def filt(preds, nms_thr, max_img, max_cat):
    byc = defaultdict(list)
    for p in preds:
        x, y, w, h = p['bbox']
        area = w * h
        cid = p['category_id']
        if area < 50 or area > 7000:
            continue
        byc[(p['image_id'], cid)].append(p)
    byi = defaultdict(list)
    for key, items in byc.items():
        byi[key[0]].extend(sorted(nms(items, nms_thr), key=lambda z: z['score'], reverse=True)[:max_cat])
    out = []
    for iid, items in byi.items():
        out.extend(sorted(items, key=lambda z: z['score'], reverse=True)[:max_img])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--api', required=True)
    ap.add_argument('--crop')
    ap.add_argument('--image-ids', default='0,1')
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    ids = [int(x) for x in args.image_ids.split(',') if x.strip()] if args.image_ids.strip().lower() != 'all' else None
    gt = load_json(GT)
    if ids is None:
        ids = sorted(im['id'] for im in gt['images'])
    root = Path(__file__).resolve().parent
    cands = load_sam3_candidates(
        gt,
        [Path(p) for p in sorted(glob.glob(str(root / 'predictions_gpu*.jsonl')))],
        ids,
        40,
        10000,
        0.0,
        0.65,
        1000,
    )
    api = load_json(args.api)
    crop = load_json(args.crop) if args.crop else []
    best = None
    for dist in [12, 18, 25, 32, 40, 55, 80]:
        snapped = snap(api, cands, dist)
        for use_crop in [False, True]:
            base = snapped + (crop if use_crop else [])
            for nms_thr in [0.25, 0.35, 0.45, 0.55]:
                for max_img in [16, 24, 32, 50, 100]:
                    for max_cat in [6, 10, 16, 40]:
                        preds = filt(base, nms_thr, max_img, max_cat)
                        if not preds:
                            continue
                        save_json(preds, '_tmp_snap_eval.json')
                        idset = set(ids)
                        sub = dict(gt)
                        sub['images'] = [im for im in gt['images'] if im['id'] in idset]
                        sub['annotations'] = [a for a in gt['annotations'] if a['image_id'] in idset]
                        save_json(sub, '_tmp_snap_gt.json')
                        m = evaluate(Path('_tmp_snap_gt.json'), Path('_tmp_snap_eval.json'))
                        rec = (m['mAP'], m['mAP50'], dist, use_crop, nms_thr, max_img, max_cat, preds, m)
                        if best is None or rec[:2] > best[:2]:
                            best = rec
                            print('best', round(rec[0], 4), round(rec[1], 4), 'n', len(preds), 'params', rec[2:7])
    save_json(best[7], args.out)
    save_json({'mAP': best[8]['mAP'], 'mAP50': best[8]['mAP50'], 'stats': best[8]['stats'], 'params': best[2:7]}, str(Path(args.out).with_suffix('')) + '_eval.json')
    print('saved', args.out, 'params', best[2:7])


if __name__ == '__main__':
    main()
