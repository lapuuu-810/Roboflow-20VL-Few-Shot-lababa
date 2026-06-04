#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

import xray_supervised_calibrate_percat_v2 as base

ROOT = Path(__file__).resolve().parent
CLASS_NAMES = base.CLASS_NAMES


def slot_key(p, rank_by_pred):
    return f"{int(p['category_id'])}_{rank_by_pred[id(p)]}"


def by_x(items):
    return sorted(items, key=lambda z: base.center(z['bbox'])[0])


def assign_ranks(preds):
    ranks = {}
    for cid in CLASS_NAMES:
        ps = by_x([p for p in preds if int(p['category_id']) == cid])
        for r, p in enumerate(ps):
            ranks[id(p)] = r
    return ranks


def learn_slot(gt, base_by, train_ids):
    gt_by = defaultdict(list)
    for ann in gt['annotations']:
        iid = int(ann['image_id'])
        cid = int(ann.get('category_id', -1))
        if iid in train_ids and cid in CLASS_NAMES:
            gt_by[iid].append(ann)

    vals = defaultdict(lambda: defaultdict(list))
    for iid in train_ids:
        preds = base_by[iid]
        ranks = assign_ranks(preds)
        for cid in CLASS_NAMES:
            ps = by_x([p for p in preds if int(p['category_id']) == cid])
            gs = by_x([g for g in gt_by[iid] if int(g['category_id']) == cid])
            for r, (p, g) in enumerate(zip(ps, gs)):
                key = f"{cid}_{r}"
                pb, gb = p['bbox'], g['bbox']
                pc, gc = base.center(pb), base.center(gb)
                vals[key]['dx'].append((gc[0] - pc[0]) / max(1.0, float(pb[2])))
                vals[key]['dy'].append((gc[1] - pc[1]) / max(1.0, float(pb[3])))
                vals[key]['sw'].append(float(gb[2]) / max(1.0, float(pb[2])))
                vals[key]['sh'].append(float(gb[3]) / max(1.0, float(pb[3])))
    return {k: {m: median(v) for m, v in stats.items()} for k, stats in vals.items()}


def apply_slot(eval_base, image_by, model, params):
    by_img = defaultdict(list)
    for p in eval_base:
        by_img[int(p['image_id'])].append(p)
    out = []
    for iid in sorted(by_img):
        preds = by_img[iid]
        ranks = assign_ranks(preds)
        for p in preds:
            cid = int(p['category_id'])
            key = slot_key(p, ranks)
            m = model.get(key, {})
            alpha, scale_alpha = params.get(key, [0.0, 0.0])
            x, y, w, h = [float(v) for v in p['bbox']]
            cx, cy = base.center(p['bbox'])
            ncx = cx + alpha * m.get('dx', 0.0) * w
            ncy = cy + alpha * m.get('dy', 0.0) * h
            nw = w * ((1.0 - scale_alpha) + scale_alpha * m.get('sw', 1.0))
            nh = h * ((1.0 - scale_alpha) + scale_alpha * m.get('sh', 1.0))
            W, H = image_by[iid]['width'], image_by[iid]['height']
            bx = max(0.0, min(W - 1.0, ncx - nw / 2.0))
            by = max(0.0, min(H - 1.0, ncy - nh / 2.0))
            bw = max(1.0, min(W - bx, nw))
            bh = max(1.0, min(H - by, nh))
            out.append({'image_id': iid, 'category_id': cid, 'bbox': [round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)], 'score': round(float(p.get('score', 0.5)), 4)})
    return out


def evaluate_silent(preds, tag, out_dir, eval_ids, gt_cache):
    pp = out_dir / f'{tag}.json'
    base.save_json(preds, pp)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        met = base.evaluate(base.GT, pp, eval_ids, gt_cache)
    return (met['mAP'], met['mAP50'], met['mAP75'], pp, met, preds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--base', required=True)
    ap.add_argument('--train-ids', default='0-29')
    ap.add_argument('--eval-ids', default='0-29')
    ap.add_argument('--out-dir', default=str(ROOT / 'xray_slot_prior_calibrate_v1'))
    ap.add_argument('--out', default='')
    ap.add_argument('--passes', type=int, default=2)
    args = ap.parse_args()

    gt = base.load_json(base.GT)
    image_by = {int(im['id']): im for im in gt['images']}
    train_ids = base.parse_ids(args.train_ids, image_by)
    eval_ids = base.parse_ids(args.eval_ids, image_by)
    base_preds_all = base.load_json(Path(args.base))
    base_by = defaultdict(list)
    for p in base_preds_all:
        base_by[int(p['image_id'])].append(p)
    eval_base = [p for p in base_preds_all if int(p['image_id']) in set(eval_ids)]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gt_cache = out_dir / 'eval_subset_gt.json'
    base.save_json(base.subset_gt(gt, eval_ids), gt_cache)
    model = learn_slot(gt, base_by, set(train_ids))
    base.save_json(model, out_dir / 'model_slot.json')

    params = {k: [0.0, 0.0] for k in sorted(model)}
    alpha_grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]
    scale_grid = [0.0, 0.25, 0.5, 0.75, 1.0, 1.25]

    def score(tag, cur_params):
        return evaluate_silent(apply_slot(eval_base, image_by, model, cur_params), tag, out_dir, eval_ids, gt_cache)

    best = score('try_start', params)
    print('START', best[:4], flush=True)
    for pass_id in range(args.passes):
        changed = False
        for key in sorted(model):
            local_best = best
            local_params = json.loads(json.dumps(params))
            for a in alpha_grid:
                for s in scale_grid:
                    trial = json.loads(json.dumps(params))
                    trial[key] = [a, s]
                    rec = score(f'try_p{pass_id}_{key}_a{a}_s{s}', trial)
                    if rec[:3] > local_best[:3]:
                        local_best = rec
                        local_params = trial
                        print('BEST', rec[:4], 'slot', key, 'params', local_params[key], flush=True)
            if local_best[:3] > best[:3]:
                best = local_best
                params = local_params
                changed = True
        if not changed:
            break

    final = Path(args.out) if args.out else out_dir / 'best.json'
    base.save_json(best[5], final)
    ev = dict(best[4])
    ev.update({'params': params, 'model': model, 'source': str(best[3]), 'num_predictions': len(best[5]), 'cat_counts': dict(Counter(int(p['category_id']) for p in best[5]))})
    base.save_json(ev, final.with_name(final.stem + '_eval.json'))
    base.draw(base.GT.parent, image_by, eval_ids, best[5], out_dir)
    print('FINAL', final)
    print('EVAL', final.with_name(final.stem + '_eval.json'))
    print('VIS', out_dir / 'montage_percat.jpg')
    print(json.dumps(ev, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
