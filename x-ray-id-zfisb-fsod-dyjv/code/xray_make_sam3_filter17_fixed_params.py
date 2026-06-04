#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import sam3_finger_bone_filter17 as sam

# 30-image best SAM3 filter17 checkpoint try_0010 params.
BEST_PARAMS_0_29 = (0.05, 40, 2500, 2.0, 80, 0.65, 1.0, 3.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--image-ids', default='all')
    ap.add_argument('--out-dir', default=str(sam.ROOT / 'sam3_finger_bone_filter17_full_current'))
    ap.add_argument('--params', default='0.05,40,2500,2.0,80,0.65,1.0,3.5')
    args = ap.parse_args()

    params = tuple(float(x) for x in args.params.split(','))
    params = (params[0], int(params[1]), int(params[2]), params[3], int(params[4]), params[5], params[6], params[7])

    gt = sam.load_json(sam.GT)
    image_by_id = {int(im['id']): im for im in gt['images']}
    ids = sam.parse_ids(args.image_ids, image_by_id)

    tpl_by = defaultdict(list)
    for p in sam.load_json(sam.TEMPLATE):
        if int(p['image_id']) in set(ids):
            tpl_by[int(p['image_id'])].append(p)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    preds, debug = sam.make_preds(ids, image_by_id, tpl_by, params)
    final = out_dir / 'sam3_finger_bone_filter17_best.json'
    sam.save_json(preds, final)
    metrics = sam.eval_preds(sam.GT, final, ids)
    metrics.update({
        'params': params,
        'source': 'fixed_params_from_0_29_try_0010',
        'num_predictions': len(preds),
        'count_by_image': dict(Counter(p['image_id'] for p in preds)),
        'cat_counts': dict(Counter(p['category_id'] for p in preds)),
        'debug': debug,
    })
    sam.save_json(metrics, out_dir / 'sam3_finger_bone_filter17_best_eval.json')
    sam.draw_vis(ids, image_by_id, preds, out_dir, debug)
    print('FINAL', final)
    print('EVAL', out_dir / 'sam3_finger_bone_filter17_best_eval.json')
    print('VIS', out_dir / 'montage_fixed17_clean.jpg')
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == '__main__':
    main()
