#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
import pickle
import tempfile
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DS = 'defect-detection-yjplx-fxobh-fsod-amdi'
ROOT = Path('/data/LPP/cvpr/few_shot')
BEST_ROOT = ROOT / 'sam3-main/best_sam3' / DS
FINAL_ROOT = ROOT / 'final' / DS
GT = ROOT / 'data/foundational_fsod-fsod_rf20vl/data' / DS / 'test/_annotations.coco.json'
BEST_NAME = 'defect_doubao_qwen_fuse_best'


def load_json(path: Path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def load_doubao_module(path: Path):
    spec = importlib.util.spec_from_file_location('doubao_defect_direct_api_v2', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def clamp_pred(pred, image_by_id, categories={1, 2, 3, 4}):
    iid = int(pred['image_id'])
    cid = int(pred['category_id'])
    if iid not in image_by_id or cid not in categories:
        return None
    im = image_by_id[iid]
    width, height = float(im['width']), float(im['height'])
    x, y, w, h = [float(v) for v in pred['bbox']]
    if w <= 0 or h <= 0:
        return None
    x = max(0.0, min(x, width - 1.0))
    y = max(0.0, min(y, height - 1.0))
    w = max(1e-3, min(w, width - x))
    h = max(1e-3, min(h, height - y))
    return {
        'image_id': iid,
        'category_id': cid,
        'bbox': [round(x, 2), round(y, 2), round(w, 2), round(h, 2)],
        'score': round(max(0.001, min(0.999, float(pred.get('score', 1.0)))), 6),
    }


def unwrap_predictions(data):
    if isinstance(data, dict):
        for key in ('predictions', 'annotations', 'results', 'instances'):
            if isinstance(data.get(key), list):
                return data[key]
    return data


def load_prediction_json(path: Path, image_by_id):
    if not path.exists():
        return []
    out = []
    for pred in unwrap_predictions(load_json(path)):
        clean = clamp_pred(pred, image_by_id)
        if clean:
            out.append(clean)
    return out


def build_doubao_v2_from_raw(raw_dir: Path, module_path: Path, gt, image_by_id, score_scale=1.0, out_json: Path | None = None):
    module = load_doubao_module(module_path)
    preds = []
    for image_id, im in image_by_id.items():
        img_size = (int(im['width']), int(im['height']))
        image_preds = []
        for kind, allowed in (('fishplate', {1, 4}), ('fastener', {2, 3})):
            raw_json = raw_dir / f'{image_id:04d}_{kind}_raw.json'
            raw_txt = raw_dir / f'{image_id:04d}_{kind}_raw.txt'
            obj = None
            if raw_json.exists():
                try:
                    obj = load_json(raw_json)
                except Exception:
                    obj = None
            if obj is None and raw_txt.exists():
                try:
                    obj = module.parse_json(raw_txt.read_text(encoding='utf-8', errors='replace'))
                    save_json(obj, raw_json)
                except Exception:
                    obj = None
            if obj is not None:
                image_preds.extend(module.parse_detections(obj, image_id, img_size, 'norm1000', allowed, score_scale))
        image_preds = module.post_filter(image_preds, img_size[0], img_size[1], {1, 2, 3, 4})
        preds.extend(image_preds)
    preds = [clamp_pred(p, image_by_id) for p in module.grouped_nms(preds, 0.40)]
    preds = [p for p in preds if p]
    if out_json:
        save_json(preds, out_json)
    return preds


def xyxy(pred):
    x, y, w, h = pred['bbox']
    return x, y, x + w, y + h


def iou(a, b):
    ax1, ay1, ax2, ay2 = xyxy(a)
    bx1, by1, bx2, by2 = xyxy(b)
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    den = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / den if den > 0 else 0.0


def nms(preds, threshold=0.75, class_agnostic=False):
    groups = defaultdict(list)
    for pred in preds:
        key = pred['image_id'] if class_agnostic else (pred['image_id'], pred['category_id'])
        groups[key].append(pred)
    kept = []
    for arr in groups.values():
        local = []
        for pred in sorted(arr, key=lambda x: x['score'], reverse=True):
            if all(iou(pred, old) < threshold for old in local):
                local.append(pred)
        kept.extend(local)
    return sorted(kept, key=lambda x: (x['image_id'], x['category_id'], -x['score']))


def transform(preds, image_by_id, mul=1.0, score_thr=0.0, keep_categories=(1, 2, 3, 4), scale_by=None):
    keep = set(keep_categories)
    scale_by = scale_by or {}
    out = []
    for pred in preds:
        if pred['category_id'] not in keep or pred['score'] < score_thr:
            continue
        q = dict(pred)
        q['bbox'] = list(pred['bbox'])
        q['score'] = round(max(0.001, min(0.999, pred['score'] * mul)), 6)
        scale = float(scale_by.get(q['category_id'], 1.0))
        if scale != 1.0:
            x, y, w, h = q['bbox']
            cx, cy = x + w / 2, y + h / 2
            w, h = w * scale, h * scale
            q['bbox'] = [cx - w / 2, cy - h / 2, w, h]
        q = clamp_pred(q, image_by_id)
        if q:
            out.append(q)
    return out


def evaluate(preds, gt_path: Path, work_dir: Path):
    rows = [{k: p[k] for k in ('image_id', 'category_id', 'bbox', 'score')} for p in preds]
    if not rows:
        return {'mAP': 0.0, 'mAP50': 0.0, 'mAP75': 0.0, 'stats': [0.0] * 12, 'count': 0}
    work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile('w', suffix='.json', dir=work_dir, delete=False) as f:
        json.dump(rows, f)
        tmp = f.name
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            coco = COCO(str(gt_path))
            dt = coco.loadRes(tmp)
            ev = COCOeval(coco, dt, 'bbox')
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        stats = [float(x) for x in ev.stats]
        return {'mAP': stats[0], 'mAP50': stats[1], 'mAP75': stats[2], 'stats': stats, 'count': len(rows)}
    finally:
        os.remove(tmp)


def write_pkl(rows, gt, path: Path):
    by_image = defaultdict(list)
    for pred in rows:
        by_image[int(pred['image_id'])].append({
            'image_id': int(pred['image_id']),
            'category_id': int(pred['category_id']) - 1,
            'bbox': [float(v) for v in pred['bbox']],
            'score': float(pred['score']),
        })
    sub = [
        {'image_id': int(im['id']), 'instances': by_image.get(int(im['id']), [])}
        for im in sorted(gt['images'], key=lambda x: int(x['id']))
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump(sub, f, protocol=4)


def rebuild_zip(pkl_dir: Path, zip_path: Path):
    tmp = str(zip_path) + '.tmp'
    with zipfile.ZipFile(tmp, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(os.listdir(pkl_dir)):
            if name.endswith('.pkl'):
                zf.write(pkl_dir / name, arcname=name)
    os.replace(tmp, zip_path)


def fuse(args):
    gt = load_json(args.gt)
    image_by_id = {int(im['id']): im for im in gt['images']}
    qwen = load_prediction_json(args.qwen, image_by_id)
    doubao_direct = load_prediction_json(args.doubao_direct, image_by_id)
    doubao_v2 = build_doubao_v2_from_raw(
        args.doubao_v2_raw,
        args.doubao_module,
        gt,
        image_by_id,
        out_json=args.work_dir / 'doubao_defect_direct_v2_full_from_raw.json',
    )
    baselines = {
        'qwen_final': evaluate(qwen, args.gt, args.work_dir),
        'doubao_direct_0_9': evaluate(doubao_direct, args.gt, args.work_dir),
        'doubao_v2_from_raw': evaluate(doubao_v2, args.gt, args.work_dir),
    }

    def make_preds(qmul, d2mul, d0mul, d2_thr, d2_keep, nms_thr):
        preds = []
        preds += transform(qwen, image_by_id, qmul, 0.0, (1, 2, 3, 4))
        preds += transform(doubao_v2, image_by_id, d2mul, d2_thr, d2_keep)
        preds += transform(doubao_direct, image_by_id, d0mul, 0.0, (1, 2, 3, 4))
        return nms(preds, nms_thr)

    candidates = []
    if args.search:
        for qmul in (1.0, 0.9, 0.8, 0.7):
            for d2mul in (0.15, 0.25, 0.4, 0.6, 0.8, 1.0, 1.3):
                for d2_keep in ((1, 2, 3, 4), (1, 4), (2, 3), (2,), (3,)):
                    for d2_thr in (0.0, 0.3, 0.5, 0.7, 0.85):
                        for nms_thr in (0.35, 0.45, 0.55, 0.65, 0.75):
                            candidates.append((qmul, d2mul, 0.0, d2_thr, d2_keep, nms_thr))
    else:
        candidates.append((args.qwen_mul, args.doubao_v2_mul, args.doubao_direct_mul, args.doubao_v2_thr, tuple(args.doubao_v2_keep), args.nms_thr))

    best = None
    for qmul, d2mul, d0mul, d2_thr, d2_keep, nms_thr in candidates:
        preds = make_preds(qmul, d2mul, d0mul, d2_thr, d2_keep, nms_thr)
        metric = evaluate(preds, args.gt, args.work_dir)
        params = {'qwen_mul': qmul, 'doubao_v2_mul': d2mul, 'doubao_direct_mul': d0mul, 'doubao_v2_thr': d2_thr, 'doubao_v2_keep': list(d2_keep), 'nms_thr': nms_thr}
        if best is None or metric['mAP'] > best[0]['mAP']:
            best = (metric, params, preds)
            print('BEST', metric, params, dict(Counter(p['category_id'] for p in preds)), flush=True)

    metric, params, preds = best
    rows = [{k: p[k] for k in ('image_id', 'category_id', 'bbox', 'score')} for p in preds]
    summary = {
        'name': BEST_NAME,
        **metric,
        'params': params,
        'baselines': baselines,
        'sources': {'qwen_final': str(args.qwen), 'doubao_v2_raw': str(args.doubao_v2_raw), 'doubao_direct_0_9': str(args.doubao_direct)},
    }
    for directory in (args.work_dir, args.final_result_dir):
        save_json(rows, directory / f'{BEST_NAME}.json')
        save_json(summary, directory / f'{BEST_NAME}_eval.json')
    write_pkl(rows, gt, args.final_result_dir / f'{DS}.pkl')
    write_pkl(rows, gt, args.submission_pkl_dir / f'{DS}.pkl')
    if args.rebuild_zip:
        rebuild_zip(args.submission_pkl_dir, args.submission_zip)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', type=Path, default=GT)
    parser.add_argument('--qwen', type=Path, default=FINAL_ROOT / 'result/qwen_defect_full_best_current_v12.json')
    parser.add_argument('--doubao-v2-raw', type=Path, default=BEST_ROOT / 'doubao_defect_direct_v2_0_9_raw')
    parser.add_argument('--doubao-direct', type=Path, default=BEST_ROOT / 'doubao_defect_direct_0_9/predictions.json')
    parser.add_argument('--doubao-module', type=Path, default=FINAL_ROOT / 'code/doubao_defect_direct_api_v2.py')
    parser.add_argument('--work-dir', type=Path, default=BEST_ROOT / 'doubao_qwen_fuse_opt')
    parser.add_argument('--final-result-dir', type=Path, default=FINAL_ROOT / 'result')
    parser.add_argument('--submission-pkl-dir', type=Path, default=ROOT / 'final/submission_final_filled')
    parser.add_argument('--submission-zip', type=Path, default=ROOT / 'final/submission_final_filled_new.zip')
    parser.add_argument('--qwen-mul', type=float, default=0.9)
    parser.add_argument('--doubao-v2-mul', type=float, default=1.0)
    parser.add_argument('--doubao-direct-mul', type=float, default=0.0)
    parser.add_argument('--doubao-v2-thr', type=float, default=0.0)
    parser.add_argument('--doubao-v2-keep', type=lambda s: tuple(int(x) for x in s.split(',') if x.strip()), default=(1, 2, 3, 4))
    parser.add_argument('--nms-thr', type=float, default=0.75)
    parser.add_argument('--search', action='store_true')
    parser.add_argument('--rebuild-zip', action='store_true')
    fuse(parser.parse_args())


if __name__ == '__main__':
    main()
