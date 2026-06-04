#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

import qwen_wb_direct_api_v2 as base

ROOT = Path(__file__).resolve().parent
DATA = base.DATA
GT = base.GT
TRAIN = base.TRAIN
VALID = base.VALID
CLASS = base.CLASS
CLASSES = [1, 2, 3]
NAME_TO_ID = {v.lower(): k for k, v in CLASS.items()}


def image_url(img: Image.Image, max_side=1400):
    img = img.convert('RGB')
    if max(img.size) > max_side:
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=92)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode()


def pil_context(img: Image.Image, bbox):
    out = img.convert('RGB').copy()
    dr = ImageDraw.Draw(out)
    x, y, w, h = [float(v) for v in bbox]
    width = max(4, round(max(out.size) / 300))
    dr.rectangle([x, y, x + w, y + h], outline=(255, 0, 0), width=width)
    dr.rectangle([0, 0, 520, 46], fill=(255, 255, 255))
    dr.text((10, 14), 'Classify only the red-boxed pig', fill=(255, 0, 0))
    return out


def pil_crop(img: Image.Image, bbox, pad=0.45):
    x, y, w, h = [float(v) for v in bbox]
    p = pad * max(w, h)
    x0, y0 = max(0, x - p), max(0, y - p)
    x1, y1 = min(img.width, x + w + p), min(img.height, y + h + p)
    crop = img.crop((x0, y0, x1, y1)).convert('RGB')
    sx = crop.width / max(1e-6, x1 - x0)
    sy = crop.height / max(1e-6, y1 - y0)
    rb = [(x - x0) * sx, (y - y0) * sy, (x + w - x0) * sx, (y + h - y0) * sy]
    dr = ImageDraw.Draw(crop)
    dr.rectangle(rb, outline=(255, 0, 0), width=5)
    dr.rectangle([0, 0, 400, 40], fill=(255, 255, 255))
    dr.text((8, 11), 'Target pig is inside red box', fill=(255, 0, 0))
    return crop


def read_cv(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def cv_crop(img, bbox, pad=0.2):
    H, W = img.shape[:2]
    x, y, w, h = [float(v) for v in bbox]
    p = pad * max(w, h)
    x0 = max(0, int(round(x - p)))
    y0 = max(0, int(round(y - p)))
    x1 = min(W, int(round(x + w + p)))
    y1 = min(H, int(round(y + h + p)))
    return img[y0:max(y0 + 1, y1), x0:max(x0 + 1, x1)]


def feature(img, bbox, image_size, pad=0.2):
    crop = cv_crop(img, bbox, pad)
    if crop.size == 0:
        crop = np.zeros((32, 32, 3), np.uint8)
    crop = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    feats = []
    for ch, bins, ran in [
        (hsv[:, :, 0], 12, [0, 180]),
        (hsv[:, :, 1], 8, [0, 256]),
        (hsv[:, :, 2], 8, [0, 256]),
        (lab[:, :, 1], 8, [0, 256]),
        (lab[:, :, 2], 8, [0, 256]),
    ]:
        hist = cv2.calcHist([ch], [0], None, [bins], ran).reshape(-1)
        hist = hist / (hist.sum() + 1e-9)
        feats.extend(hist.tolist())
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    feats.extend([float(gray.mean()) / 255.0, float(gray.std()) / 255.0])
    feats.append(float(cv2.Canny(gray, 80, 160).mean()) / 255.0)
    x, y, w, h = [float(v) for v in bbox]
    W, H = image_size
    feats.extend([
        math.log(max(w * h, 1.0)) / math.log(W * H + 1.0),
        w / max(h, 1.0),
        w / max(W, 1.0),
        h / max(H, 1.0),
        (x + w / 2.0) / max(W, 1.0),
        (y + h / 2.0) / max(H, 1.0),
    ])
    return np.array(feats, dtype=np.float32)


def load_knn_refs(pad=0.2):
    xs, ys = [], []
    for split, ann_path in [('train', TRAIN), ('valid', VALID)]:
        data = base.load_json(ann_path)
        image_by = {im['id']: im for im in data['images']}
        cache = {}
        for ann in data['annotations']:
            cid = int(ann['category_id'])
            if cid not in CLASSES:
                continue
            im = image_by[ann['image_id']]
            if ann['image_id'] not in cache:
                cache[ann['image_id']] = read_cv(DATA / split / im['file_name'])
            xs.append(feature(cache[ann['image_id']], ann['bbox'], (im['width'], im['height']), pad))
            ys.append(cid)
    X = np.stack(xs)
    y = np.array(ys, dtype=np.int32)
    mu = X.mean(axis=0)
    sig = X.std(axis=0) + 1e-6
    return (X - mu) / sig, y, mu, sig


def knn_probs(f, Xn, y, k=9, temp=3.0):
    d = np.sqrt(((Xn - f) ** 2).sum(axis=1))
    idx = np.argsort(d)[:k]
    weights = np.exp(-d[idx] / temp)
    probs = {cid: 1e-6 for cid in CLASSES}
    neighbors = []
    for rank, (ii, wt) in enumerate(zip(idx, weights), 1):
        cid = int(y[ii])
        probs[cid] += float(wt)
        neighbors.append({'rank': rank, 'category_id': cid, 'category_name': CLASS[cid], 'distance': float(d[ii])})
    total = sum(probs.values())
    return {CLASS[cid]: probs[cid] / total for cid in CLASSES}, neighbors


def prompt(knn, neighbors, orig_category_id):
    neighbor_text = ', '.join(f"#{n['rank']} {n['category_name']} d={n['distance']:.2f}" for n in neighbors[:5])
    return f"""
You are classifying one detected pig for a no-gradient in-context prompting pipeline.
The red-boxed pig is the only target. Use the full-scene image and the close crop.

Labels:
1 Adult: mature pig, bulky/deep body, thick neck/shoulders, full-grown proportions. A distant adult can look small.
2 Juvenile: young growing pig, between adult and piglet, slimmer adolescent proportions, not fully mature.
3 Piglet: baby or very young pig, baby-like head/body proportions, short legs, small/round body. A close-up piglet can fill the crop.

Auxiliary non-gradient KNN visual prior from labeled in-context examples:
- Original detector category: {orig_category_id} {CLASS.get(int(orig_category_id), 'unknown')}
- KNN probabilities: Adult={knn['Adult']:.3f}, Juvenile={knn['Juvenile']:.3f}, Piglet={knn['Piglet']:.3f}
- Nearest labels summary: {neighbor_text}

Use the KNN prior as a hint, not as ground truth. Override it if the image clearly disagrees.
Judge age/proportions, not pixel size or crop scale. Use nearby pigs only as relative context.
Return ONLY JSON:
{{"category_id":1,"category_name":"Adult","probabilities":{{"Adult":0.70,"Juvenile":0.20,"Piglet":0.10}},"confidence":0.70,"reason":"short"}}
""".strip()


def call_api(ctx, crop, text, api_key, model, timeout=120, retries=1, max_side=1400):
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': text},
            {'type': 'image_url', 'image_url': {'url': image_url(ctx, max_side)}},
            {'type': 'image_url', 'image_url': {'url': image_url(crop, max_side)}},
        ]}],
        'temperature': 0,
        'response_format': {'type': 'json_object'},
        'enable_thinking': False,
        'thinking': {'type': 'disabled'},
    }
    data = json.dumps(payload).encode()
    api_key = (api_key or '').strip().replace('\ufeff', '')
    last = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            'https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',
            data=data,
            headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
            method='POST',
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                obj = json.loads(resp.read().decode())
            return obj['choices'][0]['message']['content']
        except urllib.error.HTTPError as e:
            last = RuntimeError(f'HTTP {e.code}: ' + e.read().decode(errors='replace'))
        except Exception as e:
            last = e
        time.sleep(2 + attempt * 2)
    raise RuntimeError(last)


def parse_json(text):
    text = text.strip().replace('```json', '').replace('```', '').strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'\{.*\}', text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def probs_from_obj(obj):
    raw = obj.get('probabilities') or {}
    probs = {name: 0.0 for name in ['Adult', 'Juvenile', 'Piglet']}
    if isinstance(raw, dict):
        for name in probs:
            val = raw.get(name, raw.get(name.lower()))
            if val is not None:
                try:
                    probs[name] = max(0.0, float(val))
                except Exception:
                    pass
    if sum(probs.values()) <= 0:
        cid = obj.get('category_id')
        if cid is None:
            cid = NAME_TO_ID.get(str(obj.get('category_name', '')).strip().lower(), 1)
        try:
            cid = int(cid)
        except Exception:
            cid = 1
        probs[CLASS.get(cid, 'Adult')] = 1.0
    total = sum(probs.values())
    return {name: probs[name] / total for name in probs}


def build(preds, probs, top2_scale=0.04, min_top2_prob=0.20):
    out = []
    for p, pr in zip(preds, probs):
        ranked = sorted(pr, key=lambda name: pr[name], reverse=True)
        top = ranked[0]
        q = dict(p)
        q['category_id'] = NAME_TO_ID[top.lower()]
        q['score'] = round(float(p.get('score', 0.8)) * (0.72 + 0.28 * pr[top]), 6)
        out.append(q)
        second = ranked[1]
        if pr[second] >= min_top2_prob:
            r = dict(p)
            r['category_id'] = NAME_TO_ID[second.lower()]
            r['score'] = round(float(p.get('score', 0.8)) * top2_scale * pr[second] / max(pr[top], 1e-6), 6)
            out.append(r)
    return [{
        'image_id': int(p['image_id']),
        'category_id': int(p['category_id']),
        'bbox': [round(float(v), 2) for v in p['bbox']],
        'score': round(float(p['score']), 6),
    } for p in out]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pred', type=Path, required=True)
    ap.add_argument('--out-dir', type=Path, required=True)
    ap.add_argument('--model', default='qwen3.5-plus')
    ap.add_argument('--api-key', default=os.environ.get('DASHSCOPE_API_KEY') or os.environ.get('QWEN_API_KEY'))
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--timeout', type=int, default=120)
    ap.add_argument('--retries', type=int, default=1)
    ap.add_argument('--limit', type=int, default=0, help='Limit number of prediction rows; prefer --image-ids for fair eval.')
    ap.add_argument('--image-ids', default='', help='Comma/range image ids, e.g. 0-29. Keeps all predictions on those images.')
    ap.add_argument('--k', type=int, default=9)
    ap.add_argument('--temp', type=float, default=3.0)
    ap.add_argument('--knn-pad', type=float, default=0.2)
    ap.add_argument('--crop-pad', type=float, default=0.45)
    ap.add_argument('--top2-scale', type=float, default=0.04)
    ap.add_argument('--min-top2-prob', type=float, default=0.20)
    args = ap.parse_args()
    if not args.api_key:
        raise SystemExit('set DASHSCOPE_API_KEY/QWEN_API_KEY')
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out_dir / 'raw'
    crop_dir = args.out_dir / 'crops'
    raw_dir.mkdir(exist_ok=True)
    crop_dir.mkdir(exist_ok=True)

    Xn, y, mu, sig = load_knn_refs(args.knn_pad)
    gt = base.load_json(GT)
    image_by = {int(im['id']): im for im in gt['images']}
    preds = base.load_json(args.pred)
    eval_image_ids = sorted(image_by)
    if args.image_ids:
        wanted = set(base.parse_ids(args.image_ids, image_by))
        preds = [p for p in preds if int(p['image_id']) in wanted]
        eval_image_ids = sorted(wanted)
    elif args.limit > 0:
        preds = preds[:args.limit]
        eval_image_ids = sorted({int(p['image_id']) for p in preds})

    pil_cache = {}
    cv_cache = {}
    final_probs = []
    summaries = []
    for i, p in enumerate(preds):
        im = image_by[int(p['image_id'])]
        iid = int(p['image_id'])
        if iid not in pil_cache:
            pil_cache[iid] = Image.open(DATA / 'test' / im['file_name']).convert('RGB')
            cv_cache[iid] = read_cv(DATA / 'test' / im['file_name'])
        f = (feature(cv_cache[iid], p['bbox'], (im['width'], im['height']), args.knn_pad) - mu) / sig
        knn, neighbors = knn_probs(f, Xn, y, args.k, args.temp)
        ctx = pil_context(pil_cache[iid], p['bbox'])
        crop = pil_crop(pil_cache[iid], p['bbox'], args.crop_pad)
        crop.save(crop_dir / f'{i:04d}_img{iid}.jpg')
        rp = raw_dir / f'{i:04d}_img{iid}.json'
        tp = raw_dir / f'{i:04d}_img{iid}.txt'
        prompt_text = prompt(knn, neighbors, int(p['category_id']))
        if args.resume and rp.exists():
            obj = json.load(open(rp))
        else:
            txt = call_api(ctx, crop, prompt_text, args.api_key, args.model, args.timeout, args.retries)
            tp.write_text(txt)
            obj = parse_json(txt)
            json.dump(obj, open(rp, 'w'), indent=2)
        probs = probs_from_obj(obj)
        final_probs.append(probs)
        top = max(probs, key=probs.get)
        summaries.append({'index': i, 'image_id': iid, 'old_category_id': p['category_id'], 'knn': knn, 'qwen_probs': probs, 'raw': obj})
        print(i, 'img', iid, 'old', p['category_id'], 'knn', knn, 'qwen->', NAME_TO_ID[top.lower()], probs, flush=True)

    out = build(preds, final_probs, args.top2_scale, args.min_top2_prob)
    json.dump(out, open(args.out_dir / 'predictions.json', 'w'), indent=2)
    json.dump(summaries, open(args.out_dir / 'knn_prompt_raw.json', 'w'), indent=2)
    metrics = base.evaluate(out, eval_image_ids, args.out_dir)
    metrics.update({
        'source': str(args.pred),
        'method': 'qwen_crop_reclass_with_knn_prompt_prior',
        'model': args.model,
        'predictions': len(out),
        'eval_image_ids': eval_image_ids,
        'counts': dict(Counter(p['category_id'] for p in out)),
        'k': args.k,
        'temp': args.temp,
        'top2_scale': args.top2_scale,
        'min_top2_prob': args.min_top2_prob,
    })
    json.dump(metrics, open(args.out_dir / 'eval.json', 'w'), indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
