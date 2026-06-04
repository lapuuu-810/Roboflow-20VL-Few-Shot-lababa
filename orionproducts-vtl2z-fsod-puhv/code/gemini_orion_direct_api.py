#!/usr/bin/env python3
from __future__ import annotations

import argparse, base64, io, json, os, re, time, urllib.error, urllib.request
from pathlib import Path
from collections import defaultdict
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
DATA = Path('/data/LPP/few_shot/data/foundational_fsod-fsod_rf20vl/data/orionproducts-vtl2z-fsod-puhv')
GT = DATA / 'test' / '_annotations.coco.json'

CLASS_NAMES = {
    1: 'Candy Boom',
    2: 'Chocopie Dark',
    3: 'Chocopie Nor',
    4: 'Marine Boy',
    5: 'OStar Red',
    6: 'OStar Yellow',
    7: 'Swing Maxx',
    8: 'Swing Nor',
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def image_b64(img):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=94)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def parse_ids(text, image_by_id):
    if text.strip().lower() == 'all':
        return sorted(image_by_id)
    return [int(x) for x in text.split(',') if x.strip()]


def make_prompt(width, height):
    names = '\n'.join(f'{cid} {name}' for cid, name in CLASS_NAMES.items())
    return f"""
Output exactly one valid JSON object. No explanation.
You are doing zero-shot object detection on a retail shelf/product image.
Image size: width={width}, height={height}. Return absolute pixel coordinates.

Detect every visible individual packaged product that belongs to one of these official classes:
{names}

Rules:
- Draw a tight box around each visible product package, not the whole shelf or group.
- Detect repeated instances separately.
- Ignore products that are not one of the 8 official classes.
- If a package is partly occluded but the product identity is still clear, detect the visible package extent.
- Use the printed package appearance, logo text, color, and shape to choose the class.
- Do not merge adjacent products into one box.
- Avoid duplicate boxes for the same product instance.

Return schema:
{{"detections":[{{"category_id":6,"category_name":"OStar Yellow","bbox":[x1,y1,x2,y2],"confidence":0.92}}]}}
All bbox coordinates must be numbers in [0,width] and [0,height].
""".strip()


def extract_response_text(obj):
    texts = []
    for cand in obj.get('candidates', []):
        content = cand.get('content', {})
        for part in content.get('parts', []):
            text = part.get('text')
            if isinstance(text, str):
                texts.append(text)
    if texts:
        return '\n'.join(texts)
    if isinstance(obj.get('text'), str):
        return obj['text']
    raise RuntimeError('Cannot extract text from Gemini response: ' + json.dumps(obj, ensure_ascii=False)[:1000])


def call_api(img, prompt, api_key, model, timeout, retries):
    payload = {
        'contents': [{
            'role': 'user',
            'parts': [
                {
                    'inline_data': {
                        'mime_type': 'image/jpeg',
                        'data': image_b64(img),
                    }
                },
                {'text': prompt},
            ],
        }],
        'generationConfig': {
            'temperature': 0,
            'responseMimeType': 'application/json',
        },
    }
    data = json.dumps(payload).encode('utf-8')
    api_key = (api_key or '').strip().replace('\ufeff', '')
    last = None
    url = f'https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent'
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            url,
            data=data,
            headers={'x-goog-api-key': api_key, 'Content-Type': 'application/json'},
            method='POST',
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                obj = json.loads(resp.read().decode('utf-8'))
            return extract_response_text(obj)
        except urllib.error.HTTPError as e:
            last = RuntimeError(f'HTTP {e.code}: ' + e.read().decode('utf-8', errors='replace'))
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


def clamp_box(box, width, height, coord_mode):
    if not box or len(box) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    if coord_mode == 'norm1000':
        x1, x2 = x1 / 1000 * width, x2 / 1000 * width
        y1, y2 = y1 / 1000 * height, y2 / 1000 * height
    if x2 <= x1 or y2 <= y1:
        return None
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]


def iou(a, b):
    ax, ay, aw, ah = map(float, a)
    bx, by, bw, bh = map(float, b)
    inter = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0


def nms(preds, thr):
    out = []
    for cid in sorted(set(p['category_id'] for p in preds)):
        keep = []
        for pred in sorted([p for p in preds if p['category_id'] == cid], key=lambda x: x['score'], reverse=True):
            if all(iou(pred['bbox'], prev['bbox']) < thr for prev in keep):
                keep.append(pred)
        out.extend(keep)
    return out


def draw_vis(img, preds, path):
    vis = img.copy().convert('RGB')
    draw = ImageDraw.Draw(vis)
    colors = [(255, 0, 0), (0, 180, 255), (255, 160, 0), (0, 220, 80), (210, 80, 255), (255, 80, 160), (255, 230, 0), (0, 230, 210)]
    for p in preds:
        x, y, w, h = p['bbox']
        col = colors[(p['category_id'] - 1) % len(colors)]
        draw.rectangle((x, y, x + w, y + h), outline=col, width=3)
        draw.text((x, y), f"{p['category_id']} {p['score']:.2f}", fill=col)
    vis.save(path)


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
    ap.add_argument('--api-key', default=os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY'))
    ap.add_argument('--model', default='gemini-2.5-flash')
    ap.add_argument('--gt', default=str(GT))
    ap.add_argument('--image-ids', default='0,1,2,3,4')
    ap.add_argument('--out', default=str(ROOT / 'gemini_orion_direct_test.json'))
    ap.add_argument('--raw-dir', default=str(ROOT / 'gemini_orion_direct_raw'))
    ap.add_argument('--coord-mode', default='pixel', choices=['pixel', 'norm1000'])
    ap.add_argument('--timeout', type=int, default=150)
    ap.add_argument('--retries', type=int, default=1)
    ap.add_argument('--min-score', type=float, default=0.05)
    ap.add_argument('--nms', type=float, default=0.45)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--evaluate', action='store_true')
    args = ap.parse_args()
    if not args.api_key:
        raise SystemExit('set GEMINI_API_KEY or GOOGLE_API_KEY')

    gt = load_json(args.gt)
    image_by_id = {im['id']: im for im in gt['images']}
    ids = parse_ids(args.image_ids, image_by_id)
    image_dir = Path(args.gt).parent
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    preds = []
    for n, image_id in enumerate(ids, 1):
        meta = image_by_id[image_id]
        img = Image.open(image_dir / meta['file_name']).convert('RGB')
        raw_json = raw_dir / f'{image_id:04d}_raw.json'
        if args.resume and raw_json.exists():
            obj = load_json(raw_json)
        else:
            prompt = make_prompt(img.width, img.height)
            (raw_dir / f'{image_id:04d}_prompt.txt').write_text(prompt)
            print(f'[{n}/{len(ids)}] API direct image_id={image_id} {meta["file_name"]}', flush=True)
            raw = call_api(img, prompt, args.api_key, args.model, args.timeout, args.retries)
            (raw_dir / f'{image_id:04d}_raw.txt').write_text(raw)
            obj = parse_json(raw)
            save_json(obj, raw_json)

        dets = []
        for det in obj.get('detections', obj.get('items', [])):
            try:
                cid = int(det.get('category_id', 0) or 0)
            except Exception:
                continue
            if cid not in CLASS_NAMES:
                continue
            box = clamp_box(det.get('bbox'), img.width, img.height, args.coord_mode)
            if not box:
                continue
            score = float(det.get('confidence', det.get('score', 0.8)) or 0.8)
            if score < args.min_score:
                continue
            dets.append({'image_id': image_id, 'category_id': cid, 'bbox': box, 'score': round(max(0.001, min(0.999, score)), 6)})
        dets = nms(dets, args.nms)
        preds.extend(dets)
        draw_vis(img, dets, raw_dir / f'{image_id:04d}_vis.jpg')
        print(f'  detections={len(dets)}', flush=True)

    save_json(preds, args.out)
    print(f'wrote {len(preds)} predictions to {args.out}')
    if args.evaluate:
        eval_gt = Path(args.gt)
        if args.image_ids.strip().lower() != 'all':
            idset = set(ids)
            sub = dict(gt)
            sub['images'] = [im for im in gt['images'] if im['id'] in idset]
            sub['annotations'] = [a for a in gt['annotations'] if a['image_id'] in idset]
            eval_gt = Path(str(Path(args.out).with_suffix('')) + '_subset_gt.json')
            save_json(sub, eval_gt)
        metrics = evaluate(eval_gt, Path(args.out))
        save_json(metrics, str(Path(args.out).with_suffix('')) + '_eval.json')
        print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
