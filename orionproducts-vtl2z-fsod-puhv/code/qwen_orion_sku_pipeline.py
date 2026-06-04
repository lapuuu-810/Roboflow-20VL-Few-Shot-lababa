#!/usr/bin/env python3
from __future__ import annotations

import argparse, base64, io, json, os, re, time, urllib.error, urllib.request
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw

ROOT = Path('/data/LPP/cvpr/few_shot/sam3-main/best_sam3/orionproducts-vtl2z-fsod-puhv')
DATA = Path('/data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data/orionproducts-vtl2z-fsod-puhv')
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


def image_url(img):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=94)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def call_api(img, prompt, api_key, model, timeout, retries):
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': prompt},
            {'type': 'image_url', 'image_url': {'url': image_url(img)}},
        ]}],
        'temperature': 0,
        'response_format': {'type': 'json_object'},
        'enable_thinking': False,
        'thinking': {'type': 'disabled'},
    }
    data = json.dumps(payload).encode('utf-8')
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
                obj = json.loads(resp.read().decode('utf-8'))
            return obj['choices'][0]['message']['content']
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
        if m:
            candidate = m.group(0)
            try:
                return json.loads(candidate)
            except Exception:
                # Common VLM failure: valid prefix plus dangling malformed tail.
                for key in ('"evidence"', '"reason"', '"category_id"', '"detections"', '"items"'):
                    pos = candidate.rfind(',' + key)
                    if pos > 0:
                        repaired = candidate[:pos] + '}'
                        try:
                            return json.loads(repaired)
                        except Exception:
                            pass
        return {'category_id': 0, 'confidence': 0.0, 'ocr_text': '', 'evidence': 'invalid_json'}


def parse_ids(text, image_by_id):
    if text.strip().lower() == 'all':
        return sorted(image_by_id)
    return [int(x) for x in text.split(',') if x.strip()]


def iou(a, b):
    ax, ay, aw, ah = map(float, a)
    bx, by, bw, bh = map(float, b)
    inter = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0


def nms(items, thr):
    out = []
    for p in sorted(items, key=lambda x: x['score'], reverse=True):
        if all(iou(p['bbox'], q['bbox']) < thr for q in out):
            out.append(p)
    return out


def clamp_xyxy(box, width, height, coord_mode='pixel'):
    if not box or len(box) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    if coord_mode == 'norm1000' or (coord_mode == 'auto' and (max(x1, x2) > width or max(y1, y2) > height) and max(x1, y1, x2, y2) <= 1000):
        x1, x2 = x1 / 1000 * width, x2 / 1000 * width
        y1, y2 = y1 / 1000 * height, y2 / 1000 * height
    if x2 <= x1 or y2 <= y1:
        return None
    x1 = max(0, min(width - 1, x1))
    y1 = max(0, min(height - 1, y1))
    x2 = max(x1 + 1, min(width, x2))
    y2 = max(y1 + 1, min(height, y2))
    return [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]


def crop_candidate(img, bbox, context=0.25, max_side=768):
    W, H = img.size
    x, y, w, h = [float(v) for v in bbox]
    pad = max(8, max(w, h) * context)
    x1, y1 = max(0, int(x - pad)), max(0, int(y - pad))
    x2, y2 = min(W, int(x + w + pad)), min(H, int(y + h + pad))
    crop = img.crop((x1, y1, x2, y2)).convert('RGB')
    if max(crop.size) > max_side:
        s = max_side / float(max(crop.size))
        crop = crop.resize((int(crop.width * s), int(crop.height * s)), Image.Resampling.LANCZOS)
    return crop


def mask_refine_bbox(row):
    # Stage 2: use existing SAM mask if available; fallback to SAM bbox.
    path = row.get('mask_path')
    if not path or not Path(path).exists():
        return row['bbox']
    try:
        m = Image.open(path).convert('L')
        pix = m.load()
        xs, ys = [], []
        for y in range(m.height):
            for x in range(m.width):
                if pix[x, y] > 0:
                    xs.append(x)
                    ys.append(y)
        if not xs:
            return row['bbox']
        x1, x2, y1, y2 = min(xs), max(xs) + 1, min(ys), max(ys) + 1
        if x2 <= x1 or y2 <= y1:
            return row['bbox']
        return [round(float(x1), 2), round(float(y1), 2), round(float(x2 - x1), 2), round(float(y2 - y1), 2)]
    except Exception:
        return row['bbox']


def build_knowledge_prompt():
    names = '\n'.join(f'{cid}: {name}' for cid, name in CLASS_NAMES.items())
    return f"""
Output one valid JSON object. No explanation.
Build a zero-shot visual/OCR SKU knowledge base for these Orion packaged product classes:
{names}

For each class, provide:
- expected visible text tokens or romanized words
- dominant package colors
- shape/layout cues
- common confusion classes and how to separate them

Schema:
{{"classes":{{"1":{{"name":"Candy Boom","tokens":[],"colors":[],"visual_cues":[],"confusions":[]}}}}}}
""".strip()


def stage0_knowledge(raw_dir, api_key, model, timeout, retries, resume):
    raw_json = raw_dir / 'stage0_knowledge.json'
    if resume and raw_json.exists():
        return load_json(raw_json)
    prompt = build_knowledge_prompt()
    # Tiny blank image because the compatible API endpoint expects image_url for this script.
    img = Image.new('RGB', (32, 32), 'white')
    raw = call_api(img, prompt, api_key, model, timeout, retries)
    (raw_dir / 'stage0_knowledge_raw.txt').write_text(raw)
    obj = parse_json(raw)
    save_json(obj, raw_json)
    return obj


def load_sam_candidates(gt, ids, jsonl_paths, min_area, max_area, min_score, candidate_nms, topk):
    name_to_id = {im['file_name']: im['id'] for im in gt['images']}
    idset = set(ids)
    by = defaultdict(list)
    for p in jsonl_paths:
        if not p.exists():
            continue
        for line in open(p):
            if not line.strip():
                continue
            r = json.loads(line)
            iid = name_to_id.get(os.path.basename(r.get('image_path', '')))
            if iid not in idset:
                continue
            x1, y1, x2, y2 = [float(v) for v in r['box']]
            if x2 <= x1 or y2 <= y1:
                continue
            area = (x2 - x1) * (y2 - y1)
            score = float(r.get('score', 0.5))
            if area < min_area or area > max_area or score < min_score:
                continue
            row = {
                'image_id': iid,
                'bbox': [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                'score': score,
                'source': 'sam',
                'sam3_category_id': int(r.get('category_id') or 0),
                'mask_path': r.get('mask_path'),
            }
            row['bbox'] = mask_refine_bbox(row)
            by[iid].append(row)
    out = []
    for iid in ids:
        out.extend(nms(by[iid], candidate_nms)[:topk])
    return out


def direct_prompt(width, height):
    names = '\n'.join(f'{cid} {name}' for cid, name in CLASS_NAMES.items())
    return f"""
Output exactly one valid JSON object. No explanation.
High-recall product proposal on a retail shelf image. Image size {width}x{height}.
Find likely individual packaged products from these classes:
{names}

Return many candidate boxes. It is better to include uncertain products than to miss true products.
Do not merge adjacent products. Ignore whole shelf boxes.
Schema: {{"detections":[{{"bbox":[x1,y1,x2,y2],"confidence":0.70}}]}}
""".strip()


def stage1_qwen_direct(img, image_id, raw_dir, api_key, model, timeout, retries, resume, coord_mode):
    raw_json = raw_dir / f'stage1_direct_{image_id:04d}.json'
    if resume and raw_json.exists():
        obj = load_json(raw_json)
    else:
        raw = call_api(img, direct_prompt(img.width, img.height), api_key, model, timeout, retries)
        (raw_dir / f'stage1_direct_{image_id:04d}.txt').write_text(raw)
        obj = parse_json(raw)
        save_json(obj, raw_json)
    out = []
    for det in obj.get('detections', obj.get('items', [])):
        box = clamp_xyxy(det.get('bbox', det.get('bbox_2d')), img.width, img.height, coord_mode)
        if not box:
            continue
        out.append({'image_id': image_id, 'bbox': box, 'score': float(det.get('confidence', det.get('score', 0.5)) or 0.5), 'source': 'qwen_direct'})
    return out


def classify_prompt(knowledge):
    names = '\n'.join(f'{cid} {name}' for cid, name in CLASS_NAMES.items())
    kb = json.dumps(knowledge, ensure_ascii=False)[:3500]
    return f"""
Output exactly one valid JSON object. No explanation.
Classify ONE cropped candidate product package from an Orion retail shelf image.
The crop may include context around the red-box candidate, but judge the central product.

Official classes:
{names}

Zero-shot SKU knowledge:
{kb}

Tasks:
1. Read any visible text/OCR on the package.
2. Decide if the candidate is one of the official classes.
3. Return the best category_id and confidence. Use category_id=0 for background/unknown/wrong product.

Schema:
{{"category_id":6,"category_name":"OStar Yellow","confidence":0.88,"ocr_text":"", "evidence":"yellow OStar package"}}
""".strip()


def stage3_classify(img, cand, rank, knowledge, raw_dir, api_key, model, timeout, retries, resume, max_side, score_mode):
    stem = f'stage3_cls_{cand["image_id"]:04d}_{rank:04d}'
    raw_json = raw_dir / f'{stem}.json'
    crop = crop_candidate(img, cand['bbox'], context=0.35, max_side=max_side)
    if resume and raw_json.exists():
        obj = load_json(raw_json)
    else:
        prompt = classify_prompt(knowledge)
        crop.save(raw_dir / f'{stem}.jpg')
        raw = call_api(crop, prompt, api_key, model, timeout, retries)
        (raw_dir / f'{stem}.txt').write_text(raw)
        obj = parse_json(raw)
        save_json(obj, raw_json)
    try:
        cid = int(obj.get('category_id', 0) or 0)
    except Exception:
        cid = 0
    conf = float(obj.get('confidence', obj.get('score', 0.0)) or 0.0)
    if cid not in CLASS_NAMES:
        cid = 0
    sam_score = float(cand.get('score', 0.5))
    if score_mode == 'qconf':
        score = conf
    elif score_mode == 'samboost':
        score = 0.55 * conf + 0.45 * sam_score
    else:
        score = 0.75 * conf + 0.25 * sam_score
    return {
        'image_id': cand['image_id'],
        'category_id': cid,
        'bbox': cand['bbox'],
        'score': round(max(0.001, min(0.999, score)), 6),
        'qwen_confidence': conf,
        'source': cand.get('source', 'candidate'),
        'ocr_text': obj.get('ocr_text', ''),
        'evidence': obj.get('evidence', ''),
    }


def review_prompt(items):
    names = '\n'.join(f'{cid} {name}' for cid, name in CLASS_NAMES.items())
    ids = [it['index'] for it in items]
    return f"""
Output exactly one valid JSON object. No explanation.
You see a montage of uncertain product crop candidates. Each crop has an ID.
Recheck only these official classes:
{names}

For every ID, decide keep/category/confidence. Use keep=false for background, unknown products, duplicates, or unclear crops.
IDs: {ids}
Schema: {{"items":[{{"index":0,"keep":true,"category_id":6,"confidence":0.82}},{{"index":1,"keep":false,"confidence":0.20}}]}}
""".strip()


def make_montage(crops, cell=180):
    cols = min(5, max(1, len(crops)))
    rows = (len(crops) + cols - 1) // cols
    canvas = Image.new('RGB', (cols * cell, rows * cell), 'white')
    draw = ImageDraw.Draw(canvas)
    for i, (idx, crop) in enumerate(crops):
        r, c = divmod(i, cols)
        tile = crop.copy()
        tile.thumbnail((cell - 8, cell - 24), Image.Resampling.LANCZOS)
        x, y = c * cell + 4, r * cell + 20
        canvas.paste(tile, (x, y))
        draw.rectangle((c * cell, r * cell, (c + 1) * cell - 1, (r + 1) * cell - 1), outline=(0, 0, 0), width=1)
        draw.text((c * cell + 4, r * cell + 3), str(idx), fill=(255, 0, 0))
    return canvas


def stage4_review(img, preds, raw_dir, api_key, model, timeout, retries, resume, max_side, low, high):
    uncertain = [(i, p) for i, p in enumerate(preds) if p['category_id'] in CLASS_NAMES and low <= p['qwen_confidence'] < high]
    if not uncertain:
        return preds
    reviewed = {i: p for i, p in enumerate(preds)}
    for start in range(0, len(uncertain), 15):
        batch = uncertain[start:start + 15]
        stem = f'stage4_review_{preds[0]["image_id"]:04d}_{start:03d}'
        raw_json = raw_dir / f'{stem}.json'
        crops = [(idx, crop_candidate(img, pred['bbox'], 0.35, max_side)) for idx, pred in batch]
        if resume and raw_json.exists():
            obj = load_json(raw_json)
        else:
            montage = make_montage(crops)
            montage.save(raw_dir / f'{stem}.jpg')
            raw = call_api(montage, review_prompt([{'index': idx} for idx, _ in batch]), api_key, model, timeout, retries)
            (raw_dir / f'{stem}.txt').write_text(raw)
            obj = parse_json(raw)
            save_json(obj, raw_json)
        for item in obj.get('items', obj.get('detections', [])):
            try:
                idx = int(item.get('index', -1))
                cid = int(item.get('category_id', 0) or 0)
            except Exception:
                continue
            if idx not in reviewed:
                continue
            if not item.get('keep', True) or cid not in CLASS_NAMES:
                reviewed[idx]['category_id'] = 0
                continue
            conf = float(item.get('confidence', item.get('score', reviewed[idx]['score'])) or reviewed[idx]['score'])
            reviewed[idx]['category_id'] = cid
            reviewed[idx]['score'] = round(max(0.001, min(0.999, conf)), 6)
    return [reviewed[i] for i in range(len(preds))]


def shelf_prompt(items):
    names = '\n'.join(f'{cid} {name}' for cid, name in CLASS_NAMES.items())
    lines = []
    for p in items:
        x, y, w, h = [round(float(v), 1) for v in p['bbox']]
        lines.append(f"{p['index']}: class={p['category_id']} {CLASS_NAMES.get(p['category_id'])}, score={p['score']}, xywh={[x,y,w,h]}")
    return f"""
Output exactly one valid JSON object. No explanation.
Global shelf consistency check for Orion SKU detections.
Official classes:
{names}

Rules:
- Keep repeated products, but remove duplicate boxes for the same visible instance.
- Adjacent products often form rows/columns; class labels should be visually plausible.
- Do not force a row to one class if packages are visibly different.

Candidates:
{chr(10).join(lines)}

Return changed/removed candidates only:
{{"updates":[{{"index":3,"keep":true,"category_id":6,"confidence":0.81}},{{"index":7,"keep":false}}]}}
""".strip()


def draw_index_panel(img, preds, max_side=1400):
    panel = img.copy().convert('RGB')
    draw = ImageDraw.Draw(panel)
    for i, p in enumerate(preds):
        x, y, w, h = p['bbox']
        draw.rectangle((x, y, x + w, y + h), outline=(255, 0, 0), width=2)
        draw.rectangle((x, max(0, y - 16), x + 34, max(16, y)), fill=(255, 255, 0))
        draw.text((x + 2, max(0, y - 15)), str(i), fill=(0, 0, 0))
    if max(panel.size) > max_side:
        s = max_side / float(max(panel.size))
        panel = panel.resize((int(panel.width * s), int(panel.height * s)), Image.Resampling.LANCZOS)
    return panel


def stage5_shelf_rerank(img, preds, image_id, raw_dir, api_key, model, timeout, retries, resume):
    preds = [dict(p, index=i) for i, p in enumerate(preds) if p['category_id'] in CLASS_NAMES]
    if not preds:
        return []
    raw_json = raw_dir / f'stage5_shelf_{image_id:04d}.json'
    if resume and raw_json.exists():
        obj = load_json(raw_json)
    else:
        panel = draw_index_panel(img, preds)
        panel.save(raw_dir / f'stage5_shelf_{image_id:04d}.jpg')
        raw = call_api(panel, shelf_prompt(preds), api_key, model, timeout, retries)
        (raw_dir / f'stage5_shelf_{image_id:04d}.txt').write_text(raw)
        obj = parse_json(raw)
        save_json(obj, raw_json)
    by_idx = {p['index']: p for p in preds}
    for item in obj.get('updates', obj.get('items', [])):
        try:
            idx = int(item.get('index', -1))
        except Exception:
            continue
        if idx not in by_idx:
            continue
        if not item.get('keep', True):
            by_idx[idx]['category_id'] = 0
            continue
        cid = int(item.get('category_id', by_idx[idx]['category_id']) or by_idx[idx]['category_id'])
        if cid in CLASS_NAMES:
            by_idx[idx]['category_id'] = cid
        if 'confidence' in item:
            by_idx[idx]['score'] = round(max(0.001, min(0.999, float(item['confidence']))), 6)
    out = []
    for p in by_idx.values():
        p.pop('index', None)
        if p['category_id'] in CLASS_NAMES:
            out.append(p)
    return out


def final_fuse(preds, score_thr, nms_thr, max_per_image):
    by = defaultdict(list)
    for p in preds:
        if p['category_id'] not in CLASS_NAMES or p['score'] < score_thr:
            continue
        by[(p['image_id'], p['category_id'])].append(p)
    by_image = defaultdict(list)
    for items in by.values():
        by_image[items[0]['image_id']].extend(nms(items, nms_thr))
    out = []
    for image_id, items in by_image.items():
        out.extend(sorted(items, key=lambda x: x['score'], reverse=True)[:max_per_image])
    return [{k: v for k, v in p.items() if k in ('image_id', 'category_id', 'bbox', 'score')} for p in out]


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
    ap.add_argument('--api-key', default=os.environ.get('DASHSCOPE_API_KEY') or os.environ.get('QWEN_API_KEY'))
    ap.add_argument('--model', default='qwen3.5-plus')
    ap.add_argument('--gt', default=str(GT))
    ap.add_argument('--image-ids', default='0,1,2,3,4')
    ap.add_argument('--sam3-jsonl', nargs='*', default=[str(p) for p in sorted(ROOT.glob('predictions_gpu*.jsonl'))])
    ap.add_argument('--out', default=str(ROOT / 'qwen_orion_sku_pipeline_test.json'))
    ap.add_argument('--raw-dir', default=str(ROOT / 'qwen_orion_sku_pipeline_raw'))
    ap.add_argument('--use-qwen-direct', action='store_true')
    ap.add_argument('--use-review', action='store_true')
    ap.add_argument('--use-shelf-rerank', action='store_true')
    ap.add_argument('--topk-per-image', type=int, default=80)
    ap.add_argument('--min-area', type=float, default=80)
    ap.add_argument('--max-area', type=float, default=200000)
    ap.add_argument('--sam-min-score', type=float, default=0.0)
    ap.add_argument('--candidate-nms', type=float, default=0.45)
    ap.add_argument('--direct-coord-mode', default='auto', choices=['pixel', 'norm1000', 'auto'])
    ap.add_argument('--direct-topk-per-image', type=int, default=30)
    ap.add_argument('--crop-max-side', type=int, default=768)
    ap.add_argument('--stage3-score-mode', default='blend', choices=['blend', 'qconf', 'samboost'])
    ap.add_argument('--final-score-thr', type=float, default=0.05)
    ap.add_argument('--final-nms', type=float, default=0.30)
    ap.add_argument('--max-per-image', type=int, default=120)
    ap.add_argument('--timeout', type=int, default=150)
    ap.add_argument('--retries', type=int, default=1)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--evaluate', action='store_true')
    args = ap.parse_args()
    if not args.api_key:
        raise SystemExit('set DASHSCOPE_API_KEY/QWEN_API_KEY')

    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    gt = load_json(args.gt)
    image_by_id = {im['id']: im for im in gt['images']}
    ids = parse_ids(args.image_ids, image_by_id)
    image_dir = Path(args.gt).parent

    knowledge = stage0_knowledge(raw_dir, args.api_key, args.model, args.timeout, args.retries, args.resume)
    sam_cands = load_sam_candidates(
        gt, ids, [Path(p) for p in args.sam3_jsonl],
        args.min_area, args.max_area, args.sam_min_score,
        args.candidate_nms, args.topk_per_image,
    )
    save_json(sam_cands, raw_dir / 'stage1_sam_candidates.json')
    by_image = defaultdict(list)
    for c in sam_cands:
        by_image[c['image_id']].append(c)

    all_preds = []
    for image_id in ids:
        meta = image_by_id[image_id]
        img = Image.open(image_dir / meta['file_name']).convert('RGB')
        cands = list(by_image.get(image_id, []))
        if args.use_qwen_direct:
            direct_cands = stage1_qwen_direct(img, image_id, raw_dir, args.api_key, args.model, args.timeout, args.retries, args.resume, args.direct_coord_mode)
            direct_cands = nms(direct_cands, args.candidate_nms)[:args.direct_topk_per_image]
            # Do not let direct proposals evict SAM candidates before crop classification.
            cands = cands[:args.topk_per_image] + direct_cands
        print(f'image_id={image_id} candidates={len(cands)}', flush=True)

        preds = []
        for rank, cand in enumerate(cands):
            print(f'  stage3 classify {rank + 1}/{len(cands)}', flush=True)
            preds.append(stage3_classify(img, cand, rank, knowledge, raw_dir, args.api_key, args.model, args.timeout, args.retries, args.resume, args.crop_max_side, args.stage3_score_mode))
        if args.use_review:
            preds = stage4_review(img, preds, raw_dir, args.api_key, args.model, args.timeout, args.retries, args.resume, args.crop_max_side, 0.35, 0.70)
        if args.use_shelf_rerank:
            preds = stage5_shelf_rerank(img, preds, image_id, raw_dir, args.api_key, args.model, args.timeout, args.retries, args.resume)
        all_preds.extend(preds)

    final_preds = final_fuse(all_preds, args.final_score_thr, args.final_nms, args.max_per_image)
    save_json(final_preds, args.out)
    print(f'wrote {len(final_preds)} predictions to {args.out}')
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
