#!/usr/bin/env python3
"""Candidate-level Gemini API classifier/reboxer for dental SAM3 proposals.

This pipeline treats SAM3 boxes as localization proposals and asks Gemini only to
classify, reject, adjust, or split each candidate. It mirrors the saved Qwen and
Doubao flows so the output can be fused by the same SAM3 snap-fuse stage.

Small run from existing SAM3 candidates:
  cd /data/LPP/cvpr/few_shot/sam3-main/best_sam3/dentalai-i4clz-fsod-fsuo
  GEMINI_API_KEY='<key>' python gemini_dental_sam3_candidate_api.py \
    --image-ids 0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
    --topk-per-image 35 \
    --out gemini_dental_sam3cand_20img_v1.json \
    --raw-dir gemini_dental_sam3cand_20img_v1_raw \
    --resume --evaluate
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path('/data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data/dentalai-i4clz-fsod-fsuo')
GT = DATA_ROOT / 'test' / '_annotations.coco.json'
CLASS_NAMES = {1: 'Cavity', 2: 'Fillings', 3: 'Impacted Tooth', 4: 'Implant'}

RESPONSE_SCHEMA = {
    'type': 'object',
    'properties': {
        'detections': {
            'type': 'array',
            'items': {
                'type': 'object',
                'properties': {
                    'category_id': {'type': 'integer'},
                    'category_name': {'type': 'string'},
                    'bbox': {
                        'type': 'array',
                        'items': {'type': 'number'},
                        'minItems': 4,
                        'maxItems': 4,
                    },
                    'use_original_box': {'type': 'boolean'},
                    'confidence': {'type': 'number'},
                    'reason': {'type': 'string'},
                },
                'required': ['category_id', 'bbox', 'confidence'],
            },
        },
    },
    'required': ['detections'],
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


def iou_box(a, b):
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    inter = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0.0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def nms(items, thr):
    keep = []
    for item in sorted(items, key=lambda x: x['score'], reverse=True):
        if all(iou_box(item['bbox'], old['bbox']) < thr for old in keep):
            keep.append(item)
    return keep


def parse_ids(text, image_by_id):
    if text.strip().lower() == 'all':
        return sorted(image_by_id)
    return [int(x) for x in text.split(',') if x.strip()]


def load_sam3_candidates(gt, jsonl_paths, ids, min_area, max_area, min_score, nms_thr, topk):
    name_to_id = {im['file_name']: im['id'] for im in gt['images']}
    idset = set(ids)
    by = defaultdict(list)
    for path in jsonl_paths:
        if not path.exists():
            continue
        with open(path) as f:
            for line in f:
                if not line.strip():
                    continue
                row = json.loads(line)
                image_id = name_to_id.get(os.path.basename(row.get('image_path', '')))
                if image_id not in idset:
                    continue
                x1, y1, x2, y2 = [float(v) for v in row['box']]
                if x2 <= x1 or y2 <= y1:
                    continue
                area = (x2 - x1) * (y2 - y1)
                score = float(row.get('score', 0.5))
                if area < min_area or area > max_area or score < min_score:
                    continue
                by[image_id].append({
                    'image_id': image_id,
                    'bbox': [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)],
                    'score': score,
                    'sam3_prompt_category_id': int(row.get('category_id', 0) or 0),
                })
    out = []
    for image_id in ids:
        out.extend(nms(by.get(image_id, []), nms_thr)[:topk])
    return out


def make_crop(full, bbox, context, max_side):
    W, H = full.size
    x, y, w, h = [float(v) for v in bbox]
    pad = max(12.0, max(w, h) * context)
    x1 = max(0, int(round(x - pad)))
    y1 = max(0, int(round(y - pad)))
    x2 = min(W, int(round(x + w + pad)))
    y2 = min(H, int(round(y + h + pad)))
    crop = full.crop((x1, y1, x2, y2)).convert('RGB')
    scale = 1.0
    if max(crop.size) > max_side:
        scale = max_side / float(max(crop.size))
        crop = crop.resize((int(round(crop.width * scale)), int(round(crop.height * scale))), Image.Resampling.LANCZOS)
    red = [(x - x1) * scale, (y - y1) * scale, (x + w - x1) * scale, (y + h - y1) * scale]
    draw = ImageDraw.Draw(crop)
    for t in range(3):
        draw.rectangle((red[0] - t, red[1] - t, red[2] + t, red[3] + t), outline=(255, 0, 0))
    return crop, (x1, y1, scale), red


def make_prompt(cw, ch, red_box, candidate_box):
    return f"""
Do not think step by step. Do not explain. Output exactly one valid JSON object and nothing else.
You are classifying and calibrating one SAM3 candidate on a panoramic dental X-ray crop.
A red rectangle marks the SAM3 candidate. Crop size: width={cw}, height={ch}.
Return bbox coordinates as normalized integers in 0-1000 relative to this crop.

Official classes:
1 Cavity: local dark carious/radiolucent lesion on a tooth. Usually a small dark defect, not a normal gap/background.
2 Fillings: restoration on/in natural tooth crown. Usually bright white patch/crown/amalgam/composite, often tooth-sized vertical rectangles.
3 Impacted Tooth: unerupted/sideways/abnormally positioned tooth, often posterior/wisdom tooth. Box the whole impacted tooth, not normal erupted teeth.
4 Implant: artificial screw/post/abutment/implant-supported crown. Metallic root/post/screw shape, often narrow/tall and very bright.

Task, optimized for COCO mAP:
- Judge the RED SAM3 box itself, not just whether a target is somewhere in the crop.
- Keep only if the red box would overlap one official annotation by IoU >= 0.50.
- Reject partial boxes that cover only the brightest core of a filling, only the crown edge, only background, or only part of an implant/cavity.
- Reject duplicate/neighbor boxes when the red box is clearly shifted onto the adjacent normal tooth or jaw.
- Correct the class if SAM3 prompt label was wrong.
- Prefer the original red/SAM3 box as the final localization; evaluation boxes are instance-level and often include the full restoration/tooth region, not just the brightest pixels.
- Only adjust the bbox when the red box clearly includes large background or misses a visible part of the target.
- If unsure whether red box IoU would reach 0.50, return an empty detections list.
- Use varied confidence: 0.95 for very likely IoU>=0.75, 0.80 for likely IoU>=0.50, below 0.65 for borderline.

Original red candidate box in crop xyxy pixels: {[round(v, 1) for v in red_box]}.
Original SAM3 full-image xywh box: {[round(float(v), 2) for v in candidate_box]}.

Return schema:
{{"detections":[{{"category_id":2,"category_name":"Fillings","bbox":[x1_norm,y1_norm,x2_norm,y2_norm],"use_original_box":true,"confidence":0.92,"reason":"bright restoration on natural tooth crown"}}]}}
If no valid target: {{"detections":[]}}
""".strip()


def normalize_api_key(api_key):
    if not api_key:
        return api_key
    return str(api_key).strip().replace('\ufeff', '')


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
    api_key = normalize_api_key(api_key)
    payload = {
        'contents': [{
            'role': 'user',
            'parts': [
                {
                    'inline_data': {
                        'mime_type': 'image/jpeg',
                        'data': image_b64(img),
                    },
                },
                {'text': prompt},
            ],
        }],
        'generationConfig': {
            'temperature': 0,
            'responseMimeType': 'application/json',
            'responseJsonSchema': RESPONSE_SCHEMA,
        },
    }
    data = json.dumps(payload).encode('utf-8')
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
            last = RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}")
        except Exception as e:
            last = e
        time.sleep(2 + attempt * 2)
    raise RuntimeError(last)


def parse_json(text):
    text = text.strip().replace('```json', '').replace('```', '').strip()
    text = re.sub(r'</?think[^>]*>', ' ', text)
    candidates = [text]
    match = re.search(r'\{.*\}', text, flags=re.S)
    if match:
        candidates.append(match.group(0))
    last = None
    for cand in candidates:
        cleaned = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f]', ' ', cand)
        for variant in [cand, cleaned]:
            try:
                return json.loads(variant, strict=False)
            except Exception as exc:
                last = exc
    if '[]' in text or 'no valid' in text.lower() or 'empty detections' in text.lower():
        return {'detections': []}
    return {'detections': [], 'parse_error': str(last) if last else 'unparseable_gemini_response'}


def crop_norm_xyxy_to_full_xywh(box, origin_scale, full_size, crop_size, coord_mode):
    if not box or len(box) != 4:
        return None
    ox, oy, scale = origin_scale
    W, H = full_size
    cw, ch = crop_size
    x1, y1, x2, y2 = [float(v) for v in box]
    if coord_mode == 'norm1000' or (coord_mode == 'auto' and max(x1, y1, x2, y2) <= 1000):
        x1 = x1 / 1000.0 * cw
        x2 = x2 / 1000.0 * cw
        y1 = y1 / 1000.0 * ch
        y2 = y2 / 1000.0 * ch
    if x2 <= x1 or y2 <= y1:
        return None
    x1 = ox + x1 / scale
    y1 = oy + y1 / scale
    x2 = ox + x2 / scale
    y2 = oy + y2 / scale
    x1 = max(0.0, min(W - 1.0, x1))
    y1 = max(0.0, min(H - 1.0, y1))
    x2 = max(x1 + 1.0, min(float(W), x2))
    y2 = max(y1 + 1.0, min(float(H), y2))
    return [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]


def accept_box(candidate_box, new_box, max_shift, min_inter_candidate):
    cx = candidate_box[0] + candidate_box[2] / 2.0
    cy = candidate_box[1] + candidate_box[3] / 2.0
    nx = new_box[0] + new_box[2] / 2.0
    ny = new_box[1] + new_box[3] / 2.0
    diag = max(1.0, (candidate_box[2] ** 2 + candidate_box[3] ** 2) ** 0.5)
    shift = ((cx - nx) ** 2 + (cy - ny) ** 2) ** 0.5 / diag
    inter = iou_box(candidate_box, new_box)
    return shift <= max_shift and inter >= min_inter_candidate


def post_filter(preds, nms_thr):
    out = []
    min_area = {1: 90, 2: 90, 3: 180, 4: 120}
    max_area = {1: 4200, 2: 5200, 3: 6500, 4: 5600}
    max_ratio = {1: 6.5, 2: 6.5, 3: 4.5, 4: 7.0}
    for p in preds:
        x, y, w, h = [float(v) for v in p['bbox']]
        cid = int(p['category_id'])
        area = w * h
        ratio = max(w / max(h, 1.0), h / max(w, 1.0))
        if area < min_area[cid] or area > max_area[cid] or ratio > max_ratio[cid]:
            continue
        out.append(p)
    grouped = defaultdict(list)
    for p in out:
        grouped[(p['image_id'], p['category_id'])].append(p)
    keep = []
    for items in grouped.values():
        keep.extend(nms(items, nms_thr))
    return keep


def choose_output_box(cand_box, api_box, source):
    if source == 'candidate' or not api_box:
        return [float(v) for v in cand_box]
    if source == 'api':
        return api_box
    cand_area = max(1.0, float(cand_box[2]) * float(cand_box[3]))
    api_area = max(1.0, float(api_box[2]) * float(api_box[3]))
    overlap = iou_box(cand_box, api_box)
    if overlap >= 0.25 and api_area < cand_area * 0.65:
        return [float(v) for v in cand_box]
    return api_box


def evaluate(gt_path, pred_path):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = COCO(str(gt_path))
    dt = gt.loadRes(str(pred_path))
    ev = COCOeval(gt, dt, 'bbox')
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return {'mAP': float(ev.stats[0]), 'mAP50': float(ev.stats[1]), 'mAP75': float(ev.stats[2]), 'stats': [float(x) for x in ev.stats]}


def load_or_parse_raw(raw_json_path: Path, raw_txt_path: Path):
    if raw_json_path.exists():
        return load_json(raw_json_path)
    if raw_txt_path.exists():
        raw = raw_txt_path.read_text(encoding='utf-8', errors='replace')
        obj = parse_json(raw)
        raw_json_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
        return obj
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--api-key', default=os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY'))
    ap.add_argument('--model', default='gemini-2.5-flash')
    ap.add_argument('--gt', default=str(GT))
    ap.add_argument('--image-ids', default='0,1,2,3,4')
    ap.add_argument('--sam3-jsonl', nargs='*', default=[str(p) for p in sorted(ROOT.glob('predictions_gpu*.jsonl'))])
    ap.add_argument('--out', default=str(ROOT / 'gemini_dental_sam3cand_test.json'))
    ap.add_argument('--raw-dir', default=str(ROOT / 'gemini_dental_sam3cand_raw'))
    ap.add_argument('--min-area', type=float, default=80.0)
    ap.add_argument('--max-area', type=float, default=5000.0)
    ap.add_argument('--min-score', type=float, default=0.0)
    ap.add_argument('--candidate-nms', type=float, default=0.65)
    ap.add_argument('--topk-per-image', type=int, default=35)
    ap.add_argument('--context', type=float, default=1.15)
    ap.add_argument('--max-side', type=int, default=720)
    ap.add_argument('--coord-mode', default='norm1000', choices=['norm1000', 'pixel', 'auto'])
    ap.add_argument('--accept-shift', type=float, default=0.85)
    ap.add_argument('--accept-iou', type=float, default=0.03)
    ap.add_argument('--box-source', default='hybrid', choices=['hybrid', 'candidate', 'api'])
    ap.add_argument('--sam3-score-weight', type=float, default=0.25)
    ap.add_argument('--score-scale', type=float, default=1.0)
    ap.add_argument('--final-nms', type=float, default=0.45)
    ap.add_argument('--timeout', type=int, default=120)
    ap.add_argument('--retries', type=int, default=1)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--api-only-cached', action='store_true')
    ap.add_argument('--evaluate', action='store_true')
    ap.add_argument('--sleep', type=float, default=0.0)
    args = ap.parse_args()
    if not args.api_key and not args.api_only_cached:
        raise SystemExit('set GEMINI_API_KEY/GOOGLE_API_KEY, pass --api-key, or use --api-only-cached')

    gt = load_json(args.gt)
    image_by_id = {im['id']: im for im in gt['images']}
    ids = parse_ids(args.image_ids, image_by_id)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    image_dir = Path(args.gt).parent
    candidates = load_sam3_candidates(gt, [Path(p) for p in args.sam3_jsonl], ids, args.min_area, args.max_area, args.min_score, args.candidate_nms, args.topk_per_image)
    save_json(candidates, raw_dir / 'selected_candidates.json')
    print(f'selected_candidates={len(candidates)} images={len(ids)}')

    preds = []
    cand_by_image = defaultdict(list)
    for cand in candidates:
        cand_by_image[cand['image_id']].append(cand)
    total = len(candidates)
    done = 0
    for image_id in ids:
        meta = image_by_id[image_id]
        img = Image.open(image_dir / meta['file_name']).convert('RGB')
        for rank, cand in enumerate(cand_by_image.get(image_id, [])):
            done += 1
            crop, origin_scale, red = make_crop(img, cand['bbox'], args.context, args.max_side)
            prompt = make_prompt(crop.width, crop.height, red, cand['bbox'])
            stem = f'{image_id:04d}_{rank:03d}'
            raw_json_path = raw_dir / f'{stem}_raw.json'
            raw_txt_path = raw_dir / f'{stem}_raw.txt'
            obj = load_or_parse_raw(raw_json_path, raw_txt_path) if args.resume else None
            if obj is not None:
                pass
            elif args.api_only_cached:
                continue
            else:
                (raw_dir / f'{stem}_prompt.txt').write_text(prompt, encoding='utf-8')
                if rank < 3:
                    crop.save(raw_dir / f'{stem}_crop.jpg')
                print(f'[{done}/{total}] API classify image_id={image_id} cand={rank} file={meta["file_name"]}', flush=True)
                raw = call_api(crop, prompt, args.api_key, args.model, args.timeout, args.retries)
                raw_txt_path.write_text(raw, encoding='utf-8')
                obj = parse_json(raw)
                raw_json_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
                if args.sleep > 0:
                    time.sleep(args.sleep)
            for det in obj.get('detections', obj.get('results', [])):
                try:
                    cid = int(det.get('category_id', 0) or 0)
                except (TypeError, ValueError):
                    continue
                if cid not in CLASS_NAMES:
                    continue
                box = crop_norm_xyxy_to_full_xywh(det.get('bbox', []), origin_scale, img.size, crop.size, args.coord_mode)
                box = choose_output_box(cand['bbox'], box, args.box_source)
                if not box:
                    continue
                if not accept_box(cand['bbox'], box, args.accept_shift, args.accept_iou):
                    continue
                api_score = float(det.get('confidence', det.get('score', 0.82)) or 0.82)
                sam3_score = float(cand.get('score', 0.5))
                score = ((1.0 - args.sam3_score_weight) * api_score + args.sam3_score_weight * sam3_score) * args.score_scale
                preds.append({'image_id': image_id, 'category_id': cid, 'bbox': box, 'score': round(max(0.001, min(0.999, score)), 6)})
        print(f'image_id={image_id} raw_preds_so_far={len(preds)}', flush=True)

    preds = post_filter(preds, args.final_nms)
    save_json(preds, args.out)
    print(f'wrote {len(preds)} predictions to {args.out}')
    if args.evaluate:
        eval_gt = Path(args.gt)
        if args.image_ids.strip().lower() != 'all':
            idset = set(ids)
            sub = dict(gt)
            sub['images'] = [im for im in gt['images'] if im['id'] in idset]
            sub['annotations'] = [ann for ann in gt['annotations'] if ann['image_id'] in idset]
            eval_gt = Path(str(Path(args.out).with_suffix('')) + '_subset_gt.json')
            save_json(sub, eval_gt)
        metrics = evaluate(eval_gt, Path(args.out))
        eval_path = str(Path(args.out).with_suffix('')) + '_eval.json'
        save_json(metrics, eval_path)
        print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
