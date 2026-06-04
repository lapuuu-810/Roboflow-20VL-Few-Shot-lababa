#!/usr/bin/env python3
"""Qwen API pipeline for defect-detection-yjplx-fxobh-fsod-amdi.

The dataset labels both bad and good rail joint parts:
  1 defective fishplate
  2 fastener
  3 missing fastener
  4 non defective fishplate

Typical full run:
  DASHSCOPE_API_KEY='<your_api_key>' python qwen_defect_detection_api_pipeline.py \
    --image-ids all \
    --out qwen_defect_api_full_v1.json \
    --raw-dir qwen_defect_api_full_v1_raw \
    --coord-mode norm1000 \
    --evaluate
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
DATA_ROOT = Path('/data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data/defect-detection-yjplx-fxobh-fsod-amdi')
GT = DATA_ROOT / 'test' / '_annotations.coco.json'
MASK_DIR = ROOT / 'masks'

CLASS_NAMES = {
    1: 'defective fishplate',
    2: 'fastener',
    3: 'missing fastener',
    4: 'non defective fishplate',
}


def load_json(path: str | Path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path: str | Path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def image_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=92)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def make_prompt(width: int, height: int) -> str:
    return f"""
Do not think step by step. Do not explain. Output exactly one valid JSON object and nothing else.
You are doing object detection for railway fishplate and fastener defect inspection.
Image size: width={width}, height={height}. Return coordinates as normalized integers in the 0-1000 range: x_norm=x/width*1000, y_norm=y/height*1000.

Detect only these official categories:
1 defective fishplate: a bad/damaged rail fishplate or joint bar. Use this only when the fishplate is visibly broken, cracked, bent, severely damaged, missing a structural part, or clearly defective. Do not label an intact fishplate as defective.
2 fastener: a present rail fastening component that holds the rail to the sleeper/tie, such as a rail clip, clamp, anchor, screw spike assembly, washer/bolt head, or installed fastener seat. Box each visible installed rail fastener tightly. Do not label small bolt-like marks on the fishplate body unless they are clearly annotated rail fasteners.
3 missing fastener: an expected rail fastening/clip position where the physical fastener is absent. This is usually an empty clip seat, vacant clamp/anchor position, empty fastening hole/seat, or aligned missing location beside the rail/sleeper. Box the missing local position tightly, not the whole sleeper or fishplate.
4 non defective fishplate: an intact/good fishplate or joint bar with no visible defect. Box the whole visible fishplate/joint bar, not the rail, sleeper, or individual fasteners.

Critical visual distinctions and required inspection procedure:
- First identify the rail fastening pattern along the rails and sleepers/ties. There may be an upper rail side and a lower rail side in the same image, or left/right symmetric fastener positions. Inspect all repeated rail-clip/clamp positions, including edge positions.
- For every expected rail-fastening position, choose exactly one local label: category 2 if a physical rail clip/clamp/anchor/fastener is installed; category 3 if that expected position is empty, recessed, flat, dark, or only the fastener seat/hole remains.
- Missing fastener examples: empty rail clip seat, vacant clamp/anchor position, empty hole/seat on the sleeper near the rail, dark oval/circular depression with no raised installed component, or an aligned missing position beside neighboring present rail fasteners.
- Present fastener examples: installed rail clip, clamp, anchor, screw spike/bolt head/washer assembly, or physically raised fastening component near the rail/sleeper. These are category 2.
- Fishplate/joint bar classes are separate: category 1 or 4 should box the whole fishplate only. Do not convert individual fishplate bolt holes into category 2/3 unless they are part of the rail-fastener annotations.
- Bad part vs good part: category 1 is only visibly defective fishplate; category 4 is intact fishplate. If a fishplate/joint bar is cracked, broken, heavily bent, structurally damaged, or visually bad, use category 1 instead of category 4.
- Do not output both category 1 and category 4 for the same fishplate.
- Do not label rail heads, sleepers, ballast stones, shadows, background holes, text, or random marks.
- Use tight boxes. Large fishplate boxes are allowed only for categories 1 and 4. Fastener and missing-fastener boxes should be small local boxes around each position.
- Detect all visible target instances, including partially visible edge fasteners or edge missing positions. Avoid duplicate boxes for the same object.

Return schema:
{{"detections":[{{"category_id":3,"category_name":"missing fastener","bbox":[x1_norm,y1_norm,x2_norm,y2_norm],"confidence":0.94,"reason":"empty bolt hole in aligned fastening position"}}]}}
All bbox coordinates must be integers in [0,1000].
""".strip()


def call_api(img: Image.Image, prompt: str, api_key: str, model: str, timeout: int, retries: int) -> str:
    payload = {
        'model': model,
        'messages': [
            {
                'role': 'user',
                'content': [
                    {'type': 'text', 'text': prompt},
                    {'type': 'image_url', 'image_url': {'url': image_url(img)}},
                ],
            }
        ],
        'temperature': 0,
        'response_format': {'type': 'json_object'},
        'enable_thinking': False,
        'thinking': {'type': 'disabled'},
    }
    data = json.dumps(payload).encode('utf-8')
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
            last = RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}")
        except Exception as e:
            last = e
        time.sleep(2 + attempt * 2)
    raise RuntimeError(last)


def parse_json(text: str):
    text = text.strip().replace('```json', '').replace('```', '').strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r'\{.*\}', text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def clamp_box_xyxy_to_xywh(box, width: int, height: int, coord_mode: str):
    if not box or len(box) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    if coord_mode == 'norm1000' or (coord_mode == 'auto' and max(x1, y1, x2, y2) <= 1000):
        x1 = x1 / 1000.0 * width
        x2 = x2 / 1000.0 * width
        y1 = y1 / 1000.0 * height
        y2 = y2 / 1000.0 * height
    if x2 <= x1 or y2 <= y1:
        return None
    x1 = max(0.0, min(width - 1.0, x1))
    y1 = max(0.0, min(height - 1.0, y1))
    x2 = max(x1 + 1.0, min(float(width), x2))
    y2 = max(y1 + 1.0, min(float(height), y2))
    return [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]


def box_area(box) -> float:
    return max(0.0, float(box[2]) * float(box[3]))


def iou(a, b) -> float:
    ax, ay, aw, ah = [float(v) for v in a['bbox']]
    bx, by, bw, bh = [float(v) for v in b['bbox']]
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    inter = max(0.0, min(ax2, bx2) - max(ax, bx)) * max(0.0, min(ay2, by2) - max(ay, by))
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def nms(preds, thr: float = 0.45):
    out = []
    for cid in sorted(set(p['category_id'] for p in preds)):
        keep = []
        items = sorted([p for p in preds if p['category_id'] == cid], key=lambda x: x['score'], reverse=True)
        for pred in items:
            if all(iou(pred, old) < thr for old in keep):
                keep.append(pred)
        out.extend(keep)
    return out


def post_filter(preds, width: int, height: int):
    image_area = float(width * height)
    out = []
    for pred in preds:
        cid = pred['category_id']
        area = box_area(pred['bbox'])
        frac = area / image_area
        x, y, w, h = pred['bbox']
        ratio = max(w / max(h, 1.0), h / max(w, 1.0))
        if cid in (2, 3):
            if area < 600 or frac > 0.08 or ratio > 8.0:
                continue
        if cid in (1, 4):
            if frac < 0.005 or frac > 0.75:
                continue
        out.append(pred)
    return nms(out, 0.42)


def draw_vis(img_path: Path, dets, out_path: Path):
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)
    colors = {
        1: (255, 0, 0),
        2: (0, 220, 0),
        3: (255, 150, 0),
        4: (0, 150, 255),
    }
    for pred in dets:
        x, y, w, h = pred['bbox']
        col = colors.get(pred['category_id'], (255, 0, 255))
        draw.rectangle((x, y, x + w, y + h), outline=col, width=4)
        label = f"{pred['category_id']}:{CLASS_NAMES[pred['category_id']]} {pred['score']:.2f}"
        draw.text((x, max(0, y - 14)), label, fill=col)
    img.save(out_path)


def convert_masks_to_predictions(gt_path: Path, mask_dir: Path, min_area: float, score: float):
    import numpy as np

    gt = load_json(gt_path)
    stem_to_image = {Path(im['file_name']).stem: im for im in gt['images']}
    preds = []
    if not mask_dir.exists():
        return preds
    for image_mask_dir in sorted(mask_dir.iterdir()):
        if not image_mask_dir.is_dir() or image_mask_dir.name not in stem_to_image:
            continue
        image_id = stem_to_image[image_mask_dir.name]['id']
        for mask_path in sorted(image_mask_dir.glob('*.png')):
            try:
                category_id = int(mask_path.name.split('_', 1)[0])
            except ValueError:
                continue
            if category_id not in CLASS_NAMES:
                continue
            arr = np.array(Image.open(mask_path).convert('L')) > 0
            if not arr.any():
                continue
            ys, xs = np.where(arr)
            x1, x2 = int(xs.min()), int(xs.max()) + 1
            y1, y2 = int(ys.min()), int(ys.max()) + 1
            area = float((x2 - x1) * (y2 - y1))
            if area < min_area:
                continue
            preds.append({
                'image_id': image_id,
                'category_id': category_id,
                'bbox': [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                'score': score,
            })
    return nms(preds, 0.50)


def evaluate(gt_path: Path, pred_path: Path):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    coco_gt = COCO(str(gt_path))
    coco_dt = coco_gt.loadRes(str(pred_path))
    ev = COCOeval(coco_gt, coco_dt, 'bbox')
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return {'mAP': float(ev.stats[0]), 'mAP50': float(ev.stats[1]), 'mAP75': float(ev.stats[2]), 'stats': [float(x) for x in ev.stats]}


def parse_one_image_response(obj, image_id: int, image_size, score_scale: float, coord_mode: str):
    preds = []
    for det in obj.get('detections', obj.get('results', [])):
        try:
            cid = int(det.get('category_id', 0) or 0)
        except (TypeError, ValueError):
            continue
        if cid not in CLASS_NAMES:
            continue
        bbox = clamp_box_xyxy_to_xywh(det.get('bbox', []), image_size[0], image_size[1], coord_mode)
        if not bbox:
            continue
        conf = float(det.get('confidence', det.get('score', 0.80)) or 0.80)
        preds.append({
            'image_id': image_id,
            'category_id': cid,
            'bbox': bbox,
            'score': round(max(0.001, min(0.999, conf * score_scale)), 6),
        })
    return post_filter(preds, image_size[0], image_size[1])


def run_api(args):
    if not args.api_key:
        raise SystemExit('set DASHSCOPE_API_KEY/QWEN_API_KEY or pass --api-key')
    gt = load_json(args.gt)
    image_by_id = {im['id']: im for im in gt['images']}
    image_dir = Path(args.gt).parent
    ids = sorted(image_by_id) if args.image_ids.strip().lower() == 'all' else [int(x) for x in args.image_ids.split(',') if x.strip()]
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    all_preds = []
    for image_id in ids:
        meta = image_by_id[image_id]
        img_path = image_dir / meta['file_name']
        img = Image.open(img_path).convert('RGB')
        prompt = make_prompt(*img.size)
        (raw_dir / f'{image_id:04d}_prompt.txt').write_text(prompt)
        raw_json_path = raw_dir / f'{image_id:04d}_raw.json'
        if args.resume and raw_json_path.exists():
            print(f'reuse raw image_id={image_id} file={meta["file_name"]}', flush=True)
            obj = load_json(raw_json_path)
        else:
            print(f'API full detect image_id={image_id} file={meta["file_name"]}', flush=True)
            raw = call_api(img, prompt, args.api_key, args.model, args.timeout, args.retries)
            (raw_dir / f'{image_id:04d}_raw.txt').write_text(raw)
            obj = parse_json(raw)
            raw_json_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
            if args.sleep > 0:
                time.sleep(args.sleep)
        preds = parse_one_image_response(obj, image_id, img.size, args.score_scale, args.coord_mode)
        all_preds.extend(preds)
        if args.visualize:
            draw_vis(img_path, preds, raw_dir / f'{image_id:04d}_vis.jpg')
        print(f'  kept={len(preds)}', flush=True)
    save_json(all_preds, args.out)
    return all_preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', default=os.environ.get('DASHSCOPE_API_KEY') or os.environ.get('QWEN_API_KEY'))
    parser.add_argument('--model', default='qwen3.5-plus')
    parser.add_argument('--gt', default=str(GT))
    parser.add_argument('--image-ids', default='all')
    parser.add_argument('--out', default=str(ROOT / 'qwen_defect_api_full_v1.json'))
    parser.add_argument('--raw-dir', default=str(ROOT / 'qwen_defect_api_full_v1_raw'))
    parser.add_argument('--timeout', type=int, default=120)
    parser.add_argument('--retries', type=int, default=2)
    parser.add_argument('--coord-mode', default='norm1000', choices=['norm1000', 'pixel', 'auto'])
    parser.add_argument('--score-scale', type=float, default=1.0)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    parser.add_argument('--resume', action='store_true', help='reuse existing raw-dir/*_raw.json responses')
    parser.add_argument('--sleep', type=float, default=0.0, help='seconds to sleep between API requests')
    parser.add_argument('--masks-only', action='store_true', help='convert existing SAM3 masks to COCO bbox predictions and exit')
    parser.add_argument('--mask-dir', default=str(MASK_DIR))
    parser.add_argument('--mask-min-area', type=float, default=800.0)
    parser.add_argument('--mask-score', type=float, default=0.50)
    args = parser.parse_args()

    if args.masks_only:
        preds = convert_masks_to_predictions(Path(args.gt), Path(args.mask_dir), args.mask_min_area, args.mask_score)
        save_json(preds, args.out)
        print(f'wrote {len(preds)} mask-derived predictions to {args.out}')
    else:
        run_api(args)

    if args.evaluate:
        metrics = evaluate(Path(args.gt), Path(args.out))
        eval_path = str(Path(args.out).with_suffix('')) + '_eval.json'
        save_json(metrics, eval_path)
        print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
