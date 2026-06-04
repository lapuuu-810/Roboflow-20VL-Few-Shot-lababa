#!/usr/bin/env python3
"""Use Qwen API to re-box aquarium jellyfish bell/head candidates.

Small API test:
  DASHSCOPE_API_KEY='<your_api_key>' python \
    /data/LPP/cvpr/few_shot/sam3-main/best_sam3/aquarium-combined-fsod-gjvb/qwen_aquarium_jellyfish_api_rebox.py \
    --image-ids 17,23,39 \
    --max-jellyfish-per-image 32 \
    --out /data/LPP/cvpr/few_shot/sam3-main/best_sam3/aquarium-combined-fsod-gjvb/qwen_jelly_api_rebox_3img_v1.json \
    --raw-dir /data/LPP/cvpr/few_shot/sam3-main/best_sam3/aquarium-combined-fsod-gjvb/qwen_jelly_api_rebox_3img_v1_raw \
    --evaluate

The benchmark convention for this dataset marks the visible jellyfish bell/head. This
script asks the VLM to output a new tight bell/head bbox in the local crop coordinate
system; it does not apply a fixed geometric shrink.
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
GT = Path('/data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data/aquarium-combined-fsod-gjvb/test/_annotations.coco.json')
BASE = ROOT / 'baseline_best.json'
CLASS_NAMES = {1: 'fish', 2: 'jellyfish', 3: 'penguin', 4: 'puffin', 5: 'shark', 6: 'starfish', 7: 'stingray'}


def load_json(p):
    with open(p) as f:
        return json.load(f)


def save_json(x, p):
    with open(p, 'w') as f:
        json.dump(x, f, indent=2, ensure_ascii=False)


def image_url(img):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=92)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def make_crop(full, bbox, context=0.75, max_side=900):
    W, H = full.size
    x, y, w, h = [float(v) for v in bbox]
    pad = max(18.0, max(w, h) * context)
    x1 = int(clamp(round(x - pad), 0, W - 1))
    y1 = int(clamp(round(y - pad), 0, H - 1))
    x2 = int(clamp(round(x + w + pad), x1 + 1, W))
    y2 = int(clamp(round(y + h + pad), y1 + 1, H))
    crop = full.crop((x1, y1, x2, y2)).convert('RGB')
    scale = 1.0
    if max(crop.size) > max_side:
        scale = max_side / float(max(crop.size))
        crop = crop.resize((int(round(crop.size[0] * scale)), int(round(crop.size[1] * scale))), Image.Resampling.LANCZOS)
    rx1 = (x - x1) * scale
    ry1 = (y - y1) * scale
    rx2 = (x + w - x1) * scale
    ry2 = (y + h - y1) * scale
    draw = ImageDraw.Draw(crop)
    for t in range(3):
        draw.rectangle((rx1 - t, ry1 - t, rx2 + t, ry2 + t), outline=(255, 0, 0))
    return crop, (x1, y1, scale), [rx1, ry1, rx2, ry2]


def make_prompt(crop_w, crop_h, red_box):
    return f"""
Do not think step by step. Do not explain. Output exactly one valid JSON object and nothing else.
You are calibrating one aquarium detection candidate.
The image is a local crop. A red rectangle shows the original detector candidate.
Crop size: width={crop_w}, height={crop_h}. Coordinates must be normalized integer coordinates in the 0-1000 range within this crop: x_norm = x / crop_width * 1000, y_norm = y / crop_height * 1000.

Task:
- Decide whether there is a visible jellyfish bell/head corresponding to the red candidate.
- If yes, output a tight bbox around the visible jellyfish bell/head only.
- Do NOT include long trailing tentacles/tail strands in the bbox unless they are fused into the head silhouette.
- Do NOT box water surface waves, foam, splash, bubbles, glare, reflections, rocks, plants, tank background, or empty water.
- If there are multiple jellyfish inside/near the red candidate, choose the one whose bell/head overlaps or is closest to the red candidate center.
- The official category remains category_id=2 jellyfish.

Original red candidate box in crop xyxy: {[round(v, 1) for v in red_box]}.

Return schema:
{{"keep":true,"category_id":2,"category_name":"jellyfish","bbox":[x1_norm,y1_norm,x2_norm,y2_norm],"confidence":0.93,"reason":"tight bell/head only"}}
If the red candidate is not a jellyfish bell/head, return:
{{"keep":false,"category_id":2,"category_name":"jellyfish","bbox":[],"confidence":0.90,"false_type":"water reflection","reason":"no jellyfish bell/head visible"}}
""".strip()


def call_api(img, prompt, api_key, model, timeout=120, retries=2):
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': [{'type': 'text', 'text': prompt}, {'type': 'image_url', 'image_url': {'url': image_url(img)}}]}],
        'temperature': 0,
        'response_format': {'type': 'json_object'},
        'enable_thinking': False,
        'thinking': {'type': 'disabled'},
    }
    data = json.dumps(payload).encode('utf-8')
    last = None
    for i in range(retries + 1):
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
        time.sleep(2 + i * 2)
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


def crop_xyxy_to_full_xywh(box, origin_scale, full_size, crop_size=None, coord_mode='norm1000'):
    if not box or len(box) != 4:
        return None
    ox, oy, scale = origin_scale
    W, H = full_size
    x1, y1, x2, y2 = [float(v) for v in box]
    if coord_mode == 'norm1000' or (coord_mode == 'auto' and max(x1, y1, x2, y2) <= 1000 and crop_size):
        cw, ch = crop_size
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
    x1 = clamp(x1, 0, W - 1)
    y1 = clamp(y1, 0, H - 1)
    x2 = clamp(x2, x1 + 1, W)
    y2 = clamp(y2, y1 + 1, H)
    return [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]



def box_area(b):
    return max(1.0, float(b[2]) * float(b[3]))


def box_center(b):
    return float(b[0]) + float(b[2]) / 2.0, float(b[1]) + float(b[3]) / 2.0


def box_intersection(a, b):
    ax, ay, aw, ah = [float(v) for v in a]
    bx, by, bw, bh = [float(v) for v in b]
    ax2, ay2 = ax + aw, ay + ah
    bx2, by2 = bx + bw, by + bh
    return max(0.0, min(ax2, bx2) - max(ax, bx)) * max(0.0, min(ay2, by2) - max(ay, by))


def accept_rebox(old_box, new_box, max_shift, min_area_ratio, max_area_ratio, min_inside):
    ox, oy = box_center(old_box)
    nx, ny = box_center(new_box)
    diag = max(1.0, (float(old_box[2]) ** 2 + float(old_box[3]) ** 2) ** 0.5)
    shift = ((nx - ox) ** 2 + (ny - oy) ** 2) ** 0.5 / diag
    area_ratio = box_area(new_box) / box_area(old_box)
    inside = box_intersection(old_box, new_box) / box_area(new_box)
    ok = shift <= max_shift and min_area_ratio <= area_ratio <= max_area_ratio and inside >= min_inside
    return ok, {'shift': round(shift, 4), 'area_ratio': round(area_ratio, 4), 'inside_old': round(inside, 4)}

def draw_vis(img_path, dets, out_path):
    img = Image.open(img_path).convert('RGB')
    d = ImageDraw.Draw(img)
    for p in dets:
        x, y, w, h = p['bbox']
        col = (255, 0, 255) if p['category_id'] == 2 else (0, 220, 0)
        d.rectangle((x, y, x + w, y + h), outline=col, width=3)
        d.text((x, y), f"{p['category_id']}:{CLASS_NAMES[p['category_id']]} {p['score']:.2f}", fill=col)
    img.save(out_path)


def evaluate(gt_path, pred_path):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt = COCO(str(gt_path))
    dt = gt.loadRes(str(pred_path))
    ev = COCOeval(gt, dt, 'bbox')
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return {'mAP': float(ev.stats[0]), 'mAP50': float(ev.stats[1]), 'mAP75': float(ev.stats[2]), 'stats': [float(x) for x in ev.stats]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--api-key', default=os.environ.get('DASHSCOPE_API_KEY') or os.environ.get('QWEN_API_KEY'))
    ap.add_argument('--model', default='qwen3.5-plus')
    ap.add_argument('--pred', default=str(BASE))
    ap.add_argument('--gt', default=str(GT))
    ap.add_argument('--image-ids', default='17,23,39')
    ap.add_argument('--out', default=str(ROOT / 'qwen_jelly_api_rebox_test.json'))
    ap.add_argument('--raw-dir', default=str(ROOT / 'qwen_jelly_api_rebox_raw'))
    ap.add_argument('--max-jellyfish-per-image', type=int, default=32)
    ap.add_argument('--min-score', type=float, default=0.20)
    ap.add_argument('--keep-thr', type=float, default=0.70)
    ap.add_argument('--drop-thr', type=float, default=0.90)
    ap.add_argument('--context', type=float, default=0.75)
    ap.add_argument('--accept-shift', type=float, default=0.5)
    ap.add_argument('--accept-area-min', type=float, default=0.15)
    ap.add_argument('--accept-area-max', type=float, default=1.3)
    ap.add_argument('--accept-inside', type=float, default=0.9)
    ap.add_argument('--timeout', type=int, default=120)
    ap.add_argument('--retries', type=int, default=2)
    ap.add_argument('--coord-mode', default='norm1000', choices=['norm1000', 'pixel', 'auto'])
    ap.add_argument('--evaluate', action='store_true')
    args = ap.parse_args()
    if not args.api_key:
        raise SystemExit('set DASHSCOPE_API_KEY or --api-key')

    gt = load_json(args.gt)
    img_by_id = {im['id']: im for im in gt['images']}
    image_dir = Path(args.gt).parent
    preds = load_json(args.pred)
    ids = sorted(img_by_id) if args.image_ids.strip().lower() == 'all' else [int(x) for x in args.image_ids.split(',') if x.strip()]
    idset = set(ids)
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    by = defaultdict(list)
    for p in preds:
        by[p['image_id']].append(dict(p))

    final = []
    logs = []
    for iid in sorted(by):
        if iid not in idset:
            final.extend(by[iid])
            continue
        im = img_by_id[iid]
        img_path = image_dir / im['file_name']
        full = Image.open(img_path).convert('RGB')
        arr = sorted(by[iid], key=lambda p: p['score'], reverse=True)
        jelly_seen = 0
        image_out = []
        for p in arr:
            if p['category_id'] != 2 or p['score'] < args.min_score or jelly_seen >= args.max_jellyfish_per_image:
                image_out.append(p)
                continue
            jelly_seen += 1
            crop, origin_scale, red_box = make_crop(full, p['bbox'], args.context)
            prompt = make_prompt(crop.size[0], crop.size[1], red_box)
            stem = f'{iid:04d}_jelly_{jelly_seen:02d}'
            crop.save(raw_dir / f'{stem}_crop.jpg')
            (raw_dir / f'{stem}_prompt.txt').write_text(prompt)
            print(f'API jellyfish rebox image_id={iid} candidate={jelly_seen} score={p["score"]:.3f}', flush=True)
            raw = call_api(crop, prompt, args.api_key, args.model, args.timeout, args.retries)
            (raw_dir / f'{stem}_raw.txt').write_text(raw)
            obj = parse_json(raw)
            (raw_dir / f'{stem}_raw.json').write_text(json.dumps(obj, indent=2, ensure_ascii=False))
            conf = float(obj.get('confidence') or 0)
            keep = bool(obj.get('keep', True))
            q = dict(p)
            action = 'keep_original'
            gate = {}
            if keep and conf >= args.keep_thr:
                new_box = crop_xyxy_to_full_xywh(obj.get('bbox', []), origin_scale, full.size, crop.size, args.coord_mode)
                if new_box:
                    ok, gate = accept_rebox(p['bbox'], new_box, args.accept_shift, args.accept_area_min, args.accept_area_max, args.accept_inside)
                    if ok:
                        q['bbox'] = new_box
                        q['score'] = round(max(0.001, min(0.999, q['score'] * (0.82 + 0.18 * conf))), 6)
                        action = 'rebox'
                    else:
                        action = 'reject_rebox'
            elif (not keep) and conf >= args.drop_thr:
                action = 'drop'
                logs.append({'image_id': iid, 'old_bbox': p['bbox'], 'action': action, 'confidence': conf, 'false_type': obj.get('false_type', ''), 'reason': obj.get('reason', '')})
                continue
            image_out.append(q)
            logs.append({'image_id': iid, 'old_bbox': p['bbox'], 'new_bbox': q['bbox'], 'action': action, 'confidence': conf, 'gate': gate, 'reason': obj.get('reason', '')})
        final.extend(image_out)
        draw_vis(img_path, image_out, raw_dir / f'{iid:04d}_merged_vis.jpg')
    save_json(final, args.out)
    save_json(logs, str(Path(args.out).with_suffix('')) + '_log.json')
    print('wrote', args.out, 'preds', len(final), 'changes', sum(1 for x in logs if x['action'] != 'keep_original'))
    if args.evaluate:
        res = evaluate(args.gt, args.out)
        save_json(res, str(Path(args.out).with_suffix('')) + '_eval.json')
        print(json.dumps(res, indent=2))


if __name__ == '__main__':
    main()
