#!/usr/bin/env python3
"""Direct Qwen API inference v2 for defect-detection-yjplx-fxobh-fsod-amdi.

This script deliberately uses API outputs only. It splits the task into
specialized passes instead of asking one prompt to solve all four classes:
  - fishplate pass: category 1 vs 4, whole fishplate boxes only
  - fastener pass: category 2 vs 3, local present/missing positions only
  - optional grid fastener pass: crop-level fastener/missing detection

Example small-batch run:
  DASHSCOPE_API_KEY='...' python qwen_defect_direct_api_v2.py \
    --image-ids 136,115,88,48,26,152 \
    --out qwen_defect_direct_api_v2_small.json \
    --raw-dir qwen_defect_direct_api_v2_small_raw \
    --passes fastener,grid \
    --resume --visualize --evaluate
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
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
DATA_ROOT = Path('/data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data/defect-detection-yjplx-fxobh-fsod-amdi')
GT = DATA_ROOT / 'test' / '_annotations.coco.json'
TRAIN = DATA_ROOT / 'train' / '_annotations.coco.json'

CLASS_NAMES = {
    1: 'defective fishplate',
    2: 'fastener',
    3: 'missing fastener',
    4: 'non defective fishplate',
}
COLORS = {
    1: (255, 0, 0),
    2: (0, 220, 0),
    3: (255, 150, 0),
    4: (0, 120, 255),
}


def load_json(path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def image_url(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=92)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def parse_json(text: str):
    text = text.strip().replace('```json', '').replace('```', '').strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r'\{.*\}', text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def call_api(images, prompt, api_key, model, timeout, retries):
    content = [{'type': 'text', 'text': prompt}]
    for label, img in images:
        content.append({'type': 'text', 'text': label})
        content.append({'type': 'image_url', 'image_url': {'url': image_url(img)}})
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': content}],
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


def clamp_xyxy_to_xywh(box, width, height, coord_mode):
    if not box or len(box) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    if coord_mode == 'norm1000' or (coord_mode == 'auto' and max(x1, y1, x2, y2) <= 1000):
        x1, x2 = x1 / 1000.0 * width, x2 / 1000.0 * width
        y1, y2 = y1 / 1000.0 * height, y2 / 1000.0 * height
    if x2 <= x1 or y2 <= y1:
        return None
    x1 = max(0.0, min(width - 1.0, x1))
    y1 = max(0.0, min(height - 1.0, y1))
    x2 = max(x1 + 1.0, min(float(width), x2))
    y2 = max(y1 + 1.0, min(float(height), y2))
    return [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]


def map_crop_box(box, crop_rect, crop_size, full_size, coord_mode):
    crop_box = clamp_xyxy_to_xywh(box, crop_size[0], crop_size[1], coord_mode)
    if not crop_box:
        return None
    ox, oy, x2, y2 = crop_rect
    rw, rh = x2 - ox, y2 - oy
    sx, sy = rw / float(crop_size[0]), rh / float(crop_size[1])
    x, y, w, h = crop_box
    mapped = [ox + x * sx, oy + y * sy, w * sx, h * sy]
    return clamp_xyxy_to_xywh([mapped[0], mapped[1], mapped[0] + mapped[2], mapped[1] + mapped[3]], full_size[0], full_size[1], 'pixel')


def iou(a, b):
    ax, ay, aw, ah = [float(v) for v in a['bbox']]
    bx, by, bw, bh = [float(v) for v in b['bbox']]
    inter = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0.0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def grouped_nms(preds, thr):
    out = []
    groups = defaultdict(list)
    for pred in preds:
        groups[(pred['image_id'], pred['category_id'])].append(pred)
    for items in groups.values():
        keep = []
        for pred in sorted(items, key=lambda x: x['score'], reverse=True):
            if all(iou(pred, old) < thr for old in keep):
                keep.append(pred)
        out.extend(keep)
    return out


def draw_reference(img_path, anns, max_side=1200):
    img = Image.open(img_path).convert('RGB')
    scale = 1.0
    if max(img.size) > max_side:
        scale = max_side / float(max(img.size))
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
    d = ImageDraw.Draw(img)
    for ann in anns:
        x, y, w, h = [float(v) * scale for v in ann['bbox']]
        cid = ann['category_id']
        d.rectangle((x, y, x + w, y + h), outline=COLORS[cid], width=5)
        d.text((x, max(0, y - 18)), f"{cid} {CLASS_NAMES[cid]}", fill=COLORS[cid])
    return img


def draw_crop_reference(img_path, ann, context=1.8, max_side=900):
    full = Image.open(img_path).convert('RGB')
    W, H = full.size
    x, y, w, h = [float(v) for v in ann['bbox']]
    pad = max(32.0, max(w, h) * context)
    x1 = max(0, int(round(x - pad)))
    y1 = max(0, int(round(y - pad)))
    x2 = min(W, int(round(x + w + pad)))
    y2 = min(H, int(round(y + h + pad)))
    crop = full.crop((x1, y1, x2, y2)).convert('RGB')
    scale = 1.0
    if max(crop.size) > max_side:
        scale = max_side / float(max(crop.size))
        crop = crop.resize((round(crop.width * scale), round(crop.height * scale)), Image.Resampling.LANCZOS)
    rx1 = (x - x1) * scale
    ry1 = (y - y1) * scale
    rx2 = (x + w - x1) * scale
    ry2 = (y + h - y1) * scale
    cid = ann['category_id']
    d = ImageDraw.Draw(crop)
    for t in range(5):
        d.rectangle((rx1 - t, ry1 - t, rx2 + t, ry2 + t), outline=COLORS[cid])
    label = f"{cid} {CLASS_NAMES[cid]} example"
    d.rectangle((0, 0, min(crop.width, 18 + len(label) * 9), 28), fill=(0, 0, 0))
    d.text((6, 7), label, fill=COLORS[cid])
    return crop


def pick_reference_payload(raw_dir, mode='crop', per_class=1, categories=(1, 4, 2, 3), prefix='reference'):
    train = load_json(TRAIN)
    anns_by_image = defaultdict(list)
    image_by_id = {im['id']: im for im in train['images']}
    for ann in train['annotations']:
        anns_by_image[ann['image_id']].append(ann)

    payload = []
    used_images = set()
    for target in categories:
        candidates = []
        for iid, anns in anns_by_image.items():
            target_anns = [a for a in anns if a['category_id'] == target]
            if not target_anns:
                continue
            c = Counter(a['category_id'] for a in anns)
            best_ann = max(target_anns, key=lambda a: float(a['bbox'][2]) * float(a['bbox'][3]))
            area = float(best_ann['bbox'][2]) * float(best_ann['bbox'][3])
            score = 1000.0 * area + 10 * c[target] + 2 * len(c) + min(sum(c.values()), 8)
            candidates.append((score, iid, best_ann, anns))
        for rank, (_, iid, ann, anns) in enumerate(sorted(candidates, reverse=True)[:per_class], 1):
            img_path = DATA_ROOT / 'train' / image_by_id[iid]['file_name']
            if mode in ('crop', 'both'):
                ref = draw_crop_reference(img_path, ann)
                name = f'{prefix}_crop_cat{target}_{rank}.jpg'
                ref.save(raw_dir / name, quality=92)
                payload.append((f'Reference crop for category {target} {CLASS_NAMES[target]}. The red box is the example target.', ref))
            if mode in ('full', 'both') and iid not in used_images:
                ref = draw_reference(img_path, anns)
                name = f'{prefix}_full_cat{target}_{rank}.jpg'
                ref.save(raw_dir / name, quality=92)
                payload.append((f'Reference full image containing category {target} {CLASS_NAMES[target]} with colored GT boxes.', ref))
                used_images.add(iid)
    return payload


def fishplate_prompt(width, height):
    return f"""
Do not think step by step. Do not explain. Output exactly one valid JSON object and nothing else.
You will see reference images with labels, then one final target image. Detect ONLY fishplate/joint-bar objects in the final target image.
Target image size: width={width}, height={height}. Return bbox coordinates normalized to 0-1000.

Allowed categories:
1 defective fishplate: the whole visible fishplate/joint bar when it is bad, damaged, bent, cracked, broken, deformed, incomplete, or visually defective.
4 non defective fishplate: the whole visible fishplate/joint bar when it is intact/good.

Rules:
- Output category 1 or category 4 only. Never output fastener or missing-fastener boxes in this pass.
- Fishplate boxes are large, around the whole long metal joint plate/bar, including its visible bolt-hole region.
- Decide bad vs good from the whole fishplate. A defective fishplate may be broken, bent, short/missing a part, heavily damaged, or visibly abnormal compared with reference examples.
- Do not output both category 1 and category 4 for the same fishplate. Choose exactly one label per fishplate.
- If no fishplate is visible, return {{"detections":[]}}.

Return schema:
{{"detections":[{{"category_id":1,"category_name":"defective fishplate","bbox":[x1_norm,y1_norm,x2_norm,y2_norm],"confidence":0.92}}]}}
""".strip()


def fastener_prompt(width, height, crop_note='full image'):
    return f"""
Do not think step by step. Do not explain. Output exactly one valid JSON object and nothing else.
You will see reference images with labels, then one final target image/crop. Detect ONLY rail fastening positions in the final target image/crop.
Current view: {crop_note}. Image size: width={width}, height={height}. Return bbox coordinates normalized to 0-1000 relative to this current view.

Allowed categories:
2 fastener: present/installed rail clip, clamp, anchor, spike assembly, rail-seat bolt/washer, or physically raised fastening component at the rail foot and sleeper/tie seat.
3 missing fastener: expected fastening position where the physical fastener is absent: empty clip seat, vacant clamp/anchor position, empty hole/seat, dark or flat depression, or symmetric aligned position with no raised component.

Required localization procedure:
- First identify the steel rail(s), then identify the concrete sleepers/ties: the light gray rectangular cement bars crossing under the rail.
- Rail fasteners exist ONLY on concrete sleeper/tie surfaces, immediately beside the rail foot. They are not in ballast stones, grass, dirt, shadows, rail head, or the long fishplate/joint-bar body.
- Use the rail as a symmetry axis. Expected fastening positions form paired left/right or upper/lower slots around the rail foot on each visible concrete sleeper.
- For each visible concrete sleeper that crosses the rail, inspect the two symmetric rail-seat positions beside the rail foot. Output a box only if the position lies on that sleeper/tie and aligns with the repeated sleeper pattern.
- For EACH expected position, output exactly one local box: category 2 if a physical raised fastener is installed, category 3 if that aligned sleeper position is empty/missing.
- Missing fasteners often look like lower/darker empty seats or vacant holes on the concrete sleeper at the symmetric position. Use the neighboring and opposite-side fasteners to infer the missing slot.
- Box tightly around the local component/seat only, not the whole sleeper, fishplate, rail, ballast, or background.
- Reject isolated bolt-like blobs that are not on a concrete sleeper/tie rail-seat. Reject bolts on fishplates, rail joints, image borders, ballast, and shadows even if they are round/metallic.
- A fishplate/joint-bar is the long metal plate connecting rail ends. Its bolt heads, washers, bolt holes, and slots are NOT category 2 or 3 in this pass.
- If the current view only shows a fishplate/joint-bar and its bolts/holes, and no concrete sleeper/tie rail-seat fastening positions, return {{"positions":[]}}.
- Do not use bolts along the middle of a fishplate as the repeated fastening row. Rail fasteners are attached around the rail foot on concrete sleepers/ties.
- Do not output categories 1 or 4 in this pass.
- Edge and partially visible expected positions should be included only when the concrete sleeper/tie and rail-seat location are visible enough to place a tight local box.

Output schema, prefer positions over detections:
{{"positions":[{{"slot_id":"upper-left-1","status":"missing","category_id":3,"bbox":[x1_norm,y1_norm,x2_norm,y2_norm],"confidence":0.90,"reason":"empty aligned seat with no raised metal fastener"}}]}}

Status decision rule:
- Use status=present and category_id=2 only when a raised metal clip/bolt/clamp is physically visible.
- Use status=missing and category_id=3 when the expected position has only a flat/dark/empty seat, hole, shadowed depression, or no raised component.
- Do not mark every expected position as present. Many target images intentionally contain missing fasteners; compare opposite-side and neighboring slots.
""".strip()


def windows(width, height, rows, cols, overlap):
    step_x = width / cols
    step_y = height / rows
    pad_x = step_x * overlap
    pad_y = step_y * overlap
    out = []
    for r in range(rows):
        for c in range(cols):
            x1 = max(0, int(round(c * step_x - pad_x)))
            y1 = max(0, int(round(r * step_y - pad_y)))
            x2 = min(width, int(round((c + 1) * step_x + pad_x)))
            y2 = min(height, int(round((r + 1) * step_y + pad_y)))
            out.append((f'r{r}_c{c}', (x1, y1, x2, y2)))
    return out


def parse_detections(obj, image_id, img_size, coord_mode, allowed, score_scale, crop_rect=None, full_size=None):
    preds = []
    rows = []
    rows.extend(obj.get('positions', []))
    rows.extend(obj.get('detections', obj.get('results', [])))
    for det in rows:
        status = str(det.get('status', '')).strip().lower()
        if status in ('present', 'installed', 'fastener'):
            cid = 2
        elif status in ('missing', 'absent', 'empty'):
            cid = 3
        else:
            try:
                cid = int(det.get('category_id', 0) or 0)
            except (TypeError, ValueError):
                continue
        if cid not in allowed:
            continue
        if crop_rect is None:
            bbox = clamp_xyxy_to_xywh(det.get('bbox', []), img_size[0], img_size[1], coord_mode)
        else:
            bbox = map_crop_box(det.get('bbox', []), crop_rect, img_size, full_size, coord_mode)
        if not bbox:
            continue
        conf = float(det.get('confidence', det.get('score', 0.8)) or 0.8)
        preds.append({
            'image_id': image_id,
            'category_id': cid,
            'bbox': bbox,
            'score': round(max(0.001, min(0.999, conf * score_scale)), 6),
        })
    return preds


def post_filter(preds, width, height, keep_categories=None):
    image_area = float(width * height)
    out = []
    keep_categories = set(keep_categories or (2, 3))
    for pred in preds:
        cid = pred['category_id']
        if cid not in keep_categories:
            continue
        x, y, w, h = pred['bbox']
        area = w * h
        frac = area / image_area
        ratio = max(w / max(h, 1.0), h / max(w, 1.0))
        if cid in (1, 4):
            if area < 100000 or frac > 0.75 or ratio > 8:
                continue
        if cid in (2, 3):
            if area < 2500 or frac > 0.14 or ratio > 7:
                continue
        out.append(pred)
    return grouped_nms(out, 0.40)


def draw_vis(img_path, dets, out_path):
    img = Image.open(img_path).convert('RGB')
    d = ImageDraw.Draw(img)
    for pred in dets:
        x, y, w, h = pred['bbox']
        color = COLORS[pred['category_id']]
        d.rectangle((x, y, x + w, y + h), outline=color, width=4)
        d.text((x, max(0, y - 16)), f"{pred['category_id']} {pred['score']:.2f}", fill=color)
    img.save(out_path)


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


def subset_gt(gt_path, image_ids, out_path):
    gt = load_json(gt_path)
    keep = set(image_ids)
    sub = dict(gt)
    sub['images'] = [im for im in gt['images'] if im['id'] in keep]
    sub['annotations'] = [ann for ann in gt['annotations'] if ann['image_id'] in keep]
    save_json(sub, out_path)
    return out_path


def run(args):
    if not args.api_key:
        raise SystemExit('set DASHSCOPE_API_KEY/QWEN_API_KEY or pass --api-key')
    gt = load_json(args.gt)
    image_by_id = {im['id']: im for im in gt['images']}
    image_dir = Path(args.gt).parent
    image_ids = sorted(image_by_id) if args.image_ids.strip().lower() == 'all' else [int(x) for x in args.image_ids.split(',') if x.strip()]
    passes = {x.strip() for x in args.passes.split(',') if x.strip()}
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    fishplate_ref_payload = pick_reference_payload(raw_dir, args.reference_mode, args.ref_per_class, (1, 4), 'fishplate_ref') if args.references else []
    fastener_ref_payload = pick_reference_payload(raw_dir, args.reference_mode, args.ref_per_class, (2, 3), 'fastener_ref') if args.references else []
    all_preds = []
    for image_id in image_ids:
        meta = image_by_id[image_id]
        img_path = image_dir / meta['file_name']
        img = Image.open(img_path).convert('RGB')
        image_preds = []
        if 'fishplate' in passes:
            key = f'{image_id:04d}_fishplate'
            prompt = fishplate_prompt(*img.size)
            (raw_dir / f'{key}_prompt.txt').write_text(prompt)
            raw_json = raw_dir / f'{key}_raw.json'
            if args.resume and raw_json.exists():
                obj = load_json(raw_json)
                print(f'reuse {key}', flush=True)
            else:
                print(f'API fishplate image_id={image_id} file={meta["file_name"]}', flush=True)
                raw = call_api(fishplate_ref_payload + [('Target image. Detect only this final image.', img)], prompt, args.api_key, args.model, args.timeout, args.retries)
                (raw_dir / f'{key}_raw.txt').write_text(raw)
                obj = parse_json(raw)
                raw_json.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
                if args.sleep > 0:
                    time.sleep(args.sleep)
            image_preds.extend(parse_detections(obj, image_id, img.size, args.coord_mode, {1, 4}, args.score_scale))

        if 'fastener' in passes:
            key = f'{image_id:04d}_fastener'
            prompt = fastener_prompt(*img.size)
            (raw_dir / f'{key}_prompt.txt').write_text(prompt)
            raw_json = raw_dir / f'{key}_raw.json'
            if args.resume and raw_json.exists():
                obj = load_json(raw_json)
                print(f'reuse {key}', flush=True)
            else:
                print(f'API fastener image_id={image_id} file={meta["file_name"]}', flush=True)
                raw = call_api(fastener_ref_payload + [('Target image. Detect only this final image.', img)], prompt, args.api_key, args.model, args.timeout, args.retries)
                (raw_dir / f'{key}_raw.txt').write_text(raw)
                obj = parse_json(raw)
                raw_json.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
                if args.sleep > 0:
                    time.sleep(args.sleep)
            image_preds.extend(parse_detections(obj, image_id, img.size, args.coord_mode, {2, 3}, args.score_scale))

        if 'grid' in passes:
            for crop_name, rect in windows(img.width, img.height, args.rows, args.cols, args.overlap):
                crop = img.crop(rect).convert('RGB')
                if args.upscale != 1:
                    crop_api = crop.resize((round(crop.width * args.upscale), round(crop.height * args.upscale)), Image.Resampling.BICUBIC)
                else:
                    crop_api = crop
                key = f'{image_id:04d}_{crop_name}'
                prompt = fastener_prompt(crop_api.width, crop_api.height, f'crop {crop_name} from a larger image')
                (raw_dir / f'{key}_prompt.txt').write_text(prompt)
                raw_json = raw_dir / f'{key}_raw.json'
                if args.resume and raw_json.exists():
                    obj = load_json(raw_json)
                    print(f'reuse {key}', flush=True)
                else:
                    print(f'API grid image_id={image_id} crop={crop_name} rect={rect}', flush=True)
                    raw = call_api(fastener_ref_payload + [('Target crop. Detect only this final crop.', crop_api)], prompt, args.api_key, args.model, args.timeout, args.retries)
                    (raw_dir / f'{key}_raw.txt').write_text(raw)
                    obj = parse_json(raw)
                    raw_json.write_text(json.dumps(obj, ensure_ascii=False, indent=2))
                    if args.sleep > 0:
                        time.sleep(args.sleep)
                image_preds.extend(parse_detections(obj, image_id, crop_api.size, args.coord_mode, {2, 3}, args.score_scale, rect, img.size))

        image_preds = post_filter(image_preds, img.width, img.height, args.keep_categories)
        all_preds.extend(image_preds)
        print(f'  kept={len(image_preds)} counts={dict(Counter(p["category_id"] for p in image_preds))}', flush=True)
        if args.visualize:
            draw_vis(img_path, image_preds, raw_dir / f'{image_id:04d}_vis.jpg')
    all_preds = grouped_nms(all_preds, 0.40)
    save_json(all_preds, args.out)
    return image_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--api-key', default=os.environ.get('DASHSCOPE_API_KEY') or os.environ.get('QWEN_API_KEY'))
    parser.add_argument('--model', default='qwen3.5-plus')
    parser.add_argument('--gt', default=str(GT))
    parser.add_argument('--image-ids', default='all')
    parser.add_argument('--out', default=str(ROOT / 'qwen_defect_direct_api_v2.json'))
    parser.add_argument('--raw-dir', default=str(ROOT / 'qwen_defect_direct_api_v2_raw'))
    parser.add_argument('--passes', default='fishplate,fastener')
    parser.add_argument('--keep-categories', type=lambda s: tuple(int(x) for x in s.split(',') if x.strip()), default=(1,2,3,4), help='categories to keep in final direct output')
    parser.add_argument('--references', action='store_true', default=True)
    parser.add_argument('--no-references', dest='references', action='store_false')
    parser.add_argument('--reference-mode', choices=['crop', 'full', 'both'], default='full')
    parser.add_argument('--ref-per-class', type=int, default=1)
    parser.add_argument('--rows', type=int, default=2)
    parser.add_argument('--cols', type=int, default=2)
    parser.add_argument('--overlap', type=float, default=0.12)
    parser.add_argument('--upscale', type=float, default=1.6)
    parser.add_argument('--timeout', type=int, default=240)
    parser.add_argument('--retries', type=int, default=2)
    parser.add_argument('--coord-mode', choices=['norm1000', 'pixel', 'auto'], default='norm1000')
    parser.add_argument('--score-scale', type=float, default=1.0)
    parser.add_argument('--sleep', type=float, default=0.0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--evaluate', action='store_true')
    args = parser.parse_args()
    image_ids = run(args)
    if args.evaluate:
        gt_path = Path(args.gt)
        if args.image_ids.strip().lower() != 'all':
            gt_path = Path(str(Path(args.out).with_suffix('')) + '_gt_subset.json')
            subset_gt(Path(args.gt), image_ids, gt_path)
        metrics = evaluate(gt_path, Path(args.out))
        eval_path = str(Path(args.out).with_suffix('')) + '_eval.json'
        save_json(metrics, eval_path)
        print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
