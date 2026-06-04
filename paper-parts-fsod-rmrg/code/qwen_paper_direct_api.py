#!/usr/bin/env python3
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
DATA_ROOT = Path('/data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data/paper-parts-fsod-rmrg')
GT = DATA_ROOT / 'test' / '_annotations.coco.json'

CATEGORY_HINTS = {
    1: 'author: author names below or near the title, not affiliations',
    2: 'chapter: large chapter heading or chapter label',
    3: 'equation: mathematical formula body, excluding the equation number',
    4: 'equation number: standalone equation index such as (1), (2.3), aligned near equation margin',
    5: 'figure: image, plot, diagram, chart, photograph, or graphical panel',
    6: 'figure caption: caption text directly describing a figure, often starts with Fig. or Figure',
    7: 'footnote: small note text at page bottom or separated footnote area',
    8: 'list of content heading: table-of-contents heading line',
    9: 'list of content text: entries inside a table of contents or list of figures/tables',
    10: 'page number: page numeral or running page indicator',
    11: 'paragraph: normal body text paragraph blocks',
    12: 'reference text: bibliography/reference entries',
    13: 'section: main section heading',
    14: 'subsection: subsection heading',
    15: 'subsubsection: lower-level subsubsection heading',
    16: 'table: tabular data grid or table body',
    17: 'table caption: caption text directly describing a table',
    18: 'table of contents text: text entries in a table of contents',
    19: 'title: document title or paper title',
}

COLORS = [
    (230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200),
    (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230),
    (210, 245, 60), (250, 190, 212), (0, 128, 128), (220, 190, 255),
    (170, 110, 40), (255, 250, 200), (128, 0, 0), (170, 255, 195),
    (128, 128, 0), (255, 215, 180), (0, 0, 128),
]


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def image_url(img: Image.Image, quality: int = 92) -> str:
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=quality)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def resize_for_api(img: Image.Image, max_side: int):
    if max_side <= 0 or max(img.size) <= max_side:
        return img, 1.0
    scale = max_side / float(max(img.size))
    resized = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
    return resized, scale


def parse_ids(text: str, image_by_id):
    text = text.strip().lower()
    if text == 'all':
        return sorted(image_by_id)
    if '-' in text and ',' not in text:
        a, b = [int(x) for x in text.split('-', 1)]
        return [i for i in sorted(image_by_id) if a <= i <= b]
    return [int(x) for x in text.split(',') if x.strip()]


def make_prompt(width: int, height: int, category_names):
    class_lines = []
    for cid in sorted(category_names):
        if cid == 0:
            continue
        class_lines.append(f'{cid}: {category_names[cid]} - {CATEGORY_HINTS.get(cid, category_names[cid])}')
    classes = '\n'.join(class_lines)
    return f"""
Output exactly one valid JSON object and no explanation.
You are doing document layout object detection on a scanned academic paper page.
The image sent to you has width={width}, height={height} pixels.

Detect all visible layout regions belonging to these official classes:
{classes}

Rules:
- Return tight rectangular boxes around complete layout regions.
- Use one box per paragraph block, heading, caption, figure, table, equation, equation number, footnote, page number, title, author block, or reference entry/block.
- For normal body text, group lines into paragraph boxes. Do not box each text line separately.
- For references, box each reference entry if separable; otherwise use compact reference-text blocks.
- For figure/table captions, separate the caption from the figure/table body.
- For equations with a number, return the equation as class 3 and the number as class 4.
- Ignore decorative borders, page background, and the dataset placeholder class 0.
- Avoid duplicate boxes for the same region.
- If unsure about a class, choose the closest official layout class.

Return schema:
{{"detections":[{{"category_id":11,"category_name":"paragraph","bbox":[x1,y1,x2,y2],"confidence":0.87}}]}}
Coordinates must be absolute pixel coordinates in the sent image, with 0 <= x1 < x2 <= {width} and 0 <= y1 < y2 <= {height}.
""".strip()


def call_api(img: Image.Image, prompt: str, api_key: str, model: str, timeout: int, retries: int):
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
    last = None
    api_key = (api_key or '').strip().replace('\ufeff', '')
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


def parse_json(text: str):
    text = text.strip().replace('```json', '').replace('```', '').strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'\{.*\}', text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def clamp_xyxy_to_xywh(box, sent_width: int, sent_height: int, scale_back: float, coord_mode: str):
    if not box or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except Exception:
        return None
    full_w = sent_width * scale_back
    full_h = sent_height * scale_back
    max_coord = max(abs(x1), abs(y1), abs(x2), abs(y2))
    use_norm1000 = coord_mode == 'norm1000' or (coord_mode == 'auto' and max_coord <= 1000.0)
    if max_coord <= 1.5:
        x1, x2 = x1 * full_w, x2 * full_w
        y1, y2 = y1 * full_h, y2 * full_h
    elif use_norm1000:
        x1, x2 = x1 / 1000.0 * full_w, x2 / 1000.0 * full_w
        y1, y2 = y1 / 1000.0 * full_h, y2 / 1000.0 * full_h
    else:
        x1, y1, x2, y2 = x1 * scale_back, y1 * scale_back, x2 * scale_back, y2 * scale_back
    if x2 <= x1 or y2 <= y1:
        return None
    x1 = max(0.0, min(full_w - 1.0, x1))
    y1 = max(0.0, min(full_h - 1.0, y1))
    x2 = max(x1 + 1.0, min(full_w, x2))
    y2 = max(y1 + 1.0, min(full_h, y2))
    return [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]


def box_iou(a, b):
    ax, ay, aw, ah = [float(v) for v in a['bbox']]
    bx, by, bw, bh = [float(v) for v in b['bbox']]
    inter = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0.0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def nms(preds, thr: float):
    out = []
    groups = defaultdict(list)
    for p in preds:
        groups[(p['image_id'], p['category_id'])].append(p)
    for items in groups.values():
        keep = []
        for p in sorted(items, key=lambda x: x['score'], reverse=True):
            if all(box_iou(p, q) < thr for q in keep):
                keep.append(p)
        out.extend(keep)
    return out


def draw_vis(img_path: Path, preds, out_path: Path, category_names):
    img = Image.open(img_path).convert('RGB')
    d = ImageDraw.Draw(img)
    for p in preds:
        x, y, w, h = p['bbox']
        cid = p['category_id']
        color = COLORS[(cid - 1) % len(COLORS)]
        d.rectangle((x, y, x + w, y + h), outline=color, width=4)
        label = f'{cid}:{category_names.get(cid, cid)} {p["score"]:.2f}'
        d.text((x, max(0, y - 14)), label, fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def evaluate(gt_path: Path, pred_path: Path, image_ids=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    eval_gt_path = gt_path
    if image_ids is not None:
        gt_data = load_json(gt_path)
        keep = set(int(i) for i in image_ids)
        gt_data['images'] = [im for im in gt_data['images'] if int(im['id']) in keep]
        gt_data['annotations'] = [ann for ann in gt_data['annotations'] if int(ann['image_id']) in keep]
        eval_gt_path = pred_path.with_name(pred_path.stem + '_subset_gt.json')
        save_json(gt_data, eval_gt_path)
    coco = COCO(str(eval_gt_path))
    with open(pred_path) as f:
        preds = json.load(f)
    if image_ids is not None:
        keep = set(int(i) for i in image_ids)
        preds = [p for p in preds if int(p['image_id']) in keep]
        filtered_pred_path = pred_path.with_name(pred_path.stem + '_subset_preds.json')
        save_json(preds, filtered_pred_path)
        pred_path = filtered_pred_path
    if not preds:
        stats = [0.0] * 12
        return {'mAP': 0.0, 'mAP50': 0.0, 'mAP75': 0.0, 'stats': stats}
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
    ap.add_argument('--image-ids', default='0-9')
    ap.add_argument('--out', default=str(ROOT / 'qwen_paper_direct_api_v1.json'))
    ap.add_argument('--raw-dir', default=str(ROOT / 'qwen_paper_direct_api_v1_raw'))
    ap.add_argument('--max-side', type=int, default=1600)
    ap.add_argument('--coord-mode', default='auto', choices=['auto', 'pixel', 'norm1000'])
    ap.add_argument('--timeout', type=int, default=180)
    ap.add_argument('--retries', type=int, default=1)
    ap.add_argument('--min-score', type=float, default=0.03)
    ap.add_argument('--nms', type=float, default=0.35)
    ap.add_argument('--max-per-image', type=int, default=80)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--visualize', action='store_true')
    ap.add_argument('--evaluate', action='store_true')
    args = ap.parse_args()
    gt_path = Path(args.gt)
    gt = load_json(gt_path)
    category_names = {int(c['id']): c['name'] for c in gt['categories']}
    valid_cats = set(category_names) - {0}
    image_by_id = {int(im['id']): im for im in gt['images']}
    ids = parse_ids(args.image_ids, image_by_id)
    image_dir = gt_path.parent
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    all_preds = []
    for idx, image_id in enumerate(ids, 1):
        meta = image_by_id[image_id]
        img_path = image_dir / meta['file_name']
        full_img = Image.open(img_path).convert('RGB')
        api_img, scale = resize_for_api(full_img, args.max_side)
        scale_back = 1.0 / scale
        raw_json = raw_dir / f'{image_id:04d}_raw.json'
        raw_txt = raw_dir / f'{image_id:04d}_raw.txt'
        prompt_path = raw_dir / f'{image_id:04d}_prompt.txt'
        if args.resume and raw_json.exists():
            obj = load_json(raw_json)
        else:
            if not args.api_key:
                raise SystemExit('set DASHSCOPE_API_KEY or QWEN_API_KEY')
            prompt = make_prompt(api_img.width, api_img.height, category_names)
            prompt_path.write_text(prompt)
            print(f'[{idx}/{len(ids)}] API image_id={image_id} {meta["file_name"]} sent={api_img.width}x{api_img.height}', flush=True)
            raw = call_api(api_img, prompt, args.api_key, args.model, args.timeout, args.retries)
            raw_txt.write_text(raw)
            obj = parse_json(raw)
            save_json(obj, raw_json)

        image_preds = []
        for det in obj.get('detections', obj.get('items', obj if isinstance(obj, list) else [])):
            if not isinstance(det, dict):
                continue
            try:
                cid = int(det.get('category_id', det.get('class_id')))
            except Exception:
                continue
            if cid not in valid_cats:
                continue
            score = float(det.get('confidence', det.get('score', 0.55)))
            if score < args.min_score:
                continue
            bbox = clamp_xyxy_to_xywh(det.get('bbox') or det.get('box'), api_img.width, api_img.height, scale_back, args.coord_mode)
            if not bbox:
                continue
            if bbox[2] * bbox[3] < 8:
                continue
            image_preds.append({'image_id': image_id, 'category_id': cid, 'bbox': bbox, 'score': round(max(0.001, min(0.999, score)), 4)})
        image_preds = nms(image_preds, args.nms)
        image_preds = sorted(image_preds, key=lambda p: p['score'], reverse=True)[:args.max_per_image]
        all_preds.extend(image_preds)
        save_json({'image_id': image_id, 'file_name': meta['file_name'], 'predictions': image_preds}, raw_dir / f'{image_id:04d}_parsed.json')
        if args.visualize:
            draw_vis(img_path, image_preds, raw_dir / f'{image_id:04d}_vis.jpg', category_names)
        print(f'  parsed {len(image_preds)} detections', flush=True)

    all_preds = nms(all_preds, args.nms)
    save_json(all_preds, Path(args.out))
    print(f'wrote {len(all_preds)} predictions to {args.out}', flush=True)
    if args.evaluate:
        metrics = evaluate(gt_path, Path(args.out), ids)
        eval_path = Path(args.out).with_name(Path(args.out).stem + '_eval.json')
        save_json(metrics, eval_path)
        print(json.dumps(metrics, indent=2), flush=True)


if __name__ == '__main__':
    main()
