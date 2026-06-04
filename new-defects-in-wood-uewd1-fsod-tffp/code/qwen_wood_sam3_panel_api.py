#!/usr/bin/env python3
from __future__ import annotations

import argparse, base64, io, json, os, re, time, urllib.error, urllib.request
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw

from qwen_wood_sam3_candidate_api import (
    ROOT, GT, CLASS_NAMES, load_json, save_json, load_sam3_candidates,
    iou, nms, evaluate
)


def image_url(img):
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=94)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode('ascii')


def parse_ids(text, image_by_id):
    if text.strip().lower() == 'all':
        return sorted(image_by_id)
    return [int(x) for x in text.split(',') if x.strip()]


def make_panel(img, candidates, start_index, max_side, panel_mode='edge', panel_pad=56):
    W, H = img.size
    if panel_mode == 'local' and candidates:
        xs, ys = [], []
        for c in candidates:
            x, y, w, h = [float(v) for v in c['bbox']]
            xs.extend([x, x + w])
            ys.extend([y, y + h])
        x1 = max(0, int(min(xs) - panel_pad))
        y1 = max(0, int(min(ys) - panel_pad))
        x2 = min(W, int(max(xs) + panel_pad))
        y2 = min(H, int(max(ys) + panel_pad))
        if x2 - x1 < 220:
            cx = (x1 + x2) // 2
            x1, x2 = max(0, cx - 110), min(W, cx + 110)
        if y2 - y1 < 160:
            cy = (y1 + y2) // 2
            y1, y2 = max(0, cy - 80), min(H, cy + 80)
        panel = img.crop((x1, y1, x2, y2)).convert('RGB')
        origin = (x1, y1)
    else:
        panel = img.copy().convert('RGB')
        origin = (0, 0)

    draw = ImageDraw.Draw(panel)
    palette = [
        (255, 0, 0), (0, 170, 255), (255, 160, 0), (0, 220, 80),
        (210, 80, 255), (255, 80, 160), (255, 230, 0), (0, 230, 210)
    ]
    ox, oy = origin

    if panel_mode == 'edge':
        top_items, bottom_items = [], []
        for local_idx, c in enumerate(candidates):
            idx = start_index + local_idx
            x, y, w, h = [float(v) for v in c['bbox']]
            x -= ox
            y -= oy
            cx, cy = x + w / 2, y + h / 2
            item = (local_idx, idx, x, y, w, h, cx, cy)
            if cy < panel.height / 2:
                top_items.append(item)
            else:
                bottom_items.append(item)
        for band, items in [('top', top_items), ('bottom', bottom_items)]:
            items = sorted(items, key=lambda t: t[6])
            n = max(1, len(items))
            for pos, item in enumerate(items):
                local_idx, idx, x, y, w, h, cx, cy = item
                color = palette[local_idx % len(palette)]
                draw.rectangle((x, y, x + w, y + h), outline=color, width=3)
                lx = int((pos + 0.5) * panel.width / n) - 16
                lx = max(0, min(panel.width - 34, lx))
                ly = 3 if band == 'top' else panel.height - 21
                anchor_y = ly + 20 if band == 'top' else ly
                draw.line((lx + 17, anchor_y, cx, cy), fill=color, width=2)
                draw.rectangle((lx, ly, lx + 34, ly + 18), fill=color)
                draw.text((lx + 3, ly + 2), str(idx), fill=(0, 0, 0))
    else:
        label_slots = []
        for local_idx, c in enumerate(candidates):
            idx = start_index + local_idx
            x, y, w, h = [float(v) for v in c['bbox']]
            x -= ox
            y -= oy
            color = palette[local_idx % len(palette)]
            draw.rectangle((x, y, x + w, y + h), outline=color, width=3)
            tx = max(0, min(panel.width - 34, int(x)))
            ty = max(0, int(y) - 19)
            while any(abs(tx - a) < 34 and abs(ty - b) < 18 for a, b in label_slots):
                ty = min(panel.height - 18, ty + 18)
            label_slots.append((tx, ty))
            draw.rectangle((tx, ty, tx + 34, ty + 18), fill=color)
            draw.text((tx + 3, ty + 2), str(idx), fill=(0, 0, 0))

    if max(panel.size) > max_side:
        scale = max_side / float(max(panel.size))
        panel = panel.resize((int(round(panel.width * scale)), int(round(panel.height * scale))), Image.Resampling.LANCZOS)
    return panel


def make_prompt(candidates, start_index):
    ids = [start_index + i for i in range(len(candidates))]
    return f"""
Output exactly one valid JSON object. No explanation.
You see a wood board image with colored candidate boxes. Each box has an ID label connected to it.
For every visible candidate ID, decide whether that boxed region is one of the target defects and assign the best class.
Use only the visual boxed region. Do not invent boxes.

Classes:
1 Crack: long thin dark split or line.
2 Dead knot: dark dead knot, usually black/dark brown circular or oval knot.
3 Holes: small dark round holes or voids.
4 Live knot: intact knot with wood grain/ring texture, generally lighter than dead knot.
5 knot with crack: a knot region that visibly contains or connects to a crack.

Keep only clear target defects. Set keep=false for normal wood grain, background, duplicate/shifted boxes, partial wrong regions, or uncertain marks.
Return every ID exactly once.

IDs: {ids}
Schema: {{"items":[{{"index":0,"keep":true,"category_id":2,"confidence":0.93}},{{"index":1,"keep":false,"confidence":0.20}}]}}
""".strip()


def call_api(img, prompt, api_key, model, timeout, retries):
    api_key = (api_key or '').strip().replace('\ufeff', '')
    payload = {
        'model': model,
        'messages': [{'role': 'user', 'content': [
            {'type': 'text', 'text': prompt},
            {'type': 'image_url', 'image_url': {'url': image_url(img)}}
        ]}],
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


def parse_json(text):
    text = text.strip().replace('```json', '').replace('```', '').strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r'\{.*\}', text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def iter_response_items(obj):
    items = obj.get('items', obj.get('detections', []))
    for item in items:
        if isinstance(item, dict):
            yield item
        elif isinstance(item, str):
            for m in re.finditer(r'\{[^{}]*"index"[^{}]*\}', item):
                try:
                    yield json.loads(m.group(0))
                except Exception:
                    pass


def post_filter(preds, nms_thr, min_score, max_per_image):
    by = defaultdict(list)
    for p in preds:
        if p['score'] < min_score:
            continue
        x, y, w, h = [float(v) for v in p['bbox']]
        area = w * h
        cid = int(p['category_id'])
        ratio = max(w / max(h, 1), h / max(w, 1))
        if cid == 3 and area > 12000:
            continue
        if cid in (2, 4, 5) and ratio > 6:
            continue
        by[(p['image_id'], cid)].append(p)

    out_by_image = defaultdict(list)
    for items in by.values():
        for p in nms(items, nms_thr):
            out_by_image[p['image_id']].append(p)

    out = []
    for iid, items in out_by_image.items():
        out.extend(sorted(items, key=lambda z: z['score'], reverse=True)[:max_per_image])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--api-key', default=os.environ.get('DASHSCOPE_API_KEY') or os.environ.get('QWEN_API_KEY'))
    ap.add_argument('--model', default='qwen3.5-plus')
    ap.add_argument('--gt', default=str(GT))
    ap.add_argument('--image-ids', default='0,1')
    ap.add_argument('--sam3-jsonl', nargs='*', default=[str(p) for p in sorted(ROOT.glob('predictions_gpu*.jsonl'))])
    ap.add_argument('--out', default=str(ROOT / 'qwen_wood_sam3panel_test.json'))
    ap.add_argument('--raw-dir', default=str(ROOT / 'qwen_wood_sam3panel_raw'))
    ap.add_argument('--topk-per-image', type=int, default=40)
    ap.add_argument('--batch-size', type=int, default=20)
    ap.add_argument('--min-area', type=float, default=20)
    ap.add_argument('--max-area', type=float, default=2e6)
    ap.add_argument('--min-score', type=float, default=0)
    ap.add_argument('--candidate-nms', type=float, default=.65)
    ap.add_argument('--max-side', type=int, default=1400)
    ap.add_argument('--panel-mode', default='edge', choices=['full', 'local', 'edge'])
    ap.add_argument('--panel-pad', type=int, default=56)
    ap.add_argument('--timeout', type=int, default=150)
    ap.add_argument('--retries', type=int, default=1)
    ap.add_argument('--final-nms', type=float, default=.45)
    ap.add_argument('--min-keep-score', type=float, default=.35)
    ap.add_argument('--max-per-image', type=int, default=80)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--api-only-cached', action='store_true')
    ap.add_argument('--evaluate', action='store_true')
    args = ap.parse_args()

    if not args.api_key and not args.api_only_cached:
        raise SystemExit('set DASHSCOPE_API_KEY/QWEN_API_KEY or use --api-only-cached')

    gt = load_json(args.gt)
    image_by_id = {im['id']: im for im in gt['images']}
    ids = parse_ids(args.image_ids, image_by_id)
    image_dir = Path(args.gt).parent
    raw_dir = Path(args.raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_sam3_candidates(
        gt, [Path(p) for p in args.sam3_jsonl], ids,
        args.min_area, args.max_area, args.min_score,
        args.candidate_nms, args.topk_per_image
    )
    save_json(candidates, raw_dir / 'selected_candidates.json')

    by = defaultdict(list)
    for c in candidates:
        by[c['image_id']].append(c)

    preds = []
    total_batches = sum((len(by[i]) + args.batch_size - 1) // args.batch_size for i in ids)
    bi = 0
    for iid in ids:
        meta = image_by_id[iid]
        img = Image.open(image_dir / meta['file_name']).convert('RGB')
        cands = by.get(iid, [])
        for start in range(0, len(cands), args.batch_size):
            batch = cands[start:start + args.batch_size]
            bi += 1
            stem = f'{iid:04d}_{start:03d}'
            raw_json = raw_dir / f'{stem}_raw.json'
            if args.resume and raw_json.exists():
                obj = load_json(raw_json)
            elif args.api_only_cached:
                continue
            else:
                panel = make_panel(img, batch, start, args.max_side, args.panel_mode, args.panel_pad)
                prompt = make_prompt(batch, start)
                (raw_dir / f'{stem}_prompt.txt').write_text(prompt)
                panel.save(raw_dir / f'{stem}_panel.jpg')
                print(f'[{bi}/{total_batches}] API panel image_id={iid} candidates={start}-{start + len(batch) - 1}', flush=True)
                raw = call_api(panel, prompt, args.api_key, args.model, args.timeout, args.retries)
                (raw_dir / f'{stem}_raw.txt').write_text(raw)
                obj = parse_json(raw)
                save_json(obj, raw_json)

            index_to_cand = {start + i: c for i, c in enumerate(batch)}
            for item in iter_response_items(obj):
                try:
                    idx = int(item.get('index', -1))
                    cid = int(item.get('category_id', 0) or 0)
                except Exception:
                    continue
                if not item.get('keep', True) or idx not in index_to_cand or cid not in CLASS_NAMES:
                    continue
                cand = index_to_cand[idx]
                conf = float(item.get('confidence', item.get('score', 0.82)) or 0.82)
                score = max(.001, min(.999, conf))
                preds.append({'image_id': iid, 'category_id': cid, 'bbox': [float(v) for v in cand['bbox']], 'score': round(score, 6)})

    preds = post_filter(preds, args.final_nms, args.min_keep_score, args.max_per_image)
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
