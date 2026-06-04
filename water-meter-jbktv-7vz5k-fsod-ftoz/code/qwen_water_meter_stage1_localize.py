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
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent
DS = "water-meter-jbktv-7vz5k-fsod-ftoz"
DATA = Path("/data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data") / DS
GT = DATA / "test" / "_annotations.coco.json"


def load_json(path: Path):
    return json.load(open(path))


def image_url(img: Image.Image, max_side: int):
    img = img.convert("RGB")
    w, h = img.size
    scale = 1.0
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=94)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), img.size, scale


def prompt(w: int, h: int) -> str:
    return f"""
You are doing STAGE 1 localization for a water-meter digit detector.
Your only job is to locate the complete odometer reading window / digit strip.

Return ONE bbox that contains EVERY visible numeric digit belonging to the water meter reading.
This bbox must cover the full left-to-right digit row, including partially cut off first/last digits and faint red/black rolling digits.
It is better for this bbox to be slightly larger than too tight. Do not crop off any digit edge.
Ignore screws, dial needles, labels, unit text, meter casing, shadows, and background.
Do not return separate digit boxes in this stage.

Target image size: {w}x{h}.
Coordinates must be 0-1000 normalized coordinates for the provided image. Use [x1,y1,x2,y2].

Return ONLY JSON:
{{
  "meter_bbox": [x1,y1,x2,y2],
  "confidence": 0.95,
  "notes": "complete digit row covered"
}}
""".strip()


def call_api(img: Image.Image, api_key: str, model: str, timeout: int, retries: int, max_side: int):
    img_url, sent_size, scale = image_url(img, max_side)
    payload = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt(*sent_size)},
                    {"type": "image_url", "image_url": {"url": img_url}},
                ],
            }
        ],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "enable_thinking": False,
        "thinking": {"type": "disabled"},
    }
    data = json.dumps(payload).encode()
    api_key = (api_key or "").strip().replace("\ufeff", "")
    last = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            data=data,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                obj = json.loads(resp.read().decode())
            return obj["choices"][0]["message"]["content"], sent_size, scale
        except urllib.error.HTTPError as exc:
            last = RuntimeError(f"HTTP {exc.code}: " + exc.read().decode(errors="replace"))
        except Exception as exc:
            last = exc
        time.sleep(2 + attempt * 2)
    raise RuntimeError(last)


def parse_json(text: str):
    text = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        return json.loads(match.group(0))


def box_xyxy(box, sent_size, scale: float, full_size, pad_px: float):
    if not box or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except Exception:
        return None
    sw, sh = sent_size
    width, height = full_size
    # Qwen-VL bboxes are 0-1000 normalized on the image sent to the API.
    x1, x2 = x1 / 1000 * sw, x2 / 1000 * sw
    y1, y2 = y1 / 1000 * sh, y2 / 1000 * sh
    if scale > 0:
        x1, x2 = x1 / scale, x2 / scale
        y1, y2 = y1 / scale, y2 / scale
    x1, y1, x2, y2 = x1 - pad_px, y1 - pad_px, x2 + pad_px, y2 + pad_px
    x1 = max(0.0, min(width - 1.0, x1))
    y1 = max(0.0, min(height - 1.0, y1))
    x2 = max(0.0, min(width, x2))
    y2 = max(0.0, min(height, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def parse_ids(spec: str, image_by):
    if spec == "all":
        return sorted(image_by)
    ids = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = map(int, part.split("-", 1))
            ids.extend(range(a, b + 1))
        else:
            ids.append(int(part))
    return [iid for iid in ids if iid in image_by]


def gt_union_by_image(gt):
    out = {}
    for ann in gt["annotations"]:
        iid = int(ann["image_id"])
        x, y, w, h = [float(v) for v in ann["bbox"]]
        out.setdefault(iid, []).append([x, y, x + w, y + h])
    return {
        iid: [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
        for iid, boxes in out.items()
    }


def contains(outer, inner, tol: float):
    return bool(
        outer
        and outer[0] <= inner[0] + tol
        and outer[1] <= inner[1] + tol
        and outer[2] >= inner[2] - tol
        and outer[3] >= inner[3] - tol
    )


def eval_panels(records, image_ids, out_dir: Path):
    gt = load_json(GT)
    gt_union = gt_union_by_image(gt)
    by_id = {int(r["image_id"]): r for r in records}
    failures = []
    ok = 0
    for iid in image_ids:
        rec = by_id.get(iid)
        pred = rec.get("meter_bbox") if rec else None
        target = gt_union[iid]
        if contains(pred, target, 0.0):
            ok += 1
        else:
            failures.append({"image_id": iid, "file_name": rec.get("file_name") if rec else None, "gt_union": target, "meter_bbox": pred})
    metrics = {
        "images": len(image_ids),
        "contains_all_gt": ok,
        "containment_rate": ok / max(1, len(image_ids)),
        "failures": failures,
    }
    json.dump(metrics, open(out_dir / "panel_eval.json", "w"), indent=2)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-ids", default="all")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "qwen_water_meter_stage1_localize")
    parser.add_argument("--model", default="qwen3.5-plus")
    parser.add_argument("--api-key", default=os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY"))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--pad-px", type=float, default=10.0, help="Safety padding applied after API localization.")
    args = parser.parse_args()
    if not args.api_key:
        raise SystemExit("set DASHSCOPE_API_KEY/QWEN_API_KEY or pass --api-key")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    gt = load_json(GT)
    image_by = {int(im["id"]): im for im in gt["images"]}
    image_ids = parse_ids(args.image_ids, image_by)
    records = []
    for iid in image_ids:
        im = image_by[iid]
        img = Image.open(DATA / "test" / im["file_name"]).convert("RGB")
        raw_path = raw_dir / f"{iid:04d}.txt"
        meta_path = raw_dir / f"{iid:04d}_meta.json"
        if args.resume and raw_path.exists() and meta_path.exists():
            raw = raw_path.read_text()
            meta = load_json(meta_path)
            sent_size = tuple(meta["sent_size"])
            scale = float(meta["scale"])
        else:
            raw, sent_size, scale = call_api(img, args.api_key, args.model, args.timeout, args.retries, args.max_side)
            raw_path.write_text(raw)
            json.dump({"sent_size": sent_size, "scale": scale, "file_name": im["file_name"]}, open(meta_path, "w"), indent=2)
        try:
            obj = parse_json(raw)
        except Exception as exc:
            print("parse_fail", iid, exc, raw[:200], flush=True)
            obj = {}
        bbox = box_xyxy(obj.get("meter_bbox"), sent_size, scale, (im["width"], im["height"]), args.pad_px)
        rec = {
            "image_id": iid,
            "file_name": im["file_name"],
            "meter_bbox": bbox,
            "confidence": float(obj.get("confidence", 0.0) or 0.0),
            "pad_px": args.pad_px,
        }
        records.append(rec)
        print("image", iid, "meter_bbox", bbox, flush=True)

    json.dump(records, open(args.out_dir / "panel_localization.json", "w"), indent=2)
    metrics = eval_panels(records, image_ids, args.out_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
