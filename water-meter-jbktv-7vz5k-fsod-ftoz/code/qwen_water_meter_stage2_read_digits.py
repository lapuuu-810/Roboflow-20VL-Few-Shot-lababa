#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parent
DS = "water-meter-jbktv-7vz5k-fsod-ftoz"
DATA = Path("/data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data") / DS
GT = DATA / "test" / "_annotations.coco.json"
METRIC_KEYS = ["mAP", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large", "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]


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
    img.save(buf, format="JPEG", quality=95)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), img.size, scale


def prompt(w: int, h: int) -> str:
    return f"""
You are doing STAGE 2 digit recognition inside an already localized water-meter reading strip crop.
The image is only the meter reading strip/panel. Read the numeric odometer digits from left to right.

Return every visible digit 0-9 in the reading row with a tight bbox around that digit within THIS CROP.
Include partially visible first/last digits if they are part of the reading.
Ignore panel borders, screws, shadows, labels, unit text, ticks, and non-digit marks.
Be conservative on digit identity, but do not omit a visible reading digit.

Crop image size: {w}x{h}.
Coordinates must be 0-1000 normalized coordinates for this crop. Use [x1,y1,x2,y2].

Return ONLY JSON:
{{
  "reading": "xxx",
  "digits": [
    {{"digit": "0", "bbox": [x1,y1,x2,y2], "confidence": 0.95}}
  ]
}}
""".strip()


def call_api(img: Image.Image, api_key: str, model: str, timeout: int, retries: int, max_side: int):
    img_url, sent_size, scale = image_url(img, max_side)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [{"type": "text", "text": prompt(*sent_size)}, {"type": "image_url", "image_url": {"url": img_url}}]}],
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


def crop_box(panel, im, pad):
    x1, y1, x2, y2 = [float(v) for v in panel]
    return [max(0.0, x1 - pad), max(0.0, y1 - pad), min(float(im["width"]), x2 + pad), min(float(im["height"]), y2 + pad)]


def local_box_to_full(box, sent_size, scale, crop_xyxy, crop_size, full_size, normalized_mode=False):
    if not box or len(box) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except Exception:
        return None
    sw, sh = sent_size
    cw, ch = crop_size
    W, H = full_size
    if normalized_mode or (not (0 <= x1 <= cw and 0 <= x2 <= cw and 0 <= y1 <= ch and 0 <= y2 <= ch) and max(x1, y1, x2, y2) <= 1000):
        x1, x2 = x1 / 1000 * sw, x2 / 1000 * sw
        y1, y2 = y1 / 1000 * sh, y2 / 1000 * sh
    if scale > 0:
        x1, x2 = x1 / scale, x2 / scale
        y1, y2 = y1 / scale, y2 / scale
    ox, oy = crop_xyxy[0], crop_xyxy[1]
    x1, x2 = x1 + ox, x2 + ox
    y1, y2 = y1 + oy, y2 + oy
    x1 = max(0.0, min(W - 1.0, x1))
    y1 = max(0.0, min(H - 1.0, y1))
    x2 = max(0.0, min(W, x2))
    y2 = max(0.0, min(H, y2))
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1, 2), round(y1, 2), round(x2, 2), round(y2, 2)]


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = box
    return [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]


def parse_ids(spec, image_by):
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


def evaluate(preds, image_ids, out_dir):
    if not preds:
        return {k: 0.0 for k in METRIC_KEYS}
    subset = load_json(GT)
    keep = set(image_ids)
    subset["images"] = [im for im in subset["images"] if int(im["id"]) in keep]
    subset["annotations"] = [ann for ann in subset["annotations"] if int(ann["image_id"]) in keep]
    subset_path = out_dir / "subset_gt.json"
    json.dump(subset, open(subset_path, "w"))
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(subset_path))
        dt = coco.loadRes(preds)
        ev = COCOeval(coco, dt, "bbox")
        ev.params.imgIds = image_ids
        ev.evaluate(); ev.accumulate(); ev.summarize()
    return {k: float(v) for k, v in zip(METRIC_KEYS, ev.stats)}


def reads_to_preds(reads):
    preds = []
    for item in reads:
        for det in item.get("digits", []):
            digit = str(det.get("digit", "")).strip()
            if not re.fullmatch(r"\d", digit):
                continue
            conf = float(det.get("confidence", 0.85) or 0.85)
            preds.append({"image_id": int(item["image_id"]), "category_id": int(digit) + 1, "bbox": xyxy_to_xywh(det["bbox"]), "score": round(max(0.01, min(0.99, conf)), 4)})
    return preds


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--panel-json", type=Path, required=True)
    parser.add_argument("--image-ids", default="all")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "qwen_water_meter_stage2_read_digits")
    parser.add_argument("--model", default="qwen3.5-plus")
    parser.add_argument("--api-key", default=os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY"))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--max-side", type=int, default=1200)
    parser.add_argument("--crop-pad", type=float, default=2.0)
    parser.add_argument(
        "--bbox-coord-mode",
        choices=["norm1000", "auto", "absolute"],
        default="norm1000",
        help="Coordinate convention for Qwen digit bboxes on the crop image.",
    )
    args = parser.parse_args()
    if not args.api_key:
        raise SystemExit("set DASHSCOPE_API_KEY/QWEN_API_KEY or pass --api-key")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out_dir / "raw"; raw_dir.mkdir(exist_ok=True)
    crop_dir = args.out_dir / "crops"; crop_dir.mkdir(exist_ok=True)

    gt = load_json(GT)
    image_by = {int(im["id"]): im for im in gt["images"]}
    panels = {int(r["image_id"]): r for r in load_json(args.panel_json)}
    image_ids = parse_ids(args.image_ids, image_by)
    reads = []
    for iid in image_ids:
        im = image_by[iid]
        panel = panels[iid]["meter_bbox"]
        full = Image.open(DATA / "test" / im["file_name"]).convert("RGB")
        cbox = crop_box(panel, im, args.crop_pad)
        crop = full.crop(tuple(round(v) for v in cbox))
        crop.save(crop_dir / f"{iid:04d}.jpg", quality=95)
        raw_path = raw_dir / f"{iid:04d}.txt"
        meta_path = raw_dir / f"{iid:04d}_meta.json"
        if args.resume and raw_path.exists() and meta_path.exists():
            raw = raw_path.read_text(); meta = load_json(meta_path); sent_size = tuple(meta["sent_size"]); scale = float(meta["scale"])
        else:
            raw, sent_size, scale = call_api(crop, args.api_key, args.model, args.timeout, args.retries, args.max_side)
            raw_path.write_text(raw)
            json.dump({"sent_size": sent_size, "scale": scale, "file_name": im["file_name"], "crop_box": cbox, "crop_size": crop.size}, open(meta_path, "w"), indent=2)
        try:
            obj = parse_json(raw)
        except Exception as exc:
            print("parse_fail", iid, exc, raw[:200], flush=True); obj = {}
        raw_digits = obj.get("digits", [])
        # Qwen digit bboxes are expected to be 0-1000 normalized on the crop image.
        # Keep auto/absolute modes only for debugging old cache variants.
        normalized_mode = args.bbox_coord_mode == "norm1000"
        if args.bbox_coord_mode == "auto":
            for det in raw_digits:
                b = det.get("bbox")
                if b and len(b) == 4:
                    try:
                        x1, y1, x2, y2 = [float(v) for v in b]
                    except Exception:
                        continue
                    if max(x1, y1, x2, y2) <= 1000 and not (0 <= x1 <= crop.width and 0 <= x2 <= crop.width and 0 <= y1 <= crop.height and 0 <= y2 <= crop.height):
                        normalized_mode = True
                        break
        digits = []
        for det in raw_digits:
            digit = str(det.get("digit", "")).strip()
            if not re.fullmatch(r"\d", digit):
                continue
            box = local_box_to_full(det.get("bbox"), sent_size, scale, cbox, crop.size, (im["width"], im["height"]), normalized_mode=normalized_mode)
            if box:
                digits.append({"digit": digit, "bbox": box, "confidence": float(det.get("confidence", 0.85) or 0.85)})
        digits.sort(key=lambda d: (d["bbox"][0] + d["bbox"][2]) / 2)
        rec = {"image_id": iid, "file_name": im["file_name"], "meter_bbox": panel, "crop_box": cbox, "reading": "".join(d["digit"] for d in digits) or str(obj.get("reading", "")), "digits": digits}
        reads.append(rec)
        print("image", iid, "digits", len(digits), "reading", rec["reading"], flush=True)

    json.dump(reads, open(args.out_dir / "qwen_stage2_reads.json", "w"), indent=2)
    preds = reads_to_preds(reads)
    json.dump(preds, open(args.out_dir / "predictions_qwen_stage2.json", "w"), indent=2)
    metrics = evaluate(preds, image_ids, args.out_dir)
    metrics.update({"images": len(image_ids), "predictions": len(preds), "image_ids": image_ids})
    json.dump(metrics, open(args.out_dir / "eval_qwen_stage2.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
