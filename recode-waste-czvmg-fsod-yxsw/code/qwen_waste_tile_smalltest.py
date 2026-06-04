#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from PIL import Image

import qwen_waste_direct_api as base


def make_tiles(width: int, height: int, grid: int, overlap: float):
    tiles = []
    step_x = width / grid
    step_y = height / grid
    pad_x = step_x * overlap / 2.0
    pad_y = step_y * overlap / 2.0
    for gy in range(grid):
        for gx in range(grid):
            x1 = max(0, int(round(gx * step_x - pad_x)))
            y1 = max(0, int(round(gy * step_y - pad_y)))
            x2 = min(width, int(round((gx + 1) * step_x + pad_x)))
            y2 = min(height, int(round((gy + 1) * step_y + pad_y)))
            tiles.append((f"tile{grid}x{grid}_{gy}_{gx}", [x1, y1, x2, y2]))
    return tiles


def run_tile_qwen(args):
    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
    if not api_key:
        raise SystemExit("Set DASHSCOPE_API_KEY/QWEN_API_KEY or pass --api-key")
    gt = base.load_json(base.GT)
    image_by = {int(im["id"]): im for im in gt["images"]}
    ids = base.parse_image_ids(args.image_ids, sorted(image_by))
    raw_dir = args.out_dir / "raw_tiles"
    raw_dir.mkdir(parents=True, exist_ok=True)
    records_path = args.out_dir / "qwen_tile_records.json"
    existing = {}
    if args.resume and records_path.exists():
        for rec in base.load_json(records_path):
            existing[(int(rec["image_id"]), rec["view_name"])] = rec
    records = []
    total = len(ids) * args.grid * args.grid
    done_count = 0
    for iid in ids:
        im = image_by[iid]
        img = Image.open(base.DATA / "test" / im["file_name"]).convert("RGB")
        for view_name, crop_box in make_tiles(img.width, img.height, args.grid, args.overlap):
            key = (iid, view_name)
            done_count += 1
            if key in existing:
                records.append(existing[key])
                continue
            crop = img.crop(tuple(crop_box))
            raw, sent_size = base.call_api(crop, api_key, args.model, args.timeout, args.retries, args.max_side)
            rec = {"image_id": iid, "file_name": im["file_name"], "view_name": view_name, "crop_box": crop_box, "raw": raw, "sent_size": list(sent_size)}
            try:
                rec["parsed"] = base.parse_json(raw)
            except Exception as e:
                rec["parse_error"] = repr(e)
                rec["parsed"] = {"detections": []}
            records.append(rec)
            base.save_json(rec, raw_dir / f"{iid:04d}_{view_name}.json")
            base.save_json(records, records_path)
            print(f"[{done_count}/{total}] image {iid} {view_name}: raw chars={len(raw)}", flush=True)
    base.save_json(records, records_path)
    return records, image_by


def norm1000_tile_to_full_xywh(box, crop_box, image_width: int, image_height: int):
    if not isinstance(box, list) or len(box) != 4:
        return None
    ox1, oy1, ox2, oy2 = [float(v) for v in crop_box]
    cw = ox2 - ox1
    ch = oy2 - oy1
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = ox1 + x1 / 1000.0 * cw
    x2 = ox1 + x2 / 1000.0 * cw
    y1 = oy1 + y1 / 1000.0 * ch
    y2 = oy1 + y2 / 1000.0 * ch
    if x2 <= x1 or y2 <= y1:
        return None
    x1 = max(0.0, min(image_width - 1.0, x1))
    y1 = max(0.0, min(image_height - 1.0, y1))
    x2 = max(x1 + 1.0, min(float(image_width), x2))
    y2 = max(y1 + 1.0, min(float(image_height), y2))
    return [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]


def convert_tile_records(records, image_by, args):
    preds = []
    audit = []
    for rec in records:
        iid = int(rec["image_id"])
        im = image_by[iid]
        parsed = rec.get("parsed") or {}
        if isinstance(parsed, list):
            dets = parsed
        elif isinstance(parsed, dict):
            dets = parsed.get("detections", [])
        else:
            dets = []
        if not isinstance(dets, list):
            dets = []
        for det in dets:
            cid = base.label_to_id(det.get("label", det.get("category_name", det.get("category"))))
            if cid is None:
                audit.append({"image_id": iid, "view_name": rec.get("view_name"), "skip": "unknown_label", "det": det})
                continue
            bbox = norm1000_tile_to_full_xywh(det.get("bbox_2d", det.get("bbox")), rec["crop_box"], int(im["width"]), int(im["height"]))
            if not bbox:
                audit.append({"image_id": iid, "view_name": rec.get("view_name"), "skip": "bad_box", "det": det})
                continue
            score = float(det.get("confidence", det.get("score", args.default_score)))
            if score < args.score_thr:
                continue
            preds.append({"image_id": iid, "category_id": cid, "bbox": bbox, "score": round(max(0.001, min(0.999, score)), 6), "source": rec.get("view_name")})
    preds = base.nms(preds, args.nms_thr)
    return preds, audit


def load_full_preds(path: Path, image_ids: set[int]):
    if not path:
        return []
    if not path.exists():
        raise FileNotFoundError(f"--full-preds does not exist: {path}")
    rows = base.load_json(path)
    return [p for p in rows if int(p["image_id"]) in image_ids]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=base.ROOT / "qwen_tile_smalltest")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="qwen3.5-plus")
    parser.add_argument("--image-ids", default=None)
    parser.add_argument("--grid", type=int, default=2)
    parser.add_argument("--overlap", type=float, default=0.12)
    parser.add_argument("--max-side", type=int, default=1200)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-api", action="store_true")
    parser.add_argument("--score-thr", type=float, default=0.0)
    parser.add_argument("--default-score", type=float, default=0.70)
    parser.add_argument("--nms-thr", type=float, default=0.45)
    parser.add_argument("--full-preds", type=Path, default=None)
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    gt = base.load_json(base.GT)
    image_by = {int(im["id"]): im for im in gt["images"]}
    if args.skip_api:
        records = base.load_json(args.out_dir / "qwen_tile_records.json")
    else:
        records, image_by = run_tile_qwen(args)
    tile_preds, audit = convert_tile_records(records, image_by, args)
    image_ids = sorted({int(r["image_id"]) for r in records})
    base.save_json(tile_preds, args.out_dir / "predictions_tile_only.json")
    tile_metrics = base.evaluate(tile_preds, args.out_dir, image_ids)
    tile_metrics.update({"images": len(image_ids), "records": len(records), "predictions": len(tile_preds), "image_ids": image_ids, "counts": dict(Counter(int(p["category_id"]) for p in tile_preds))})
    base.save_json(tile_metrics, args.out_dir / "eval_tile_only.json")

    full_preds = load_full_preds(args.full_preds, set(image_ids)) if args.full_preds else []
    fused_preds = base.nms(full_preds + tile_preds, args.nms_thr)
    base.save_json(fused_preds, args.out_dir / "predictions_fused.json")
    fused_metrics = base.evaluate(fused_preds, args.out_dir, image_ids)
    fused_metrics.update({"images": len(image_ids), "records": len(records), "predictions": len(fused_preds), "image_ids": image_ids, "full_predictions": len(full_preds), "tile_predictions": len(tile_preds), "counts": dict(Counter(int(p["category_id"]) for p in fused_preds))})
    base.save_json(fused_metrics, args.out_dir / "eval_fused.json")
    base.save_json(audit, args.out_dir / "audit.json")
    base.write_pkl(fused_preds, image_by, args.out_dir / f"{base.DS}.pkl")
    if args.vis:
        base.visualize(fused_preds, image_by, args.out_dir, 24)
    print(json.dumps({"tile_only": tile_metrics, "fused": fused_metrics}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
