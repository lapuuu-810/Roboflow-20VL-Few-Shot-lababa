#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import qwen_waste_direct_api as base


def xyxy(pred):
    x, y, w, h = map(float, pred["bbox"])
    return [x, y, x + w, y + h]


def box_iou(a, b):
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


def parse_allowed(text: str):
    if text.strip().lower() in {"all", "*"}:
        return None
    return {int(x) for x in text.split(",") if x.strip()}


def filter_tile(full, tile, image_by, args):
    allowed = parse_allowed(args.allowed_classes)
    out = []
    for pred in tile:
        cid = int(pred["category_id"])
        if allowed is not None and cid not in allowed:
            continue
        im = image_by[int(pred["image_id"])]
        width, height = float(im["width"]), float(im["height"])
        x, y, w, h = map(float, pred["bbox"])
        if (w * h) / (width * height) < args.min_area_frac:
            continue
        if min(w / width, h / height) < args.min_side_frac:
            continue
        pbox = xyxy(pred)
        same_iou = max(
            [box_iou(pbox, xyxy(old)) for old in full if int(old["image_id"]) == int(pred["image_id"]) and int(old["category_id"]) == cid],
            default=0.0,
        )
        any_iou = max(
            [box_iou(pbox, xyxy(old)) for old in full if int(old["image_id"]) == int(pred["image_id"])],
            default=0.0,
        )
        if same_iou > args.max_full_iou_same or any_iou > args.max_full_iou_any:
            continue
        item = {k: v for k, v in pred.items() if k != "source"}
        item["score"] = round(float(item.get("score", args.default_score)) * args.tile_score_scale, 6)
        out.append(item)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full-preds", type=Path, required=True)
    parser.add_argument("--tile-preds", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--image-ids", default=None)
    parser.add_argument("--allowed-classes", default="all")
    parser.add_argument("--max-full-iou-same", type=float, default=0.15)
    parser.add_argument("--max-full-iou-any", type=float, default=0.25)
    parser.add_argument("--min-area-frac", type=float, default=0.0)
    parser.add_argument("--min-side-frac", type=float, default=0.06)
    parser.add_argument("--nms-thr", type=float, default=0.45)
    parser.add_argument("--tile-score-scale", type=float, default=1.0)
    parser.add_argument("--default-score", type=float, default=0.70)
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gt = base.load_json(base.GT)
    image_by = {int(im["id"]): im for im in gt["images"]}
    all_ids = sorted(image_by)
    image_ids = base.parse_image_ids(args.image_ids, all_ids) if args.image_ids else sorted({int(p["image_id"]) for p in base.load_json(args.full_preds)} | {int(p["image_id"]) for p in base.load_json(args.tile_preds)})
    keep_ids = set(image_ids)
    full = [p for p in base.load_json(args.full_preds) if int(p["image_id"]) in keep_ids]
    tile = [p for p in base.load_json(args.tile_preds) if int(p["image_id"]) in keep_ids]
    tile_keep = filter_tile(full, tile, image_by, args)
    fused = base.nms(full + tile_keep, args.nms_thr)

    base.save_json(tile_keep, args.out_dir / "tile_kept.json")
    base.save_json(fused, args.out_dir / "predictions.json")
    metrics = base.evaluate(fused, args.out_dir, image_ids)
    metrics.update({
        "images": len(image_ids),
        "image_ids": image_ids,
        "full_predictions": len(full),
        "tile_predictions": len(tile),
        "tile_kept": len(tile_keep),
        "predictions": len(fused),
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "counts": dict(Counter(int(p["category_id"]) for p in fused)),
    })
    base.save_json(metrics, args.out_dir / "eval.json")
    base.write_pkl(fused, image_by, args.out_dir / f"{base.DS}.pkl")
    if args.vis:
        base.visualize(fused, image_by, args.out_dir, 24)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
