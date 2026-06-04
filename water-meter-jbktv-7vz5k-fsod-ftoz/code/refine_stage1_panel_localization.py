#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

DS = "water-meter-jbktv-7vz5k-fsod-ftoz"
DATA = Path("/data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data") / DS
GT = DATA / "test" / "_annotations.coco.json"


def load_json(path: Path):
    return json.load(open(path))


def union_boxes(boxes):
    boxes = [b for b in boxes if b]
    if not boxes:
        return None
    return [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]


def clamp_full_width(box, image, ypad):
    if not box:
        return None
    return [0.0, round(max(0.0, box[1] - ypad), 2), float(image["width"]), round(min(float(image["height"]), box[3] + ypad), 2)]


def gt_union_by_image(gt):
    grouped = {}
    for ann in gt["annotations"]:
        iid = int(ann["image_id"])
        x, y, w, h = [float(v) for v in ann["bbox"]]
        grouped.setdefault(iid, []).append([x, y, x + w, y + h])
    return {iid: union_boxes(boxes) for iid, boxes in grouped.items()}


def contains(outer, inner, tol=0.0):
    return bool(
        outer
        and inner
        and outer[0] <= inner[0] + tol
        and outer[1] <= inner[1] + tol
        and outer[2] >= inner[2] - tol
        and outer[3] >= inner[3] - tol
    )


def evaluate(records, image_ids, out_dir):
    gt = load_json(GT)
    gt_union = gt_union_by_image(gt)
    by_id = {int(r["image_id"]): r for r in records}
    failures = []
    ok = 0
    for iid in image_ids:
        rec = by_id.get(iid)
        pred = rec.get("meter_bbox") if rec else None
        target = gt_union[iid]
        if contains(pred, target):
            ok += 1
        else:
            failures.append({"image_id": iid, "file_name": rec.get("file_name") if rec else None, "gt_union": target, "meter_bbox": pred})
    metrics = {"images": len(image_ids), "contains_all_gt": ok, "containment_rate": ok / max(1, len(image_ids)), "failures": failures}
    json.dump(metrics, open(out_dir / "panel_eval_refined.json", "w"), indent=2)
    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary-json", type=Path, required=True, help="panel_localization.json from stage1 API localization")
    parser.add_argument("--aux-json", type=Path, help="optional qwen_reads.json or another panel localization JSON")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--ypad", type=float, default=10.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gt = load_json(GT)
    images = {int(im["id"]): im for im in gt["images"]}
    primary = {int(r["image_id"]): r for r in load_json(args.primary_json)}
    aux = {}
    if args.aux_json and args.aux_json.exists():
        aux = {int(r["image_id"]): r for r in load_json(args.aux_json)}

    records = []
    for iid in sorted(images):
        im = images[iid]
        boxes = []
        sources = []
        if iid in primary and primary[iid].get("meter_bbox"):
            boxes.append(primary[iid]["meter_bbox"])
            sources.append("primary")
        if iid in aux and aux[iid].get("meter_bbox"):
            boxes.append(aux[iid]["meter_bbox"])
            sources.append("aux")
        merged = clamp_full_width(union_boxes(boxes), im, args.ypad)
        records.append({"image_id": iid, "file_name": im["file_name"], "meter_bbox": merged, "sources": sources, "ypad": args.ypad})

    json.dump(records, open(args.out_dir / "panel_localization_refined.json", "w"), indent=2)
    metrics = evaluate(records, sorted(images), args.out_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
