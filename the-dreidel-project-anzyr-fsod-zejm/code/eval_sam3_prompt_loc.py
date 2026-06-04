#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DS = "the-dreidel-project-anzyr-fsod-zejm"
DATA = Path("/data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data") / DS
GT = DATA / "test/_annotations.coco.json"
METRIC_KEYS = [
    "mAP", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large",
    "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large",
]


def load_json(path: Path):
    return json.load(open(path))


def mask_size(path):
    try:
        with Image.open(path) as img:
            return img.size
    except Exception:
        return None


def load_rows(pred_root: Path):
    gt = load_json(GT)
    file_to_img = {im["file_name"]: im for im in gt["images"]}
    size_cache = {}
    rows = []
    for fp in sorted(pred_root.glob("predictions_gpu*.jsonl")):
        for line in fp.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            im = file_to_img.get(os.path.basename(row.get("image_path", "")))
            if im is None:
                continue
            mp = row.get("mask_path")
            if mp not in size_cache:
                size_cache[mp] = mask_size(mp) if mp else None
            src = size_cache[mp] or (im["width"], im["height"])
            sw, sh = src
            W, H = float(im["width"]), float(im["height"])
            x1, y1, x2, y2 = [float(v) for v in row["box"]]
            x1 *= W / max(1.0, float(sw))
            x2 *= W / max(1.0, float(sw))
            y1 *= H / max(1.0, float(sh))
            y2 *= H / max(1.0, float(sh))
            x1 = max(0.0, min(W - 1.0, x1))
            y1 = max(0.0, min(H - 1.0, y1))
            x2 = max(0.0, min(W, x2))
            y2 = max(0.0, min(H, y2))
            if x2 <= x1 or y2 <= y1:
                continue
            rows.append({
                "image_id": int(im["id"]),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(row.get("score", 0.5)),
                "prompt": row.get("source_prompt") or row.get("prompt", ""),
            })
    return gt, rows


def iou(a, b):
    ax, ay, aw, ah = map(float, a)
    bx, by, bw, bh = map(float, b)
    inter = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0.0, min(ay + ah, by + bh) - max(ay, by)
    )
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def parse_ids(text, image_by):
    if text == "all":
        return sorted(image_by)
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = map(int, part.split("-", 1))
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    return [iid for iid in out if iid in image_by]


def eval_oracle(preds, image_ids):
    if not preds:
        return {k: 0.0 for k in METRIC_KEYS}
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(GT))
        dt = coco.loadRes(preds)
        ev = COCOeval(coco, dt, "bbox")
        ev.params.imgIds = image_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {k: float(v) for k, v in zip(METRIC_KEYS, ev.stats)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred-root", type=Path, required=True)
    parser.add_argument("--image-ids", default="0-29")
    parser.add_argument("--match-iou", type=float, default=0.3)
    parser.add_argument("--out-json", type=Path)
    args = parser.parse_args()

    gt, rows = load_rows(args.pred_root)
    image_by = {int(im["id"]): im for im in gt["images"]}
    classes = {int(c["id"]): c["name"] for c in gt["categories"]}
    image_ids = parse_ids(args.image_ids, image_by)
    image_id_set = set(image_ids)
    anns = [a for a in gt["annotations"] if int(a["image_id"]) in image_id_set]
    by_image = defaultdict(list)
    for row in rows:
        if row["image_id"] in image_id_set:
            by_image[row["image_id"]].append(row)

    recall = {}
    for thr in [0.3, 0.5, 0.75]:
        hit = Counter()
        total = Counter()
        best_ious = []
        for ann in anns:
            cid = int(ann["category_id"])
            total[cid] += 1
            best = max([iou(ann["bbox"], row["bbox"]) for row in by_image[ann["image_id"]]] or [0.0])
            best_ious.append(best)
            if best >= thr:
                hit[cid] += 1
        recall[str(thr)] = {
            "overall": sum(hit.values()) / max(1, len(anns)),
            "mean_best_iou": sum(best_ious) / max(1, len(best_ious)),
            "per_class": {classes[cid]: hit[cid] / total[cid] for cid in total},
        }

    by_prompt = {}
    for prompt in sorted({row["prompt"] for row in rows}):
        prompt_by_image = defaultdict(list)
        for row in rows:
            if row["image_id"] in image_id_set and row["prompt"] == prompt:
                prompt_by_image[row["image_id"]].append(row)
        hit = Counter()
        total = Counter()
        for ann in anns:
            cid = int(ann["category_id"])
            total[cid] += 1
            best = max([iou(ann["bbox"], row["bbox"]) for row in prompt_by_image[ann["image_id"]]] or [0.0])
            if best >= 0.5:
                hit[cid] += 1
        overall = sum(hit.values()) / max(1, len(anns))
        if overall > 0:
            by_prompt[prompt] = {
                "candidates": sum(len(v) for v in prompt_by_image.values()),
                "recall50": overall,
                "per_class": {classes[cid]: hit[cid] / total[cid] for cid in total if hit[cid]},
            }

    oracle = []
    for row in rows:
        if row["image_id"] not in image_id_set:
            continue
        same = [ann for ann in anns if ann["image_id"] == row["image_id"]]
        best_iou, best_ann = max(
            [(iou(ann["bbox"], row["bbox"]), ann) for ann in same],
            default=(0.0, None),
            key=lambda item: item[0],
        )
        if best_ann is not None and best_iou >= args.match_iou:
            oracle.append({
                "image_id": row["image_id"],
                "category_id": int(best_ann["category_id"]),
                "bbox": [round(float(v), 2) for v in row["bbox"]],
                "score": round(float(row["score"]), 6),
            })
    oracle_metrics = eval_oracle(oracle, image_ids)

    result = {
        "pred_root": str(args.pred_root),
        "image_ids": image_ids,
        "images": len(image_ids),
        "gt": len(anns),
        "candidates": sum(len(v) for v in by_image.values()),
        "prompt_counts": dict(Counter(row["prompt"] for row in rows if row["image_id"] in image_id_set)),
        "gt_counts": dict(Counter(classes[int(a["category_id"])] for a in anns)),
        "recall": recall,
        "by_prompt": by_prompt,
        "oracle_match_iou": args.match_iou,
        "oracle_predictions": len(oracle),
        "oracle_metrics": oracle_metrics,
    }
    if args.out_json:
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        json.dump(result, open(args.out_json, "w"), indent=2)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
