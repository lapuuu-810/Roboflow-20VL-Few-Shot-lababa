#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pickle
import shutil
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


DS = "soda-bottles-fsod-haga"
COLOR_DS = "soda-bottles-color-prompt-sam3"
ROOT = Path("/data/LPP/cvpr/few_shot/sam3-main/best_sam3") / DS
SAM3_ROOT = Path("/data/LPP/cvpr/few_shot/sam3-main")
DATA = Path("/data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data") / DS
GT = DATA / "test/_annotations.coco.json"
KEYS = ["mAP", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large", "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
COLOR_PROMPTS = {
    1: "red soda bottle",
    2: "orange soda bottle",
    3: "green soda bottle",
}


def load_json(path: Path):
    return json.load(open(path))


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = map(float, box)
    return [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]


def xywh_to_xyxy(box):
    x, y, w, h = map(float, box)
    return [x, y, x + w, y + h]


def iou(a, b):
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


def expand_xyxy(box, sx, sy, width, height):
    x1, y1, x2, y2 = map(float, box)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = (x2 - x1) * sx
    h = (y2 - y1) * sy
    out = [max(0.0, cx - w / 2.0), max(0.0, cy - h / 2.0), min(float(width), cx + w / 2.0), min(float(height), cy + h / 2.0)]
    return None if out[2] <= out[0] or out[3] <= out[1] else out


def prepare_color_dataset(args):
    color_root = args.work_root / COLOR_DS
    test_dir = color_root / "test"
    if color_root.exists() and not args.preserve_work:
        shutil.rmtree(color_root)
    test_dir.mkdir(parents=True, exist_ok=True)

    gt = load_json(GT)
    category_map = {}
    for cat in gt["categories"]:
        cid = int(cat["id"])
        category_map[cid] = {"original_name": cat["name"], "prompt_name": COLOR_PROMPTS.get(cid, cat["name"])}
        if cid in COLOR_PROMPTS:
            cat["name"] = COLOR_PROMPTS[cid]
            cat["supercategory"] = "color soda bottle"

    for im in gt["images"]:
        src = DATA / "test" / im["file_name"]
        dst = test_dir / im["file_name"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        if not dst.exists():
            try:
                os.symlink(src, dst)
            except OSError:
                shutil.copy2(src, dst)

    json.dump(gt, open(test_dir / "_annotations.coco.json", "w"), indent=2)
    json.dump(category_map, open(color_root / "color_prompt_category_map.json", "w"), indent=2)
    return args.work_root, category_map


def run_sam3(args, data_root):
    if args.skip_sam3:
        return
    cmd = [
        sys.executable,
        str(SAM3_ROOT / "scripts/batch_segment_data_test.py"),
        "--data-root",
        str(data_root),
        "--sam3-root",
        str(SAM3_ROOT),
        "--checkpoint",
        str(args.checkpoint),
        "--output",
        str(args.sam3_output_parent),
        "--gpus",
        args.gpus,
        "--batch-size",
        str(args.batch_size),
        "--threshold",
        str(args.threshold),
        "--mode",
        "categories",
        "--num-workers",
        str(args.num_workers),
        "--max-dets-per-query",
        str(args.max_dets_per_query),
    ]
    if args.preserve_output:
        cmd.append("--preserve-output")
    subprocess.run(cmd, check=True)


def load_rows(pred_dir: Path):
    gt = load_json(GT)
    file_to_img = {im["file_name"]: im for im in gt["images"]}
    rows = []
    missing = Counter()
    for fp in sorted(pred_dir.glob("predictions_gpu*.jsonl")):
        for line in open(fp):
            if not line.strip():
                continue
            rec = json.loads(line)
            im = file_to_img.get(os.path.basename(rec.get("image_path", "")))
            if im is None:
                missing["image"] += 1
                continue
            width, height = float(im["width"]), float(im["height"])
            x1, y1, x2, y2 = map(float, rec["box"])
            box = [max(0.0, min(width - 1.0, x1)), max(0.0, min(height - 1.0, y1)), max(0.0, min(width, x2)), max(0.0, min(height, y2))]
            if box[2] <= box[0] or box[3] <= box[1]:
                missing["bad_box"] += 1
                continue
            cid = int(rec.get("category_id", 0))
            if cid not in COLOR_PROMPTS:
                continue
            rows.append({"image_id": int(im["id"]), "category_id": cid, "bbox_xyxy": box, "score": float(rec.get("score", 0.0)), "prompt": rec.get("prompt", ""), "source_prompt": rec.get("source_prompt", "")})
    return gt, rows, missing


def nms(rows, thr, class_agnostic=False):
    groups = defaultdict(list)
    for row in rows:
        groups[0 if class_agnostic else int(row["category_id"])].append(row)
    kept = []
    for items in groups.values():
        accepted = []
        for row in sorted(items, key=lambda r: float(r["score"]), reverse=True):
            if all(iou(row["bbox_xyxy"], prev["bbox_xyxy"]) < thr for prev in accepted):
                accepted.append(row)
        kept.extend(accepted)
    return kept


def build_predictions(rows, image_by, args):
    by_image = defaultdict(list)
    raw_kept = []
    for row in rows:
        if float(row["score"]) < args.score_thr:
            continue
        im = image_by[int(row["image_id"])]
        box = expand_xyxy(row["bbox_xyxy"], args.scale_x, args.scale_y, im["width"], im["height"])
        if not box:
            continue
        w = box[2] - box[0]
        h = box[3] - box[1]
        area = w * h
        ar = w / max(h, 1e-9)
        width_frac = w / float(im["width"])
        height_frac = h / float(im["height"])
        area_frac = area / (float(im["width"]) * float(im["height"]))
        if width_frac < args.min_w_frac or height_frac < args.min_h_frac or area_frac < args.min_area_frac:
            continue
        if not (args.min_w <= w <= args.max_w and args.min_h <= h <= args.max_h):
            continue
        if not (args.min_area <= area <= args.max_area and args.min_ar <= ar <= args.max_ar):
            continue
        item = {**row, "bbox_xyxy": box}
        by_image[int(row["image_id"])].append(item)
        raw_kept.append(item)

    preds = []
    for iid, items in by_image.items():
        kept = nms(items, args.nms_thr, args.class_agnostic_nms)
        kept = sorted(kept, key=lambda r: float(r["score"]), reverse=True)
        if args.topk_image > 0:
            kept = kept[: args.topk_image]
        for row in kept:
            preds.append({"image_id": iid, "category_id": int(row["category_id"]), "bbox": xyxy_to_xywh(row["bbox_xyxy"]), "score": round(float(row["score"]) * args.score_scale, 6)})
    return sorted(preds, key=lambda p: (int(p["image_id"]), -float(p["score"]))), raw_kept


def evaluate(preds, out_dir):
    if not preds:
        return {k: 0.0 for k in KEYS}
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(GT))
        dt = coco.loadRes(preds)
        ev = COCOeval(coco, dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {k: float(v) for k, v in zip(KEYS, ev.stats)}


def write_pkl(preds, image_by, path):
    by = defaultdict(list)
    for pred in preds:
        by[int(pred["image_id"])].append({"image_id": int(pred["image_id"]), "category_id": int(pred["category_id"]) - 1, "bbox": np.array(pred["bbox"], dtype=np.float32), "score": float(pred["score"])})
    sub = [{"image_id": int(im["id"]), "instances": by.get(int(im["id"]), [])} for im in image_by.values()]
    pickle.dump(sub, open(path, "wb"), protocol=4)


def visualize(preds, out_dir, limit=24):
    gt = load_json(GT)
    image_by = {int(im["id"]): im for im in gt["images"]}
    by = defaultdict(list)
    for pred in preds:
        by[int(pred["image_id"])].append(pred)
    vis = out_dir / "vis"
    vis.mkdir(parents=True, exist_ok=True)
    colors = {1: (220, 30, 30), 2: (250, 140, 20), 3: (40, 180, 80)}
    paths = []
    for iid in sorted(image_by)[:limit]:
        im = image_by[iid]
        img = Image.open(DATA / "test" / im["file_name"]).convert("RGB")
        draw = ImageDraw.Draw(img)
        for pred in by.get(iid, [])[:160]:
            x, y, w, h = pred["bbox"]
            color = colors.get(int(pred["category_id"]), (30, 130, 240))
            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
            draw.text((x, max(0, y - 12)), f"{pred['category_id']}:{pred['score']:.2f}", fill=color)
        img.thumbnail((900, 900))
        out_path = vis / f"{iid:04d}.jpg"
        img.save(out_path, quality=92)
        paths.append(out_path)

    thumbs = []
    for path in paths:
        img = Image.open(path).convert("RGB")
        img.thumbnail((300, 220))
        thumbs.append((path.stem, img.copy()))
    if not thumbs:
        return
    cols = 4
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 300, rows * 245), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for idx, (name, img) in enumerate(thumbs):
        x = (idx % cols) * 300
        y = (idx // cols) * 245
        draw.text((x + 4, y + 4), f"image {name}", fill=(0, 0, 0))
        sheet.paste(img, (x, y + 22))
    sheet.save(out_dir / "sample_contact_sheet.jpg", quality=92)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work-root", type=Path, default=SAM3_ROOT / "work_dirs/soda_bottles_color_prompt_data")
    parser.add_argument("--sam3-output-parent", type=Path, default=SAM3_ROOT / "sam3_soda_bottles_color_prompt_results")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "sam3_color_prompt")
    parser.add_argument("--checkpoint", type=Path, default=SAM3_ROOT / "sam3_weight/sam3.pt")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--max-dets-per-query", type=int, default=-1)
    parser.add_argument("--score-thr", type=float, default=0.0)
    parser.add_argument("--nms-thr", type=float, default=0.6)
    parser.add_argument("--topk-image", type=int, default=40)
    parser.add_argument("--class-agnostic-nms", action="store_true")
    parser.add_argument("--scale-x", type=float, default=1.0)
    parser.add_argument("--scale-y", type=float, default=1.0)
    parser.add_argument("--score-scale", type=float, default=1.0)
    parser.add_argument("--min-w", type=float, default=1.0)
    parser.add_argument("--max-w", type=float, default=1e9)
    parser.add_argument("--min-h", type=float, default=1.0)
    parser.add_argument("--max-h", type=float, default=1e9)
    parser.add_argument("--min-area", type=float, default=1.0)
    parser.add_argument("--max-area", type=float, default=1e12)
    parser.add_argument("--min-w-frac", type=float, default=0.0)
    parser.add_argument("--min-h-frac", type=float, default=0.0)
    parser.add_argument("--min-area-frac", type=float, default=0.0)
    parser.add_argument("--min-ar", type=float, default=0.05)
    parser.add_argument("--max-ar", type=float, default=5.0)
    parser.add_argument("--skip-sam3", action="store_true")
    parser.add_argument("--preserve-output", action="store_true")
    parser.add_argument("--preserve-work", action="store_true")
    parser.add_argument("--vis", action="store_true")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data_root, category_map = prepare_color_dataset(args)
    run_sam3(args, data_root)

    gt, rows, missing = load_rows(args.sam3_output_parent / COLOR_DS)
    image_by = {int(im["id"]): im for im in gt["images"]}
    preds, raw_kept = build_predictions(rows, image_by, args)
    json.dump(preds, open(args.out_dir / "predictions.json", "w"), indent=2)
    json.dump(raw_kept, open(args.out_dir / "raw_kept_records.json", "w"), indent=2)
    metrics = evaluate(preds, args.out_dir)
    metrics.update({
        "images": len(image_by),
        "predictions": len(preds),
        "method": "sam3_color_prompt_mapped_to_original_classes",
        "category_map": category_map,
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
        "missing": dict(missing),
        "counts": dict(Counter(int(p["category_id"]) for p in preds)),
    })
    json.dump(metrics, open(args.out_dir / "eval.json", "w"), indent=2)
    write_pkl(preds, image_by, args.out_dir / f"{DS}.pkl")
    if args.vis:
        visualize(preds, args.out_dir)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
