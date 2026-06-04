#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DS = "lacrosse-object-detection-fsod-uxkt"
ROOT = Path(__file__).resolve().parent
DATA = Path("/data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data") / DS
GT = DATA / "test" / "_annotations.coco.json"
FALLBACK = Path("/data/LPP/cvpr/few_shot/sub/sam3_foundational_submission") / f"{DS}.pkl"
KEYS = ["mAP", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large", "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
CLASS_NAMES = {1: "Goalie", 2: "Longpole", 3: "Referee", 4: "Shortstick"}


def load_json(path: Path):
    return json.load(open(path))


def save_json(obj, path: Path):
    json.dump(obj, open(path, "w"), indent=2, ensure_ascii=False)


def load_fallback(path: Path):
    data = pickle.load(open(path, "rb"))
    rows = data if isinstance(data, list) else [{"image_id": k, "instances": v} for k, v in data.items()]
    preds = []
    for row in rows:
        iid = int(row["image_id"])
        for inst in row.get("instances", []):
            preds.append({
                "image_id": iid,
                "category_id": int(inst["category_id"]) + 1,
                "bbox": [round(float(x), 2) for x in inst["bbox"]],
                "score": float(inst.get("score", 0.5)),
            })
    return preds


def xyxy(p):
    x, y, w, h = map(float, p["bbox"])
    return [x, y, x + w, y + h]


def iou(a, b):
    a = xyxy(a)
    b = xyxy(b)
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


def nms(preds, thr: float):
    groups = defaultdict(list)
    for p in preds:
        groups[(int(p["image_id"]), int(p["category_id"]))].append(p)
    out = []
    for rows in groups.values():
        keep = []
        for p in sorted(rows, key=lambda r: float(r["score"]), reverse=True):
            if all(iou(p, q) < thr for q in keep):
                keep.append(p)
        out.extend(keep)
    return sorted(out, key=lambda p: (int(p["image_id"]), -float(p["score"])))


def evaluate(preds):
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


def write_pkl(preds, path: Path):
    gt = load_json(GT)
    by = defaultdict(list)
    for p in preds:
        by[int(p["image_id"])].append({
            "image_id": int(p["image_id"]),
            "category_id": int(p["category_id"]) - 1,
            "bbox": np.array(p["bbox"], dtype=np.float32),
            "score": float(p["score"]),
        })
    sub = [{"image_id": int(im["id"]), "instances": by.get(int(im["id"]), [])} for im in gt["images"]]
    pickle.dump(sub, open(path, "wb"), protocol=4)


def visualize(preds, out_dir: Path, count: int):
    gt = load_json(GT)
    image_by = {int(im["id"]): im for im in gt["images"]}
    by = defaultdict(list)
    for p in preds:
        by[int(p["image_id"])].append(p)
    colors = {1: (30, 140, 230), 2: (230, 150, 20), 3: (220, 40, 40), 4: (60, 180, 80)}
    vis = out_dir / "vis"
    vis.mkdir(exist_ok=True)
    thumbs = []
    for iid in sorted(image_by)[:count]:
        im = image_by[iid]
        img = Image.open(DATA / "test" / im["file_name"]).convert("RGB")
        draw = ImageDraw.Draw(img)
        for p in by.get(iid, [])[:100]:
            x, y, w, h = p["bbox"]
            color = colors.get(int(p["category_id"]), (255, 0, 255))
            draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
            draw.text((x, max(0, y - 14)), f"{CLASS_NAMES[int(p['category_id'])]} {p['score']:.2f}", fill=color)
        img.thumbnail((900, 900))
        img.save(vis / f"{iid:04d}.jpg", quality=92)
        tile = img.copy()
        tile.thumbnail((320, 220))
        thumbs.append((iid, tile))
    if not thumbs:
        return
    cols = 4
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 320, rows * 245), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for i, (iid, tile) in enumerate(thumbs):
        x = (i % cols) * 320
        y = (i // cols) * 245
        draw.text((x + 4, y + 4), f"image {iid}", fill=(0, 0, 0))
        sheet.paste(tile, (x, y + 22))
    sheet.save(out_dir / "contact_sheet.jpg", quality=92)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-preds", type=Path, required=True)
    parser.add_argument("--fallback-pkl", type=Path, default=FALLBACK)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "qwen35_fallback_fuse_best")
    parser.add_argument("--qwen-scale", type=float, default=0.85)
    parser.add_argument("--fallback-scale", type=float, default=0.65)
    parser.add_argument("--nms-thr", type=float, default=0.65)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--vis-count", type=int, default=24)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    qwen = load_json(args.qwen_preds)
    fallback = load_fallback(args.fallback_pkl)
    preds = []
    for p in qwen:
        pp = dict(p)
        pp["score"] = round(max(0.001, min(0.999, float(pp["score"]) * args.qwen_scale)), 6)
        preds.append(pp)
    for p in fallback:
        pp = dict(p)
        pp["score"] = round(max(0.001, min(0.999, float(pp["score"]) * args.fallback_scale)), 6)
        preds.append(pp)
    preds = nms(preds, args.nms_thr)
    metrics = evaluate(preds)
    metrics.update({
        "images": len(load_json(GT)["images"]),
        "predictions": len(preds),
        "source_qwen": len(qwen),
        "source_fallback": len(fallback),
        "counts": dict(Counter(int(p["category_id"]) for p in preds)),
        "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()},
    })
    save_json(preds, args.out_dir / "predictions.json")
    save_json(metrics, args.out_dir / "eval.json")
    write_pkl(preds, args.out_dir / f"{DS}.pkl")
    if args.vis:
        visualize(preds, args.out_dir, args.vis_count)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
