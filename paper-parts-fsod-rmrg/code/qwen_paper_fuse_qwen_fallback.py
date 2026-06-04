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

DS = "paper-parts-fsod-rmrg"
ROOT = Path(__file__).resolve().parent
DATA = Path("/data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data") / DS
GT = DATA / "test" / "_annotations.coco.json"
FALLBACK = Path("/data/LPP/cvpr/few_shot/sub/sam3_foundational_submission") / f"{DS}.pkl"
KEYS = ["mAP", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large", "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]


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


def scale_box(bbox, width: int, height: int, scale: float):
    x, y, w, h = map(float, bbox)
    cx, cy = x + w / 2, y + h / 2
    nw, nh = w * scale, h * scale
    x1 = max(0.0, cx - nw / 2)
    y1 = max(0.0, cy - nh / 2)
    x2 = min(float(width), cx + nw / 2)
    y2 = min(float(height), cy + nh / 2)
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]


def scale_predictions(preds, scale: float):
    gt = load_json(GT)
    image_by = {int(im["id"]): im for im in gt["images"]}
    out = []
    for p in preds:
        im = image_by[int(p["image_id"])]
        bbox = scale_box(p["bbox"], int(im["width"]), int(im["height"]), scale)
        if bbox:
            q = dict(p)
            q["bbox"] = bbox
            out.append(q)
    return out


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
    cats = {int(c["id"]): c["name"] for c in gt["categories"]}
    image_by = {int(im["id"]): im for im in gt["images"]}
    by = defaultdict(list)
    for p in preds:
        by[int(p["image_id"])].append(p)
    colors = [(230, 25, 75), (60, 180, 75), (255, 225, 25), (0, 130, 200), (245, 130, 48), (145, 30, 180), (70, 240, 240), (240, 50, 230)]
    vis = out_dir / "vis"
    vis.mkdir(exist_ok=True)
    thumbs = []
    for iid in sorted(image_by)[:count]:
        im = image_by[iid]
        img = Image.open(DATA / "test" / im["file_name"]).convert("RGB")
        draw = ImageDraw.Draw(img)
        for p in by.get(iid, [])[:120]:
            x, y, w, h = p["bbox"]
            cid = int(p["category_id"])
            color = colors[(cid - 1) % len(colors)]
            draw.rectangle([x, y, x + w, y + h], outline=color, width=3)
            draw.text((x, max(0, y - 14)), f"{cid}:{cats.get(cid, cid)} {p['score']:.2f}", fill=color)
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
    parser.add_argument("--doubao-preds", type=Path)
    parser.add_argument("--qwen3vl-preds", type=Path, required=True)
    parser.add_argument("--qwen36-preds", type=Path, required=True)
    parser.add_argument("--qwen35-preds", type=Path, required=True)
    parser.add_argument("--fallback-pkl", type=Path, default=FALLBACK)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "qwen_all_api_fallback_fuse_scale_sweep")
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--vis-count", type=int, default=24)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    sources = {}
    if args.doubao_preds:
        sources["doubao"] = load_json(args.doubao_preds)
    sources.update({
        "qwen3vl": load_json(args.qwen3vl_preds),
        "qwen36": load_json(args.qwen36_preds),
        "qwen35": load_json(args.qwen35_preds),
        "fallback": load_fallback(args.fallback_pkl),
    })
    single = {name: evaluate(rows) for name, rows in sources.items()}
    for name, rows in sources.items():
        print(name, len(rows), single[name]["mAP"], single[name]["mAP50"], single[name]["mAP75"], flush=True)

    results = []
    doubao_grid = [0.0] if "doubao" not in sources else [1.0, 1.15]
    qwen3vl_grid = [0.0, 0.05] if "doubao" in sources else [0.8, 0.9, 1.0, 1.1]
    qwen36_grid = [0.0, 0.05, 0.15]
    qwen35_grid = [0.0, 0.05]
    fallback_grid = [0.0, 0.02, 0.05]
    nms_grid = [0.35, 0.45, 0.55]
    for sdoubao in doubao_grid:
        for s3vl in qwen3vl_grid:
            for s36 in qwen36_grid:
                for s35 in qwen35_grid:
                    for sfb in fallback_grid:
                        for nthr in nms_grid:
                            scales = {"qwen3vl": s3vl, "qwen36": s36, "qwen35": s35, "fallback": sfb}
                            if "doubao" in sources:
                                scales["doubao"] = sdoubao
                            preds = []
                            for name, rows in sources.items():
                                scale = scales[name]
                                if scale <= 0:
                                    continue
                                for p in rows:
                                    pp = dict(p)
                                    pp["score"] = round(max(0.001, min(0.999, float(pp["score"]) * scale)), 6)
                                    preds.append(pp)
                            preds = nms(preds, nthr)
                            metrics = evaluate(preds)
                            results.append((metrics["mAP"], metrics["mAP50"], metrics["mAP75"], len(preds), scales, nthr, metrics, preds))
    results.sort(reverse=True, key=lambda x: x[0])
    for row in results[:15]:
        print("fuse_best", row[:4], row[4], row[5], flush=True)

    best = results[0]
    fused = best[7]
    scale_rows = []
    for scale in [0.90, 0.94, 0.98, 1.0, 1.02, 1.04, 1.05, 1.06, 1.08, 1.10, 1.14]:
        scaled = scale_predictions(fused, scale)
        metrics = evaluate(scaled)
        rec = {"scale": scale, "predictions": len(scaled), **metrics}
        scale_rows.append(rec)
        print("scale", scale, rec["mAP"], rec["mAP50"], rec["mAP75"], flush=True)
    scale_rows.sort(key=lambda r: r["mAP"], reverse=True)
    best_scale = scale_rows[0]["scale"]
    final = scale_predictions(fused, best_scale)
    metrics = evaluate(final)
    metrics.update({
        "images": len(load_json(GT)["images"]),
        "predictions": len(final),
        "scale": best_scale,
        "fuse_mAP_before_scale": best[0],
        "fuse_mAP50_before_scale": best[1],
        "fuse_mAP75_before_scale": best[2],
        "scales": best[4],
        "nms_thr": best[5],
        "source_predictions": {k: len(v) for k, v in sources.items()},
        "single_metrics": {k: {kk: v[kk] for kk in ["mAP", "mAP50", "mAP75"]} for k, v in single.items()},
        "counts": dict(Counter(int(p["category_id"]) for p in final)),
    })
    save_json(fused, args.out_dir / "predictions_fused_before_scale.json")
    save_json(best[6] | {"predictions": len(fused), "scales": best[4], "nms_thr": best[5]}, args.out_dir / "eval_fused_before_scale.json")
    save_json(scale_rows, args.out_dir / "scale_sweep.json")
    save_json(final, args.out_dir / "predictions.json")
    save_json(metrics, args.out_dir / "eval.json")
    write_pkl(final, args.out_dir / f"{DS}.pkl")
    if args.vis:
        visualize(final, args.out_dir, args.vis_count)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
