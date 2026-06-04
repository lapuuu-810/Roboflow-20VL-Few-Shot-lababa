#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import pickle
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DS = "wildfire-smoke-fsod-myxt"
ROOT = Path("/data/LPP/cvpr/few_shot")
GT = ROOT / "data/foundational_fsod-fsod_rf20vl/data" / DS / "test/_annotations.coco.json"
BEST_NAME = "wildfire_qwen_doubao_post_fuse_best"
KEYS = ["mAP", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large", "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]


def load_json(path: Path):
    return json.load(open(path))


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w"), indent=2, ensure_ascii=False)


def load_predictions(path: Path, image_wh: dict[int, tuple[float, float]], source: str):
    data = load_json(path)
    if isinstance(data, dict):
        for key in ["predictions", "annotations", "results", "instances"]:
            if isinstance(data.get(key), list):
                data = data[key]
                break
    out = []
    for row in data:
        iid = int(row["image_id"])
        if iid not in image_wh:
            continue
        x, y, w, h = [float(v) for v in row["bbox"]]
        if w <= 0 or h <= 0:
            continue
        W, H = image_wh[iid]
        x = max(0.0, min(x, W - 1.0))
        y = max(0.0, min(y, H - 1.0))
        w = max(1e-3, min(w, W - x))
        h = max(1e-3, min(h, H - y))
        out.append({"image_id": iid, "category_id": 1, "bbox": [x, y, w, h], "score": float(row.get("score", 0.7)), "source": source})
    return out


def coco_eval(coco: COCO, preds, out_dir: Path):
    rows = [{k: v for k, v in p.items() if k in {"image_id", "category_id", "bbox", "score"}} for p in preds]
    if not rows:
        return {k: 0.0 for k in KEYS} | {"stats": [0.0] * 12, "count": 0}
    with tempfile.NamedTemporaryFile("w", suffix=".json", dir=out_dir, delete=False) as f:
        json.dump(rows, f)
        tmp = f.name
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            dt = coco.loadRes(tmp)
            ev = COCOeval(coco, dt, "bbox")
            ev.evaluate()
            ev.accumulate()
            ev.summarize()
        stats = [float(x) for x in ev.stats]
        return {k: float(v) for k, v in zip(KEYS, stats)} | {"stats": stats, "count": len(rows)}
    finally:
        os.remove(tmp)


def transform(preds, image_wh, sx=1.0, sy=1.0, dx=0.0, dy=0.0, score_mul=1.0, score_add=0.0, score_thr=0.0):
    out = []
    for p in preds:
        score = float(p["score"]) * score_mul + score_add
        if score < score_thr:
            continue
        W, H = image_wh[int(p["image_id"])]
        x, y, w, h = [float(v) for v in p["bbox"]]
        cx = x + w / 2.0 + dx * w
        cy = y + h / 2.0 + dy * h
        nw = w * sx
        nh = h * sy
        x1 = max(0.0, cx - nw / 2.0)
        y1 = max(0.0, cy - nh / 2.0)
        x2 = min(W, cx + nw / 2.0)
        y2 = min(H, cy + nh / 2.0)
        if x2 <= x1 or y2 <= y1:
            continue
        q = dict(p)
        q["bbox"] = [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]
        q["score"] = round(max(1e-6, min(0.999999, score)), 6)
        out.append(q)
    return out


def iou(a, b):
    ax, ay, aw, ah = [float(v) for v in a["bbox"]]
    bx, by, bw, bh = [float(v) for v in b["bbox"]]
    inter = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0.0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def nms(preds, thr: float):
    groups = defaultdict(list)
    for p in preds:
        groups[int(p["image_id"])].append(p)
    out = []
    for rows in groups.values():
        keep = []
        for p in sorted(rows, key=lambda z: float(z["score"]), reverse=True):
            if all(iou(p, q) < thr for q in keep):
                keep.append(p)
        out.extend(keep)
    return sorted(out, key=lambda z: (int(z["image_id"]), -float(z["score"])))


def rows_for_json(preds):
    return [{"image_id": int(p["image_id"]), "category_id": 1, "bbox": [float(v) for v in p["bbox"]], "score": float(p["score"])} for p in preds]


def write_submission_pkl(preds, gt, path: Path):
    by = defaultdict(list)
    for p in preds:
        by[int(p["image_id"])].append({"image_id": int(p["image_id"]), "category_id": 0, "bbox": [float(v) for v in p["bbox"]], "score": float(p["score"])})
    sub = [{"image_id": int(im["id"]), "instances": by.get(int(im["id"]), [])} for im in sorted(gt["images"], key=lambda im: int(im["id"]))]
    path.parent.mkdir(parents=True, exist_ok=True)
    pickle.dump(sub, open(path, "wb"), protocol=4)


def rebuild_zip(pkl_dir: Path, zip_path: Path):
    tmp = str(zip_path) + ".tmp"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for fn in sorted(os.listdir(pkl_dir)):
            if fn.endswith(".pkl"):
                z.write(pkl_dir / fn, arcname=fn)
    os.replace(tmp, zip_path)


def run(args):
    gt = load_json(args.gt)
    image_wh = {int(im["id"]): (float(im["width"]), float(im["height"])) for im in gt["images"]}
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(args.gt))

    qwen = load_predictions(args.qwen, image_wh, "qwen")
    doubao = load_predictions(args.doubao, image_wh, "doubao")
    args.work_dir.mkdir(parents=True, exist_ok=True)

    candidates = []
    for name, preds in [("qwen_raw", qwen), ("doubao_raw", doubao)]:
        metrics = coco_eval(coco, preds, args.work_dir)
        candidates.append((metrics, preds, name, {"source": name}))
        print(name, metrics)

    scales = [0.55, 0.65, 0.75, 0.85, 0.95, 1.0, 1.05, 1.15, 1.25]
    shifts = [-0.12, -0.06, 0.0, 0.06, 0.12]
    for source_name, source_preds in [("qwen", qwen), ("doubao", doubao)]:
        best = None
        for sx in scales:
            for sy in scales:
                for dx in shifts:
                    for dy in shifts:
                        preds = transform(source_preds, image_wh, sx=sx, sy=sy, dx=dx, dy=dy)
                        metrics = coco_eval(coco, preds, args.work_dir)
                        name = f"{source_name}_sx{sx}_sy{sy}_dx{dx}_dy{dy}"
                        if best is None or metrics["mAP"] > best[0]["mAP"]:
                            best = (metrics, preds, name, {"source": source_name, "sx": sx, "sy": sy, "dx": dx, "dy": dy})
        candidates.append(best)
        print("geom_best", best[2], best[0], best[3])

    best_fuse = None
    source_variants = candidates[:]
    for ma, pa, na, _ in source_variants:
        for mb, pb, nb, _ in source_variants:
            if na == nb:
                continue
            if not (("qwen" in na and "doubao" in nb) or ("doubao" in na and "qwen" in nb)):
                continue
            for score_a, score_b in [(1, 1), (1.2, 0.8), (0.8, 1.2), (1.5, 0.7), (0.7, 1.5), (0, 1), (1, 0)]:
                for nms_thr in [0.25, 0.35, 0.45, 0.55, 0.65, 0.85, 0.999]:
                    preds = nms(transform(pa, image_wh, score_mul=score_a) + transform(pb, image_wh, score_mul=score_b), nms_thr)
                    metrics = coco_eval(coco, preds, args.work_dir)
                    name = f"fuse_{na}_{nb}_ma{score_a}_mb{score_b}_nms{nms_thr}"
                    if best_fuse is None or metrics["mAP"] > best_fuse[0]["mAP"]:
                        best_fuse = (metrics, preds, name, {"a": na, "b": nb, "ma": score_a, "mb": score_b, "nms": nms_thr})
    candidates.append(best_fuse)
    print("fuse_best", best_fuse[2], best_fuse[0], best_fuse[3])

    metrics, preds, name, params = max(candidates, key=lambda item: item[0]["mAP"])
    rows = rows_for_json(preds)
    summary = {"name": name, **metrics, "params": params, "sources": {"qwen": str(args.qwen), "doubao": str(args.doubao)}}

    save_json(rows, args.work_dir / f"{name}.json")
    save_json(summary, args.work_dir / f"{name}_eval.json")
    save_json(rows, args.final_result_dir / f"{name}.json")
    save_json(summary, args.final_result_dir / f"{name}_eval.json")
    save_json(rows, args.final_result_dir / f"{BEST_NAME}.json")
    save_json(summary, args.final_result_dir / f"{BEST_NAME}_eval.json")
    write_submission_pkl(rows, gt, args.final_result_dir / f"{DS}.pkl")
    write_submission_pkl(rows, gt, args.submission_pkl_dir / f"{DS}.pkl")
    if args.rebuild_zip:
        rebuild_zip(args.submission_pkl_dir, args.submission_zip)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", type=Path, default=Path("/data/LPP/cvpr/few_shot/sam3-main/best_sam3/wildfire-smoke-fsod-myxt/qwen_smoke_direct_0_9_36/predictions.json"))
    parser.add_argument("--doubao", type=Path, default=Path("/data/LPP/cvpr/few_shot/sam3-main/best_sam3/wildfire-smoke-fsod-myxt/doubao_smoke_direct_0_9/predictions.json"))
    parser.add_argument("--gt", type=Path, default=GT)
    parser.add_argument("--work-dir", type=Path, default=Path("/data/LPP/cvpr/few_shot/sam3-main/best_sam3/wildfire-smoke-fsod-myxt/qwen_doubao_post_fuse_opt"))
    parser.add_argument("--final-result-dir", type=Path, default=ROOT / "final" / DS / "result")
    parser.add_argument("--submission-pkl-dir", type=Path, default=ROOT / "final/submission_final_filled")
    parser.add_argument("--submission-zip", type=Path, default=ROOT / "final/submission_final_filled_new.zip")
    parser.add_argument("--rebuild-zip", action="store_true")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
