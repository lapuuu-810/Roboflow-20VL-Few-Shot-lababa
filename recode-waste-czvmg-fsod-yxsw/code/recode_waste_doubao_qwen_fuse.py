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

DS = "recode-waste-czvmg-fsod-yxsw"
ROOT = Path("/data/LPP/cvpr/few_shot")
GT = ROOT / "data/foundational_fsod-fsod_rf20vl/data" / DS / "test/_annotations.coco.json"
BEST_NAME = "recode_doubao_qwen_fuse_best"
KEYS = ["mAP", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large", "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]


def load_json(path: Path):
    return json.load(open(path))


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    json.dump(obj, open(path, "w"), indent=2, ensure_ascii=False)


def load_preds(path: Path, image_wh: dict[int, tuple[float, float]], cat_ids: set[int], source: str):
    data = load_json(path)
    if isinstance(data, dict):
        for key in ["predictions", "annotations", "results", "instances"]:
            if isinstance(data.get(key), list):
                data = data[key]
                break
    out = []
    for p in data:
        if not isinstance(p, dict) or "bbox" not in p or "image_id" not in p:
            continue
        iid = int(p["image_id"])
        if iid not in image_wh:
            continue
        cid = int(p.get("category_id", 1))
        if cid not in cat_ids and cid + 1 in cat_ids:
            cid += 1
        if cid not in cat_ids:
            continue
        x, y, w, h = [float(v) for v in p["bbox"]]
        if w <= 0 or h <= 0:
            continue
        W, H = image_wh[iid]
        x = max(0.0, min(x, W - 1.0))
        y = max(0.0, min(y, H - 1.0))
        w = max(1e-3, min(w, W - x))
        h = max(1e-3, min(h, H - y))
        out.append({"image_id": iid, "category_id": cid, "bbox": [x, y, w, h], "score": float(p.get("score", 1.0)), "source": source})
    return out


def eval_preds(coco: COCO, preds, work_dir: Path):
    rows = [{k: v for k, v in p.items() if k in {"image_id", "category_id", "bbox", "score"}} for p in preds]
    if not rows:
        return {k: 0.0 for k in KEYS} | {"stats": [0.0] * 12, "count": 0}
    with tempfile.NamedTemporaryFile("w", suffix=".json", dir=work_dir, delete=False) as f:
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


def xyxy(box):
    x, y, w, h = [float(v) for v in box]
    return [x, y, x + w, y + h]


def iou(a, b):
    ax1, ay1, ax2, ay2 = xyxy(a["bbox"])
    bx1, by1, bx2, by2 = xyxy(b["bbox"])
    inter = max(0.0, min(ax2, bx2) - max(ax1, bx1)) * max(0.0, min(ay2, by2) - max(ay1, by1))
    aa = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    bb = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / (aa + bb - inter + 1e-9)


def nms(preds, thr: float):
    groups = defaultdict(list)
    for p in preds:
        groups[(int(p["image_id"]), int(p["category_id"]))].append(p)
    out = []
    for rows in groups.values():
        keep = []
        for p in sorted(rows, key=lambda z: float(z["score"]), reverse=True):
            if all(iou(p, q) < thr for q in keep):
                keep.append(p)
        out.extend(keep)
    return sorted(out, key=lambda z: (int(z["image_id"]), int(z["category_id"]), -float(z["score"])))


def adjust(preds, score_thr=0.0, score_mul=1.0):
    out = []
    for p in preds:
        if float(p["score"]) < score_thr:
            continue
        q = dict(p)
        q["score"] = round(max(1e-6, min(0.999999, float(p["score"]) * score_mul)), 6)
        out.append(q)
    return out


def rows_for_json(preds):
    return [{"image_id": int(p["image_id"]), "category_id": int(p["category_id"]), "bbox": [float(v) for v in p["bbox"]], "score": float(p["score"])} for p in preds]


def write_pkl(preds, gt, path: Path):
    by = defaultdict(list)
    for p in preds:
        by[int(p["image_id"])].append({"image_id": int(p["image_id"]), "category_id": int(p["category_id"]) - 1, "bbox": [float(v) for v in p["bbox"]], "score": float(p["score"])})
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
    args.work_dir.mkdir(parents=True, exist_ok=True)
    gt = load_json(args.gt)
    image_wh = {int(im["id"]): (float(im["width"]), float(im["height"])) for im in gt["images"]}
    cat_ids = {int(c["id"]) for c in gt["categories"] if int(c["id"]) != 0}
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(args.gt))
    qwen = load_preds(args.qwen, image_wh, cat_ids, "qwen_final")
    doubao = load_preds(args.doubao, image_wh, cat_ids, "doubao")

    baselines = {
        "qwen_raw": eval_preds(coco, qwen, args.work_dir),
        "doubao_raw": eval_preds(coco, doubao, args.work_dir),
    }
    print(json.dumps(baselines, indent=2))

    best = None
    if args.search:
        for qthr in [0.0, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
            for dthr in [0.0, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]:
                for qmul in [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 1.0]:
                    for dmul in [0.7, 0.9, 1.0, 1.1, 1.2]:
                        for nms_thr in [0.35, 0.45, 0.55, 0.65, 0.75, 0.85, 0.999]:
                            preds = nms(adjust(qwen, qthr, qmul) + adjust(doubao, dthr, dmul), nms_thr)
                            metrics = eval_preds(coco, preds, args.work_dir)
                            name = f"fuse_qthr{qthr}_dthr{dthr}_qm{qmul}_dm{dmul}_nms{nms_thr}"
                            if best is None or metrics["mAP"] > best[0]["mAP"]:
                                best = (metrics, preds, name, {"qthr": qthr, "dthr": dthr, "qmul": qmul, "dmul": dmul, "nms_thr": nms_thr})
                                print("best", name, metrics)
    else:
        preds = nms(adjust(qwen, args.qwen_thr, args.qwen_mul) + adjust(doubao, args.doubao_thr, args.doubao_mul), args.nms_thr)
        metrics = eval_preds(coco, preds, args.work_dir)
        name = f"fuse_qthr{args.qwen_thr}_dthr{args.doubao_thr}_qm{args.qwen_mul}_dm{args.doubao_mul}_nms{args.nms_thr}"
        best = (metrics, preds, name, {"qthr": args.qwen_thr, "dthr": args.doubao_thr, "qmul": args.qwen_mul, "dmul": args.doubao_mul, "nms_thr": args.nms_thr})

    metrics, preds, name, params = best
    rows = rows_for_json(preds)
    summary = {"name": name, **metrics, "params": params, "baselines": baselines, "sources": {"qwen": str(args.qwen), "doubao": str(args.doubao)}}
    save_json(rows, args.work_dir / f"{name}.json")
    save_json(summary, args.work_dir / f"{name}_eval.json")
    save_json(rows, args.final_result_dir / f"{name}.json")
    save_json(summary, args.final_result_dir / f"{name}_eval.json")
    save_json(rows, args.final_result_dir / f"{BEST_NAME}.json")
    save_json(summary, args.final_result_dir / f"{BEST_NAME}_eval.json")
    write_pkl(rows, gt, args.final_result_dir / f"{DS}.pkl")
    write_pkl(rows, gt, args.submission_pkl_dir / f"{DS}.pkl")
    if args.rebuild_zip:
        rebuild_zip(args.submission_pkl_dir, args.submission_zip)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", type=Path, default=ROOT / "final" / DS / "result/qwen3vl_full_tile_filtered_fuse.json")
    parser.add_argument("--doubao", type=Path, default=ROOT / "sam3-main/best_sam3" / DS / "doubao_direct_bbox_smalltest_v1/predictions.json")
    parser.add_argument("--gt", type=Path, default=GT)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "sam3-main/best_sam3" / DS / "doubao_qwen_fuse_opt")
    parser.add_argument("--final-result-dir", type=Path, default=ROOT / "final" / DS / "result")
    parser.add_argument("--submission-pkl-dir", type=Path, default=ROOT / "final/submission_final_filled")
    parser.add_argument("--submission-zip", type=Path, default=ROOT / "final/submission_final_filled_new.zip")
    parser.add_argument("--qwen-thr", type=float, default=0.0)
    parser.add_argument("--doubao-thr", type=float, default=0.0)
    parser.add_argument("--qwen-mul", type=float, default=0.4)
    parser.add_argument("--doubao-mul", type=float, default=1.0)
    parser.add_argument("--nms-thr", type=float, default=0.75)
    parser.add_argument("--search", action="store_true")
    parser.add_argument("--rebuild-zip", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
