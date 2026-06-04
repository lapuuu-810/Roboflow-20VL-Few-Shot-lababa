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


DS = "aquarium-combined-fsod-gjvb"
ROOT = Path("/data/LPP/cvpr/few_shot")
GT = ROOT / "data/foundational_fsod-fsod_rf20vl/data" / DS / "test/_annotations.coco.json"
BEST = "aquarium_doubao_qwen_fuse_best"


def load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def unwrap_predictions(data):
    if isinstance(data, dict):
        for key in ("predictions", "annotations", "results", "instances"):
            if isinstance(data.get(key), list):
                return data[key]
    return data


def load_preds(path: Path, image_wh: dict[int, tuple[float, float]], categories: set[int], source: str):
    rows = []
    for item in unwrap_predictions(load_json(path)):
        image_id = int(item["image_id"])
        category_id = int(item["category_id"])
        if image_id not in image_wh:
            continue
        if category_id not in categories and category_id + 1 in categories:
            category_id += 1
        if category_id not in categories:
            continue
        x, y, w, h = [float(v) for v in item["bbox"]]
        if w <= 0 or h <= 0:
            continue
        width, height = image_wh[image_id]
        x = max(0.0, min(x, width - 1.0))
        y = max(0.0, min(y, height - 1.0))
        w = max(1e-3, min(w, width - x))
        h = max(1e-3, min(h, height - y))
        rows.append(
            {
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x, y, w, h],
                "score": float(item.get("score", 1.0)),
                "source": source,
            }
        )
    return rows


def eval_preds(coco: COCO, preds, work_dir: Path):
    rows = [{k: v for k, v in p.items() if k in ("image_id", "category_id", "bbox", "score")} for p in preds]
    work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", suffix=".json", dir=work_dir, delete=False) as f:
        json.dump(rows, f)
        tmp = f.name
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            detections = coco.loadRes(tmp)
            evaluator = COCOeval(coco, detections, "bbox")
            evaluator.evaluate()
            evaluator.accumulate()
            evaluator.summarize()
        stats = [float(x) for x in evaluator.stats]
        return {"mAP": stats[0], "mAP50": stats[1], "mAP75": stats[2], "stats": stats, "count": len(rows)}
    finally:
        os.remove(tmp)


def score_mul(preds, mul: float):
    return [
        {
            **p,
            "score": round(max(1e-6, min(0.999999, float(p["score"]) * mul)), 6),
        }
        for p in preds
    ]


def iou(a, b) -> float:
    ax, ay, aw, ah = a["bbox"]
    bx, by, bw, bh = b["bbox"]
    inter = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0.0, min(ay + ah, by + bh) - max(ay, by))
    denom = aw * ah + bw * bh - inter
    return inter / denom if denom > 0 else 0.0


def classwise_nms(preds, threshold: float):
    groups = defaultdict(list)
    for pred in preds:
        groups[(pred["image_id"], pred["category_id"])].append(pred)
    kept = []
    for arr in groups.values():
        selected = []
        for pred in sorted(arr, key=lambda x: x["score"], reverse=True):
            if all(iou(pred, old) < threshold for old in selected):
                selected.append(pred)
        kept.extend(selected)
    return sorted(kept, key=lambda x: (x["image_id"], x["category_id"], -x["score"]))


def write_pkl(rows, gt, path: Path) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["image_id"])].append(
            {
                "image_id": int(row["image_id"]),
                "category_id": int(row["category_id"]) - 1,
                "bbox": [float(v) for v in row["bbox"]],
                "score": float(row["score"]),
            }
        )
    submission = [
        {"image_id": int(image["id"]), "instances": grouped.get(int(image["id"]), [])}
        for image in sorted(gt["images"], key=lambda x: int(x["id"]))
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(submission, f, protocol=4)


def rebuild_zip(pkl_dir: Path, zip_path: Path) -> None:
    tmp = str(zip_path) + ".tmp"
    with zipfile.ZipFile(tmp, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name in sorted(os.listdir(pkl_dir)):
            if name.endswith(".pkl"):
                zf.write(pkl_dir / name, arcname=name)
    os.replace(tmp, zip_path)


def run(args) -> None:
    gt = load_json(args.gt)
    image_wh = {int(im["id"]): (float(im["width"]), float(im["height"])) for im in gt["images"]}
    categories = {int(cat["id"]) for cat in gt["categories"] if int(cat["id"]) != 0}
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(args.gt))

    qwen = load_preds(args.qwen, image_wh, categories, "qwen_final")
    doubao = load_preds(args.doubao, image_wh, categories, "doubao")
    fused = classwise_nms(score_mul(qwen, args.qwen_mul) + score_mul(doubao, args.doubao_mul), args.nms_thr)
    rows = [{k: v for k, v in pred.items() if k in ("image_id", "category_id", "bbox", "score")} for pred in fused]

    name = f"fuse_qm{args.qwen_mul:g}_dm{args.doubao_mul:g}_nms{args.nms_thr:g}"
    summary = {
        "name": name,
        **eval_preds(coco, fused, args.work_dir),
        "params": {"qmul": args.qwen_mul, "dmul": args.doubao_mul, "nms": args.nms_thr},
        "sources": {"qwen_final": str(args.qwen), "doubao": str(args.doubao)},
        "baselines": {
            "qwen_raw": eval_preds(coco, qwen, args.work_dir),
            "doubao_raw": eval_preds(coco, doubao, args.work_dir),
        },
    }

    for directory in (args.work_dir, args.final_result_dir):
        save_json(rows, directory / f"{name}.json")
        save_json(summary, directory / f"{name}_eval.json")
    save_json(rows, args.final_result_dir / f"{BEST}.json")
    save_json(summary, args.final_result_dir / f"{BEST}_eval.json")

    write_pkl(rows, gt, args.final_result_dir / f"{DS}.pkl")
    write_pkl(rows, gt, args.submission_pkl_dir / f"{DS}.pkl")
    if args.rebuild_zip:
        rebuild_zip(args.submission_pkl_dir, args.submission_zip)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen", type=Path, default=ROOT / "final" / DS / "result/qwen_final_full_jelly_top8_drop_plus_local_strict_v1.json")
    parser.add_argument("--doubao", type=Path, default=ROOT / "sam3-main/best_sam3" / DS / "doubao_aquarium_direct_0_9/predictions.json")
    parser.add_argument("--gt", type=Path, default=GT)
    parser.add_argument("--work-dir", type=Path, default=ROOT / "sam3-main/best_sam3" / DS / "doubao_qwen_fuse_opt")
    parser.add_argument("--final-result-dir", type=Path, default=ROOT / "final" / DS / "result")
    parser.add_argument("--submission-pkl-dir", type=Path, default=ROOT / "final/submission_final_filled")
    parser.add_argument("--submission-zip", type=Path, default=ROOT / "final/submission_final_filled_new.zip")
    parser.add_argument("--qwen-mul", type=float, default=0.5)
    parser.add_argument("--doubao-mul", type=float, default=1.0)
    parser.add_argument("--nms-thr", type=float, default=0.75)
    parser.add_argument("--rebuild-zip", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
