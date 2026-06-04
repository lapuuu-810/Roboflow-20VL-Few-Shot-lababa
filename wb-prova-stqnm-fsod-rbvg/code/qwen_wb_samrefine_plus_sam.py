#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

import qwen_wb_direct_api_v2 as base


def iou(a, b):
    ax, ay, aw, ah = map(float, a)
    bx, by, bw, bh = map(float, b)
    inter = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0.0, min(ay + ah, by + bh) - max(ay, by)
    )
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def nms(preds, thr):
    out = []
    for p in sorted(preds, key=lambda z: z["score"], reverse=True):
        if all(iou(p["bbox"], q["bbox"]) < thr for q in out):
            out.append(p)
    return out


def clean_preds(preds):
    return [
        {
            "image_id": int(p["image_id"]),
            "category_id": int(p["category_id"]),
            "bbox": [round(float(v), 2) for v in p["bbox"]],
            "score": round(float(p["score"]), 4),
        }
        for p in preds
    ]


def load_sam_rows(sam_dir: Path, image_by):
    file_to_img = {im["file_name"]: im for im in image_by.values()}
    rows = defaultdict(list)
    for fp in sorted(sam_dir.glob("predictions_gpu*.jsonl")):
        for line in fp.read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            im = file_to_img.get(os.path.basename(r.get("image_path", "")))
            if im is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in r["box"]]
            x1 = max(0.0, min(float(im["width"]) - 1.0, x1))
            y1 = max(0.0, min(float(im["height"]) - 1.0, y1))
            x2 = max(0.0, min(float(im["width"]), x2))
            y2 = max(0.0, min(float(im["height"]), y2))
            if x2 <= x1 or y2 <= y1:
                continue
            rows[int(im["id"])].append(
                {
                    "image_id": int(im["id"]),
                    "category_id": int(r.get("category_id", 1)),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],
                    "score": float(r.get("score", 0.0)),
                    "prompt": str(r.get("prompt", "")).lower(),
                }
            )
    return rows


def refine_with_sam(qwen_preds, sam_by_img, match_iou):
    refined = []
    for p in qwen_preds:
        candidates = sam_by_img.get(int(p["image_id"]), [])
        best = None
        for s in candidates:
            ov = iou(p["bbox"], s["bbox"])
            if best is None or ov > best[0]:
                best = (ov, s)
        if best and best[0] >= match_iou:
            q = dict(p)
            q["bbox"] = [round(float(v), 2) for v in best[1]["bbox"]]
            q["score"] = round(float(p["score"]) * (0.92 + 0.04 * best[0]), 4)
            refined.append(q)
        else:
            refined.append(p)
    return refined


def add_sam_recall(preds, sam_by_img, image_ids, topk, score_thr, overlap_thr, score_scale):
    out = list(preds)
    for iid in image_ids:
        added = 0
        for s in sorted(sam_by_img.get(int(iid), []), key=lambda z: z["score"], reverse=True):
            if s["score"] < score_thr:
                continue
            if all(iou(s["bbox"], p["bbox"]) < overlap_thr for p in out if p["image_id"] == iid):
                out.append(
                    {
                        "image_id": int(iid),
                        "category_id": 1,
                        "bbox": [round(float(v), 2) for v in s["bbox"]],
                        "score": round(float(s["score"]) * score_scale, 4),
                    }
                )
                added += 1
                if added >= topk:
                    break
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--qwen-dir", type=Path, required=True)
    ap.add_argument(
        "--sam-dir",
        type=Path,
        default=base.ROOT,
        help="Directory containing predictions_gpu*.jsonl SAM candidate files.",
    )
    ap.add_argument("--image-ids", default="all")
    ap.add_argument("--match-iou", type=float, default=0.35)
    ap.add_argument("--sam-score-thr", type=float, default=0.05)
    ap.add_argument("--sam-add-topk", type=int, default=1)
    ap.add_argument("--sam-overlap-thr", type=float, default=0.20)
    ap.add_argument("--sam-score-scale", type=float, default=0.05)
    ap.add_argument("--nms-thr", type=float, default=0.85)
    args = ap.parse_args()

    gt = base.load_json(base.GT)
    image_by = {int(im["id"]): im for im in gt["images"]}
    image_ids = base.parse_ids(args.image_ids, image_by)
    qwen_preds = base.load_json(args.qwen_dir / "predictions.json")
    sam_by_img = load_sam_rows(args.sam_dir, image_by)

    refined = refine_with_sam(qwen_preds, sam_by_img, args.match_iou)
    refined = clean_preds(nms(refined, args.nms_thr))
    json.dump(refined, open(args.qwen_dir / "predictions_samrefine.json", "w"), indent=2)
    refined_eval = base.evaluate(clean_preds(refined), image_ids, args.qwen_dir)
    refined_eval.update({"images": len(image_ids), "predictions": len(refined), "image_ids": image_ids})
    json.dump(refined_eval, open(args.qwen_dir / "eval_samrefine.json", "w"), indent=2)

    plus = add_sam_recall(
        refined,
        sam_by_img,
        image_ids,
        args.sam_add_topk,
        args.sam_score_thr,
        args.sam_overlap_thr,
        args.sam_score_scale,
    )
    plus = clean_preds(nms(plus, args.nms_thr))
    json.dump(plus, open(args.qwen_dir / "predictions_samrefine_plus_sam.json", "w"), indent=2)
    plus_eval = base.evaluate(clean_preds(plus), image_ids, args.qwen_dir)
    plus_eval.update({"images": len(image_ids), "predictions": len(plus), "image_ids": image_ids})
    json.dump(plus_eval, open(args.qwen_dir / "eval_samrefine_plus_sam.json", "w"), indent=2)
    print(json.dumps({"samrefine": refined_eval, "samrefine_plus_sam": plus_eval}, indent=2))


if __name__ == "__main__":
    main()
