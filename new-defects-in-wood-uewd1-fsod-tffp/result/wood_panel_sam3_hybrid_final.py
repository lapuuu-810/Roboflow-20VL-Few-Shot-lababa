#!/usr/bin/env python3
"""
Reproduce the current best final output for new-defects-in-wood-uewd1-fsod-tffp.

Input dependency:
  qwen_wood_sam3panel_all_edge_v1_raw/selected_candidates.json
  qwen_wood_sam3panel_all_edge_v1_raw/*_raw.json

Final strategy:
  - class 1 Crack: use SAM3 candidate class
  - class 2 Dead knot: use Qwen panel + SAM3 candidate class
  - class 3 Holes: use SAM3 candidate class
  - class 4 Live knot: use Qwen panel
  - class 5 knot with crack: use Qwen panel + SAM3 candidate class

Command:
  python wood_panel_sam3_hybrid_final.py \
    --raw-dir qwen_wood_sam3panel_all_edge_v1_raw \
    --out wood_qwen_panel_sam3_hybrid_all_v2.json
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


CLASS_NAMES = {
    1: "Crack",
    2: "Dead knot",
    3: "Holes",
    4: "Live knot",
    5: "knot with crack",
}


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path: Path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def iou(a, b):
    ax, ay, aw, ah = map(float, a)
    bx, by, bw, bh = map(float, b)
    inter = max(0, min(ax + aw, bx + bw) - max(ax, bx)) * max(0, min(ay + ah, by + bh) - max(ay, by))
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0


def nms(items, thr):
    out = []
    for pred in sorted(items, key=lambda x: x["score"], reverse=True):
        if all(iou(pred["bbox"], kept["bbox"]) < thr for kept in out):
            out.append(pred)
    return out


def post_filter(preds, nms_thr=0.35, max_per_image=100):
    by_class = defaultdict(list)
    for pred in preds:
        by_class[(pred["image_id"], pred["category_id"])].append(pred)

    by_image = defaultdict(list)
    for items in by_class.values():
        for pred in nms(items, nms_thr):
            by_image[pred["image_id"]].append(pred)

    out = []
    for image_id, items in by_image.items():
        out.extend(sorted(items, key=lambda x: x["score"], reverse=True)[:max_per_image])
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="qwen_wood_sam3panel_all_edge_v1_raw")
    parser.add_argument("--out", default="wood_qwen_panel_sam3_hybrid_all_v2.json")
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--nms", type=float, default=0.35)
    parser.add_argument("--sam-scale", type=float, default=0.7)
    args = parser.parse_args()

    raw_dir = Path(args.raw_dir)
    candidates = load_json(raw_dir / "selected_candidates.json")
    by_image = defaultdict(list)
    for cand in candidates:
        by_image[cand["image_id"]].append(cand)

    qwen_preds = []
    for image_id, cands in by_image.items():
        for start in range(0, len(cands), args.batch_size):
            raw_path = raw_dir / f"{image_id:04d}_{start:03d}_raw.json"
            if not raw_path.exists():
                continue
            obj = load_json(raw_path)
            index_to_cand = {start + i: cand for i, cand in enumerate(cands[start:start + args.batch_size])}
            for item in obj.get("items", obj.get("detections", [])):
                if not isinstance(item, dict):
                    continue
                try:
                    idx = int(item.get("index", -1))
                    category_id = int(item.get("category_id", 0) or 0)
                except Exception:
                    continue
                if not item.get("keep", True) or idx not in index_to_cand or category_id not in CLASS_NAMES:
                    continue
                cand = index_to_cand[idx]
                score = float(item.get("confidence", item.get("score", 0.82)) or 0.82)
                qwen_preds.append({
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": [float(v) for v in cand["bbox"]],
                    "score": max(0.001, min(0.999, score)),
                })

    preds = []
    # Qwen is useful for knot categories.
    for pred in qwen_preds:
        if pred["category_id"] in (2, 4, 5):
            preds.append(pred)

    # SAM3 is better for cracks and holes; keep SAM3 as an additional source for 2/5.
    for cand in candidates:
        category_id = int(cand.get("sam3_category_id") or 0)
        if category_id not in (1, 2, 3, 5):
            continue
        preds.append({
            "image_id": cand["image_id"],
            "category_id": category_id,
            "bbox": [float(v) for v in cand["bbox"]],
            "score": max(0.001, min(0.999, float(cand.get("score", 0.5)) * args.sam_scale)),
        })

    final_preds = post_filter(preds, args.nms)
    save_json(final_preds, Path(args.out))
    print(f"wrote {len(final_preds)} predictions to {args.out}")


if __name__ == "__main__":
    main()
