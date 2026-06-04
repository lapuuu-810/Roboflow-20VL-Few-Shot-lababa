#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
from collections import defaultdict
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parent
DS = "water-meter-jbktv-7vz5k-fsod-ftoz"
DATA = Path("/data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data") / DS
GT = DATA / "test" / "_annotations.coco.json"
KEYS = ["mAP", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large", "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]


def load_json(path: Path):
    return json.load(open(path))


def xyxy_to_xywh(box):
    x1, y1, x2, y2 = map(float, box)
    return [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]


def center(box):
    return (float(box[0] + box[2]) / 2.0, float(box[1] + box[3]) / 2.0)


def center_in(box, region, pad=0.0):
    cx, cy = center(box)
    return region[0] - pad <= cx <= region[2] + pad and region[1] - pad <= cy <= region[3] + pad


def load_sam_rows(sam_dir: Path, image_by):
    file_to_img = {im["file_name"]: im for im in image_by.values()}
    rows = defaultdict(list)
    for fp in sorted(sam_dir.glob("predictions_gpu*.jsonl")):
        for line in fp.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            im = file_to_img.get(os.path.basename(rec.get("image_path", "")))
            if im is None:
                continue
            x1, y1, x2, y2 = [float(v) for v in rec["box"]]
            width, height = float(im["width"]), float(im["height"])
            box = [
                max(0.0, min(width - 1.0, x1)),
                max(0.0, min(height - 1.0, y1)),
                max(0.0, min(width, x2)),
                max(0.0, min(height, y2)),
            ]
            if box[2] <= box[0] or box[3] <= box[1]:
                continue
            rows[int(im["id"])].append(
                {
                    "image_id": int(im["id"]),
                    "category_id": int(rec.get("category_id", 1)),
                    "bbox_xyxy": box,
                    "score": float(rec.get("score", 0.0)),
                    "prompt": str(rec.get("prompt", "")),
                    "mask_path": rec.get("mask_path"),
                }
            )
    return rows


def evaluate(preds, image_ids, out_dir: Path):
    if not preds:
        return {k: 0.0 for k in KEYS}
    subset = load_json(GT)
    keep = set(image_ids)
    subset["images"] = [im for im in subset["images"] if int(im["id"]) in keep]
    subset["annotations"] = [ann for ann in subset["annotations"] if int(ann["image_id"]) in keep]
    subset_path = out_dir / "subset_gt.json"
    json.dump(subset, open(subset_path, "w"))
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(subset_path))
        dt = coco.loadRes(preds)
        ev = COCOeval(coco, dt, "bbox")
        ev.params.imgIds = image_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {k: float(v) for k, v in zip(KEYS, ev.stats)}


def reading_from_record(rec):
    reading = str(rec.get("reading", "")).strip()
    if reading:
        return "".join(ch for ch in reading if ch.isdigit())
    return "".join(str(d.get("digit", "")) for d in rec.get("digits", []) if str(d.get("digit", "")).isdigit())


def make_slot_box(panel, image, index, count, width_scale, height_scale):
    x1, y1, x2, y2 = [float(v) for v in panel]
    step = (x2 - x1) / max(1, count)
    cx = x1 + (index + 0.5) * step
    cy = (y1 + y2) / 2.0
    w = step * width_scale
    h = (y2 - y1) * height_scale
    return [
        max(0.0, cx - w / 2.0),
        max(0.0, cy - h / 2.0),
        min(float(image["width"]), cx + w / 2.0),
        min(float(image["height"]), cy + h / 2.0),
    ]


def build_predictions(reads, panels, sam_by_img, image_by, args):
    preds = []
    audit = []
    for rec in reads:
        iid = int(rec["image_id"])
        if iid not in panels:
            continue
        reading = reading_from_record(rec)
        if not reading:
            continue
        panel = panels[iid]["meter_bbox"]
        slot_count = len(reading)
        step = (panel[2] - panel[0]) / max(1, slot_count)
        used = set()
        for idx, digit_char in enumerate(reading):
            digit = int(digit_char)
            category_id = digit + 1
            expected_x = panel[0] + (idx + 0.5) * step
            prompt = f"digit {digit}"
            candidates = []
            for cand_idx, row in enumerate(sam_by_img.get(iid, [])):
                if cand_idx in used:
                    continue
                if int(row["category_id"]) != category_id:
                    continue
                box = row["bbox_xyxy"]
                if not center_in(box, panel, args.panel_pad):
                    continue
                cx, cy = center(box)
                slot_dist = abs(cx - expected_x) / (step + 1e-9)
                if slot_dist > args.max_slot_dist:
                    continue
                candidates.append((slot_dist, -float(row["score"]), cand_idx, row))
            if candidates:
                slot_dist, neg_score, cand_idx, row = min(candidates, key=lambda x: (x[0], x[1]))
                used.add(cand_idx)
                score = max(0.01, min(0.99, args.sam_score_base + args.sam_score_weight * float(row["score"]) - args.slot_penalty * min(slot_dist, 2.0)))
                preds.append({"image_id": iid, "category_id": category_id, "bbox": xyxy_to_xywh(row["bbox_xyxy"]), "score": round(score, 4)})
                audit.append({"image_id": iid, "slot": idx, "prompt": prompt, "source": "sam3", "slot_dist": round(float(slot_dist), 4), "sam_score": round(float(row["score"]), 4)})
            elif args.use_slot_fallback:
                box = make_slot_box(panel, image_by[iid], idx, slot_count, args.slot_width_scale, args.slot_height_scale)
                preds.append({"image_id": iid, "category_id": category_id, "bbox": xyxy_to_xywh(box), "score": args.fallback_score})
                audit.append({"image_id": iid, "slot": idx, "prompt": prompt, "source": "slot_fallback"})
            else:
                audit.append({"image_id": iid, "slot": idx, "prompt": prompt, "source": "dropped"})
    return sorted(preds, key=lambda p: (int(p["image_id"]), p["bbox"][0])), audit


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--qwen-json", type=Path, required=True, help="Existing Step2 qwen_stage2_reads.json; Qwen is not rerun.")
    parser.add_argument("--panel-json", type=Path, required=True)
    parser.add_argument("--sam-dir", type=Path, default=ROOT, help="Directory containing existing predictions_gpu*.jsonl SAM3 prompt results.")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--panel-pad", type=float, default=4.0)
    parser.add_argument("--max-slot-dist", type=float, default=2.0)
    parser.add_argument("--use-slot-fallback", action="store_true")
    parser.add_argument("--slot-width-scale", type=float, default=0.62)
    parser.add_argument("--slot-height-scale", type=float, default=0.55)
    parser.add_argument("--fallback-score", type=float, default=0.25)
    parser.add_argument("--sam-score-base", type=float, default=0.85)
    parser.add_argument("--sam-score-weight", type=float, default=0.10)
    parser.add_argument("--slot-penalty", type=float, default=0.03)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    gt = load_json(GT)
    image_by = {int(im["id"]): im for im in gt["images"]}
    image_ids = sorted(image_by)
    reads = load_json(args.qwen_json)
    panels = {int(r["image_id"]): r for r in load_json(args.panel_json)}
    sam_by_img = load_sam_rows(args.sam_dir, image_by)
    preds, audit = build_predictions(reads, panels, sam_by_img, image_by, args)
    json.dump(preds, open(args.out_dir / "predictions.json", "w"), indent=2)
    json.dump(audit, open(args.out_dir / "audit.json", "w"), indent=2)
    metrics = evaluate(preds, image_ids, args.out_dir)
    metrics.update({"images": len(image_ids), "predictions": len(preds), "image_ids": image_ids})
    json.dump(metrics, open(args.out_dir / "eval.json", "w"), indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
