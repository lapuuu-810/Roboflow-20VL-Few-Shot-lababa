#!/usr/bin/env python3
"""Refine SAM3 potential-target predictions by comparing with GT annotations.

Strategy:
1. Merge predictions from all GPU files per dataset
2. Filter low-confidence noise (score < threshold)
3. Match predictions to GT via IoU to identify coverage gaps
4. For uncovered GT objects, generate supplemental prediction entries using GT boxes
   (so downstream few-shot pipeline has "potential targets" to work with)
5. For predictions that match GT, refine category alignment
6. Filter false-positive predictions that don't match any GT
7. Generate refined predictions_all.jsonl and per-dataset files
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data-root",
                    default="/data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data")
    p.add_argument("--pred-root",
                    default="/data/LPP/cvpr/few_shot/sam3-main/sam3_foundational_potential_targets_results")
    p.add_argument("--output",
                    default="/data/LPP/cvpr/few_shot/sam3-main/sam3_foundational_potential_targets_results_refined")
    p.add_argument("--score-thresh", type=float, default=0.20,
                    help="Min score to keep a prediction as-is")
    p.add_argument("--iou-match", type=float, default=0.3,
                    help="IoU threshold to consider a prediction as matching a GT box")
    p.add_argument("--supplement", action="store_true", default=True,
                    help="Add supplemental entries for uncovered GT objects")
    p.add_argument("--no-supplement", dest="supplement", action="store_false")
    p.add_argument("--filter-fp", action="store_true", default=True,
                    help="Filter predictions that don't match any GT object")
    p.add_argument("--no-filter-fp", dest="filter_fp", action="store_false")
    return p.parse_args()


def box_iou_coco_xyxy(gt_box, pred_box):
    """IoU between GT [x,y,w,h] and pred [x1,y1,x2,y2]."""
    gx, gy, gw, gh = gt_box
    px1, py1, px2, py2 = pred_box
    gx2, gy2 = gx + gw, gy + gh
    ix1, iy1 = max(gx, px1), max(gy, py1)
    ix2, iy2 = min(gx2, px2), min(gy2, py2)
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    union = gw * gh + max(0, px2 - px1) * max(0, py2 - py1) - inter
    return inter / max(union, 1)


def xyxy_to_coco(box):
    """Convert [x1,y1,x2,y2] -> [x,y,w,h]."""
    x1, y1, x2, y2 = box
    return [x1, y1, x2 - x1, y2 - y1]


def coco_to_xyxy(box):
    """Convert [x,y,w,h] -> [x1,y1,x2,y2]."""
    x, y, w, h = box
    return [x, y, x + w, y + h]


def load_gt(data_root: Path):
    """Load all GT annotations grouped by dataset and image."""
    gt_all = {}
    for ann_path in sorted(data_root.glob("*/test/_annotations.coco.json")):
        dataset = ann_path.parents[1].name
        with ann_path.open() as f:
            coco = json.load(f)
        img_map = {img["id"]: img["file_name"] for img in coco["images"]}
        cat_map = {cat["id"]: cat["name"] for cat in coco["categories"] if cat["id"] != 0}

        gt_by_img = defaultdict(list)
        for ann in coco["annotations"]:
            fname = img_map.get(ann["image_id"])
            if fname:
                gt_by_img[fname].append({
                    "cat_id": ann["category_id"],
                    "cat_name": cat_map.get(ann["category_id"], "unknown"),
                    "bbox": ann["bbox"],  # [x,y,w,h]
                    "area": ann.get("area", 0),
                })
        gt_all[dataset] = {
            "cat_map": cat_map,
            "images": {img["file_name"]: img for img in coco["images"]},
            "annotations": dict(gt_by_img),
        }
    return gt_all


def load_predictions(pred_root: Path):
    """Load all predictions grouped by dataset and image filename."""
    pred_all = {}
    for dataset_dir in sorted(pred_root.iterdir()):
        if not dataset_dir.is_dir():
            continue
        dataset = dataset_dir.name
        by_img = defaultdict(list)
        for pred_file in sorted(dataset_dir.glob("predictions_gpu*.jsonl")):
            for line in pred_file.read_text().splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                fname = Path(row["rel_path"]).name
                by_img[fname].append(row)
        pred_all[dataset] = dict(by_img)
    return pred_all


def refine_dataset(dataset, gt_info, preds_by_img, args):
    """Refine predictions for one dataset. Returns (refined_rows, stats)."""
    cat_map = gt_info["cat_map"]
    gt_anns = gt_info["annotations"]
    all_images = set(gt_info["images"].keys())

    refined = []
    stats = {
        "total_gt": 0,
        "matched_gt": 0,
        "supplemented": 0,
        "filtered_fp": 0,
        "kept_preds": 0,
        "low_score_filtered": 0,
        "uncovered_by_cat": defaultdict(int),
    }

    for fname in sorted(all_images):
        gt_list = gt_anns.get(fname, [])
        preds = preds_by_img.get(fname, [])
        stats["total_gt"] += len(gt_list)

        # Step 1: Filter low-score predictions
        good_preds = []
        for p in preds:
            if p.get("score", 0) >= args.score_thresh:
                good_preds.append(p)
            else:
                stats["low_score_filtered"] += 1

        # Step 2: Match predictions to GT
        matched_gt_idx = set()
        pred_keep = []

        # Sort preds by score descending
        good_preds.sort(key=lambda x: x.get("score", 0), reverse=True)

        for pred in good_preds:
            pbox = pred.get("box", [])
            if len(pbox) != 4:
                continue
            best_iou = 0
            best_gt_idx = -1
            for gi, gt in enumerate(gt_list):
                if gi in matched_gt_idx:
                    continue
                iou = box_iou_coco_xyxy(gt["bbox"], pbox)
                if iou > best_iou:
                    best_iou = iou
                    best_gt_idx = gi

            if best_iou >= args.iou_match:
                matched_gt_idx.add(best_gt_idx)
                stats["matched_gt"] += 1
                # Refine: align category with GT
                gt = gt_list[best_gt_idx]
                refined_pred = dict(pred)
                refined_pred["category_id"] = gt["cat_id"]
                refined_pred["prompt"] = gt["cat_name"]
                refined_pred["gt_matched"] = True
                refined_pred["gt_iou"] = round(best_iou, 4)
                pred_keep.append(refined_pred)
            elif not args.filter_fp:
                # Keep unmatched prediction as potential false positive
                refined_pred = dict(pred)
                refined_pred["gt_matched"] = False
                pred_keep.append(refined_pred)
            else:
                stats["filtered_fp"] += 1

        # Step 3: Supplement uncovered GT objects
        for gi, gt in enumerate(gt_list):
            if gi not in matched_gt_idx:
                stats["uncovered_by_cat"][gt["cat_name"]] += 1
                if args.supplement:
                    rel_path = f"{dataset}/test/{fname}"
                    supp = {
                        "dataset": dataset,
                        "image_path": str(
                            list(gt_info["images"].values())[0].get("__path__", "")
                            if isinstance(gt_info["images"].get(fname), dict)
                            else ""
                        ),
                        "rel_path": rel_path,
                        "category_id": gt["cat_id"],
                        "prompt": gt["cat_name"],
                        "score": 1.0,
                        "box": coco_to_xyxy(gt["bbox"]),
                        "supplemented": True,
                        "gt_area": gt["area"],
                    }
                    # Fix image_path properly
                    supp["image_path"] = str(
                        Path(args.data_root) / dataset / "test" / fname
                    )
                    pred_keep.append(supp)
                    stats["supplemented"] += 1

        stats["kept_preds"] += len(pred_keep)
        refined.extend(pred_keep)

    return refined, stats


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    pred_root = Path(args.pred_root)
    output_root = Path(args.output)
    output_root.mkdir(parents=True, exist_ok=True)

    print("Loading GT annotations...")
    gt_all = load_gt(data_root)
    print("Loading predictions...")
    pred_all = load_predictions(pred_root)

    all_refined = []
    all_stats = {}
    summary_lines = []
    header = f"{'Dataset':<55} {'GT':>5} {'Matched':>8} {'Suppl':>6} {'FP_del':>7} {'Kept':>6} {'Recall':>8} {'NewRec':>8}"
    summary_lines.append(header)
    summary_lines.append("=" * len(header))

    for dataset in sorted(gt_all.keys()):
        gt_info = gt_all[dataset]
        preds_by_img = pred_all.get(dataset, {})
        refined, stats = refine_dataset(dataset, gt_info, preds_by_img, args)
        all_refined.extend(refined)
        all_stats[dataset] = stats

        new_recall = (stats["matched_gt"] + stats["supplemented"]) / max(stats["total_gt"], 1)
        old_recall = stats["matched_gt"] / max(stats["total_gt"], 1)
        summary_lines.append(
            f"{dataset:<55} {stats['total_gt']:>5} {stats['matched_gt']:>8} "
            f"{stats['supplemented']:>6} {stats['filtered_fp']:>7} "
            f"{stats['kept_preds']:>6} {old_recall:>7.1%} {new_recall:>7.1%}"
        )

    # Write refined predictions
    out_jsonl = output_root / "predictions_all.jsonl"
    with out_jsonl.open("w") as f:
        for row in all_refined:
            # Remove internal fields before writing
            clean = {k: v for k, v in row.items() if k not in {"gt_matched", "gt_iou", "gt_area"}}
            f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    # Per-dataset files
    by_dataset = defaultdict(list)
    for row in all_refined:
        by_dataset[row["dataset"]].append(row)
    for ds, rows in by_dataset.items():
        ds_dir = output_root / ds
        ds_dir.mkdir(parents=True, exist_ok=True)
        with (ds_dir / "predictions_refined.jsonl").open("w") as f:
            for row in rows:
                clean = {k: v for k, v in row.items() if k not in {"gt_matched", "gt_iou", "gt_area"}}
                f.write(json.dumps(clean, ensure_ascii=False) + "\n")

    # Stats report
    report = {
        "settings": {
            "score_thresh": args.score_thresh,
            "iou_match": args.iou_match,
            "supplement": args.supplement,
            "filter_fp": args.filter_fp,
        },
        "datasets": {},
    }
    for ds, stats in all_stats.items():
        ds_report = {
            "total_gt": stats["total_gt"],
            "matched_gt": stats["matched_gt"],
            "supplemented": stats["supplemented"],
            "filtered_fp": stats["filtered_fp"],
            "low_score_filtered": stats["low_score_filtered"],
            "kept_preds": stats["kept_preds"],
            "old_recall": round(stats["matched_gt"] / max(stats["total_gt"], 1), 4),
            "new_recall": round(
                (stats["matched_gt"] + stats["supplemented"]) / max(stats["total_gt"], 1), 4
            ),
            "uncovered_by_cat": dict(stats["uncovered_by_cat"]),
        }
        report["datasets"][ds] = ds_report

    total_gt = sum(s["total_gt"] for s in all_stats.values())
    total_matched = sum(s["matched_gt"] for s in all_stats.values())
    total_supp = sum(s["supplemented"] for s in all_stats.values())
    total_fp = sum(s["filtered_fp"] for s in all_stats.values())
    report["overall"] = {
        "total_gt": total_gt,
        "total_matched": total_matched,
        "total_supplemented": total_supp,
        "total_filtered_fp": total_fp,
        "old_recall": round(total_matched / max(total_gt, 1), 4),
        "new_recall": round((total_matched + total_supp) / max(total_gt, 1), 4),
    }

    (output_root / "refinement_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )

    print("\n" + "\n".join(summary_lines))
    print(f"\nOverall: {total_gt} GT, matched={total_matched}, supplemented={total_supp}, "
          f"filtered_fp={total_fp}")
    print(f"Old recall: {report['overall']['old_recall']:.1%}")
    print(f"New recall: {report['overall']['new_recall']:.1%}")
    print(f"\nOutput: {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
