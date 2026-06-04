#!/usr/bin/env python3
"""Analyze and fix SAM3 predictions against ground truth."""

from __future__ import annotations

import argparse
import collections
import json
import math
from pathlib import Path
from typing import Any


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data",
    )
    parser.add_argument(
        "--pred-root",
        default="/data/LPP/cvpr/few_shot/sam3-main/sam3_foundational_potential_targets_results",
    )
    parser.add_argument(
        "--output",
        default="/data/LPP/cvpr/few_shot/sam3-main/prediction_analysis.json",
    )
    return parser.parse_args()


def load_gt_data(data_root: Path) -> dict[str, dict]:
    """Load ground truth data for all datasets."""
    gt_data = {}
    for ds_dir in sorted(data_root.iterdir()):
        if not ds_dir.is_dir():
            continue
        coco_path = ds_dir / "test" / "_annotations.coco.json"
        if not coco_path.exists():
            continue

        with open(coco_path) as f:
            coco = json.load(f)

        # Count annotations per category
        cat_counts = collections.Counter()
        cat_images = collections.Counter()
        for ann in coco.get("annotations", []):
            cat_id = ann["category_id"]
            cat_counts[cat_id] += 1
            cat_images[cat_id] = ann.get("image_id", -1)

        cat_names = {c["id"]: c["name"] for c in coco.get("categories", [])}

        # Count images per category (unique images)
        cat_unique_images = collections.Counter()
        image_cats = collections.defaultdict(set)
        for ann in coco.get("annotations", []):
            image_cats[ann["image_id"]].add(ann["category_id"])
        for img_id, cats in image_cats.items():
            for cat_id in cats:
                cat_unique_images[cat_id] += 1

        gt_data[ds_dir.name] = {
            "images": len(coco.get("images", [])),
            "annotations": len(coco.get("annotations", [])),
            "categories": cat_names,
            "cat_counts": dict(cat_counts),
            "cat_images": dict(cat_unique_images),
        }

    return gt_data


def load_predictions(pred_root: Path) -> dict[str, dict]:
    """Load predictions for all datasets."""
    pred_data = {}
    for ds_dir in sorted(pred_root.iterdir()):
        if not ds_dir.is_dir() or not (ds_dir / "predictions_gpu0.jsonl").exists():
            continue

        # Load all predictions
        all_preds = []
        for gpu_file in ds_dir.glob("predictions_gpu*.jsonl"):
            with open(gpu_file) as f:
                for line in f:
                    if line.strip():
                        all_preds.append(json.loads(line))

        # Group by category/prompt
        cat_preds = collections.defaultdict(list)
        for pred in all_preds:
            cat_preds[pred["prompt"]].append(pred)

        # Count predictions per category
        cat_counts = {k: len(v) for k, v in cat_preds.items()}

        # Count unique images per category
        cat_images = collections.Counter()
        for pred in all_preds:
            cat_images[pred["prompt"]] += 1

        # Group by image
        img_preds = collections.defaultdict(list)
        for pred in all_preds:
            img_preds[pred["rel_path"]].append(pred)

        pred_data[ds_dir.name] = {
            "total_predictions": len(all_preds),
            "cat_counts": cat_counts,
            "cat_images": dict(cat_images),
            "unique_images": len(img_preds),
            "predictions": all_preds,
        }

    return pred_data


def analyze_dataset(ds_name: str, gt_info: dict, pred_info: dict) -> dict:
    """Analyze a single dataset."""
    analysis = {
        "dataset": ds_name,
        "gt_images": gt_info["images"],
        "gt_annotations": gt_info["annotations"],
        "pred_total": pred_info["total_predictions"],
        "pred_unique_images": pred_info["unique_images"],
        "prediction_ratio": round(pred_info["total_predictions"] / max(gt_info["annotations"], 1), 2),
        "category_analysis": [],
        "issues": [],
    }

    gt_cats = gt_info["categories"]
    gt_cat_counts = gt_info["cat_counts"]
    pred_cat_counts = pred_info["cat_counts"]

    # Analyze each GT category
    for cat_id, cat_name in gt_cats.items():
        gt_count = gt_cat_counts.get(cat_id, 0)
        # Find matching predictions (exact match or case-insensitive)
        pred_count = pred_cat_counts.get(cat_name, 0)
        if pred_count == 0:
            # Try case-insensitive match
            for k, v in pred_cat_counts.items():
                if k.lower() == cat_name.lower():
                    pred_count = v
                    break

        cat_analysis = {
            "category_id": cat_id,
            "category_name": cat_name,
            "gt_count": gt_count,
            "pred_count": pred_count,
            "ratio": round(pred_count / max(gt_count, 1), 2),
        }
        analysis["category_analysis"].append(cat_analysis)

        # Identify issues
        if pred_count == 0 and gt_count > 0:
            analysis["issues"].append(f"MISSING: {cat_name} has {gt_count} GT but 0 predictions")
        elif pred_count > 0 and gt_count == 0:
            analysis["issues"].append(f"EXTRA: {cat_name} has predictions but no GT")
        elif cat_analysis["ratio"] < 0.5 and gt_count >= 5:
            analysis["issues"].append(f"LOW: {cat_name} has only {pred_count}/{gt_count} predictions")
        elif cat_analysis["ratio"] > 3.0 and gt_count >= 5:
            analysis["issues"].append(f"HIGH: {cat_name} has {pred_count}/{gt_count} predictions (possible duplicates)")

    # Check for extra predictions not in GT
    gt_cat_names = set(gt_cats.values())
    for pred_cat, count in pred_cat_counts.items():
        if pred_cat not in gt_cat_names:
            analysis["issues"].append(f"EXTRA_CATEGORY: {pred_cat} has {count} predictions but is not a GT category")

    return analysis


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    pred_root = Path(args.pred_root)

    print("Loading ground truth data...")
    gt_data = load_gt_data(data_root)

    print("Loading predictions...")
    pred_data = load_predictions(pred_root)

    print("Analyzing datasets...")
    analyses = []
    for ds_name in gt_data:
        if ds_name in pred_data:
            analysis = analyze_dataset(ds_name, gt_data[ds_name], pred_data[ds_name])
            analyses.append(analysis)

    # Sort by number of issues
    analyses.sort(key=lambda x: len(x["issues"]), reverse=True)

    # Generate summary
    summary = {
        "total_datasets": len(analyses),
        "datasets_with_issues": sum(1 for a in analyses if a["issues"]),
        "total_issues": sum(len(a["issues"]) for a in analyses),
        "analyses": analyses,
    }

    # Save analysis
    with open(args.output, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(f"\nAnalysis saved to: {args.output}")
    print(f"\nSummary:")
    print(f"  Total datasets: {summary['total_datasets']}")
    print(f"  Datasets with issues: {summary['datasets_with_issues']}")
    print(f"  Total issues: {summary['total_issues']}")

    # Print top issues
    print("\nTop issues by dataset:")
    for analysis in analyses[:10]:
        if analysis["issues"]:
            print(f"\n{analysis['dataset']}:")
            for issue in analysis["issues"][:5]:
                print(f"  - {issue}")


if __name__ == "__main__":
    main()
