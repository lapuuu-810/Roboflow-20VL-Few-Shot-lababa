#!/usr/bin/env python3
"""Filter and post-process SAM3 predictions."""

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
        default="/data/LPP/cvpr/few_shot/sam3-main/sam3_filtered_predictions",
    )
    parser.add_argument("--min-score", type=float, default=0.15, help="Minimum confidence score")
    parser.add_argument("--max-predictions-per-image", type=int, default=500, help="Max predictions per image")
    parser.add_argument(
        "--target-ratio",
        type=float,
        default=3.0,
        help="Target ratio of predictions to GT annotations per category",
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
        for ann in coco.get("annotations", []):
            cat_counts[ann["category_id"]] += 1

        cat_names = {c["id"]: c["name"] for c in coco.get("categories", [])}
        gt_cat_names = set(cat_names.values())

        gt_data[ds_dir.name] = {
            "images": len(coco.get("images", [])),
            "annotations": len(coco.get("annotations", [])),
            "categories": cat_names,
            "cat_counts": dict(cat_counts),
            "gt_cat_names": gt_cat_names,
        }

    return gt_data


def normalize_category_name(name: str) -> str:
    """Normalize category name for matching."""
    return name.lower().strip().replace("_", " ").replace("-", " ")


def find_matching_gt_category(pred_prompt: str, gt_cat_names: set[str]) -> str | None:
    """Find matching GT category for a prediction prompt."""
    normalized_prompt = normalize_category_name(pred_prompt)

    # Exact match
    if pred_prompt in gt_cat_names:
        return pred_prompt

    # Case-insensitive match
    for gt_name in gt_cat_names:
        if normalize_category_name(gt_name) == normalized_prompt:
            return gt_name

    # Partial match
    for gt_name in gt_cat_names:
        if normalized_prompt in normalize_category_name(gt_name) or normalize_category_name(gt_name) in normalized_prompt:
            return gt_name

    return None


def calculate_dynamic_threshold(
    gt_count: int, pred_count: int, target_ratio: float
) -> float:
    """Calculate dynamic score threshold based on prediction/GT ratio."""
    if gt_count == 0 or pred_count == 0:
        return 0.0

    ratio = pred_count / gt_count
    if ratio <= target_ratio:
        return 0.0  # No filtering needed

    # Very conservative - only filter when ratio is way over target
    excess_ratio = ratio / target_ratio
    if excess_ratio < 2.0:
        return 0.0  # Don't filter if less than 2x over target
    threshold = min(0.3, 0.1 * math.log2(excess_ratio))
    return max(0.0, min(0.3, threshold))


def filter_predictions_for_dataset(
    ds_name: str,
    predictions: list[dict],
    gt_info: dict,
    min_score: float,
    max_per_image: int,
    target_ratio: float,
) -> list[dict]:
    """Filter predictions for a single dataset."""
    gt_cat_names = gt_info["gt_cat_names"]
    gt_cat_counts = gt_info["cat_counts"]
    cat_names = gt_info["categories"]

    # Group predictions by category
    pred_by_category = collections.defaultdict(list)
    for pred in predictions:
        prompt = pred["prompt"]
        gt_match = find_matching_gt_category(prompt, gt_cat_names)
        if gt_match:
            pred_by_category[gt_match].append(pred)

    filtered = []
    for gt_cat_name, preds in pred_by_category.items():
        # Find GT count for this category
        gt_count = 0
        for cat_id, name in cat_names.items():
            if name == gt_cat_name:
                gt_count = gt_cat_counts.get(cat_id, 0)
                break

        # Calculate dynamic threshold
        dynamic_threshold = calculate_dynamic_threshold(
            gt_count, len(preds), target_ratio
        )

        # Sort by score
        preds.sort(key=lambda x: x["score"], reverse=True)

        # Apply dynamic threshold - only use dynamic_threshold if it's higher than min_score
        kept = []
        for pred in preds:
            effective_threshold = max(min_score, dynamic_threshold)
            if pred["score"] >= effective_threshold:
                kept.append(pred)

        # Limit number of predictions
        if len(kept) > max_per_image:
            kept = kept[:max_per_image]

        filtered.extend(kept)

    # Group by image and apply per-image limit
    by_image = collections.defaultdict(list)
    for pred in filtered:
        by_image[pred["rel_path"]].append(pred)

    final_filtered = []
    for rel_path, preds in by_image.items():
        preds.sort(key=lambda x: x["score"], reverse=True)
        final_filtered.extend(preds[:max_per_image])

    return final_filtered


def main():
    args = parse_args()
    data_root = Path(args.data_root)
    pred_root = Path(args.pred_root)
    output_root = Path(args.output)

    # Create output directory
    output_root.mkdir(parents=True, exist_ok=True)

    print("Loading ground truth data...")
    gt_data = load_gt_data(data_root)

    # Process each dataset
    for ds_dir in sorted(pred_root.iterdir()):
        if not ds_dir.is_dir():
            continue

        ds_name = ds_dir.name
        if ds_name not in gt_data:
            continue

        print(f"\nProcessing {ds_name}...")

        # Load all predictions
        all_preds = []
        for gpu_file in ds_dir.glob("predictions_gpu*.jsonl"):
            with open(gpu_file) as f:
                for line in f:
                    if line.strip():
                        all_preds.append(json.loads(line))

        if not all_preds:
            print(f"  No predictions found, skipping...")
            continue

        # Filter predictions
        filtered = filter_predictions_for_dataset(
            ds_name,
            all_preds,
            gt_data[ds_name],
            args.min_score,
            args.max_predictions_per_image,
            args.target_ratio,
        )

        print(f"  Original: {len(all_preds)}, Filtered: {len(filtered)}")

        # Save filtered predictions
        ds_output = output_root / ds_name
        ds_output.mkdir(parents=True, exist_ok=True)

        with open(ds_output / "filtered_predictions.jsonl", "w") as f:
            for pred in filtered:
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")

        # Generate summary
        summary = {
            "dataset": ds_name,
            "original_count": len(all_preds),
            "filtered_count": len(filtered),
            "category_counts": collections.Counter(p["prompt"] for p in filtered),
        }

        with open(ds_output / "filter_summary.json", "w") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    # Generate overall summary
    overall_summary = {"datasets": {}}
    total_original = 0
    total_filtered = 0

    for ds_dir in sorted(output_root.iterdir()):
        if not ds_dir.is_dir():
            continue
        summary_path = ds_dir / "filter_summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            overall_summary["datasets"][ds_dir.name] = {
                "original": summary["original_count"],
                "filtered": summary["filtered_count"],
                "reduction": round(
                    1 - summary["filtered_count"] / max(summary["original_count"], 1), 2
                ),
            }
            total_original += summary["original_count"]
            total_filtered += summary["filtered_count"]

    overall_summary["total_original"] = total_original
    overall_summary["total_filtered"] = total_filtered
    overall_summary["overall_reduction"] = round(
        1 - total_filtered / max(total_original, 1), 2
    )

    with open(output_root / "overall_summary.json", "w") as f:
        json.dump(overall_summary, f, indent=2, ensure_ascii=False)

    print(f"\nOverall Summary:")
    print(f"  Total original: {total_original}")
    print(f"  Total filtered: {total_filtered}")
    print(f"  Reduction: {overall_summary['overall_reduction']:.1%}")


if __name__ == "__main__":
    main()
