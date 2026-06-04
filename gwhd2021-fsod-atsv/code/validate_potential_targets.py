#!/usr/bin/env python3
"""Validate SAM3 potential-target outputs against COCO test image lists."""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data",
    )
    parser.add_argument(
        "--output",
        default="/data/LPP/cvpr/few_shot/sam3-main/sam3_foundational_potential_targets_results",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    output_root = Path(args.output)

    expected = []
    for ann_path in sorted(data_root.glob("*/test/_annotations.coco.json")):
        dataset = ann_path.parents[1].name
        with ann_path.open() as f:
            coco = json.load(f)
        for image in coco.get("images", []):
            expected.append((dataset, image["file_name"]))

    rows_by_rel = collections.Counter()
    rows_by_dataset = collections.Counter()
    for pred_path in output_root.glob("*/predictions_gpu*.jsonl"):
        for line in pred_path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            rows_by_rel[row["rel_path"]] += 1
            rows_by_dataset[row["dataset"]] += 1

    missing_overlay = []
    empty_prediction = []
    for dataset, file_name in expected:
        stem = Path(file_name).stem
        overlay = output_root / dataset / "overlays" / f"{stem}.jpg"
        rel_path = f"{dataset}/test/{file_name}"
        if not overlay.exists():
            missing_overlay.append(rel_path)
        elif rows_by_rel[rel_path] == 0:
            empty_prediction.append(rel_path)

    report = {
        "expected_images": len(expected),
        "overlay_count": len(list(output_root.glob("*/overlays/*.jpg"))),
        "mask_count": len(list(output_root.glob("*/masks/*/*.png"))),
        "prediction_rows": sum(rows_by_rel.values()),
        "missing_overlay_count": len(missing_overlay),
        "empty_prediction_count": len(empty_prediction),
        "missing_overlay": missing_overlay,
        "empty_prediction": empty_prediction,
        "empty_by_dataset": dict(collections.Counter(p.split("/")[0] for p in empty_prediction)),
        "prediction_rows_by_dataset": dict(rows_by_dataset),
    }

    (output_root / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2)
    )
    (output_root / "missing_overlay_images.txt").write_text(
        "\n".join(missing_overlay) + ("\n" if missing_overlay else "")
    )
    (output_root / "empty_prediction_images.txt").write_text(
        "\n".join(empty_prediction) + ("\n" if empty_prediction else "")
    )

    print(json.dumps({k: report[k] for k in [
        "expected_images",
        "overlay_count",
        "mask_count",
        "prediction_rows",
        "missing_overlay_count",
        "empty_prediction_count",
        "empty_by_dataset",
    ]}, ensure_ascii=False, indent=2))
    print(f"report={output_root / 'validation_report.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
