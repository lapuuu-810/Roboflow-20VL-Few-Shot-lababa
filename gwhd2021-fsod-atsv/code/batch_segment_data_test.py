#!/usr/bin/env python3
"""Batch SAM3 segmentation for the few_shot data_test split on multiple GPUs."""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import shutil
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image, ImageDraw


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
DEFAULT_ALL_PROMPTS = (
    "object",
    "thing",
    "item",
    "person",
    "animal",
    "vehicle",
    "product",
    "food",
    "plant",
    "tool",
    "equipment",
    "text",
    "sign",
    "part",
)


@dataclass(frozen=True)
class ImageTask:
    dataset: str
    image_path: str
    rel_path: str
    categories: tuple[tuple[int, str, str], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data-root",
        default="/data/LPP/cvpr/few_shot/code/data_test",
        help="Root containing */test/_annotations.coco.json and test images.",
    )
    parser.add_argument(
        "--sam3-root",
        default="/data/LPP/cvpr/few_shot/sam3-main",
        help="SAM3 repo root.",
    )
    parser.add_argument(
        "--checkpoint",
        default="/data/LPP/cvpr/few_shot/sam3-main/sam3_weight/sam3.pt",
        help="Local SAM3 checkpoint.",
    )
    parser.add_argument(
        "--output",
        default="/data/LPP/cvpr/few_shot/sam3-main/sam3_data_test_seg_results",
        help="Directory for masks, overlays, and JSONL predictions.",
    )
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=0.35)
    parser.add_argument(
        "--mode",
        choices=["categories", "all"],
        default="categories",
        help="categories uses COCO category names; all uses generic class-agnostic prompts.",
    )
    parser.add_argument(
        "--all-prompts",
        default=",".join(DEFAULT_ALL_PROMPTS),
        help="Comma-separated prompts for --mode all.",
    )
    parser.add_argument("--dedup-iou", type=float, default=0.82)
    parser.add_argument("--min-mask-area", type=int, default=16)
    parser.add_argument("--max-dets-per-query", type=int, default=-1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--preserve-output", action="store_true")
    parser.add_argument("--rerun-existing", action="store_true")
    parser.add_argument(
        "--image-list",
        default=None,
        help="Optional newline-separated rel paths like dataset/test/image.jpg.",
    )
    parser.add_argument(
        "--prompt-alias-file",
        default=None,
        help="Optional JSON mapping dataset -> category name -> prompt aliases.",
    )
    parser.add_argument("--limit-images", type=int, default=0)
    return parser.parse_args()


def load_tasks(
    data_root: Path,
    limit_images: int = 0,
    mode: str = "categories",
    all_prompts: tuple[str, ...] = DEFAULT_ALL_PROMPTS,
    image_filter: set[str] | None = None,
    prompt_aliases: dict | None = None,
) -> list[ImageTask]:
    tasks: list[ImageTask] = []
    for ann_path in sorted(data_root.glob("*/test/_annotations.coco.json")):
        dataset = ann_path.parents[1].name
        with ann_path.open() as f:
            coco = json.load(f)
        if mode == "all":
            categories = tuple(
                (idx + 1, prompt, "all_objects") for idx, prompt in enumerate(all_prompts)
            )
        else:
            ann_counts = collections.Counter(
                int(a["category_id"]) for a in coco.get("annotations", [])
            )
            category_rows = []
            for c in coco.get("categories", []):
                category_id = int(c.get("id", -1))
                if category_id == 0 or ann_counts.get(category_id, 0) <= 0:
                    continue
                category_name = str(c["name"])
                prompts = [normalize_prompt(category_name)]
                prompts.extend(
                    prompt_aliases.get(dataset, {}).get(category_name, [])
                    if prompt_aliases
                    else []
                )
                seen_prompts = set()
                for prompt in prompts:
                    prompt = normalize_prompt(prompt)
                    if prompt in seen_prompts:
                        continue
                    seen_prompts.add(prompt)
                    category_rows.append((category_id, prompt, category_name))
            categories = tuple(category_rows)
        if not categories:
            continue
        image_names = [img["file_name"] for img in coco.get("images", [])]
        if not image_names:
            image_names = [
                p.name for p in sorted(ann_path.parent.iterdir()) if p.suffix.lower() in IMAGE_EXTS
            ]
        for image_name in image_names:
            image_path = ann_path.parent / image_name
            if not image_path.exists():
                continue
            rel_path = f"{dataset}/test/{image_name}"
            if image_filter is not None and rel_path not in image_filter:
                continue
            tasks.append(ImageTask(dataset, str(image_path), rel_path, categories))
    tasks.sort(key=lambda t: t.rel_path)
    if limit_images > 0:
        tasks = tasks[:limit_images]
    return tasks


def normalize_prompt(name: str) -> str:
    name = name.strip().replace("_", " ").replace("-", " ")
    if len(name) == 1 and name.isdigit():
        return f"digit {name}"
    return name


def chunks(items: list[ImageTask], size: int) -> Iterable[list[ImageTask]]:
    for start in range(0, len(items), size):
        yield items[start : start + size]


def create_datapoint_helpers():
    from sam3.train.data.sam3_image_dataset import (
        Datapoint,
        FindQueryLoaded,
        Image as SAMImage,
        InferenceMetadata,
    )

    counter = {"value": 1}

    def make_datapoint(task: ImageTask, image: Image.Image, query_map: dict[int, dict]):
        datapoint = Datapoint(find_queries=[], images=[])
        width, height = image.size
        datapoint.images = [SAMImage(data=image, objects=[], size=[height, width])]
        for category_id, prompt, category_name in task.categories:
            query_id = counter["value"]
            counter["value"] += 1
            query_map[query_id] = {
                "dataset": task.dataset,
                "image_path": task.image_path,
                "rel_path": task.rel_path,
                "category_id": category_id,
                "category_name": category_name,
                "prompt": prompt,
            }
            datapoint.find_queries.append(
                FindQueryLoaded(
                    query_text=prompt,
                    image_id=0,
                    object_ids_output=[],
                    is_exhaustive=True,
                    query_processing_order=0,
                    inference_metadata=InferenceMetadata(
                        coco_image_id=query_id,
                        original_image_id=query_id,
                        original_category_id=category_id,
                original_size=[height, width],
                        object_id=0,
                        frame_index=0,
                    ),
                )
            )
        return datapoint

    return make_datapoint


def safe_stem(path: str) -> str:
    return Path(path).stem.replace("/", "_")


def color_for_index(index: int) -> tuple[int, int, int]:
    rng = random.Random(index + 20260511)
    return tuple(rng.randint(32, 255) for _ in range(3))


def save_result_group(
    output_root: Path,
    task: ImageTask,
    image: Image.Image,
    records: list[dict],
    predictions_name: str,
):
    dataset_dir = output_root / task.dataset
    mask_dir = dataset_dir / "masks" / safe_stem(task.image_path)
    overlay_dir = dataset_dir / "overlays"
    mask_dir.mkdir(parents=True, exist_ok=True)
    overlay_dir.mkdir(parents=True, exist_ok=True)

    overlay = image.convert("RGBA")
    overlay_layer = Image.new("RGBA", overlay.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay_layer)
    instance_index = 0

    for record in records:
        masks = record.pop("_masks")
        for local_idx, mask in enumerate(masks):
            mask_np = mask.astype(np.uint8) * 255
            if mask_np.shape != (overlay.height, overlay.width):
                mask_np = np.array(
                    Image.fromarray(mask_np).resize(overlay.size, resample=Image.Resampling.NEAREST)
                )
            mask_name = (
                f"{record['category_id']:03d}_{safe_stem(record['prompt'])}_"
                f"{instance_index:04d}.png"
            )
            mask_path = mask_dir / mask_name
            Image.fromarray(mask_np).save(mask_path)

            color = color_for_index(instance_index)
            colored = Image.new("RGBA", overlay.size, (*color, 96))
            overlay_layer.paste(colored, (0, 0), Image.fromarray(mask_np))
            box = record["boxes"][local_idx]
            draw.rectangle(box, outline=(*color, 220), width=2)

            saved = dict(record)
            saved["score"] = record["scores"][local_idx]
            saved["box"] = box
            del saved["scores"]
            del saved["boxes"]
            saved["mask_path"] = str(mask_path)
            append_jsonl(dataset_dir / predictions_name, saved)
            instance_index += 1

    overlay.alpha_composite(overlay_layer)
    overlay.convert("RGB").save(overlay_dir / f"{safe_stem(task.image_path)}.jpg", quality=92)


def mask_iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    intersection = np.logical_and(mask_a, mask_b).sum()
    if intersection == 0:
        return 0.0
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / max(union, 1))


def dedup_records(records: list[dict], iou_threshold: float, min_area: int) -> list[dict]:
    instances: list[dict] = []
    for record in records:
        for score, box, mask in zip(record["scores"], record["boxes"], record["_masks"]):
            area = int(mask.sum())
            if area < min_area:
                continue
            item = {k: v for k, v in record.items() if k not in {"scores", "boxes", "_masks"}}
            item.update(score=float(score), box=box, mask=mask, area=area)
            instances.append(item)

    instances.sort(key=lambda item: item["score"], reverse=True)
    kept: list[dict] = []
    for item in instances:
        if any(mask_iou(item["mask"], prev["mask"]) >= iou_threshold for prev in kept):
            continue
        kept.append(item)

    deduped: list[dict] = []
    for idx, item in enumerate(kept):
        row = {k: v for k, v in item.items() if k not in {"score", "box", "mask", "area"}}
        row["category_id"] = 1
        row["prompt"] = "all_objects"
        row["source_prompt"] = item["prompt"]
        row["scores"] = [item["score"]]
        row["boxes"] = [item["box"]]
        row["_masks"] = [item["mask"]]
        deduped.append(row)
    return deduped


def append_jsonl(path: Path, row: dict):
    with path.open("a") as f:
        f.write(json.dumps(row, ensure_ascii=False) + "\n")


def worker(rank: int, gpus: list[int], task_splits: list[list[ImageTask]], args: argparse.Namespace):
    gpu = gpus[rank]
    tasks = task_splits[rank]
    device = torch.device(f"cuda:{gpu}")
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    os.environ.setdefault("USE_PERFLIB", "1")

    sys.path.insert(0, args.sam3_root)
    from sam3 import build_sam3_image_model
    from sam3.eval.postprocessors import PostProcessImage
    from sam3.model.utils.misc import copy_data_to_device
    from sam3.train.data.collator import collate_fn_api as collate
    from sam3.train.transforms.basic_for_api import (
        ComposeAPI,
        NormalizeAPI,
        RandomResizeAPI,
        ToTensorAPI,
    )

    bpe_path = str(Path(args.sam3_root) / "sam3/assets/bpe_simple_vocab_16e6.txt.gz")
    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=args.checkpoint,
        load_from_HF=False,
        device="cuda",
        compile=args.compile,
    )
    transform = ComposeAPI(
        transforms=[
            RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    postprocessor = PostProcessImage(
        max_dets_per_img=args.max_dets_per_query,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=args.threshold,
        to_cpu=True,
    )
    make_datapoint = create_datapoint_helpers()
    output_root = Path(args.output)

    done = 0
    start_time = time.time()
    for batch_tasks in chunks(tasks, args.batch_size):
        active_tasks: list[ImageTask] = []
        images: list[Image.Image] = []
        datapoints = []
        query_map: dict[int, dict] = {}
        for task in batch_tasks:
            if args.skip_existing:
                overlay_path = output_root / task.dataset / "overlays" / f"{safe_stem(task.image_path)}.jpg"
                if overlay_path.exists() and not args.rerun_existing:
                    continue
            image = Image.open(task.image_path).convert("RGB")
            active_tasks.append(task)
            images.append(image)
            datapoints.append(transform(make_datapoint(task, image, query_map)))
        if not datapoints:
            continue

        batch = collate(datapoints, dict_key="dummy")["dummy"]
        batch = copy_data_to_device(batch, device, non_blocking=True)
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            output = model(batch)
            processed = postprocessor.process_results(output, batch.find_metadatas)

        grouped: dict[str, list[dict]] = {}
        for query_id, result in processed.items():
            meta = query_map[query_id]
            scores = result["scores"].detach().cpu().float().tolist()
            boxes = result["boxes"].detach().cpu().float().tolist()
            masks_tensor = result["masks"].detach().cpu()
            if masks_tensor.ndim == 4:
                masks_tensor = masks_tensor[:, 0]
            masks = [m.numpy().astype(bool) for m in masks_tensor]
            if not masks:
                continue
            row = {
                **meta,
                "scores": scores,
                "boxes": boxes,
                "_masks": masks,
            }
            grouped.setdefault(meta["image_path"], []).append(row)

        for task, image in zip(active_tasks, images):
            records = grouped.get(task.image_path, [])
            if args.mode == "all":
                records = dedup_records(records, args.dedup_iou, args.min_mask_area)
            save_result_group(
                output_root,
                task,
                image,
                records,
                predictions_name=f"predictions_gpu{gpu}.jsonl",
            )
            done += 1
            if done % 8 == 0:
                elapsed = max(time.time() - start_time, 1e-6)
                print(
                    f"[gpu {gpu}] processed {done}/{len(tasks)} images "
                    f"({done / elapsed:.2f} img/s)",
                    flush=True,
                )


def split_tasks(tasks: list[ImageTask], n: int) -> list[list[ImageTask]]:
    return [tasks[i::n] for i in range(n)]


def main() -> int:
    args = parse_args()
    output_root = Path(args.output)
    if output_root.exists() and not args.skip_existing and not args.preserve_output:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    all_prompts = tuple(p.strip() for p in args.all_prompts.split(",") if p.strip())
    image_filter = None
    if args.image_list:
        image_filter = {
            line.strip()
            for line in Path(args.image_list).read_text().splitlines()
            if line.strip()
        }
    prompt_aliases = None
    if args.prompt_alias_file:
        prompt_aliases = json.loads(Path(args.prompt_alias_file).read_text())
    tasks = load_tasks(
        Path(args.data_root),
        args.limit_images,
        args.mode,
        all_prompts,
        image_filter,
        prompt_aliases,
    )
    gpus = [int(x) for x in args.gpus.split(",") if x.strip()]
    if args.num_workers > 0:
        gpus = gpus[: args.num_workers]
    if not tasks:
        raise RuntimeError(f"No test images found under {args.data_root}")
    if not gpus:
        raise RuntimeError("No GPUs requested")

    manifest = {
        "data_root": args.data_root,
        "checkpoint": args.checkpoint,
        "images": len(tasks),
        "gpus": gpus,
        "batch_size_per_gpu": args.batch_size,
        "threshold": args.threshold,
        "mode": args.mode,
        "all_prompts": all_prompts,
        "dedup_iou": args.dedup_iou,
        "min_mask_area": args.min_mask_area,
        "image_list": args.image_list,
        "prompt_alias_file": args.prompt_alias_file,
        "started_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with (output_root / "run_manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)

    print(
        f"Running SAM3 segmentation: {len(tasks)} images, {len(gpus)} GPUs, "
        f"batch size {args.batch_size}/GPU, output={output_root}",
        flush=True,
    )
    task_splits = split_tasks(tasks, len(gpus))
    mp.spawn(worker, args=(gpus, task_splits, args), nprocs=len(gpus), join=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
