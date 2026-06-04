#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import contextlib
import io
import json
import os
import pickle
import re
import time
import urllib.error
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DS = "recode-waste-czvmg-fsod-yxsw"
ROOT = Path(__file__).resolve().parent
DATA = Path("/data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data") / DS
GT = DATA / "test" / "_annotations.coco.json"
KEYS = ["mAP", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large", "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
CLASS_NAMES = {
    1: "aggregate",
    2: "cardboard",
    3: "hard plastic",
    4: "metal",
    5: "soft plastic",
    6: "timber",
}
CLASS_ALIASES = {
    "aggregate": 1, "gravel": 1, "stone": 1, "stones": 1, "rubble": 1, "crushed stone": 1, "碎石": 1, "骨料": 1,
    "cardboard": 2, "paperboard": 2, "carton": 2, "硬纸板": 2,
    "hard plastic": 3, "rigid plastic": 3, "plastic container": 3, "硬质塑料": 3,
    "metal": 4, "金属": 4,
    "soft plastic": 5, "film plastic": 5, "plastic bag": 5, "柔性塑料": 5, "软质塑料": 5,
    "timber": 6, "wood": 6, "wooden piece": 6, "木材": 6,
}


def load_json(path: Path):
    return json.load(open(path))


def save_json(obj, path: Path):
    json.dump(obj, open(path, "w"), indent=2, ensure_ascii=False)


def image_url(img: Image.Image, max_side: int) -> tuple[str, tuple[int, int]]:
    img = img.convert("RGB")
    if max(img.size) > max_side:
        scale = max_side / max(img.size)
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii"), img.size


def make_prompt() -> str:
    return """
Detect all visible waste objects in the image and return tight bounding boxes.
Use only these classes:
- aggregate: gravel, crushed stone, rubble, concrete or rock fragments.
- cardboard: brown paperboard, carton pieces, corrugated cardboard, paper packaging.
- hard plastic: rigid plastic pieces, bottles, caps, containers, trays or solid plastic parts.
- metal: metallic cans, wires, foil, shiny or rusty metal parts.
- soft plastic: flexible plastic bags, film, wrappers, sheets or crumpled thin plastic.
- timber: wood pieces, sticks, boards, branches or other wooden material.
Return only JSON: {"detections":[{"bbox_2d":[x1,y1,x2,y2],"label":"hard plastic","confidence":0.90}]}.
Coordinates must be integers from 0 to 1000.
""".strip()


def call_api(img: Image.Image, api_key: str, model: str, timeout: int, retries: int, max_side: int) -> tuple[str, tuple[int, int]]:
    url, sent_size = image_url(img, max_side)
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": [{"type": "text", "text": make_prompt()}, {"type": "image_url", "image_url": {"url": url}}]}],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "enable_thinking": False,
        "thinking": {"type": "disabled"},
    }
    data = json.dumps(payload).encode("utf-8")
    last = None
    for attempt in range(retries + 1):
        req = urllib.request.Request(
            "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
            data=data,
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                obj = json.loads(resp.read().decode("utf-8"))
            return obj["choices"][0]["message"]["content"], sent_size
        except urllib.error.HTTPError as e:
            last = RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8', errors='replace')}")
        except Exception as e:
            last = e
        time.sleep(2 + attempt * 2)
    raise RuntimeError(last)


def parse_json(text: str):
    text = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, flags=re.S)
        if not m:
            raise
        return json.loads(m.group(0))


def label_to_id(label) -> int | None:
    if label is None:
        return None
    if isinstance(label, int) and label in CLASS_NAMES:
        return label
    text = str(label).strip().lower().replace("_", "-").replace("  ", " ")
    text = text.replace("-", " ")
    if text in CLASS_ALIASES:
        return CLASS_ALIASES[text]
    for name, cid in CLASS_ALIASES.items():
        if name in text:
            return cid
    return None


def norm1000_to_xywh(box, width: int, height: int):
    if not isinstance(box, list) or len(box) != 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in box]
    x1 = x1 / 1000.0 * width
    x2 = x2 / 1000.0 * width
    y1 = y1 / 1000.0 * height
    y2 = y2 / 1000.0 * height
    if x2 <= x1 or y2 <= y1:
        return None
    x1 = max(0.0, min(width - 1.0, x1))
    y1 = max(0.0, min(height - 1.0, y1))
    x2 = max(x1 + 1.0, min(float(width), x2))
    y2 = max(y1 + 1.0, min(float(height), y2))
    return [round(x1, 2), round(y1, 2), round(x2 - x1, 2), round(y2 - y1, 2)]


def xywh_to_xyxy(box):
    x, y, w, h = map(float, box)
    return [x, y, x + w, y + h]


def iou(a, b):
    a = xywh_to_xyxy(a["bbox"])
    b = xywh_to_xyxy(b["bbox"])
    inter = max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / (aa + bb - inter + 1e-9)


def nms(preds, thr: float):
    groups = defaultdict(list)
    for p in preds:
        groups[(int(p["image_id"]), int(p["category_id"]))].append(p)
    out = []
    for rows in groups.values():
        keep = []
        for p in sorted(rows, key=lambda r: float(r["score"]), reverse=True):
            if all(iou(p, q) < thr for q in keep):
                keep.append(p)
        out.extend(keep)
    return sorted(out, key=lambda p: (int(p["image_id"]), -float(p["score"])))


def evaluate(preds, out_dir: Path, image_ids: list[int] | None = None):
    if not preds:
        return {k: 0.0 for k in KEYS}
    gt = load_json(GT)
    if image_ids is not None:
        keep = set(int(i) for i in image_ids)
        gt["images"] = [im for im in gt["images"] if int(im["id"]) in keep]
        gt["annotations"] = [ann for ann in gt["annotations"] if int(ann["image_id"]) in keep]
        subset_path = out_dir / "subset_gt.json"
        save_json(gt, subset_path)
        gt_path = subset_path
    else:
        gt_path = GT
        keep = None
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(gt_path))
        dt = coco.loadRes(preds)
        ev = COCOeval(coco, dt, "bbox")
        if keep is not None:
            ev.params.imgIds = sorted(keep)
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {k: float(v) for k, v in zip(KEYS, ev.stats)}


def write_pkl(preds, image_by, path: Path):
    by = defaultdict(list)
    for p in preds:
        by[int(p["image_id"])].append({"image_id": int(p["image_id"]), "category_id": int(p["category_id"]) - 1, "bbox": np.array(p["bbox"], dtype=np.float32), "score": float(p["score"])})
    sub = [{"image_id": int(im["id"]), "instances": by.get(int(im["id"]), [])} for im in image_by.values()]
    pickle.dump(sub, open(path, "wb"), protocol=4)


def parse_image_ids(text: str | None, all_ids: list[int]):
    if not text:
        return all_ids
    out = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = [int(x) for x in part.split("-", 1)]
            out.extend(range(a, b + 1))
        else:
            out.append(int(part))
    keep = set(all_ids)
    return [i for i in out if i in keep]


def run_qwen(args):
    api_key = args.api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
    if not api_key:
        raise SystemExit("Set DASHSCOPE_API_KEY/QWEN_API_KEY or pass --api-key")
    gt = load_json(GT)
    image_by = {int(im["id"]): im for im in gt["images"]}
    ids = parse_image_ids(args.image_ids, sorted(image_by))
    raw_dir = args.out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    records = []
    done = {int(r["image_id"]): r for r in load_json(args.out_dir / "qwen_records.json")} if args.resume and (args.out_dir / "qwen_records.json").exists() else {}
    for idx, iid in enumerate(ids, 1):
        if iid in done:
            records.append(done[iid])
            continue
        im = image_by[iid]
        img = Image.open(DATA / "test" / im["file_name"]).convert("RGB")
        raw, sent_size = call_api(img, api_key, args.model, args.timeout, args.retries, args.max_side)
        rec = {"image_id": iid, "file_name": im["file_name"], "raw": raw, "sent_size": list(sent_size)}
        try:
            rec["parsed"] = parse_json(raw)
        except Exception as e:
            rec["parse_error"] = repr(e)
            rec["parsed"] = {"detections": []}
        records.append(rec)
        save_json(rec, raw_dir / f"{iid:04d}_raw.json")
        save_json(records, args.out_dir / "qwen_records.json")
        print(f"[{idx}/{len(ids)}] image {iid}: raw chars={len(raw)}", flush=True)
    save_json(records, args.out_dir / "qwen_records.json")
    return records, image_by


def convert_records(records, image_by, args):
    preds = []
    audit = []
    for rec in records:
        iid = int(rec["image_id"])
        im = image_by[iid]
        parsed = rec.get("parsed") or {}
        if isinstance(parsed, list):
            dets = parsed
        elif isinstance(parsed, dict):
            dets = parsed.get("detections", [])
        else:
            dets = []
        if not isinstance(dets, list):
            dets = []
        for det in dets:
            if not isinstance(det, dict):
                audit.append({"image_id": iid, "skip": "non_dict_detection", "det": det})
                continue
            cid = label_to_id(det.get("label", det.get("category_name", det.get("category"))))
            if cid is None:
                audit.append({"image_id": iid, "skip": "unknown_label", "det": det})
                continue
            box = det.get("bbox_2d", det.get("bbox"))
            bbox = norm1000_to_xywh(box, int(im["width"]), int(im["height"]))
            if not bbox:
                audit.append({"image_id": iid, "skip": "bad_box", "det": det})
                continue
            score = float(det.get("confidence", det.get("score", args.default_score)))
            if score < args.score_thr:
                continue
            preds.append({"image_id": iid, "category_id": cid, "bbox": bbox, "score": round(max(0.001, min(0.999, score)), 6)})
    preds = nms(preds, args.nms_thr)
    return preds, audit


def visualize(preds, image_by, out_dir: Path, count: int):
    by = defaultdict(list)
    for p in preds:
        by[int(p["image_id"])].append(p)
    colors = {1: (120, 120, 120), 2: (220, 170, 30), 3: (60, 140, 230), 4: (210, 50, 50), 5: (180, 70, 210), 6: (120, 80, 35)}
    vis = out_dir / "vis"
    vis.mkdir(exist_ok=True)
    thumbs = []
    for iid in sorted(image_by)[:count]:
        im = image_by[iid]
        img = Image.open(DATA / "test" / im["file_name"]).convert("RGB")
        draw = ImageDraw.Draw(img)
        for p in by.get(iid, [])[:160]:
            x, y, w, h = p["bbox"]
            color = colors.get(int(p["category_id"]), (255, 0, 255))
            draw.rectangle([x, y, x + w, y + h], outline=color, width=2)
            draw.text((x, max(0, y - 12)), f"{p['category_id']}:{p['score']:.2f}", fill=color)
        img.thumbnail((900, 900))
        op = vis / f"{iid:04d}.jpg"
        img.save(op, quality=92)
        tile = img.copy()
        tile.thumbnail((320, 220))
        thumbs.append((iid, tile))
    if not thumbs:
        return
    cols = 4
    rows = (len(thumbs) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * 320, rows * 245), (245, 245, 245))
    draw = ImageDraw.Draw(sheet)
    for i, (iid, tile) in enumerate(thumbs):
        x = (i % cols) * 320
        y = (i // cols) * 245
        draw.text((x + 4, y + 4), f"image {iid}", fill=(0, 0, 0))
        sheet.paste(tile, (x, y + 22))
    sheet.save(out_dir / "contact_sheet.jpg", quality=92)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=ROOT / "qwen_direct_bbox")
    parser.add_argument("--api-key", default=None)
    parser.add_argument("--model", default="qwen3.5-plus")
    parser.add_argument("--image-ids", default=None)
    parser.add_argument("--max-side", type=int, default=1600)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-api", action="store_true")
    parser.add_argument("--score-thr", type=float, default=0.0)
    parser.add_argument("--default-score", type=float, default=0.70)
    parser.add_argument("--nms-thr", type=float, default=0.45)
    parser.add_argument("--vis", action="store_true")
    parser.add_argument("--vis-count", type=int, default=24)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    gt = load_json(GT)
    image_by = {int(im["id"]): im for im in gt["images"]}
    if args.skip_api:
        records = load_json(args.out_dir / "qwen_records.json")
    else:
        records, image_by = run_qwen(args)
    preds, audit = convert_records(records, image_by, args)
    save_json(preds, args.out_dir / "predictions.json")
    save_json(audit, args.out_dir / "audit.json")
    eval_image_ids = sorted({int(r["image_id"]) for r in records})
    metrics = evaluate(preds, args.out_dir, eval_image_ids)
    metrics.update({"images": len(eval_image_ids), "total_test_images": len(image_by), "records": len(records), "image_ids": eval_image_ids, "predictions": len(preds), "params": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "counts": dict(Counter(int(p["category_id"]) for p in preds))})
    save_json(metrics, args.out_dir / "eval.json")
    write_pkl(preds, image_by, args.out_dir / f"{DS}.pkl")
    if args.vis:
        visualize(preds, image_by, args.out_dir, args.vis_count)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
