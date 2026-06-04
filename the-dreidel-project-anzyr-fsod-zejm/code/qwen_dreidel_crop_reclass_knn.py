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

import cv2
import numpy as np
from PIL import Image, ImageDraw
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import dreidel_sam_candidate_pipeline as sam_pipe

ROOT = Path(__file__).resolve().parent
DS = "the-dreidel-project-anzyr-fsod-zejm"
DATA = Path("/data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data") / DS
GT = DATA / "test/_annotations.coco.json"
TRAIN = DATA / "train/_annotations.coco.json"
VALID = DATA / "valid/_annotations.coco.json"
METRIC_KEYS = [
    "mAP", "mAP50", "mAP75", "mAP_small", "mAP_medium", "mAP_large",
    "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large",
]

ID_TO_CODE = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F"}
CODE_TO_ID = {v: k for k, v in ID_TO_CODE.items()}
CODE_TEXT = {
    "A": "四面小陀螺主体，静止或清晰可见的完整/部分陀螺，不是单独字母。",
    "B": "吉梅尔面，陀螺上的希伯来字母 ג，对应 get all。",
    "C": "海伊面，陀螺上的希伯来字母 ה，对应 take half。",
    "D": "努恩面，陀螺上的希伯来字母 נ，对应 nothing。",
    "E": "辛恩面，陀螺上的希伯来字母 ש，对应 put one in。",
    "F": "正在旋转的小陀螺主体，常有运动模糊、倾斜姿态或旋转状态。",
    "Z": "背景、阴影、桌面、玩具纹理、非目标，或无法确认的裁剪。",
}


def load_json(path: Path):
    return json.load(open(path))


def image_url(img: Image.Image, max_side=900):
    img = img.convert("RGB")
    w, h = img.size
    scale = 1.0
    if max(w, h) > max_side:
        scale = max_side / max(w, h)
        img = img.resize((round(w * scale), round(h * scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=92)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def crop_pil(img: Image.Image, bbox, pad=0.12):
    x, y, w, h = [float(v) for v in bbox]
    p = pad * max(w, h)
    x0 = max(0, x - p)
    y0 = max(0, y - p)
    x1 = min(img.width, x + w + p)
    y1 = min(img.height, y + h + p)
    return img.crop((x0, y0, x1, y1))


def make_reference(max_side=1200):
    colors = {"A": (220, 30, 30), "B": (30, 130, 240), "C": (40, 170, 70),
              "D": (240, 150, 20), "E": (180, 60, 210), "F": (20, 180, 180)}
    samples = []
    for split, ann_path in [("train", TRAIN), ("valid", VALID)]:
        data = load_json(ann_path)
        image_by = {im["id"]: im for im in data["images"]}
        per = Counter()
        for ann in data["annotations"]:
            cid = int(ann["category_id"])
            code = ID_TO_CODE.get(cid)
            if not code or per[code] >= 2:
                continue
            im = image_by[ann["image_id"]]
            img = Image.open(DATA / split / im["file_name"]).convert("RGB")
            crop = crop_pil(img, ann["bbox"], pad=0.20)
            crop.thumbnail((260, 210), Image.Resampling.LANCZOS)
            draw = ImageDraw.Draw(crop)
            draw.rectangle([1, 1, crop.width - 2, crop.height - 2], outline=colors[code], width=4)
            draw.rectangle([0, 0, 80, 28], fill=(255, 255, 255))
            draw.text((7, 7), code, fill=colors[code])
            samples.append((code, crop.copy()))
            per[code] += 1
        if all(per[ID_TO_CODE[cid]] >= 2 for cid in ID_TO_CODE):
            break

    cols = []
    for code in ["A", "B", "C", "D", "E", "F"]:
        imgs = [img for c, img in samples if c == code][:2]
        if not imgs:
            continue
        w = max(img.width for img in imgs)
        h = sum(img.height for img in imgs) + 8 * (len(imgs) - 1)
        col = Image.new("RGB", (w, h), (245, 245, 245))
        y = 0
        for img in imgs:
            col.paste(img, (0, y))
            y += img.height + 8
        cols.append(col)
    w = sum(col.width for col in cols) + 12 * (len(cols) - 1)
    h = max(col.height for col in cols)
    out = Image.new("RGB", (w, h), (245, 245, 245))
    x = 0
    for col in cols:
        out.paste(col, (x, 0))
        x += col.width + 12
    if max(out.size) > max_side:
        out.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    return out


def prompt(knn_prior):
    prior_text = ", ".join(f"{ID_TO_CODE[cid]}={knn_prior.get(cid, 0):.2f}" for cid in range(1, 7))
    desc = "\n".join(f"{code}: {CODE_TEXT[code]}" for code in ["A", "B", "C", "D", "E", "F", "Z"])
    return f"""
第一张图是参考样例，样例只使用 A-F 代号。第二张图是一个候选裁剪，请只判断这个裁剪最像哪个代号。

代号说明：
{desc}

一个非训练的 few-shot 视觉检索器给出的参考倾向是：{prior_text}。它只是参考，若图像证据不一致，以第二张裁剪为准。

请注意：
- 不要输出英文类别名。
- 如果裁剪主要是背景、桌面、阴影、模糊纹理，输出 Z。
- 如果是陀螺主体，选 A 或 F；如果是陀螺面上的单个希伯来字母，选 B/C/D/E。
- 输出 JSON，格式为 {{"label":"A","confidence":0.90,"probs":{{"A":0.1,"B":0.1,"C":0.1,"D":0.1,"E":0.1,"F":0.1,"Z":0.4}}}}。
""".strip()


def call_api(ref_img, crop_img, knn_prior, api_key, model, timeout=120, retries=1):
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt(knn_prior)},
                {"type": "image_url", "image_url": {"url": image_url(ref_img, 1200)}},
                {"type": "image_url", "image_url": {"url": image_url(crop_img, 900)}},
            ],
        }],
        "temperature": 0,
        "response_format": {"type": "json_object"},
        "enable_thinking": False,
        "thinking": {"type": "disabled"},
    }
    data = json.dumps(payload).encode()
    api_key = (api_key or "").strip().replace("\ufeff", "")
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
                obj = json.loads(resp.read().decode())
            return obj["choices"][0]["message"]["content"]
        except urllib.error.HTTPError as exc:
            last = RuntimeError(f"HTTP {exc.code}: " + exc.read().decode(errors="replace"))
        except Exception as exc:
            last = exc
        time.sleep(2 + attempt * 2)
    raise RuntimeError(last)


def parse_response(text):
    text = text.strip().replace("```json", "").replace("```", "").strip()
    try:
        obj = json.loads(text)
    except Exception:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise
        obj = json.loads(match.group(0))
    label = str(obj.get("label", "")).strip().upper()
    probs = obj.get("probs") or {}
    out = {code: 0.0 for code in ["A", "B", "C", "D", "E", "F", "Z"]}
    for code in out:
        try:
            out[code] = float(probs.get(code, 0.0))
        except Exception:
            out[code] = 0.0
    if sum(out.values()) <= 0:
        conf = float(obj.get("confidence", 0.75) or 0.75)
        if label in out:
            out[label] = conf
            rest = (1.0 - conf) / 6.0
            for code in out:
                if code != label:
                    out[code] = rest
    total = sum(max(0.0, v) for v in out.values())
    if total > 0:
        out = {code: max(0.0, v) / total for code, v in out.items()}
    return label, out, obj


def parse_ids(text, image_by):
    if text == "all":
        return sorted(image_by)
    ids = []
    for part in text.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = map(int, part.split("-", 1))
            ids.extend(range(a, b + 1))
        else:
            ids.append(int(part))
    return [iid for iid in ids if iid in image_by]


def iou(a, b):
    ax, ay, aw, ah = map(float, a)
    bx, by, bw, bh = map(float, b)
    inter = max(0.0, min(ax + aw, bx + bw) - max(ax, bx)) * max(
        0.0, min(ay + ah, by + bh) - max(ay, by)
    )
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def nms(preds, thr, classwise=True):
    groups = defaultdict(list)
    for pred in preds:
        groups[pred["category_id"] if classwise else 0].append(pred)
    out = []
    for group in groups.values():
        keep = []
        for pred in sorted(group, key=lambda z: z["score"], reverse=True):
            if all(iou(pred["bbox"], old["bbox"]) < thr for old in keep):
                keep.append(pred)
        out.extend(keep)
    return out


def evaluate(preds, image_ids, out_dir):
    if not preds:
        return {key: 0.0 for key in METRIC_KEYS}
    subset = load_json(GT)
    keep = set(image_ids)
    subset["images"] = [im for im in subset["images"] if im["id"] in keep]
    subset["annotations"] = [ann for ann in subset["annotations"] if ann["image_id"] in keep]
    gt_path = out_dir / "subset_gt.json"
    json.dump(subset, open(gt_path, "w"))
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(gt_path))
        dt = coco.loadRes(preds)
        ev = COCOeval(coco, dt, "bbox")
        ev.params.imgIds = image_ids
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {key: float(value) for key, value in zip(METRIC_KEYS, ev.stats)}


def write_pkl(preds, image_by, out_path):
    by = defaultdict(list)
    for pred in preds:
        by[pred["image_id"]].append({
            "image_id": int(pred["image_id"]),
            "category_id": int(pred["category_id"]) - 1,
            "bbox": np.array(pred["bbox"], dtype=np.float32),
            "score": float(pred["score"]),
        })
    sub = [{"image_id": iid, "instances": by.get(iid, [])} for iid in sorted(image_by)]
    pickle.dump(sub, open(out_path, "wb"), protocol=4)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-preds", type=Path, default=ROOT / "dreidel_shape_dedup_0_9_best/predictions.json")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--image-ids", default="0-9")
    parser.add_argument("--model", default="qwen3.5-plus")
    parser.add_argument("--api-key", default=os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY"))
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--qwen-weight", type=float, default=0.65)
    parser.add_argument("--bg-thr", type=float, default=0.60)
    parser.add_argument("--nms-thr", type=float, default=0.50)
    parser.add_argument("--max-candidates", type=int, default=120)
    args = parser.parse_args()
    if not args.api_key:
        raise SystemExit("set DASHSCOPE_API_KEY/QWEN_API_KEY")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = args.out_dir / "raw"
    raw_dir.mkdir(exist_ok=True)
    gt = load_json(GT)
    image_by = {int(im["id"]): im for im in gt["images"]}
    image_ids = parse_ids(args.image_ids, image_by)
    candidates = [p for p in load_json(args.source_preds) if int(p["image_id"]) in set(image_ids)]
    candidates = sorted(candidates, key=lambda z: z["score"], reverse=True)[:args.max_candidates]

    Xn, y, mu, sig = sam_pipe.knn_model(pad=0.18)
    ref_img = make_reference()
    ref_img.save(args.out_dir / "reference_codes.jpg")
    cache_pil = {}
    cache_cv = {}
    preds = []
    raw_rows = []
    for idx, cand in enumerate(candidates):
        iid = int(cand["image_id"])
        im = image_by[iid]
        if iid not in cache_pil:
            cache_pil[iid] = Image.open(DATA / "test" / im["file_name"]).convert("RGB")
            cache_cv[iid] = cv2.imread(str(DATA / "test" / im["file_name"]), cv2.IMREAD_COLOR)
        feat = sam_pipe.feature(cache_cv[iid], cand["bbox"], (im["width"], im["height"]), pad=0.18)
        knn = sam_pipe.knn_probs((feat - mu) / sig, Xn, y, k=9, temp=3.0)
        raw_path = raw_dir / f"{idx:04d}_img{iid}.txt"
        if args.resume and raw_path.exists():
            text = raw_path.read_text()
        else:
            crop = crop_pil(cache_pil[iid], cand["bbox"], pad=0.16)
            text = call_api(ref_img, crop, knn, args.api_key, args.model, args.timeout, args.retries)
            raw_path.write_text(text)
        try:
            label, qprobs, obj = parse_response(text)
        except Exception as exc:
            print("parse_fail", idx, iid, exc, text[:160], flush=True)
            continue
        fused = {}
        for cid in range(1, 7):
            code = ID_TO_CODE[cid]
            fused[cid] = args.qwen_weight * qprobs.get(code, 0.0) + (1.0 - args.qwen_weight) * knn.get(cid, 0.0)
        bg = qprobs.get("Z", 0.0)
        cid = max(fused, key=fused.get)
        score = float(cand["score"]) * (0.35 + 0.65 * fused[cid])
        raw_rows.append({"index": idx, "image_id": iid, "old_category_id": cand["category_id"], "label": label, "qwen_probs": qprobs, "knn": knn, "fused": fused, "bg": bg, "raw": obj})
        if bg >= args.bg_thr and bg > max(fused.values()):
            continue
        preds.append({
            "image_id": iid,
            "category_id": int(cid),
            "bbox": [round(float(v), 2) for v in cand["bbox"]],
            "score": round(max(0.0001, min(0.9999, score)), 6),
        })
        print(idx, "img", iid, "old", cand["category_id"], "qwen", label, "new", cid, "bg", round(bg, 3), flush=True)

    preds = nms(preds, args.nms_thr, classwise=True)
    preds = [{
        "image_id": int(p["image_id"]),
        "category_id": int(p["category_id"]),
        "bbox": [round(float(v), 2) for v in p["bbox"]],
        "score": round(float(p["score"]), 6),
    } for p in preds]
    json.dump(preds, open(args.out_dir / "predictions.json", "w"), indent=2)
    json.dump(raw_rows, open(args.out_dir / "qwen_knn_raw.json", "w"), indent=2)
    metrics = evaluate(preds, image_ids, args.out_dir)
    metrics.update({
        "images": len(image_ids),
        "predictions": len(preds),
        "image_ids": image_ids,
        "source": str(args.source_preds),
        "qwen_weight": args.qwen_weight,
        "bg_thr": args.bg_thr,
        "counts": dict(Counter(p["category_id"] for p in preds)),
    })
    json.dump(metrics, open(args.out_dir / "eval.json", "w"), indent=2)
    write_pkl(preds, image_by, args.out_dir / f"{DS}.pkl")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
