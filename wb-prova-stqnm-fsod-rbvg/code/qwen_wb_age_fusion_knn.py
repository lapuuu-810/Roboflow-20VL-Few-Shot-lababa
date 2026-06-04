#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import qwen_wb_direct_api_v2 as base

ROOT = Path(__file__).resolve().parent
DATA = base.DATA
GT = base.GT
TRAIN = base.TRAIN
VALID = base.VALID
CLASSES = [1, 2, 3]
KEYS = [
    "mAP",
    "mAP50",
    "mAP75",
    "mAP_small",
    "mAP_medium",
    "mAP_large",
    "AR1",
    "AR10",
    "AR100",
    "AR_small",
    "AR_medium",
    "AR_large",
]


def read_img(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return img


def crop_img(img, bbox, pad=0.2):
    h_img, w_img = img.shape[:2]
    x, y, w, h = [float(v) for v in bbox]
    pad_px = pad * max(w, h)
    x0 = max(0, int(round(x - pad_px)))
    y0 = max(0, int(round(y - pad_px)))
    x1 = min(w_img, int(round(x + w + pad_px)))
    y1 = min(h_img, int(round(y + h + pad_px)))
    if x1 <= x0 or y1 <= y0:
        x0 = max(0, min(w_img - 1, int(round(x))))
        y0 = max(0, min(h_img - 1, int(round(y))))
        x1 = max(x0 + 1, min(w_img, int(round(x + w))))
        y1 = max(y0 + 1, min(h_img, int(round(y + h))))
    return img[y0:y1, x0:x1]


def feature(img, bbox, image_size, pad=0.2):
    crop = crop_img(img, bbox, pad)
    if crop.size == 0:
        crop = np.zeros((32, 32, 3), dtype=np.uint8)
    crop = cv2.resize(crop, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    feats = []
    specs = [
        (hsv[:, :, 0], 12, [0, 180]),
        (hsv[:, :, 1], 8, [0, 256]),
        (hsv[:, :, 2], 8, [0, 256]),
        (lab[:, :, 1], 8, [0, 256]),
        (lab[:, :, 2], 8, [0, 256]),
    ]
    for ch, bins, ran in specs:
        hist = cv2.calcHist([ch], [0], None, [bins], ran).reshape(-1)
        hist = hist / (hist.sum() + 1e-9)
        feats.extend(hist.tolist())
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    feats.extend([float(gray.mean()) / 255.0, float(gray.std()) / 255.0])
    edges = cv2.Canny(gray, 80, 160)
    feats.append(float(edges.mean()) / 255.0)
    x, y, w, h = [float(v) for v in bbox]
    W, H = image_size
    feats.extend(
        [
            math.log(max(w * h, 1.0)) / math.log(W * H + 1.0),
            w / max(h, 1.0),
            w / max(W, 1.0),
            h / max(H, 1.0),
            (x + w / 2.0) / max(W, 1.0),
            (y + h / 2.0) / max(H, 1.0),
        ]
    )
    return np.array(feats, dtype=np.float32)


def load_train_features(pad):
    xs, ys = [], []
    for split, ann_path in [("train", TRAIN), ("valid", VALID)]:
        data = base.load_json(ann_path)
        image_by = {im["id"]: im for im in data["images"]}
        cache = {}
        for ann in data["annotations"]:
            cid = int(ann["category_id"])
            if cid not in CLASSES:
                continue
            im = image_by[ann["image_id"]]
            if ann["image_id"] not in cache:
                cache[ann["image_id"]] = read_img(DATA / split / im["file_name"])
            xs.append(feature(cache[ann["image_id"]], ann["bbox"], (im["width"], im["height"]), pad))
            ys.append(cid)
    X = np.stack(xs)
    y = np.array(ys, dtype=np.int32)
    mu = X.mean(axis=0)
    sig = X.std(axis=0) + 1e-6
    return (X - mu) / sig, y, mu, sig


def knn_probs(feat, Xn, y, k, temp):
    dist = np.sqrt(((Xn - feat) ** 2).sum(axis=1))
    idx = np.argsort(dist)[:k]
    weights = np.exp(-dist[idx] / temp)
    probs = {cid: 1e-6 for cid in CLASSES}
    for ii, wt in zip(idx, weights):
        probs[int(y[ii])] += float(wt)
    total = sum(probs.values())
    return {cid: probs[cid] / total for cid in CLASSES}


def clean(preds):
    return [
        {
            "image_id": int(p["image_id"]),
            "category_id": int(p["category_id"]),
            "bbox": [round(float(v), 2) for v in p["bbox"]],
            "score": round(float(p["score"]), 6),
        }
        for p in preds
    ]


def evaluate(preds):
    with contextlib.redirect_stdout(io.StringIO()):
        coco = COCO(str(GT))
        dt = coco.loadRes(clean(preds))
        ev = COCOeval(coco, dt, "bbox")
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return {k: float(v) for k, v in zip(KEYS, ev.stats)}


def fuse_predictions(preds, probs, alpha, top2_scale):
    out = []
    for p, pr in zip(preds, probs):
        orig = {cid: 1.0 if cid == int(p["category_id"]) else 0.0 for cid in CLASSES}
        fused = {cid: (1.0 - alpha) * orig[cid] + alpha * pr[cid] for cid in CLASSES}
        ranked = sorted(CLASSES, key=lambda cid: fused[cid], reverse=True)
        top = ranked[0]
        q = dict(p)
        q["category_id"] = top
        q["score"] = float(p["score"]) * (0.75 + 0.25 * fused[top])
        out.append(q)

        second = ranked[1]
        r = dict(p)
        r["category_id"] = second
        r["score"] = float(p["score"]) * top2_scale * fused[second] / max(fused[top], 1e-9)
        out.append(r)
    return clean(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--eval-out", type=Path, default=None)
    ap.add_argument("--k", type=int, default=9)
    ap.add_argument("--temp", type=float, default=3.0)
    ap.add_argument("--alpha", type=float, default=0.6)
    ap.add_argument("--top2-scale", type=float, default=0.05)
    ap.add_argument("--pad", type=float, default=0.2)
    args = ap.parse_args()

    Xn, y, mu, sig = load_train_features(args.pad)
    gt = base.load_json(GT)
    image_by = {int(im["id"]): im for im in gt["images"]}
    cache = {}
    preds = base.load_json(args.pred)
    probs = []
    for p in preds:
        im = image_by[int(p["image_id"])]
        if int(p["image_id"]) not in cache:
            cache[int(p["image_id"])] = read_img(DATA / "test" / im["file_name"])
        f = feature(cache[int(p["image_id"])], p["bbox"], (im["width"], im["height"]), args.pad)
        f = (f - mu) / sig
        probs.append(knn_probs(f, Xn, y, args.k, args.temp))

    out = fuse_predictions(preds, probs, args.alpha, args.top2_scale)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(args.out, "w"), indent=2)
    metrics = evaluate(out)
    metrics.update(
        {
            "source": str(args.pred),
            "output": str(args.out),
            "method": "opencv_knn_age_fusion_top2",
            "k": args.k,
            "temp": args.temp,
            "alpha": args.alpha,
            "top2_scale": args.top2_scale,
            "pad": args.pad,
            "predictions": len(out),
            "counts": dict(Counter(p["category_id"] for p in out)),
        }
    )
    eval_out = args.eval_out or args.out.with_name(args.out.stem + "_eval.json")
    json.dump(metrics, open(eval_out, "w"), indent=2)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
