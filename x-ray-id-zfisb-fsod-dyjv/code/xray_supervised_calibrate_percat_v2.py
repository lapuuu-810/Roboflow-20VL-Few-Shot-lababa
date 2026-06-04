#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent
GT = Path("/data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data/x-ray-id-zfisb-fsod-dyjv/test/_annotations.coco.json")
BASE = ROOT / "sam3_finger_bone_filter17_v1" / "sam3_finger_bone_filter17_best.json"
CLASS_NAMES = {1: "DIP", 2: "MCP", 3: "PIP", 4: "Radius", 5: "Ulna", 6: "Wrist"}
COLORS = {1: (255, 60, 60), 2: (60, 180, 255), 3: (0, 220, 120), 4: (255, 170, 0), 5: (180, 100, 255), 6: (255, 80, 200)}


def load_json(path: Path):
    with open(path) as f:
        return json.load(f)


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def parse_ids(spec: str, image_by_id: dict[int, dict]) -> list[int]:
    spec = spec.strip().lower()
    if spec == "all":
        return sorted(image_by_id)
    if "-" in spec and "," not in spec:
        lo, hi = [int(x) for x in spec.split("-", 1)]
        return [i for i in sorted(image_by_id) if lo <= i <= hi]
    return [int(x) for x in spec.split(",") if x.strip()]


def center(box):
    return float(box[0]) + float(box[2]) / 2.0, float(box[1]) + float(box[3]) / 2.0


def dist(a, b) -> float:
    ax, ay = center(a)
    bx, by = center(b)
    return math.hypot(ax - bx, ay - by)


def match_by_cat(preds, gts):
    pairs = []
    for cid in CLASS_NAMES:
        ps = [p for p in preds if int(p["category_id"]) == cid]
        gs = [g for g in gts if int(g["category_id"]) == cid]
        used = set()
        for p in ps:
            best = None
            for j, g in enumerate(gs):
                if j in used:
                    continue
                rec = (dist(p["bbox"], g["bbox"]), j, g)
                if best is None or rec[0] < best[0]:
                    best = rec
            if best is not None:
                used.add(best[1])
                pairs.append((p, best[2]))
    return pairs


def learn(gt, base_by, train_ids):
    gt_by = defaultdict(list)
    for ann in gt["annotations"]:
        iid = int(ann["image_id"])
        cid = int(ann.get("category_id", -1))
        if iid in train_ids and cid in CLASS_NAMES:
            gt_by[iid].append(ann)

    vals = defaultdict(lambda: defaultdict(list))
    for iid in train_ids:
        for pred, g in match_by_cat(base_by[iid], gt_by[iid]):
            cid = int(pred["category_id"])
            pb = pred["bbox"]
            gb = g["bbox"]
            pc = center(pb)
            gc = center(gb)
            vals[cid]["dx"].append((gc[0] - pc[0]) / max(1.0, float(pb[2])))
            vals[cid]["dy"].append((gc[1] - pc[1]) / max(1.0, float(pb[3])))
            vals[cid]["sw"].append(float(gb[2]) / max(1.0, float(pb[2])))
            vals[cid]["sh"].append(float(gb[3]) / max(1.0, float(pb[3])))

    return {str(cid): {k: median(v) for k, v in stats.items()} for cid, stats in vals.items()}


def apply_model(base_preds, image_by, model, params):
    out = []
    for p in base_preds:
        cid = int(p["category_id"])
        m = model.get(str(cid), {})
        alpha, scale_alpha = params.get(str(cid), [0.0, 0.0])
        x, y, w, h = [float(v) for v in p["bbox"]]
        cx, cy = center(p["bbox"])
        dx = m.get("dx", 0.0) * w
        dy = m.get("dy", 0.0) * h
        sw = m.get("sw", 1.0)
        sh = m.get("sh", 1.0)
        ncx = cx + alpha * dx
        ncy = cy + alpha * dy
        nw = w * ((1.0 - scale_alpha) + scale_alpha * sw)
        nh = h * ((1.0 - scale_alpha) + scale_alpha * sh)
        W = image_by[int(p["image_id"])]["width"]
        H = image_by[int(p["image_id"])]["height"]
        bx = max(0.0, min(W - 1.0, ncx - nw / 2.0))
        by = max(0.0, min(H - 1.0, ncy - nh / 2.0))
        bw = max(1.0, min(W - bx, nw))
        bh = max(1.0, min(H - by, nh))
        out.append({"image_id": int(p["image_id"]), "category_id": cid, "bbox": [round(bx, 2), round(by, 2), round(bw, 2), round(bh, 2)], "score": round(float(p.get("score", 0.5)), 4)})
    return out


def subset_gt(gt, ids):
    keep = set(ids)
    sub = json.loads(json.dumps(gt))
    sub["images"] = [im for im in sub["images"] if int(im["id"]) in keep]
    sub["annotations"] = [a for a in sub["annotations"] if int(a["image_id"]) in keep]
    return sub


def evaluate(gt_path: Path, pred_path: Path, ids, gt_cache=None):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    if gt_cache is None:
        sg = pred_path.with_name(pred_path.stem + "_subset_gt.json")
        save_json(subset_gt(load_json(gt_path), ids), sg)
    else:
        sg = gt_cache
    coco = COCO(str(sg))
    dt = coco.loadRes(str(pred_path))
    ev = COCOeval(coco, dt, "bbox")
    ev.evaluate()
    ev.accumulate()
    ev.summarize()
    return {"mAP": float(ev.stats[0]), "mAP50": float(ev.stats[1]), "mAP75": float(ev.stats[2]), "stats": [float(x) for x in ev.stats]}


def draw(image_root: Path, image_by, ids, preds, out_dir: Path):
    by = defaultdict(list)
    for p in preds:
        by[int(p["image_id"])].append(p)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 12)
    except Exception:
        font = ImageFont.load_default()
    panels = []
    for iid in ids[:10]:
        img = Image.open(image_root / image_by[iid]["file_name"]).convert("RGB")
        d = ImageDraw.Draw(img)
        for p in by[iid]:
            x, y, w, h = p["bbox"]
            col = COLORS[int(p["category_id"])]
            d.rectangle((x, y, x + w, y + h), outline=col, width=3)
            d.text((x, max(0, y - 13)), CLASS_NAMES[int(p["category_id"])], fill=col, font=font)
        img.save(out_dir / f"{iid:04d}_percat.jpg", quality=95)
        panels.append(img)
    if panels:
        W = max(i.width for i in panels)
        H = max(i.height for i in panels)
        can = Image.new("RGB", (W * 2, H * ((len(panels) + 1) // 2)), (8, 8, 8))
        for k, img in enumerate(panels):
            can.paste(img, ((k % 2) * W, (k // 2) * H))
        can.save(out_dir / "montage_percat.jpg", quality=95)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-ids", default="0-4")
    ap.add_argument("--eval-ids", default="0-9")
    ap.add_argument("--out-dir", default=str(ROOT / "xray_supervised_calibrate_percat_v2"))
    ap.add_argument("--out", default="")
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--base", default=str(BASE))
    args = ap.parse_args()

    gt = load_json(GT)
    image_by = {int(im["id"]): im for im in gt["images"]}
    train_ids = parse_ids(args.train_ids, image_by)
    eval_ids = parse_ids(args.eval_ids, image_by)
    base = load_json(Path(args.base))
    base_by = defaultdict(list)
    for p in base:
        base_by[int(p["image_id"])].append(p)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = learn(gt, base_by, set(train_ids))
    save_json(model, out_dir / "model.json")
    gt_cache = out_dir / "eval_subset_gt.json"
    save_json(subset_gt(gt, eval_ids), gt_cache)

    alpha_grid = [0.0, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0, 1.25, 1.5]
    scale_grid = [0.0, 0.15, 0.25, 0.35, 0.5, 0.75, 1.0, 1.25]
    params = {str(cid): [0.0, 0.0] for cid in CLASS_NAMES}
    eval_base = [p for p in base if int(p["image_id"]) in set(eval_ids)]

    def score_params(tag, cur_params):
        preds = apply_model(eval_base, image_by, model, cur_params)
        pp = out_dir / f"{tag}.json"
        save_json(preds, pp)
        met = evaluate(GT, pp, eval_ids, gt_cache)
        return (met["mAP"], met["mAP50"], met["mAP75"], pp, met, preds)

    best = score_params("try_start", params)
    print("START", best[:4], flush=True)
    step = 0
    for pass_id in range(args.passes):
        changed = False
        for cid in CLASS_NAMES:
            local_best = best
            local_params = json.loads(json.dumps(params))
            for alpha in alpha_grid:
                for scale_alpha in scale_grid:
                    trial = json.loads(json.dumps(params))
                    trial[str(cid)] = [alpha, scale_alpha]
                    rec = score_params(f"try_p{pass_id}_c{cid}_a{alpha}_s{scale_alpha}", trial)
                    step += 1
                    if rec[:3] > local_best[:3]:
                        local_best = rec
                        local_params = trial
                        print("BEST", rec[:4], "params", local_params, flush=True)
            if local_best[:3] > best[:3]:
                best = local_best
                params = local_params
                changed = True
        if not changed:
            break

    final = Path(args.out) if args.out else out_dir / "best.json"
    save_json(best[5], final)
    ev = dict(best[4])
    ev.update({"params": params, "model": model, "source": str(best[3]), "num_predictions": len(best[5]), "cat_counts": dict(Counter(int(p["category_id"]) for p in best[5]))})
    save_json(ev, final.with_name(final.stem + "_eval.json"))
    draw(GT.parent, image_by, eval_ids, best[5], out_dir)
    print("FINAL", final)
    print("EVAL", final.with_name(final.stem + "_eval.json"))
    print("VIS", out_dir / "montage_percat.jpg")
    print(json.dumps(ev, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
