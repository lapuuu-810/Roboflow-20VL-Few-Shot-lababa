#!/usr/bin/env python3
########################################################################################
# FINAL REPRODUCTION: actions-zzid2-zb1hq-fsod-amih
########################################################################################
# Final result directory:
#   /data/LPP/cvpr/few_shot/sam3-main/best_sam3/actions-zzid2-zb1hq-fsod-amih/qwen36_27b_full_gtprompt_2img_b2——final
#
# Final output files:
#   refined_coco_predictions.json
#   actions-zzid2-zb1hq-fsod-amih.pkl
#   full_metrics.json
#   raw_qwen_results.jsonl
#   refined_predictions.jsonl
#
# Final verified metrics:
#   mAP   = 0.15560081140053694
#   mAP50 = 0.3121049858911415
#   mAP75 = 0.13429024923894065
#
# Launch command:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/conda/envs/qwen/bin/python \
#     /data/LPP/cvpr/few_shot/sam3-main/best_sam3/actions-zzid2-zb1hq-fsod-amih/qwen_volleyball_pipeline.py \
#     --pred-dir /data/LPP/cvpr/few_shot/sam3-main/best_sam3/actions-zzid2-zb1hq-fsod-amih \
#     --image-dir /data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data/actions-zzid2-zb1hq-fsod-amih/test \
#     --ann-path /data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data/actions-zzid2-zb1hq-fsod-amih/test/_annotations.coco.json \
#     --model-path /data/LPP/cvpr/model/Qwen/Qwen3.6-27B \
#     --output-dir /data/LPP/cvpr/few_shot/sam3-main/best_sam3/actions-zzid2-zb1hq-fsod-amih/qwen36_27b_full_gtprompt_2img_b2——final \
#     --num-gpus 4 \
#     --batch-size 2 \
#     --use-annotated-view \
#     --use-candidate-montage
#
# Resume command if interrupted:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/conda/envs/qwen/bin/python \
#     /data/LPP/cvpr/few_shot/sam3-main/best_sam3/actions-zzid2-zb1hq-fsod-amih/qwen_volleyball_pipeline.py \
#     --model-path /data/LPP/cvpr/model/Qwen/Qwen3.6-27B \
#     --output-dir /data/LPP/cvpr/few_shot/sam3-main/best_sam3/actions-zzid2-zb1hq-fsod-amih/qwen36_27b_full_gtprompt_2img_b2——final \
#     --num-gpus 4 \
#     --batch-size 2 \
#     --resume
########################################################################################
"""Qwen-assisted volleyball action refinement for SAM3 proposals.

Pipeline:
1. Load SAM3 person / ball proposals.
2. Remove obvious off-court noise with geometry, score filters, and NMS.
3. Ask Qwen3.5-VL to reason over one image plus indexed candidate boxes.
4. Fuse Qwen output with deterministic constraints: at most one ball, on-court
   player cap, team side consistency, valid labels only.
5. Save refined JSONL, COCO detection JSON, submission-style PKL, and previews.

The script is designed for four-GPU batch inference. Each worker owns one GPU and
loads one model copy, which is faster and more predictable than a single
device_map="auto" process for this per-image VLM workload.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.multiprocessing as mp
from PIL import Image, ImageDraw


ROOT = Path("/data/LPP/cvpr/few_shot/sam3-main/best_sam3/actions-zzid2-zb1hq-fsod-amih")
DATA_ROOT = Path("/data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data")
DATASET = "actions-zzid2-zb1hq-fsod-amih"
IMAGE_DIR = DATA_ROOT / DATASET / "test"
ANN_PATH = IMAGE_DIR / "_annotations.coco.json"
MODEL_PATH = Path("/data/LPP/cvpr/model/Qwen/Qwen3.5-9B")

ACTION_TO_COCO_ID = {
    "Attack": 1,
    "Block": 2,
    "Defense": 3,
    "Serve": 4,
    "Set": 5,
    "ball": 6,
}
SUBMISSION_ID = {name: coco_id - 1 for name, coco_id in ACTION_TO_COCO_ID.items()}
VALID_ACTIONS = {"Attack", "Block", "Defense", "Serve", "Set"}


@dataclass
class Candidate:
    idx: int
    source_id: int
    kind: str
    box: list[float]
    score: float
    prompt: str
    rel_path: str
    image_path: str
    raw: dict[str, Any]

    @property
    def area(self) -> float:
        return max(0.0, self.box[2] - self.box[0]) * max(0.0, self.box[3] - self.box[1])

    @property
    def center(self) -> tuple[float, float]:
        return ((self.box[0] + self.box[2]) / 2.0, (self.box[1] + self.box[3]) / 2.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3 + Qwen volleyball action refinement")
    parser.add_argument("--pred-dir", type=Path, default=ROOT)
    parser.add_argument("--image-dir", type=Path, default=IMAGE_DIR)
    parser.add_argument("--ann-path", type=Path, default=ANN_PATH)
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "qwen_refined")
    parser.add_argument("--num-gpus", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size per GPU worker, or global batch in sharded mode")
    parser.add_argument("--sharded-model", action="store_true", help="Load one large model sharded across visible GPUs instead of one model copy per GPU")
    parser.add_argument("--limit-images", type=int, default=0, help="Debug limit; 0 means all")
    parser.add_argument("--resume", action="store_true", help="Skip images already in raw_qwen_results.jsonl")
    parser.add_argument("--dry-run", action="store_true", help="Only inspect inputs and write prefilter summary")
    parser.add_argument("--skip-viz", action="store_true")
    parser.add_argument("--use-annotated-view", action="store_true", default=True, help="Feed Qwen an image with candidate boxes and indices drawn on it")
    parser.add_argument("--use-candidate-montage", action="store_true", default=True, help="Feed Qwen a crop montage of indexed candidates for clearer target selection")
    parser.add_argument("--no-candidate-montage", dest="use_candidate_montage", action="store_false")
    parser.add_argument("--include-clean-original", action="store_true", default=False, help="Also send the clean original image; disabled by default for 27B stability")
    parser.add_argument("--fewshot-reference-image", type=Path, default=None, help="Optional train/valid GT protocol reference montage image")

    parser.add_argument("--person-category-ids", default="0")
    parser.add_argument("--ball-category-ids", default="1,6")
    parser.add_argument("--person-score", type=float, default=0.30)
    parser.add_argument("--ball-score", type=float, default=0.20)
    parser.add_argument("--person-nms", type=float, default=0.72)
    parser.add_argument("--ball-nms", type=float, default=0.55)
    parser.add_argument("--max-persons", type=int, default=18)
    parser.add_argument("--max-balls", type=int, default=5)
    parser.add_argument("--max-on-court", type=int, default=12)
    parser.add_argument("--target-only", action="store_true", default=True, help="Keep only ball plus one action category near the ball")
    parser.add_argument("--selection-mode", choices=["llm", "geometry", "hybrid"], default="llm", help="Target player selection strategy after Qwen parsing")
    parser.add_argument("--max-target-players", type=int, default=2, help="Max target players; Block may legitimately keep multiple same-side blockers")
    parser.add_argument("--llm-min-confidence", type=float, default=0.45)
    parser.add_argument("--llm-max-block-players", type=int, default=2)
    parser.add_argument("--llm-max-other-players", type=int, default=1)
    parser.add_argument("--near-ball-factor", type=float, default=1.55, help="Geometry fallback only: keep extra same-action players if similarly close")
    parser.add_argument("--net-x-ratio", type=float, default=0.50, help="Approximate vertical net split; players across this line are treated as different sides")
    parser.add_argument("--side-margin-ratio", type=float, default=0.035, help="Tolerance around the net line for near-net players")
    parser.add_argument("--geometry-reselect", action="store_true", default=True, help="After Qwen picks an action, reselect nearest player candidates on the ball side")
    parser.add_argument("--max-ball-distance-ratio", type=float, default=0.34, help="Reject non-serve target players too far from the selected ball")
    parser.add_argument("--action-fallback", choices=["none", "confusion", "all"], default="confusion", help="Add low-score action alternatives for selected players to improve mAP under action ambiguity")
    parser.add_argument("--fallback-score-factor", type=float, default=0.30)
    parser.add_argument("--rank-fallback-k", type=int, default=0, help="Add low-score candidates from Qwen rank list as recall fallback")
    parser.add_argument("--rank-fallback-factor", type=float, default=0.18)
    parser.add_argument("--ball-fallback-k", type=int, default=0, help="Add extra low-score ball candidates for small-ball recall")
    parser.add_argument("--ball-fallback-factor", type=float, default=0.18)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.05)
    return parser.parse_args()


def id_set(raw: str) -> set[int]:
    return {int(x) for x in raw.split(",") if x.strip()}


def load_annotations(path: Path) -> tuple[dict[str, int], dict[int, str], list[dict[str, Any]]]:
    with open(path, "r") as f:
        coco = json.load(f)
    name_to_id = {img["file_name"]: int(img["id"]) for img in coco["images"]}
    id_to_name = {int(img["id"]): img["file_name"] for img in coco["images"]}
    return name_to_id, id_to_name, coco["images"]


def box_iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-6)


def nms(cands: list[Candidate], threshold: float) -> list[Candidate]:
    kept: list[Candidate] = []
    for cand in sorted(cands, key=lambda c: c.score, reverse=True):
        if all(box_iou(cand.box, prev.box) < threshold for prev in kept):
            kept.append(cand)
    return kept


def infer_kind(pred: dict[str, Any], person_ids: set[int], ball_ids: set[int]) -> str | None:
    typ = str(pred.get("type") or "").lower()
    prompt = str(pred.get("prompt") or "").lower()
    cat = int(pred.get("category_id", -999))
    if typ == "ball" or "ball" in prompt or cat in ball_ids:
        return "ball"
    if typ == "person" or prompt in {"person", "player", "athlete"} or cat in person_ids:
        return "person"
    return None


def load_predictions(args: argparse.Namespace) -> dict[str, list[Candidate]]:
    person_ids = id_set(args.person_category_ids)
    ball_ids = id_set(args.ball_category_ids)
    by_image: dict[str, list[Candidate]] = defaultdict(list)
    source_id = 0
    for pred_file in sorted(args.pred_dir.glob("predictions_gpu*.jsonl")):
        with open(pred_file, "r") as f:
            for line in f:
                if not line.strip():
                    continue
                pred = json.loads(line)
                kind = infer_kind(pred, person_ids, ball_ids)
                if kind is None:
                    continue
                box = [float(v) for v in pred["box"]]
                cand = Candidate(
                    idx=-1,
                    source_id=source_id,
                    kind=kind,
                    box=box,
                    score=float(pred.get("score", 0.0)),
                    prompt=str(pred.get("prompt", "")),
                    rel_path=str(pred.get("rel_path", "")),
                    image_path=str(pred.get("image_path", "")),
                    raw=pred,
                )
                source_id += 1
                by_image[cand.rel_path].append(cand)
    return by_image


def court_likelihood(c: Candidate, width: int, height: int) -> float:
    x1, y1, x2, y2 = c.box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx, cy = c.center
    area_ratio = (w * h) / max(width * height, 1)
    border_penalty = 0.0
    if cx < width * 0.04 or cx > width * 0.96:
        border_penalty += 0.20
    if cy < height * 0.04 or cy > height * 0.98:
        border_penalty += 0.15
    lower_bonus = 0.18 if c.kind == "person" and height * 0.20 <= cy <= height * 0.92 else 0.0
    size_bonus = min(0.20, math.log1p(area_ratio * 200.0) / 20.0)
    return c.score + lower_bonus + size_bonus - border_penalty


def prefilter_image(cands: list[Candidate], image_path: Path, args: argparse.Namespace) -> list[Candidate]:
    with Image.open(image_path) as im:
        width, height = im.size

    persons: list[Candidate] = []
    balls: list[Candidate] = []
    for cand in cands:
        x1, y1, x2, y2 = cand.box
        w = x2 - x1
        h = y2 - y1
        if w <= 1 or h <= 1:
            continue
        if cand.kind == "person":
            if cand.score < args.person_score:
                continue
            if cand.area < max(900, width * height * 0.00035):
                continue
            if h < height * 0.045:
                continue
            if y2 < height * 0.16:
                continue
            persons.append(cand)
        elif cand.kind == "ball":
            if cand.score < args.ball_score:
                continue
            if cand.area < 12 or cand.area > width * height * 0.02:
                continue
            balls.append(cand)

    persons = nms(persons, args.person_nms)
    balls = nms(balls, args.ball_nms)
    persons = sorted(persons, key=lambda c: court_likelihood(c, width, height), reverse=True)[: args.max_persons]
    balls = sorted(balls, key=lambda c: c.score, reverse=True)[: args.max_balls]

    merged = persons + balls
    for i, cand in enumerate(merged):
        cand.idx = i
    return merged


def load_existing_results(path: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return done
    with open(path, "r") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line)
            done[item["rel_path"]] = item
    return done


def make_prompt(rel_path: str, cands: list[Candidate], width: int, height: int) -> str:
    rows = []
    for c in cands:
        x1, y1, x2, y2 = c.box
        cx, cy = c.center
        rows.append(
            f"[{c.idx}] {c.kind} score={c.score:.3f} "
            f"box=[{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}] "
            f"point=({cx:.0f},{cy:.0f}) norm_center=({cx/width:.3f},{cy/height:.3f}) "
            f"size=({(x2-x1)/width:.3f},{(y2-y1)/height:.3f}) prompt={c.prompt}"
        )
    det_text = "\n".join(rows)
    return f"""You are refining volleyball few-shot detection annotations from indexed SAM3 candidates.
Your job is to reproduce the dataset annotation protocol, not to detect every visible player.

Image id: {Path(rel_path).name}
Image size: {width}x{height}

Visual inputs:
- If a reference image is provided first, it shows train/valid GT examples of this dataset's annotation protocol. Learn what kind of athlete is annotated for each class.
- The full-court candidate image has boxes and indices. Use it for net side, global play, and ball trajectory.
- The indexed crop montage shows candidates clearly. Use it to inspect complete athletes, partial crops, balls, and distractors.
- The clean original image, when present, helps judge pose, hands, ball contact, and net relation.

Candidate boxes:
{det_text}

Dataset annotation protocol learned from GT statistics:
- Most images annotate exactly one action class and one target athlete. Do not annotate all players.
- Two target athletes are common mainly for Block: keep two same-side blockers only when both jump/raise hands as one blocking wall at the net.
- With a visible ball, the target athlete is usually below or near the ball, on the same play side of the net, and visually interacting with the ball trajectory. This is a visual relation, not simply nearest-center distance.
- Serve is special: the server can be farther below the ball, isolated behind the end line or in a toss/serve motion.
- Ball should be included when a candidate is a visible volleyball. Ignore logos, lights, heads, shoes, court marks, and duplicate balls.

Action definitions and confusions:
1. Attack: hitter/spiker, arm swing or approach/jump to hit the ball over the net. Not a blocker with vertical hands.
2. Block: one or two front-row players at the net with raised hands forming a block wall. Both targets must be active blockers on the same side.
3. Defense: receiver/digger/passer with low stance or platform arms, reacting to a lower incoming ball. Not all back-row players.
4. Serve: isolated server/toss/serving arm action, often farther from the net and ball.
5. Set: setter under/near the ball, hands above forehead preparing overhead set. Not a blocker at the net.

Decision checklist:
1. Select the best ball candidate if a real ball is visible.
2. Infer the active play side and action type from pose, hands, net relation, and ball trajectory.
3. Rank candidate PERSON boxes by how likely they are the GT target. Prefer complete athlete boxes over partial body crops.
4. Final targets must be from one action class. For non-Block actions return one best target. For Block return one or two active blockers.
5. If uncertain between actions for the same person, still choose the visually most likely action; fallback categories will be added later.

Return ONLY compact JSON:
{{"b": ball_index_or_null, "action": "Attack"|"Block"|"Defense"|"Serve"|"Set", "side": "left"|"right"|"unknown", "p": [[idx,"Attack"|"Block"|"Defense"|"Serve"|"Set",conf], ...], "rank": [[idx,conf], ...], "r": {{}}}}

Rules:
- "p" is the final GT-style target set.
- "rank" lists up to 5 plausible target PERSON indices in descending likelihood, including the final target(s).
- Keep at most 1 player for Attack/Defense/Serve/Set.
- Keep at most 2 players for Block only when both are true active blockers.
- Use candidate indices. Do not invent or adjust bounding boxes; leave "r" empty unless a numeric candidate key is absolutely necessary. JSON only, no markdown, no prose."""


def parse_json_object(text: str) -> dict[str, Any]:
    text = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.S)
    if fenced:
        text = fenced.group(1)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        text = text[start : end + 1]
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}


def clamp_box(box: Any, fallback: list[float], width: int, height: int) -> list[float]:
    if not isinstance(box, list) or len(box) != 4:
        return fallback
    try:
        x1, y1, x2, y2 = [float(v) for v in box]
    except (TypeError, ValueError):
        return fallback
    x1 = min(max(x1, 0.0), float(width))
    x2 = min(max(x2, 0.0), float(width))
    y1 = min(max(y1, 0.0), float(height))
    y2 = min(max(y2, 0.0), float(height))
    if x2 <= x1 or y2 <= y1:
        return fallback
    return [x1, y1, x2, y2]


def deterministic_fuse(parsed: dict[str, Any], cands: list[Candidate], width: int, height: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    by_idx = {c.idx: c for c in cands}
    outputs: list[dict[str, Any]] = []
    refinements = parse_refinements(parsed.get("r"))

    ball_obj = parsed.get("ball") if isinstance(parsed.get("ball"), dict) else {}
    ball_idx = parsed.get("b", ball_obj.get("index"))
    if isinstance(ball_idx, int) and ball_idx in by_idx and by_idx[ball_idx].kind == "ball":
        cand = by_idx[ball_idx]
        box = clamp_box(refinements.get(ball_idx, ball_obj.get("refined_box")), cand.box, width, height)
        outputs.append(make_output(cand, "ball", "ball", None, box, ball_obj.get("confidence", cand.score), "qwen_ball"))
    else:
        balls = [c for c in cands if c.kind == "ball"]
        if balls:
            cand = max(balls, key=lambda c: c.score)
            outputs.append(make_output(cand, "ball", "ball", None, cand.box, cand.score, "fallback_top_ball"))
    outputs.extend(ball_fallback_candidates(cands, outputs, args))

    player_items = parsed.get("p", parsed.get("players", []))
    if not isinstance(player_items, list):
        player_items = []
    seen = set()
    players: list[dict[str, Any]] = []
    for item in player_items:
        idx, team, label, conf, refined = normalize_player_item(item, refinements)
        if not isinstance(idx, int) or idx in seen or idx not in by_idx:
            continue
        seen.add(idx)
        cand = by_idx[idx]
        if cand.kind != "person":
            continue
        label = normalize_action_label(label)
        if label is None:
            continue
        team = normalize_team(team)
        conf = safe_float(conf, cand.score)
        # SAM3 candidate boxes are more stable than free-form LLM box edits.
        box = cand.box
        players.append(make_output(cand, label, "person", team, box, conf, "qwen_player"))

    if len(players) > args.max_on_court:
        players.sort(key=lambda x: (x["llm_confidence"], x["score"]), reverse=True)
        players = players[: args.max_on_court]

    selected_ball = None
    desired_action = normalize_action_label(parsed.get("action"))
    if getattr(args, "target_only", True):
        selected_ball = next((x for x in outputs if x["kind"] == "ball"), None)
        mode = getattr(args, "selection_mode", "llm")
        if mode == "geometry" or (mode == "hybrid" and not players):
            players = geometry_reselect_players(cands, players, selected_ball, desired_action, width, height, args)
        elif mode == "hybrid":
            players = llm_select_players(players, selected_ball, desired_action, args)
            if not players:
                players = geometry_reselect_players(cands, [], selected_ball, desired_action, width, height, args)
        else:
            players = llm_select_players(players, selected_ball, desired_action, args)
        players.extend(rank_fallback_players(parsed, by_idx, players, desired_action, args))

    enforce_team_consistency(players, width)
    outputs.extend(players)
    return outputs



def ball_fallback_candidates(cands: list[Candidate], selected_outputs: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    k = int(getattr(args, "ball_fallback_k", 0))
    if k <= 0:
        return []
    selected_boxes = [o["box"] for o in selected_outputs if o.get("kind") == "ball"]
    balls = sorted((c for c in cands if c.kind == "ball"), key=lambda c: c.score, reverse=True)
    out: list[dict[str, Any]] = []
    for cand in balls:
        if len(out) >= k:
            break
        if any(box_iou(cand.box, box) > 0.75 for box in selected_boxes):
            continue
        confidence = cand.score * float(getattr(args, "ball_fallback_factor", 0.18))
        out.append(make_output(cand, "ball", "ball", None, cand.box, confidence, "ball_rank_fallback"))
        selected_boxes.append(cand.box)
    return out

def parse_refinements(raw: Any) -> dict[int, list[float]]:
    refinements: dict[int, list[float]] = {}
    if not isinstance(raw, dict):
        return refinements
    for key, value in raw.items():
        try:
            idx = int(key)
        except (TypeError, ValueError):
            continue
        if isinstance(value, list) and len(value) == 4:
            refinements[idx] = value
    return refinements


def normalize_player_item(item: Any, refinements: dict[int, list[float]]) -> tuple[Any, Any, Any, Any, Any]:
    if isinstance(item, list):
        idx = item[0] if len(item) > 0 else None
        if len(item) >= 4:
            team = item[1]
            label = item[2]
            conf = item[3]
        else:
            team = None
            label = item[1] if len(item) > 1 else None
            conf = item[2] if len(item) > 2 else None
        refined = refinements.get(idx) if isinstance(idx, int) else None
        return idx, team, label, conf, refined
    if isinstance(item, dict):
        if item.get("keep") is False:
            return None, None, None, None, None
        idx = item.get("index")
        refined = refinements.get(idx) if isinstance(idx, int) else item.get("refined_box")
        return idx, item.get("team"), item.get("label"), item.get("confidence"), refined
    return None, None, None, None, None


def normalize_team(team: Any) -> str | None:
    if not isinstance(team, str):
        return None
    aliases = {
        "att": "attacking",
        "attack": "attacking",
        "attacking": "attacking",
        "def": "defending",
        "defense": "defending",
        "defending": "defending",
    }
    return aliases.get(team.strip().lower())


def candidate_to_player_output(cand: Candidate, label: str, confidence: float, source: str) -> dict[str, Any]:
    return make_output(cand, label, "person", None, cand.box, confidence, source)


def side_of_box(box: list[float], width: int, args: argparse.Namespace) -> int:
    cx = (box[0] + box[2]) / 2.0
    net_x = width * float(args.net_x_ratio)
    margin = width * float(args.side_margin_ratio)
    if cx < net_x - margin:
        return -1
    if cx > net_x + margin:
        return 1
    return 0


def side_distance_penalty(box: list[float], target_side: int, width: int, args: argparse.Namespace) -> float:
    if target_side == 0:
        return 0.0
    side = side_of_box(box, width, args)
    if side == 0 or side == target_side:
        return 0.0
    return width * 0.75


def rank_fallback_players(
    parsed: dict[str, Any],
    by_idx: dict[int, Candidate],
    selected: list[dict[str, Any]],
    desired_action: str | None,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    k = int(getattr(args, "rank_fallback_k", 0))
    if k <= 0 or desired_action not in VALID_ACTIONS:
        return []
    raw_rank = parsed.get("rank", [])
    if not isinstance(raw_rank, list):
        return []
    selected_idx = {p.get("source_index") for p in selected}
    out: list[dict[str, Any]] = []
    for item in raw_rank:
        if len(out) >= k:
            break
        if isinstance(item, list) and item:
            idx = item[0]
            conf = item[1] if len(item) > 1 else None
        elif isinstance(item, dict):
            idx = item.get("index")
            conf = item.get("confidence")
        else:
            continue
        if not isinstance(idx, int) or idx in selected_idx or idx not in by_idx:
            continue
        cand = by_idx[idx]
        if cand.kind != "person":
            continue
        confidence = safe_float(conf, cand.score) * float(args.rank_fallback_factor)
        out.append(make_output(cand, desired_action, "person", None, cand.box, confidence, "qwen_rank_fallback"))
        selected_idx.add(idx)
    return out


def llm_select_players(
    players: list[dict[str, Any]],
    ball: dict[str, Any] | None,
    desired_action: str | None,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if not players:
        return []
    if desired_action not in VALID_ACTIONS:
        counts = Counter(p["label"] for p in players)
        desired_action = max(
            counts,
            key=lambda k: (counts[k], max(p["llm_confidence"] for p in players if p["label"] == k)),
        )
    same = [p for p in players if p["label"] == desired_action and p["llm_confidence"] >= args.llm_min_confidence]
    if not same:
        same = [p for p in players if p["label"] == desired_action]
    if not same:
        return []

    same.sort(key=lambda p: (p["llm_confidence"], p["sam3_score"], p["score"]), reverse=True)
    limit = args.llm_max_block_players if desired_action == "Block" else args.llm_max_other_players
    limit = max(1, min(args.max_target_players, int(limit)))
    selected: list[dict[str, Any]] = []
    seen_boxes: list[list[float]] = []
    for p in same:
        if any(box_iou(p["box"], box) > 0.88 for box in seen_boxes):
            continue
        selected.append(p)
        seen_boxes.append(p["box"])
        if len(selected) >= limit:
            break
    return selected


def geometry_reselect_players(
    cands: list[Candidate],
    qwen_players: list[dict[str, Any]],
    ball: dict[str, Any] | None,
    desired_action: str | None,
    width: int,
    height: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    person_cands = [c for c in cands if c.kind == "person"]
    if not person_cands:
        return []
    if desired_action not in VALID_ACTIONS:
        if qwen_players:
            if ball is not None:
                ranked_qwen = sorted(qwen_players, key=lambda p: (distance_to_ball(p, ball), -p["llm_confidence"], -p["score"]))
                desired_action = ranked_qwen[0]["label"]
            else:
                desired_action = max(
                    Counter(p["label"] for p in qwen_players),
                    key=lambda k: max(p["llm_confidence"] for p in qwen_players if p["label"] == k),
                )
        else:
            desired_action = "Attack"

    qwen_by_idx = {p["source_index"]: p for p in qwen_players}
    if ball is None:
        if qwen_players:
            same = [p for p in qwen_players if p["label"] == desired_action]
            if same:
                same.sort(key=lambda p: (p["llm_confidence"], p["score"]), reverse=True)
                chosen_side = side_of_box(same[0]["box"], width, args)
                same_side = [p for p in same if side_of_box(p["box"], width, args) in {0, chosen_side}]
                return same_side[: args.max_target_players]
        person_cands.sort(key=lambda c: (c.score, -abs(c.center[0] - width * args.net_x_ratio)), reverse=True)
        chosen = person_cands[:1]
        return [candidate_to_player_output(c, desired_action, c.score, "geometry_no_ball") for c in chosen]

    ball_side = side_of_box(ball["box"], width, args)
    if ball_side == 0 and qwen_players:
        nearest_qwen = min(qwen_players, key=lambda p: distance_to_ball(p, ball))
        ball_side = side_of_box(nearest_qwen["box"], width, args)

    max_dist = math.hypot(width, height) * float(args.max_ball_distance_ratio)
    if desired_action == "Serve":
        max_dist = math.hypot(width, height) * 0.65

    ranked = []
    for cand in person_cands:
        base = candidate_to_player_output(cand, desired_action, qwen_by_idx.get(cand.idx, {}).get("llm_confidence", cand.score), "geometry_reselect")
        d = distance_to_ball(base, ball)
        penalty = side_distance_penalty(cand.box, ball_side, width, args)
        qwen_bonus = -width * 0.08 if cand.idx in qwen_by_idx else 0.0
        complete_bonus = min(width * 0.06, 0.10 * max(0.0, cand.box[3] - cand.box[1]))
        score_key = d + penalty + qwen_bonus - complete_bonus
        ranked.append((score_key, d, cand, base))
    ranked.sort(key=lambda x: (x[0], -x[2].score))

    selected: list[dict[str, Any]] = []
    first_dist = None
    for score_key, d, cand, base in ranked:
        cand_side = side_of_box(cand.box, width, args)
        if ball_side != 0 and cand_side not in {0, ball_side}:
            continue
        if d > max_dist and desired_action != "Serve":
            continue
        if not selected:
            selected.append(base)
            first_dist = max(d, 1.0)
            continue
        if len(selected) >= args.max_target_players:
            break
        if cand_side not in {0, side_of_box(selected[0]["box"], width, args)}:
            continue
        if first_dist is not None and d <= first_dist * args.near_ball_factor:
            selected.append(base)
    if not selected and ranked:
        selected.append(ranked[0][3])
    return selected


def select_target_players(
    players: list[dict[str, Any]],
    ball: dict[str, Any] | None,
    desired_action: str | None,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    if not players:
        return []

    label_counts = Counter(p["label"] for p in players)
    if desired_action not in VALID_ACTIONS:
        if ball is not None:
            # Choose the label of the closest high-confidence player to the ball.
            ranked = sorted(players, key=lambda p: (distance_to_ball(p, ball), -p["llm_confidence"], -p["score"]))
            desired_action = ranked[0]["label"]
        else:
            desired_action = max(label_counts, key=lambda k: (label_counts[k], max(p["llm_confidence"] for p in players if p["label"] == k)))

    same_label = [p for p in players if p["label"] == desired_action]
    if not same_label:
        return []

    if ball is None:
        same_label.sort(key=lambda p: (p["llm_confidence"], p["score"]), reverse=True)
        return same_label[: args.max_target_players]

    ranked = sorted(same_label, key=lambda p: (distance_to_ball(p, ball), -p["llm_confidence"], -p["score"]))
    selected = ranked[:1]
    if args.max_target_players > 1 and len(ranked) > 1:
        first_dist = max(distance_to_ball(ranked[0], ball), 1.0)
        for cand in ranked[1:]:
            if len(selected) >= args.max_target_players:
                break
            if distance_to_ball(cand, ball) <= first_dist * args.near_ball_factor:
                selected.append(cand)
    return selected


def distance_to_ball(player: dict[str, Any], ball: dict[str, Any]) -> float:
    box = player["box"]
    bx = (ball["box"][0] + ball["box"][2]) / 2.0
    by = (ball["box"][1] + ball["box"][3]) / 2.0
    nearest_x = min(max(bx, box[0]), box[2])
    nearest_y = min(max(by, box[1]), box[3])
    edge_dist = math.hypot(nearest_x - bx, nearest_y - by)
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    center_dist = math.hypot(cx - bx, cy - by)
    height = max(1.0, box[3] - box[1])
    width = max(1.0, box[2] - box[0])
    completeness_bonus = min(45.0, 0.12 * height + 0.03 * width)
    below_bonus = 0.90 if cy >= by else 1.0
    # GT boxes are complete athletes: the ball is often near an arm/head, not near bbox center.
    return max(0.0, edge_dist + 0.12 * center_dist - completeness_bonus) * below_bonus


def safe_float(value: Any, fallback: float) -> float:
    try:
        v = float(value)
        if math.isfinite(v):
            return min(max(v, 0.0), 1.0)
    except (TypeError, ValueError):
        pass
    return float(fallback)


def normalize_action_label(label: Any) -> str | None:
    if not isinstance(label, str):
        return None
    label = label.strip()
    aliases = {
        "attack": "Attack",
        "attacking": "Attack",
        "spike": "Attack",
        "spiking": "Attack",
        "block": "Block",
        "blocking": "Block",
        "defense": "Defense",
        "defending": "Defense",
        "dig": "Defense",
        "receive": "Defense",
        "serve": "Serve",
        "serving": "Serve",
        "set": "Set",
        "setting": "Set",
    }
    return aliases.get(label.lower(), label if label in VALID_ACTIONS else None)


def make_output(
    cand: Candidate,
    label: str,
    kind: str,
    team: str | None,
    box: list[float],
    confidence: Any,
    source: str,
) -> dict[str, Any]:
    score = float(cand.score) * (0.55 + 0.45 * safe_float(confidence, cand.score))
    return {
        "dataset": DATASET,
        "image_path": cand.image_path,
        "rel_path": cand.rel_path,
        "source_id": cand.source_id,
        "source_index": cand.idx,
        "source_prompt": cand.prompt,
        "kind": kind,
        "type": kind,
        "label": label,
        "team": team,
        "category_id": ACTION_TO_COCO_ID[label],
        "submission_category_id": SUBMISSION_ID[label],
        "box": box,
        "bbox": [box[0], box[1], max(0.0, box[2] - box[0]), max(0.0, box[3] - box[1])],
        "score": score,
        "sam3_score": float(cand.score),
        "llm_confidence": safe_float(confidence, cand.score),
        "refine_source": source,
    }


def enforce_team_consistency(players: list[dict[str, Any]], width: int) -> None:
    if not players:
        return
    known = [p for p in players if p.get("team") in {"attacking", "defending"}]
    if len({p.get("team") for p in known}) >= 2:
        return
    xs = [p["box"][0] + p["box"][2] for p in players]
    if not xs:
        return
    median = float(np.median(xs)) / 2.0
    for p in players:
        cx = (p["box"][0] + p["box"][2]) / 2.0
        if p.get("team") not in {"attacking", "defending"}:
            p["team"] = "attacking" if cx >= median else "defending"
    _ = width


def worker_main(gpu_id: int, jobs: list[dict[str, Any]], args_dict: dict[str, Any], result_queue: mp.Queue) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    args = argparse.Namespace(**args_dict)
    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    torch.cuda.set_device(0)
    processor = AutoProcessor.from_pretrained(str(args.model_path), trust_remote_code=True)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(args.model_path),
        dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    for start in range(0, len(jobs), args.batch_size):
        batch = jobs[start : start + args.batch_size]
        messages = []
        for job in batch:
            content = []
            if job.get("fewshot_reference_image"):
                content.append({"type": "image", "image": job["fewshot_reference_image"]})
            content.append({"type": "image", "image": job["image_path"]})
            if job.get("montage_image_path"):
                content.append({"type": "image", "image": job["montage_image_path"]})
            if job.get("include_clean_original") and job.get("orig_image_path") and job.get("orig_image_path") != job["image_path"]:
                content.append({"type": "image", "image": job["orig_image_path"]})
            content.append({"type": "text", "text": job["prompt"]})
            messages.append([
                {
                    "role": "user",
                    "content": content,
                }
            ])
        texts = [
            processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
            for m in messages
        ]
        batch_images = []
        for job in batch:
            imgs = []
            if job.get("fewshot_reference_image"):
                imgs.append(Image.open(job["fewshot_reference_image"]).convert("RGB"))
            imgs.append(Image.open(job["image_path"]).convert("RGB"))
            if job.get("montage_image_path"):
                imgs.append(Image.open(job["montage_image_path"]).convert("RGB"))
            if job.get("include_clean_original") and job.get("orig_image_path") and job.get("orig_image_path") != job["image_path"]:
                imgs.append(Image.open(job["orig_image_path"]).convert("RGB"))
            batch_images.append(imgs)
        inputs = processor(text=texts, images=batch_images, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=0.9,
            )
        input_len = inputs["input_ids"].shape[1]
        responses = processor.batch_decode(generated[:, input_len:], skip_special_tokens=True)
        for job, response in zip(batch, responses):
            result_queue.put({"rel_path": job["rel_path"], "response": response, "gpu": gpu_id})
        del inputs, generated
        torch.cuda.empty_cache()


def run_qwen_jobs_sharded(jobs: list[dict[str, Any]], args: argparse.Namespace, raw_path: Path) -> dict[str, dict[str, Any]]:
    existing = load_existing_results(raw_path) if args.resume else {}
    pending = [job for job in jobs if job["rel_path"] not in existing]
    print(f"Qwen sharded jobs: total={len(jobs)} pending={len(pending)} resume_hits={len(existing)}")
    if not pending:
        return existing

    from transformers import AutoProcessor, Qwen3_5ForConditionalGeneration

    processor = AutoProcessor.from_pretrained(str(args.model_path), trust_remote_code=True)
    if hasattr(processor, "tokenizer"):
        processor.tokenizer.padding_side = "left"
        if processor.tokenizer.pad_token is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
    max_memory = {i: "22GiB" for i in range(min(args.num_gpus, torch.cuda.device_count()))}
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        str(args.model_path),
        dtype=torch.bfloat16,
        device_map="auto",
        max_memory=max_memory,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    input_device = next(model.parameters()).device

    results = dict(existing)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    with open(raw_path, "a", buffering=1) as f:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            messages = []
            for job in batch:
                content = [{"type": "image", "image": job["image_path"]}]
                if job.get("orig_image_path") and job.get("orig_image_path") != job["image_path"]:
                    content.append({"type": "image", "image": job["orig_image_path"]})
                content.append({"type": "text", "text": job["prompt"]})
                messages.append([{"role": "user", "content": content}])
            texts = [
                processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True, enable_thinking=False)
                for m in messages
            ]
            batch_images = []
            for job in batch:
                imgs = [Image.open(job["image_path"]).convert("RGB")]
                if job.get("montage_image_path"):
                    imgs.append(Image.open(job["montage_image_path"]).convert("RGB"))
                if job.get("orig_image_path") and job.get("orig_image_path") != job["image_path"]:
                    imgs.append(Image.open(job["orig_image_path"]).convert("RGB"))
                batch_images.append(imgs)
            inputs = processor(text=texts, images=batch_images, return_tensors="pt", padding=True).to(input_device)
            with torch.inference_mode():
                generated = model.generate(
                    **inputs,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=args.temperature > 0,
                    temperature=args.temperature,
                    top_p=0.9,
                )
            input_len = inputs["input_ids"].shape[1]
            responses = processor.batch_decode(generated[:, input_len:], skip_special_tokens=True)
            for job, response in zip(batch, responses):
                item = {"rel_path": job["rel_path"], "response": response, "gpu": "sharded"}
                results[job["rel_path"]] = item
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
                completed += 1
            print(f"[qwen-sharded] {completed}/{len(pending)} done")
            del inputs, generated
            torch.cuda.empty_cache()
    return results


def run_qwen_jobs(jobs: list[dict[str, Any]], args: argparse.Namespace, raw_path: Path) -> dict[str, dict[str, Any]]:
    existing = load_existing_results(raw_path) if args.resume else {}
    pending = [job for job in jobs if job["rel_path"] not in existing]
    print(f"Qwen jobs: total={len(jobs)} pending={len(pending)} resume_hits={len(existing)}")
    if not pending:
        return existing

    num_gpus = min(args.num_gpus, torch.cuda.device_count())
    if num_gpus <= 0:
        raise RuntimeError("No CUDA GPU found for Qwen inference")

    chunks = [pending[i::num_gpus] for i in range(num_gpus)]
    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    args_dict = vars(args).copy()
    processes = []
    for gpu_id, chunk in enumerate(chunks):
        if not chunk:
            continue
        proc = ctx.Process(target=worker_main, args=(gpu_id, chunk, args_dict, queue))
        proc.start()
        processes.append(proc)

    results = dict(existing)
    completed = 0
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    with open(raw_path, "a", buffering=1) as f:
        while completed < len(pending):
            item = queue.get()
            rel_path = item["rel_path"]
            results[rel_path] = item
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            completed += 1
            if completed % 20 == 0 or completed == len(pending):
                print(f"[qwen] {completed}/{len(pending)} done")

    failed = []
    for proc in processes:
        proc.join()
        if proc.exitcode != 0:
            failed.append(proc.exitcode)
    if failed:
        raise RuntimeError(f"Qwen worker failure exitcodes={failed}")
    return results


def make_annotated_view(image_path: Path, cands: list[Candidate], out_path: Path) -> Path:
    image = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(image)
    for c in cands:
        color = (255, 230, 0) if c.kind == "ball" else (0, 210, 255)
        width = 5 if c.kind == "ball" else 3
        draw.rectangle(c.box, outline=color, width=width)
        label = f"{c.idx}:{'B' if c.kind == 'ball' else 'P'}"
        tx, ty = c.box[0], max(0, c.box[1] - 24)
        draw.rectangle([tx, ty, tx + 9 * len(label) + 8, ty + 22], fill=(0, 0, 0))
        draw.text((tx + 4, ty + 3), label, fill=color)
        cx, cy = c.center
        r = 4 if c.kind == "ball" else 3
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=color, width=2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, quality=94)
    return out_path


def make_candidate_montage(image_path: Path, cands: list[Candidate], out_path: Path) -> Path:
    image = Image.open(image_path).convert("RGB")
    tile_w, tile_h = 220, 260
    cols = 4
    rows = max(1, math.ceil(len(cands) / cols))
    montage = Image.new("RGB", (cols * tile_w, rows * tile_h), (24, 24, 24))
    draw = ImageDraw.Draw(montage)
    for pos, c in enumerate(cands):
        col = pos % cols
        row = pos // cols
        x0 = col * tile_w
        y0 = row * tile_h
        x1, y1, x2, y2 = c.box
        pad_x = max(8.0, (x2 - x1) * 0.18)
        pad_y = max(8.0, (y2 - y1) * 0.18)
        crop_box = [
            max(0, int(x1 - pad_x)),
            max(0, int(y1 - pad_y)),
            min(image.width, int(x2 + pad_x)),
            min(image.height, int(y2 + pad_y)),
        ]
        crop = image.crop(crop_box)
        crop.thumbnail((tile_w - 16, tile_h - 44), Image.Resampling.LANCZOS)
        px = x0 + (tile_w - crop.width) // 2
        py = y0 + 34 + (tile_h - 44 - crop.height) // 2
        montage.paste(crop, (px, py))
        color = (255, 230, 0) if c.kind == "ball" else (0, 210, 255)
        label = f"{c.idx}:{'BALL' if c.kind == 'ball' else 'PERSON'} s={c.score:.2f}"
        draw.rectangle([x0, y0, x0 + tile_w - 1, y0 + tile_h - 1], outline=color, width=2)
        draw.rectangle([x0, y0, x0 + tile_w - 1, y0 + 28], fill=(0, 0, 0))
        draw.text((x0 + 6, y0 + 7), label, fill=color)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    montage.save(out_path, quality=94)
    return out_path


def build_jobs(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, list[Candidate]], dict[str, tuple[int, int]]]:
    name_to_id, _id_to_name, images = load_annotations(args.ann_path)
    raw_by_image = load_predictions(args)
    jobs = []
    filtered_by_image: dict[str, list[Candidate]] = {}
    sizes: dict[str, tuple[int, int]] = {}
    for img in sorted(images, key=lambda x: int(x["id"])):
        rel_path = f"{DATASET}/test/{img['file_name']}"
        image_path = args.image_dir / img["file_name"]
        if not image_path.exists():
            continue
        cands = prefilter_image(raw_by_image.get(rel_path, []), image_path, args)
        if not cands:
            continue
        width, height = int(img["width"]), int(img["height"])
        filtered_by_image[rel_path] = cands
        sizes[rel_path] = (width, height)
        qwen_image_path = image_path
        if args.use_annotated_view:
            qwen_image_path = make_annotated_view(
                image_path,
                cands,
                args.output_dir / "qwen_inputs" / f"boxed_{img['file_name']}",
            )
        montage_image_path = None
        if args.use_candidate_montage:
            montage_image_path = make_candidate_montage(
                image_path,
                cands,
                args.output_dir / "qwen_inputs" / f"montage_{img['file_name']}",
            )
        jobs.append({
            "rel_path": rel_path,
            "image_id": name_to_id[img["file_name"]],
            "image_path": str(qwen_image_path),
            "montage_image_path": str(montage_image_path) if montage_image_path else None,
            "fewshot_reference_image": str(args.fewshot_reference_image) if args.fewshot_reference_image else None,
            "include_clean_original": bool(args.include_clean_original),
            "orig_image_path": str(image_path),
            "prompt": make_prompt(rel_path, cands, width, height),
        })
    if args.limit_images:
        jobs = jobs[: args.limit_images]
        keep = {job["rel_path"] for job in jobs}
        filtered_by_image = {k: v for k, v in filtered_by_image.items() if k in keep}
        sizes = {k: v for k, v in sizes.items() if k in keep}
    return jobs, filtered_by_image, sizes


COCO_ID_TO_ACTION = {v: k for k, v in ACTION_TO_COCO_ID.items()}
CONFUSION_FALLBACKS = {
    "Attack": ["Block", "Set", "Serve"],
    "Block": ["Attack", "Set"],
    "Defense": ["Attack", "Serve"],
    "Serve": ["Attack", "Defense"],
    "Set": ["Attack", "Block"],
}


def expand_action_fallbacks(items: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    if args.action_fallback == "none":
        return items
    expanded = list(items)
    for item in items:
        if item.get("kind") != "person" or item.get("label") not in VALID_ACTIONS:
            continue
        if args.action_fallback == "all":
            labels = [x for x in sorted(VALID_ACTIONS) if x != item["label"]]
        else:
            labels = CONFUSION_FALLBACKS.get(item["label"], [])
        for label in labels:
            alt = dict(item)
            alt["label"] = label
            alt["category_id"] = ACTION_TO_COCO_ID[label]
            alt["submission_category_id"] = SUBMISSION_ID[label]
            alt["score"] = float(item["score"]) * float(args.fallback_score_factor)
            alt["refine_source"] = "qwen_action_fallback"
            expanded.append(alt)
    return expanded


def write_outputs(
    args: argparse.Namespace,
    qwen_results: dict[str, dict[str, Any]],
    filtered_by_image: dict[str, list[Candidate]],
    sizes: dict[str, tuple[int, int]],
) -> list[dict[str, Any]]:
    primary: list[dict[str, Any]] = []
    for rel_path, cands in filtered_by_image.items():
        response = qwen_results.get(rel_path, {}).get("response", "")
        parsed = parse_json_object(response)
        width, height = sizes[rel_path]
        primary.extend(deterministic_fuse(parsed, cands, width, height, args))
    refined = expand_action_fallbacks(primary, args)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with open(args.output_dir / "primary_predictions.jsonl", "w") as f:
        for item in primary:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    with open(args.output_dir / "refined_predictions.jsonl", "w") as f:
        for item in refined:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    name_to_id, _id_to_name, images = load_annotations(args.ann_path)
    coco_preds = []
    by_image_id: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for item in refined:
        file_name = Path(item["rel_path"]).name
        image_id = name_to_id[file_name]
        coco_item = {
            "image_id": image_id,
            "category_id": int(item["category_id"]),
            "bbox": [round(float(v), 2) for v in item["bbox"]],
            "score": round(float(item["score"]), 6),
        }
        coco_preds.append(coco_item)
        by_image_id[image_id].append(item)
    with open(args.output_dir / "refined_coco_predictions.json", "w") as f:
        json.dump(coco_preds, f)

    submission = []
    for img in images:
        image_id = int(img["id"])
        instances = []
        for item in by_image_id.get(image_id, []):
            instances.append({
                "image_id": image_id,
                "category_id": int(item["submission_category_id"]),
                "bbox": np.array(item["bbox"], dtype=np.float64),
                "score": float(item["score"]),
            })
        submission.append({"image_id": image_id, "instances": instances})
    with open(args.output_dir / f"{DATASET}.pkl", "wb") as f:
        pickle.dump(submission, f, protocol=4)

    stats = {
        "images_with_candidates": len(filtered_by_image),
        "primary_predictions": len(primary),
        "refined_predictions": len(refined),
        "action_fallback": args.action_fallback,
        "fallback_score_factor": args.fallback_score_factor,
        "by_label": Counter(item["label"] for item in refined),
        "by_team": Counter(str(item.get("team")) for item in refined if item["kind"] == "person" and item.get("refine_source") != "qwen_action_fallback"),
        "model": str(args.model_path),
    }
    stats["by_label"] = dict(stats["by_label"])
    stats["by_team"] = dict(stats["by_team"])
    with open(args.output_dir / "pipeline_stats.json", "w") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    return refined


def visualize(args: argparse.Namespace, refined: list[dict[str, Any]], max_workers: int = 8) -> None:
    by_image: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in refined:
        by_image[item["rel_path"]].append(item)
    out_dir = args.output_dir / "previews"
    out_dir.mkdir(parents=True, exist_ok=True)
    colors = {
        "Attack": (255, 60, 60, 220),
        "Block": (255, 150, 30, 220),
        "Defense": (60, 130, 255, 220),
        "Serve": (170, 80, 255, 220),
        "Set": (40, 190, 120, 220),
        "ball": (255, 235, 40, 240),
    }

    def one(rel_path: str, items: list[dict[str, Any]]) -> None:
        img_name = Path(rel_path).name
        image = Image.open(args.image_dir / img_name).convert("RGB")
        draw = ImageDraw.Draw(image)
        for item in items:
            if item.get("refine_source") == "qwen_action_fallback":
                continue
            box = item["box"]
            label = item["label"]
            color = colors[label]
            width = 4 if label == "ball" else 2
            draw.rectangle(box, outline=color, width=width)
            text = label if label == "ball" else f"{label}/{item.get('team') or '?'}"
            draw.rectangle([box[0], max(0, box[1] - 18), box[0] + len(text) * 7 + 4, max(0, box[1] - 1)], fill=(0, 0, 0))
            draw.text((box[0] + 2, max(0, box[1] - 17)), text, fill=color)
        image.save(out_dir / f"qwen_{img_name}", quality=92)

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for rel_path, items in by_image.items():
            ex.submit(one, rel_path, items)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    jobs, filtered_by_image, sizes = build_jobs(args)
    raw_count = sum(1 for _ in args.pred_dir.glob("predictions_gpu*.jsonl"))
    summary = {
        "pred_dir": str(args.pred_dir),
        "pred_files": raw_count,
        "images_with_candidates": len(filtered_by_image),
        "jobs": len(jobs),
        "num_gpus": args.num_gpus,
        "batch_size_per_gpu": args.batch_size,
        "sharded_model": args.sharded_model,
        "use_annotated_view": args.use_annotated_view,
        "use_candidate_montage": args.use_candidate_montage,
        "include_clean_original": args.include_clean_original,
        "fewshot_reference_image": str(args.fewshot_reference_image) if args.fewshot_reference_image else None,
        "action_fallback": args.action_fallback,
        "fallback_score_factor": args.fallback_score_factor,
        "rank_fallback_k": args.rank_fallback_k,
        "rank_fallback_factor": args.rank_fallback_factor,
        "ball_fallback_k": args.ball_fallback_k,
        "ball_fallback_factor": args.ball_fallback_factor,
        "selection_mode": args.selection_mode,
        "llm_min_confidence": args.llm_min_confidence,
        "geometry_reselect": args.geometry_reselect,
        "net_x_ratio": args.net_x_ratio,
        "side_margin_ratio": args.side_margin_ratio,
        "max_ball_distance_ratio": args.max_ball_distance_ratio,
    }
    with open(args.output_dir / "prefilter_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))
    if args.dry_run:
        print(f"Dry run summary saved to {args.output_dir / 'prefilter_summary.json'}")
        return

    raw_qwen_path = args.output_dir / "raw_qwen_results.jsonl"
    started = time.time()
    if args.sharded_model:
        qwen_results = run_qwen_jobs_sharded(jobs, args, raw_qwen_path)
    else:
        qwen_results = run_qwen_jobs(jobs, args, raw_qwen_path)
    refined = write_outputs(args, qwen_results, filtered_by_image, sizes)
    if not args.skip_viz:
        visualize(args, refined)
    print(f"Done in {(time.time() - started) / 60:.1f} min")
    print(f"Outputs: {args.output_dir}")


if __name__ == "__main__":
    main()
