#!/usr/bin/env python3
"""Conservative local-Qwen calibration for aquarium detections.

Small local test:
  CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/conda/envs/qwen/bin/python \
    /data/LPP/cvpr/few_shot/sam3-main/best_sam3/aquarium-combined-fsod-gjvb/qwen_aquarium_local_safe_filter.py \
    --image-ids 1,18,29,31,49 \
    --max-candidates-per-image 32 \
    --chunk-size 4 \
    --out /data/LPP/cvpr/few_shot/sam3-main/best_sam3/aquarium-combined-fsod-gjvb/qwen_local_safe_5img_v1.json \
    --evaluate \
    --continue-on-error \
    --save-debug /data/LPP/cvpr/few_shot/sam3-main/best_sam3/aquarium-combined-fsod-gjvb/qwen_local_safe_5img_v1_debug

This script deliberately avoids jellyfish box editing. Jellyfish head/bell re-boxing is
handled by qwen_aquarium_jellyfish_api_rebox.py.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import qwen_aquarium_local_filter as base


def prompt_for(preds):
    meta = []
    for i, p in enumerate(preds):
        meta.append({
            'candidate_id': i,
            'orig_category_id': p['category_id'],
            'orig_category': base.CLASS_NAMES.get(p['category_id']),
            'score': round(p['score'], 3),
            'bbox_xywh': [round(v, 1) for v in p['bbox']],
        })
    return (
        'Do not think step by step. Output exactly one JSON object and nothing else.\n'
        'You are conservatively verifying aquarium object-detection candidates. Each tile has a red candidate box, context view, and exact crop.\n'
        'Official categories: 1 fish, 2 jellyfish, 3 penguin, 4 puffin, 5 shark, 6 starfish, 7 stingray.\n'
        'Your task in this run is ONLY to fix obvious non-jellyfish errors. Do not edit category 2 jellyfish candidates.\n'
        'Delete a candidate only when the red box is clearly NOT an animal: water surface wave, splash, foam, bubbles, reflection, glare, tank background, rock/plant, glass mark, empty water, text/label.\n'
        'If an animal is visible, keep it. Prefer relabeling over deleting.\n'
        'Class distinctions: shark is elongated with shark body/fins; stingray is flat diamond/wing-shaped ray; fish is ordinary fish body; penguin/puffin are birds.\n'
        'Use keep=false only for obvious background false positives, with confidence >=0.98.\n'
        'Use class relabeling only for shark-to-stingray confusion when a predicted shark is very clearly a flat stingray, with confidence >=0.95.\n'
        'Return schema: {"results":[{"candidate_id":0,"keep":true,"category_id":1,"category_name":"fish","confidence":0.95,"reason":"clear fish body"}]}\n'
        f'Candidates: {json.dumps(meta, ensure_ascii=False)}'
    )


def safe_apply_decisions(chunk, decisions, drop_thr=0.98, relabel_thr=0.95):
    by = {int(d.get('candidate_id', -1)): d for d in decisions if isinstance(d, dict)}
    out, logs = [], []
    background_words = (
        'water', 'wave', 'splash', 'foam', 'bubble', 'reflection', 'glare',
        'background', 'rock', 'plant', 'glass', 'empty', 'label', 'surface'
    )
    allowed_relabels = {(5, 7)}
    for i, p in enumerate(chunk):
        d = by.get(i)
        q = dict(p)
        if not d:
            out.append(q)
            continue
        old = int(q['category_id'])
        if old == 2:
            out.append(q)
            continue
        conf = float(d.get('confidence') or 0)
        keep = bool(d.get('keep', True))
        reason = str(d.get('reason', '')).lower()
        new = int(d.get('category_id') or old) if keep else old
        if (not keep and conf >= drop_thr and any(w in reason for w in background_words)):
            logs.append({'candidate_id': i, 'action': 'drop', 'old': old, 'confidence': conf, 'reason': d.get('reason', ''), 'bbox': q['bbox']})
            continue
        if keep and (old, new) in allowed_relabels and conf >= relabel_thr:
            q['category_id'] = new
            q['score'] = round(max(0.001, min(0.999, q['score'] * (0.78 + 0.22 * conf))), 6)
            logs.append({'candidate_id': i, 'action': 'relabel', 'old': old, 'new': new, 'confidence': conf, 'reason': d.get('reason', ''), 'bbox': q['bbox']})
        out.append(q)
    return out, logs


base.prompt_for = prompt_for
base.apply_decisions = safe_apply_decisions


if __name__ == '__main__':
    base.main()
