#!/usr/bin/env python3
"""Local Qwen aquarium candidate verifier.

Purpose:
  - Filter water-surface splash / wave / foam / reflection false positives.
  - Fix coarse class confusions among fish, shark, stingray, penguin, puffin, jellyfish.
  - Keep jellyfish boxes focused on the visible bell/head. This script does not blindly
    shrink every jellyfish; it asks the VLM to verify whether the candidate is a target.

Small test example:
  CUDA_VISIBLE_DEVICES=0,1,2,3 /opt/conda/envs/qwen/bin/python \
    /data/LPP/cvpr/few_shot/sam3-main/best_sam3/aquarium-combined-fsod-gjvb/qwen_aquarium_local_filter.py \
    --image-ids 1,17,18,29,31 \
    --max-candidates-per-image 32 \
    --chunk-size 4 \
    --out /data/LPP/cvpr/few_shot/sam3-main/best_sam3/aquarium-combined-fsod-gjvb/qwen_local_filter_test.json \
    --evaluate \
    --save-debug /data/LPP/cvpr/few_shot/sam3-main/best_sam3/aquarium-combined-fsod-gjvb/qwen_local_filter_debug
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from transformers import AutoModelForImageTextToText, AutoProcessor

ROOT = Path('/data/LPP/cvpr/few_shot/sam3-main/best_sam3/aquarium-combined-fsod-gjvb')
ANN = Path('/data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data/aquarium-combined-fsod-gjvb/test/_annotations.coco.json')
BASE = ROOT / 'baseline_best.json'
MODEL = Path('/data/LPP/cvpr/model/Qwen/Qwen3.5-9B')
CLASS_NAMES = {1:'fish',2:'jellyfish',3:'penguin',4:'puffin',5:'shark',6:'starfish',7:'stingray'}
VALID = set(CLASS_NAMES)


def load_json(p):
    with open(p) as f: return json.load(f)

def save_json(x,p):
    with open(p,'w') as f: json.dump(x,f,indent=2,ensure_ascii=False)

def build_montage(img_path, preds, tile_w=300, tile_h=360, cols=2):
    img=Image.open(img_path).convert('RGB')
    W,H=img.size
    tiles=[]
    for i,p in enumerate(preds):
        x,y,w,h=p['bbox']
        pad=max(16,int(max(w,h)*0.35))
        x1=max(0,int(round(x-pad))); y1=max(0,int(round(y-pad)))
        x2=min(W,int(round(x+w+pad))); y2=min(H,int(round(y+h+pad)))
        ctx=img.crop((x1,y1,x2,y2))
        d=ImageDraw.Draw(ctx)
        rx1=int(round(x-x1)); ry1=int(round(y-y1)); rx2=int(round(x+w-x1)); ry2=int(round(y+h-y1))
        for t in range(3): d.rectangle((rx1-t,ry1-t,rx2+t,ry2+t),outline=(255,0,0))
        exact=img.crop((max(0,int(round(x))),max(0,int(round(y))),min(W,int(round(x+w))),min(H,int(round(y+h))))).convert('RGB')
        exact.thumbnail((tile_w-12,145), Image.Resampling.LANCZOS)
        ctx.thumbnail((tile_w-12,145), Image.Resampling.LANCZOS)
        tile=Image.new('RGB',(tile_w,tile_h),'white')
        td=ImageDraw.Draw(tile)
        td.rectangle((0,0,tile_w,31),fill=(245,245,245))
        td.text((6,8),f"#{i} {p['category_id']}:{CLASS_NAMES.get(p['category_id'],'?')} s={p['score']:.2f}",fill=(0,0,0))
        tile.paste(ctx,((tile_w-ctx.size[0])//2,38))
        td.text((6,190),'exact crop:',fill=(0,0,0))
        tile.paste(exact,((tile_w-exact.size[0])//2,212))
        tiles.append(tile)
    rows=(len(tiles)+cols-1)//cols
    out=Image.new('RGB',(cols*tile_w,rows*tile_h),'white')
    for i,t in enumerate(tiles): out.paste(t,((i%cols)*tile_w,(i//cols)*tile_h))
    return out

def prompt_for(preds):
    meta=[]
    for i,p in enumerate(preds):
        meta.append({'candidate_id':i,'orig_category_id':p['category_id'],'orig_category':CLASS_NAMES.get(p['category_id']),'score':round(p['score'],3),'bbox_xywh':[round(v,1) for v in p['bbox']]})
    return (
        'Do not think step by step. Output exactly one JSON object and nothing else.\n'
        'You are verifying aquarium object-detection candidates. Each tile has a red candidate box, context view, and exact crop.\n'
        'Official categories: 1 fish, 2 jellyfish, 3 penguin, 4 puffin, 5 shark, 6 starfish, 7 stingray.\n'
        'Important negative examples: water surface wave, splash, foam, bubbles, reflections, glare, rocks, plants, tank background, glass marks, labels, empty water are NOT targets.\n'
        'Jellyfish rule: the benchmark labels the visible bell/head region, not long trailing tentacles/tail. If the red box mainly contains tentacles or water texture without a bell/head, drop it. If it contains a clear jellyfish bell/head, keep as jellyfish.\n'
        'Class distinctions: shark is elongated with shark body/fins; stingray is flat diamond/wing-shaped ray; fish is ordinary fish body; penguin/puffin are birds, not water splashes.\n'
        'For each candidate return: candidate_id, keep, category_id, category_name, confidence, reason.\n'
        'Use keep=false with confidence >=0.75 for obvious splash/wave/foam/reflection/background false positives.\n'
        'Return schema: {"results":[{"candidate_id":0,"keep":true,"category_id":1,"category_name":"fish","confidence":0.90,"reason":"clear fish body"}]}\n'
        f'Candidates: {json.dumps(meta, ensure_ascii=False)}'
    )

def parse_json(text):
    text=text.strip().replace('```json','').replace('```','').strip()
    try: return json.loads(text)
    except Exception:
        m=re.search(r'\{.*\}',text,flags=re.S)
        if not m: raise
        return json.loads(m.group(0))

class VLM:
    def __init__(self, model_path, dtype='bfloat16', device_map='auto'):
        torch_dtype=getattr(torch,dtype) if dtype!='auto' else 'auto'
        print('Loading processor',model_path,flush=True)
        self.processor=AutoProcessor.from_pretrained(model_path,trust_remote_code=True)
        print('Loading model',model_path,flush=True)
        self.model=AutoModelForImageTextToText.from_pretrained(model_path,torch_dtype=torch_dtype,device_map=device_map,trust_remote_code=True,low_cpu_mem_usage=True)
        self.model.eval(); print('Model loaded.',flush=True)
    def generate(self,img,prompt,max_new_tokens=768):
        messages=[{'role':'user','content':[{'type':'image'},{'type':'text','text':prompt}]}]
        text=self.processor.apply_chat_template(messages,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs=self.processor(text=[text],images=[img],return_tensors='pt')
        first=next(self.model.parameters()).device
        inputs={k:(v.to(first) if hasattr(v,'to') else v) for k,v in inputs.items()}
        with torch.inference_mode():
            out=self.model.generate(**inputs,max_new_tokens=max_new_tokens,do_sample=False,temperature=None,top_p=None)
        gen=out[:,inputs['input_ids'].shape[1]:]
        return self.processor.batch_decode(gen,skip_special_tokens=True,clean_up_tokenization_spaces=False)[0].strip()

def apply_decisions(chunk, decisions, drop_thr=0.75, relabel_thr=0.80):
    by={int(d.get('candidate_id',-1)):d for d in decisions if isinstance(d,dict)}
    out=[]; logs=[]
    for i,p in enumerate(chunk):
        d=by.get(i); q=dict(p)
        if not d:
            out.append(q); continue
        conf=float(d.get('confidence') or 0)
        keep=bool(d.get('keep',True))
        new=int(d.get('category_id') or q['category_id']) if keep else q['category_id']
        if not keep and conf>=drop_thr:
            logs.append({'candidate_id':i,'action':'drop','old':q['category_id'],'confidence':conf,'reason':d.get('reason',''),'bbox':q['bbox']})
            continue
        if keep and new in VALID and new!=q['category_id'] and conf>=relabel_thr:
            old=q['category_id']; q['category_id']=new; q['score']=round(max(0.001,min(0.999,q['score']*(0.70+0.30*conf))),6)
            logs.append({'candidate_id':i,'action':'relabel','old':old,'new':new,'confidence':conf,'reason':d.get('reason',''),'bbox':q['bbox']})
        out.append(q)
    return out, logs

def evaluate(gt_path,pred_path):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt=COCO(str(gt_path)); dt=gt.loadRes(str(pred_path)); ev=COCOeval(gt,dt,'bbox')
    ev.evaluate(); ev.accumulate(); ev.summarize()
    return {'mAP':float(ev.stats[0]),'mAP50':float(ev.stats[1]),'mAP75':float(ev.stats[2]),'stats':[float(x) for x in ev.stats]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--pred',default=str(BASE)); ap.add_argument('--gt',default=str(ANN)); ap.add_argument('--model-path',default=str(MODEL))
    ap.add_argument('--image-ids',default='1,17,18,29,31'); ap.add_argument('--out',default=str(ROOT/'qwen_local_filter_test.json'))
    ap.add_argument('--max-candidates-per-image',type=int,default=32); ap.add_argument('--chunk-size',type=int,default=4)
    ap.add_argument('--save-debug',default=''); ap.add_argument('--evaluate',action='store_true'); ap.add_argument('--continue-on-error',action='store_true')
    args=ap.parse_args()
    gt=load_json(args.gt); img_by_id={im['id']:im for im in gt['images']}; image_dir=Path(args.gt).parent
    preds=load_json(args.pred); by=defaultdict(list)
    for p in preds: by[p['image_id']].append(p)
    ids=sorted(by) if args.image_ids.lower()=='all' else [int(x) for x in args.image_ids.split(',') if x.strip()]
    if args.save_debug: Path(args.save_debug).mkdir(parents=True,exist_ok=True)
    vlm=VLM(args.model_path)
    final=[]; all_logs=[]; processed=set()
    for n,iid in enumerate(ids,1):
        processed.add(iid); im=img_by_id[iid]; img_path=image_dir/im['file_name']
        arr=sorted(by[iid],key=lambda x:x['score'],reverse=True)
        group=arr[:args.max_candidates_per_image]; rest=arr[args.max_candidates_per_image:]
        changed=[]
        for s in range(0,len(group),args.chunk_size):
            chunk=group[s:s+args.chunk_size]; ci=s//args.chunk_size
            montage=build_montage(img_path,chunk); prompt=prompt_for(chunk)
            print(f'[{n}/{len(ids)}] image_id={iid} chunk={ci} candidates={len(chunk)}',flush=True)
            if args.save_debug:
                stem=Path(im['file_name']).stem
                montage.save(Path(args.save_debug)/f'{iid:04d}_{stem}_chunk{ci}.jpg')
                (Path(args.save_debug)/f'{iid:04d}_{stem}_chunk{ci}.prompt.txt').write_text(prompt)
            try:
                raw=vlm.generate(montage,prompt)
                if args.save_debug: (Path(args.save_debug)/f'{iid:04d}_chunk{ci}_raw.txt').write_text(raw)
                obj=parse_json(raw); decisions=obj.get('results',obj if isinstance(obj,list) else [])
                if args.save_debug: (Path(args.save_debug)/f'{iid:04d}_chunk{ci}_raw.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2))
                out,logs=apply_decisions(chunk,decisions)
            except Exception as e:
                print('[warn]',iid,ci,e,flush=True)
                if not args.continue_on_error: raise
                out,logs=chunk,[]
            for l in logs: l['image_id']=iid; l['chunk']=ci
            all_logs.extend(logs); changed.extend(out)
        final.extend(changed+rest)
    for iid,arr in by.items():
        if iid not in processed: final.extend(arr)
    save_json(final,args.out); save_json(all_logs,str(Path(args.out).with_suffix(''))+'_log.json')
    print('wrote',args.out,'preds',len(final),'changes',len(all_logs))
    if args.evaluate:
        res=evaluate(args.gt,args.out); save_json(res,str(Path(args.out).with_suffix(''))+'_eval.json'); print(json.dumps(res,indent=2))
if __name__=='__main__': main()
