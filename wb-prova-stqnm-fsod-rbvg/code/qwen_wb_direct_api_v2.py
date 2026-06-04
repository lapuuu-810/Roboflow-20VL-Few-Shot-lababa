#!/usr/bin/env python3
from __future__ import annotations

import argparse, base64, contextlib, io, json, os, re, time, urllib.request, urllib.error
from collections import defaultdict
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT = Path(__file__).resolve().parent
DS = 'wb-prova-stqnm-fsod-rbvg'
DATA = Path('/data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data') / DS
GT = DATA / 'test' / '_annotations.coco.json'
TRAIN = DATA / 'train' / '_annotations.coco.json'
VALID = DATA / 'valid' / '_annotations.coco.json'
CLASS = {1: 'Adult', 2: 'Juvenile', 3: 'Piglet'}
NAME_TO_ID = {v.lower(): k for k, v in CLASS.items()}
METRIC_KEYS = ['mAP','mAP50','mAP75','mAP_small','mAP_medium','mAP_large','AR1','AR10','AR100','AR_small','AR_medium','AR_large']


def load_json(p): return json.load(open(p))

def image_url(img: Image.Image, max_side=1600):
    img = img.convert('RGB')
    w,h = img.size
    scale = 1.0
    if max(w,h) > max_side:
        scale = max_side / max(w,h)
        img = img.resize((round(w*scale), round(h*scale)), Image.Resampling.LANCZOS)
    buf = io.BytesIO(); img.save(buf, format='JPEG', quality=90)
    return 'data:image/jpeg;base64,' + base64.b64encode(buf.getvalue()).decode(), img.size, scale


def make_reference(max_side=1600):
    samples = []
    colors = {1:(230,30,30),2:(30,170,40),3:(30,80,240)}
    for split, ann_path in [('train', TRAIN), ('valid', VALID)]:
        d = load_json(ann_path)
        img_by = {im['id']: im for im in d['images']}
        per = {1:0, 2:0, 3:0}
        for ann in d['annotations']:
            cid = int(ann['category_id'])
            if cid not in CLASS or per[cid] >= 2:
                continue
            im = img_by[ann['image_id']]
            img = Image.open(DATA/split/im['file_name']).convert('RGB')
            x,y,w,h = [float(v) for v in ann['bbox']]
            pad = max(30, 0.15 * max(w,h))
            x0,y0=max(0,x-pad),max(0,y-pad)
            x1,y1=min(img.width,x+w+pad),min(img.height,y+h+pad)
            crop = img.crop((x0,y0,x1,y1))
            crop.thumbnail((360,260), Image.Resampling.LANCZOS)
            sx, sy = crop.width / max(1,(x1-x0)), crop.height / max(1,(y1-y0))
            bx=[(x-x0)*sx,(y-y0)*sy,(x+w-x0)*sx,(y+h-y0)*sy]
            dr=ImageDraw.Draw(crop); col=colors[cid]
            dr.rectangle(bx, outline=col, width=4)
            dr.rectangle([0,0,210,32], fill=(255,255,255))
            dr.text((6,7), f'{CLASS[cid]} reference', fill=col)
            samples.append((cid,crop.copy()))
            per[cid]+=1
        if all(v>=2 for v in per.values()):
            break
    # arrange 3 columns by class, up to 2 examples each
    cols=[]
    for cid in [1,2,3]:
        imgs=[im for c,im in samples if c==cid][:2]
        W=max(i.width for i in imgs); H=sum(i.height for i in imgs)+12*(len(imgs)-1)
        col=Image.new('RGB',(W,H),(245,245,245)); y=0
        for im in imgs:
            col.paste(im,(0,y)); y+=im.height+12
        cols.append(col)
    W=sum(c.width for c in cols)+18*(len(cols)-1); H=max(c.height for c in cols)
    out=Image.new('RGB',(W,H),(245,245,245)); x=0
    for c in cols:
        out.paste(c,(x,0)); x+=c.width+18
    if max(out.size)>max_side:
        out.thumbnail((max_side,max_side), Image.Resampling.LANCZOS)
    return out


def prompt(w,h):
    return f"""
You are detecting pigs in a farm image for COCO bbox evaluation.
Target categories, exact dataset labels:
1 Adult, 2 Juvenile, 3 Piglet.

The first image is a reference montage with colored boxes and label text. Use it to learn this dataset's visual labeling style. The second image is the target image; detect only the target image.
Important: classify by the dataset examples and the animal's apparent age/relative body size, not by how large the crop appears. A single large-looking close-up can still be Piglet if it matches piglet proportions.

Return ALL visible pig instances. Search the whole image, including background, corners, partially occluded animals, and small pigs. If multiple pigs are visible, output multiple boxes. Use tight boxes around the visible whole body. Do not box people, floor, rails, shadows, feeders, walls, or merged groups. Do not duplicate the same animal.

Coordinates may be absolute pixels for the target image size {w}x{h}, or 0-1000 normalized. Use [x1,y1,x2,y2].
Return ONLY JSON:
{{"detections":[{{"category_id":1,"category_name":"Adult","bbox":[x1,y1,x2,y2],"confidence":0.90}}]}}
""".strip()


def call_api(img, ref, api_key, model, timeout=120, retries=1, max_side=1600):
    ref_url,_,_ = image_url(ref, max_side=max_side)
    img_url,sent_size,scale = image_url(img, max_side=max_side)
    payload = {
        'model': model,
        'messages': [{'role':'user','content':[{'type':'text','text':prompt(*sent_size)}, {'type':'image_url','image_url':{'url':ref_url}}, {'type':'image_url','image_url':{'url':img_url}}]}],
        'temperature': 0,
        'response_format': {'type':'json_object'},
        'enable_thinking': False,
        'thinking': {'type':'disabled'},
    }
    data=json.dumps(payload).encode()
    last=None; api_key=(api_key or '').strip().replace('\ufeff','')
    for a in range(retries+1):
        req=urllib.request.Request('https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions', data=data, headers={'Authorization':f'Bearer {api_key}','Content-Type':'application/json'}, method='POST')
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                obj=json.loads(resp.read().decode())
            return obj['choices'][0]['message']['content'], sent_size, scale
        except urllib.error.HTTPError as e:
            last=RuntimeError(f'HTTP {e.code}: '+e.read().decode(errors='replace'))
        except Exception as e:
            last=e
        time.sleep(2+a*2)
    raise RuntimeError(last)


def parse_json(text):
    text=text.strip().replace('```json','').replace('```','').strip()
    try: return json.loads(text)
    except Exception:
        m=re.search(r'\{.*\}', text, flags=re.S)
        if not m: raise
        return json.loads(m.group(0))


def box_to_xywh(box, sent_size, scale, full_size):
    if not box or len(box)!=4: return None
    try: x1,y1,x2,y2=[float(v) for v in box]
    except Exception: return None
    sw,sh=sent_size; W,H=full_size
    if max(x1,y1,x2,y2) <= 1000 and (x2 <= 1000 and y2 <= 1000):
        x1=x1/1000*sw; x2=x2/1000*sw; y1=y1/1000*sh; y2=y2/1000*sh
    if scale > 0:
        x1/=scale; x2/=scale; y1/=scale; y2/=scale
    x1=max(0,min(W-1,x1)); y1=max(0,min(H-1,y1)); x2=max(0,min(W,x2)); y2=max(0,min(H,y2))
    if x2<=x1 or y2<=y1: return None
    return [round(x1,2), round(y1,2), round(x2-x1,2), round(y2-y1,2)]


def parse_ids(s, image_by):
    if s == 'all': return sorted(image_by)
    ids=[]
    for part in s.split(','):
        part=part.strip()
        if not part: continue
        if '-' in part:
            a,b=map(int,part.split('-',1)); ids.extend(range(a,b+1))
        else: ids.append(int(part))
    return [i for i in ids if i in image_by]


def evaluate(preds, image_ids, out_dir):
    if not preds: return {k:0.0 for k in METRIC_KEYS}
    subset = load_json(GT)
    subset['images']=[im for im in subset['images'] if im['id'] in set(image_ids)]
    subset['annotations']=[a for a in subset['annotations'] if a['image_id'] in set(image_ids)]
    sg=out_dir/'subset_gt.json'; json.dump(subset, open(sg,'w'))
    with contextlib.redirect_stdout(io.StringIO()):
        coco=COCO(str(sg)); dt=coco.loadRes(preds); ev=COCOeval(coco,dt,'bbox'); ev.params.imgIds=image_ids; ev.evaluate(); ev.accumulate(); ev.summarize()
    return {k:float(v) for k,v in zip(METRIC_KEYS, ev.stats)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--image-ids', default='0-7')
    ap.add_argument('--out-dir', type=Path, default=ROOT/'qwen_wb_direct_api_v1')
    ap.add_argument('--model', default='qwen3.5-plus')
    ap.add_argument('--api-key', default=os.environ.get('DASHSCOPE_API_KEY') or os.environ.get('QWEN_API_KEY'))
    ap.add_argument('--timeout', type=int, default=120)
    ap.add_argument('--retries', type=int, default=1)
    ap.add_argument('--resume', action='store_true')
    ap.add_argument('--max-side', type=int, default=1600)
    args=ap.parse_args()
    if not args.api_key: raise SystemExit('set DASHSCOPE_API_KEY/QWEN_API_KEY')
    args.out_dir.mkdir(parents=True, exist_ok=True); raw_dir=args.out_dir/'raw'; raw_dir.mkdir(exist_ok=True)
    gt=load_json(GT); image_by={int(im['id']):im for im in gt['images']}; ids=parse_ids(args.image_ids,image_by)
    ref=make_reference(args.max_side); ref.save(args.out_dir/'reference.jpg')
    preds=[]
    for iid in ids:
        im=image_by[iid]; img=Image.open(DATA/'test'/im['file_name']).convert('RGB')
        raw_path=raw_dir/f'{iid:04d}.txt'
        meta_path=raw_dir/f'{iid:04d}_meta.json'
        if args.resume and raw_path.exists() and meta_path.exists():
            raw=raw_path.read_text(); meta=json.load(open(meta_path)); sent_size=tuple(meta['sent_size']); scale=meta['scale']
        else:
            raw,sent_size,scale=call_api(img, ref, args.api_key, args.model, args.timeout, args.retries, args.max_side)
            raw_path.write_text(raw); json.dump({'sent_size':sent_size,'scale':scale,'file_name':im['file_name']}, open(meta_path,'w'))
        try: obj=parse_json(raw)
        except Exception as e:
            print('parse_fail',iid,e,raw[:200]); continue
        dets=obj.get('detections', obj if isinstance(obj,list) else [])
        for d in dets:
            cid=d.get('category_id')
            if cid is None:
                cid=NAME_TO_ID.get(str(d.get('category_name','')).strip().lower())
            try: cid=int(cid)
            except Exception: continue
            if cid not in CLASS: continue
            bbox=box_to_xywh(d.get('bbox'), sent_size, scale, (im['width'], im['height']))
            if not bbox: continue
            conf=float(d.get('confidence', d.get('score', 0.8)) or 0.8)
            preds.append({'image_id':iid,'category_id':cid,'bbox':bbox,'score':round(max(0.01,min(0.99,conf)),4)})
        print('image',iid,'raw_dets',len(dets),'kept_total',len(preds), flush=True)
    json.dump(preds, open(args.out_dir/'predictions.json','w'), indent=2)
    met=evaluate(preds, ids, args.out_dir); met.update({'images':len(ids),'predictions':len(preds),'image_ids':ids})
    json.dump(met, open(args.out_dir/'eval.json','w'), indent=2)
    print(json.dumps(met, indent=2))

if __name__=='__main__': main()
