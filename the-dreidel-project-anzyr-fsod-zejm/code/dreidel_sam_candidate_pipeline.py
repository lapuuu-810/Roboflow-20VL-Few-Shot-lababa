#!/usr/bin/env python3
from __future__ import annotations

import argparse
import contextlib
import io
import json
import math
import os
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DS = 'the-dreidel-project-anzyr-fsod-zejm'
ROOT = Path(__file__).resolve().parent
DATA = Path('/data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data') / DS
GT = DATA / 'test/_annotations.coco.json'
TRAIN = DATA / 'train/_annotations.coco.json'
VALID = DATA / 'valid/_annotations.coco.json'
KEYS = ['mAP','mAP50','mAP75','mAP_small','mAP_medium','mAP_large','AR1','AR10','AR100','AR_small','AR_medium','AR_large']
CLASSES = {1:'Dreidel',2:'Gimel',3:'Hay',4:'Nun',5:'Shin',6:'Spinning Dreidel'}


def load_json(p): return json.load(open(p))


def image_ids_arg(s, image_by):
    if not s or s == 'all': return sorted(image_by)
    out=[]
    for part in s.split(','):
        part=part.strip()
        if not part: continue
        if '-' in part:
            a,b=map(int,part.split('-',1)); out.extend(range(a,b+1))
        else: out.append(int(part))
    return [i for i in out if i in image_by]


def iou(a,b):
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    inter=max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by))
    return inter/(aw*ah+bw*bh-inter+1e-9)


def nms(ps, thr, class_agn=True):
    if thr < 0: return list(ps)
    groups=defaultdict(list)
    for p in ps: groups[0 if class_agn else p['category_id']].append(p)
    out=[]
    for gs in groups.values():
        keep=[]
        for p in sorted(gs,key=lambda z:z['score'],reverse=True):
            if all(iou(p['bbox'],q['bbox'])<thr for q in keep): keep.append(p)
        out.extend(keep)
    return out


def expand_box(box, scale, W, H):
    x,y,w,h=map(float,box); cx=x+w/2; cy=y+h/2; nw=w*scale; nh=h*scale
    x0=max(0,cx-nw/2); y0=max(0,cy-nh/2); x1=min(W,cx+nw/2); y1=min(H,cy+nh/2)
    if x1<=x0 or y1<=y0: return None
    return [x0,y0,x1-x0,y1-y0]


def mask_size(mask_path):
    try:
        with Image.open(mask_path) as im: return im.size
    except Exception:
        return None


def load_rows(candidate_dir: Path):
    gt=load_json(GT); file_to_img={im['file_name']:im for im in gt['images']}
    rows=[]; missing=Counter(); size_cache={}
    for fp in sorted(candidate_dir.glob('predictions_gpu*.jsonl')):
        for line in open(fp):
            if not line.strip(): continue
            r=json.loads(line)
            im=file_to_img.get(os.path.basename(r.get('image_path','')))
            if im is None:
                missing['image']+=1; continue
            mp=r.get('mask_path')
            if mp not in size_cache: size_cache[mp]=mask_size(mp) if mp else None
            src=size_cache[mp]
            if not src:
                src=(im['width'], im['height'])
            sw,sh=src; W,H=float(im['width']),float(im['height'])
            x1,y1,x2,y2=map(float,r['box'])
            # SAM boxes are in the mask/source coordinate frame. Resize independently to COCO image frame.
            x1*=W/max(1,sw); x2*=W/max(1,sw); y1*=H/max(1,sh); y2*=H/max(1,sh)
            x1=max(0,min(W-1,x1)); y1=max(0,min(H-1,y1)); x2=max(0,min(W,x2)); y2=max(0,min(H,y2))
            if x2<=x1 or y2<=y1:
                missing['degenerate']+=1; continue
            cid=int(r.get('category_id',1))
            if cid not in CLASSES: continue
            rows.append({'image_id':int(im['id']),'category_id':cid,'bbox':[x1,y1,x2-x1,y2-y1],'score':float(r.get('score',0.5)),'prompt':r.get('prompt',''),'src_size':src})
    return gt, rows, missing


def evaluate(preds, image_ids):
    if not preds: return {k:0.0 for k in KEYS}
    with contextlib.redirect_stdout(io.StringIO()):
        coco=COCO(str(GT)); dt=coco.loadRes(preds); ev=COCOeval(coco,dt,'bbox'); ev.params.imgIds=image_ids; ev.evaluate(); ev.accumulate(); ev.summarize()
    return {k:float(v) for k,v in zip(KEYS,ev.stats)}


def build(rows, image_by, image_ids, topk=20, score_thr=0.0, nms_thr=0.6, scale=1.0, class_agn=True):
    by=defaultdict(list)
    for r in rows:
        if r['image_id'] not in image_ids or r['score'] < score_thr: continue
        im=image_by[r['image_id']]
        box=expand_box(r['bbox'], scale, im['width'], im['height'])
        if not box: continue
        q={'image_id':r['image_id'],'category_id':int(r['category_id']),'bbox':[round(float(v),2) for v in box],'score':round(float(r['score']),6)}
        by[r['image_id']].append(q)
    out=[]
    for iid in image_ids:
        ps=nms(by.get(iid,[]), nms_thr, class_agn)
        out.extend(sorted(ps,key=lambda z:z['score'],reverse=True)[:topk])
    return out


def read_cv(path):
    im=cv2.imread(str(path), cv2.IMREAD_COLOR)
    if im is None: raise FileNotFoundError(path)
    return im


def crop_cv(img, bbox, pad=0.2):
    H,W=img.shape[:2]; x,y,w,h=map(float,bbox); p=pad*max(w,h)
    x0=max(0,int(round(x-p))); y0=max(0,int(round(y-p))); x1=min(W,int(round(x+w+p))); y1=min(H,int(round(y+h+p)))
    return img[y0:max(y0+1,y1), x0:max(x0+1,x1)]


def feature(img, bbox, image_size, pad=0.2):
    crop=crop_cv(img,bbox,pad)
    if crop.size==0: crop=np.zeros((32,32,3),np.uint8)
    crop=cv2.resize(crop,(64,64),interpolation=cv2.INTER_AREA)
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV); lab=cv2.cvtColor(crop,cv2.COLOR_BGR2LAB)
    feats=[]
    for ch,bins,ran in [(hsv[:,:,0],16,[0,180]),(hsv[:,:,1],10,[0,256]),(hsv[:,:,2],10,[0,256]),(lab[:,:,0],10,[0,256]),(lab[:,:,1],10,[0,256]),(lab[:,:,2],10,[0,256])]:
        hist=cv2.calcHist([ch],[0],None,[bins],ran).reshape(-1); hist=hist/(hist.sum()+1e-9); feats.extend(hist.tolist())
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    feats.extend([gray.mean()/255, gray.std()/255, cv2.Canny(gray,80,160).mean()/255])
    x,y,w,h=map(float,bbox); W,H=image_size
    feats.extend([math.log(max(w*h,1))/math.log(W*H+1), math.log(max(w/max(h,1),1e-3)), w/W, h/H, (x+w/2)/W, (y+h/2)/H])
    return np.array(feats,dtype=np.float32)


def knn_model(pad=0.2):
    X=[]; y=[]
    for split,ann_path in [('train',TRAIN),('valid',VALID)]:
        d=load_json(ann_path); im_by={im['id']:im for im in d['images']}; cache={}
        for a in d['annotations']:
            cid=int(a['category_id'])
            if cid not in CLASSES: continue
            im=im_by[a['image_id']]
            if a['image_id'] not in cache: cache[a['image_id']]=read_cv(DATA/split/im['file_name'])
            X.append(feature(cache[a['image_id']],a['bbox'],(im['width'],im['height']),pad)); y.append(cid)
    X=np.stack(X); y=np.array(y,dtype=np.int32); mu=X.mean(0); sig=X.std(0)+1e-6
    return (X-mu)/sig,y,mu,sig


def knn_probs(f,Xn,y,k=9,temp=3.0):
    d=np.sqrt(((Xn-f)**2).sum(1)); idx=np.argsort(d)[:k]; w=np.exp(-d[idx]/temp)
    pr={cid:1e-6 for cid in CLASSES}
    for ii,wt in zip(idx,w): pr[int(y[ii])]+=float(wt)
    s=sum(pr.values()); return {cid:pr[cid]/s for cid in CLASSES}


def apply_knn(preds, image_by, alpha=0.6, top2_scale=0.05, k=9, temp=3.0, pad=0.2):
    Xn,y,mu,sig=knn_model(pad); cache={}; out=[]
    for p in preds:
        im=image_by[p['image_id']]
        if p['image_id'] not in cache: cache[p['image_id']]=read_cv(DATA/'test'/im['file_name'])
        f=(feature(cache[p['image_id']],p['bbox'],(im['width'],im['height']),pad)-mu)/sig
        pr=knn_probs(f,Xn,y,k,temp)
        orig={cid:(1.0 if cid==int(p['category_id']) else 0.0) for cid in CLASSES}
        fused={cid:(1-alpha)*orig[cid]+alpha*pr[cid] for cid in CLASSES}
        ranked=sorted(CLASSES,key=lambda c:fused[c],reverse=True)
        q=dict(p); q['category_id']=ranked[0]; q['score']=round(float(p['score'])*(0.75+0.25*fused[ranked[0]]),6); out.append(q)
        r=dict(p); r['category_id']=ranked[1]; r['score']=round(float(p['score'])*top2_scale*fused[ranked[1]]/max(fused[ranked[0]],1e-9),6); out.append(r)
    return out


def write_pkl(preds, image_by, out_path):
    by=defaultdict(list)
    for p in preds:
        by[p['image_id']].append({'image_id':int(p['image_id']),'category_id':int(p['category_id'])-1,'bbox':np.array(p['bbox'],dtype=np.float32),'score':float(p['score'])})
    sub=[{'image_id':iid,'instances':by.get(iid,[])} for iid in sorted(image_by)]
    pickle.dump(sub,open(out_path,'wb'),protocol=4)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--candidate-dir',type=Path,default=ROOT)
    ap.add_argument('--out-dir',type=Path,required=True)
    ap.add_argument('--image-ids',default='0-9')
    ap.add_argument('--topk',type=int,default=20)
    ap.add_argument('--score-thr',type=float,default=0.0)
    ap.add_argument('--nms-thr',type=float,default=0.6)
    ap.add_argument('--scale',type=float,default=1.0)
    ap.add_argument('--class-nms',action='store_true')
    ap.add_argument('--knn',action='store_true')
    args=ap.parse_args()
    gt,rows,missing=load_rows(args.candidate_dir); image_by={int(im['id']):im for im in gt['images']}; ids=image_ids_arg(args.image_ids,image_by)
    preds=build(rows,image_by,ids,args.topk,args.score_thr,args.nms_thr,args.scale,not args.class_nms)
    if args.knn: preds=apply_knn(preds,image_by)
    met=evaluate(preds,ids); met.update({'images':len(ids),'predictions':len(preds),'image_ids':ids,'topk':args.topk,'score_thr':args.score_thr,'nms_thr':args.nms_thr,'scale':args.scale,'knn':args.knn,'missing':dict(missing),'counts':dict(Counter(p['category_id'] for p in preds))})
    args.out_dir.mkdir(parents=True,exist_ok=True)
    json.dump(preds,open(args.out_dir/'predictions.json','w'),indent=2)
    json.dump(met,open(args.out_dir/'eval.json','w'),indent=2)
    write_pkl(preds,image_by,args.out_dir/f'{DS}.pkl')
    print(json.dumps(met,indent=2))

if __name__=='__main__': main()

# Note: contour-symbol experiments are kept in dreidel_symbol_contour_0_9_best.
