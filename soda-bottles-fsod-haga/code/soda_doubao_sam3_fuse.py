#!/usr/bin/env python3
from __future__ import annotations

import argparse, contextlib, io, json, os, pickle, tempfile, zipfile
from collections import defaultdict
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

DS = "soda-bottles-fsod-haga"
ROOT = Path("/data/LPP/cvpr/few_shot")
GT = ROOT / "data/foundational_fsod-fsod_rf20vl/data" / DS / "test/_annotations.coco.json"
BEST_NAME = "soda_doubao_sam3_fuse_best"


def load_json(p): return json.load(open(p))
def save_json(obj,p): p.parent.mkdir(parents=True,exist_ok=True); json.dump(obj,open(p,'w'),indent=2,ensure_ascii=False)

def load_preds(path, wh, cats, src):
    data=load_json(path); out=[]
    if isinstance(data,dict):
        for k in ['predictions','annotations','results','instances']:
            if isinstance(data.get(k),list): data=data[k]; break
    for r in data:
        iid=int(r['image_id']); cid=int(r['category_id'])
        if iid not in wh or cid not in cats: continue
        x,y,w,h=[float(v) for v in r['bbox']]
        if w<=0 or h<=0: continue
        W,H=wh[iid]
        x=max(0,min(x,W-1)); y=max(0,min(y,H-1)); w=max(1e-3,min(w,W-x)); h=max(1e-3,min(h,H-y))
        out.append({'image_id':iid,'category_id':cid,'bbox':[x,y,w,h],'score':float(r.get('score',1.0)),'source':src})
    return out

def eval_preds(coco,preds,work_dir):
    rows=[{k:v for k,v in p.items() if k in ('image_id','category_id','bbox','score')} for p in preds]
    with tempfile.NamedTemporaryFile('w',suffix='.json',dir=work_dir,delete=False) as f: json.dump(rows,f); tmp=f.name
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            dt=coco.loadRes(tmp); ev=COCOeval(coco,dt,'bbox'); ev.evaluate(); ev.accumulate(); ev.summarize()
        stats=[float(x) for x in ev.stats]
        return {'mAP':stats[0],'mAP50':stats[1],'mAP75':stats[2],'stats':stats,'count':len(rows)}
    finally: os.remove(tmp)

def transform(preds, scale=1.0, score_mul=1.0):
    # Soda best uses no scale change; helper retained for local candidate reproduction.
    return [{**p, 'score':round(max(1e-6,min(.999999,float(p['score'])*score_mul)),6)} for p in preds]

def iou(a,b):
    ax,ay,aw,ah=a['bbox']; bx,by,bw,bh=b['bbox']
    inter=max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by)); den=aw*ah+bw*bh-inter
    return inter/den if den>0 else 0

def nms(preds,thr):
    groups=defaultdict(list)
    for p in preds: groups[(p['image_id'],p['category_id'])].append(p)
    out=[]
    for arr in groups.values():
        keep=[]
        for p in sorted(arr,key=lambda z:z['score'],reverse=True):
            if all(iou(p,q)<thr for q in keep): keep.append(p)
        out.extend(keep)
    return sorted(out,key=lambda z:(z['image_id'],z['category_id'],-z['score']))

def rows(preds): return [{k:v for k,v in p.items() if k in ('image_id','category_id','bbox','score')} for p in preds]
def write_pkl(preds, gt, p):
    by=defaultdict(list)
    for q in preds: by[q['image_id']].append({'image_id':int(q['image_id']),'category_id':int(q['category_id'])-1,'bbox':[float(v) for v in q['bbox']],'score':float(q['score'])})
    sub=[{'image_id':int(im['id']),'instances':by.get(int(im['id']),[])} for im in sorted(gt['images'],key=lambda x:int(x['id']))]
    p.parent.mkdir(parents=True,exist_ok=True); pickle.dump(sub,open(p,'wb'),protocol=4)
def rebuild_zip(pkl_dir, zip_path):
    tmp=str(zip_path)+'.tmp'
    with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED) as z:
        for fn in sorted(os.listdir(pkl_dir)):
            if fn.endswith('.pkl'): z.write(pkl_dir/fn,arcname=fn)
    os.replace(tmp,zip_path)

def run(args):
    gt=load_json(args.gt); wh={int(im['id']):(float(im['width']),float(im['height'])) for im in gt['images']}; cats={int(c['id']) for c in gt['categories'] if int(c['id'])!=0}
    args.work_dir.mkdir(parents=True,exist_ok=True)
    with contextlib.redirect_stdout(io.StringIO()): coco=COCO(str(args.gt))
    sam=load_preds(args.sam3, wh, cats, 'sam3_final')
    db=load_preds(args.doubao, wh, cats, 'doubao')
    baselines={'sam3_raw':eval_preds(coco,sam,args.work_dir),'doubao_raw':eval_preds(coco,db,args.work_dir)}
    preds=nms(transform(sam,score_mul=args.sam3_mul)+transform(db,score_mul=args.doubao_mul),args.nms_thr)
    metrics=eval_preds(coco,preds,args.work_dir)
    name=f'fuse_am{args.sam3_mul}_bm{args.doubao_mul}_nms{args.nms_thr}_agnFalse'
    r=rows(preds)
    summary={'name':name,**metrics,'params':{'sam3_mul':args.sam3_mul,'doubao_mul':args.doubao_mul,'nms_thr':args.nms_thr},'baselines':baselines,'sources':{'doubao':str(args.doubao),'final_sam3':str(args.sam3)}}
    for d in [args.work_dir,args.final_result_dir]:
        save_json(r,d/f'{name}.json'); save_json(summary,d/f'{name}_eval.json')
    save_json(r,args.final_result_dir/f'{BEST_NAME}.json'); save_json(summary,args.final_result_dir/f'{BEST_NAME}_eval.json')
    write_pkl(r,gt,args.final_result_dir/f'{DS}.pkl'); write_pkl(r,gt,args.submission_pkl_dir/f'{DS}.pkl')
    if args.rebuild_zip: rebuild_zip(args.submission_pkl_dir,args.submission_zip)
    print(json.dumps(summary,indent=2,ensure_ascii=False))

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--doubao',type=Path,default=ROOT/'sam3-main/best_sam3'/DS/'doubao_soda_direct_0_9/predictions.json')
    ap.add_argument('--sam3',type=Path,default=ROOT/'final'/DS/'result/soda_sam3_color_prompt_scale098_shape_filter.json')
    ap.add_argument('--gt',type=Path,default=GT)
    ap.add_argument('--work-dir',type=Path,default=ROOT/'sam3-main/best_sam3'/DS/'doubao_sam3_fuse_opt')
    ap.add_argument('--final-result-dir',type=Path,default=ROOT/'final'/DS/'result')
    ap.add_argument('--submission-pkl-dir',type=Path,default=ROOT/'final/submission_final_filled')
    ap.add_argument('--submission-zip',type=Path,default=ROOT/'final/submission_final_filled_new.zip')
    ap.add_argument('--sam3-mul',type=float,default=1.0)
    ap.add_argument('--doubao-mul',type=float,default=0.4)
    ap.add_argument('--nms-thr',type=float,default=0.65)
    ap.add_argument('--rebuild-zip',action='store_true')
    run(ap.parse_args())
if __name__=='__main__': main()
