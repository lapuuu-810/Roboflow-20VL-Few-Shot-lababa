#!/usr/bin/env python3
from __future__ import annotations
import argparse, contextlib, io, json, os, pickle, tempfile, zipfile
from collections import defaultdict
from pathlib import Path
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
DS='all-elements-fsod-mebv'; ROOT=Path('/data/LPP/cvpr/few_shot')
GT=ROOT/'data/foundational_fsod-fsod_rf20vl/data'/DS/'test/_annotations.coco.json'; BEST='all_elements_doubao_qwen_fuse_best'
def load_json(p): return json.load(open(p))
def save_json(x,p): p.parent.mkdir(parents=True,exist_ok=True); json.dump(x,open(p,'w'),indent=2,ensure_ascii=False)
def load_preds(path,wh,cats,src):
 data=load_json(path); out=[]
 if isinstance(data,dict):
  for k in ['predictions','annotations','results','instances']:
   if isinstance(data.get(k),list): data=data[k]; break
 for r in data:
  iid=int(r['image_id']); cid=int(r['category_id']); x,y,w,h=[float(v) for v in r['bbox']]
  if iid not in wh: continue
  if cid not in cats and cid+1 in cats: cid+=1
  if cid not in cats or w<=0 or h<=0: continue
  W,H=wh[iid]; x=max(0,min(x,W-1)); y=max(0,min(y,H-1)); w=max(1e-3,min(w,W-x)); h=max(1e-3,min(h,H-y))
  out.append({'image_id':iid,'category_id':cid,'bbox':[x,y,w,h],'score':float(r.get('score',1.0)),'source':src})
 return out
def evalp(coco,preds,work):
 rows=[{k:v for k,v in p.items() if k in ('image_id','category_id','bbox','score')} for p in preds]
 with tempfile.NamedTemporaryFile('w',suffix='.json',dir=work,delete=False) as f: json.dump(rows,f); tmp=f.name
 try:
  with contextlib.redirect_stdout(io.StringIO()):
   dt=coco.loadRes(tmp); ev=COCOeval(coco,dt,'bbox'); ev.evaluate(); ev.accumulate(); ev.summarize()
  st=[float(x) for x in ev.stats]; return {'mAP':st[0],'mAP50':st[1],'mAP75':st[2],'stats':st,'count':len(rows)}
 finally: os.remove(tmp)
def trans(preds,mul=1.0): return [{**p,'score':round(max(1e-6,min(.999999,float(p['score'])*mul)),6)} for p in preds]
def iou(a,b):
 ax,ay,aw,ah=a['bbox']; bx,by,bw,bh=b['bbox']; inter=max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by)); den=aw*ah+bw*bh-inter; return inter/den if den>0 else 0
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
def write_pkl(rows,gt,p):
 by=defaultdict(list)
 for q in rows: by[q['image_id']].append({'image_id':int(q['image_id']),'category_id':int(q['category_id'])-1,'bbox':[float(v) for v in q['bbox']],'score':float(q['score'])})
 sub=[{'image_id':int(im['id']),'instances':by.get(int(im['id']),[])} for im in sorted(gt['images'],key=lambda x:int(x['id']))]
 p.parent.mkdir(parents=True,exist_ok=True); pickle.dump(sub,open(p,'wb'),protocol=4)
def rebuild(pkl_dir,zip_path):
 tmp=str(zip_path)+'.tmp'
 with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED) as z:
  for fn in sorted(os.listdir(pkl_dir)):
   if fn.endswith('.pkl'): z.write(pkl_dir/fn,arcname=fn)
 os.replace(tmp,zip_path)
def run(args):
 gt=load_json(args.gt); wh={int(im['id']):(float(im['width']),float(im['height'])) for im in gt['images']}; cats={int(c['id']) for c in gt['categories'] if int(c['id'])!=0}; args.work_dir.mkdir(parents=True,exist_ok=True)
 with contextlib.redirect_stdout(io.StringIO()): coco=COCO(str(args.gt))
 q=load_preds(args.qwen,wh,cats,'qwen_final'); d=load_preds(args.doubao,wh,cats,'doubao')
 preds=nms(trans(q,args.qwen_mul)+trans(d,args.doubao_mul),args.nms_thr); rows=[{k:v for k,v in p.items() if k in ('image_id','category_id','bbox','score')} for p in preds]
 summary={'name':f'fuse_qm{args.qwen_mul}_dm{args.doubao_mul}_nms{args.nms_thr}_agnFalse',**evalp(coco,preds,args.work_dir),'params':{'qmul':args.qwen_mul,'dmul':args.doubao_mul,'nms':args.nms_thr,'agnostic':False},'baselines':{'qwen_raw':evalp(coco,q,args.work_dir),'doubao_raw':evalp(coco,d,args.work_dir)},'sources':{'qwen_final':str(args.qwen),'doubao':str(args.doubao)}}
 for dd in [args.work_dir,args.final_result_dir]: save_json(rows,dd/(summary['name']+'.json')); save_json(summary,dd/(summary['name']+'_eval.json'))
 save_json(rows,args.final_result_dir/(BEST+'.json')); save_json(summary,args.final_result_dir/(BEST+'_eval.json'))
 write_pkl(rows,gt,args.final_result_dir/(DS+'.pkl')); write_pkl(rows,gt,args.submission_pkl_dir/(DS+'.pkl'))
 if args.rebuild_zip: rebuild(args.submission_pkl_dir,args.submission_zip)
 print(json.dumps(summary,indent=2,ensure_ascii=False))
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--qwen',type=Path,default=ROOT/'final'/DS/'result/qwen_api_direct_full_v1.json'); ap.add_argument('--doubao',type=Path,default=ROOT/'sam3-main/best_sam3'/DS/'doubao_all_elements_direct_0_9/predictions.json'); ap.add_argument('--gt',type=Path,default=GT); ap.add_argument('--work-dir',type=Path,default=ROOT/'sam3-main/best_sam3'/DS/'doubao_qwen_fuse_opt'); ap.add_argument('--final-result-dir',type=Path,default=ROOT/'final'/DS/'result'); ap.add_argument('--submission-pkl-dir',type=Path,default=ROOT/'final/submission_final_filled'); ap.add_argument('--submission-zip',type=Path,default=ROOT/'final/submission_final_filled_new.zip'); ap.add_argument('--qwen-mul',type=float,default=1.0); ap.add_argument('--doubao-mul',type=float,default=0.3); ap.add_argument('--nms-thr',type=float,default=0.65); ap.add_argument('--rebuild-zip',action='store_true'); run(ap.parse_args())
if __name__=='__main__': main()
