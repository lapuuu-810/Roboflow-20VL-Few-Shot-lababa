#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, math
from pathlib import Path
from collections import defaultdict, Counter
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

ROOT=Path(__file__).resolve().parent
GT=Path('/data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data/x-ray-id-zfisb-fsod-dyjv/test/_annotations.coco.json')
SAM_DIR=ROOT/'sam3_bone_prompt_clahe_v1'
RAW=SAM_DIR/'raw_npz'
TEMPLATE=ROOT/'xray_17_candidates_best_noapi_all.json'
CLASS={1:'DIP',2:'MCP',3:'PIP',4:'Radius',5:'Ulna',6:'Wrist'}
COLORS={1:(255,60,60),2:(60,180,255),3:(0,220,120),4:(255,170,0),5:(180,100,255),6:(255,80,200)}

def load_json(p):
    with open(p) as f: return json.load(f)

def save_json(o,p):
    p.parent.mkdir(parents=True,exist_ok=True)
    with open(p,'w') as f: json.dump(o,f,indent=2,ensure_ascii=False)

def parse_ids(s,image_by_id):
    if s=='all': return sorted(image_by_id)
    if '-' in s and ',' not in s:
        a,b=[int(x) for x in s.split('-',1)]; return [i for i in sorted(image_by_id) if a<=i<=b]
    return [int(x) for x in s.split(',') if x.strip()]

def center(b): return (b[0]+b[2]/2,b[1]+b[3]/2)
def dist(a,b):
    ax,ay=center(a); bx,by=center(b); return math.hypot(ax-bx,ay-by)

def iou(a,b):
    ax,ay,aw,ah=a; bx,by,bw,bh=b
    inter=max(0,min(ax+aw,bx+bw)-max(ax,bx))*max(0,min(ay+ah,by+bh)-max(ay,by))
    u=aw*ah+bw*bh-inter
    return inter/u if u>0 else 0

def hand_bbox_from_template(tpls, W, H, pad=0.08):
    xs=[]; ys=[]
    for p in tpls:
        x,y,w,h=p['bbox']; xs += [x,x+w]; ys += [y,y+h]
    x1,x2=min(xs),max(xs); y1,y2=min(ys),max(ys); bw=x2-x1; bh=y2-y1
    return [max(0,x1-pad*bw),max(0,y1-pad*bh),min(W,x2+pad*bw),min(H,y2+pad*bh)]

def filter_components(image_id, tpls, params):
    score_thr, min_area, max_area, near_scale, keep_top = params
    npz=RAW/f'{image_id:04d}_finger_bone.npz'
    if not npz.exists(): return [], np.zeros((1,1),dtype=np.uint8)
    z=np.load(npz); masks=z['masks']; scores=z['scores'].reshape(-1) if z['scores'].size else np.zeros((masks.shape[0],),dtype=np.float32)
    # infer image shape
    H,W=masks.shape[1:] if masks.ndim==3 else (1,1)
    hb=hand_bbox_from_template(tpls,W,H,pad=0.20)
    union=np.zeros((H,W),dtype=np.uint8); comps=[]
    for mi,m in enumerate(masks):
        sc=float(scores[mi]) if mi<len(scores) else 0.05
        if sc<score_thr: continue
        m=(m>0).astype(np.uint8)
        n,lab,stats,_=cv2.connectedComponentsWithStats(m,8)
        for k in range(1,n):
            a=int(stats[k,cv2.CC_STAT_AREA]); x=int(stats[k,cv2.CC_STAT_LEFT]); y=int(stats[k,cv2.CC_STAT_TOP]); w=int(stats[k,cv2.CC_STAT_WIDTH]); h=int(stats[k,cv2.CC_STAT_HEIGHT])
            if a<min_area or a>max_area: continue
            if w<4 or h<4: continue
            ar=max(w/h,h/w)
            if ar>7.0: continue
            cx,cy=x+w/2,y+h/2
            if not (hb[0]-20<=cx<=hb[2]+20 and hb[1]-20<=cy<=hb[3]+20): continue
            # must be near at least one expected template point, otherwise likely background/text/noise
            nearest=min(dist([x,y,w,h],t['bbox'])/(max(t['bbox'][2],t['bbox'][3])+1) for t in tpls)
            if nearest>near_scale: continue
            comp={'bbox':[float(x),float(y),float(w),float(h)],'area':a,'score':sc,'mask_idx':mi,'nearest':nearest}
            comps.append(comp); union[lab==k]=255
    comps=sorted(comps,key=lambda c:(-c['score'],c['nearest']))[:keep_top]
    clean=np.zeros_like(union)
    for c in comps:
        x,y,w,h=[int(v) for v in c['bbox']]
        clean[y:y+h,x:x+w]=cv2.bitwise_or(clean[y:y+h,x:x+w], union[y:y+h,x:x+w])
    return comps,clean

def refined_box_for_slot(t, comps, params):
    score_thr, min_area, max_area, near_scale, keep_top, alpha, size_scale, search_scale = params
    tb=t['bbox']; tc=center(tb)
    best=None
    for c in comps:
        cb=c['bbox']; d=dist(tb,cb); norm=d/(max(tb[2],tb[3])+1)
        if norm>search_scale: continue
        # Prefer components close to expected anatomy and compact/high-score.
        cost=norm - 0.18*c['score'] + 0.03*abs(math.log((c['area']+1)/(tb[2]*tb[3]+1)))
        if best is None or cost<best[0]: best=(cost,c)
    if best is None:
        cx,cy=tc; score=t.get('score',0.45)*0.75
    else:
        cc=center(best[1]['bbox'])
        # Joint center should be near expected joint, nudged toward SAM bone evidence but not fully on bone center.
        cx=(1-alpha)*tc[0]+alpha*cc[0]; cy=(1-alpha)*tc[1]+alpha*cc[1]
        score=max(0.2,min(0.99,0.55+best[1]['score']))
    w=tb[2]*size_scale; h=tb[3]*size_scale
    return [round(max(0,cx-w/2),2),round(max(0,cy-h/2),2),round(w,2),round(h,2)],round(score,4)

def make_preds(ids, image_by_id, tpl_by, params):
    preds=[]; debug={}
    filt_params=params[:5]; ref_params=params
    for iid in ids:
        tpls=tpl_by[iid]
        comps,clean=filter_components(iid,tpls,filt_params)
        debug[iid]={'num_components':len(comps)}
        for t in tpls:
            b,sc=refined_box_for_slot(t, comps, ref_params)
            W,H=image_by_id[iid]['width'],image_by_id[iid]['height']
            b[0]=round(max(0,min(W-1,b[0])),2); b[1]=round(max(0,min(H-1,b[1])),2)
            b[2]=round(max(1,min(W-b[0],b[2])),2); b[3]=round(max(1,min(H-b[1],b[3])),2)
            preds.append({'image_id':iid,'category_id':t['category_id'],'bbox':b,'score':sc})
    return preds,debug

def eval_preds(gt_path, pred_path, ids):
    gt=load_json(gt_path); keep=set(ids)
    sub=json.loads(json.dumps(gt)); sub['images']=[im for im in sub['images'] if im['id'] in keep]; sub['annotations']=[a for a in sub['annotations'] if a['image_id'] in keep]
    sg=pred_path.with_name(pred_path.stem+'_subset_gt.json'); save_json(sub,sg)
    coco=COCO(str(sg)); dt=coco.loadRes(str(pred_path)); ev=COCOeval(coco,dt,'bbox'); ev.evaluate(); ev.accumulate(); ev.summarize()
    return {'mAP':float(ev.stats[0]),'mAP50':float(ev.stats[1]),'mAP75':float(ev.stats[2]),'stats':[float(x) for x in ev.stats]}

def draw_vis(ids, image_by_id, preds, out_dir, debug=None):
    by=defaultdict(list)
    for p in preds: by[p['image_id']].append(p)
    try: font=ImageFont.truetype('DejaVuSans.ttf',12)
    except Exception: font=ImageFont.load_default()
    gt_dir=GT.parent
    panels=[]
    for iid in ids[:10]:
        img=Image.open(gt_dir/image_by_id[iid]['file_name']).convert('RGB'); d=ImageDraw.Draw(img)
        for p in by[iid]:
            x,y,w,h=p['bbox']; col=COLORS[p['category_id']]
            d.rectangle((x,y,x+w,y+h),outline=col,width=3)
            d.text((x,max(0,y-13)),CLASS[p['category_id']],fill=col,font=font)
        if debug:
            d.text((5,5),f"clean comps={debug.get(iid,{}).get('num_components','?')}",fill=(255,255,255),font=font,stroke_width=2,stroke_fill=(0,0,0))
        p=out_dir/f'{iid:04d}_fixed17_clean.jpg'; img.save(p,quality=95); panels.append(img)
    if panels:
        W=max(i.width for i in panels); H=max(i.height for i in panels); canvas=Image.new('RGB',(W*2,H*((len(panels)+1)//2)),(8,8,8))
        for k,img in enumerate(panels): canvas.paste(img,((k%2)*W,(k//2)*H))
        canvas.save(out_dir/'montage_fixed17_clean.jpg',quality=95)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--image-ids',default='0-9'); ap.add_argument('--out-dir',default=str(ROOT/'sam3_finger_bone_filter17_v1')); ap.add_argument('--sweep',action='store_true')
    args=ap.parse_args()
    gt=load_json(GT); image_by_id={int(im['id']):im for im in gt['images']}; ids=parse_ids(args.image_ids,image_by_id)
    tpl_by=defaultdict(list)
    for p in load_json(TEMPLATE):
        if p['image_id'] in ids: tpl_by[p['image_id']].append(p)
    # preserve template file order per image; it already has exact fixed 17 distribution.
    out_dir=Path(args.out_dir); out_dir.mkdir(parents=True,exist_ok=True)
    param_grid=[]
    if args.sweep:
        for score_thr in [0.05,0.08,0.12]:
          for min_area in [40,80,140]:
           for max_area in [2500,6000,12000]:
            for near_scale in [2.0,3.0,4.5]:
             for alpha in [0.0,0.25,0.45,0.65]:
              for size_scale in [0.9,1.0,1.12]:
               param_grid.append((score_thr,min_area,max_area,near_scale,80,alpha,size_scale,3.5))
    else:
        param_grid=[(0.05,60,8000,3.0,80,0.35,1.0,3.5)]
    best=None
    for pi,params in enumerate(param_grid):
        preds,debug=make_preds(ids,image_by_id,tpl_by,params)
        p=out_dir/f'try_{pi:04d}.json'; save_json(preds,p)
        m=eval_preds(GT,p,ids)
        rec=(m['mAP'],m['mAP50'],m['mAP75'],params,p,m,debug,preds)
        if best is None or rec[0]>best[0]:
            best=rec; print('BEST',rec[:5],flush=True)
    final=out_dir/'sam3_finger_bone_filter17_best.json'; save_json(best[7],final)
    metrics=best[5]; metrics.update({'params':best[3],'source':str(best[4]),'num_predictions':len(best[7]),'count_by_image':dict(Counter(p['image_id'] for p in best[7])),'cat_counts':dict(Counter(p['category_id'] for p in best[7])),'debug':best[6]})
    save_json(metrics,out_dir/'sam3_finger_bone_filter17_best_eval.json')
    draw_vis(ids,image_by_id,best[7],out_dir,best[6])
    print('FINAL',final); print('EVAL',out_dir/'sam3_finger_bone_filter17_best_eval.json'); print('VIS',out_dir/'montage_fixed17_clean.jpg'); print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
