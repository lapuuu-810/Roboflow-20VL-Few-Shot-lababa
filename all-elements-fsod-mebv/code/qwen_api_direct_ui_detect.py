#!/usr/bin/env python3
########################################################################################
# FINAL REPRODUCTION: all-elements-fsod-mebv
########################################################################################
# Final result file:
#   /data/LPP/cvpr/few_shot/sam3-main/best_sam3/all-elements-fsod-mebv/qwen_api_direct_full_v1——final.json
#
# Final raw/API directory:
#   /data/LPP/cvpr/few_shot/sam3-main/best_sam3/all-elements-fsod-mebv/qwen_api_direct_full_v1_raw
#
# Final output/eval files:
#   qwen_api_direct_full_v1.json
#   qwen_api_direct_full_v1_eval.json
#
# Final verified metrics:
#   mAP   = 0.3934398820601464
#   mAP50 = 0.6395450345982888
#   mAP75 = 0.3698538816389302
#
# Launch command:
#   DASHSCOPE_API_KEY='<your_api_key>' python \
#     /data/LPP/cvpr/few_shot/sam3-main/best_sam3/all-elements-fsod-mebv/qwen_api_direct_ui_detect.py \
#     --model qwen3.5-plus \
#     --image-ids all \
#     --out /data/LPP/cvpr/few_shot/sam3-main/best_sam3/all-elements-fsod-mebv/qwen_api_direct_full_v1.json \
#     --raw-dir /data/LPP/cvpr/few_shot/sam3-main/best_sam3/all-elements-fsod-mebv/qwen_api_direct_full_v1_raw \
#     --timeout 120 \
#     --retries 2 \
#     --coord-mode norm1000
#
# Evaluation-only command when running inside an environment without pycocotools:
#   python - <<'PY_EVAL'
#   import json
#   from pathlib import Path
#   from pycocotools.coco import COCO
#   from pycocotools.cocoeval import COCOeval
#   gt = '/data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data/all-elements-fsod-mebv/test/_annotations.coco.json'
#   pred = '/data/LPP/cvpr/few_shot/sam3-main/best_sam3/all-elements-fsod-mebv/qwen_api_direct_full_v1.json'
#   out = '/data/LPP/cvpr/few_shot/sam3-main/best_sam3/all-elements-fsod-mebv/qwen_api_direct_full_v1_eval.json'
#   coco_gt = COCO(gt)
#   coco_dt = coco_gt.loadRes(pred)
#   ev = COCOeval(coco_gt, coco_dt, 'bbox')
#   ev.evaluate(); ev.accumulate(); ev.summarize()
#   res = {'mAP': float(ev.stats[0]), 'mAP50': float(ev.stats[1]), 'mAP75': float(ev.stats[2]), 'stats': [float(x) for x in ev.stats]}
#   Path(out).write_text(json.dumps(res, indent=2))
#   PY_EVAL
########################################################################################
import argparse
import base64
import io
import json
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent
GT = Path('/data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data/all-elements-fsod-mebv/test/_annotations.coco.json')
BASELINE = ROOT / 'tmp_baseline.json'

CLASS_NAMES = {1:'Button',2:'Check box',3:'Checked Radio button',4:'Checked box',5:'Dropdown box',6:'Dropdown expand',7:'Icon',8:'Radio button',9:'Scroll bar',10:'Text box'}


def load_json(p):
    with open(p) as f: return json.load(f)

def save_json(x,p):
    with open(p,'w') as f: json.dump(x,f,indent=2,ensure_ascii=False)

def image_url(img):
    buf=io.BytesIO(); img.save(buf,format='JPEG',quality=92)
    return 'data:image/jpeg;base64,'+base64.b64encode(buf.getvalue()).decode('ascii')

def make_prompt(width,height):
    return f"""
Do not think step by step. Do not explain. Output exactly one valid JSON object and nothing else.
You are doing fine-grained UI element object detection on the full image.
Image size: width={width}, height={height}. Return absolute pixel coordinates.

Detect only these official categories:
1 Button: clickable command rectangle, action button such as submit/login/next/save/clear.
2 Check box: small empty square form-control checkbox, not a square icon tile.
3 Checked Radio button: small circular radio option with inner dot, filled center, selected mark.
4 Checked box: small square checkbox with check/tick/fill, not a decorative square icon.
5 Dropdown box: full select/combo field rectangle with selected text/value and usually a down arrow.
6 Dropdown expand: the expanded/open dropdown list or popup panel area. In this dataset it can be a tall rectangle covering the opened menu/list, not only the small arrow.
7 Icon: standalone UI symbol/glyph such as search/calendar/menu/user/star, including small chevron/down-arrow icons inside dropdown fields and square icon buttons. Mouse cursor or pointer arrow is NOT a target.
8 Radio button: small empty circular ring option, unselected radio control.
9 Scroll bar: scroll track/thumb on edge of page or container.
10 Text box: editable input/search field rectangle, not a command button.

Critical rules:
- Empty circular ring => category 8 Radio button.
- Circular control with center dot/fill/selected mark => category 3 Checked Radio button.
- Big square tile containing cursor/picture/symbol => category 7 Icon or ignore if it is a mouse cursor; never checkbox.
- Only small form-control squares in lists/forms are category 2/4.
- Full closed dropdown/select field is category 5. Opened dropdown list/popup/panel is category 6. Small chevron arrow inside a closed dropdown is category 7 Icon if it is individually visible.
- Detect all visible target instances. For dropdown UI, include the full dropdown box, the small arrow icon if visible, and the expanded dropdown panel if open. Avoid duplicate boxes for the same object.
- Use tight boxes around the visible control, not around nearby text labels.
- If uncertain between icon and form control, prefer the form control only when its checkbox/radio/dropdown function is visually clear.

Return schema:
{{"detections":[{{"category_id":8,"category_name":"Radio button","bbox":[x1,y1,x2,y2],"confidence":0.95,"reason":"empty circular ring"}}]}}
All bbox coordinates must be integers in [0,width] and [0,height].
""".strip()

def call_api(img,prompt,api_key,model,timeout=90,retries=1):
    payload={'model':model,'messages':[{'role':'user','content':[{'type':'text','text':prompt},{'type':'image_url','image_url':{'url':image_url(img)}}]}],'temperature':0,'response_format':{'type':'json_object'},'enable_thinking':False,'thinking':{'type':'disabled'}}
    data=json.dumps(payload).encode('utf-8'); last=None
    for i in range(retries):
        req=urllib.request.Request('https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions',data=data,headers={'Authorization':f'Bearer {api_key}','Content-Type':'application/json'},method='POST')
        try:
            with urllib.request.urlopen(req,timeout=timeout) as resp: obj=json.loads(resp.read().decode('utf-8'))
            return obj['choices'][0]['message']['content']
        except urllib.error.HTTPError as e:
            last=RuntimeError(f"HTTP {e.code}: {e.read().decode('utf-8',errors='replace')}")
        except Exception as e: last=e
        time.sleep(2+i*2)
    raise RuntimeError(last)

def parse_json(text):
    text=text.strip().replace('```json','').replace('```','').strip()
    try: return json.loads(text)
    except Exception:
        m=re.search(r'\{.*\}',text,flags=re.S)
        if not m: raise
        return json.loads(m.group(0))

def clamp_box(b,w,h,coord_mode='auto'):
    if len(b)!=4: return None
    x1,y1,x2,y2=[float(v) for v in b]
    if coord_mode == 'norm1000' or (coord_mode == 'auto' and max(x1,y1,x2,y2) <= 1000 and (w > 1000 or h < 900)):
        x1 = x1 / 1000.0 * w
        x2 = x2 / 1000.0 * w
        y1 = y1 / 1000.0 * h
        y2 = y2 / 1000.0 * h
    if x2<=x1 or y2<=y1:
        x2=x1+max(1,x2); y2=y1+max(1,y2)
    x1=max(0,min(w-1,x1)); y1=max(0,min(h-1,y1)); x2=max(x1+1,min(w,x2)); y2=max(y1+1,min(h,y2))
    return [round(x1,2),round(y1,2),round(x2-x1,2),round(y2-y1,2)]

def nms(preds,thr=0.45):
    def iou(a,b):
        ax,ay,aw,ah=a['bbox']; bx,by,bw,bh=b['bbox']; ax2=ax+aw; ay2=ay+ah; bx2=bx+bw; by2=by+bh
        ix=max(0,min(ax2,bx2)-max(ax,bx)); iy=max(0,min(ay2,by2)-max(ay,by)); inter=ix*iy; union=aw*ah+bw*bh-inter
        return inter/union if union>0 else 0
    out=[]
    for c in sorted(set(p['category_id'] for p in preds)):
        keep=[]
        for p in sorted([p for p in preds if p['category_id']==c],key=lambda x:x['score'],reverse=True):
            if all(iou(p,k)<thr for k in keep): keep.append(p)
        out.extend(keep)
    return out

def draw_vis(img_path,dets,out_path):
    img=Image.open(img_path).convert('RGB'); d=ImageDraw.Draw(img)
    colors={3:(255,0,255),8:(0,180,255),7:(0,220,0),2:(255,180,0),4:(255,80,0),5:(120,0,255),6:(0,0,255)}
    for p in dets:
        x,y,w,h=p['bbox']; c=p['category_id']; col=colors.get(c,(255,0,0))
        d.rectangle((x,y,x+w,y+h),outline=col,width=3); d.text((x,y),f"{c}:{CLASS_NAMES.get(c,'?')} {p['score']:.2f}",fill=col)
    img.save(out_path)

def evaluate(gt_path,pred_path):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    gt=COCO(str(gt_path)); dt=gt.loadRes(str(pred_path)); ev=COCOeval(gt,dt,'bbox'); ev.evaluate(); ev.accumulate(); ev.summarize()
    return {'mAP':float(ev.stats[0]),'mAP50':float(ev.stats[1]),'mAP75':float(ev.stats[2]),'stats':[float(x) for x in ev.stats]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--api-key',default=os.environ.get('DASHSCOPE_API_KEY') or os.environ.get('QWEN_API_KEY'))
    ap.add_argument('--model',default='qwen3.5-plus')
    ap.add_argument('--image-ids',default='20,62', help='comma-separated image ids, or all')
    ap.add_argument('--baseline',default=str(BASELINE)); ap.add_argument('--gt',default=str(GT))
    ap.add_argument('--out',default=str(ROOT/'qwen_api_direct_test.json'))
    ap.add_argument('--raw-dir',default=str(ROOT/'qwen_api_direct_raw'))
    ap.add_argument('--timeout',type=int,default=120); ap.add_argument('--retries',type=int,default=1); ap.add_argument('--score-scale',type=float,default=1.0); ap.add_argument('--coord-mode',default='auto',choices=['auto','pixel','norm1000'])
    args=ap.parse_args()
    if not args.api_key: raise SystemExit('set DASHSCOPE_API_KEY or --api-key')
    gt=load_json(args.gt); baseline=load_json(args.baseline); img_by_id={im['id']:im for im in gt['images']}; ids=sorted(img_by_id) if args.image_ids.strip().lower() == 'all' else [int(x) for x in args.image_ids.split(',') if x.strip()]
    raw_dir=Path(args.raw_dir); raw_dir.mkdir(parents=True,exist_ok=True); direct=[]
    for iid in ids:
        im=img_by_id[iid]; img_path=Path(args.gt).parent/im['file_name']; img=Image.open(img_path).convert('RGB'); prompt=make_prompt(*img.size)
        (raw_dir/f'{iid:04d}_prompt.txt').write_text(prompt)
        print(f'API direct detect image_id={iid} size={img.size}',flush=True)
        raw=call_api(img,prompt,args.api_key,args.model,args.timeout,args.retries); (raw_dir/f'{iid:04d}_raw.txt').write_text(raw)
        obj=parse_json(raw); (raw_dir/f'{iid:04d}_raw.json').write_text(json.dumps(obj,ensure_ascii=False,indent=2))
        dets=[]
        for dct in obj.get('detections',obj.get('results',[])):
            cid=int(dct.get('category_id',0) or 0)
            if cid not in CLASS_NAMES: continue
            bbox=clamp_box(dct.get('bbox',[]),img.size[0],img.size[1],args.coord_mode)
            if not bbox: continue
            score=float(dct.get('confidence',dct.get('score',0.8)) or 0.8)*args.score_scale
            dets.append({'image_id':iid,'category_id':cid,'bbox':bbox,'score':round(max(0.001,min(0.999,score)),6)})
        dets=nms(dets,0.45); direct.extend(dets); draw_vis(img_path,dets,raw_dir/f'{iid:04d}_direct_vis.jpg'); print(f'  detections={len(dets)}',flush=True)
    merged=[p for p in baseline if p['image_id'] not in set(ids)]+direct; save_json(merged,args.out)
    metrics=evaluate(args.gt,args.out); save_json(metrics,str(Path(args.out).with_suffix(''))+'_eval.json'); print(json.dumps(metrics,indent=2))
if __name__=='__main__': main()
