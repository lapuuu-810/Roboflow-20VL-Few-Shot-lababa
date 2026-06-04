# defect-detection-yjplx-fxobh-fsod-amdi Code Reproduction

This folder stores the scripts used to produce or post-process the saved results for this dataset. Run commands from any working directory unless a script explicitly requires otherwise.

## Code Files

### `qwen_defect_detection_api_pipeline.py`

Command:

```bash
export DASHSCOPE_API_KEY="你的Qwen_API_KEY"
python /data/LPP/cvpr/few_shot/final/defect-detection-yjplx-fxobh-fsod-amdi/code/qwen_defect_detection_api_pipeline.py --resume
```

Notes: API scripts may require provider credentials and can be run on a subset first via their `--image-ids`/resume-style options when available.

### `qwen_defect_direct_api_v2.py`

Command:

```bash
export DASHSCOPE_API_KEY="你的Qwen_API_KEY"
python /data/LPP/cvpr/few_shot/final/defect-detection-yjplx-fxobh-fsod-amdi/code/qwen_defect_direct_api_v2.py --resume
```

Notes: API scripts may require provider credentials and can be run on a subset first via their `--image-ids`/resume-style options when available.


### `doubao_defect_direct_api_v2.py`

Doubao version of the Qwen v2 multi-pass logic. It keeps the same fishplate / fastener / grid passes and replaces only the API backend with Doubao Responses API.

Small-batch command:

```bash
export ARK_API_KEY="你的Doubao_API_KEY"
python /data/LPP/cvpr/few_shot/final/defect-detection-yjplx-fxobh-fsod-amdi/code/doubao_defect_direct_api_v2.py \
  --image-ids 0,1,2,3,4,5,6,7,8,9 \
  --out /data/LPP/cvpr/few_shot/sam3-main/best_sam3/defect-detection-yjplx-fxobh-fsod-amdi/doubao_defect_direct_v2_0_9.json \
  --raw-dir /data/LPP/cvpr/few_shot/sam3-main/best_sam3/defect-detection-yjplx-fxobh-fsod-amdi/doubao_defect_direct_v2_0_9_raw \
  --passes fishplate,fastener \
  --coord-mode norm1000 \
  --reference-mode full \
  --ref-per-class 1 \
  --timeout 240 \
  --retries 2 \
  --resume \
  --visualize \
  --evaluate
```

For higher fastener recall, add the crop pass: `--passes fishplate,fastener,grid`.

### `qwen_defect_full_fusion_v2.py`

Command:

```bash
python /data/LPP/cvpr/few_shot/final/defect-detection-yjplx-fxobh-fsod-amdi/code/qwen_defect_full_fusion_v2.py
```

Notes: local post-processing, filtering, calibration, fusion, or result-conversion script.

### `qwen_defect_stage1_merge.py`

Command:

```bash
python /data/LPP/cvpr/few_shot/final/defect-detection-yjplx-fxobh-fsod-amdi/code/qwen_defect_stage1_merge.py
```

Notes: local pipeline or helper script for reproducing the saved result.


### `defect_doubao_qwen_fuse.py`

Offline fusion script. It rebuilds Doubao v2 predictions from raw JSON/TXT files, fuses them with the Qwen final result, evaluates on GT, writes final JSON/PKL, and optionally rebuilds the global submission zip.

Reproduce current final result:

```bash
python /data/LPP/cvpr/few_shot/final/defect-detection-yjplx-fxobh-fsod-amdi/code/defect_doubao_qwen_fuse.py \
  --qwen-mul 0.9 \
  --doubao-v2-mul 1.0 \
  --doubao-direct-mul 0.0 \
  --doubao-v2-thr 0.0 \
  --doubao-v2-keep 1,2,3,4 \
  --nms-thr 0.75 \
  --rebuild-zip
```

Expected metrics: mAP `0.3050309094446834`, mAP50 `0.5887439424804463`, mAP75 `0.29977797515314636`.

To repeat the parameter sweep, add `--search`.

## Saved Results

Evaluation/result pairs:
- `defect_doubao_qwen_fuse_best.json` with `defect_doubao_qwen_fuse_best_eval.json`: mAP=0.3050309094446834, mAP50=0.5887439424804463, mAP75=0.29977797515314636
- `qwen_defect_full_best_current_v12.json` with `qwen_defect_full_best_current_v12_eval.json`: mAP=0.2920368853328822, mAP50=0.5363153446963971, mAP75=0.2845897192073042

## Final Submission

For the global submission bundle, the dataset PKL should be present at:

```text
/data/LPP/cvpr/few_shot/final/submission_final_filled/defect-detection-yjplx-fxobh-fsod-amdi.pkl
```

Rebuild the global zip after updating any PKL:

```bash
python - <<'PY'
import os, zipfile
base='/data/LPP/cvpr/few_shot/final/submission_final_filled'
out='/data/LPP/cvpr/few_shot/final/submission_final_filled_new.zip'
tmp=out+'.tmp'
with zipfile.ZipFile(tmp,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for fn in sorted(os.listdir(base)):
        if fn.endswith('.pkl'):
            z.write(os.path.join(base,fn), arcname=fn)
os.replace(tmp,out)
PY
```
