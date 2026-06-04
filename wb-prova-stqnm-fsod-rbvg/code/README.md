# wb-prova-stqnm-fsod-rbvg Code Reproduction

This folder stores the scripts used to produce or post-process the saved results for this dataset. Run commands from any working directory unless a script explicitly requires otherwise.

## Code Files

### `qwen_wb_age_fusion_knn.py`

Command:

```bash
python /data/LPP/cvpr/few_shot/final/wb-prova-stqnm-fsod-rbvg/code/qwen_wb_age_fusion_knn.py
```

Notes: local post-processing, filtering, calibration, fusion, or result-conversion script.

### `qwen_wb_direct_api_v2.py`

Command:

```bash
export DASHSCOPE_API_KEY="你的Qwen_API_KEY"
python /data/LPP/cvpr/few_shot/final/wb-prova-stqnm-fsod-rbvg/code/qwen_wb_direct_api_v2.py --resume
```

Notes: API scripts may require provider credentials and can be run on a subset first via their `--image-ids`/resume-style options when available.

### `qwen_wb_knn_prompt_reclass.py`

Command:

```bash
python /data/LPP/cvpr/few_shot/final/wb-prova-stqnm-fsod-rbvg/code/qwen_wb_knn_prompt_reclass.py
```

Notes: local post-processing, filtering, calibration, fusion, or result-conversion script.

### `qwen_wb_samrefine_plus_sam.py`

Command:

```bash
python /data/LPP/cvpr/few_shot/final/wb-prova-stqnm-fsod-rbvg/code/qwen_wb_samrefine_plus_sam.py
```

Notes: local post-processing, filtering, calibration, fusion, or result-conversion script.

### `wb_sam3_topk_postprocess_v1.py`

Command:

```bash
python /data/LPP/cvpr/few_shot/final/wb-prova-stqnm-fsod-rbvg/code/wb_sam3_topk_postprocess_v1.py
```

Notes: local post-processing, filtering, calibration, fusion, or result-conversion script.

## Saved Results

Submission PKL files:
- `wb-prova-stqnm-fsod-rbvg.pkl`

Evaluation/result pairs:
- `wb_qwen_samrefine_scale105_knn_qwen_pigprotect.json` with `wb_qwen_samrefine_scale105_knn_qwen_pigprotect_eval.json`: mAP=0.5558791854022012, mAP50=0.7136838321118446, mAP75=0.6346112159103828, AR100=0.7275940156661841, predictions=440
- `wb_sam3_topk_postprocess_v1.json` with `wb_sam3_topk_postprocess_v1_eval.json`: mAP=0.12381853832034223, mAP50=0.1699355203326999, mAP75=0.1471539364949942, AR100=0.37354439561255465

## Final Submission

For the global submission bundle, the dataset PKL should be present at:

```text
/data/LPP/cvpr/few_shot/final/submission_final_filled/wb-prova-stqnm-fsod-rbvg.pkl
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
