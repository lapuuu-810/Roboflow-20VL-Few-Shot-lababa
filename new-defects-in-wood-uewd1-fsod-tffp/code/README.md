# new-defects-in-wood-uewd1-fsod-tffp Code Reproduction

This folder stores the scripts used to produce or post-process the saved results for this dataset. Run commands from any working directory unless a script explicitly requires otherwise.

## Code Files

### `qwen_wood_direct_api.py`

Command:

```bash
export DASHSCOPE_API_KEY="你的Qwen_API_KEY"
python /data/LPP/cvpr/few_shot/final/new-defects-in-wood-uewd1-fsod-tffp/code/qwen_wood_direct_api.py --resume
```

Notes: API scripts may require provider credentials and can be run on a subset first via their `--image-ids`/resume-style options when available.

### `qwen_wood_fuse_qwen_fallback.py`

Command:

```bash
python /data/LPP/cvpr/few_shot/final/new-defects-in-wood-uewd1-fsod-tffp/code/qwen_wood_fuse_qwen_fallback.py
```

Notes: local post-processing, filtering, calibration, fusion, or result-conversion script.

### `qwen_wood_sam3_panel_api.py`

Command:

```bash
export DASHSCOPE_API_KEY="你的Qwen_API_KEY"
python /data/LPP/cvpr/few_shot/final/new-defects-in-wood-uewd1-fsod-tffp/code/qwen_wood_sam3_panel_api.py --resume
```

Notes: API scripts may require provider credentials and can be run on a subset first via their `--image-ids`/resume-style options when available.

### `wood_panel_sam3_hybrid_final.py`

Command:

```bash
python /data/LPP/cvpr/few_shot/final/new-defects-in-wood-uewd1-fsod-tffp/code/wood_panel_sam3_hybrid_final.py
```

Notes: local pipeline or helper script for reproducing the saved result.

## Saved Results

Submission PKL files:
- `new-defects-in-wood-uewd1-fsod-tffp.pkl`

Evaluation/result pairs:
- `eval_fused_before_scale.json`: mAP=0.19059896598182802, mAP50=0.3654980060802297, mAP75=0.18102650007800003, AR100=0.39703239508306865, predictions=1108
- `wood_qwen_all_api_fallback_fuse_scale100.json` with `wood_qwen_all_api_fallback_fuse_scale100_eval.json`: mAP=0.19059896598182802, mAP50=0.3654980060802297, mAP75=0.18102650007800003, AR100=0.39703239508306865, predictions=1108
- `wood_qwen_panel_sam3_hybrid_all_v2.json` with `wood_qwen_panel_sam3_hybrid_all_v2_eval.json`: mAP=0.09550504613839672, mAP50=0.17220662124646202, mAP75=0.1013306085276887

## Final Submission

For the global submission bundle, the dataset PKL should be present at:

```text
/data/LPP/cvpr/few_shot/final/submission_final_filled/new-defects-in-wood-uewd1-fsod-tffp.pkl
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
