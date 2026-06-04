# lacrosse-object-detection-fsod-uxkt Code Reproduction

This folder stores the scripts used to produce or post-process the saved results for this dataset. Run commands from any working directory unless a script explicitly requires otherwise.

## Code Files

### `qwen_lacrosse_direct_api.py`

Command:

```bash
export DASHSCOPE_API_KEY="你的Qwen_API_KEY"
python /data/LPP/cvpr/few_shot/final/lacrosse-object-detection-fsod-uxkt/code/qwen_lacrosse_direct_api.py --resume
```

Notes: API scripts may require provider credentials and can be run on a subset first via their `--image-ids`/resume-style options when available.

### `qwen_lacrosse_fuse_fallback.py`

Command:

```bash
python /data/LPP/cvpr/few_shot/final/lacrosse-object-detection-fsod-uxkt/code/qwen_lacrosse_fuse_fallback.py
```

Notes: local post-processing, filtering, calibration, fusion, or result-conversion script.

## Saved Results

Submission PKL files:
- `lacrosse-object-detection-fsod-uxkt.pkl`

Evaluation/result pairs:
- `eval_scale105.json`: mAP=0.3546529895834813, mAP50=0.5308666045351101, mAP75=0.38041422533765223, AR100=0.5156712032639141, predictions=650
- `lacrosse_all_sources_fuse_scale104.json` with `lacrosse_all_sources_fuse_scale104_eval.json`: mAP=0.3611478014280608, mAP50=0.531613988212289, mAP75=0.3824128140687196, AR100=0.5220117829740591, predictions=650

## Final Submission

For the global submission bundle, the dataset PKL should be present at:

```text
/data/LPP/cvpr/few_shot/final/submission_final_filled/lacrosse-object-detection-fsod-uxkt.pkl
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
