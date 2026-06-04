# wildfire-smoke-fsod-myxt Code Reproduction

This folder stores the scripts used to produce or post-process the saved results for this dataset. Run commands from any working directory unless a script explicitly requires otherwise.

## Code Files

### `doubao_wildfire_smoke_direct_api.py`

Command:

```bash
export ARK_API_KEY="你的豆包API_KEY"
python /data/LPP/cvpr/few_shot/final/wildfire-smoke-fsod-myxt/code/doubao_wildfire_smoke_direct_api.py --image-ids 0-74 --resume --vis
```

Notes: API scripts may require provider credentials and can be run on a subset first via their `--image-ids`/resume-style options when available.

### `qwen_wildfire_smoke_direct_api.py`

Command:

```bash
export DASHSCOPE_API_KEY="你的Qwen_API_KEY"
python /data/LPP/cvpr/few_shot/final/wildfire-smoke-fsod-myxt/code/qwen_wildfire_smoke_direct_api.py --image-ids 0-74 --resume --vis
```

Notes: API scripts may require provider credentials and can be run on a subset first via their `--image-ids`/resume-style options when available.

### `wildfire_qwen_doubao_post_fuse.py`

Command:

```bash
python /data/LPP/cvpr/few_shot/final/wildfire-smoke-fsod-myxt/code/wildfire_qwen_doubao_post_fuse.py --rebuild-zip
```

Notes: local post-processing, filtering, calibration, fusion, or result-conversion script.

## Saved Results

Submission PKL files:
- `wildfire-smoke-fsod-myxt.pkl`

Evaluation/result pairs:
- `fuse_qwen_sx1.05_sy1.05_dx-0.06_dy0.0_doubao_sx1.05_sy1.05_dx0.0_dy0.0_ma0_mb1_nms0.85.json` with `fuse_qwen_sx1.05_sy1.05_dx-0.06_dy0.0_doubao_sx1.05_sy1.05_dx0.0_dy0.0_ma0_mb1_nms0.85_eval.json`: mAP=0.29434374480371167, mAP50=0.7252477509348415, mAP75=0.15873866638687037, count=141
- `wildfire_qwen_doubao_post_fuse_best.json` with `wildfire_qwen_doubao_post_fuse_best_eval.json`: mAP=0.29434374480371167, mAP50=0.7252477509348415, mAP75=0.15873866638687037, count=141

## Final Submission

For the global submission bundle, the dataset PKL should be present at:

```text
/data/LPP/cvpr/few_shot/final/submission_final_filled/wildfire-smoke-fsod-myxt.pkl
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
