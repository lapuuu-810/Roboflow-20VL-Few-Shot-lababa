# paper-parts-fsod-rmrg Code Reproduction

This folder stores the scripts used to produce or post-process the saved results for this dataset. Run commands from any working directory unless a script explicitly requires otherwise.

## Code Files

### `qwen_paper_direct_api.py`

Command:

```bash
export DASHSCOPE_API_KEY="你的Qwen_API_KEY"
python /data/LPP/cvpr/few_shot/final/paper-parts-fsod-rmrg/code/qwen_paper_direct_api.py --resume
```

Notes: API scripts may require provider credentials and can be run on a subset first via their `--image-ids`/resume-style options when available.

### `qwen_paper_fuse_qwen_fallback.py`

Command:

```bash
python /data/LPP/cvpr/few_shot/final/paper-parts-fsod-rmrg/code/qwen_paper_fuse_qwen_fallback.py
```

Notes: local post-processing, filtering, calibration, fusion, or result-conversion script.

## Saved Results

Submission PKL files:
- `paper-parts-fsod-rmrg.pkl`

Evaluation/result pairs:
- `doubao_base_qwen_fallback_fuse.json` with `doubao_base_qwen_fallback_fuse_eval.json`: mAP=0.4431154769893519, mAP50=0.6795553267955764, mAP75=0.44913748373955575, AR100=0.6130837408732692, predictions=5869
- `qwen_paper_direct_api_v1_all_norm1000.json` with `qwen_paper_direct_api_v1_all_norm1000_eval.json`: mAP=0.18593828069913015, mAP50=0.40046538691511235, mAP75=0.1495991429857239

## Final Submission

For the global submission bundle, the dataset PKL should be present at:

```text
/data/LPP/cvpr/few_shot/final/submission_final_filled/paper-parts-fsod-rmrg.pkl
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
