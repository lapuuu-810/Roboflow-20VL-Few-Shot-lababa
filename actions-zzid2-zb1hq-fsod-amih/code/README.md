# actions-zzid2-zb1hq-fsod-amih Code Reproduction

This folder stores the scripts used to produce or post-process the saved results for this dataset. Run commands from any working directory unless a script explicitly requires otherwise.

## Code Files

### `qwen_volleyball_pipeline.py`

Command:

```bash
python /data/LPP/cvpr/few_shot/final/actions-zzid2-zb1hq-fsod-amih/code/qwen_volleyball_pipeline.py
```

Notes: local pipeline or helper script for reproducing the saved result.

## Saved Results

Submission PKL files:
- `actions-zzid2-zb1hq-fsod-amih.pkl`

Evaluation/result pairs:
- `refined_coco_predictions.json` with `refined_coco_predictions_eval.json`
- `full_metrics.json`: mAP=0.15560081140053694, mAP50=0.3121049858911415, mAP75=0.13429024923894065

## Final Submission

For the global submission bundle, the dataset PKL should be present at:

```text
/data/LPP/cvpr/few_shot/final/submission_final_filled/actions-zzid2-zb1hq-fsod-amih.pkl
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
