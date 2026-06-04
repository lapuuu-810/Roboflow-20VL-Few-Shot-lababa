# aerial-airport-7ap9o-fsod-ddgc Code Reproduction

This folder stores the scripts used to produce or post-process the saved results for this dataset. Run commands from any working directory unless a script explicitly requires otherwise.

## Code Files

### `reproduce_boxcal_reg_area.py`

Command:

```bash
python /data/LPP/cvpr/few_shot/final/aerial-airport-7ap9o-fsod-ddgc/code/reproduce_boxcal_reg_area.py
```

Notes: local pipeline or helper script for reproducing the saved result.

## Saved Results

Submission PKL files:
- `aerial-airport-7ap9o-fsod-ddgc.pkl`

Evaluation/result pairs:
- `metrics.json`: mAP=0.5588982300093266, mAP50=0.9014014389126267, mAP75=0.6422519602593655

## Final Submission

For the global submission bundle, the dataset PKL should be present at:

```text
/data/LPP/cvpr/few_shot/final/submission_final_filled/aerial-airport-7ap9o-fsod-ddgc.pkl
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
