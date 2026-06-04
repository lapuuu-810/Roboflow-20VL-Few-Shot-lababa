# the-dreidel-project-anzyr-fsod-zejm Code Reproduction

This folder stores the scripts used to produce or post-process the saved results for this dataset. Run commands from any working directory unless a script explicitly requires otherwise.

## Code Files

### `dreidel_sam_candidate_pipeline.py`

Command:

```bash
python /data/LPP/cvpr/few_shot/final/the-dreidel-project-anzyr-fsod-zejm/code/dreidel_sam_candidate_pipeline.py
```

Notes: local pipeline or helper script for reproducing the saved result.

### `eval_sam3_prompt_loc.py`

Command:

```bash
python /data/LPP/cvpr/few_shot/final/the-dreidel-project-anzyr-fsod-zejm/code/eval_sam3_prompt_loc.py
```

Notes: local pipeline or helper script for reproducing the saved result.

### `qwen_dreidel_crop_reclass_knn.py`

Command:

```bash
python /data/LPP/cvpr/few_shot/final/the-dreidel-project-anzyr-fsod-zejm/code/qwen_dreidel_crop_reclass_knn.py
```

Notes: local post-processing, filtering, calibration, fusion, or result-conversion script.

## Saved Results

Submission PKL files:
- `the-dreidel-project-anzyr-fsod-zejm.pkl`

Evaluation/result pairs:
- `dreidel_sam3v2_qwen_type_split_symbol_bias_best.json` with `dreidel_sam3v2_qwen_type_split_symbol_bias_best_eval.json`: mAP=0.4616192348425306, mAP50=0.6329728649670718, mAP75=0.5121174607636728, AR100=0.5681071657652036, predictions=299

## Final Submission

For the global submission bundle, the dataset PKL should be present at:

```text
/data/LPP/cvpr/few_shot/final/submission_final_filled/the-dreidel-project-anzyr-fsod-zejm.pkl
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
