# orionproducts-vtl2z-fsod-puhv Code Reproduction

This folder stores the scripts used to produce or post-process the saved results for this dataset. Run commands from any working directory unless a script explicitly requires otherwise.

## Code Files

### `qwen_orion_sku_pipeline.py`

Command:

```bash
python /data/LPP/cvpr/few_shot/final/orionproducts-vtl2z-fsod-puhv/code/qwen_orion_sku_pipeline.py
```

Notes: local pipeline or helper script for reproducing the saved result.

### `run_orion_sku_pipeline_all_v1_best.sh`

Command:

```bash
bash /data/LPP/cvpr/few_shot/final/orionproducts-vtl2z-fsod-puhv/code/run_orion_sku_pipeline_all_v1_best.sh
```

Notes: shell wrapper for the saved pipeline configuration.

## Saved Results

Submission PKL files:
- `orionproducts-vtl2z-fsod-puhv.pkl`

Evaluation/result pairs:
- `doubao_base_final_fuse.json` with `doubao_base_final_fuse_eval.json`: mAP=0.19041114823263472, mAP50=0.3222129482717699, mAP75=0.19778113920351767, AR100=0.3674985706963154, predictions=1039
- `eval_fused_before_scale.json`: mAP=0.18117582216419587, mAP50=0.31645740098938824, mAP75=0.18370977636768518, AR100=0.356122655901136, predictions=1039
- `qwen_orion_sku_pipeline_all_v1_best.json` with `qwen_orion_sku_pipeline_all_v1_best_eval.json`: mAP=0.10013577386719326, mAP50=0.1483244086612559, mAP75=0.12141423442404438

## Final Submission

For the global submission bundle, the dataset PKL should be present at:

```text
/data/LPP/cvpr/few_shot/final/submission_final_filled/orionproducts-vtl2z-fsod-puhv.pkl
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


## Gemini Additions

Gemini versions of the Orion direct detector and SAM3 fusion pipeline are also included.

Direct whole-image detection:

```bash
export GEMINI_API_KEY="your_key"
bash /data/LPP/few_shot/final/orionproducts-vtl2z-fsod-puhv/code/run_gemini_orion_direct_all_v1.sh
```

SAM3 + Gemini fusion pipeline:

```bash
export GEMINI_API_KEY="your_key"
bash /data/LPP/few_shot/final/orionproducts-vtl2z-fsod-puhv/code/run_gemini_orion_sku_pipeline_all_v1_best.sh
```

Files:
- `gemini_orion_direct_api.py`: Gemini direct full-image detection
- `gemini_orion_sku_pipeline.py`: Gemini knowledge/direct/classification fusion pipeline
- `run_gemini_orion_direct_all_v1.sh`: direct detector wrapper
- `run_gemini_orion_sku_pipeline_all_v1_best.sh`: fusion pipeline wrapper
