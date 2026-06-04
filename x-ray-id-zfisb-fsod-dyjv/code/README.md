# x-ray-id-zfisb-fsod-dyjv Code Reproduction

This folder stores the scripts used to produce or post-process the saved results for this dataset. Run commands from any working directory unless a script explicitly requires otherwise.

## Code Files

### `sam3_finger_bone_filter17.py`

Command:

```bash
python /data/LPP/cvpr/few_shot/final/x-ray-id-zfisb-fsod-dyjv/code/sam3_finger_bone_filter17.py
```

Notes: local post-processing, filtering, calibration, fusion, or result-conversion script.

### `xray_make_sam3_filter17_fixed_params.py`

Command:

```bash
python /data/LPP/cvpr/few_shot/final/x-ray-id-zfisb-fsod-dyjv/code/xray_make_sam3_filter17_fixed_params.py
```

Notes: local post-processing, filtering, calibration, fusion, or result-conversion script.

### `xray_run_full_current_best.sh`

Command:

```bash
bash /data/LPP/cvpr/few_shot/final/x-ray-id-zfisb-fsod-dyjv/code/xray_run_full_current_best.sh
```

Notes: shell wrapper for the saved pipeline configuration.

### `xray_slot_prior_calibrate_v1.py`

Command:

```bash
python /data/LPP/cvpr/few_shot/final/x-ray-id-zfisb-fsod-dyjv/code/xray_slot_prior_calibrate_v1.py
```

Notes: local post-processing, filtering, calibration, fusion, or result-conversion script.

### `xray_supervised_calibrate_percat_v2.py`

Command:

```bash
python /data/LPP/cvpr/few_shot/final/x-ray-id-zfisb-fsod-dyjv/code/xray_supervised_calibrate_percat_v2.py
```

Notes: local post-processing, filtering, calibration, fusion, or result-conversion script.

## Saved Results

Evaluation/result pairs:
- `best.json` with `best_eval.json`: mAP=0.07290475632370547, mAP50=0.2167689578734597, mAP75=0.02272161242108239
- `best_eval_before_clahe_wh130.json`: mAP=0.03317756945214547, mAP50=0.12167569450240316, mAP75=0.009555436412139
- `current_pkl_eval.json`: mAP=0.07290475632370547, mAP50=0.2167689578734597, mAP75=0.02272161242108239
- `current_pkl_eval_predictions.json`
- `doubao_xray_realesrgan_clahe_full_whscale_130_130.json` with `doubao_xray_realesrgan_clahe_full_whscale_130_130_eval.json`: mAP=0.07290475632370547, mAP50=0.2167689578734597, mAP75=0.02272161242108239

## Final Submission

For the global submission bundle, the dataset PKL should be present at:

```text
/data/LPP/cvpr/few_shot/final/submission_final_filled/x-ray-id-zfisb-fsod-dyjv.pkl
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
