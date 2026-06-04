# water-meter-jbktv-7vz5k-fsod-ftoz Code Reproduction

This folder stores the scripts used to produce or post-process the saved results for this dataset. Run commands from any working directory unless a script explicitly requires otherwise.

## Code Files

### `qwen_water_meter_stage1_localize.py`

Command:

```bash
python /data/LPP/cvpr/few_shot/final/water-meter-jbktv-7vz5k-fsod-ftoz/code/qwen_water_meter_stage1_localize.py
```

Notes: local pipeline or helper script for reproducing the saved result.

### `qwen_water_meter_stage2_read_digits.py`

Command:

```bash
python /data/LPP/cvpr/few_shot/final/water-meter-jbktv-7vz5k-fsod-ftoz/code/qwen_water_meter_stage2_read_digits.py
```

Notes: local pipeline or helper script for reproducing the saved result.

### `refine_stage1_panel_localization.py`

Command:

```bash
python /data/LPP/cvpr/few_shot/final/water-meter-jbktv-7vz5k-fsod-ftoz/code/refine_stage1_panel_localization.py
```

Notes: local post-processing, filtering, calibration, fusion, or result-conversion script.

### `sam3_from_qwen_number_prompt.py`

Command:

```bash
python /data/LPP/cvpr/few_shot/final/water-meter-jbktv-7vz5k-fsod-ftoz/code/sam3_from_qwen_number_prompt.py
```

Notes: local pipeline or helper script for reproducing the saved result.

## Saved Results

Submission PKL files:
- `water-meter-jbktv-7vz5k-fsod-ftoz.pkl`

Evaluation/result pairs:
- `water_meter_qwen_reading_sam3_slot_scale1115.json` with `water_meter_qwen_reading_sam3_slot_scale1115_eval.json`: mAP=0.19478993205259887, mAP50=0.4562274441877299, mAP75=0.1743701048043861, AR100=0.23814876432666193, predictions=439
- `water_meter_realesrgan_eval_fused_before_scale.json`: mAP=0.39642564717125367, mAP50=0.9086699459429565, mAP75=0.30009754677717093, AR100=0.5419240949874373, predictions=1129
- `water_meter_realesrgan_vlm_final_fuse.json` with `water_meter_realesrgan_vlm_final_fuse_eval.json`: mAP=0.39831292666994444, mAP50=0.9059701186984151, mAP75=0.28693627802994925, AR100=0.5415475576284201, predictions=1129

## Final Submission

For the global submission bundle, the dataset PKL should be present at:

```text
/data/LPP/cvpr/few_shot/final/submission_final_filled/water-meter-jbktv-7vz5k-fsod-ftoz.pkl
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
