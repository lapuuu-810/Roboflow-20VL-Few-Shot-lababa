# DentalAI Final Result

Current final submission files:
- `best.json`
- `best_eval.json`
- `qwen_dental_sam3crop_all_top35_best_v9.json`
- `qwen_dental_sam3crop_all_top35_best_v9_eval.json`
- `dentalai-i4clz-fsod-fsuo.pkl`

Final source:
- `/data/LPP/cvpr/few_shot/sam3-main/best_sam3/dentalai-i4clz-fsod-fsuo/doubao_qwen_v9_fuse_best.json`
- Qwen v9 full result fused with Doubao raw full result
- Full coverage: 128 images, 3745 predictions
- Local Roboflow-GT mAP: 0.0360235758724364

`qwen_dental_snapfuse_2img_v7` is preserved only as a reference because it covers only 2 images, despite its higher local mAP.

Category-id correction:
- Local Roboflow IDs include dummy `none=0` and use `Cavity=1`, `Fillings=2`, `Impacted Tooth=3`, `Implant=4`.
- Server IDs are 0-based without `none`: `Cavity=0`, `Fillings=1`, `Impacted Tooth=2`, `Implant=3`.
- Final JSON/PKL are remapped with `category_id - 1`.
- The local Roboflow-ID source is saved as `qwen_dental_sam3crop_all_top35_best_v9_local_roboflow_ids.json`.

PKL format matches the reference submission format: outer list of image records, `instances` list, and `bbox` stored as `numpy.ndarray(dtype=float32, shape=(4,))`.
