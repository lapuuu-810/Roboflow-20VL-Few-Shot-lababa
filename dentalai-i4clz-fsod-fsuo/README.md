# dentalai-i4clz-fsod-fsuo

Final submission result: `result/best.json`
Final eval reference: `result/best_eval.json`
Submission PKL: `result/dentalai-i4clz-fsod-fsuo.pkl`

Current final source:
- `doubao_qwen_v9_fuse_best.json`
- Qwen v9 full result fused with Doubao raw full result
- Full test coverage: 128 images, 3745 predictions
- Local Roboflow-GT mAP reference: 0.0360235758724364

Why not `qwen_dental_snapfuse_2img_v7`:
- It has local mAP 0.10117282650250842, but only contains predictions for image_id 0 and 1.
- It is therefore not suitable as the full 128-image submission result.

Category-id note:
- Local Roboflow COCO has a dummy `none` class at `category_id=0`.
- Local IDs: `Cavity=1`, `Fillings=2`, `Impacted Tooth=3`, `Implant=4`.
- Server IDs remove `none`: `Cavity=0`, `Fillings=1`, `Impacted Tooth=2`, `Implant=3`.
- Final JSON/PKL use `server_category_id = roboflow_category_id - 1`.
- The current fused source includes Cavity from Doubao and Fillings/Impacted Tooth/Implant from both sources.

The global submission zip has been rebuilt at:

```text
/data/LPP/cvpr/few_shot/final/submission_final_filled_new.zip
```
