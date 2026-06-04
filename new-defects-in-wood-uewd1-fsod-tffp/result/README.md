# Result

Selected output:

- `wood_qwen_all_api_fallback_fuse_scale100.json`: COCO bbox predictions.
- `wood_qwen_all_api_fallback_fuse_scale100_eval.json`: COCO bbox evaluation.
- `new-defects-in-wood-uewd1-fsod-tffp.pkl`: EvalAI-style submission pickle.
- `scale_sweep.json`: center-fixed bbox scale sweep.
- `predictions_fused_before_scale.json`: fused predictions before scale sweep.
- `sample_contact_sheet.jpg`: visualization sample.

Metrics:

```json
{
  "scale": 1.0,
  "mAP": 0.19059896598182802,
  "mAP50": 0.3654980060802297,
  "mAP75": 0.18102650007800003,
  "AR100": 0.39703239508306865,
  "predictions": 1108
}
```

Postprocess: three Qwen direct detectors (`qwen3-vl-plus`, `qwen3.6-plus`, `qwen3.5-plus`) plus original fallback are fused by score scaling and class-wise NMS; bbox scale sweep selected 1.0.
