# Result

Selected output:

- `lacrosse_all_sources_fuse_scale104.json`: COCO bbox predictions.
- `lacrosse_all_sources_fuse_scale104_eval.json`: COCO bbox evaluation.
- `lacrosse-object-detection-fsod-uxkt.pkl`: EvalAI-style submission pickle.
- `scale_sweep.json`: center-fixed bbox scale sweep.
- `predictions_scale105.json`, `eval_scale105.json`: fixed 1.05 scale check.

Metrics:

```json
{
  "scale": 1.04,
  "mAP": 0.3611478014280608,
  "mAP50": 0.531613988212289,
  "mAP75": 0.3824128140687196,
  "AR100": 0.5220117829740591,
  "predictions": 650
}
```

Postprocess: four-source fusion (`qwen3vl`, `qwen3.6`, `qwen3.5`, fallback) followed by bbox center fixed scale 1.04.
