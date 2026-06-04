# lacrosse-object-detection-fsod-uxkt

Final selected result for the lacrosse dataset.

- Result: `result/`
- Code: `code/qwen_lacrosse_direct_api.py`, `code/qwen_lacrosse_fuse_fallback.py`
- Method: qwen3-vl, qwen3.6, qwen3.5 and fallback predictions fused, then center-fixed bbox scale 1.04.
- Metrics: mAP 0.3611478014280608, mAP50 0.531613988212289, mAP75 0.3824128140687196
