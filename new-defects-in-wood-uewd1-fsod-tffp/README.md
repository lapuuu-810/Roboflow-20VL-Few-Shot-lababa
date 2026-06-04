# new-defects-in-wood-uewd1-fsod-tffp

Final selected result for the wood-defect dataset.

- Result: `result/`
- Code: `code/qwen_wood_direct_api.py`, `code/qwen_wood_fuse_qwen_fallback.py`, `code/qwen_wood_sam3_panel_api.py`
- Method: qwen3-vl, qwen3.6, qwen3.5 direct bbox detections fused with original fallback; center-fixed bbox scale sweep selected scale 1.0.
- Metrics: mAP 0.19059896598182802, mAP50 0.3654980060802297, mAP75 0.18102650007800003
