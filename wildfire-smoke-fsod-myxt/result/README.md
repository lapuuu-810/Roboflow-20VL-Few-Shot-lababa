# Wildfire Smoke Result

Best result: `wildfire_qwen_doubao_post_fuse_best.json`
Best eval: `wildfire_qwen_doubao_post_fuse_best_eval.json`
PKL: `wildfire-smoke-fsod-myxt.pkl`

Source predictions:
- Qwen: `/data/LPP/cvpr/few_shot/sam3-main/best_sam3/wildfire-smoke-fsod-myxt/qwen_smoke_direct_0_9_36/predictions.json`
- Doubao: `/data/LPP/cvpr/few_shot/sam3-main/best_sam3/wildfire-smoke-fsod-myxt/doubao_smoke_direct_0_9/predictions.json`

Metrics:
- mAP: 0.29434374480371167
- mAP50: 0.7252477509348415
- mAP75: 0.15873866638687037
- predictions: 141

Best params: `{'a': 'qwen_sx1.05_sy1.05_dx-0.06_dy0.0', 'b': 'doubao_sx1.05_sy1.05_dx0.0_dy0.0', 'ma': 0, 'mb': 1, 'nms': 0.85}`
