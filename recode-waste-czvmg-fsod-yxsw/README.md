# Recode Waste Current Best

Final result JSON: `/data/LPP/cvpr/few_shot/final/recode-waste-czvmg-fsod-yxsw/result/recode_doubao_qwen_fuse_best.json`
Final eval JSON: `/data/LPP/cvpr/few_shot/final/recode-waste-czvmg-fsod-yxsw/result/recode_doubao_qwen_fuse_best_eval.json`
Submission PKL: `/data/LPP/cvpr/few_shot/final/recode-waste-czvmg-fsod-yxsw/result/recode-waste-czvmg-fsod-yxsw.pkl`

Metrics:
- mAP: 0.457203077865662
- mAP50: 0.5548593886677572
- mAP75: 0.4878751742394232
- AR100: 0.5591050428157964

Method: Doubao direct bbox result as base, fused with existing final Qwen tile-filtered result. Best params: qwen score multiplier 0.4, doubao score multiplier 1.0, class-wise NMS 0.75.
PKL category ids are zero-based for submission format.
