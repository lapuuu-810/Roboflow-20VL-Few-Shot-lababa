# Soda Bottles Current Best

Final result JSON: `/data/LPP/cvpr/few_shot/final/soda-bottles-fsod-haga/result/soda_doubao_sam3_fuse_best.json`
Final eval JSON: `/data/LPP/cvpr/few_shot/final/soda-bottles-fsod-haga/result/soda_doubao_sam3_fuse_best_eval.json`
Submission PKL: `/data/LPP/cvpr/few_shot/final/soda-bottles-fsod-haga/result/soda-bottles-fsod-haga.pkl`

Metrics:
- mAP: 0.29053820762491916
- mAP50: 0.7067611510550772
- mAP75: 0.19813326160543998

Method: current final SAM3 color-prompt result fused with Doubao direct color-bottle detections. Best simple fusion uses SAM3 score multiplier 1.0, Doubao score multiplier 0.4, class-wise NMS 0.65.
