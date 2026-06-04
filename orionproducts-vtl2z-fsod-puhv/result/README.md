# Orion Products Doubao Base Fusion

Final result: `/data/LPP/cvpr/few_shot/final/orionproducts-vtl2z-fsod-puhv/result/doubao_base_final_fuse.json`
Final eval: `/data/LPP/cvpr/few_shot/final/orionproducts-vtl2z-fsod-puhv/result/doubao_base_final_fuse_eval.json`
Submission pkl: `/data/LPP/cvpr/few_shot/final/orionproducts-vtl2z-fsod-puhv/result/orionproducts-vtl2z-fsod-puhv.pkl`

Metrics:
- mAP: 0.19041114823263472
- mAP50: 0.3222129482717699
- mAP75: 0.19778113920351767

Fusion:
- Base: Doubao direct all norm1000 (`doubao_orion_direct_all_norm1000.json`)
- Aux: existing final Qwen/SAM SKU pipeline result (`qwen_orion_sku_pipeline_all_v1_best.json`)
- Score scales: doubao 0.85, final 0.8, min_score 0.0
- NMS threshold: 0.75
- Final box scale: 1.035
