# Paper Parts Doubao Base Qwen/Fallback Fusion

Final result: `/data/LPP/cvpr/few_shot/final/paper-parts-fsod-rmrg/result/doubao_base_qwen_fallback_fuse.json`
Final eval: `/data/LPP/cvpr/few_shot/final/paper-parts-fsod-rmrg/result/doubao_base_qwen_fallback_fuse_eval.json`
Submission pkl: `/data/LPP/cvpr/few_shot/final/paper-parts-fsod-rmrg/result/paper-parts-fsod-rmrg.pkl`

Metrics:
- mAP: 0.4431154769893519
- mAP50: 0.6795553267955764
- mAP75: 0.44913748373955575

Fusion:
- Base: Doubao direct API all norm1000
- Aux: Qwen3VL all norm1000, Qwen36 all norm1000, Qwen3.5/Qwen v1 all norm1000, original SAM3 fallback
- Score scales: doubao 1.0, qwen3vl 0.05, qwen36 0.15, qwen35 0.05, fallback 0.02
- NMS threshold: 0.55
- Final box scale: 1.035
