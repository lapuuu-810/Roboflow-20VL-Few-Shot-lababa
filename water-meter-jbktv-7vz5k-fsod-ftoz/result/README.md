# Water Meter Real-ESRGAN VLM Fusion

Final result: `/data/LPP/cvpr/few_shot/final/water-meter-jbktv-7vz5k-fsod-ftoz/result/water_meter_realesrgan_vlm_final_fuse.json`
Final eval: `/data/LPP/cvpr/few_shot/final/water-meter-jbktv-7vz5k-fsod-ftoz/result/water_meter_realesrgan_vlm_final_fuse_eval.json`
Submission pkl: `/data/LPP/cvpr/few_shot/final/water-meter-jbktv-7vz5k-fsod-ftoz/result/water-meter-jbktv-7vz5k-fsod-ftoz.pkl`

Metrics:
- mAP: 0.39831292666994444
- mAP50: 0.9059701186984151
- mAP75: 0.28693627802994925

Fusion:
- Sources: Doubao Real-ESRGAN direct, Qwen Real-ESRGAN direct, Qwen v2 Real-ESRGAN direct, Qwen raw_1 Real-ESRGAN direct, existing final Qwen/SAM slot result
- Score scales: doubao 0.8, qwen 0.0, qwen_v2 0.1, qwen_raw1 0.08, final 0.2
- NMS threshold: 0.75
- Final box scale: 0.99
