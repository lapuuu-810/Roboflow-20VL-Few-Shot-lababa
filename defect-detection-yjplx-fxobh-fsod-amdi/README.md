# defect-detection-yjplx-fxobh-fsod-amdi

Final result: `result/defect_doubao_qwen_fuse_best.json`
Final eval: `result/defect_doubao_qwen_fuse_best_eval.json`
Submission PKL: `result/defect-detection-yjplx-fxobh-fsod-amdi.pkl`

Metrics:
- mAP: 0.3050309094446834
- mAP50: 0.5887439424804463
- mAP75: 0.29977797515314636
- predictions: 2544

Baselines:
- Original Qwen final: mAP 0.29209036523818244, mAP50 0.5363153446963971, mAP75 0.2845897192073042
- Doubao v2 from raw: mAP 0.21920504697759075, mAP50 0.4371196671682071, mAP75 0.21849836784892923
- Early Doubao direct 0-9: mAP 0.014757372404860622

Best fusion parameters:
- `qwen_mul=0.9`
- `doubao_v2_mul=1.0`
- `doubao_direct_mul=0.0`
- `doubao_v2_keep=1,2,3,4`
- `doubao_v2_thr=0.0`
- `nms_thr=0.75`

Code:
- `code/doubao_defect_direct_api_v2.py`: Doubao multi-pass inference, following the previous Qwen v2 logic.
- `code/defect_doubao_qwen_fuse.py`: aggregates Doubao raw outputs, fuses with Qwen final, evaluates, writes JSON/PKL, and can rebuild submission zip.
