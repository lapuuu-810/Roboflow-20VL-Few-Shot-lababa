# orionproducts-vtl2z-fsod-puhv

Final calibrated output for the Orion products SKU detection dataset.

## Result
- Final prediction: `result/qwen_orion_sku_pipeline_all_v1_best.json`
- Evaluation: `result/qwen_orion_sku_pipeline_all_v1_best_eval.json`
- Metrics: mAP 0.10013577386719326, mAP50 0.1483244086612559, mAP75 0.12141423442404438

## Reproduce
The final submitted file was produced by the Qwen + SAM3 SKU pipeline over all images:

```bash
cd /data/LPP/cvpr/few_shot/sam3-main/best_sam3/orionproducts-vtl2z-fsod-puhv

DASHSCOPE_API_KEY='<your_key>' conda run -n qwen python qwen_orion_sku_pipeline.py \
  --image-ids all \
  --topk-per-image 40 \
  --min-area 80 \
  --max-area 200000 \
  --sam-min-score 0.0 \
  --candidate-nms 0.45 \
  --crop-max-side 768 \
  --stage3-score-mode qconf \
  --final-score-thr 0.05 \
  --final-nms 0.25 \
  --max-per-image 120 \
  --out qwen_orion_sku_pipeline_all_v1_best.json \
  --raw-dir qwen_orion_sku_pipeline_all_v1_best_raw \
  --resume \
  --evaluate
```

Key code copies are stored in `code/`. The raw intermediate crop/classification directory is kept in the original run directory.
