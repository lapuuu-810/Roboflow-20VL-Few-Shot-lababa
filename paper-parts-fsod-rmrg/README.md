# paper-parts-fsod-rmrg

Final direct Qwen API output for the paper-parts document layout detection dataset.

## Result
- Final prediction: `result/qwen_paper_direct_api_v1_all_norm1000.json`
- Evaluation: `result/qwen_paper_direct_api_v1_all_norm1000_eval.json`
- Metrics: mAP 0.18593828069913015, mAP50 0.40046538691511235, mAP75 0.1495991429857239

## Reproduce
The final submitted file was produced by direct Qwen API layout detection over all 500 test images. Qwen returned coordinates in 0-1000 space, so the run uses `--coord-mode norm1000` to scale boxes back to the original image size.

```bash
cd /data/LPP/cvpr/few_shot/sam3-main/best_sam3/paper-parts-fsod-rmrg

DASHSCOPE_API_KEY='<your_key>' python qwen_paper_direct_api.py \
  --image-ids all \
  --out qwen_paper_direct_api_v1_all_norm1000.json \
  --raw-dir qwen_paper_direct_api_v1_all_raw \
  --resume \
  --visualize \
  --evaluate \
  --coord-mode norm1000
```

Key code copy is stored in `code/`. The raw intermediate API cache is kept in the original run directory: `/data/LPP/cvpr/few_shot/sam3-main/best_sam3/paper-parts-fsod-rmrg/qwen_paper_direct_api_v1_all_raw`.
