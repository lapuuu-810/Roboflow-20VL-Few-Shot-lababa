# DentalAI Code Reproduction

## Selected Qwen Snap-Fuse Result

The selected result is `qwen_dental_snapfuse_2img_v7`, produced by snapping API predictions to SAM3 candidates and sweeping snap/NMS/top-k parameters.

Reproduce from existing API prediction file:

```bash
cd /data/LPP/cvpr/few_shot/sam3-main/best_sam3/dentalai-i4clz-fsod-fsuo
python snap_fuse_qwen_sam3.py \
  --api qwen_dental_sam3crop_2img_top35_metric_v9.json \
  --image-ids 0,1 \
  --out qwen_dental_snapfuse_2img_v7.json
```

Expected metrics:
- mAP: 0.10117282650250842
- mAP50: 0.1503865989435823

## Doubao Same-Flow Test

A Doubao version of the same candidate-classification flow is also saved here:

```bash
export ARK_API_KEY="your_key"
bash /data/LPP/cvpr/few_shot/final/dentalai-i4clz-fsod-fsuo/code/run_doubao_snapfuse_2img.sh
```

This is for comparison only; the final selected result remains `qwen_dental_snapfuse_2img_v7`.


## Gemini Same-Flow Test

A Gemini version following the same candidate API and SAM3 snap-fuse flow is also included:

```bash
export GEMINI_API_KEY="your_key"
bash /data/LPP/cvpr/few_shot/final/dentalai-i4clz-fsod-fsuo/code/run_gemini_snapfuse_2img.sh
```

Main files:

- `gemini_dental_sam3_candidate_api.py`: Gemini candidate classification / rebox stage
- `snap_fuse_gemini_sam3.py`: snap Gemini predictions back to SAM3 candidates and sweep fusion params
- `run_gemini_snapfuse_2img.sh`: 2-image sanity run
- `run_gemini_snapfuse_full.sh`: full-image-id run
