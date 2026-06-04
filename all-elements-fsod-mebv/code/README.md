# all-elements-fsod-mebv Code Reproduction

## `doubao_all_elements_direct_api.py`

Doubao direct UI-element inference.

```bash
export ARK_API_KEY="你的豆包API_KEY"
python /data/LPP/cvpr/few_shot/final/all-elements-fsod-mebv/code/doubao_all_elements_direct_api.py \
  --image-ids 0-69 \
  --out-dir /data/LPP/cvpr/few_shot/sam3-main/best_sam3/all-elements-fsod-mebv/doubao_all_elements_direct_0_9 \
  --coord-mode norm1000 \
  --max-side 1600 \
  --score-thr 0.0 \
  --nms-thr 0.45 \
  --vis \
  --resume
```

## `qwen_api_direct_ui_detect.py`

Original Qwen direct UI detector used by the baseline final result.

```bash
export DASHSCOPE_API_KEY="你的Qwen_API_KEY"
python /data/LPP/cvpr/few_shot/final/all-elements-fsod-mebv/code/qwen_api_direct_ui_detect.py \
  --model qwen3.5-plus \
  --image-ids all \
  --out /data/LPP/cvpr/few_shot/final/all-elements-fsod-mebv/result/qwen_api_direct_full_v1.json \
  --raw-dir /data/LPP/cvpr/few_shot/sam3-main/best_sam3/all-elements-fsod-mebv/qwen_api_direct_full_v1_raw \
  --timeout 120 \
  --retries 2 \
  --coord-mode norm1000
```

## `all_elements_doubao_qwen_fuse.py`

Reproduces the current best Doubao + Qwen fusion and updates JSON/eval/PKL/zip.

```bash
python /data/LPP/cvpr/few_shot/final/all-elements-fsod-mebv/code/all_elements_doubao_qwen_fuse.py --rebuild-zip
```

## Current Best Result

- `result/all_elements_doubao_qwen_fuse_best.json`
- `result/all_elements_doubao_qwen_fuse_best_eval.json`
- mAP: 0.44070799392120397
- mAP50: 0.7144901363309829
- mAP75: 0.4264263832198452
