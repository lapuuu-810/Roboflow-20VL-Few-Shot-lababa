# recode-waste-czvmg-fsod-yxsw Code Reproduction

## `qwen_waste_direct_api.py`

```bash
export DASHSCOPE_API_KEY="你的Qwen_API_KEY"
python /data/LPP/cvpr/few_shot/final/recode-waste-czvmg-fsod-yxsw/code/qwen_waste_direct_api.py --resume
```

Qwen API inference helper; credentials required.

## `qwen_waste_fuse_tile_filtered.py`

```bash
python /data/LPP/cvpr/few_shot/final/recode-waste-czvmg-fsod-yxsw/code/qwen_waste_fuse_tile_filtered.py
```

Local pipeline, tiling, filtering, or fusion helper.

## `qwen_waste_tile_smalltest.py`

```bash
python /data/LPP/cvpr/few_shot/final/recode-waste-czvmg-fsod-yxsw/code/qwen_waste_tile_smalltest.py
```

Local pipeline, tiling, filtering, or fusion helper.

## `recode_waste_doubao_qwen_fuse.py`

```bash
python /data/LPP/cvpr/few_shot/final/recode-waste-czvmg-fsod-yxsw/code/recode_waste_doubao_qwen_fuse.py --rebuild-zip
```

Reproduces the current best Doubao + existing-final-Qwen fusion and writes result JSON/eval/PKL/zip.

## Current Best Result

- `result/recode_doubao_qwen_fuse_best.json`
- `result/recode_doubao_qwen_fuse_best_eval.json`
- mAP: 0.457203077865662
- mAP50: 0.5548593886677572
- mAP75: 0.4878751742394232
