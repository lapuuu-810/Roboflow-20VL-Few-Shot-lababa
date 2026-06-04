# aquarium-combined-fsod-gjvb Code Reproduction

Run commands from any working directory. API scripts require their provider keys; the fusion script is fully offline.

## Final Fusion

Reproduce the saved best result and rebuild the global submission zip:

```bash
python /data/LPP/cvpr/few_shot/final/aquarium-combined-fsod-gjvb/code/aquarium_doubao_qwen_fuse.py \
  --qwen /data/LPP/cvpr/few_shot/final/aquarium-combined-fsod-gjvb/result/qwen_final_full_jelly_top8_drop_plus_local_strict_v1.json \
  --doubao /data/LPP/cvpr/few_shot/sam3-main/best_sam3/aquarium-combined-fsod-gjvb/doubao_aquarium_direct_0_9/predictions.json \
  --gt /data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data/aquarium-combined-fsod-gjvb/test/_annotations.coco.json \
  --qwen-mul 0.5 \
  --doubao-mul 1.0 \
  --nms-thr 0.75 \
  --rebuild-zip
```

Expected metrics:
- mAP: 0.4172279529973448
- mAP50: 0.705831244569279
- mAP75: 0.39152212340008996

## Doubao Direct Inference

Small batch test command:

```bash
export ARK_API_KEY="你的Doubao_API_KEY"
python /data/LPP/cvpr/few_shot/final/aquarium-combined-fsod-gjvb/code/doubao_aquarium_direct_api.py \
  --gt /data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data/aquarium-combined-fsod-gjvb/test/_annotations.coco.json \
  --image-dir /data/LPP/cvpr/few_shot/data/foundational_fsod-fsod_rf20vl/data/aquarium-combined-fsod-gjvb/test \
  --image-ids 0-9 \
  --out /data/LPP/cvpr/few_shot/sam3-main/best_sam3/aquarium-combined-fsod-gjvb/doubao_aquarium_direct_0_9/predictions.json \
  --raw-dir /data/LPP/cvpr/few_shot/sam3-main/best_sam3/aquarium-combined-fsod-gjvb/doubao_aquarium_direct_0_9/raw \
  --coord-mode norm1000 \
  --evaluate \
  --visualize \
  --resume
```

Full run uses `--image-ids all` with the same output directory.

## Previous Qwen Baseline

The previous final baseline is kept in `result/qwen_final_full_jelly_top8_drop_plus_local_strict_v1.json` and can be reproduced by the existing Qwen/local filter scripts in this folder. Its eval is mAP `0.3945189020415528`.
