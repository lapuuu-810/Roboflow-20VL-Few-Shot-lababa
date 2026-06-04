# Aquarium Combined Final Result

Current final submission files:
- `best.json`
- `best_eval.json`
- `aquarium_doubao_qwen_fuse_best.json`
- `aquarium_doubao_qwen_fuse_best_eval.json`
- `aquarium-combined-fsod-gjvb.pkl`

Best metrics on local `test/_annotations.coco.json`, before category-id remap:
- mAP: 0.4172279529973448
- mAP50: 0.705831244569279
- mAP75: 0.39152212340008996
- predictions: 1147

Important category-id correction:
- Local Roboflow GT includes dummy `none` class at `category_id=0`.
- Server GT removes `none`, so true categories are 0-based: `fish=0, jellyfish=1, penguin=2, puffin=3, shark=4, starfish=5, stingray=6`.
- The final `best.json` and PKL have been remapped from Roboflow IDs using `category_id - 1`.
- The local-eval, pre-remap JSON is saved as `aquarium_doubao_qwen_fuse_best_local_roboflow_ids.json`.

This result fuses the previous Qwen final with the Doubao direct result:
- Previous Qwen final mAP: 0.3945189020415528
- Doubao direct mAP: 0.36980939818799
- Best parameters: `qwen_mul=0.5`, `doubao_mul=1.0`, `nms_thr=0.75`

The same PKL has been written to:

```text
/data/LPP/cvpr/few_shot/final/submission_final_filled/aquarium-combined-fsod-gjvb.pkl
```

The global submission zip is rebuilt at:

```text
/data/LPP/cvpr/few_shot/final/submission_final_filled_new.zip
```
