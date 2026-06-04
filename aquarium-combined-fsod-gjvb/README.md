# aquarium-combined-fsod-gjvb

Final submission result: `result/best.json`
Final eval reference: `result/best_eval.json`
Submission PKL: `result/aquarium-combined-fsod-gjvb.pkl`

Metrics, computed before category-id remap on the local Roboflow-style GT:
- mAP: 0.4172279529973448
- mAP50: 0.705831244569279
- mAP75: 0.39152212340008996
- predictions: 1147

Category-id note:
- The Roboflow COCO export contains a dummy `none` class at `category_id=0`, so local annotations use `fish=1, jellyfish=2, ...`.
- The evaluation server removes the dummy class and uses `fish=0, jellyfish=1, ...`.
- Final submission files in this directory are remapped with `server_category_id = roboflow_category_id - 1`.
- The pre-remap local-eval JSON is preserved as `result/aquarium_doubao_qwen_fuse_best_local_roboflow_ids.json`.

Baselines:
- Original final Qwen result: mAP 0.3945189020415528, mAP50 0.6504319973730669, mAP75 0.38236417286963886
- Doubao direct result: mAP 0.36980939818799, mAP50 0.62485189353376, mAP75 0.3469655134189594

Best fusion:
- Qwen source: `result/qwen_final_full_jelly_top8_drop_plus_local_strict_v1.json`
- Doubao source: `/data/LPP/cvpr/few_shot/sam3-main/best_sam3/aquarium-combined-fsod-gjvb/doubao_aquarium_direct_0_9/predictions.json`
- Score weights: qwen `0.5`, doubao `1.0`
- Class-wise NMS: `0.75`

Code:
- `code/aquarium_doubao_qwen_fuse.py`: reproduces the fusion JSON/eval/PKL. For server submission, apply the final category-id remap described above.
- `code/doubao_aquarium_direct_api.py`: Doubao direct inference script used to produce the added prediction source.
- Existing Qwen/local filter scripts are kept for reproducing the previous final baseline.
