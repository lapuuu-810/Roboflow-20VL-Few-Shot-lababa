# Defect Detection Final Result

Current final files:
- `defect_doubao_qwen_fuse_best.json`
- `defect_doubao_qwen_fuse_best_eval.json`
- `defect-detection-yjplx-fxobh-fsod-amdi.pkl`

Metrics:
- mAP: 0.3050309094446834
- mAP50: 0.5887439424804463
- mAP75: 0.29977797515314636

This result fuses `qwen_defect_full_best_current_v12.json` with Doubao v2 raw predictions from:

```text
/data/LPP/cvpr/few_shot/sam3-main/best_sam3/defect-detection-yjplx-fxobh-fsod-amdi/doubao_defect_direct_v2_0_9_raw
```

The same PKL has been written to the global submission folder and `submission_final_filled_new.zip` has been rebuilt.
