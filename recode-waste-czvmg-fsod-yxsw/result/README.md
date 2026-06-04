# Recode Waste Result

Best result: `recode_doubao_qwen_fuse_best.json`
Best eval: `recode_doubao_qwen_fuse_best_eval.json`
PKL: `recode-waste-czvmg-fsod-yxsw.pkl`

Sources:
- Doubao: `/data/LPP/cvpr/few_shot/sam3-main/best_sam3/recode-waste-czvmg-fsod-yxsw/doubao_direct_bbox_smalltest_v1/predictions.json`
- Existing final Qwen: `/data/LPP/cvpr/few_shot/final/recode-waste-czvmg-fsod-yxsw/result/qwen3vl_full_tile_filtered_fuse.json`

Metrics:
- mAP: 0.457203077865662
- mAP50: 0.5548593886677572
- mAP75: 0.4878751742394232
- predictions: 9832

Baselines:
- Qwen raw: 0.3697908979179474
- Doubao raw: 0.39952445383084295

Reproduce:
```bash
python /data/LPP/cvpr/few_shot/final/recode-waste-czvmg-fsod-yxsw/code/recode_waste_doubao_qwen_fuse.py --rebuild-zip
```
