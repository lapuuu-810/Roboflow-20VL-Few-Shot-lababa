# All Elements Result

Best result: `all_elements_doubao_qwen_fuse_best.json`
Best eval: `all_elements_doubao_qwen_fuse_best_eval.json`
PKL: `all-elements-fsod-mebv.pkl`

Sources:
- Doubao: `/data/LPP/cvpr/few_shot/sam3-main/best_sam3/all-elements-fsod-mebv/doubao_all_elements_direct_0_9/predictions.json`
- Existing final Qwen: `/data/LPP/cvpr/few_shot/final/all-elements-fsod-mebv/result/qwen_api_direct_full_v1.json`

Metrics:
- mAP: 0.44070799392120397
- mAP50: 0.7144901363309829
- mAP75: 0.4264263832198452
- predictions: 1268

Baselines:
- Qwen raw: 0.3934398820601464
- Doubao raw: 0.4025573582177416

Reproduce:
```bash
python /data/LPP/cvpr/few_shot/final/all-elements-fsod-mebv/code/all_elements_doubao_qwen_fuse.py --rebuild-zip
```
