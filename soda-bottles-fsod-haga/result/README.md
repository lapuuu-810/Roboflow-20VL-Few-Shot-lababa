# Soda Bottles Result

Best result: `soda_doubao_sam3_fuse_best.json`
Best eval: `soda_doubao_sam3_fuse_best_eval.json`
PKL: `soda-bottles-fsod-haga.pkl`

Sources:
- Doubao: `/data/LPP/cvpr/few_shot/sam3-main/best_sam3/soda-bottles-fsod-haga/doubao_soda_direct_0_9/predictions.json`
- Existing final SAM3: `/data/LPP/cvpr/few_shot/final/soda-bottles-fsod-haga/result/soda_sam3_color_prompt_scale098_shape_filter.json`

Metrics:
- mAP: 0.29053820762491916
- mAP50: 0.7067611510550772
- mAP75: 0.19813326160543998
- predictions: 6396

Baselines:
- SAM3 raw: 0.27221210619077024
- Doubao raw: 0.2777564065822576

Reproduce:
```bash
python /data/LPP/cvpr/few_shot/final/soda-bottles-fsod-haga/code/soda_doubao_sam3_fuse.py --rebuild-zip
```
