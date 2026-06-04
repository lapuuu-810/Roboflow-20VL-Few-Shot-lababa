# soda-bottles-fsod-haga Code Reproduction

## `doubao_soda_direct_api.py`

Doubao direct color/brand soda bottle inference.

```bash
export ARK_API_KEY="你的豆包API_KEY"
python /data/LPP/cvpr/few_shot/final/soda-bottles-fsod-haga/code/doubao_soda_direct_api.py \
  --image-ids 0-225 \
  --out-dir /data/LPP/cvpr/few_shot/sam3-main/best_sam3/soda-bottles-fsod-haga/doubao_soda_direct_0_9 \
  --coord-mode norm1000 \
  --max-side 1600 \
  --score-thr 0.0 \
  --nms-thr 0.45 \
  --vis \
  --resume
```

## `soda_sam3_color_prompt.py`

Reproduces the SAM3 color-prompt baseline.

```bash
python /data/LPP/cvpr/few_shot/final/soda-bottles-fsod-haga/code/soda_sam3_color_prompt.py \
  --skip-sam3 \
  --preserve-work \
  --out-dir /data/LPP/cvpr/few_shot/sam3-main/best_sam3/soda-bottles-fsod-haga/sam3_color_prompt_scale098_shape_filter \
  --scale-x 0.98 \
  --scale-y 0.98 \
  --nms-thr 0.5 \
  --topk-image 50 \
  --min-w-frac 0.05 \
  --min-h-frac 0.1 \
  --min-area-frac 0.006 \
  --vis
```

## `run_sam3_color_prompt.sh`

Shell wrapper for SAM3 color-prompt processing.

```bash
bash /data/LPP/cvpr/few_shot/final/soda-bottles-fsod-haga/code/run_sam3_color_prompt.sh
```

## `soda_doubao_sam3_fuse.py`

Reproduces the current best Doubao + SAM3 fusion and updates JSON/eval/PKL/zip.

```bash
python /data/LPP/cvpr/few_shot/final/soda-bottles-fsod-haga/code/soda_doubao_sam3_fuse.py --rebuild-zip
```

## Current Best Result

- `result/soda_doubao_sam3_fuse_best.json`
- `result/soda_doubao_sam3_fuse_best_eval.json`
- mAP: 0.30115553422799274
- mAP50: 0.7087816923443996
- mAP75: 0.21753031329731698
