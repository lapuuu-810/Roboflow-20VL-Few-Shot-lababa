# gwhd2021-fsod-atsv

This dataset was filled from saved SAM3 initial predictions and optimized with post-processing search over score threshold, class-wise NMS, per-image cap, and center-fixed box scaling.

Best local mAP: `0.239591`. Submitted category IDs are remapped from local Roboflow IDs by subtracting one because the local export contains a placeholder class at `category_id=0`.
