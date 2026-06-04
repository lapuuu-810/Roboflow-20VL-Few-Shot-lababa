# Result

- `best.json`: final detections using server category IDs.
- `sam3_initial_best_local_roboflow_ids.json`: local Roboflow category IDs for local COCO evaluation.
- `best_eval.json`: local evaluation, search space, and selected post-processing parameters.
- `gwhd2021-fsod-atsv.pkl`: submission-format pickle.

Best searched local mAP: `0.239591`. Parameters: `{'thr': 0.0, 'nms': 0.55, 'cap': 100, 'sx': 1.1, 'sy': 1.1}`.
