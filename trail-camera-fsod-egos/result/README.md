# Result

- `best.json`: final detections using server category IDs.
- `sam3_initial_best_local_roboflow_ids.json`: local Roboflow category IDs for local COCO evaluation.
- `best_eval.json`: local evaluation, search space, and selected post-processing parameters.
- `trail-camera-fsod-egos.pkl`: submission-format pickle.

Best searched local mAP: `0.626806`. Parameters: `{'thr': 0.0, 'nms': 0.65, 'cap': 50, 'sx': 1.1, 'sy': 1.1}`.
