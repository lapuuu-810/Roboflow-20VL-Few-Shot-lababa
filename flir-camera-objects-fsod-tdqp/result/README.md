# Result

- `best.json`: final detections using server category IDs.
- `sam3_initial_best_local_roboflow_ids.json`: local Roboflow category IDs for local COCO evaluation.
- `best_eval.json`: local evaluation, search space, and selected post-processing parameters.
- `flir-camera-objects-fsod-tdqp.pkl`: submission-format pickle.

Best searched local mAP: `0.374949`. Parameters: `{'thr': 0.0, 'nms': 0.75, 'cap': 50, 'sx': 1.04, 'sy': 1.04}`.
