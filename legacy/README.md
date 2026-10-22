# Legacy (v1)

The original scripts and write-ups, kept unchanged for reference. **Don't cite the numbers in
`reports/`.** They were affected by the issues listed in `../CHANGES.md`, most importantly:

- Recall was computed for the benign class (sklearn's `pos_label=1`), so "100% recall / zero missed
  cancers" meant zero benign cases flagged.
- Augmentation and preprocessing ran before the train/test split, and model selection used the test set.
- `hyperparameter_tuning.py` used random numbers in place of gradients and predictions; no circuit
  was ever trained during the "search".
- Several reports describe configurations that don't match the code, and some reported metric
  combinations (e.g. 78.68% accuracy + 100% recall + 68.57% precision) can't come from any single
  confusion matrix.

The scripts here import each other (`ensemble_model.py` imports `quantum_nn.py` from this folder),
so run them from inside `legacy/` if you need to reproduce v1 behaviour.

`tools/` holds the v1 debugging scripts (GPU checks, a segfault repro), also unused by v2.
