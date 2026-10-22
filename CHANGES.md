# Changes in v2

v1 is preserved unchanged in `legacy/` (code) and `legacy/reports/` (the old write-ups).
The old reports' numbers were produced by the issues listed below and should not be cited.

## Correctness bugs fixed

| # | v1 behaviour | Effect | v2 |
|---|---|---|---|
| 1 | Labels left as sklearn's `0 = malignant, 1 = benign`, and every `recall_score` used the default `pos_label=1` | "100% recall / zero false negatives / zero missed cancers" actually meant zero *benign* cases flagged. Missed cancers were in the cell printed as "False Positives". | Labels re-coded so `1 = malignant`. Sensitivity, missed cancers and false alarms are reported by name. |
| 2 | SMOTE-style augmentation ran **before** the train/test split | Test set contained synthetic points interpolated from training patients | No synthetic data; class imbalance handled with a weighted loss |
| 3 | StandardScaler and PCA fit on the full dataset before splitting | Test statistics leaked into preprocessing | Every transform is fit on the training split only |
| 4 | Test features mapped to [0, π] with the **test set's own** min/max (one global min/max over all features) | Train and test features on different scales; low-variance PCA components squashed | Per-feature MinMax fit on training data, applied to val/test, clipped to [0, π] |
| 5 | Best epoch and early stopping chosen by **test** accuracy | Reported test score optimistically biased | Separate validation split for early stopping, threshold choice and tuning. The test set is scored once per run. |
| 6 | Gradient = parameter-shift applied to the BCE **loss** | Parameter shift is exact for expectation values, not for a nonlinear loss, so the gradient was biased | Parameter shift for d⟨Z⟩/dθ, chain rule through the loss. Checked against finite differences (relative error ~1e-9). |
| 7 | Mini-batches never shuffled; 8 fixed batches per epoch | Every epoch trained on the same first 256 of 556 samples | Full shuffled pass over the training set every epoch |
| 8 | 6 PCA features computed, but the circuit only encoded the first 4 | Two features silently ignored | Feature count must fit the encoding (RY on each qubit, then RZ); the tuner chooses 4 or 6 |
| 9 | No random seeds | Results varied run to run | NumPy, scikit-learn and CUDA-Q (`cudaq.set_random_seed`) all seeded; multi-seed runs report mean ± std |
| 10 | `hyperparameter_tuning.py` used `np.random.randn` as the "gradient" and `np.random.binomial(1, 0.7)` as the "predictions" | The "10,800-configuration search" never ran a circuit; its "best config" was chosen from random numbers | `tune_qnn.py` trains real models and ranks them on validation loss |
| 11 | Two `nvidia-smi` subprocesses launched on every mini-batch for the progress bar | Slowed training down | Removed from the training loop |

## Modelling improvements

- **Hybrid readout:** `p(malignant) = sigmoid(w·⟨Z₀⟩ + b)` with trainable classical `w, b`, instead of the fixed `(⟨Z⟩+1)/2`. The circuit no longer has to push ⟨Z⟩ to ±1 to express confidence.
- **Data re-uploading** is available (`--reupload`), but validation tuning found it worse on this dataset, so it is off by default.
- **Class-weighted loss** (balanced) instead of synthetic oversampling.
- **Exact expectation values** in simulation by default (`--train-shots 0`). Shot noise is still available with `--train-shots/--eval-shots N`.
- **Sensitivity-targeted threshold:** chosen on validation to reach `--target-sensitivity` (default 0.95), then applied unchanged to test. Results are reported at both 0.5 and the tuned threshold.
- **Fair baselines:** logistic regression, calibrated RBF SVM, random forest and MLP on the *same* splits, on both the quantum features and all 30 features, plus a QNN + logistic-regression soft-vote ensemble. All except the MLP use balanced class weights (sklearn's MLP doesn't support them).
- **Honest uncertainty:** multi-seed mean ± sample std, plus a bootstrap 95% CI over test patients (the across-seed spread alone understates uncertainty on a 114-patient test set).
- **Tuned defaults:** 4 qubits × 3 layers, 6 PCA features, no re-uploading, lr 0.05, chosen by `tune_qnn.py` on validation loss (`tuning_v2.json`).

## Engineering

- One CLI (`argparse`); config in a frozen dataclass.
- Swappable backends: `--backend cudaq` (GPU `nvidia` target, falls back to `qpp-cpu`) or `--backend numpy` (exact statevector simulator of the identical circuit). `--check-backend` verifies they agree (uses the FP64 `nvidia` option, since the GPU default is FP32).
- Results written to JSON (per-run metrics, training curves, final parameters, summary).
- No `plt.show()`, so the script doesn't stall in WSL or over SSH.

## Added after v2: experiment options (defaults unchanged)

- `--entangler ring|brick`: the CNOT pattern. `ring` (default) is the v2 circuit; `brick` is a
  nearest-neighbour brickwork that avoids the barren plateau found in `EXPERIMENTS.md`. Both are
  implemented in the CUDA-Q kernel and every simulator, and `--check-backend` tests both.
- `--init uniform|small`: parameter initialisation (`small` = N(0, 0.1²), near identity). It didn't
  help with the barren plateau, so `uniform` stays the default.
- `experiments/`: noise models (CUDA-Q Kraus channels + density-matrix simulator), the scaling
  study, an adjoint-gradient simulator for fast CPU runs, and plotting.
- `tests/test_simulators.py`: 75 checks of every simulator against dense-matrix references.
