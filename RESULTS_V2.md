# v2 results

All numbers are from `results_v2.json` and its CUDA-Q reproduction `results_v2_cudaq.json` (10 seeds, exact expectation values) and
`results_v2_shots1024.json` (5 seeds, 1024 shots for training and evaluation), produced by
`quantum_nn.py` with its default config. Hyperparameters were chosen by `tune_qnn.py` using
validation data only (`tuning_v2.json`).

**Reproduced on CUDA-Q.** All 10 seeds were rerun with CUDA-Q on an NVIDIA GPU
(`nvidia` target, FP64; `results_v2_cudaq.json`). Every seed stopped at the same epoch and ended
with trained parameters within 1e-8 of the NumPy run. Test-set predictions match to the 6
decimals stored, so every number below is identical on both backends. The original numbers came
from `--backend numpy`, an exact statevector simulator of the same circuit. On CUDA-Q each seed
took 11–33 minutes to train, against about 20 seconds with NumPy, because at 4 qubits per-call
overhead dominates.

## Setup

- Wisconsin Diagnostic Breast Cancer, 569 patients (212 malignant). Positive class = malignant.
- Fixed stratified test set: 114 patients (42 malignant). Validation: 20% of the remaining data, re-drawn per seed.
- Model: 4 qubits, 3 layers, 6 PCA features (RY on 4 qubits + RZ on 2), no re-uploading,
  24 circuit parameters + 2 classical readout parameters. Adam, lr 0.05, batch 32, class-weighted BCE,
  early stopping on validation loss (patience 10, max 80 epochs).

## Tuning (validation loss, mean of 3 seeds, 36 configurations)

| Rank | Layers | Features | Re-upload | lr | Params | Val loss | Val AUC |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 6 | no | 0.05 | 24 | 0.097 ± 0.026 | 0.991 |
| 2 | 3 | 6 | no | 0.10 | 24 | 0.100 ± 0.032 | 0.991 |
| 3 | 2 | 6 | no | 0.05 | 16 | 0.102 ± 0.035 | 0.991 |
| best with re-upload | 4 | 4 | yes | 0.10 | 32 | 0.129 ± 0.008 | 0.990 |

Using all 6 PCA features helped. Data re-uploading consistently made validation loss worse here
(probably overfitting with ~360 training samples), so it is off by default.

## Test results, threshold 0.5 (10 seeds)

| Model | Accuracy | Sensitivity | Specificity | AUC | Missed cancers (of 42) |
|---|---|---|---|---|---|
| **Hybrid QNN** | **96.6% ± 1.1** | **95.0% ± 2.1** | **97.5% ± 1.9** | **0.996** | **2.1 (1–4)** |
| Logistic regression, same 6 features | 96.7% ± 0.8 | 93.6% ± 1.1 | 98.5% ± 1.7 | 0.996 | 2.7 (2–3) |
| MLP, same 6 features | 96.9% ± 1.0 | 93.6% ± 2.3 | 98.9% ± 0.9 | 0.997 | 2.7 (2–5) |
| RBF SVM, same 6 features | 95.8% ± 0.4 | 91.0% ± 1.0 | 98.6% ± 0.0 | 0.995 | 3.8 (3–4) |
| Random forest, same 6 features | 94.5% ± 0.6 | 91.0% ± 1.0 | 96.5% ± 0.7 | 0.992 | 3.8 (3–4) |
| Logistic regression, all 30 features | 97.9% ± 0.6 | 96.2% ± 1.2 | 98.9% ± 0.9 | 0.996 | 1.6 (1–2) |
| QNN + logistic regression (6 features), soft vote | 96.8% ± 0.6 | 94.8% ± 1.0 | 97.9% ± 1.0 | 0.997 | 2.2 (2–3) |

± is the standard deviation across seeds on one fixed test set. Uncertainty from the test set's
size is larger: the QNN's bootstrap 95% CI over test patients is **accuracy 93.7–99.1%,
sensitivity 88.6–99.8%, AUC 0.99–1.00**.

With 1024 measurement shots (5 seeds) the QNN gets 96.3% ± 0.4 accuracy, 94.8% ± 1.1
sensitivity and 0.996 AUC, so the result holds up under realistic shot noise.

## What this does and doesn't show

- The hybrid QNN is a competitive classifier: about 96–97% accuracy with roughly 2 missed cancers in 42,
  using 26 trainable parameters.
- It **matches but does not beat** classical models given the same 6 features. Logistic regression on all
  30 features is slightly better than everything else. There is no quantum advantage on this dataset,
  which is the expected outcome for a small tabular problem.
- Soft-voting the QNN with logistic regression did not improve on either model alone.
- The validation-tuned "95% sensitivity" threshold did **not** transfer reliably to test
  (QNN 92.6% ± 8.0 sensitivity). The validation split has only ~34 malignant cases, too few to set a
  threshold that precisely. The 0.5 threshold with a class-weighted loss was better. Results at both
  thresholds are in the JSON.
- The learned readout weight `w` ends up large (8–15 across seeds), meaning ⟨Z₀⟩ varies over a narrow range and the
  classical scale stretches it. It's a legitimate hybrid design, but the quantum circuit is acting as a
  learned feature map feeding a one-dimensional logistic readout.
- The test set has 114 patients. Differences of about 1 point between models are within noise.
