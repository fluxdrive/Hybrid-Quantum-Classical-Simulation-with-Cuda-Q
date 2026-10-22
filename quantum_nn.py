"""
Hybrid quantum-classical classifier for the Wisconsin Breast Cancer dataset (CUDA-Q)
====================================================================================

v2. See CHANGES.md for what changed from v1 and why.

Pipeline
--------
1. Data: labels are re-coded so 1 = malignant (the positive class), so "recall" means
   cancer sensitivity and a false negative means a missed cancer.
   Fixed stratified test split; a validation split is carved out of the training data.
   StandardScaler -> PCA -> per-feature min/max to [0, pi] are fit on the training split only.
2. Circuit: angle encoding (RY for the first n_qubits features, RZ for the rest), optionally
   re-uploaded before every variational layer; each layer is RY+RZ on every qubit followed by
   a CNOT ring. Readout is <Z_0>.
3. Hybrid readout: p(malignant) = sigmoid(w * <Z_0> + b) with trainable classical w, b.
4. Training: class-weighted binary cross-entropy, Adam, shuffled mini-batches over the full
   training set, exact gradients (parameter-shift for d<Z>/d(theta), chain rule through the
   loss), early stopping on validation loss with best-weights restore.
5. Decision threshold: chosen on the validation set to reach a target cancer sensitivity,
   then applied unchanged to the test set. The test set is used exactly once per run.
6. Comparison: logistic regression, RBF SVM, random forest and MLP on the identical splits,
   on both the quantum model's features and the full 30 features, plus a QNN + classical
   soft-vote ensemble.

Backends
--------
--backend cudaq  : CUDA-Q (`nvidia` GPU target by default, falls back to `qpp-cpu`)
--backend numpy  : exact NumPy statevector simulator of the same circuit (no CUDA-Q needed)
--check-backend  : compare the two on random inputs (requires CUDA-Q)

Examples
--------
python quantum_nn.py --backend cudaq --seeds 0 1 2 3 4
python quantum_nn.py --backend numpy --seeds 0 1 2 3 4 5 6 7 8 9 --out results_v2.json
python quantum_nn.py --check-backend
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass, replace

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.calibration import CalibratedClassifierCV
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.svm import SVC

try:
    import cudaq

    HAVE_CUDAQ = True
except ImportError:  # the NumPy backend still works
    cudaq = None
    HAVE_CUDAQ = False


# ==============================================================================
# CONFIGURATION
# ==============================================================================

@dataclass(frozen=True)
class Config:
    # circuit
    n_qubits: int = 4
    n_layers: int = 3
    n_features: int = 6          # PCA components; must be <= 2 * n_qubits
    reupload: bool = False       # re-encode features before every layer (tuning found it worse)
    entangler: str = "ring"      # "ring": CNOT chain + wrap-around; "brick": nearest-neighbour brickwork
    # optimisation
    epochs: int = 80
    batch_size: int = 32
    lr: float = 0.05
    patience: int = 10           # epochs without validation-loss improvement
    class_weight: str = "balanced"   # "balanced" or "none"
    init: str = "uniform"        # "uniform": theta ~ U(0, 2pi); "small": theta ~ N(0, init_scale^2)
    init_scale: float = 0.1
    # data
    test_fraction: float = 0.2
    val_fraction: float = 0.2    # fraction of the non-test data
    split_seed: int = 42         # fixes the test split across all runs
    # measurement
    train_shots: int = 0         # 0 = exact expectation values
    eval_shots: int = 0
    # decision rule
    target_sensitivity: float = 0.95
    # backend
    backend: str = "numpy"
    cudaq_target: str = "nvidia"
    cudaq_option: str = ""       # e.g. "fp64" for the nvidia target

    @property
    def n_params(self) -> int:
        return self.n_qubits * 2 * self.n_layers

    def validate(self) -> None:
        if not 1 <= self.n_features <= 2 * self.n_qubits:
            raise ValueError("n_features must be between 1 and 2 * n_qubits")
        if self.class_weight not in ("balanced", "none"):
            raise ValueError("class_weight must be 'balanced' or 'none'")
        if self.entangler not in ("ring", "brick"):
            raise ValueError("entangler must be 'ring' or 'brick'")
        if self.init not in ("uniform", "small"):
            raise ValueError("init must be 'uniform' or 'small'")
        if not 0.0 < self.target_sensitivity <= 1.0:
            raise ValueError("target_sensitivity must be in (0, 1]")


# ==============================================================================
# DATA
# ==============================================================================

@dataclass
class Split:
    Xq_train: np.ndarray   # quantum features in [0, pi]
    Xq_val: np.ndarray
    Xq_test: np.ndarray
    Xf_train: np.ndarray   # all 30 features, standardised (for classical baselines)
    Xf_val: np.ndarray
    Xf_test: np.ndarray
    y_train: np.ndarray    # 1 = malignant
    y_val: np.ndarray
    y_test: np.ndarray
    pca_variance: float


def load_split(cfg: Config, seed: int) -> Split:
    data = load_breast_cancer()
    X = data.data
    y = (data.target == 0).astype(int)   # sklearn: 0 = malignant. Re-code so 1 = malignant.

    X_rest, X_test, y_rest, y_test = train_test_split(
        X, y, test_size=cfg.test_fraction, random_state=cfg.split_seed, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_rest, y_rest, test_size=cfg.val_fraction, random_state=seed, stratify=y_rest)

    std = StandardScaler().fit(X_train)
    Xf = [std.transform(a) for a in (X_train, X_val, X_test)]

    pca = PCA(n_components=cfg.n_features, random_state=0).fit(Xf[0])
    angle = MinMaxScaler(feature_range=(0.0, np.pi)).fit(pca.transform(Xf[0]))
    Xq = [np.clip(angle.transform(pca.transform(a)), 0.0, np.pi) for a in Xf]

    return Split(*Xq, *Xf, y_train, y_val, y_test, float(pca.explained_variance_ratio_.sum()))


# ==============================================================================
# CIRCUIT BACKENDS
# ==============================================================================
# Both backends implement exactly this circuit:
#
#   for layer in range(n_layers):
#       if layer == 0 or reupload:
#           RY(x[i]) on qubit i            for i <  n_qubits
#           RZ(x[i]) on qubit i - n_qubits for i >= n_qubits
#       for q in qubits: RY(theta[k]) q; RZ(theta[k+1]) q; k += 2
#       entangler "ring":  CNOT(q, q+1) for q = 0 .. n_qubits-2, then CNOT(n_qubits-1, 0)
#       entangler "brick": CNOT on pairs (0,1),(2,3),... then (1,2),(3,4),...  (no wrap-around)
#   return <Z_0>
#
# expectations(X, thetas, shots) -> array of shape (len(thetas), len(X))

if HAVE_CUDAQ:
    @cudaq.kernel
    def qnn_kernel(feats: list[float], theta: list[float],
                   n_qubits: int, n_layers: int, reupload: int, brick: int):
        q = cudaq.qvector(n_qubits)
        for layer in range(n_layers):
            # true when layer == 0 or reupload == 1 (integer form keeps the kernel simple)
            if layer * (1 - reupload) == 0:
                for i in range(len(feats)):
                    if i < n_qubits:
                        ry(feats[i], q[i])
                    else:
                        rz(feats[i], q[i - n_qubits])
            for j in range(n_qubits):
                ry(theta[2 * (layer * n_qubits + j)], q[j])
                rz(theta[2 * (layer * n_qubits + j) + 1], q[j])
            if brick == 1:
                for j in range(0, n_qubits - 1, 2):
                    cx(q[j], q[j + 1])
                for j in range(1, n_qubits - 1, 2):
                    cx(q[j], q[j + 1])
            else:
                for j in range(n_qubits - 1):
                    cx(q[j], q[j + 1])
                if n_qubits > 1:
                    cx(q[n_qubits - 1], q[0])


class CudaqBackend:
    name = "cudaq"

    def __init__(self, cfg: Config, seed: int):
        if not HAVE_CUDAQ:
            raise RuntimeError("CUDA-Q is not installed; use --backend numpy")
        self.cfg = cfg
        try:
            if cfg.cudaq_option:
                cudaq.set_target(cfg.cudaq_target, option=cfg.cudaq_option)
            else:
                cudaq.set_target(cfg.cudaq_target)
        except Exception as e:  # e.g. no GPU
            print(f"  [cudaq] target '{cfg.cudaq_target}' unavailable ({e}); using qpp-cpu")
            cudaq.set_target("qpp-cpu")
        cudaq.set_random_seed(seed)
        self.hamiltonian = cudaq.spin.z(0)

    def expectations(self, X: np.ndarray, thetas: np.ndarray, shots: int = 0) -> np.ndarray:
        c = self.cfg
        kwargs = {"shots_count": shots} if shots > 0 else {}
        futures = [
            cudaq.observe_async(qnn_kernel, self.hamiltonian, x.tolist(), th.tolist(),
                                c.n_qubits, c.n_layers, int(c.reupload), int(c.entangler == "brick"),
                                **kwargs)
            for th in thetas for x in X
        ]
        vals = np.array([f.get().expectation() for f in futures], dtype=float)
        return vals.reshape(len(thetas), len(X))


class NumpyBackend:
    """Exact statevector simulation. State shape: (P, S, 2, ..., 2), axis 2+q is qubit q."""
    name = "numpy"

    def __init__(self, cfg: Config, seed: int):
        self.cfg = cfg
        self.rng = np.random.default_rng(10_000 + seed)

    @staticmethod
    def _ry(t):
        c, s = np.cos(t / 2), np.sin(t / 2)
        return np.stack([np.stack([c, -s], -1), np.stack([s, c], -1)], -2).astype(complex)

    @staticmethod
    def _rz(t):
        a, b = np.exp(-0.5j * t), np.exp(0.5j * t)
        z = np.zeros_like(a)
        return np.stack([np.stack([a, z], -1), np.stack([z, b], -1)], -2)

    def _apply_1q(self, st, M, q):
        # M: (P, S, 2, 2) or broadcastable to it
        nq = self.cfg.n_qubits
        st = np.moveaxis(st, 2 + q, -1)
        M = np.broadcast_to(M, st.shape[:2] + (2, 2))
        st = np.matmul(M.reshape(st.shape[:2] + (1,) * (nq - 1) + (2, 2)), st[..., None])[..., 0]
        return np.moveaxis(st, -1, 2 + q)

    @staticmethod
    def _apply_cx(st, c, t):
        st = st.copy()
        idx = [slice(None)] * st.ndim
        idx[2 + c] = 1
        sub = st[tuple(idx)]
        st[tuple(idx)] = np.flip(sub, axis=(2 + t) - (1 if t > c else 0))
        return st

    def exact(self, X: np.ndarray, thetas: np.ndarray) -> np.ndarray:
        c = self.cfg
        nq = c.n_qubits
        P, S = thetas.shape[0], X.shape[0]
        st = np.zeros((P, S) + (2,) * nq, dtype=complex)
        st[(slice(None), slice(None)) + (0,) * nq] = 1.0
        enc_ry = [self._ry(X[:, i])[None] for i in range(X.shape[1])]
        enc_rz = [self._rz(X[:, i])[None] for i in range(X.shape[1])]
        k = 0
        for layer in range(c.n_layers):
            if layer == 0 or c.reupload:
                for i in range(X.shape[1]):
                    if i < nq:
                        st = self._apply_1q(st, enc_ry[i], i)
                    else:
                        st = self._apply_1q(st, enc_rz[i], i - nq)
            for j in range(nq):
                st = self._apply_1q(st, self._ry(thetas[:, k])[:, None], j)
                st = self._apply_1q(st, self._rz(thetas[:, k + 1])[:, None], j)
                k += 2
            for a, b in cnot_pairs(nq, c.entangler):
                st = self._apply_cx(st, a, b)
        p0 = (np.abs(st) ** 2)[:, :, 0].reshape(P, S, -1).sum(-1)
        return 2.0 * p0 - 1.0

    def expectations(self, X: np.ndarray, thetas: np.ndarray, shots: int = 0) -> np.ndarray:
        ev = self.exact(X, thetas)
        if shots <= 0:
            return ev
        p0 = np.clip((ev + 1.0) / 2.0, 0.0, 1.0)
        return (2.0 * self.rng.binomial(shots, p0) - shots) / shots


def cnot_pairs(n_qubits: int, entangler: str) -> list[tuple[int, int]]:
    """(control, target) pairs of one entangling layer, in application order."""
    if entangler == "brick":
        return ([(j, j + 1) for j in range(0, n_qubits - 1, 2)] +
                [(j, j + 1) for j in range(1, n_qubits - 1, 2)])
    return [(j, j + 1) for j in range(n_qubits - 1)] + ([(n_qubits - 1, 0)] if n_qubits > 1 else [])


def make_backend(cfg: Config, seed: int):
    return CudaqBackend(cfg, seed) if cfg.backend == "cudaq" else NumpyBackend(cfg, seed)


# ==============================================================================
# MODEL
# ==============================================================================

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


class HybridQNN:
    """Variational circuit + trainable classical readout: p = sigmoid(w * <Z_0> + b)."""

    def __init__(self, cfg: Config, backend, rng: np.random.Generator):
        self.cfg, self.backend = cfg, backend
        if cfg.init == "small":   # near-identity start; mitigates barren plateaus in wide circuits
            self.theta = rng.normal(0.0, cfg.init_scale, cfg.n_params)
        else:
            self.theta = rng.uniform(0.0, 2.0 * np.pi, cfg.n_params)
        self.w, self.b = 1.0, 0.0

    def get(self):
        return self.theta.copy(), self.w, self.b

    def set(self, state):
        self.theta, self.w, self.b = state[0].copy(), state[1], state[2]

    def predict_proba(self, X: np.ndarray, shots: int | None = None) -> np.ndarray:
        shots = self.cfg.eval_shots if shots is None else shots
        ev = self.backend.expectations(X, self.theta[None, :], shots)[0]
        return sigmoid(self.w * ev + self.b)

    def loss_and_grad(self, X: np.ndarray, y: np.ndarray, cw: np.ndarray):
        """Weighted BCE and its exact gradient w.r.t. (theta, w, b)."""
        K = self.cfg.n_params
        thetas = np.repeat(self.theta[None, :], 2 * K + 1, axis=0)
        for k in range(K):
            thetas[1 + 2 * k, k] += np.pi / 2
            thetas[2 + 2 * k, k] -= np.pi / 2
        E = self.backend.expectations(X, thetas, self.cfg.train_shots)   # (2K+1, S)
        e0 = E[0]
        dE = (E[1::2] - E[2::2]) / 2.0                                  # (K, S), parameter shift

        p = np.clip(sigmoid(self.w * e0 + self.b), 1e-7, 1 - 1e-7)
        sw = cw / cw.sum()
        loss = float(-(sw * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum())
        dlogit = sw * (p - y)                                            # (S,)
        g_theta = dE @ (dlogit * self.w)
        g_w = float(dlogit @ e0)
        g_b = float(dlogit.sum())
        return loss, g_theta, g_w, g_b


class Adam:
    def __init__(self, n, lr, b1=0.9, b2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, b1, b2, eps
        self.m, self.v, self.t = np.zeros(n), np.zeros(n), 0

    def step(self, params, grad):
        self.t += 1
        self.m = self.b1 * self.m + (1 - self.b1) * grad
        self.v = self.b2 * self.v + (1 - self.b2) * grad ** 2
        mh = self.m / (1 - self.b1 ** self.t)
        vh = self.v / (1 - self.b2 ** self.t)
        return params - self.lr * mh / (np.sqrt(vh) + self.eps)


def class_weights(y: np.ndarray, mode: str) -> np.ndarray:
    if mode == "none":
        return np.ones_like(y, dtype=float)
    n, pos = len(y), y.sum()
    return np.where(y == 1, n / (2.0 * pos), n / (2.0 * (n - pos)))


def weighted_bce(p, y, cw):
    p = np.clip(p, 1e-7, 1 - 1e-7)
    sw = cw / cw.sum()
    return float(-(sw * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum())


def train(model: HybridQNN, split: Split, rng: np.random.Generator, verbose: bool = True):
    cfg = model.cfg
    Xtr, ytr, Xva, yva = split.Xq_train, split.y_train, split.Xq_val, split.y_val
    cw_tr = class_weights(ytr, cfg.class_weight)
    cw_va = class_weights(yva, cfg.class_weight)
    opt = Adam(cfg.n_params + 2, cfg.lr)

    best_val, best_state, best_epoch, bad = np.inf, model.get(), -1, 0
    history = []
    t0 = time.time()
    for epoch in range(cfg.epochs):
        order = rng.permutation(len(ytr))
        losses = []
        for s in range(0, len(order), cfg.batch_size):
            idx = order[s:s + cfg.batch_size]
            loss, g_t, g_w, g_b = model.loss_and_grad(Xtr[idx], ytr[idx], cw_tr[idx])
            flat = np.concatenate([model.theta, [model.w, model.b]])
            flat = opt.step(flat, np.concatenate([g_t, [g_w, g_b]]))
            model.theta, model.w, model.b = flat[:-2], flat[-2], flat[-1]
            losses.append(loss)

        p_val = model.predict_proba(Xva)
        val_loss = weighted_bce(p_val, yva, cw_va)
        val_auc = roc_auc_score(yva, p_val)
        history.append({"epoch": epoch, "train_loss": round(float(np.mean(losses)), 4),
                        "val_loss": round(val_loss, 4), "val_auc": round(float(val_auc), 4)})
        if verbose:
            print(f"    epoch {epoch:3d} | train loss {np.mean(losses):.4f} | "
                  f"val loss {val_loss:.4f} | val AUC {val_auc:.4f}")
        if val_loss < best_val - 1e-4:
            best_val, best_state, best_epoch, bad = val_loss, model.get(), epoch, 0
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    model.set(best_state)
    return {"best_epoch": best_epoch, "best_val_loss": round(float(best_val), 4),
            "epochs_run": len(history), "train_seconds": round(time.time() - t0, 1),
            "history": history}


# ==============================================================================
# EVALUATION
# ==============================================================================

def pick_threshold(p_val: np.ndarray, y_val: np.ndarray, target_sensitivity: float) -> float:
    """Highest threshold whose validation sensitivity is >= target (max specificity under the
    sensitivity constraint)."""
    pos = np.sort(p_val[y_val == 1])
    k = int(np.floor((1.0 - target_sensitivity) * len(pos) + 1e-9))  # positives allowed below
    return float(pos[k]) if len(pos) else 0.5


def metrics(y_true: np.ndarray, p: np.ndarray, threshold: float) -> dict:
    y_pred = (p >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = tp / (tp + fn) if tp + fn else 0.0
    spec = tn / (tn + fp) if tn + fp else 0.0
    prec = tp / (tp + fp) if tp + fp else 0.0
    f1 = 2 * prec * sens / (prec + sens) if prec + sens else 0.0
    return {
        "threshold": round(float(threshold), 4),
        "accuracy": round(float((tp + tn) / len(y_true)), 4),
        "sensitivity": round(float(sens), 4),       # malignant recall
        "specificity": round(float(spec), 4),
        "precision": round(float(prec), 4),
        "f1": round(float(f1), 4),
        "auc": round(float(roc_auc_score(y_true, p)), 4),
        "missed_cancers": int(fn),
        "false_alarms": int(fp),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def classical_baselines(split: Split, seed: int) -> dict:
    models = {
        "logreg": lambda: LogisticRegression(max_iter=5000, class_weight="balanced"),
        "svm_rbf": lambda: CalibratedClassifierCV(SVC(class_weight="balanced"), cv=5),
        "random_forest": lambda: RandomForestClassifier(
            n_estimators=300, class_weight="balanced", random_state=seed),
        "mlp": lambda: MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=2000, random_state=seed),
    }
    feature_sets = {
        "qfeat": (split.Xq_train, split.Xq_val, split.Xq_test),
        "all30": (split.Xf_train, split.Xf_val, split.Xf_test),
    }
    out = {}
    for fname, (Xtr, Xva, Xte) in feature_sets.items():
        for mname, make in models.items():
            m = make().fit(Xtr, split.y_train)
            pv, pt = m.predict_proba(Xva)[:, 1], m.predict_proba(Xte)[:, 1]
            out[f"{mname}/{fname}"] = {"p_val": pv, "p_test": pt}
    return out


def evaluate_run(cfg: Config, seed: int, verbose: bool = True) -> dict:
    cfg.validate()
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    split = load_split(cfg, seed)
    backend = make_backend(cfg, seed)
    model = HybridQNN(cfg, backend, rng)
    if verbose:
        print(f"\n  seed {seed}: train {len(split.y_train)}, val {len(split.y_val)}, "
              f"test {len(split.y_test)} | PCA variance kept {split.pca_variance:.3f} | "
              f"{cfg.n_params} circuit params + 2 readout")
    info = train(model, split, rng, verbose)

    p_val = model.predict_proba(split.Xq_val)
    p_test = model.predict_proba(split.Xq_test)
    thr = pick_threshold(p_val, split.y_val, cfg.target_sensitivity)
    preds = {"qnn": (p_test, thr)}
    result = {
        "seed": seed,
        "training": {k: v for k, v in info.items() if k != "history"},
        "history": info["history"],
        "qnn": {"default_0.5": metrics(split.y_test, p_test, 0.5),
                "val_tuned": metrics(split.y_test, p_test, thr)},
        "baselines": {},
        "ensembles": {},
    }

    base = classical_baselines(split, seed)
    for name, d in base.items():
        t = pick_threshold(d["p_val"], split.y_val, cfg.target_sensitivity)
        preds[name] = (d["p_test"], t)
        result["baselines"][name] = {"default_0.5": metrics(split.y_test, d["p_test"], 0.5),
                                     "val_tuned": metrics(split.y_test, d["p_test"], t)}
    for partner in ("logreg/qfeat", "logreg/all30"):
        pv = (p_val + base[partner]["p_val"]) / 2
        pt = (p_test + base[partner]["p_test"]) / 2
        t = pick_threshold(pv, split.y_val, cfg.target_sensitivity)
        preds[f"qnn+{partner}"] = (pt, t)
        result["ensembles"][f"qnn+{partner}"] = {"default_0.5": metrics(split.y_test, pt, 0.5),
                                                 "val_tuned": metrics(split.y_test, pt, t)}
    result["final_params"] = {"theta": model.theta.tolist(), "w": model.w, "b": model.b}
    result["test_predictions"] = {n: {"p_test": np.round(p, 6).tolist(), "val_tuned_threshold": t}
                                  for n, (p, t) in preds.items()}
    result["y_test"] = split.y_test.tolist()
    return result


# ==============================================================================
# REPORTING
# ==============================================================================

KEYS = ("accuracy", "sensitivity", "specificity", "auc", "missed_cancers")


def bootstrap_ci(runs: list[dict], name: str, n_boot: int = 2000, seed: int = 0) -> dict:
    """95% CI of the seed-averaged test metrics, resampling test PATIENTS (the +/- across seeds
    only reflects training randomness on one fixed test set)."""
    from scipy.stats import rankdata
    y = np.array(runs[0]["y_test"])
    P = np.array([r["test_predictions"][name]["p_test"] for r in runs])            # (R, N)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), (n_boot, len(y)))                                 # (B, N)
    yb, pb = y[idx], P[:, idx]                                                      # (B,N), (R,B,N)
    npos = yb.sum(1)
    ok = (npos > 0) & (npos < len(y))
    ranks = rankdata(pb, axis=-1)
    auc = ((ranks * yb).sum(-1) - npos * (npos + 1) / 2) / (npos * (len(y) - npos))  # (R, B)
    out = {"auc": np.percentile(auc.mean(0)[ok], [2.5, 97.5])}
    for rule in ("default_0.5", "val_tuned"):
        thr = np.array([0.5 if rule == "default_0.5" else r["test_predictions"][name]["val_tuned_threshold"]
                        for r in runs])[:, None, None]
        pred = pb >= thr
        acc = (pred == yb).mean(-1).mean(0)
        sens = ((pred & (yb == 1)).sum(-1) / np.maximum(npos, 1)).mean(0)
        out[rule] = {"accuracy": np.percentile(acc[ok], [2.5, 97.5]),
                     "sensitivity": np.percentile(sens[ok], [2.5, 97.5])}
    r4 = lambda a: [round(float(a[0]), 4), round(float(a[1]), 4)]
    return {"auc": r4(out["auc"]),
            **{rule: {k: r4(v) for k, v in out[rule].items()} for rule in ("default_0.5", "val_tuned")}}


def aggregate(runs: list[dict]) -> dict:
    def summarise(getter, name):
        out = {}
        ci = bootstrap_ci(runs, name)
        for rule in ("default_0.5", "val_tuned"):
            vals = {k: np.array([getter(r)[rule][k] for r in runs], dtype=float) for k in KEYS}
            out[rule] = {k: {"mean": round(float(v.mean()), 4),
                             "std": round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4),
                             "min": round(float(v.min()), 4), "max": round(float(v.max()), 4)}
                         for k, v in vals.items()}
            out[rule]["test_ci95"] = {**ci[rule], "auc": ci["auc"]}
        return out

    agg = {"qnn": summarise(lambda r: r["qnn"], "qnn")}
    for name in runs[0]["baselines"]:
        agg[name] = summarise(lambda r, n=name: r["baselines"][n], name)
    for name in runs[0]["ensembles"]:
        agg[name] = summarise(lambda r, n=name: r["ensembles"][n], name)
    return agg


def print_table(agg: dict, n_runs: int, n_test: int, n_pos: int) -> None:
    for rule, title in (("default_0.5", "threshold 0.5"),
                        ("val_tuned", "threshold tuned on validation for sensitivity")):
        print(f"\n  Test set ({n_test} patients, {n_pos} malignant), {n_runs} runs, {title}:")
        print(f"  {'model':26s} {'accuracy':>15s} {'sensitivity':>15s} {'specificity':>15s} "
              f"{'AUC':>15s} {'missed':>13s}")
        for name, d in agg.items():
            row = d[rule]
            cell = lambda k: f"{row[k]['mean']:.3f} ± {row[k]['std']:.3f}"
            miss = f"{row['missed_cancers']['mean']:.1f} ({row['missed_cancers']['min']:.0f}-" \
                   f"{row['missed_cancers']['max']:.0f})"
            print(f"  {name:26s} {cell('accuracy'):>15s} {cell('sensitivity'):>15s} "
                  f"{cell('specificity'):>15s} {cell('auc'):>15s} {miss:>13s}")
        ci = agg["qnn"][rule]["test_ci95"]
        print(f"  QNN 95% CI over test patients: accuracy {ci['accuracy']}, "
              f"sensitivity {ci['sensitivity']}, AUC {ci['auc']}")
        print("  (± is the spread across seeds on one fixed test set; the CI is the uncertainty "
              "from the test set's size)")


def check_backend(cfg: Config, n: int = 20) -> None:
    """Verify the NumPy simulator against CUDA-Q on random inputs."""
    if not HAVE_CUDAQ:
        sys.exit("CUDA-Q is not installed, so there is nothing to compare against.")
    rng = np.random.default_rng(0)
    worst = 0.0
    for reupload, ent in ((False, "ring"), (True, "ring"), (False, "brick"), (True, "brick")):
        for nf in (cfg.n_qubits, min(2 * cfg.n_qubits, cfg.n_qubits + 2)):
            c = replace(cfg, reupload=reupload, n_features=nf, entangler=ent)
            X = rng.uniform(0, np.pi, (n, nf))
            th = rng.uniform(0, 2 * np.pi, (3, c.n_params))
            a = CudaqBackend(replace(c, cudaq_option="fp64" if c.cudaq_target == "nvidia" else ""),
                             0).expectations(X, th, 0)
            b = NumpyBackend(c, 0).expectations(X, th, 0)
            diff = float(np.abs(a - b).max())
            worst = max(worst, diff)
            print(f"  {ent:5s} reupload={reupload!s:5s} n_features={nf}: max |cudaq - numpy| = {diff:.2e}")
    print("  PASS" if worst < 1e-5 else "  MISMATCH: do not trust NumPy-backend results")


# ==============================================================================
# MAIN
# ==============================================================================

def parse_args() -> tuple[Config, argparse.Namespace]:
    d = Config()
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["cudaq", "numpy"],
                    default="cudaq" if HAVE_CUDAQ else "numpy")
    ap.add_argument("--cudaq-target", default=d.cudaq_target)
    ap.add_argument("--cudaq-option", default=d.cudaq_option, help='e.g. "fp64"')
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-qubits", type=int, default=d.n_qubits)
    ap.add_argument("--n-layers", type=int, default=d.n_layers)
    ap.add_argument("--n-features", type=int, default=d.n_features)
    ap.add_argument("--reupload", action="store_true")
    ap.add_argument("--entangler", choices=["ring", "brick"], default=d.entangler)
    ap.add_argument("--epochs", type=int, default=d.epochs)
    ap.add_argument("--batch-size", type=int, default=d.batch_size)
    ap.add_argument("--lr", type=float, default=d.lr)
    ap.add_argument("--patience", type=int, default=d.patience)
    ap.add_argument("--class-weight", choices=["balanced", "none"], default=d.class_weight)
    ap.add_argument("--init", choices=["uniform", "small"], default=d.init)
    ap.add_argument("--init-scale", type=float, default=d.init_scale)
    ap.add_argument("--train-shots", type=int, default=d.train_shots)
    ap.add_argument("--eval-shots", type=int, default=d.eval_shots)
    ap.add_argument("--target-sensitivity", type=float, default=d.target_sensitivity)
    ap.add_argument("--out", default="results_v2.json")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--check-backend", action="store_true")
    ap.add_argument("--merge", nargs="+", metavar="JSON",
                    help="combine result files from separate runs (e.g. one per seed) into --out")
    a = ap.parse_args()
    cfg = Config(n_qubits=a.n_qubits, n_layers=a.n_layers, n_features=a.n_features,
                 reupload=a.reupload, entangler=a.entangler, epochs=a.epochs, batch_size=a.batch_size, lr=a.lr,
                 patience=a.patience, class_weight=a.class_weight, init=a.init, init_scale=a.init_scale, train_shots=a.train_shots,
                 eval_shots=a.eval_shots, target_sensitivity=a.target_sensitivity,
                 backend=a.backend, cudaq_target=a.cudaq_target, cudaq_option=a.cudaq_option)
    return cfg, a


def save_results(path: str, cfg: Config, runs: list[dict]) -> dict:
    agg = aggregate(runs)
    with open(path, "w") as f:
        json.dump({"config": asdict(cfg), "seeds": [r["seed"] for r in runs], "summary": agg,
                   "runs": runs}, f, indent=1, default=float)
    return agg


def merge(paths: list[str], out: str) -> None:
    """Combine result files (e.g. one per seed, each from its own process) into one summary."""
    files = [json.load(open(p)) for p in paths]
    cfgs = [{k: v for k, v in f["config"].items()} for f in files]
    if any(c != cfgs[0] for c in cfgs[1:]):
        sys.exit("Refusing to merge: the files were produced with different configurations.")
    runs = sorted((r for f in files for r in f["runs"]), key=lambda r: r["seed"])
    seeds = [r["seed"] for r in runs]
    if len(set(seeds)) != len(seeds):
        sys.exit(f"Refusing to merge: duplicate seeds {seeds}.")
    cfg = Config(**cfgs[0])
    agg = save_results(out, cfg, runs)
    y = np.array(runs[0]["y_test"])
    print(f"Merged {len(paths)} files, seeds {seeds}, backend={cfg.backend}")
    print_table(agg, len(runs), len(y), int(y.sum()))
    print(f"\n  Saved {out}")


def main() -> None:
    cfg, args = parse_args()
    if args.check_backend:
        check_backend(cfg)
        return
    if args.merge:
        merge(args.merge, args.out)
        return
    cfg.validate()
    print("=" * 78)
    print(f"Hybrid QNN v2 | backend={cfg.backend} | {cfg.n_qubits} qubits x {cfg.n_layers} layers | "
          f"{cfg.n_features} features | reupload={cfg.reupload} | entangler={cfg.entangler}")
    print("=" * 78)

    runs = []
    for s in args.seeds:
        runs.append(evaluate_run(cfg, s, verbose=not args.quiet))
        agg = save_results(args.out, cfg, runs)      # saved after every seed, so a crash loses nothing
        print(f"  [saved {len(runs)}/{len(args.seeds)} seeds to {args.out}]")
    split = load_split(cfg, args.seeds[0])
    print_table(agg, len(runs), len(split.y_test), int(split.y_test.sum()))
    print(f"\n  Saved {args.out}")


if __name__ == "__main__":
    main()
