"""
Scaling study for the v2 hybrid QNN.

  --part timing     How long CUDA-Q takes to simulate the circuit as qubits grow, CPU (qpp-cpu)
                    vs GPU (nvidia). Latency of one call and throughput over a batch of circuits.
  --part width      Accuracy vs circuit width: n qubits encode 2n PCA components (up to all 30
                    features at 15 qubits). Logistic regression on the same features is the reference.
  --part gradients  Trainability: variance of d<Z_0>/d(theta) over random parameters as qubits grow.
                    Exponentially shrinking variance is a barren plateau.

Examples
  python experiments/scaling_study.py --part timing --targets qpp-cpu nvidia --max-qubits 30
  python experiments/scaling_study.py --part width --backend cudaq --widths 2 4 6 8 10 12 15 --seeds 0 1 2
  python experiments/scaling_study.py --part gradients --backend cudaq --max-qubits 20
  (--backend numpy runs width/gradients on the exact NumPy simulator with adjoint gradients)
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
from sklearn.linear_model import LogisticRegression

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
import quantum_nn as Q  # noqa: E402
from fastsim import AdjointBackend, AdjointQNN, AdjointSim  # noqa: E402

N_LAYERS = 3


def feats_for(n: int) -> int:
    return min(2 * n, 30)


# ==============================================================================
# Part 1: simulation timing (CUDA-Q)
# ==============================================================================

def set_target(name: str):
    import cudaq
    if name == "nvidia-fp64":
        cudaq.set_target("nvidia", option="fp64")
    else:
        cudaq.set_target(name)
    return cudaq


def time_target(target: str, qubits: list[int], batch: int, reps: int, budget: float) -> list[dict]:
    rows = []
    rng = np.random.default_rng(0)
    if target == "numpy":
        sim_for = lambda n: AdjointSim(n, N_LAYERS)
    else:
        cudaq = set_target(target)
        H = cudaq.spin.z(0)
    for n in qubits:
        nf = feats_for(n)
        K = 2 * n * N_LAYERS
        X = rng.uniform(0, np.pi, (batch, nf))
        Th = rng.uniform(0, 2 * np.pi, (batch, K))
        try:
            if target == "numpy":
                sim = sim_for(n)
                one = lambda: sim.expect(X[:1], Th[:1])
                many = lambda: sim.expect(X, Th)
            else:
                def one():
                    return cudaq.observe(Q.qnn_kernel, H, X[0].tolist(), Th[0].tolist(), n, N_LAYERS, 0, 0).expectation()

                def many():
                    fs = [cudaq.observe_async(Q.qnn_kernel, H, X[i].tolist(), Th[i].tolist(), n, N_LAYERS, 0, 0)
                          for i in range(batch)]
                    return [f.get().expectation() for f in fs]
            t0 = time.perf_counter()
            one()                                   # warm-up (JIT + allocation)
            warm = time.perf_counter() - t0
            lat = []
            for _ in range(reps):
                t0 = time.perf_counter()
                one()
                lat.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            many()
            per_circ = (time.perf_counter() - t0) / batch
        except Exception as e:  # out of memory, unsupported target, ...
            print(f"  {target:12s} n={n:2d}: stopped ({type(e).__name__}: {str(e)[:80]})")
            break
        row = {"target": target, "n_qubits": n, "warmup_s": warm, "latency_s": float(np.median(lat)),
               "per_circuit_s_batched": per_circ, "statevector_mb_fp64": (2 ** n) * 16 / 1e6}
        rows.append(row)
        print(f"  {target:12s} n={n:2d}: latency {row['latency_s'] * 1e3:9.2f} ms | "
              f"batched {per_circ * 1e3:9.2f} ms/circuit")
        if row["latency_s"] > budget:
            print(f"  {target:12s} stopping: one call exceeded {budget}s")
            break
    return rows


# ==============================================================================
# Part 2: accuracy vs width
# ==============================================================================

def width_run(n: int, seed: int, backend: str, epochs: int, init: str = "uniform",
              entangler: str = "ring") -> dict:
    cfg = Q.Config(n_qubits=n, n_features=feats_for(n), n_layers=N_LAYERS, epochs=epochs, init=init,
                   entangler=entangler,
                   backend="cudaq" if backend == "cudaq" else "numpy")
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    split = Q.load_split(cfg, seed)
    if backend == "cudaq":
        model = Q.HybridQNN(cfg, Q.CudaqBackend(cfg, seed), rng)
    else:
        model = AdjointQNN(cfg, AdjointBackend(cfg, seed), rng)
    t0 = time.time()
    info = Q.train(model, split, rng, verbose=False)
    p = model.predict_proba(split.Xq_test)
    lr = LogisticRegression(max_iter=5000, class_weight="balanced").fit(split.Xq_train, split.y_train)
    return {"n_qubits": n, "n_features": cfg.n_features, "seed": seed, "init": init, "entangler": entangler,
            "n_params": cfg.n_params + 2,
            "pca_variance": round(split.pca_variance, 4),
            "qnn": Q.metrics(split.y_test, p, 0.5),
            "logreg_same_features": Q.metrics(split.y_test, lr.predict_proba(split.Xq_test)[:, 1], 0.5),
            "best_epoch": info["best_epoch"], "train_seconds": round(time.time() - t0, 1)}


def _width_job(args):
    return width_run(*args)


# ==============================================================================
# Part 3: gradient variance vs qubits (barren plateaus)
# ==============================================================================

def grad_variance(n: int, n_samples: int, backend: str, layers: int, init: str = "uniform",
                  entangler: str = "ring", init_scale: float = 0.1, chunk: int = 25) -> dict:
    """Var over random theta (and random training inputs) of d<Z0>/d(theta_k)."""
    cfg = Q.Config(n_qubits=n, n_features=feats_for(n), n_layers=layers, entangler=entangler)
    split = Q.load_split(cfg, 0)
    rng = np.random.default_rng(1000 + n)
    K = cfg.n_params
    Th = (rng.normal(0, init_scale, (n_samples, K)) if init == "small"
          else rng.uniform(0, 2 * np.pi, (n_samples, K)))
    Xs = split.Xq_train[rng.integers(0, len(split.Xq_train), n_samples)]
    if backend == "cudaq":
        be = Q.CudaqBackend(cfg, 0)
        G, E = np.zeros((n_samples, K)), np.zeros(n_samples)
        for s in range(n_samples):
            sh = np.repeat(Th[s][None], 2 * K + 1, 0)
            for k in range(K):
                sh[1 + 2 * k, k] += np.pi / 2
                sh[2 + 2 * k, k] -= np.pi / 2
            ev = be.expectations(Xs[s][None], sh, 0)[:, 0]
            E[s], G[s] = ev[0], (ev[1::2] - ev[2::2]) / 2
    else:
        sim = AdjointSim(n, layers, entangler=entangler)
        Es, Gs = [], []
        for i in range(0, n_samples, chunk):
            e, g = sim.expect_and_grad(Xs[i:i + chunk], Th[i:i + chunk].copy())
            Es.append(e)
            Gs.append(g)
        E, G = np.concatenate(Es), np.concatenate(Gs)
    var_k = G.var(axis=0, ddof=1)
    return {"n_qubits": n, "layers": layers, "n_samples": n_samples, "init": init, "entangler": entangler,
            "var_grad_theta0": float(var_k[0]),            # first RY on qubit 0
            "var_grad_mean": float(var_k.mean()),           # averaged over all parameters
            "var_grad_last": float(var_k[-2]),              # last-layer RY on the last qubit
            "var_expectation": float(E.var(ddof=1))}


# ==============================================================================

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--part", choices=["timing", "width", "gradients"], required=True)
    ap.add_argument("--backend", choices=["numpy", "cudaq"], default="numpy")
    ap.add_argument("--targets", nargs="+", default=["qpp-cpu", "nvidia"],
                    help="timing: any of qpp-cpu nvidia nvidia-fp64 numpy")
    ap.add_argument("--min-qubits", type=int, default=4)
    ap.add_argument("--max-qubits", type=int, default=28)
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--time-budget", type=float, default=20.0)
    ap.add_argument("--widths", type=int, nargs="+", default=[2, 3, 4, 5, 6, 8, 10, 12])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--samples", type=int, default=200)
    ap.add_argument("--layers", type=int, nargs="+", default=[N_LAYERS])
    ap.add_argument("--inits", nargs="+", default=["uniform"], choices=["uniform", "small"])
    ap.add_argument("--entanglers", nargs="+", default=["ring", "brick"], choices=["ring", "brick"])
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    out = a.out or os.path.join(HERE, "scaling_timing.json" if a.part == "timing"
                                else f"scaling_{a.part}_{a.backend}.json")

    if a.part == "timing":
        qubits = list(range(a.min_qubits, a.max_qubits + 1, a.step))
        rows = []
        for t in a.targets:
            rows += time_target(t, qubits, a.batch, a.reps, a.time_budget)
        result = {"rows": rows}
    elif a.part == "width":
        jobs = [(n, s, a.backend, a.epochs, i, e) for e in a.entanglers for i in a.inits
                for n in a.widths for s in a.seeds]
        if a.workers > 1 and a.backend == "numpy":
            from concurrent.futures import ProcessPoolExecutor
            with ProcessPoolExecutor(a.workers) as ex:
                rows = list(ex.map(_width_job, jobs))
        else:
            rows = [_width_job(j) for j in jobs]
        summary = []
        for ent, init, n in [(e, i, n) for e in a.entanglers for i in a.inits for n in a.widths]:
            rs = [r for r in rows if r["n_qubits"] == n and r["init"] == init and r["entangler"] == ent]
            m = lambda f: (float(np.mean([f(r) for r in rs])), float(np.std([f(r) for r in rs], ddof=1)) if len(rs) > 1 else 0.0)
            row = {"entangler": ent, "init": init, "n_qubits": n, "n_features": rs[0]["n_features"], "n_params": rs[0]["n_params"],
                   "pca_variance": rs[0]["pca_variance"],
                   "qnn_acc": m(lambda r: r["qnn"]["accuracy"]), "qnn_sens": m(lambda r: r["qnn"]["sensitivity"]),
                   "qnn_auc": m(lambda r: r["qnn"]["auc"]),
                   "logreg_acc": m(lambda r: r["logreg_same_features"]["accuracy"]),
                   "train_seconds": float(np.mean([r["train_seconds"] for r in rs]))}
            summary.append(row)
            print(f"  {ent:5s} {init:7s} n={n:2d} feats={row['n_features']:2d} params={row['n_params']:3d} | QNN acc "
                  f"{row['qnn_acc'][0]:.3f} ± {row['qnn_acc'][1]:.3f}, AUC {row['qnn_auc'][0]:.3f} | "
                  f"logreg acc {row['logreg_acc'][0]:.3f} | {row['train_seconds']:.0f}s/run")
        result = {"summary": summary, "runs": rows}
    else:
        rows = []
        for ent, init, L in [(e, i, L) for e in a.entanglers for i in a.inits for L in a.layers]:
            for n in range(max(2, a.min_qubits), a.max_qubits + 1):
                t0 = time.time()
                r = grad_variance(n, a.samples, a.backend, L, init, ent)
                rows.append(r)
                print(f"  {ent:5s} {init:7s} L={L} n={n:2d}: Var[dE/dtheta] mean {r['var_grad_mean']:.3e} | first "
                      f"{r['var_grad_theta0']:.3e} | Var[E] {r['var_expectation']:.3e} ({time.time() - t0:.0f}s)")
        result = {"rows": rows}

    with open(out, "w") as f:
        json.dump({"part": a.part, "backend": a.backend, "args": vars(a), **result}, f, indent=1, default=float)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
