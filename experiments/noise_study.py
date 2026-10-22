"""
Noise study for the v2 hybrid QNN.

A. Train the default model noiselessly (one model per seed).
B. Robustness sweep: evaluate each trained model on the test set under increasing noise of four
   kinds (depolarizing gates, amplitude damping, dephasing, readout error), both
   (i)  as-is, with the readout w, b learned without noise, and
   (ii) after recalibrating only the 2 classical readout parameters on the noisy VALIDATION set.
C. Noise-aware training: train with the noise model in the loop (parameter-shift stays exact
   because each channel does not depend on theta) and compare with B(ii) at the same noise.

python experiments/noise_study.py --backend numpy --seeds 0 1 2 3 4
python experiments/noise_study.py --backend cudaq --seeds 0 1 2 3 4    # density-matrix-cpu target
python experiments/noise_study.py --check                              # CUDA-Q vs NumPy (needs CUDA-Q)
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
from noise import Noise, NoisyNumpyBackend, make_noisy_backend  # noqa: E402

SWEEPS = {
    # 2-qubit error is set 10x the 1-qubit error, the usual ratio on current hardware
    "depolarizing": [Noise("depolarizing", p / 10, p) for p in
                     (0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1)],
    "amplitude_damping": [Noise("amplitude_damping", p) for p in (0.001, 0.005, 0.01, 0.02, 0.05, 0.1)],
    "phase_flip": [Noise("phase_flip", p) for p in (0.001, 0.005, 0.01, 0.02, 0.05, 0.1)],
    "readout": [Noise("none", readout=r) for r in (0.01, 0.02, 0.05, 0.1, 0.2)],
}
TRAIN_LEVELS = {
    "nisq_like": Noise("depolarizing", 0.001, 0.01, 0.01),
    "heavy": Noise("depolarizing", 0.005, 0.05, 0.02),
}
SWEEPS["combined"] = list(TRAIN_LEVELS.values())


def strength(n: Noise) -> float:
    return n.p2 if n.kind == "depolarizing" and n.readout == 0 else (n.p1 if n.kind != "none" else n.readout)


def recalibrate(e_val, y_val):
    """Refit p = sigmoid(w*E + b) on noisy validation expectations (2 classical parameters)."""
    lr = LogisticRegression(class_weight="balanced", C=1e6, max_iter=10_000)
    lr.fit(e_val[:, None], y_val)
    return float(lr.coef_[0, 0]), float(lr.intercept_[0])


def eval_under_noise(theta, w, b, split, cfg, noise, seed, backend):
    be = make_noisy_backend(cfg, seed, noise, backend)
    e_val = be.expectations(split.Xq_val, theta[None], cfg.eval_shots)[0]
    e_test = be.expectations(split.Xq_test, theta[None], cfg.eval_shots)[0]
    if hasattr(be, "close"):
        be.close()
    p_asis = Q.sigmoid(w * e_test + b)
    w2, b2 = recalibrate(e_val, split.y_val)
    p_recal = Q.sigmoid(w2 * e_test + b2)
    return {
        "as_is": Q.metrics(split.y_test, p_asis, 0.5),
        "recalibrated": Q.metrics(split.y_test, p_recal, 0.5),
        "e_test_spread": round(float(np.std(e_test)), 5),
        "recal_w": round(w2, 3), "recal_b": round(b2, 3),
    }


def run_seed(seed: int, backend: str, do_noisy_training: bool, verbose: bool, eval_shots: int = 0) -> dict:
    cfg = Q.Config(backend="numpy", eval_shots=eval_shots)
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    split = Q.load_split(cfg, seed)

    # A. noiseless training (statevector backend; fast)
    model = Q.HybridQNN(cfg, Q.NumpyBackend(cfg, seed), rng)
    info = Q.train(model, split, rng, verbose=False)
    theta, w, b = model.theta.copy(), model.w, model.b
    p_clean = model.predict_proba(split.Xq_test)
    out = {"seed": seed, "clean": Q.metrics(split.y_test, p_clean, 0.5),
           "clean_training": {k: v for k, v in info.items() if k != "history"},
           "sweeps": {}, "noise_aware": {}}
    if verbose:
        print(f"  seed {seed}: clean test acc {out['clean']['accuracy']:.3f}")

    # B. robustness sweeps
    for family, levels in SWEEPS.items():
        out["sweeps"][family] = []
        for noise in levels:
            r = eval_under_noise(theta, w, b, split, cfg, noise, seed, backend)
            r.update({"noise": noise.__dict__, "label": noise.label(), "strength": strength(noise)})
            out["sweeps"][family].append(r)
        if verbose:
            last = out["sweeps"][family][-1]
            print(f"    {family:18s} worst level: as-is acc {last['as_is']['accuracy']:.3f}, "
                  f"recalibrated {last['recalibrated']['accuracy']:.3f}, AUC {last['as_is']['auc']:.3f}")

    # C. noise-aware training
    if do_noisy_training:  # noqa: SIM102
        for name, noise in TRAIN_LEVELS.items():
            t0 = time.time()
            rng2 = np.random.default_rng(seed)
            m2 = Q.HybridQNN(cfg, make_noisy_backend(cfg, seed, noise, backend), rng2)
            info2 = Q.train(m2, split, rng2, verbose=False)
            p = m2.predict_proba(split.Xq_test)
            out["noise_aware"][name] = {"noise": noise.__dict__, "label": noise.label(),
                                        "test": Q.metrics(split.y_test, p, 0.5),
                                        "best_epoch": info2["best_epoch"],
                                        "seconds": round(time.time() - t0, 1)}
            if verbose:
                print(f"    noise-aware {name:9s}: acc {out['noise_aware'][name]['test']['accuracy']:.3f} "
                      f"({time.time() - t0:.0f}s)")
    return out


def _run_job(job):
    return run_seed(*job)


def summarise(runs: list[dict]) -> dict:
    def ms(vals):
        v = np.array(vals, dtype=float)
        return {"mean": round(float(v.mean()), 4), "median": round(float(np.median(v)), 4),
                "std": round(float(v.std(ddof=1)) if len(v) > 1 else 0.0, 4),
                "per_seed": [round(float(x), 4) for x in v]}

    keys = ("accuracy", "sensitivity", "specificity", "auc", "missed_cancers")
    s = {"clean": {k: ms([r["clean"][k] for r in runs]) for k in keys}, "sweeps": {}, "noise_aware": {}}
    for fam in runs[0]["sweeps"]:
        s["sweeps"][fam] = []
        for i, lvl in enumerate(runs[0]["sweeps"][fam]):
            row = {"label": lvl["label"], "strength": lvl["strength"]}
            for mode in ("as_is", "recalibrated"):
                row[mode] = {k: ms([r["sweeps"][fam][i][mode][k] for r in runs]) for k in keys}
            s["sweeps"][fam].append(row)
    aware = [r for r in runs if r["noise_aware"]]
    for name in (aware[0]["noise_aware"] if aware else {}):
        rows = [r["noise_aware"][name]["test"] for r in aware]
        s["noise_aware"][name] = {"label": aware[0]["noise_aware"][name]["label"],
                                  "seeds": [r["seed"] for r in aware],
                                  **{k: ms([x[k] for x in rows]) for k in keys}}
        # matched comparison: recalibrated noiseless-trained models, same seeds, same noise
        idx = next(i for i, x in enumerate(aware[0]["sweeps"]["combined"])
                   if x["label"] == aware[0]["noise_aware"][name]["label"])
        s["noise_aware"][name]["recalibrated_same_seeds"] = {
            k: ms([r["sweeps"]["combined"][idx]["recalibrated"][k] for r in aware]) for k in keys}
    return s


def print_summary(s: dict) -> None:
    c = s["clean"]
    print(f"\nNoiseless: accuracy {c['accuracy']['mean']:.3f}, AUC {c['auc']['mean']:.3f}")
    for fam, rows in s["sweeps"].items():
        print(f"\n{fam}")
        print(f"  {'noise':42s} {'acc as-is':>10s} {'acc recal':>10s} {'sens recal':>10s} {'AUC':>7s}")
        for r in rows:
            print(f"  {r['label']:42s} {r['as_is']['accuracy']['mean']:10.3f} "
                  f"{r['recalibrated']['accuracy']['mean']:10.3f} "
                  f"{r['recalibrated']['sensitivity']['mean']:10.3f} {r['as_is']['auc']['mean']:7.3f}")
    if s["noise_aware"]:
        print("\nNoise-aware training vs noiseless training + recalibration (same noise):")
        for name, r in s["noise_aware"].items():
            recal = r["recalibrated_same_seeds"]
            print(f"  {r['label']:42s} noise-aware acc {r['accuracy']['mean']:.3f} ± {r['accuracy']['std']:.3f} | "
                  f"recalibrated acc {recal['accuracy']['mean']:.3f} ± {recal['accuracy']['std']:.3f}")


def check(n: int = 8) -> None:
    """Compare the NumPy density-matrix simulator with CUDA-Q's density-matrix-cpu target."""
    if not Q.HAVE_CUDAQ:
        sys.exit("CUDA-Q is not installed, so there is nothing to compare against.")
    from noise import NoisyCudaqBackend
    rng = np.random.default_rng(0)
    worst = 0.0
    for cfg, noise in [(Q.Config(entangler=e), nz) for e in ("ring", "brick") for nz in
                       (Noise(), Noise("depolarizing", 0.01, 0.05, 0.02),
                        Noise("amplitude_damping", 0.05), Noise("phase_flip", 0.05))]:
        Xf = rng.uniform(0, np.pi, (n, cfg.n_features))
        th = rng.uniform(0, 2 * np.pi, (2, cfg.n_params))
        cb = NoisyCudaqBackend(cfg, 0, noise)
        a = cb.expectations(Xf, th)
        cb.close()
        b = NoisyNumpyBackend(cfg, 0, noise).expectations(Xf, th)
        d = float(np.abs(a - b).max())
        worst = max(worst, d)
        print(f"  {cfg.entangler:5s} {noise.label():45s} max |cudaq - numpy| = {d:.2e}  (noise API: {cb.api})")
    print("  PASS" if worst < 1e-5 else "  MISMATCH: check the CUDA-Q noise-model API for your version")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["numpy", "cudaq"], default="numpy")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--noise-aware-seeds", type=int, nargs="*", default=[0, 1, 2],
                    help="seeds that also get noise-aware training (slow: density-matrix training)")
    ap.add_argument("--workers", type=int, default=1, help="parallel seeds (numpy backend only)")
    ap.add_argument("--eval-shots", type=int, default=0,
                    help="measurement shots for evaluation/recalibration (0 = exact expectation values)")
    ap.add_argument("--out", default=None, help="default: experiments/noise_results_<backend>.json")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        check()
        return
    jobs = [(s, a.backend, s in a.noise_aware_seeds, True, a.eval_shots) for s in a.seeds]
    if a.workers > 1 and a.backend == "numpy":
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(a.workers) as ex:
            runs = list(ex.map(_run_job, jobs))
    else:
        runs = [_run_job(j) for j in jobs]
    s = summarise(runs)
    print_summary(s)
    a.out = a.out or os.path.join(HERE, f"noise_results_{a.backend}"
                                        f"{f'_shots{a.eval_shots}' if a.eval_shots else ''}.json")
    with open(a.out, "w") as f:
        json.dump({"backend": a.backend, "seeds": a.seeds, "eval_shots": a.eval_shots, "summary": s,
                   "runs": runs}, f, indent=1,
                  default=float)
    print(f"\nSaved {a.out}")


if __name__ == "__main__":
    main()
