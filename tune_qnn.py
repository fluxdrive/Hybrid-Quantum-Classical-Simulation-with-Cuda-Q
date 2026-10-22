"""
Hyperparameter search for the v2 hybrid QNN, scored on the VALIDATION split only.

The test split is never loaded into a model here, so picking the winner cannot leak test data.
Each configuration is trained with several seeds (different validation splits and initial
parameters) and ranked by mean validation loss.

python tune_qnn.py --backend numpy --seeds 0 1 2 --workers 4 --out tuning_v2.json
"""

import argparse
import itertools
import json
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import quantum_nn as Q

GRID = {
    "n_layers": [2, 3, 4],
    "n_features": [4, 6],
    "reupload": [True, False],
    "lr": [0.02, 0.05, 0.1],
}


def run_one(args):
    params, seed, backend = args
    cfg = Q.Config(backend=backend, epochs=80, **params)
    cfg.validate()
    np.random.seed(seed)
    rng = np.random.default_rng(seed)
    split = Q.load_split(cfg, seed)
    model = Q.HybridQNN(cfg, Q.make_backend(cfg, seed), rng)
    info = Q.train(model, split, rng, verbose=False)
    p_val = model.predict_proba(split.Xq_val)
    return {**params, "seed": seed, "val_loss": info["best_val_loss"],
            "val_auc": round(float(Q.roc_auc_score(split.y_val, p_val)), 4),
            "best_epoch": info["best_epoch"], "seconds": info["train_seconds"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["numpy", "cudaq"], default="numpy")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--workers", type=int, default=1, help="use 1 with the cudaq backend")
    ap.add_argument("--out", default="tuning_v2.json")
    a = ap.parse_args()

    combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
    jobs = [(c, s, a.backend) for c in combos for s in a.seeds]
    print(f"{len(combos)} configurations x {len(a.seeds)} seeds = {len(jobs)} training runs")
    if a.workers > 1:
        with ProcessPoolExecutor(a.workers) as ex:
            rows = list(ex.map(run_one, jobs))
    else:
        rows = [run_one(j) for j in jobs]

    table = []
    for c in combos:
        r = [x for x in rows if all(x[k] == c[k] for k in c)]
        table.append({**c, "val_loss_mean": round(float(np.mean([x["val_loss"] for x in r])), 4),
                      "val_loss_std": round(float(np.std([x["val_loss"] for x in r])), 4),
                      "val_auc_mean": round(float(np.mean([x["val_auc"] for x in r])), 4),
                      "n_params": Q.Config(**{k: c[k] for k in ("n_layers",)}).n_params})
    table.sort(key=lambda t: t["val_loss_mean"])

    print(f"\n{'layers':>6} {'feats':>5} {'reup':>5} {'lr':>5} {'params':>6} {'val loss':>16} {'val AUC':>8}")
    for t in table:
        print(f"{t['n_layers']:6d} {t['n_features']:5d} {str(t['reupload']):>5} {t['lr']:5.2f} "
              f"{t['n_params']:6d} {t['val_loss_mean']:8.4f} ± {t['val_loss_std']:.4f} "
              f"{t['val_auc_mean']:8.4f}")
    with open(a.out, "w") as f:
        json.dump({"grid": GRID, "seeds": a.seeds, "ranking": table, "runs": rows}, f, indent=1)
    print(f"\nBest by validation loss: {table[0]}")


if __name__ == "__main__":
    main()
