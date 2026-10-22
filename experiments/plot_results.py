"""
Make the figures in experiments/figures/ from whichever result JSONs exist.

    python experiments/plot_results.py

Looks for (in experiments/): noise_results_*.json, scaling_gradients_*.json, scaling_width_*.json,
scaling_timing.json. CUDA-Q results are preferred over NumPy ones when both exist.
"""

import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")

# Palette: validated categorical slots 1-3 (passes all-pairs CVD checks); text in neutral inks.
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e6e5e1"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    "axes.titlesize": 11, "axes.titleweight": "bold", "legend.frameon": False,
    "lines.linewidth": 2, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
})
MARK = dict(marker="o", markersize=6.5, markeredgecolor=SURFACE, markeredgewidth=1.5)


def load(pattern):
    paths = sorted(glob.glob(os.path.join(HERE, pattern)))
    paths.sort(key=lambda p: ("cudaq" not in p, p))      # prefer CUDA-Q results
    return (json.load(open(paths[0])), os.path.basename(paths[0])) if paths else (None, None)


def end_label(ax, x, y, text, dy=0):
    ax.annotate(text, (x, y), xytext=(6, dy), textcoords="offset points", va="center",
                fontsize=9, color=INK2)


def save(fig, name, source):
    fig.text(0.99, -0.02, f"source: {source}", fontsize=7.5, color=INK2, ha="right", va="top")
    os.makedirs(FIG, exist_ok=True)
    path = os.path.join(FIG, name)
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.relpath(path, os.path.join(HERE, "..")))


def plot_gradients():
    d, src = load("scaling_gradients_*.json")
    if not d:
        return
    rows = d["rows"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    series = [("ring", "uniform", S1, "ring entangler"),
              ("brick", "uniform", S2, "brickwork entangler"),
              ("ring", "small", S3, "ring, near-zero init")]
    for ent, init, col, lab in series:
        r = sorted([x for x in rows if x["entangler"] == ent and x["init"] == init and x["layers"] == 3],
                   key=lambda x: x["n_qubits"])
        if not r:
            continue
        n = [x["n_qubits"] for x in r]
        v = [x["var_grad_mean"] for x in r]
        ax.plot(n, v, color=col, label=lab, **MARK)
        end_label(ax, n[-1], v[-1], lab)
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("qubits")
    ax.set_ylabel("Var[∂⟨Z₀⟩/∂θ]  (mean over parameters)")
    ax.set_title("Gradient variance vs. circuit width (3 layers, random parameters)", loc="left")
    ax.legend(loc="lower left")
    ax.set_xlim(right=ax.get_xlim()[1] + 4)
    save(fig, "gradient_variance.png", src)


def plot_noise():
    # prefer finite-shot results (what EXPERIMENTS.md reports), CUDA-Q before NumPy
    paths = sorted(glob.glob(os.path.join(HERE, "noise_results*.json")),
                   key=lambda p: ("shots" not in p, "cudaq" not in p, p))
    if not paths:
        return
    d, src = json.load(open(paths[0])), os.path.basename(paths[0])
    s = d["summary"]
    stat = "median" if "median" in s["clean"]["accuracy"] else "mean"
    clean = s["clean"]["accuracy"][stat]
    fams = [("depolarizing", "Depolarizing gates (2-qubit error, 1-qubit = ÷10)"),
            ("amplitude_damping", "Amplitude damping (per 1-qubit gate)"),
            ("phase_flip", "Dephasing / phase flip (per 1-qubit gate)"),
            ("readout", "Readout error")]
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 6.4), sharey=True)
    for ax, (fam, title) in zip(axes.flat, fams):
        rows = s["sweeps"][fam]
        x = [r["strength"] for r in rows]
        a = [r["as_is"]["accuracy"][stat] for r in rows]
        c = [r["recalibrated"]["accuracy"][stat] for r in rows]
        ax.axhline(clean, color=INK2, linewidth=0.8)
        ax.plot(x, a, color=S1, label="as trained", **MARK)
        ax.plot(x, c, color=S2, label="readout recalibrated", **MARK)
        ax.set_xscale("log")
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("error probability")
    for ax in axes[:, 0]:
        ax.set_ylabel("test accuracy")
    axes[0, 0].annotate(f"no noise {clean:.3f}", (axes[0, 0].get_xlim()[0], clean), xytext=(4, 5),
                        textcoords="offset points", fontsize=8.5, color=INK2)
    lo = min(min(r["as_is"]["accuracy"][stat] for r in s["sweeps"][f]) for f, _ in fams)
    axes[0, 0].set_ylim(max(0.5, lo - 0.05), 1.0)
    axes[0, 0].legend(loc="lower left")
    shots = d.get("eval_shots", 0)
    fig.suptitle("Accuracy of the noiselessly trained model under hardware noise "
                 f"({stat} of {len(d['seeds'])} seeds, "
                 f"{f'{shots} shots' if shots else 'exact expectation values'})",
                 x=0.01, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "noise_robustness.png", src)


def plot_width():
    d, src = load("scaling_width_*.json")
    if not d:
        return
    summ = d["summary"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for ent, col, lab in [("ring", S1, "QNN, ring"), ("brick", S2, "QNN, brickwork")]:
        r = sorted([x for x in summ if x["entangler"] == ent and x["init"] == "uniform"], key=lambda x: x["n_qubits"])
        if not r:
            continue
        n = np.array([x["n_qubits"] for x in r])
        m = np.array([x["qnn_acc"][0] for x in r])
        sd = np.array([x["qnn_acc"][1] for x in r])
        ax.fill_between(n, m - sd, m + sd, color=col, alpha=0.10, linewidth=0)
        ax.plot(n, m, color=col, label=lab, **MARK)
        end_label(ax, n[-1], m[-1], lab)
    r = sorted([x for x in summ if x["entangler"] == "ring" and x["init"] == "uniform"], key=lambda x: x["n_qubits"])
    if r:
        n = [x["n_qubits"] for x in r]
        m = [x["logreg_acc"][0] for x in r]
        ax.plot(n, m, color=S3, label="logistic regression, same features", **MARK)
        end_label(ax, n[-1], m[-1], "logistic reg.")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("qubits  (each encodes 2 PCA components)")
    ax.set_ylabel("test accuracy")
    ax.set_title("Accuracy vs. circuit width (mean ± sd over seeds)", loc="left")
    ax.legend(loc="lower right")
    ax.set_xlim(right=ax.get_xlim()[1] + 2)
    save(fig, "accuracy_vs_width.png", src)


def plot_timing():
    d, src = load("scaling_timing*.json")
    if not d:
        return
    rows = d["rows"]
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    names = {"qpp-cpu": "CPU (qpp-cpu)", "nvidia": "GPU (nvidia, fp32)", "nvidia-fp64": "GPU (nvidia, fp64)",
             "numpy": "NumPy reference"}
    for t, col in zip([t for t in names if any(r["target"] == t for r in rows)], (S1, S2, S3)):
        r = sorted([x for x in rows if x["target"] == t], key=lambda x: x["n_qubits"])
        n = [x["n_qubits"] for x in r]
        v = [x["per_circuit_s_batched"] * 1e3 for x in r]
        ax.plot(n, v, color=col, label=names[t], **MARK)
        end_label(ax, n[-1], v[-1], names[t])
    ax.set_yscale("log")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("qubits")
    ax.set_ylabel("ms per circuit (batched)")
    ax.set_title("CUDA-Q simulation time vs. qubits", loc="left")
    ax.legend(loc="upper left")
    ax.set_xlim(right=ax.get_xlim()[1] + 5)
    save(fig, "simulation_timing.png", src)


if __name__ == "__main__":
    plot_gradients()
    plot_noise()
    plot_width()
    plot_timing()
