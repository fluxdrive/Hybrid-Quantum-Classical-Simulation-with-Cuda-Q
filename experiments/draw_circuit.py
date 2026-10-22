"""
Draw the v2 hybrid QNN circuit (default config: 4 qubits, 6 features, 3 layers, ring entangler).

    python experiments/draw_circuit.py            # writes experiments/figures/circuit.{png,svg}
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle, Arc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#d9d8d3"
ENC, VAR, READ = "#2a78d6", "#eb6834", "#1baf7a"           # encoding / trainable / readout
TINT = {ENC: "#e3eefb", VAR: "#fce6dc", READ: "#ddf3ea"}

plt.rcParams.update({"font.size": 11, "mathtext.fontset": "cm", "savefig.facecolor": SURFACE})

N = 4
Y = {q: (N - 1 - q) * 1.0 for q in range(N)}                 # q0 on top
BW, BH = 0.78, 0.5                                          # gate box size


def gate(ax, x, q, text, color, w=BW, fs=11):
    ax.add_patch(FancyBboxPatch((x - w / 2, Y[q] - BH / 2), w, BH,
                                boxstyle="round,pad=0,rounding_size=0.06",
                                fc=TINT[color], ec=color, lw=1.4, zorder=3))
    ax.text(x, Y[q], text, ha="center", va="center", color=INK, fontsize=fs, zorder=4)


def cnot(ax, x, c, t):
    ax.plot([x, x], [Y[c], Y[t]], color=INK, lw=1.4, zorder=2)
    ax.add_patch(Circle((x, Y[c]), 0.07, fc=INK, ec=INK, zorder=3))
    ax.add_patch(Circle((x, Y[t]), 0.17, fc=SURFACE, ec=INK, lw=1.4, zorder=3))
    ax.plot([x - 0.17, x + 0.17], [Y[t], Y[t]], color=INK, lw=1.4, zorder=4)
    ax.plot([x, x], [Y[t] - 0.17, Y[t] + 0.17], color=INK, lw=1.4, zorder=4)


def meter(ax, x, q):
    w, h = 0.78, 0.5
    ax.add_patch(FancyBboxPatch((x - w / 2, Y[q] - h / 2), w, h, boxstyle="round,pad=0,rounding_size=0.06",
                                fc=TINT[READ], ec=READ, lw=1.4, zorder=3))
    ax.add_patch(Arc((x, Y[q] - 0.12), 0.48, 0.42, theta1=0, theta2=180, color=INK, lw=1.3, zorder=4))
    ax.plot([x, x + 0.17], [Y[q] - 0.12, Y[q] + 0.14], color=INK, lw=1.3, zorder=4)


def bracket(ax, x0, x1, y, text, color):
    ax.plot([x0, x0, x1, x1], [y - 0.1, y, y, y - 0.1], color=color, lw=1.4)
    ax.text((x0 + x1) / 2, y + 0.12, text, ha="center", va="bottom", fontsize=10.5, color=INK2)


fig, ax = plt.subplots(figsize=(13.5, 4.6))
ax.set_facecolor(SURFACE)
fig.patch.set_facecolor(SURFACE)

# columns
x_lab, x_e1, x_e2 = 0.0, 1.4, 2.62
x_ry, x_rz = 4.25, 5.55
x_c = [6.6, 7.3, 8.0, 8.7]
x_meas = 10.35

# wires
for q in range(N):
    end = x_meas if q == 0 else 9.45
    ax.plot([x_lab + 0.5, end], [Y[q], Y[q]], color=INK2, lw=1.1, zorder=1)
    ax.text(x_lab + 0.35, Y[q], rf"$q_{q}\ \ |0\rangle$", ha="right", va="center", fontsize=12, color=INK)

# data encoding: RY(x1..x4), then RZ(x5), RZ(x6) on q0, q1
for q in range(N):
    gate(ax, x_e1, q, rf"$R_Y(x_{q + 1})$", ENC, w=1.02)
for i, q in enumerate([0, 1]):
    gate(ax, x_e2, q, rf"$R_Z(x_{i + 5})$", ENC, w=1.02)

# trainable layer (shown once)
for q in range(N):
    gate(ax, x_ry, q, rf"$R_Y(\theta^{{y}}_{{\ell,{q}}})$", VAR, w=1.16, fs=10.5)
    gate(ax, x_rz, q, rf"$R_Z(\theta^{{z}}_{{\ell,{q}}})$", VAR, w=1.16, fs=10.5)
for x, (c, t) in zip(x_c, [(0, 1), (1, 2), (2, 3), (3, 0)]):
    cnot(ax, x, c, t)

# dashed box around one layer
lx0, lx1 = x_ry - 0.75, x_c[-1] + 0.4
ax.add_patch(Rectangle((lx0, Y[3] - 0.45), lx1 - lx0, Y[0] - Y[3] + 0.9, fill=False,
                       ec=VAR, lw=1.3, ls=(0, (4, 3)), zorder=0))
ax.text(lx1 - 0.05, Y[3] - 0.52, r"repeat for layers $\ell = 1, 2, 3$", ha="right", va="top",
        fontsize=10.5, color=INK2)

# measurement + classical readout
meter(ax, x_meas, 0)
ax.text(x_meas, Y[0] - 0.42, r"$\langle Z_0 \rangle$", ha="center", va="top", fontsize=12, color=INK)
cx = 12.65
ax.annotate("", xy=(cx - 1.25, Y[0]), xytext=(x_meas + 0.42, Y[0]),
            arrowprops=dict(arrowstyle="->", color=INK, lw=1.3))
ax.add_patch(FancyBboxPatch((cx - 1.2, Y[0] - 0.42), 2.4, 0.84, boxstyle="round,pad=0,rounding_size=0.08",
                            fc="#f0efec", ec=INK2, lw=1.3, zorder=3))
ax.text(cx, Y[0] + 0.1, r"$p = \sigma(w\,\langle Z_0\rangle + b)$", ha="center", va="center", fontsize=12.5, color=INK, zorder=4)
ax.text(cx, Y[0] - 0.22, "P(malignant)", ha="center", va="center", fontsize=10, color=INK2, zorder=4)

# group labels
top = Y[0] + 0.62
bracket(ax, x_e1 - 0.5, x_e2 + 0.5, top, "Data encoding\n6 PCA features", ENC)
ax.text((lx0 + lx1) / 2, Y[0] + 0.55, "Trainable layer: 8 rotation angles + CNOT ring\n(3 layers = 24 parameters)",
        ha="center", va="bottom", fontsize=10.5, color=INK2)
bracket(ax, x_meas - 0.45, x_meas + 0.45, top, "Measure", READ)
bracket(ax, cx - 1.2, cx + 1.2, top, "Classical readout\n(2 parameters: w, b)", INK2)

ax.set_xlim(-0.9, 14.0)
ax.set_ylim(Y[3] - 1.0, Y[0] + 1.55)
ax.set_aspect("equal")
ax.axis("off")
os.makedirs(FIG, exist_ok=True)
for ext in ("png", "svg"):
    fig.savefig(os.path.join(FIG, f"circuit.{ext}"), dpi=200, bbox_inches="tight", pad_inches=0.15)
print("wrote experiments/figures/circuit.png and circuit.svg")
