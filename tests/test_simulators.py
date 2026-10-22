"""
Correctness tests for the simulators and training math. Pure NumPy, no CUDA-Q needed.

    python tests/test_simulators.py

Every simulator is compared against an independent dense-matrix reference built with Kronecker
products (statevector and density-matrix with explicit Kraus sums), for both entanglers.
"""

import os
import sys
from functools import reduce

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "experiments"))
import quantum_nn as Q  # noqa: E402
import fastsim as F  # noqa: E402
import noise as N  # noqa: E402

I2, X, Z = np.eye(2), np.array([[0, 1], [1, 0]]), np.diag([1.0, -1.0])
FAILURES = []


def check(name, value, tol):
    ok = value < tol
    print(f"  {'PASS' if ok else 'FAIL'}  {name}: {value:.2e}")
    if not ok:
        FAILURES.append(name)


def kron(ops):
    return reduce(np.kron, ops)


def one(U, q, n):
    return kron([U if i == q else I2 for i in range(n)])


def cxm(c, t, n):
    P0, P1 = np.diag([1, 0]), np.diag([0, 1])
    return (kron([P0 if i == c else I2 for i in range(n)]) +
            kron([P1 if i == c else (X if i == t else I2) for i in range(n)]))


def RY(t):
    c, s = np.cos(t / 2), np.sin(t / 2)
    return np.array([[c, -s], [s, c]], dtype=complex)


def RZ(t):
    return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])


def ref_ops(cfg, x, th):
    """Yield (unitary, noise-site) in circuit order; noise-site = ('1q', q) or ('2q', c, t)."""
    n = cfg.n_qubits
    for layer in range(cfg.n_layers):
        if layer == 0 or cfg.reupload:
            for i, v in enumerate(x):
                q = i if i < n else i - n
                yield one(RY(v) if i < n else RZ(v), q, n), ("1q", q)
        for j in range(n):
            k = 2 * (layer * n + j)
            yield one(RY(th[k]), j, n), ("1q", j)
            yield one(RZ(th[k + 1]), j, n), ("1q", j)
        for a, b in Q.cnot_pairs(n, cfg.entangler):
            yield cxm(a, b, n), ("2q", a, b)


def ref_statevector(cfg, x, th):
    n = cfg.n_qubits
    psi = np.zeros(2 ** n, complex)
    psi[0] = 1
    for U, _ in ref_ops(cfg, x, th):
        psi = U @ psi
    return np.real(psi.conj() @ one(Z, 0, n) @ psi)


def embed2(K4, c, t, n):
    paulis = N.PAULIS
    out = np.zeros((2 ** n, 2 ** n), complex)
    for a in range(4):
        for b in range(4):
            coef = np.trace(np.kron(paulis[a], paulis[b]).conj().T @ K4) / 4
            if abs(coef) > 1e-14:
                out += coef * kron([paulis[a] if i == c else (paulis[b] if i == t else I2) for i in range(n)])
    return out


def ref_density(cfg, noise, x, th):
    n = cfg.n_qubits
    rho = np.zeros((2 ** n, 2 ** n), complex)
    rho[0, 0] = 1
    k1, k2 = N.kraus_1q(noise), N.kraus_2q(noise)
    for U, site in ref_ops(cfg, x, th):
        rho = U @ rho @ U.conj().T
        if site[0] == "1q" and k1:
            Ks = [one(K, site[1], n) for K in k1]
            rho = sum(K @ rho @ K.conj().T for K in Ks)
        if site[0] == "2q" and k2:
            Ks = [embed2(K, site[1], site[2], n) for K in k2]
            rho = sum(K @ rho @ K.conj().T for K in Ks)
    return (1 - 2 * noise.readout) * np.real(np.trace(one(Z, 0, n) @ rho))


def main():
    rng = np.random.default_rng(0)
    configs = [Q.Config(n_qubits=4, n_layers=3, n_features=6, entangler=e, reupload=r)
               for e in ("ring", "brick") for r in (False, True)]
    configs += [Q.Config(n_qubits=5, n_layers=2, n_features=7, entangler="brick"),
                Q.Config(n_qubits=3, n_layers=2, n_features=3, entangler="ring"),
                Q.Config(n_qubits=2, n_layers=3, n_features=4, entangler="brick")]

    print("Statevector simulators vs dense reference")
    for cfg in configs:
        Xs = rng.uniform(0, np.pi, (4, cfg.n_features))
        th = rng.uniform(0, 2 * np.pi, (2, cfg.n_params))
        ref = np.array([[ref_statevector(cfg, x, t) for x in Xs] for t in th])
        a = Q.NumpyBackend(cfg, 0).expectations(Xs, th)
        b = F.AdjointBackend(cfg, 0).expectations(Xs, th)
        tag = f"{cfg.n_qubits}q {cfg.entangler} reupload={cfg.reupload}"
        check(f"NumpyBackend   {tag}", np.abs(a - ref).max(), 1e-12)
        check(f"AdjointBackend {tag}", np.abs(b - ref).max(), 1e-12)

    print("Adjoint gradients vs parameter shift")
    for cfg in configs:
        Xs = rng.uniform(0, np.pi, (3, cfg.n_features))
        th = rng.uniform(0, 2 * np.pi, cfg.n_params)
        _, dE = F.AdjointSim(cfg.n_qubits, cfg.n_layers, cfg.reupload, entangler=cfg.entangler) \
            .expect_and_grad(Xs, np.broadcast_to(th, (3, len(th))).copy())
        sh = np.repeat(th[None], 2 * len(th), 0)
        for k in range(len(th)):
            sh[2 * k, k] += np.pi / 2
            sh[2 * k + 1, k] -= np.pi / 2
        E = Q.NumpyBackend(cfg, 0).expectations(Xs, sh)
        check(f"{cfg.n_qubits}q {cfg.entangler} reupload={cfg.reupload}",
              np.abs(dE - ((E[0::2] - E[1::2]) / 2).T).max(), 1e-12)

    print("Density-matrix simulator vs dense Kraus reference")
    noises = [N.Noise(), N.Noise("depolarizing", 0.01, 0.05, 0.02),
              N.Noise("amplitude_damping", 0.05), N.Noise("phase_flip", 0.07, readout=0.01)]
    for cfg in configs[:2] + configs[2:3] + configs[4:6]:
        for nz in noises:
            Xs = rng.uniform(0, np.pi, (2, cfg.n_features))
            th = rng.uniform(0, 2 * np.pi, (2, cfg.n_params))
            ref = np.array([[ref_density(cfg, nz, x, t) for x in Xs] for t in th])
            a = N.NoisyNumpyBackend(cfg, 0, nz, np.complex128).expectations(Xs, th)
            b = N.NoisyNumpyBackend(cfg, 0, nz).expectations(Xs, th)       # complex64 default
            tag = f"{cfg.n_qubits}q {cfg.entangler} {nz.label()}"
            check(f"complex128 {tag}", np.abs(a - ref).max(), 1e-12)
            check(f"complex64  {tag}", np.abs(b - ref).max(), 1e-5)

    print("Shot sampling is unbiased (density-matrix backend, 20000 shots)")
    cfg = Q.Config(entangler="brick")
    Xs, th = rng.uniform(0, np.pi, (3, 6)), rng.uniform(0, 2 * np.pi, (1, cfg.n_params))
    be = N.NoisyNumpyBackend(cfg, 0, noises[1])
    exact = be.expectations(Xs, th)
    est = np.mean([be.expectations(Xs, th, shots=20000) for _ in range(5)], axis=0)
    check("|mean of 5x20000-shot estimates - exact|", np.abs(est - exact).max(), 0.01)

    print("Kraus completeness")
    for nz in noises[1:]:
        check(f"{nz.kind} 1q", np.abs(sum(K.conj().T @ K for K in N.kraus_1q(nz)) - np.eye(2)).max(), 1e-12)
    check("depolarizing 2q", np.abs(sum(K.conj().T @ K for K in N.kraus_2q(noises[1])) - np.eye(4)).max(), 1e-12)

    print("Training gradient (loss_and_grad) vs finite differences")
    cfg = Q.Config()
    m = Q.HybridQNN(cfg, Q.NumpyBackend(cfg, 0), np.random.default_rng(1))
    m.w, m.b = 1.7, -0.3
    Xb, yb = rng.uniform(0, np.pi, (16, 6)), rng.integers(0, 2, 16)
    cw = Q.class_weights(yb, "balanced")
    L, gt, gw, gb = m.loss_and_grad(Xb, yb, cw)

    def loss_at(th, w, b):
        e = Q.NumpyBackend(cfg, 0).expectations(Xb, th[None])[0]
        return Q.weighted_bce(Q.sigmoid(w * e + b), yb, cw)

    h = 1e-6
    fd = np.array([(loss_at(m.theta + h * e, m.w, m.b) - loss_at(m.theta - h * e, m.w, m.b)) / (2 * h)
                   for e in np.eye(len(m.theta))])
    check("theta (relative)", np.abs(gt - fd).max() / np.abs(fd).max(), 1e-6)
    check("w", abs(gw - (loss_at(m.theta, m.w + h, m.b) - loss_at(m.theta, m.w - h, m.b)) / (2 * h)), 1e-7)
    check("b", abs(gb - (loss_at(m.theta, m.w, m.b + h) - loss_at(m.theta, m.w, m.b - h)) / (2 * h)), 1e-7)
    m2 = F.AdjointQNN(cfg, F.AdjointBackend(cfg, 0), np.random.default_rng(1))
    m2.w, m2.b = 1.7, -0.3
    L2, gt2, gw2, gb2 = m2.loss_and_grad(Xb, yb, cw)
    check("AdjointQNN == HybridQNN", max(abs(L - L2), np.abs(gt - gt2).max(), abs(gw - gw2), abs(gb - gb2)), 1e-12)

    print("Data and threshold invariants")
    s0, s7 = Q.load_split(cfg, 0), Q.load_split(cfg, 7)
    check("test labels identical across seeds", float(not np.array_equal(s0.y_test, s7.y_test)), 0.5)
    check("positive class is malignant (212 of 569)",
          abs(int(s0.y_train.sum() + s0.y_val.sum() + s0.y_test.sum()) - 212), 0.5)
    p = np.array([.1, .2, .3, .4, .5, .6, .7, .8, .9, .95])
    y = np.array([0, 0, 0, 0, 1, 0, 1, 1, 1, 1])
    check("pick_threshold 80% -> 0.7", abs(Q.pick_threshold(p, y, 0.8) - 0.7), 1e-12)
    check("pick_threshold 100% -> 0.5", abs(Q.pick_threshold(p, y, 1.0) - 0.5), 1e-12)

    print("\nALL PASSED" if not FAILURES else f"\n{len(FAILURES)} FAILED: {FAILURES}")
    sys.exit(1 if FAILURES else 0)


if __name__ == "__main__":
    main()
