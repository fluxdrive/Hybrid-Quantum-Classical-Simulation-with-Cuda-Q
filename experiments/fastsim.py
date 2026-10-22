"""
Fast exact statevector simulator of the v2 QNN circuit with adjoint-method gradients.

Same circuit as quantum_nn.qnn_kernel / NumpyBackend, but
  * parameters can differ per sample (theta has shape (S, K)), and
  * expect_and_grad() returns <Z_0> and d<Z_0>/d(theta) for every parameter using one forward and
    one backward pass (adjoint differentiation) instead of 2K+1 parameter-shifted circuits.

The gradients are mathematically identical to parameter shift; the adjoint method is simply a
cheaper way to compute them on a classical simulator. (On real hardware you would have to use
parameter shift, which is what the CUDA-Q backend does.)
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import quantum_nn as Q  # noqa: E402


class AdjointSim:
    def __init__(self, n_qubits: int, n_layers: int, reupload: bool = False, dtype=np.complex128,
                 entangler: str = "ring"):
        self.n, self.L, self.reupload, self.dtype = n_qubits, n_layers, reupload, dtype
        self.entangler = entangler

    @property
    def n_params(self) -> int:
        return 2 * self.n * self.L

    # ---- gate primitives on psi of shape (S, 2, ..., 2) --------------------------------
    def _idx(self, q, bit):
        return (slice(None),) * (1 + q) + (bit,)

    def _ry(self, psi, q, t):
        c = np.cos(t / 2).reshape((-1,) + (1,) * (self.n - 1))
        s = np.sin(t / 2).reshape((-1,) + (1,) * (self.n - 1))
        a0, a1 = psi[self._idx(q, 0)].copy(), psi[self._idx(q, 1)].copy()
        psi[self._idx(q, 0)] = c * a0 - s * a1
        psi[self._idx(q, 1)] = s * a0 + c * a1

    def _rz(self, psi, q, t):
        e = np.exp(-0.5j * t).reshape((-1,) + (1,) * (self.n - 1))
        psi[self._idx(q, 0)] *= e
        psi[self._idx(q, 1)] *= np.conj(e)

    def _cx(self, psi, c, t):
        idx = self._idx(c, 1)
        sub = psi[idx]
        psi[idx] = np.flip(sub, axis=t - (1 if t > c else 0) + 1).copy()

    def _apply_gen(self, psi, q, kind):
        """Return G psi for G = Y (ry) or Z (rz) on qubit q."""
        out = np.empty_like(psi)
        if kind == "ry":
            out[self._idx(q, 0)] = -1j * psi[self._idx(q, 1)]
            out[self._idx(q, 1)] = 1j * psi[self._idx(q, 0)]
        else:
            out[self._idx(q, 0)] = psi[self._idx(q, 0)]
            out[self._idx(q, 1)] = -psi[self._idx(q, 1)]
        return out

    # ---- circuit as a gate list -----------------------------------------------------------
    def gates(self, n_features: int):
        """List of (kind, qubits, source) with source ('x', i) for data or ('t', k) for params."""
        g = []
        n = self.n
        for layer in range(self.L):
            if layer == 0 or self.reupload:
                for i in range(n_features):
                    g.append(("ry", (i,), ("x", i)) if i < n else ("rz", (i - n,), ("x", i)))
            for j in range(n):
                k = 2 * (layer * n + j)
                g.append(("ry", (j,), ("t", k)))
                g.append(("rz", (j,), ("t", k + 1)))
            for a, b in Q.cnot_pairs(n, self.entangler):
                g.append(("cx", (a, b), None))
        return g

    def _angle(self, src, X, Th):
        return X[:, src[1]] if src[0] == "x" else Th[:, src[1]]

    def forward(self, X, Th):
        S = X.shape[0]
        psi = np.zeros((S,) + (2,) * self.n, dtype=self.dtype)
        psi[(slice(None),) + (0,) * self.n] = 1.0
        for kind, qs, src in self.gates(X.shape[1]):
            if kind == "cx":
                self._cx(psi, *qs)
            else:
                getattr(self, "_" + kind)(psi, qs[0], self._angle(src, X, Th))
        return psi

    @staticmethod
    def _z0(psi):
        out = psi.copy()
        out[:, 1] *= -1
        return out

    def expect(self, X, Th):
        psi = self.forward(X, Th)
        p = np.abs(psi) ** 2
        return p[:, 0].reshape(len(X), -1).sum(1) - p[:, 1].reshape(len(X), -1).sum(1)

    def expect_and_grad(self, X, Th):
        """Returns E (S,), dE (S, K)."""
        S = X.shape[0]
        psi = self.forward(X, Th)
        lam = self._z0(psi)
        E = np.real(np.sum(np.conj(psi) * lam, axis=tuple(range(1, self.n + 1))))
        dE = np.zeros((S, self.n_params))
        for kind, qs, src in reversed(self.gates(X.shape[1])):
            if kind == "cx":
                self._cx(psi, *qs)
                self._cx(lam, *qs)
                continue
            q = qs[0]
            if src[0] == "t":
                g = self._apply_gen(psi, q, kind)
                dE[:, src[1]] = np.imag(np.sum(np.conj(lam) * g, axis=tuple(range(1, self.n + 1))))
            ang = -self._angle(src, X, Th)            # apply U^dagger = U(-angle)
            getattr(self, "_" + kind)(psi, q, ang)
            getattr(self, "_" + kind)(lam, q, ang)
        return E, dE


class AdjointBackend:
    """Drop-in backend for quantum_nn (expectations) plus a fast exact gradient."""
    name = "numpy-adjoint"

    def __init__(self, cfg: Q.Config, seed: int = 0):
        self.cfg = cfg
        self.sim = AdjointSim(cfg.n_qubits, cfg.n_layers, cfg.reupload, entangler=cfg.entangler)
        self.rng = np.random.default_rng(10_000 + seed)

    def expectations(self, X, thetas, shots: int = 0):
        out = np.stack([self.sim.expect(X, np.broadcast_to(th, (len(X), len(th)))) for th in thetas])
        if shots > 0:
            p0 = np.clip((out + 1) / 2, 0, 1)
            out = (2.0 * self.rng.binomial(shots, p0) - shots) / shots
        return out


class AdjointQNN(Q.HybridQNN):
    """HybridQNN whose loss_and_grad uses adjoint gradients (identical values, far fewer passes)."""

    def loss_and_grad(self, X, y, cw):
        Th = np.broadcast_to(self.theta, (len(X), len(self.theta)))
        e0, dE = self.backend.sim.expect_and_grad(X, Th)
        p = np.clip(Q.sigmoid(self.w * e0 + self.b), 1e-7, 1 - 1e-7)
        sw = cw / cw.sum()
        loss = float(-(sw * (y * np.log(p) + (1 - y) * np.log(1 - p))).sum())
        dlogit = sw * (p - y)
        return loss, dE.T @ (dlogit * self.w), float(dlogit @ e0), float(dlogit.sum())
