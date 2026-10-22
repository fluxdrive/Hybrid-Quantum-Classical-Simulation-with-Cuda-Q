"""
Noise models for the v2 hybrid QNN, as an exact NumPy density-matrix simulator and as a CUDA-Q
noise model. Both apply the same Kraus channels in the same places:

  * after every RY / RZ gate (encoding and variational): a 1-qubit channel on that qubit
  * after every CNOT: a 2-qubit depolarizing channel on (control, target)
  * at measurement: symmetric readout error r, which maps <Z> -> (1 - 2r) <Z>

Channels (probability conventions match CUDA-Q's docstrings):
  depolarizing      1q: (1-p) rho + p/3 (X rho X + Y rho Y + Z rho Z)       p = p1
                    2q: (1-p) rho + p/15 * sum over the 15 non-identity Paulis   p = p2
  amplitude_damping K0 = [[1,0],[0,sqrt(1-g)]], K1 = [[0,sqrt(g)],[0,0]]   (after 1q gates)
  phase_flip        (1-p) rho + p Z rho Z                                   (after 1q gates)

The CUDA-Q backend builds every channel with cudaq.KrausChannel from these exact matrices.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass, field

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import quantum_nn as Q  # noqa: E402

I2 = np.eye(2, dtype=complex)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
PAULIS = [I2, X, Y, Z]


@dataclass(frozen=True)
class Noise:
    kind: str = "none"        # none | depolarizing | amplitude_damping | phase_flip
    p1: float = 0.0           # 1-qubit error per RY/RZ gate
    p2: float = 0.0           # 2-qubit depolarizing error per CNOT (depolarizing only)
    readout: float = 0.0      # symmetric readout flip probability

    def label(self) -> str:
        if self.kind == "none" and self.readout == 0:
            return "noiseless"
        parts = []
        if self.kind == "depolarizing":
            parts.append(f"depol p1={self.p1:g} p2={self.p2:g}")
        elif self.kind != "none":
            parts.append(f"{self.kind} p={self.p1:g}")
        if self.readout:
            parts.append(f"readout={self.readout:g}")
        return ", ".join(parts)


def kraus_1q(noise: Noise) -> list[np.ndarray]:
    p = noise.p1
    if noise.kind == "depolarizing":
        return [np.sqrt(1 - p) * I2] + [np.sqrt(p / 3) * P for P in (X, Y, Z)]
    if noise.kind == "amplitude_damping":
        return [np.array([[1, 0], [0, np.sqrt(1 - p)]], dtype=complex),
                np.array([[0, np.sqrt(p)], [0, 0]], dtype=complex)]
    if noise.kind == "phase_flip":
        return [np.sqrt(1 - p) * I2, np.sqrt(p) * Z]
    return []


def kraus_2q(noise: Noise) -> list[np.ndarray]:
    if noise.kind != "depolarizing" or noise.p2 == 0:
        return []
    p = noise.p2
    ops = [np.sqrt(1 - p) * np.kron(I2, I2)]
    for a in range(4):
        for b in range(4):
            if a or b:
                ops.append(np.sqrt(p / 15) * np.kron(PAULIS[a], PAULIS[b]))
    return ops


# ==============================================================================
# NumPy density-matrix backend
# ==============================================================================

class NoisyNumpyBackend:
    """rho has shape (P, S, 2,...,2 [row qubits], 2,...,2 [col qubits])."""
    name = "numpy-dm"

    def __init__(self, cfg: Q.Config, seed: int, noise: Noise, dtype=np.complex64):
        # complex64 is ~6x faster here; expectation values agree with complex128 to ~1e-6
        self.cfg, self.noise, self.dtype = cfg, noise, dtype
        self.n = cfg.n_qubits
        self.rng = np.random.default_rng(20_000 + seed)
        self.k1 = [K.astype(dtype) for K in kraus_1q(noise)]
        self.k2 = kraus_2q(noise)

    # -- helpers --------------------------------------------------------------
    def _mat_on_axis(self, rho, M, axis):
        """Contract a 2x2 matrix M (shape (P|1, S|1, 2, 2)) into one tensor axis of rho."""
        shp = M.shape[:2] + (1,) * (rho.ndim - 3)
        m = lambda i, j: M[..., i, j].reshape(shp)
        a0, a1 = np.take(rho, 0, axis=axis), np.take(rho, 1, axis=axis)
        return np.stack([m(0, 0) * a0 + m(0, 1) * a1, m(1, 0) * a0 + m(1, 1) * a1], axis=axis)

    def _unitary(self, rho, U, q):
        rho = self._mat_on_axis(rho, U, 2 + q)
        return self._mat_on_axis(rho, np.conj(U), 2 + self.n + q)

    def _cnot(self, rho, c, t):
        for off in (0, self.n):  # rows, then columns
            rho = rho.copy()
            idx = [slice(None)] * rho.ndim
            idx[2 + off + c] = 1
            sub = rho[tuple(idx)]
            tax = (2 + off + t) - (1 if t > c else 0)
            rho[tuple(idx)] = np.flip(sub, axis=tax)
        return rho

    def _fully_mix(self, rho, q):
        """I/2 (x) Tr_q(rho): completely depolarize qubit q."""
        ra, ca = 2 + q, 2 + self.n + q
        r = np.moveaxis(rho, (ra, ca), (-2, -1))
        m = 0.5 * (r[..., 0, 0] + r[..., 1, 1])
        out = np.zeros_like(r)
        out[..., 0, 0] = m
        out[..., 1, 1] = m
        return np.moveaxis(out, (-2, -1), (ra, ca))

    def _channel_1q(self, rho, q):
        if not self.k1:
            return rho
        p = self.noise.p1
        if self.noise.kind == "depolarizing":
            # (1-p) rho + p/3 (XrX + YrY + ZrZ) == (1 - 4p/3) rho + (4p/3) I/2 (x) Tr_q rho
            return (1 - 4 * p / 3) * rho + (4 * p / 3) * self._fully_mix(rho, q)
        if self.noise.kind == "phase_flip":
            # Z rho Z flips the sign of the coherences of qubit q
            sign = np.ones((2, 2), dtype=np.float32)
            sign[0, 1] = sign[1, 0] = -1
            shape = [1] * rho.ndim
            shape[2 + q], shape[2 + self.n + q] = 2, 2
            return (1 - p) * rho + p * rho * sign.reshape(shape)
        out = 0
        for K in self.k1:
            out = out + self._unitary(rho, K[None, None], q)
        return out

    def _channel_2q(self, rho, c, t):
        if not self.k2:
            return rho
        # (1-p) rho + p/15 sum_{(a,b)!=(I,I)} P rho P == (1 - 16p/15) rho + (16p/15) I/4 (x) Tr_ct rho
        p = self.noise.p2
        return (1 - 16 * p / 15) * rho + (16 * p / 15) * self._fully_mix(self._fully_mix(rho, c), t)

    # -- public ---------------------------------------------------------------
    def exact(self, Xf: np.ndarray, thetas: np.ndarray) -> np.ndarray:
        c, n = self.cfg, self.n
        P, S = thetas.shape[0], Xf.shape[0]
        rho = np.zeros((P, S) + (2,) * (2 * n), dtype=self.dtype)
        rho[(slice(None), slice(None)) + (0,) * (2 * n)] = 1.0
        ry, rz = Q.NumpyBackend._ry, Q.NumpyBackend._rz
        for layer in range(c.n_layers):
            if layer == 0 or c.reupload:
                for i in range(Xf.shape[1]):
                    q = i if i < n else i - n
                    U = (ry if i < n else rz)(Xf[:, i])[None].astype(self.dtype)
                    rho = self._channel_1q(self._unitary(rho, U, q), q)
            for j in range(n):
                k = 2 * (layer * n + j)
                rho = self._channel_1q(self._unitary(rho, ry(thetas[:, k])[:, None].astype(self.dtype), j), j)
                rho = self._channel_1q(self._unitary(rho, rz(thetas[:, k + 1])[:, None].astype(self.dtype), j), j)
            for a, b in Q.cnot_pairs(n, c.entangler):
                rho = self._channel_2q(self._cnot(rho, a, b), a, b)
        diag = np.einsum("psii->psi", rho.reshape(P, S, 2 ** n, 2 ** n)).real   # populations
        diag = diag.reshape((P, S, 2) + (2,) * (n - 1))
        p0 = diag[:, :, 0].reshape(P, S, -1).sum(-1)
        return ((1 - 2 * self.noise.readout) * (2 * p0 - 1)).astype(np.float64)

    def expectations(self, Xf, thetas, shots: int = 0):
        ev = self.exact(Xf, thetas)
        if shots <= 0:
            return ev
        p0 = np.clip((ev + 1) / 2, 0, 1)
        return (2.0 * self.rng.binomial(shots, p0) - shots) / shots


# ==============================================================================
# CUDA-Q backend (density-matrix-cpu target + explicit Kraus channels)
# ==============================================================================

def build_cudaq_noise_model(noise: Noise, n_qubits: int):
    import cudaq
    nm = cudaq.NoiseModel()
    k1, k2 = kraus_1q(noise), kraus_2q(noise)
    if k1:
        ch = cudaq.KrausChannel([np.ascontiguousarray(K, dtype=np.complex128) for K in k1])
        for op in ("ry", "rz"):
            try:
                nm.add_all_qubit_channel(op, ch)
            except Exception:
                for q in range(n_qubits):
                    nm.add_channel(op, [q], ch)
    if k2:
        ch2 = cudaq.KrausChannel([np.ascontiguousarray(K, dtype=np.complex128) for K in k2])
        try:
            nm.add_all_qubit_channel("x", ch2, num_controls=1)
        except Exception:
            pairs = set(Q.cnot_pairs(n_qubits, "ring")) | set(Q.cnot_pairs(n_qubits, "brick"))
            for a, b in sorted(pairs):
                nm.add_channel("x", [a, b], ch2)
    return nm


class NoisyCudaqBackend:
    name = "cudaq-dm"

    def __init__(self, cfg: Q.Config, seed: int, noise: Noise, target: str = "density-matrix-cpu"):
        import cudaq
        self.cudaq, self.cfg, self.noise = cudaq, cfg, noise
        cudaq.set_target(target)
        cudaq.set_random_seed(seed)
        self.H = cudaq.spin.z(0)
        self.nm = build_cudaq_noise_model(noise, cfg.n_qubits)
        self.api = None   # "kwarg": observe(..., noise_model=nm); "global": cudaq.set_noise(nm)

    def _observe(self, args, kw):
        # observe_async() does not accept noise_model in current CUDA-Q releases, so use the
        # synchronous observe(), which does. Older releases only support the global set_noise().
        if self.api != "global":
            try:
                ev = self.cudaq.observe(Q.qnn_kernel, self.H, *args, noise_model=self.nm, **kw).expectation()
                self.api = "kwarg"
                return ev
            except TypeError:
                if self.api == "kwarg":
                    raise
                self.api = "global"
                self.cudaq.set_noise(self.nm)
        return self.cudaq.observe(Q.qnn_kernel, self.H, *args, **kw).expectation()

    def expectations(self, Xf, thetas, shots: int = 0):
        c = self.cfg
        kw = {"shots_count": shots} if shots > 0 else {}
        tail = (c.n_qubits, c.n_layers, int(c.reupload), int(c.entangler == "brick"))
        ev = np.array([[self._observe((x.tolist(), th.tolist()) + tail, kw) for x in Xf]
                       for th in thetas])
        return (1 - 2 * self.noise.readout) * ev

    def close(self):
        if self.api == "global":
            self.cudaq.unset_noise()


def make_noisy_backend(cfg: Q.Config, seed: int, noise: Noise, backend: str):
    if backend == "cudaq":
        return NoisyCudaqBackend(cfg, seed, noise)
    return NoisyNumpyBackend(cfg, seed, noise)
