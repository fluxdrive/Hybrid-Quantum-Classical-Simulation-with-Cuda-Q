#!/usr/bin/env python3
"""Minimal test to isolate segfault"""

import cudaq
import numpy as np

print("Setting up CUDA-Q with GPU...")
num_gpus = cudaq.num_available_gpus()
if num_gpus > 0:
    cudaq.set_target('nvidia')
    print(f"✓ GPU backend set ({num_gpus} GPUs)")
else:
    cudaq.set_target('qpp-cpu')
    print("Using CPU backend")

# Define circuit similar to the real one
N_QUBITS = 6
N_LAYERS = 3

@cudaq.kernel
def test_circuit(feature_angles: list[float], params: list[float]):
    qubits = cudaq.qvector(N_QUBITS)
    for i in range(N_QUBITS):
        if i < len(feature_angles):
            ry(feature_angles[i], qubits[i])
    
    param_idx = 0
    for layer in range(N_LAYERS):
        for q in range(N_QUBITS):
            ry(params[param_idx], qubits[q])
            param_idx += 1
            rz(params[param_idx], qubits[q])
            param_idx += 1
        for q in range(N_QUBITS - 1):
            cx(qubits[q], qubits[q + 1])
        if layer == N_LAYERS - 1:
            cx(qubits[N_QUBITS - 1], qubits[0])

print("Running single prediction...")
features = np.random.uniform(0, np.pi, N_QUBITS)
params = np.random.uniform(0, 2*np.pi, N_QUBITS * 2 * N_LAYERS)

try:
    hamiltonian = cudaq.spin.z(0)
    result = cudaq.observe(test_circuit, hamiltonian, features.tolist(), params.tolist())
    exp_val = result.expectation()
    print(f"✓ Prediction succeeded: {exp_val:.4f}")
except Exception as e:
    print(f"✗ Prediction failed: {e}")

print("\nRunning batch of predictions...")
try:
    for i in range(5):
        features = np.random.uniform(0, np.pi, N_QUBITS)
        hamiltonian = cudaq.spin.z(0)
        result = cudaq.observe(test_circuit, hamiltonian, features.tolist(), params.tolist())
        exp_val = result.expectation()
        print(f"  Prediction {i+1}: {exp_val:.4f}")
except Exception as e:
    print(f"✗ Batch failed: {e}")

print("\n✓ Test complete!")
