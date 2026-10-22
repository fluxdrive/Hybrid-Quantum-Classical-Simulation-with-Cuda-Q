#!/usr/bin/env python3
"""Quick GPU connectivity test"""

import sys
print("Testing GPU connectivity...")

try:
    print("1. Importing cudaq...")
    import cudaq
    print("   ✓ cudaq imported")
except Exception as e:
    print(f"   ✗ Failed to import cudaq: {e}")
    sys.exit(1)

try:
    print("2. Checking available GPUs...")
    num_gpus = cudaq.num_available_gpus()
    print(f"   ✓ Available GPUs: {num_gpus}")
except Exception as e:
    print(f"   ✗ Failed to check GPUs: {e}")
    sys.exit(1)

if num_gpus > 0:
    try:
        print("3. Setting NVIDIA backend...")
        cudaq.set_target('nvidia')
        print("   ✓ NVIDIA backend set")
    except Exception as e:
        print(f"   ✗ Failed to set NVIDIA backend: {e}")
        sys.exit(1)
else:
    print("3. No GPUs available, using CPU")
    try:
        cudaq.set_target('qpp-cpu')
        print("   ✓ CPU backend set")
    except Exception as e:
        print(f"   ✗ Failed to set CPU backend: {e}")
        sys.exit(1)

try:
    print("4. Testing simple quantum circuit...")
    import numpy as np
    
    @cudaq.kernel
    def simple_circuit():
        qubits = cudaq.qvector(2)
        ry(0.5, qubits[0])
        ry(0.5, qubits[1])
        cx(qubits[0], qubits[1])
    
    hamiltonian = cudaq.spin.z(0)
    result = cudaq.observe(simple_circuit, hamiltonian)
    exp_val = result.expectation()
    print(f"   ✓ Expectation value: {exp_val:.4f}")
except Exception as e:
    print(f"   ✗ Failed to run quantum circuit: {e}")
    sys.exit(1)

print("\n✓ All GPU tests passed!")
