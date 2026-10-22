#!/usr/bin/env python3
"""Diagnostic script to check CUDA-Q backend and GPU status"""

import cudaq
import subprocess
import sys

print("=" * 70)
print("CUDA-Q BACKEND DIAGNOSTICS")
print("=" * 70)

# Check GPU availability
print("\n1. GPU AVAILABILITY:")
try:
    num_gpus = cudaq.num_available_gpus()
    print(f"   Available GPUs: {num_gpus}")
except Exception as e:
    print(f"   Error checking GPUs: {e}")

# Check current target
print("\n2. CURRENT BACKEND TARGET:")
try:
    print(f"   Active target: {cudaq.get_target()}")
except Exception as e:
    print(f"   Error: {e}")

# Try setting NVIDIA backend
print("\n3. ATTEMPTING NVIDIA BACKEND:")
try:
    cudaq.set_target('nvidia')
    print(f"   ✓ Successfully set to: {cudaq.get_target()}")
except Exception as e:
    print(f"   ✗ Failed to set NVIDIA: {e}")
    print("   Trying CPU backend...")
    try:
        cudaq.set_target('qpp-cpu')
        print(f"   ✓ Fallback to: {cudaq.get_target()}")
    except Exception as e2:
        print(f"   ✗ Failed: {e2}")

# Check nvidia-smi availability
print("\n4. NVIDIA-SMI STATUS:")
try:
    result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,driver_version', '--format=csv,noheader'],
                          capture_output=True, text=True, timeout=5)
    if result.returncode == 0:
        for line in result.stdout.strip().split('\n'):
            print(f"   GPU: {line}")
    else:
        print(f"   nvidia-smi returned error code {result.returncode}")
except FileNotFoundError:
    print("   ✗ nvidia-smi not found in PATH")
except Exception as e:
    print(f"   ✗ Error: {e}")

# Test a simple circuit on current backend
print("\n5. SIMPLE CIRCUIT TEST:")
try:
    @cudaq.kernel
    def test_circuit():
        qubit = cudaq.qubit()
        h(qubit)
        mz(qubit)
    
    result = cudaq.sample(test_circuit, shots_count=100)
    print(f"   ✓ Circuit executed successfully")
    print(f"   Measurement results: {dict(result)}")
except Exception as e:
    print(f"   ✗ Circuit execution failed: {e}")
    import traceback
    traceback.print_exc()

# Test async execution
print("\n6. ASYNC EXECUTION TEST:")
try:
    @cudaq.kernel
    def async_test():
        qubit = cudaq.qubit()
        rx(0.5, qubit)
        ry(0.3, qubit)
    
    hamiltonian = cudaq.spin.z(0)
    future = cudaq.observe_async(async_test, hamiltonian, shots_count=512)
    result = future.get()
    print(f"   ✓ Async execution works")
    print(f"   Expectation value: {result.expectation()}")
except Exception as e:
    print(f"   ✗ Async execution failed: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
