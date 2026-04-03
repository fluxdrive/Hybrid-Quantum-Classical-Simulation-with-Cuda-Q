import cudaq
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# Enable GPU
try:
    cudaq.set_target('nvidia')
    print("Target: NVIDIA GPU")
except:
    cudaq.set_target('qpp-cpu')
    print("Target: CPU (Fallback)")

# Generate simple dataset
X, y = make_classification(n_samples=100, n_features=4, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Normalize to [0, π]
X_train = (X_train - X_train.min()) / (X_train.max() - X_train.min()) * np.pi

# --- CORRECTED KERNEL ---
@cudaq.kernel
def qnn(features: list[float], params: list[float]):
    qubits = cudaq.qvector(4)
    
    # Encode features
    for i in range(4):
        ry(features[i], qubits[i])
    
    # Variational layer
    for i in range(4):
        ry(params[i], qubits[i])
        rz(params[i+4], qubits[i])
    
    # Entangle
    for i in range(3):
        cx(qubits[i], qubits[i+1])
    
    # REMOVED: mz(qubits[0]) <-- This caused the crash

# Initialize parameters
params = np.random.uniform(0, 2*np.pi, 8).tolist()

# Predict function
def predict(features, params):
    # We measure Z on qubit 0 explicitly here via Hamiltonian
    hamiltonian = cudaq.spin.z(0)
    
    # Observe calculates <Psi | Z | Psi>
    result = cudaq.observe(qnn, hamiltonian, features, params)
    
    # Expectation is between -1 and 1.
    # > 0 means closer to State |0>
    # < 0 means closer to State |1>
    return 0 if result.expectation() > 0 else 1

# Test prediction
sample = X_train[0].tolist()
prediction = predict(sample, params)
print(f"Prediction: {prediction}, True label: {y_train[0]}")