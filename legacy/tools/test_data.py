#!/usr/bin/env python3
"""Test data loading"""

import sys
print("Testing data loading...")

try:
    print("1. Importing libraries...")
    from sklearn.datasets import load_breast_cancer
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    import numpy as np
    print("   ✓ Imports successful")
except Exception as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

try:
    print("2. Loading dataset...")
    data = load_breast_cancer()
    X, y = data.data, data.target
    print(f"   ✓ Dataset loaded: {len(X)} samples, {X.shape[1]} features")
except Exception as e:
    print(f"   ✗ Failed to load dataset: {e}")
    sys.exit(1)

try:
    print("3. Preprocessing...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    print(f"   ✓ Scaling done")
    
    pca = PCA(n_components=6)
    X_selected = pca.fit_transform(X_scaled)
    print(f"   ✓ PCA done: variance explained = {pca.explained_variance_ratio_.sum():.4f}")
except Exception as e:
    print(f"   ✗ Preprocessing failed: {e}")
    sys.exit(1)

try:
    print("4. Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"   ✓ Train: {len(X_train)}, Test: {len(X_test)}")
except Exception as e:
    print(f"   ✗ Splitting failed: {e}")
    sys.exit(1)

print("\n✓ Data loading test passed!")
