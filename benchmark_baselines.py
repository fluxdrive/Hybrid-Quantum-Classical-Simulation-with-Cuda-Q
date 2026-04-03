"""
Benchmark Classical Baselines Against Quantum NN
================================================
Loads quantum model checkpoint and compares with classical models
"""

import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
import json
from collections import Counter

# ==============================================================================
# DATA PREPARATION (SAME AS QUANTUM NN)
# ==============================================================================

def augment_minority_class(X, y, target_ratio=0.9):
    """Balance dataset using SMOTE-like synthetic oversampling"""
    class_counts = Counter(y)
    minority_class = min(class_counts, key=class_counts.get)
    majority_class = max(class_counts, key=class_counts.get)
    
    X_minority = X[y == minority_class]
    n_synthetic = int(class_counts[majority_class] * target_ratio) - class_counts[minority_class]
    
    synthetic_X = []
    for _ in range(n_synthetic):
        idx1, idx2 = np.random.choice(len(X_minority), 2, replace=False)
        alpha = np.random.random()
        synthetic = alpha * X_minority[idx1] + (1 - alpha) * X_minority[idx2]
        synthetic_X.append(synthetic)
    
    X_aug = np.vstack([X, synthetic_X])
    y_aug = np.hstack([y, np.full(n_synthetic, minority_class)])
    
    return X_aug, y_aug

def load_and_prepare_data():
    """Load breast cancer dataset with PCA feature extraction"""
    print("\nLoading data...")
    
    data = load_breast_cancer()
    X, y = data.data, data.target
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    pca = PCA(n_components=6)
    X_selected = pca.fit_transform(X_scaled)
    
    X_augmented, y_augmented = augment_minority_class(X_selected, y, target_ratio=0.9)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X_augmented, y_augmented, test_size=0.2, random_state=42, stratify=y_augmented
    )
    
    scaler_quantum = StandardScaler()
    X_train_scaled = scaler_quantum.fit_transform(X_train)
    X_test_scaled = scaler_quantum.transform(X_test)
    
    return X_train_scaled, X_test_scaled, y_train, y_test

# ==============================================================================
# CLASSICAL BASELINES
# ==============================================================================

def train_random_forest(X_train, y_train, X_test, y_test):
    """Train Random Forest"""
    print("\n" + "="*70)
    print("RANDOM FOREST BASELINE")
    print("="*70)
    
    rf = RandomForestClassifier(n_estimators=100, max_depth=15, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    
    test_acc = rf.score(X_test, y_test)
    train_acc = rf.score(X_train, y_train)
    y_pred = rf.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f}")
    print(f"  Precision:      {precision:.4f}")
    print(f"  Recall:         {recall:.4f}")
    print(f"  F1-Score:       {f1:.4f}")
    
    return {
        'name': 'Random Forest',
        'train_acc': train_acc,
        'test_acc': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_svm(X_train, y_train, X_test, y_test):
    """Train SVM with hyperparameter tuning"""
    print("\n" + "="*70)
    print("SUPPORT VECTOR MACHINE (SVM) BASELINE")
    print("="*70)
    
    param_grid = {'C': [1, 10], 'gamma': ['scale', 0.001]}
    svm = SVC(kernel='rbf', random_state=42)
    grid_search = GridSearchCV(svm, param_grid, cv=3, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"  Best parameters: C={grid_search.best_params_['C']}, gamma={grid_search.best_params_['gamma']}")
    
    best_svm = grid_search.best_estimator_
    test_acc = best_svm.score(X_test, y_test)
    train_acc = best_svm.score(X_train, y_train)
    y_pred = best_svm.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f}")
    print(f"  Precision:      {precision:.4f}")
    print(f"  Recall:         {recall:.4f}")
    print(f"  F1-Score:       {f1:.4f}")
    
    return {
        'name': 'SVM',
        'train_acc': train_acc,
        'test_acc': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

def train_xgboost(X_train, y_train, X_test, y_test):
    """Train XGBoost"""
    try:
        import xgboost as xgb
        
        print("\n" + "="*70)
        print("XGBOOST BASELINE")
        print("="*70)
        
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, 
                                       random_state=42, n_jobs=-1)
        xgb_model.fit(X_train, y_train)
        
        test_acc = xgb_model.score(X_test, y_test)
        train_acc = xgb_model.score(X_train, y_train)
        y_pred = xgb_model.predict(X_test)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        
        print(f"  Train Accuracy: {train_acc:.4f}")
        print(f"  Test Accuracy:  {test_acc:.4f}")
        print(f"  Precision:      {precision:.4f}")
        print(f"  Recall:         {recall:.4f}")
        print(f"  F1-Score:       {f1:.4f}")
        
        return {
            'name': 'XGBoost',
            'train_acc': train_acc,
            'test_acc': test_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    except ImportError:
        print("\n⚠ XGBoost not installed. Skipping XGBoost baseline.")
        return None

def train_classical_nn(X_train, y_train, X_test, y_test):
    """Train Classical Neural Network"""
    print("\n" + "="*70)
    print("CLASSICAL NEURAL NETWORK BASELINE")
    print("="*70)
    
    clf = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, random_state=42)
    clf.fit(X_train, y_train)
    
    test_acc = clf.score(X_test, y_test)
    train_acc = clf.score(X_train, y_train)
    y_pred = clf.predict(X_test)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy:  {test_acc:.4f}")
    print(f"  Precision:      {precision:.4f}")
    print(f"  Recall:         {recall:.4f}")
    print(f"  F1-Score:       {f1:.4f}")
    
    return {
        'name': 'Classical NN',
        'train_acc': train_acc,
        'test_acc': test_acc,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

# ==============================================================================
# MAIN BENCHMARKING
# ==============================================================================

def main():
    print("\n" + "="*70)
    print("CLASSICAL BASELINE BENCHMARKING")
    print("="*70)
    
    # Load data
    X_train, X_test, y_train, y_test = load_and_prepare_data()
    print(f"Train set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Train baselines
    results = {}
    
    results['Random Forest'] = train_random_forest(X_train, y_train, X_test, y_test)
    results['SVM'] = train_svm(X_train, y_train, X_test, y_test)
    results['Classical NN'] = train_classical_nn(X_train, y_train, X_test, y_test)
    
    xgb_result = train_xgboost(X_train, y_train, X_test, y_test)
    if xgb_result:
        results['XGBoost'] = xgb_result
    
    # Load quantum NN results
    try:
        with open('qnn_best_checkpoint.json', 'r') as f:
            qnn_checkpoint = json.load(f)
        qnn_test_acc = qnn_checkpoint['test_accuracy']
        results['Quantum NN'] = {'name': 'Quantum NN', 'test_acc': qnn_test_acc}
    except FileNotFoundError:
        print("\n⚠ Quantum NN checkpoint not found")
    
    # Print comparison table
    print("\n" + "="*70)
    print("MODEL PERFORMANCE COMPARISON")
    print("="*70)
    print(f"\n{'Model':<20} {'Train Acc':<12} {'Test Acc':<12} {'Precision':<12} {'Recall':<12} {'F1-Score':<12}")
    print("-" * 80)
    
    for model_name in ['Quantum NN', 'Random Forest', 'SVM', 'Classical NN', 'XGBoost']:
        if model_name not in results:
            continue
        
        metrics = results[model_name]
        train = metrics.get('train_acc', None)
        test = metrics.get('test_acc', None)
        prec = metrics.get('precision', None)
        rec = metrics.get('recall', None)
        f1 = metrics.get('f1', None)
        
        train_str = f"{train:.4f}" if train is not None else "N/A"
        test_str = f"{test:.4f}" if test is not None else "N/A"
        prec_str = f"{prec:.4f}" if prec is not None else "N/A"
        rec_str = f"{rec:.4f}" if rec is not None else "N/A"
        f1_str = f"{f1:.4f}" if f1 is not None else "N/A"
        
        print(f"{model_name:<20} {train_str:<12} {test_str:<12} {prec_str:<12} {rec_str:<12} {f1_str:<12}")
    
    print("-" * 80)
    
    # Save results
    results_to_save = {}
    for model_name, metrics in results.items():
        results_to_save[model_name] = {
            'test_acc': float(metrics.get('test_acc', 0)) if metrics.get('test_acc') is not None else None,
            'train_acc': float(metrics.get('train_acc', 0)) if metrics.get('train_acc') is not None else None,
            'precision': float(metrics.get('precision', 0)) if metrics.get('precision') is not None else None,
            'recall': float(metrics.get('recall', 0)) if metrics.get('recall') is not None else None,
            'f1': float(metrics.get('f1', 0)) if metrics.get('f1') is not None else None,
        }
    
    with open('benchmark_results.json', 'w') as f:
        json.dump(results_to_save, f, indent=2)
    
    print("\n✓ Benchmark results saved to 'benchmark_results.json'")
    
    # Summary
    best_model = max(results.items(), key=lambda x: x[1].get('test_acc', 0))
    print(f"\n✓ Best Model: {best_model[0]} with {best_model[1].get('test_acc', 0):.4f} test accuracy")
    
    print("\n" + "="*70)

if __name__ == "__main__":
    main()
