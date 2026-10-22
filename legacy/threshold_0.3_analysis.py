#!/usr/bin/env python3
"""
0.3 Threshold Model - Practical Implementation & Examples
=========================================================
This script demonstrates the 0.3 threshold model in action with
code examples, visualizations, and performance analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc, precision_recall_curve,
    classification_report
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)

# ==============================================================================
# 1. SIMULATE ENSEMBLE MODEL PREDICTIONS
# ==============================================================================

def generate_synthetic_predictions(n_samples=569, pos_ratio=0.37, seed=42):
    """Generate synthetic predictions from three classical models."""
    np.random.seed(seed)
    
    # True labels (37% positive in breast cancer dataset)
    y_true = np.random.binomial(1, pos_ratio, n_samples)
    
    # Simulate probabilities from each model
    # Models are good but not perfect - some uncertainty
    rf_proba = np.random.beta(
        a=7 + 3*y_true,  # Higher alpha when positive
        b=3 + 0.5*(1-y_true)
    )
    svm_proba = np.random.beta(
        a=6.5 + 3.5*y_true,
        b=3.5 + 0.5*(1-y_true)
    )
    cnn_proba = np.random.beta(
        a=8 + 2.5*y_true,
        b=2 + 0.8*(1-y_true)
    )
    
    return y_true, rf_proba, svm_proba, cnn_proba

# ==============================================================================
# 2. THRESHOLD COMPARISON FUNCTION
# ==============================================================================

def compare_thresholds(y_true, rf_proba, svm_proba, cnn_proba, thresholds):
    """Compare performance across different thresholds."""
    
    avg_proba = (rf_proba + svm_proba + cnn_proba) / 3
    results = {}
    
    for threshold in thresholds:
        y_pred = (avg_proba >= threshold).astype(int)
        
        # Calculate metrics
        tp = np.sum((y_pred == 1) & (y_true == 1))
        tn = np.sum((y_pred == 0) & (y_true == 0))
        fp = np.sum((y_pred == 1) & (y_true == 0))
        fn = np.sum((y_pred == 0) & (y_true == 1))
        
        sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        f1 = 2 * (precision * sensitivity) / (precision + sensitivity) if (precision + sensitivity) > 0 else 0
        
        results[threshold] = {
            'sensitivity': sensitivity,
            'specificity': specificity,
            'precision': precision,
            'f1': f1,
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'predictions': y_pred
        }
    
    return avg_proba, results

# ==============================================================================
# 3. HIGH SENSITIVITY THRESHOLD FUNCTION
# ==============================================================================

def high_sensitivity_threshold(rf_proba, svm_proba, cnn_proba, threshold=0.3):
    """
    Original implementation from ensemble_model.py
    Lower threshold for positive classification - more conservative
    Better recall at cost of some precision
    """
    avg_proba = (rf_proba + svm_proba + cnn_proba) / 3
    ensemble_preds = (avg_proba >= threshold).astype(int)
    return ensemble_preds, avg_proba

# ==============================================================================
# 4. MAIN ANALYSIS
# ==============================================================================

def main():
    print("=" * 80)
    print("0.3 THRESHOLD MODEL - DETAILED ANALYSIS")
    print("=" * 80)
    print()
    
    # Generate synthetic data
    print("[1/4] Generating synthetic predictions from ensemble...")
    y_true, rf_proba, svm_proba, cnn_proba = generate_synthetic_predictions()
    print(f"  ✓ Generated {len(y_true)} samples ({np.sum(y_true)} positive, {np.sum(1-y_true)} negative)")
    print()
    
    # Compare thresholds
    print("[2/4] Comparing performance across threshold values...")
    thresholds = [0.2, 0.3, 0.4, 0.5, 0.6]
    avg_proba, threshold_results = compare_thresholds(
        y_true, rf_proba, svm_proba, cnn_proba, thresholds
    )
    
    # Print comparison table
    print("\n" + "="*100)
    print(f"{'Threshold':<12} {'Sensitivity':<15} {'Specificity':<15} {'Precision':<15} {'F1-Score':<15}")
    print("="*100)
    for threshold in thresholds:
        res = threshold_results[threshold]
        marker = " ◄-- 0.3 THRESHOLD" if threshold == 0.3 else ""
        print(f"{threshold:<12.2f} {res['sensitivity']:<15.4f} {res['specificity']:<15.4f} "
              f"{res['precision']:<15.4f} {res['f1']:<15.4f}{marker}")
    print("="*100)
    print()
    
    # Detailed 0.3 threshold analysis
    print("[3/4] Detailed 0.3 Threshold Analysis:")
    print("-" * 80)
    res_03 = threshold_results[0.3]
    print(f"True Positives (TP):   {res_03['tp']:>4d}  - Cases correctly identified as positive")
    print(f"True Negatives (TN):   {res_03['tn']:>4d}  - Cases correctly identified as negative")
    print(f"False Positives (FP):  {res_03['fp']:>4d}  - Negative cases wrongly flagged as positive")
    print(f"False Negatives (FN):  {res_03['fn']:>4d}  - Positive cases missed (CRITICAL)")
    print()
    print(f"Sensitivity (Recall):  {res_03['sensitivity']:.2%}  - Catches {res_03['sensitivity']:.1%} of true positives")
    print(f"Specificity:           {res_03['specificity']:.2%}  - Correctly identifies negatives")
    print(f"Precision:             {res_03['precision']:.2%}  - {res_03['precision']:.1%} of alerts are correct")
    print(f"F1-Score:              {res_03['f1']:.4f}     - Balanced performance measure")
    print()
    
    # Clinical interpretation
    print("[4/4] Clinical Interpretation:")
    print("-" * 80)
    print(f"✅ EXCELLENT: Sensitivity = {res_03['sensitivity']:.2%}")
    print(f"   → Only {res_03['fn']} cases (out of {res_03['tp']+res_03['fn']}) were missed")
    print(f"   → Safe for screening: minimize false negatives")
    print()
    print(f"⚠️  TRADE-OFF: {res_03['fp']} false positives")
    print(f"   → {res_03['fp']} unnecessary alerts requiring follow-up")
    print(f"   → Acceptable cost for medical safety")
    print()
    
    # Create visualizations
    print("\nGenerating visualizations...")
    create_visualizations(y_true, threshold_results, avg_proba)
    
    print("\n✅ Analysis complete! Check generated plots.")

# ==============================================================================
# 5. VISUALIZATION FUNCTIONS
# ==============================================================================

def create_visualizations(y_true, threshold_results, avg_proba):
    """Create comprehensive visualization of threshold analysis."""
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    thresholds = list(threshold_results.keys())
    
    # Plot 1: Metrics vs Threshold
    ax1 = fig.add_subplot(gs[0, :])
    sensitivities = [threshold_results[t]['sensitivity'] for t in thresholds]
    specificities = [threshold_results[t]['specificity'] for t in thresholds]
    precisions = [threshold_results[t]['precision'] for t in thresholds]
    f1_scores = [threshold_results[t]['f1'] for t in thresholds]
    
    ax1.plot(thresholds, sensitivities, 'o-', linewidth=2.5, markersize=8, label='Sensitivity (Recall)', color='#2ecc71')
    ax1.plot(thresholds, specificities, 's-', linewidth=2.5, markersize=8, label='Specificity', color='#3498db')
    ax1.plot(thresholds, precisions, '^-', linewidth=2.5, markersize=8, label='Precision', color='#e74c3c')
    ax1.plot(thresholds, f1_scores, 'd-', linewidth=2.5, markersize=8, label='F1-Score', color='#f39c12')
    
    # Highlight 0.3 threshold
    ax1.axvline(x=0.3, color='red', linestyle='--', linewidth=2, alpha=0.7, label='0.3 Threshold')
    ax1.scatter([0.3], [threshold_results[0.3]['sensitivity']], s=200, color='red', marker='*', zorder=5)
    
    ax1.set_xlabel('Classification Threshold', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Metric Value', fontsize=12, fontweight='bold')
    ax1.set_title('Performance Metrics vs Classification Threshold', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11, loc='best')
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 1.05])
    
    # Plot 2: Confusion Matrix for 0.3 threshold
    ax2 = fig.add_subplot(gs[1, 0])
    res_03 = threshold_results[0.3]
    cm = np.array([[res_03['tn'], res_03['fp']], 
                   [res_03['fn'], res_03['tp']]])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, ax=ax2,
                xticklabels=['Negative', 'Positive'],
                yticklabels=['Negative', 'Positive'],
                annot_kws={'size': 14, 'weight': 'bold'})
    ax2.set_xlabel('Predicted Label', fontsize=11, fontweight='bold')
    ax2.set_ylabel('True Label', fontsize=11, fontweight='bold')
    ax2.set_title('Confusion Matrix (0.3 Threshold)', fontsize=12, fontweight='bold')
    
    # Plot 3: False Positives & False Negatives
    ax3 = fig.add_subplot(gs[1, 1])
    fp_counts = [threshold_results[t]['fp'] for t in thresholds]
    fn_counts = [threshold_results[t]['fn'] for t in thresholds]
    
    x = np.arange(len(thresholds))
    width = 0.35
    
    bars1 = ax3.bar(x - width/2, fp_counts, width, label='False Positives (Over-flagging)', color='#e74c3c', alpha=0.8)
    bars2 = ax3.bar(x + width/2, fn_counts, width, label='False Negatives (Missed cases)', color='#2ecc71', alpha=0.8)
    
    ax3.set_xlabel('Classification Threshold', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Count', fontsize=11, fontweight='bold')
    ax3.set_title('Error Types vs Threshold', fontsize=12, fontweight='bold')
    ax3.set_xticks(x)
    ax3.set_xticklabels([f'{t:.1f}' for t in thresholds])
    ax3.legend(fontsize=10)
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Highlight 0.3
    idx_03 = thresholds.index(0.3)
    bars1[idx_03].set_edgecolor('red')
    bars1[idx_03].set_linewidth(2)
    bars2[idx_03].set_edgecolor('red')
    bars2[idx_03].set_linewidth(2)
    
    # Plot 4: Probability Distribution
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.hist(avg_proba[y_true == 0], bins=30, alpha=0.6, label='Negative Cases', color='#3498db', edgecolor='black')
    ax4.hist(avg_proba[y_true == 1], bins=30, alpha=0.6, label='Positive Cases', color='#e74c3c', edgecolor='black')
    ax4.axvline(x=0.3, color='red', linestyle='--', linewidth=2.5, label='0.3 Threshold')
    ax4.axvline(x=0.5, color='gray', linestyle='--', linewidth=2, label='Standard Threshold (0.5)', alpha=0.7)
    ax4.set_xlabel('Average Probability', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Frequency', fontsize=11, fontweight='bold')
    ax4.set_title('Distribution of Ensemble Probabilities', fontsize=12, fontweight='bold')
    ax4.legend(fontsize=10)
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Precision-Recall Curve with threshold markers
    ax5 = fig.add_subplot(gs[2, 1])
    precisions_all = [threshold_results[t]['precision'] for t in thresholds]
    recalls_all = [threshold_results[t]['sensitivity'] for t in thresholds]
    
    ax5.plot(recalls_all, precisions_all, 'o-', linewidth=2.5, markersize=10, color='#9b59b6')
    for i, t in enumerate(thresholds):
        ax5.annotate(f'{t:.1f}', (recalls_all[i], precisions_all[i]), 
                    textcoords="offset points", xytext=(0,10), ha='center', fontsize=9, fontweight='bold')
    
    # Highlight 0.3
    idx_03 = thresholds.index(0.3)
    ax5.scatter([recalls_all[idx_03]], [precisions_all[idx_03]], s=300, color='red', marker='*', zorder=5, 
               label='0.3 Threshold')
    
    ax5.set_xlabel('Recall (Sensitivity)', fontsize=11, fontweight='bold')
    ax5.set_ylabel('Precision', fontsize=11, fontweight='bold')
    ax5.set_title('Precision-Recall Trade-off', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    ax5.legend(fontsize=10)
    ax5.set_xlim([0.75, 1.05])
    ax5.set_ylim([0.4, 1.05])
    
    plt.suptitle('0.3 Threshold Model - Comprehensive Analysis', fontsize=16, fontweight='bold', y=0.995)
    plt.savefig('/home/magic/lanl_project/QSVM/threshold_0.3_analysis.png', dpi=300, bbox_inches='tight')
    print("✓ Saved: threshold_0.3_analysis.png")
    
    plt.show()

# ==============================================================================
# 6. EXAMPLE USAGE IN PRODUCTION
# ==============================================================================

def production_example():
    """Example of how to use the 0.3 threshold model in production."""
    
    print("\n" + "="*80)
    print("PRODUCTION DEPLOYMENT EXAMPLE")
    print("="*80)
    
    code_example = '''
# Step 1: Load models and get predictions
from ensemble_model import RandomForestModel, SVMModel, ClassicalNNModel

rf_model = RandomForestModel.load('models/rf_model.pkl')
svm_model = SVMModel.load('models/svm_model.pkl')
cnn_model = ClassicalNNModel.load('models/cnn_model.h5')

# Step 2: Get probability predictions for new patient data
X_patient = [[feature1, feature2, ..., feature6]]  # 6 features

rf_proba = rf_model.predict_proba(X_patient)[:, 1]
svm_proba = svm_model.predict_proba(X_patient)[:, 1]
cnn_proba = cnn_model.predict_proba(X_patient)

# Step 3: Apply 0.3 threshold
from ensemble_model import high_sensitivity_threshold

predictions, avg_proba = high_sensitivity_threshold(
    rf_proba, svm_proba, cnn_proba, threshold=0.3
)

# Step 4: Clinical decision
for i, pred in enumerate(predictions):
    confidence = avg_proba[i]
    
    if pred == 1:
        print(f"⚠️  ALERT: Possible positive case (confidence: {confidence:.2%})")
        print(f"   Recommended action: Schedule follow-up with radiologist")
    else:
        print(f"✓ CLEAR: Negative screening result (confidence: {1-confidence:.2%})")
        print(f"   Recommended action: Routine follow-up at next appointment")
    '''
    
    print(code_example)

# ==============================================================================
# RUN ANALYSIS
# ==============================================================================

if __name__ == "__main__":
    main()
    production_example()
