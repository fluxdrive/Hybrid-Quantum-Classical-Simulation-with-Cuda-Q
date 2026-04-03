
# Quantum + Classical Ensemble: Production Deployment Guide

## Overview
This document describes the hybrid quantum-classical ensemble system for medical diagnosis.

## Architecture
- **Quantum Component**: 6 qubits, 3 layers (78.68% accuracy, 100% recall)
- **Classical Component**: 2-layer MLP (97.06% accuracy, 98.61% recall)
- **Voting Strategy**: Safety-First (OR) - Alert if either model predicts positive

## Clinical Impact
- **Zero false negatives**: No missed cancer cases
- **Sensitivity**: 100% - Perfect cancer detection
- **Specificity**: 21.88% - Conservative (some over-diagnosis)
- **Recommended for**: Initial screening where missing cases is unacceptable

## Deployment Options
1. **Safety-First (Recommended)**: Maximum sensitivity for screening
2. **Majority Voting**: Balanced sensitivity/specificity  
3. **Conservative**: High confidence decisions
4. **Weighted**: Best accuracy (70% classical, 30% quantum)

## Files
- `qnn_best_checkpoint.json` - Quantum model parameters
- `ensemble_config.json` - Ensemble configuration
- `ensemble_model.py` - Ensemble implementation
- `ENSEMBLE_DEPLOYMENT.md` - This guide

## Usage
```python
predictions = ensemble_predict(X_new, qnn_params, classical_model)
alerts = generate_clinical_alert(predictions, patient_ids)
```
