# Deep-GPCM Project Summary

## Overview

This document summarizes the work completed on implementing and improving ordinal regression models for knowledge tracing, specifically focusing on CORAL-GPCM integration.

## Key Accomplishments

### 1. Fixed Prediction Methods for Ordinal Metrics

**Issue**: QWK (Quadratic Weighted Kappa) was returning NaN with soft predictions.

**Solution**: Implemented hard (argmax) predictions for all ordinal metrics since they require discrete values.

**Implementation**:
- Updated `METRIC_METHOD_MAP` in `utils/metrics.py` to use 'hard' predictions
- Modified `compute_predictions` functions to support different prediction methods
- Achieved consistent metric computation across all models

### 2. Focal Loss Training

Implemented focal loss to handle class imbalance in ordinal categories.

**Results on synthetic_OC dataset**:
- **Deep-GPCM**: QWK = 0.6496, Accuracy = 51.73%
- **Attention-GPCM (Focal)**: Unstable training, QWK = 0.4636
- **Attention-GPCM (CE)**: QWK = 0.6477, Accuracy = 52.75%
- **CORAL-GPCM (Focal)**: QWK = 0.6743, Accuracy = 53.00%

### 3. Combined Loss Implementation

Created a sophisticated combined loss function integrating:
- Focal loss (for class imbalance)
- QWK loss (for ordinal accuracy)
- CORAL loss (for rank consistency)

**Best Result (Flawed CORAL-GPCM)**:
- Combined loss (0.4 focal + 0.2 QWK + 0.4 CORAL)
- QWK = 0.6831, Accuracy = 53.67%

### 4. Discovered Critical CORAL Implementation Flaw

**Issue**: The original CORAL-GPCM implementation was mathematically equivalent to GPCM.

**Analysis**:
```python
# FLAWED: Both use the same β parameters
P_CORAL(Y=k) = CORAL_structure(α(θ - β))
P_GPCM(Y=k) = GPCM_structure(α(θ - β))
```

**Root Cause**: CORAL was using GPCM's β thresholds instead of its own τ thresholds, making the two models identical.

### 5. Implemented Proper CORAL-GPCM

**Architecture**:
- **IRT Branch**: features → α, θ, β → GPCM probabilities
- **CORAL Branch**: features → shared score → ordered thresholds τ₁ ≤ τ₂ ≤ ... ≤ τₖ
- **Integration**: Adaptive blending based on threshold distance

**Key Features**:
1. Proper separation of IRT parameters (α, θ, β) and CORAL thresholds (τ)
2. Monotonic ordering enforcement for CORAL thresholds
3. Adaptive blending mechanism
4. Full gradient flow preservation

**Implementation Details**:
- Created `CoralTauHead` class for proper CORAL threshold extraction
- Implemented `CORALGPCM` class in `models/implementations/coral_gpcm_proper.py`
- Verified mathematical correctness through comprehensive testing

### 6. Training Results for Proper CORAL-GPCM

**Configuration**:
- Dataset: synthetic_OC (200 questions, 4 categories)
- Loss: Combined (0.4 focal + 0.2 QWK + 0.4 CORAL)
- Adaptive blending: Enabled
- Training: 30 epochs with cosine annealing

**Final Results**:
- **Test QWK**: 0.7381 (best result achieved)
- **Test Accuracy**: 64.61%
- **Ordinal Accuracy**: 89.09%
- **Mean Absolute Error**: 0.4866

**Per-Category Performance**:
- Category 0: 79.72% accuracy (most frequent)
- Category 1: 30.16% accuracy
- Category 2: 30.80% accuracy
- Category 3: 58.68% accuracy

## Technical Insights

### 1. Mathematical Correctness
The proper CORAL-GPCM implementation maintains distinct parameter spaces:
- IRT β parameters vary per item and timestep
- CORAL τ thresholds are global and ordered
- This distinction enables the model to capture both item-specific difficulty and global ordinal structure

### 2. Adaptive Blending
The adaptive blending mechanism dynamically adjusts the contribution of GPCM and CORAL based on:
- Threshold distance between β and τ
- Range divergence
- Threshold correlation

### 3. Training Stability
The combined loss with proper weight balancing achieved:
- Stable convergence
- Consistent improvement across epochs
- No gradient explosion issues

## Comparison Summary

| Model | Architecture | QWK | Accuracy | Notes |
|-------|-------------|-----|----------|-------|
| Deep-GPCM (Focal) | Standard GPCM | 0.6496 | 51.73% | Baseline |
| Attention-GPCM (CE) | GPCM + Attention | 0.6477 | 52.75% | Stable |
| CORAL-GPCM (Flawed) | Incorrect fusion | 0.6831 | 53.67% | Best flawed |
| **CORAL-GPCM (Proper)** | **Correct separation** | **0.7381** | **64.61%** | **Best overall** |

## Key Learnings

1. **Ordinal metrics require discrete predictions** - Soft predictions are incompatible with QWK
2. **Mathematical validation is crucial** - The original CORAL implementation looked correct but was fundamentally flawed
3. **Combined losses work well** - Balancing focal, QWK, and CORAL losses improved performance
4. **Proper architecture matters** - Correct separation of IRT and CORAL parameters was essential
5. **Adaptive mechanisms help** - Dynamic blending outperformed fixed weights

## Future Directions

1. Test on larger, real-world datasets
2. Explore more sophisticated blending mechanisms
3. Investigate interpretability of learned τ thresholds
4. Compare with other ordinal regression approaches
5. Extend to multi-skill knowledge tracing scenarios