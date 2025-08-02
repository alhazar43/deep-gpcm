# Unified Prediction System for Deep-GPCM

## Overview

The unified prediction system provides three different methods for converting probability distributions to predictions, allowing each evaluation metric to use the most appropriate method for accurate assessment.

## Prediction Methods

### 1. Hard Predictions (Argmax)
- **Method**: Select category with highest probability
- **Formula**: `pred = argmax(P(Y=k))`
- **Output**: Discrete category indices
- **Best for**: Classification accuracy, confusion matrices, precision/recall
- **Example**: `[0.7, 0.2, 0.05, 0.05] → 0`

### 2. Soft Predictions (Expected Value)
- **Method**: Probability-weighted average of categories
- **Formula**: `pred = Σ(k * P(Y=k))` for k ∈ {0, 1, ..., K-1}
- **Output**: Continuous values in [0, K-1]
- **Best for**: MAE, MSE, correlations, QWK
- **Example**: `[0.4, 0.3, 0.2, 0.1] → 1.0`

### 3. Threshold Predictions (Cumulative/Median)
- **Method**: Find the smallest category k where P(Y ≤ k) ≥ 1 - threshold
- **Formula**: Using P(Y > k), find smallest k where P(Y > k-1) < threshold[k-1]
- **Output**: Discrete category indices (ordinal-aware)
- **Best for**: Ordinal accuracy, adjacent accuracy
- **Special case**: With threshold = 0.5, computes the median of the distribution
- **Default**: Uses threshold = 0.5 for all categories (computes median)
- **Example**: With custom thresholds [0.75, 0.5, 0.25]:
  - Need P(Y > 0) ≥ 0.75 to predict category > 0 (conservative for low categories)
  - Need P(Y > 1) ≥ 0.5 to predict category > 1 (median-based)
  - Need P(Y > 2) ≥ 0.25 to predict category 3 (liberal for high categories)

## Usage

### Basic Evaluation

```python
# Standard evaluation (hard predictions only)
python evaluate.py --model_path save_models/best_model.pth

# Multi-method evaluation
python evaluate.py --model_path save_models/best_model.pth --use_multimethod_eval
```

### Advanced Options

```python
# Custom thresholds
python evaluate.py --model_path save_models/best_model.pth \
    --use_multimethod_eval --thresholds 0.8 0.6 0.4

# Adaptive thresholds (data-driven)
python evaluate.py --model_path save_models/best_model.pth \
    --use_multimethod_eval --adaptive_thresholds

# Batch evaluation
python evaluate.py --all --use_multimethod_eval
```

### In Code

```python
from utils.predictions import compute_unified_predictions, PredictionConfig
from utils.metrics import compute_metrics_multimethod

# Configure predictions
config = PredictionConfig()
config.thresholds = [0.75, 0.5, 0.25]  # Optional custom thresholds
config.adaptive_thresholds = True       # Optional adaptive thresholds

# Compute all prediction types
predictions = compute_unified_predictions(probabilities, config=config)

# Compute metrics with appropriate methods
metrics = compute_metrics_multimethod(
    y_true=targets,
    predictions=predictions,
    y_prob=probabilities,
    n_cats=4
)
```

## Metric-Method Mapping

The system automatically selects the best prediction method for each metric:

| Metric Type | Prediction Method | Rationale |
|------------|-------------------|-----------|
| Classification (accuracy, F1) | Hard | Requires discrete categories |
| Ordinal (ordinal accuracy, QWK) | Threshold* | Respects ordinal structure |
| Regression-like (MAE, correlations) | Soft | Benefits from continuous values |
| Probability-based (cross-entropy) | Raw probabilities | Uses full distribution |

*Note: While QWK is an ordinal metric, empirical results show that hard predictions may perform better when the model produces uncertain/flat probability distributions. Threshold predictions work best when the model is well-calibrated with sharp distributions.

## Benefits

1. **Comprehensive Evaluation**: Each metric uses its optimal prediction type
2. **Backward Compatible**: Default behavior unchanged
3. **Ordinal-Aware**: Threshold predictions respect category ordering
4. **Uncertainty Quantification**: Soft predictions capture prediction confidence
5. **Flexible**: Custom thresholds and adaptive strategies supported

## Implementation Details

### Files Modified/Created
- `utils/predictions.py`: Core prediction functions
- `utils/prediction_strategies.py`: Strategy pattern implementation
- `utils/metrics.py`: Multi-method metric computation
- `utils/monitoring.py`: Performance tracking
- `evaluate.py`: CLI integration

### Edge Case Handling
- Zero probability vectors: Add epsilon and renormalize
- Numerical precision: Clamp probabilities to [ε, 1-ε]
- Tied probabilities: Consistent argmax behavior
- GPU/CPU compatibility: Automatic device handling

## Performance

- Hard predictions: ~0.1ms per batch
- Soft predictions: ~0.2ms per batch  
- Threshold predictions: ~0.4ms per batch
- Overhead: <5% for complete multi-method evaluation

## Important Considerations

### Hard vs Threshold Predictions

Both hard and threshold predictions produce discrete labels, so any metric that can be computed with one can be computed with the other. Key differences:

1. **Hard predictions (argmax)**: Select the category with highest probability, even if probabilities are similar
2. **Threshold predictions**: Use cumulative probabilities to find the median, which can be biased toward middle categories when distributions are flat

### When to Use Each Method

- **Hard predictions**: Best when the model produces uncertain/flat distributions (common in early training)
- **Threshold predictions**: Best when the model is well-calibrated with sharp probability peaks
- **Soft predictions**: Always preferable for regression-like metrics (MAE, correlations)

### Model Calibration Impact

The effectiveness of threshold predictions depends heavily on model calibration:
- Well-calibrated models with confident predictions → Threshold predictions can improve ordinal metrics
- Poorly calibrated or uncertain models → Hard predictions typically perform better

## Future Enhancements

1. **Confidence intervals** for soft predictions
2. **Per-category thresholds** for imbalanced data
3. **Ensemble strategies** combining multiple methods
4. **Adaptive method selection** based on data characteristics
5. **Calibration-aware prediction selection**