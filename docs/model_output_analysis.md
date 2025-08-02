# Model Output Format Analysis

## Phase 0: Pre-Implementation Analysis

### 1. Model Output Formats

All models return a tuple of 4 tensors:
```python
(student_abilities, item_thresholds, discrimination_params, gpcm_probs)
```

#### DeepGPCM (Base Model)
- **student_abilities**: `(batch_size, seq_len)` - Student ability parameter θ
- **item_thresholds**: `(batch_size, seq_len, n_cats-1)` - Item threshold parameters β
- **discrimination_params**: `(batch_size, seq_len)` - Discrimination parameters α
- **gpcm_probs**: `(batch_size, seq_len, n_cats)` - Final categorical probabilities

#### AttentionGPCM
- Same output format as DeepGPCM
- Internal processing uses attention refinement but output structure identical

#### HybridCORALGPCM
- Same output format as DeepGPCM
- **gpcm_probs** are blended: `(1-w)*gpcm + w*coral`
- Stores `_last_coral_logits` internally for loss computation

#### EnhancedCORALGPCM
- Same output format as DeepGPCM
- May have adaptively coupled thresholds if enabled
- May use adaptive blending if enabled
- Stores `_last_coral_logits` internally

### 2. Current Evaluation Pipeline Flow

```
1. Model Forward Pass
   ↓
2. Extract gpcm_probs (4th element of tuple)
   ↓
3. Flatten and Apply Mask (remove padding)
   ↓
4. Argmax to get predictions
   ↓
5. Compute Metrics with hard predictions
```

#### Key Code Snippets:

**Training (train.py:336)**:
```python
y_pred = all_predictions.argmax(dim=-1)
eval_metrics = compute_metrics(all_targets, y_pred, all_predictions, n_cats=model.n_cats)
```

**Evaluation (evaluate.py:350)**:
```python
y_pred = all_predictions.argmax(dim=-1)
eval_metrics = compute_metrics(all_targets, y_pred, all_predictions, n_cats=n_cats)
```

**Metrics (metrics.py:327)**:
```python
valid_preds = valid_probs.argmax(dim=-1)
```

### 3. Current Metric System

The `compute_metrics` function accepts:
- `y_true`: Ground truth labels (hard)
- `y_pred`: Predicted labels (hard, from argmax)
- `y_prob`: Probability distributions (optional)
- `n_cats`: Number of categories

Metrics that use hard predictions:
- categorical_accuracy
- ordinal_accuracy
- adjacent_accuracy
- quadratic_weighted_kappa
- mean_absolute_error
- confusion_matrix metrics
- per-category breakdowns

Metrics that use probabilities:
- cross_entropy
- mean_confidence
- entropy metrics
- expected_calibration_error

### 4. Probability Types in System

1. **GPCM Probabilities**: Always categorical, sum to 1
2. **CORAL Cumulative Logits**: `(batch, seq, n_cats-1)`
3. **CORAL Cumulative Probs**: `sigmoid(logits)`
4. **CORAL Categorical Probs**: Converted via `_cumulative_to_categorical`

### 5. Key Observations

1. **All predictions currently use argmax** - No soft predictions
2. **Metrics are computed after argmax** - Loss of probability information
3. **CORAL models convert to categorical before blending** - Unified probability space
4. **No threshold-based predictions** - Despite ordinal nature
5. **Expected value predictions not implemented** - Could improve ordinal metrics

### 6. Edge Cases to Consider

1. **All-zero probabilities**: Need epsilon smoothing
2. **Single response category**: Degenerate case
3. **Extreme ability values**: Numerical stability
4. **Tied probabilities**: Argmax behavior undefined
5. **Padding tokens**: Must be excluded correctly
6. **Very long sequences**: Memory and stability

### 7. Implementation Requirements

For unified prediction system:
1. Extract probabilities correctly from model tuple
2. Handle both categorical and cumulative formats
3. Implement three prediction methods
4. Maintain backward compatibility
5. Support all model architectures
6. Handle edge cases gracefully