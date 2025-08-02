# Comprehensive Metrics Analysis Summary

## Key Findings

### 1. QWK is an Ordinal Metric
- **Quadratic Weighted Kappa (QWK)** measures agreement between ordinal predictions, penalizing disagreements quadratically based on distance
- As an ordinal metric, QWK can theoretically benefit from ordinal-aware predictions
- However, empirical results show performance depends on model calibration

### 2. Both Hard and Threshold Predictions Produce Discrete Labels
Since both methods output discrete category labels, **any metric can be computed with either method**:
- Classification metrics (accuracy, precision, recall)
- Ordinal metrics (QWK, ordinal accuracy)
- Agreement metrics (Cohen's kappa)

### 3. Metric Performance Summary

| Metric | Best Method | Value | Notes |
|--------|-------------|-------|-------|
| Categorical Accuracy | Hard | 0.4606 | Hard predictions consistently better |
| QWK | Hard | 0.4367 | Despite being ordinal, hard performs better |
| Cohen's Kappa | Hard | 0.2241 | Non-ordinal agreement metric |
| Ordinal Accuracy | Threshold (liberal) | 0.7415 | Threshold shines for ordinal accuracy |
| Spearman Correlation | Soft | 0.5347 | Continuous predictions capture ranking |
| Kendall's Tau | Soft | 0.4085 | Continuous predictions better |
| MAE | Soft | 0.9073 | Regression metric needs continuous values |

### 4. Why Hard Predictions Outperform Threshold for QWK

Analysis reveals the model produces **flat/uncertain probability distributions**:
- Average max probability: 0.391 (quite low)
- High entropy: 1.287 (near maximum of 1.386)
- P(Y > 0) < 0.5 only 14.59% of the time
- P(Y > 1) < 0.5 for 68.57% of samples

This causes threshold predictions to be heavily biased toward category 1:
- Category 0: 2,203 predictions (14.6%)
- Category 1: 12,892 predictions (85.4%)
- Category 2: 0 predictions
- Category 3: 0 predictions

### 5. Model Calibration is Critical

The effectiveness of threshold predictions depends on model calibration:

**Well-calibrated model** (sharp, confident distributions):
- Threshold predictions can improve ordinal metrics
- Better respect for ordinal structure
- More nuanced predictions at boundaries

**Poorly calibrated model** (flat, uncertain distributions):
- Hard predictions perform better
- Threshold predictions collapse to middle categories
- Argmax still finds the highest probability

### 6. Practical Recommendations

1. **For most metrics**: Use the default mapping in METRIC_METHOD_MAP
2. **For QWK specifically**: Consider model calibration
   - Early training / uncertain model → Use hard predictions
   - Well-trained / confident model → Try threshold predictions
3. **For ordinal accuracy**: Threshold predictions usually better
4. **For correlations**: Always use soft predictions

### 7. Configuration Impact

Different threshold configurations offer trade-offs:
- **Conservative (0.8, 0.6, 0.4)**: Biased toward lower categories
- **Liberal (0.3, 0.3, 0.3)**: Biased toward higher categories  
- **Median (0.5, 0.5, 0.5)**: Balanced but can collapse to middle
- **Adaptive**: Data-driven but requires tuning

## Conclusion

The unified prediction system successfully provides three prediction methods (hard, soft, threshold) that can be used flexibly. While the theoretical mapping (e.g., QWK → threshold) makes sense, empirical performance depends heavily on model characteristics, particularly calibration. The system allows easy experimentation to find the best method for each specific model and metric.