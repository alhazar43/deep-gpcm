# QWK Probability Agreement: Ordinal Closeness Heatmap Guide

## Overview

The QWK Probability Agreement score provides a mathematically principled measure of "degree of closeness" between predicted probability distributions and true ordinal categories. Unlike traditional accuracy metrics that treat all errors equally, this measure respects the ordinal structure where being wrong by 1 category is better than being wrong by 2 categories.

## Mathematical Foundation

### QWK-Weighted Probability Agreement

**Formula:**
```
Agreement(y_true, P) = Σ_k [QWK_weight(y_true, k) × p_k]

where QWK_weight(i,j) = 1 - (i-j)²/(K-1)²
```

**Weight Matrix for K=4 categories:**
```
Distance:    0     1     2     3
Weight:    1.0  8/9   5/9   0.0
           1.0  0.889 0.556 0.0
```

### Key Properties

1. **Range**: Always produces values in [0,1]
2. **Ordinal Structure**: Quadratic penalty for category distance
3. **Interpretability**: Higher values = better probabilistic agreement
4. **Symmetry**: Same weights regardless of direction of error
5. **Theoretical Grounding**: Based on Quadratic Weighted Kappa principles

## Usage Examples

### Basic Usage

```python
from utils.ordinal_agreement import qwk_probability_agreement

# Perfect prediction for category 0
agreement = qwk_probability_agreement(0, [1.0, 0.0, 0.0, 0.0], K=4)
# Result: 1.0

# Adjacent category error
agreement = qwk_probability_agreement(0, [0.0, 1.0, 0.0, 0.0], K=4)
# Result: 0.889

# Realistic soft prediction
agreement = qwk_probability_agreement(1, [0.1, 0.7, 0.2, 0.0], K=4)
# Result: 0.933
```

### IRT Analysis Integration

```bash
# Generate QWK agreement heatmap specifically
python irt_analysis.py --dataset synthetic_500_200_4 \
                      --analysis_types temporal \
                      --visualization_type qwk_agreement

# Compare all probabilistic measures
python irt_analysis.py --dataset synthetic_500_200_4 \
                      --analysis_types temporal
```

## Interpretation Guidelines

### Agreement Score Ranges

| Score Range | Interpretation | Educational Meaning |
|-------------|----------------|-------------------|
| 0.90 - 1.00 | Excellent | Model has strong probabilistic understanding |
| 0.75 - 0.89 | Good | Model captures ordinal relationships well |
| 0.60 - 0.74 | Moderate | Some ordinal awareness, room for improvement |
| 0.45 - 0.59 | Poor | Limited ordinal understanding |
| 0.00 - 0.44 | Very Poor | Model lacks ordinal structure comprehension |

### Heatmap Color Interpretation

**Colormap: RdYlGn (Red-Yellow-Green)**
- **Dark Green (0.9-1.0)**: Excellent probabilistic agreement
- **Light Green (0.7-0.9)**: Good agreement with ordinal awareness
- **Yellow (0.5-0.7)**: Moderate agreement, some ordinal understanding
- **Orange (0.3-0.5)**: Poor agreement, limited ordinal awareness
- **Red (0.0-0.3)**: Very poor agreement, no ordinal understanding

### Comparison with Other Measures

| Measure | Strength | Use Case |
|---------|----------|----------|
| **QWK Agreement** | Principled ordinal weighting | Primary recommendation for ordinal assessment |
| Expected Score | Simple expected value | Quick overview of predicted performance level |
| Max Probability | Confidence measure | Understanding model certainty |
| Ordinal Weighted | Linear decay | Legacy comparison (less principled) |
| Entropy | Uncertainty quantification | Measuring prediction confidence |

## Example Interpretations

### Scenario 1: High Agreement (0.95)
```
True Category: 1 (Basic)
Predicted Probabilities: [0.05, 0.85, 0.10, 0.00]
```
**Interpretation**: Model has excellent understanding. High probability on correct category with reasonable uncertainty in adjacent category. No probability mass on distant categories.

### Scenario 2: Moderate Agreement (0.67)
```
True Category: 0 (Below Basic)  
Predicted Probabilities: [0.4, 0.3, 0.2, 0.1]
```
**Interpretation**: Model shows some ordinal awareness (highest probability on correct category, decreasing with distance) but lacks confidence and precision.

### Scenario 3: Poor Agreement (0.33)
```
True Category: 1 (Basic)
Predicted Probabilities: [0.1, 0.2, 0.3, 0.4]
```
**Interpretation**: Model has inverted understanding, placing highest probability on most distant category. Suggests fundamental learning issues.

## Research Applications

### Educational Assessment
- **Adaptive Testing**: Use agreement scores to select optimal next items
- **Student Modeling**: Track learning progress through probabilistic agreement
- **Item Analysis**: Identify items where models struggle with ordinal understanding

### Model Evaluation
- **Architecture Comparison**: Compare different neural architectures' ordinal understanding
- **Training Monitoring**: Track ordinal awareness development during training
- **Ablation Studies**: Quantify impact of ordinal-aware loss functions

### Quality Assurance
- **Deployment Readiness**: Ensure models meet ordinal understanding thresholds
- **Performance Monitoring**: Detect degradation in ordinal reasoning over time
- **Data Quality**: Identify problematic response patterns in datasets

## Technical Implementation

### Heatmap Generation
The QWK agreement visualization creates heatmaps where:
- **Rows**: Individual students (selected by performance diversity)
- **Columns**: Questions/time steps in sequence
- **Cell Values**: QWK agreement scores [0,1]
- **Color Encoding**: RdYlGn colormap (Red=poor, Green=excellent)

### Integration Points
1. **IRT Analysis**: `python irt_analysis.py --visualization_type qwk_agreement`
2. **Comprehensive View**: Automatically included in 4-panel analysis
3. **Standalone Utility**: `from utils.ordinal_agreement import qwk_probability_agreement`
4. **Batch Processing**: Compare across multiple models and datasets

## Best Practices

### Interpretation
1. **Context Matters**: Consider student ability level and item difficulty
2. **Temporal Patterns**: Look for learning trends over time
3. **Model Comparison**: Use relative rankings, not just absolute scores
4. **Threshold Setting**: Establish domain-specific minimum agreement levels

### Visualization
1. **Student Selection**: Use diverse performance levels for representative view
2. **Color Scaling**: Maintain consistent [0,1] range across comparisons
3. **Annotation**: Include student performance information in labels
4. **Documentation**: Always specify the ordinal categories being measured

### Research Design
1. **Baseline Comparison**: Compare against simpler ordinal measures
2. **Validation Studies**: Correlate with external ordinal understanding measures
3. **Sensitivity Analysis**: Test robustness across different category definitions
4. **Cross-Validation**: Ensure agreement patterns generalize across data splits

## Troubleshooting

### Common Issues
1. **All Zero Scores**: Check for invalid category ranges or probability formatting
2. **Uniform High Scores**: May indicate overfitting or data leakage
3. **No Ordinal Pattern**: Model may lack ordinal structure in training
4. **Visualization Artifacts**: Ensure consistent student selection across comparisons

### Performance Considerations
- QWK agreement computation is O(K) per prediction
- Heatmap generation scales with number of students and questions
- Memory usage proportional to sequence length and batch size
- Consider subsampling for very large datasets

## References

1. Cohen, J. (1968). Weighted kappa: Nominal scale agreement provision for scaled disagreement or partial credit.
2. Fleiss, J. L., & Cohen, J. (1973). The equivalence of weighted kappa and the intraclass correlation coefficient as measures of reliability.
3. Agresti, A. (2010). Analysis of Ordinal Categorical Data (2nd ed.). Chapter 2: Ordinal measures of association.