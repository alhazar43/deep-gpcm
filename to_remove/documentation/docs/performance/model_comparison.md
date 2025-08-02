# Model Comparison: Baseline vs CORAL vs Hybrid-CORAL

## 1. Baseline Deep-GPCM
**Architecture**: DKVMN + Standard GPCM probability layer
- Uses DKVMN memory networks for knowledge state tracking
- Standard GPCM (Generalized Partial Credit Model) for probability computation
- Linear transformation to logits, then softmax for probabilities
- No explicit ordinal constraints

**Strengths**:
- Simple and interpretable
- Good performance on categorical accuracy (57.0%)
- Best Quadratic Weighted Kappa (0.698)
- Best on correlation metrics (Kendall tau: 0.620, Spearman: 0.701)

**Weaknesses**:
- No ordinal structure enforcement
- Can predict inconsistent ordinal patterns
- Higher cross-entropy loss (2.078)

**Best for**: General polytomous response modeling without strong ordinal assumptions

## 2. CORAL Deep-GPCM
**Architecture**: DKVMN + CORAL ordinal regression layer
- Replaces GPCM probability layer with CORAL (COnsistent RAnk Logits)
- Uses K-1 binary classifiers for K categories
- Enforces rank monotonicity: P(Y≥1) ≥ P(Y≥2) ≥ ... ≥ P(Y≥K-1)
- Shared weight matrix with category-specific biases

**Strengths**:
- Best ordinal accuracy (87.0%)
- Lowest cross-entropy (0.999)
- Guarantees ordinal consistency in predictions
- Better calibrated probabilities

**Weaknesses**:
- Slightly lower categorical accuracy (55.2%)
- No explicit IRT parameter interpretation
- May be too restrictive for non-ordinal data

**Best for**: True ordinal data where response categories have inherent ordering

## 3. Hybrid-CORAL GPCM
**Architecture**: DKVMN + Hybrid approach combining CORAL structure with GPCM parameters
- Maintains CORAL's rank-consistent structure
- Preserves explicit IRT parameters (discrimination, difficulty)
- Combines ordinal constraints with psychometric interpretability
- Uses weighted combination of CORAL and GPCM predictions

**Strengths**:
- Balance between ordinal consistency and flexibility
- Maintains IRT parameter interpretability
- Can adapt to both ordinal and non-ordinal patterns
- Good compromise between baseline and pure CORAL

**Weaknesses**:
- More complex architecture
- Additional hyperparameters to tune
- Performance depends on weight balance

**Best for**: Educational assessment where both ordinal structure and IRT parameters are important

## Performance Summary (on synthetic_OC dataset)

| Metric | Baseline | CORAL | Hybrid-CORAL |
|--------|----------|-------|--------------|
| Categorical Accuracy | **57.0%** | 55.2% | ~56% (expected) |
| Ordinal Accuracy | 86.6% | **87.0%** | ~86.8% (expected) |
| QWK | **0.698** | 0.697 | ~0.697 |
| MAE | **0.591** | 0.602 | ~0.595 |
| Cross Entropy | 2.078 | **0.999** | ~1.5 |

## Recommended Usage

```bash
# Baseline - for general use
python train.py --model baseline --dataset synthetic_OC

# CORAL - for strictly ordinal data  
python train.py --model coral --dataset synthetic_OC --loss combined --ce_weight 0.7 --qwk_weight 0.3

# Hybrid-CORAL - for educational assessment with ordinal structure
python train.py --model hybrid_coral --dataset synthetic_OC --loss combined --coral_weight 0.5
```

## Key Differences

1. **Probability Computation**:
   - Baseline: Direct softmax on logits
   - CORAL: Cumulative link function with ordinal constraints
   - Hybrid: Weighted combination of both approaches

2. **Ordinal Guarantees**:
   - Baseline: None
   - CORAL: Strict rank consistency
   - Hybrid: Soft ordinal preferences

3. **Parameter Interpretability**:
   - Baseline: Full IRT parameters
   - CORAL: No explicit IRT parameters
   - Hybrid: Maintains IRT parameters with ordinal structure

4. **Use Cases**:
   - Baseline: General knowledge tracing
   - CORAL: Ordinal assessment (e.g., proficiency levels)
   - Hybrid: Educational testing with partial credit scoring