# Deep-GPCM: Generalized Partial Credit Model for Knowledge Tracing

Deep-GPCM extends traditional knowledge tracing to handle polytomous responses using the Generalized Partial Credit Model (GPCM) with specialized embedding strategies.

## Overview

Deep-GPCM extends traditional binary knowledge tracing to handle:
- **Partial Credit Responses**: Decimal scores in [0,1] 
- **Ordered Categories**: Discrete K-category responses {0, 1, 2, ..., K-1}
- **Multiple Embedding Strategies**: 4 different approaches for Q-A encoding

## Key Features

- **GPCM Implementation**: IRT-based polytomous prediction for K-category responses
- **Dual Data Formats**: PC (partial credit) and OC (ordered categories) support
- **Embedding Strategy Analysis**: Comprehensive comparison of neural embedding approaches
- **Ordinal Loss Function**: Specialized loss function respecting category ordering
- **Advanced Prediction Metrics**: New accuracy metrics designed for ordinal data
  - **Prediction Consistency Accuracy**: Measures consistency with ordinal training (cumulative method)
  - **Ordinal Ranking Accuracy**: Spearman correlation between predicted and true ordinal values
  - **Distribution Consistency Score**: Alignment between probability distributions and ordinal structure
- **Cross-Validation Pipeline**: Robust 5-fold validation with statistical analysis

## Usage

### Data Generation
```bash
# Generate both PC and OC formats with 4 categories
python data_gen.py --format both --categories 4 --students 800 --questions 50
```

### Training
```bash
# Single strategy training
python train.py --dataset synthetic_OC --embedding_strategy linear_decay --epochs 30

# Cross-validation training
python train_cv.py --dataset synthetic_OC --n_folds 5 --epochs 20
```

### Embedding Strategy Analysis
```bash
# Run comparison experiment (training + analysis + visualization)
python embedding_strategy_analysis.py --dataset synthetic_OC --epochs 10

# All 4 strategies and all 7 metrics by default
python embedding_strategy_analysis.py --dataset synthetic_OC --epochs 5

# Generate plots only from existing results (adaptive column layout)
python embedding_strategy_analysis.py --plot-only

# Specify custom metrics and strategies
python embedding_strategy_analysis.py --strategies ordered linear_decay --metrics categorical_acc ordinal_acc prediction_consistency_acc
```

### Model Evaluation
```bash
python evaluate.py --model_path save_models/best_model_synthetic_OC.pth --dataset synthetic_OC
```

## Data Formats

### Ordered Categories (OC)
```
48
26,9,25,18,6,29,...
2,0,2,0,3,3,...     # Categories: 0,1,2,3
```

### Partial Credit (PC)  
```
48
26,9,25,18,6,29,...
0.667,0.000,0.667,0.000,1.000,1.000,...  # Scores: [0,1]
```

## Architecture

```
Input: (questions, responses) → Embedding Strategy → DKVMN Memory → GPCM Predictor → K-category probabilities
```

## Embedding Strategies

1. **Ordered (2Q)**: Binary components `[correctness, score]`
2. **Unordered (KQ)**: One-hot category encoding
3. **Linear Decay (KQ)**: Triangular weights around response
4. **Adjacent Weighted (KQ)**: Focus on response and neighbors

## Current Status

**Completed**
- GPCM model implementation with multiple embedding strategies
- Ordinal loss functions and comprehensive evaluation metrics
- Unified embedding strategy analysis and visualization pipeline
- Cross-validation training with statistical analysis

**Performance Issues Identified** (See IMPROVEMENT_PLAN.md and TODO.md)
- Categorical accuracy: ~50% (random-level performance)
- Prediction consistency: ~37% (critical training/inference mismatch)
- Need for CORAL/CORN ordinal classification improvements

## Next Phase Implementation

Critical improvements needed:
1. Fix training/inference alignment (cumulative vs argmax prediction)
2. Implement CORAL framework for rank consistency
3. Advanced ordinal embeddings with distance-aware components
4. Probability calibration and uncertainty quantification

## Requirements

```
torch>=1.9.0
numpy>=1.20.0
scikit-learn>=0.24.0
matplotlib>=3.0.0
seaborn>=0.11.0
pandas>=1.3.0
tqdm>=4.60.0
```