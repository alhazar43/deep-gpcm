# Deep-GPCM: Generalized Partial Credit Model for Knowledge Tracing

Extension of Deep-IRT with support for polytomous (K-category) responses using the Generalized Partial Credit Model (GPCM).

## Overview

Deep-GPCM extends traditional binary knowledge tracing to handle:
- **Partial Credit Responses**: Decimal scores in [0,1] 
- **Ordered Categories**: Discrete K-category responses {0, 1, 2, ..., K-1}
- **Multiple Embedding Strategies**: 4 different approaches for Q-A encoding

## Key Features

- **GPCM Model**: Proper IRT-based polytomous prediction with 100% mathematical compliance
- **Dual Data Formats**: PC (partial credit) and OC (ordered categories) support
- **Multiple Embedding Strategies**: Three neural embedding approaches (ordered, unordered, linear decay)
- **Ordinal Loss Function**: Specialized loss function respecting category ordering
- **Comprehensive Evaluation**: Cross-validation, baseline comparisons, and statistical analysis
- **Performance Visualization**: Advanced plotting and analysis tools

## Usage

### Generate Synthetic Data
```bash
# Generate both PC and OC formats with 4 categories
python data_gen.py --format both --categories 4 --students 800 --questions 50

# Generate larger datasets for comprehensive analysis
python data_gen.py --format both --categories 4 --students 800 --output_dir data/large
```

### Train GPCM Model
```bash
# Basic training with specific embedding strategy
python train.py --dataset synthetic_OC --embedding_strategy linear_decay --epochs 30

# Train with different strategies for comparison
python train.py --dataset synthetic_OC --embedding_strategy ordered --loss_type ordinal
python train.py --dataset synthetic_OC --embedding_strategy unordered --n_cats 4
```

### Cross-Validation Training
```bash
# 5-fold cross-validation training
python train_cv.py --dataset synthetic_OC --n_folds 5 --epochs 20

# Cross-validation with specific strategy
python train_cv.py --dataset synthetic_OC --embedding_strategy linear_decay --n_folds 5
```

### Model Evaluation
```bash
# Comprehensive model evaluation
python evaluate.py --model_path save_models/best_model_synthetic_OC.pth --dataset synthetic_OC

# Evaluation with baseline comparisons
python evaluate.py --model_path save_models/best_model_synthetic_OC.pth --dataset synthetic_OC
```

### Strategy Comparison Analysis
```bash
# Compare all strategies on both PC and OC formats
python compare_strategies.py --dataset_path data/large --epochs 5

# Quick comparison with specific strategies
python compare_strategies.py --strategies linear_decay ordered --formats OC PC --epochs 3
```

### Comprehensive Strategy Analysis
```bash
# Full analysis across multiple K values and strategies
python analyze_strategies.py --dataset synthetic_OC --quick

# Detailed analysis with extended configurations
python analyze_strategies.py --dataset synthetic_OC
```

### Visualization and Reporting
```bash
# Create performance dashboard
python visualize.py --dataset synthetic_OC --save_path results/plots

# Generate comprehensive analysis plots
python visualize.py --dataset synthetic_OC
```

### GPCM Model Analysis
```bash
# Analyze GPCM compliance and prediction behavior
python gpcm_analysis.py
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
Input: (questions, responses) → Embedding → DKVMN → GPCM Predictor → K-category probabilities
```

### Embedding Strategies
1. **Ordered (2Q)**: `[low_component, high_component]` - Most intuitive for partial credit
2. **Unordered (KQ)**: One-hot style for each category
3. **Linear Decay (KQ)**: Triangular weights around actual response  
4. **Adjacent Weighted (KQ)**: Focus on actual + adjacent categories

## Implementation Status

### Phase 1 & 2 COMPLETED
- [x] Project setup and synthetic data generation
- [x] GPCM embedding strategies (ordered, unordered, linear decay)
- [x] GPCM predictor with proper probability calculation
- [x] Custom ordinal loss function and comprehensive metrics
- [x] Complete training pipeline with cross-validation support
- [x] Advanced evaluation framework with baseline comparisons
- [x] Strategy analysis and visualization tools
- [x] Model persistence and performance tracking

### Phase 3 PLANNED: Experimental Design & Validation
- [ ] Benchmark dataset evaluation (ASSISTments, EdNet, STATICS)
- [ ] Comparative analysis with state-of-the-art models
- [ ] Educational impact studies and interpretability analysis
- [ ] Publication-ready research contributions

### Phase 4 PLANNED: Performance Optimization
- [ ] Computational efficiency enhancements
- [ ] Advanced model accuracy improvements  
- [ ] Production deployment optimization
- [ ] Advanced research extensions

## Requirements

```
torch>=1.9.0
numpy>=1.20.0
scikit-learn>=0.24.0
tqdm>=4.60.0
```