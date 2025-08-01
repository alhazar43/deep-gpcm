# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Deep-GPCM is a production-ready knowledge tracing system for polytomous responses. Phase 1 enhancement completed with optimal loss functions identified: Cross-Entropy (55.0% accuracy) and Focal Loss γ=2.0 (54.6% accuracy, 83.4% ordinal). System uses linear_decay embedding strategy and comprehensive ordinal-aware evaluation metrics.

## Common Commands

### Environment Setup
```bash
# Activate conda environment (REQUIRED)
source ~/anaconda3/etc/profile.d/conda.sh && conda activate vrec-env

# Create necessary directories
mkdir -p logs save_models results/{train,test,plots}

# Note: Intel MKL threading issue is automatically fixed in all Python scripts
```

### Data Generation
```bash
# Generate standard synthetic dataset
python data_gen.py --format OC --categories 4 --students 800 --questions 50 --min_seq 10 --max_seq 50

# Generate larger dataset for stable results (current default)
python data_gen.py --format OC --categories 4 --students 800 --questions 400 --min_seq 100 --max_seq 400 --seed 42
```

### Training
```bash
# Main pipeline with automatic loss optimization (RECOMMENDED - all models included by default)
python main.py --dataset synthetic_OC --epochs 30

# Individual model training with custom loss
python train.py --model deep_gpcm --dataset synthetic_OC --epochs 30
python train.py --model coral_gpcm --loss combined --ce_weight 0.5 --coral_weight 0.5 --dataset synthetic_OC --epochs 30

# Cross-validation training (5-fold)
python train_cv.py --dataset synthetic_OC --n_folds 5 --epochs 20
```

### Evaluation and Analysis
```bash
# Complete pipeline with automatic loss optimization (RECOMMENDED - all models by default)
python main.py --dataset synthetic_OC --epochs 30 --cv_folds 5

# Individual model evaluation
python evaluate.py --model_path save_models/best_model_synthetic_OC.pth --dataset synthetic_OC

# Generate plots from existing results (no retraining needed)
python utils/plot_metrics.py

# Comprehensive analysis with advanced metrics
python comprehensive_analysis.py --save_path results/analysis
```

### Testing and Debugging
```bash
# Manual prediction testing
python manual_prediction_test.py

# Debug predictions and methods
python debug_predictions.py

# Benchmark prediction methods
python benchmark_prediction_methods.py
```

## Architecture Overview

### Core Model Components
- **DKVMN Memory Network** (`models/memory.py`): Dynamic key-value memory network for knowledge state tracking
- **Deep-GPCM Model** (`models/model.py`): Main model implementing GPCM with optimal embedding strategy
- **Optimal Loss Functions** (`models/advanced_losses.py`): Cross-Entropy and Focal Loss implementations

### Embedding Strategies 
**Optimal Strategy**: **Linear Decay (KQ)** - Triangular weights around actual response  
**Performance**: Proven optimal in systematic benchmarking, used in all Phase 1 experiments

Available strategies:
1. **Ordered (2Q)**: `[correctness_component, score_component]` - Most intuitive for partial credit
2. **Unordered (KQ)**: One-hot style encoding for each category  
3. **Linear Decay (KQ)**: Triangular weights around actual response ⭐ **OPTIMAL**
4. **Adjacent Weighted (KQ)**: Focus on actual + adjacent categories

### Data Formats
- **Ordered Categories (OC)**: Discrete responses {0, 1, 2, ..., K-1}
- **Partial Credit (PC)**: Continuous scores in [0, 1] range

### Loss Functions (Phase 1 Validated)
- **Cross-Entropy**: 55.0% categorical accuracy - Best overall performance ⭐
- **Focal Loss (γ=2.0)**: 54.6% categorical, 83.4% ordinal accuracy - Best ordinal performance ⭐
- **OrdinalLoss**: Original baseline (46.4% categorical accuracy)

## Key Implementation Details

### Model Architecture
```
Input: (questions, responses) → Embedding Strategy → DKVMN Memory → Predictor → K-category probabilities
```

### Memory Architecture
- **Key Memory Matrix**: Fixed concept embeddings learned during training
- **Value Memory Matrix**: Dynamic knowledge state updated via read/write operations
- **Attention Mechanism**: Correlation-based weighting for memory access

### Training Pipeline
- 5-fold cross-validation standard across all experiments
- Model checkpointing with best validation performance
- Comprehensive metrics tracking (categorical accuracy, ordinal accuracy, MAE, etc.)
- Advanced prediction accuracy metrics (consistency, ranking, distribution alignment)

### Evaluation Framework
- **GpcmMetrics** class with multiple prediction methods (argmax, cumulative, expected)
- Baseline comparisons with random and majority class predictors
- Statistical significance testing and confidence intervals
- ROC analysis and confusion matrix visualization

## File Organization

### Training Scripts
- `train.py`: Single-fold training with comprehensive logging
- `train_cv.py`: Cross-validation training following deep-2pl patterns
- `train_coral_enhanced.py`: CORAL integration training

### Analysis and Comparison
- `comprehensive_analysis.py`: Advanced analysis with new prediction accuracy metrics
- `compare_prediction_methods.py`: Method comparison framework
- `run_embedding_strategy_comparison.py`: Strategy benchmarking
- `benchmark_coral_integration.py`: CORAL vs GPCM performance analysis

### Utilities
- `utils/gpcm_utils.py`: Core utilities, loss functions, data loading
- `evaluation/metrics.py`: Comprehensive evaluation metrics
- `data_gen.py`: Synthetic data generation with IRT parameters

### Model Components
- `models/model.py`: Main Deep-GPCM implementation with embedding strategies
- `models/memory.py`: DKVMN memory network implementation
- `models/coral_layer.py`: CORAL ordinal classification layer
- `models/dkvmn_coral_integration.py`: CORAL-DKVMN integration

## Development Notes

### Testing Strategy
- Use `synthetic_OC` dataset for rapid prototyping (50 questions, 4 categories)
- Run embedding strategy comparisons to validate improvements
- Cross-validation ensures robust performance measurement
- Manual prediction testing for debugging specific cases

### Performance Monitoring
- Training logs saved to `logs/` directory with timestamps
- Model checkpoints in `save_models/` directory
- Results and plots organized in `results/` subdirectories
- JSON metadata tracking for experiment reproducibility

### CORAL Integration Status
- CORAL layer implemented with rank monotonicity guarantees
- Integration with DKVMN memory network completed
- Performance benchmarking shows competitive results with GPCM
- Maintains backward compatibility with existing GPCM pipeline