# Deep-GPCM: Knowledge Tracing System

Production-ready Deep Generalized Partial Credit Model for polytomous response prediction with adaptive ordinal regression and temporal IRT analysis.

## Quick Start

```bash
# Activate conda environment (REQUIRED)
source ~/anaconda3/etc/profile.d/conda.sh && conda activate vrec-env

# Single training run (no cross-validation)
python train.py --model deep_gpcm --dataset synthetic_500_200_4 --epochs 30 --n_folds 0

# K-fold cross-validation training
python train.py --model deep_gpcm --dataset synthetic_500_200_4 --epochs 30 --n_folds 5

# ⭐ Adaptive hyperparameter optimization (RECOMMENDED)
python train.py --model deep_gpcm --dataset synthetic_500_200_4 --hyperopt --hyperopt_trials 50

# Complete pipeline with adaptive optimization
python main.py --models deep_gpcm attn_gpcm_learn --dataset synthetic_500_200_4 --hyperopt --hyperopt_trials 30
```

## Main Pipeline

### Core Command
```bash
# Complete pipeline with adaptive optimization (RECOMMENDED)
python main.py --dataset synthetic_500_200_4 --epochs 30 --hyperopt --hyperopt_trials 50

# All models with standard training
python main.py --dataset synthetic_500_200_4 --epochs 30 --n_folds 5

# Advanced adaptive configuration
python main.py --dataset synthetic_500_200_4 --hyperopt --adaptive_epochs 5,15,40 --adaptive_arch --adaptive_learning
```

### Pipeline Phases
```bash
# Training only with adaptive optimization
python main.py --action train --dataset synthetic_500_200_4 --epochs 30 --hyperopt

# Evaluation only  
python main.py --action evaluate --dataset synthetic_500_200_4

# Plotting only
python main.py --action plot --dataset synthetic_500_200_4

# IRT analysis only
python main.py --action irt --dataset synthetic_500_200_4

# Cleanup only
python main.py --action clean --dataset synthetic_500_200_4
python main.py --action clean --clean-all --no-confirm  # Clean all without prompts
```

## Model Types

### Available Models
| Model | Description | Parameters | Loss Function |
|-------|-------------|------------|---------------|
| `deep_gpcm` | Core DKVMN + IRT + GPCM | ~280K | Combined (CE: 0.2, QWK: 0.2, Focal: 0.6) |
| `attn_gpcm_learn` | Attention-enhanced with learnable embeddings | ~195K | Combined (CE: 0.2, QWK: 0.2, Focal: 0.6) |
| `attn_gpcm_linear` | Attention-enhanced with linear decay embeddings | ~195K | Combined (CE: 0.2, QWK: 0.2, Focal: 0.6) |
| `stable_temporal_attn_gpcm` | Production-ready temporal attention model | ~200K | Combined (CE: 0.2, QWK: 0.2, Focal: 0.6) |

**Features:**
- 🔥 **Adaptive Hyperparameter Optimization**: Intelligent epoch allocation and expanded search space
- 🎯 **Model-Aware Parameter Search**: Only searches parameters relevant to each model
- 📊 **Automated Analysis**: AI-generated insights and recommendations
- 🛡️ **Fallback System**: Safe degradation to original optimization if needed

## Training Modes

### 1. Adaptive Hyperparameter Optimization (RECOMMENDED)
```bash
# ⭐ Best practice: Adaptive optimization with intelligent epoch allocation
python train.py --model deep_gpcm --dataset synthetic_500_200_4 --hyperopt --hyperopt_trials 50

# Advanced configuration: Custom epoch allocation and expanded search
python train.py --model deep_gpcm --dataset synthetic_500_200_4 --hyperopt \
    --adaptive_epochs 5,15,40 --adaptive_arch --adaptive_learning

# Disable adaptive features (fallback to original optimization)
python train.py --model deep_gpcm --dataset synthetic_500_200_4 --hyperopt --no_adaptive

# Quick adaptive search for development
python train.py --model deep_gpcm --dataset synthetic_500_200_4 --hyperopt --hyperopt_trials 20
```

### 2. Standard Training
```bash
# Basic single model training
python train.py --model deep_gpcm --dataset synthetic_500_200_4 --epochs 30 --n_folds 0

# K-fold cross-validation
python train.py --model deep_gpcm --dataset synthetic_500_200_4 --epochs 30 --n_folds 5

# Multiple models with standard training
python train.py --models deep_gpcm attn_gpcm_learn --dataset synthetic_500_200_4 --epochs 30 --n_folds 5
```

## ⭐ Adaptive Hyperparameter Optimization

### Key Features

**🔄 Intelligent Epoch Allocation**
- **Phase 1** (5 epochs): Quick exploration for unpromising configurations
- **Phase 2** (15 epochs): Standard evaluation for most trials
- **Phase 3** (40 epochs): Full evaluation for promising configurations

**🎯 Model-Aware Parameter Search**
- **Common Parameters**: `memory_size`, `key_dim`, `value_dim`, `final_fc_dim`, `dropout_rate`
- **Attention Models**: Additional `embed_dim`, `n_heads`, `n_cycles` parameters
- **Learning Parameters**: `lr`, `weight_decay`, `batch_size`, `grad_clip`, `label_smoothing`

**🤖 Automated Analysis & Recommendations**
- Performance pattern analysis and optimization insights
- Parameter importance ranking and interaction analysis
- Actionable recommendations for next optimization steps

**🛡️ Fallback Safety System**
- Automatic fallback to original optimization if adaptive features fail
- Configurable failure thresholds and safety mechanisms

### Usage Examples

```bash
# ⭐ Recommended: Full adaptive optimization
python train.py --model deep_gpcm --dataset synthetic_500_200_4 --hyperopt --hyperopt_trials 50

# Quick development testing
python train.py --model deep_gpcm --dataset synthetic_500_200_4 --hyperopt --hyperopt_trials 10 --adaptive_epochs 3,5,10

# Advanced: Custom configuration for attention models
python train.py --model attn_gpcm_learn --dataset synthetic_500_200_4 --hyperopt \
    --adaptive_epochs 5,20,50 --adaptive_arch --adaptive_learning --hyperopt_trials 75

# Production: Multiple models with adaptive optimization
python main.py --models deep_gpcm attn_gpcm_learn attn_gpcm_linear \
    --dataset synthetic_500_200_4 --hyperopt --hyperopt_trials 100
```

## Evaluation and Analysis

### Evaluation
```bash
# Evaluate all trained models
python evaluate.py --all --dataset synthetic_OC

# Individual model evaluation
python evaluate.py --model_path save_models/best_deep_gpcm_synthetic_OC.pth --dataset synthetic_OC

# Summary of existing results
python evaluate.py --summary_only --dataset synthetic_OC
```

### Visualization
```bash
# Generate all plots from existing results for a specific dataset
python utils/plot_metrics.py --dataset synthetic_OC

# Generate plots for all available results (no dataset filter)
python utils/plot_metrics.py

# Custom plot generation with specific models
python utils/plot_metrics.py --models deep_gpcm attn_gpcm coral_gpcm_proper --dataset synthetic_OC

# Compare all six models
python utils/plot_metrics.py --models deep_gpcm attn_gpcm coral_gpcm_proper coral_gpcm_fixed test_gpcm attn_gpcm_new --dataset synthetic_OC
```

### IRT Analysis
```bash
# Complete IRT analysis (parameter recovery + temporal analysis)
python analysis/irt_analysis.py --dataset synthetic_OC

# Binary classification datasets (automatically handles 1 threshold)
python analysis/irt_analysis.py --dataset synthetic_4000_200_2 --analysis_types recovery temporal

# Temporal analysis with rank-rank heatmaps
python analysis/irt_analysis.py --dataset synthetic_OC --analysis_types temporal

# Parameter recovery analysis with enhanced visualizations
python analysis/irt_analysis.py --dataset synthetic_OC --analysis_types recovery

# Save extracted parameters
python analysis/irt_analysis.py --dataset synthetic_OC --save_params
```

## Data Generation

### New Format (Recommended)
```bash
# Generate datasets with explicit configuration (OC format only)
python utils/data_gen.py --name synthetic_4000_200_2   # 2 categories
python utils/data_gen.py --name synthetic_4000_200_3   # 3 categories
python utils/data_gen.py --name synthetic_4000_200_5   # 5 categories

# Custom sequence lengths (override automatic 1/4 ratio)
python utils/data_gen.py --name synthetic_4000_200_2 --min_seq 20 --max_seq 100
python utils/data_gen.py --name synthetic_1000_120_3 --min_seq 30   # Custom min only

# Format: synthetic_<students>_<max_questions>_<categories>
# - students: Number of student sequences
# - max_questions: Maximum question pool size  
# - categories: Number of response categories (n_cats)
# - min_seq: Defaults to max_seq/4 (minimum 10)
# - max_seq: Defaults to min(max_questions, 200)
```

### Legacy Format  
```bash
# Standard synthetic dataset (OC format only)
python utils/data_gen.py --categories 4 --students 800 --questions 50 --seed 42

# Custom dataset sizes with explicit sequence control
python utils/data_gen.py --categories 3 --students 1000 --questions 200 --min_seq 50 --max_seq 200
```

## Cleanup Utility

### Basic Usage
```bash
# Clean specific dataset
python main.py --action clean --dataset synthetic_OC

# Clean all datasets
python main.py --action clean --clean-all

# Preview what will be deleted (dry run)
python main.py --action clean --dataset synthetic_OC --dry-run

# Clean without confirmation prompt
python main.py --action clean --dataset synthetic_OC --no-confirm

# Clean without creating backup
python main.py --action clean --dataset synthetic_OC --no-backup
```

### Integration with Pipeline
```bash
# Clean before running pipeline (start fresh)
python main.py --dataset synthetic_OC --clean

# Clean only without running pipeline
python main.py --clean-only --dataset synthetic_OC

# Run multiple experiments with fresh start each time
python main.py --dataset synthetic_OC --clean --epochs 30
python main.py --dataset synthetic_4000_200_2 --clean --epochs 30
```

### Standalone Usage
```bash
# Direct utility usage
python utils/clean_res.py --dataset synthetic_OC
python utils/clean_res.py --all --dry-run
```

## Configuration Options

### Main Arguments
- `--model`: Single model to train (`deep_gpcm`, `attn_gpcm`, etc.)
- `--models`: Multiple models to train sequentially
- `--dataset`: Dataset name (e.g., `synthetic_500_200_4`, `synthetic_OC`)
- `--epochs`: Training epochs (default: 30)
- `--n_folds`: Number of folds (0=no CV, 1=single run, >1=K-fold CV, default: 5)
- `--batch_size`: Batch size (default: 64)
- `--device`: Device selection (`cuda`/`cpu`)

### Adaptive Hyperparameter Optimization Arguments
- `--hyperopt`: Enable Bayesian hyperparameter optimization
- `--hyperopt_trials`: Number of optimization trials (default: 50)
- `--hyperopt_metric`: Metric to optimize (default: `quadratic_weighted_kappa`)
- `--adaptive`: Enable adaptive optimization features (default: True)
- `--no_adaptive`: Disable adaptive optimization, use original system
- `--adaptive_epochs`: Epoch allocation strategy (default: `5,15,40`)
- `--adaptive_arch`: Enable architectural parameter search (default: True)
- `--adaptive_learning`: Enable learning parameter search (default: True)

### Loss Function Arguments
- `--loss`: Loss type (`ce`, `qwk`, `emd`, `ordinal_ce`, `combined`)
- `--ce_weight`: Weight for CE in combined loss (default: 1.0)
- `--qwk_weight`: Weight for QWK in combined loss (default: 0.5)
- `--coral_weight`: Weight for CORAL loss (default: 0.0)

### Cleanup Arguments
- `--clean`: Clean existing results before running pipeline (start fresh)
- `--clean-only`: Only clean results without running pipeline
- `--clean-all`: Clean results for all datasets
- `--dry-run`: Preview what will be deleted without deleting
- `--no-backup`: Skip creating backup before deletion
- `--no-confirm`: Skip confirmation prompt

## Environment Setup

```bash
# Activate conda environment (REQUIRED)
source ~/anaconda3/etc/profile.d/conda.sh && conda activate vrec-env

# Clean previous outputs (using new cleanup utility)
python main.py --action clean --dataset synthetic_OC  # Clean specific dataset
python main.py --action clean --clean-all            # Clean all datasets
python main.py --action clean --dataset synthetic_OC --dry-run  # Preview what will be deleted
```

## Troubleshooting

### Common Issues
- **CUDA out of memory**: Reduce batch size (`--batch_size 32`)
- **Model loading errors**: Check model paths and naming
- **Training instability**: Adjust learning rate (`--lr 0.0005` for QWK loss)
- **Environment issues**: Ensure conda environment is activated
- **Binary classification errors**: Fixed - now handles 2-category datasets correctly
- **IRT analysis skipped**: Fixed - parameter recovery works for any number of categories

### Performance Expectations
- **Training Time**: ~10-30 minutes (30 epochs, 5-fold CV, synthetic data)
- **Memory Usage**: ~2-4GB GPU memory
- **Test Samples**: 41,115 samples (excluding padding tokens)

### Best Practices
- Use cross-validation for robust model evaluation
- Start with standard models before trying adaptive variants
- Monitor gradient norms during training (should be <10)
- Use combined losses for CORAL models for better performance