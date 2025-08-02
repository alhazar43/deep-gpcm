# Deep-GPCM: Knowledge Tracing System

Production-ready Deep Generalized Partial Credit Model for polytomous response prediction with adaptive ordinal regression and temporal IRT analysis.

## Quick Start

```bash
# Activate conda environment (REQUIRED)
source ~/anaconda3/etc/profile.d/conda.sh && conda activate vrec-env

# Train and evaluate the three main models
python train.py --models deep_gpcm attn_gpcm coral_gpcm_proper --dataset synthetic_OC --epochs 30

# Complete pipeline with legacy dataset
python main.py --dataset synthetic_OC --epochs 30

# Complete pipeline with new format datasets
python main.py --dataset synthetic_4000_200_2 --epochs 30  # 2 categories
python main.py --dataset synthetic_4000_200_3 --epochs 30  # 3 categories
python main.py --dataset synthetic_4000_200_5 --epochs 30  # 5 categories

# Pipeline with pre-execution cleanup (start fresh)
python main.py --dataset synthetic_OC --epochs 30 --clean  # Clean before training
```

## Key Features

- ✅ **Temporal IRT Modeling**: Dynamic extraction of student abilities (θ), item discriminations (α), and thresholds (β)
- ✅ **CORAL Ordinal Regression**: Proper IRT-CORAL separation with coral_gpcm_proper model
- ✅ **Adaptive Blending**: Dynamic blending of GPCM and CORAL predictions based on threshold geometry
- ✅ **Memory Networks**: DKVMN architecture with attention-based knowledge state tracking
- ✅ **Comprehensive Analysis**: Full IRT parameter analysis with temporal visualization

## Main Pipeline

### Core Command
```bash
# All models with 5-fold training (no hyperparameter tuning)
python main.py --dataset synthetic_OC --epochs 30 --n_folds 5

# With cross-validation and hyperparameter tuning
python main.py --dataset synthetic_OC --epochs 30 --n_folds 5 --cv
```

**Includes**:
- Training with 5-fold cross-validation
- Model evaluation with comprehensive metrics
- Visualization generation (7 plots)
- IRT parameter analysis with temporal heatmaps

### Pipeline Phases
```bash
# Training only
python main.py --action train --dataset synthetic_OC --epochs 30

# Evaluation only  
python main.py --action evaluate --dataset synthetic_OC

# Plotting only
python main.py --action plot --dataset synthetic_OC

# IRT analysis only
python main.py --action irt --dataset synthetic_OC

# Cleanup only
python main.py --action clean --dataset synthetic_OC
python main.py --action clean --clean-all --no-confirm  # Clean all without prompts
```

## Model Types

### Available Models
| Model | Description | Parameters | Loss Function |
|-------|-------------|------------|---------------|
| `deep_gpcm` | Core DKVMN + IRT + GPCM | ~279K | Focal loss (γ=2.0) |
| `attn_gpcm` | Attention-enhanced with refinement | ~194K | Cross-entropy |
| `coral_gpcm_proper` | Proper IRT-CORAL separation | ~274K | Combined (Focal: 0.4, QWK: 0.2, CORAL: 0.4) |

**Note**: The main.py pipeline automatically configures the optimal loss function for each model type.

## Individual Component Usage

### Training
```bash
# Standard models with cross-validation
python train.py --model deep_gpcm --dataset synthetic_OC --epochs 30 --n_folds 5

# Proper CORAL-GPCM with IRT-CORAL separation
python train.py --model coral_gpcm_proper --dataset synthetic_OC --epochs 30

# Multiple models in sequence
python train.py --models deep_gpcm attn_gpcm coral_gpcm_proper --dataset synthetic_OC --epochs 30
```

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
# Generate all plots from existing results
python utils/plot_metrics.py

# Custom plot generation with specific models
python utils/plot_metrics.py --models deep_gpcm attn_gpcm coral_gpcm_proper --dataset synthetic_OC

# Compare all three main models
python utils/plot_metrics.py --models deep_gpcm attn_gpcm coral_gpcm_proper
```

### IRT Analysis
```bash
# Complete IRT analysis (parameter recovery + temporal analysis)
python analysis/irt_analysis.py --dataset synthetic_OC

# Binary classification datasets (automatically handles 1 threshold)
python analysis/irt_analysis.py --dataset synthetic_4000_200_2 --analysis_types recovery temporal

# Temporal analysis with heatmaps
python analysis/irt_analysis.py --dataset synthetic_OC --analysis_types temporal

# Parameter recovery analysis (dynamic column layout)
python analysis/irt_analysis.py --dataset synthetic_OC --analysis_types recovery

# Save extracted parameters
python analysis/irt_analysis.py --dataset synthetic_OC --save_params
```

## Installation and Setup

### Environment Setup
```bash
# Activate conda environment (REQUIRED)
source ~/anaconda3/etc/profile.d/conda.sh && conda activate vrec-env

# Clean previous outputs (using new cleanup utility)
python main.py --action clean --dataset synthetic_OC  # Clean specific dataset
python main.py --action clean --clean-all            # Clean all datasets
python main.py --action clean --dataset synthetic_OC --dry-run  # Preview what will be deleted

# Directories created automatically by scripts
```

### Data Generation

#### New Format (Recommended)
```bash
# Generate datasets with explicit configuration
python utils/data_gen.py --name synthetic_4000_200_2   # 2 categories
python utils/data_gen.py --name synthetic_4000_200_3   # 3 categories
python utils/data_gen.py --name synthetic_4000_200_5   # 5 categories

# Format: synthetic_<students>_<max_questions>_<categories>
# - students: Number of student sequences
# - max_questions: Maximum question pool size
# - categories: Number of response categories (n_cats)
```

#### Legacy Format
```bash
# Standard synthetic dataset (current default)
python utils/data_gen.py --format OC --categories 4 --students 800 --questions 50 --seed 42

# Custom dataset sizes
python data_gen.py --format OC --categories 4 --students 1000 --questions 200 --min_seq 50 --max_seq 200
```

## Performance Results

### Current Performance (Validated from /results/)
| Model | Categorical Accuracy | Quadratic Weighted Kappa | Ordinal Accuracy | Status |
|-------|---------------------|-------------------------|------------------|--------|
| **Deep-GPCM** | **53.5%** (±1.3%) | **0.643** (±0.016) | **83.1%** (±0.7%) | ✅ VALIDATED |
| **Attn-GPCM** | 55.0% | 0.694 | ~86% | ✅ WORKING |
| **CORAL-GPCM-Proper** | 61.3% | 0.655 | 84.1% | ✅ VALIDATED (Proper IRT-CORAL) |

**Source**: Cross-validation results from `/results/train/` directory with 5-fold CV.
**Note**: coral_gpcm_proper implements the correct IRT-CORAL separation with proper τ thresholds.

## Loss Functions

### Available Losses
1. **Cross-Entropy (`ce`)**: Standard baseline loss
2. **QWK Loss (`qwk`)**: Direct Quadratic Weighted Kappa optimization
3. **EMD Loss (`emd`)**: Earth Mover's Distance for ordinal data
4. **Ordinal CE (`ordinal_ce`)**: Distance-weighted cross-entropy
5. **Combined (`combined`)**: Weighted combination of multiple losses

### Usage Examples
```bash
# Direct QWK optimization
python train.py --model coral_gpcm --loss qwk --dataset synthetic_OC

# Balanced combined loss
python train.py --model coral_gpcm --loss combined --ce_weight 0.7 --qwk_weight 0.3

# CORAL-specific combined loss
python train.py --model coral_gpcm --loss combined --ce_weight 0.5 --coral_weight 0.5
```

## Architecture Overview

### Core Pipeline
```
Input → Embedding → DKVMN Memory → IRT Parameter Extraction → GPCM Layer → Predictions
```

### Key Components
- **DKVMN Memory Networks**: Dynamic key-value memory with attention mechanisms
- **IRT Parameter Extraction**: Temporal θ, α, β parameter computation
- **GPCM Layer**: Generalized Partial Credit Model for ordinal probabilities
- **CORAL Integration (coral_gpcm_proper)**: Proper IRT-CORAL separation with distinct τ thresholds
- **Adaptive Blending**: Dynamic weighting between GPCM and CORAL predictions

### CORAL-GPCM-Proper Architecture
```
Input → Embedding → DKVMN Memory → Summary Network ┬→ IRT Branch → α, θ, β → GPCM Probs ┐
                                                    └→ CORAL Branch → τ thresholds → CORAL Probs ┘→ Adaptive Blend → Final Predictions
```

**Key Innovation**: Separate IRT parameters (β) and CORAL thresholds (τ) prevent mathematical equivalence

### Directory Structure
```
models/
├── base/
│   └── base_model.py          # Base class for all models
├── implementations/
│   ├── deep_gpcm.py           # Core DeepGPCM implementation
│   ├── attention_gpcm.py      # Attention-enhanced model
│   └── coral_gpcm_proper.py   # Proper IRT-CORAL separation
├── components/
│   ├── memory_networks.py     # DKVMN architecture
│   ├── embeddings.py          # Response embedding strategies
│   ├── irt_layers.py          # IRT parameter extraction
│   ├── attention_layers.py    # Attention mechanisms
│   └── coral_layers.py        # CORAL ordinal regression
├── adaptive/
│   └── blenders.py            # Adaptive blending mechanisms
└── __init__.py                # Model factory

training/
└── ordinal_losses.py          # Specialized ordinal loss functions

analysis/
├── irt_analysis.py            # Comprehensive IRT analysis
├── extract_beta_params.py     # Parameter extraction
└── beta_extraction_report.md  # Extraction methodology

utils/
├── plot_metrics.py            # Visualization generation
└── clean_res.py               # Dataset cleanup utility
```

## Generated Outputs

### Results Structure
```
results/
├── plots/              # 7 comprehensive visualizations
│   ├── training_metrics.png
│   ├── test_metrics.png
│   ├── confusion_matrices_test.png
│   └── category_transitions_test.png
├── irt_plots/         # IRT analysis with temporal heatmaps
│   └── [dataset]/     # Dataset-specific IRT plots
├── train/             # Training metrics and histories
│   └── [dataset]/     # Dataset-specific training results
├── valid/             # Validation results
│   └── [dataset]/     # Dataset-specific validation results
└── test/              # Test evaluation results
    └── [dataset]/     # Dataset-specific test results

saved_models/          # New model storage structure
└── [dataset]/         # Dataset-specific models
    └── best_[model]_[dataset].pth
```

### Cleanup Functionality

The project includes a comprehensive cleanup utility to manage results:

```bash
# Clean specific dataset results
python utils/clean_res.py --dataset synthetic_OC

# Preview cleanup without deleting (dry run)
python utils/clean_res.py --dataset synthetic_OC --dry-run

# Clean all datasets
python utils/clean_res.py --all --no-confirm

# Clean without creating backup
python utils/clean_res.py --dataset synthetic_OC --no-backup
```

**Cleanup Features**:
- Removes all dataset-specific files across all directories
- Supports both new (`saved_models/dataset/`) and legacy (`save_models/`) structures
- Creates timestamped backups by default (disable with `--no-backup`)
- Shows confirmation prompt by default (disable with `--no-confirm`)
- Dry-run mode to preview deletions (`--dry-run`)
- Manifest file tracks backed up files

### Visualization Types
- **Training Curves**: Loss and accuracy evolution
- **Performance Metrics**: Categorical accuracy, QWK, ordinal accuracy
- **Confusion Matrices**: Model comparison matrices
- **Temporal Heatmaps**: Student ability evolution over time
- **IRT Plots**: Item Characteristic Curves, Wright maps
- **Parameter Recovery**: Dynamic layout adapts to number of response categories

## Configuration Options

### Main Arguments
- `--models`: Model types (default: all main models)
- `--dataset`: Dataset name (e.g., `synthetic_OC`, `synthetic_4000_200_2`)
- `--epochs`: Training epochs (default: 30)
- `--n_folds`: Number of folds for k-fold training (default: 5, use 0 for no folds)
- `--cv`: Enable cross-validation with hyperparameter tuning (flag)
- `--batch_size`: Batch size (default: 64)
- `--device`: Device selection (`cuda`/`cpu`)

### Cleanup Arguments
- `--clean`: Clean existing results before running pipeline (start fresh)
- `--clean-only`: Only clean results without running pipeline
- `--clean-all`: Clean results for all datasets
- `--dry-run`: Preview what will be deleted without deleting
- `--no-backup`: Skip creating backup before deletion
- `--no-confirm`: Skip confirmation prompt

### Loss Function Arguments
- `--loss`: Loss type (`ce`, `qwk`, `emd`, `ordinal_ce`, `combined`)
- `--ce_weight`: Weight for CE in combined loss (default: 1.0)
- `--qwk_weight`: Weight for QWK in combined loss (default: 0.5)
- `--coral_weight`: Weight for CORAL loss (default: 0.0)

### Data Formats

#### Dataset Naming Conventions
- **New Format**: `synthetic_<students>_<max_questions>_<categories>`
  - Example: `synthetic_4000_200_2` = 4000 students, 200 questions, 2 categories
  - Supports dynamic n_cats: 2, 3, 4, 5, etc.
- **Legacy Format**: `synthetic_OC` (4 categories), `synthetic_PC` (3 categories)

#### Response Types
- **Ordered Categories (OC)**: Discrete responses {0, 1, ..., n_cats-1}
- **Partial Credit (PC)**: Continuous scores [0, 1]

## IRT Analysis Features

### Parameter Extraction
- **Student Abilities (θ)**: Temporal evolution of student proficiency
- **Item Discriminations (α)**: How well items differentiate students
- **Item Thresholds (β)**: Category boundary difficulties

### Analysis Types
- **Parameter Recovery**: Correlation with ground truth IRT parameters
- **Temporal Analysis**: Time-series visualization of parameter evolution
- **Static Analysis**: Traditional IRT plots and statistics

### Binary Classification Support
- **Dynamic Column Layout**: Parameter recovery plots automatically adapt to number of categories
- **Binary Datasets**: Show 3 columns (θ, α, β_0) for 2-category data
- **Multi-Category**: Show 2 + n_thresholds columns for 3+ category data
- **Automatic Detection**: Handles any number of response categories (2, 3, 4, 5, etc.)

### Temporal Insights
All parameters in Deep-GPCM are time-indexed, representing:
- **Dynamic Learning**: Student abilities evolve with each interaction
- **Context Sensitivity**: Item parameters adapt to temporal context
- **Learning Trajectories**: Individual student learning paths

## Cleanup Utility

The project includes a comprehensive cleanup utility for managing experiment results:

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

### What Gets Cleaned
- `results/train/[dataset]/` - Training results
- `results/valid/[dataset]/` - Validation results  
- `results/test/[dataset]/` - Test results
- `results/plots/*_[dataset]*.png` - Plot files
- `results/irt_plots/[dataset]/` - IRT plot directories
- `saved_models/[dataset]/` - Model files (new structure)
- `save_models/*_[dataset]*.pth` - Legacy model files

### Safety Features
- **Backups**: Creates timestamped backups in `backups/` directory (default)
- **Confirmation**: Prompts for confirmation before deletion (default)
- **Dry Run**: Preview mode shows what would be deleted
- **Manifest**: Backup includes manifest.json listing all backed up files

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

---

## Critical Issues

### ❌ BLOCKING: CORAL Design Flaw
**Problem**: CORAL uses β parameters instead of τ thresholds, making CORAL and GPCM identical.

**Impact**: 
- All CORAL-based models produce invalid results
- Adaptive blending provides no benefit (blending identical systems)
- Performance claims for CORAL variants are unreliable

**Status**: 📋 FIX PLANNED - See `analysis/CORAL_DESIGN_FLAW_ANALYSIS.md` for detailed analysis.

### ✅ WORKING: Core Deep-GPCM
- Deep-GPCM baseline model works correctly
- IRT parameter extraction validated
- Full pipeline and evaluation system functional

---

**Deep-GPCM: Bridging Neural Memory Networks with Classical Psychometrics**