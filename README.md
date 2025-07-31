# Deep-GPCM: Knowledge Tracing System

Production-ready Deep Generalized Partial Credit Model for polytomous response prediction with CORAL ordinal regression support.

## Quick Start

```bash
# Complete pipeline (training, evaluation, plotting, IRT analysis)
python main.py --dataset synthetic_OC --epochs 30

# Train with ordinal losses for better performance
python main.py --models baseline coral --loss qwk --dataset synthetic_OC

# Compare all models with combined loss
python main.py --models baseline akvmn coral hybrid_coral --loss combined --qwk_weight 0.5
```

## Environment Setup

```bash
# Activate conda environment (REQUIRED)
source ~/anaconda3/etc/profile.d/conda.sh && conda activate vrec-env

# Clean previous outputs (optional)
rm -rf save_models/ results/ logs/ irt_plots/ irt_animations/

# Create directories (done automatically by scripts)
mkdir -p save_models results/{train,test,plots} logs irt_plots irt_animations
```

## Performance Results

| Model | Categorical Accuracy | Quadratic Weighted Kappa | Ordinal Accuracy | MAE |
|-------|---------------------|-------------------------|------------------|------|
| **Baseline** | 58.3% | 0.716 | 87.2% | 0.573 |
| **AKVMN** | 55.9% | 0.694 | 86.8% | 0.600 |
| **CORAL** | TBD | TBD | TBD | TBD |
| **Hybrid CORAL** | TBD | TBD | TBD | TBD |

*Results with corrected padding handling. CORAL models expected to improve QWK by 5-10%.*

## Usage

### Complete Pipeline
```bash
# Full pipeline with all phases (recommended)
python main.py --dataset synthetic_OC --epochs 30 --cv_folds 5

# Training only
python main.py --action train --models baseline akvmn --dataset synthetic_OC --epochs 30

# Evaluation only  
python main.py --action evaluate --models baseline akvmn --dataset synthetic_OC
```

### Individual Components
```bash
# Training with cross-validation (default: 5-fold)
python train.py --model baseline --dataset synthetic_OC --epochs 30 --n_folds 5

# Training with ordinal losses
python train.py --model coral --loss qwk --dataset synthetic_OC --epochs 30
python train.py --model coral --loss combined --ce_weight 0.7 --qwk_weight 0.3

# Batch evaluation (main models only - recommended)
python evaluate.py --all --dataset synthetic_OC

# Individual model evaluation
python evaluate.py --model_path save_models/best_baseline_synthetic_OC.pth --dataset synthetic_OC

# Show summary of existing results only
python evaluate.py --summary_only --dataset synthetic_OC

# Include CV fold models (requires corresponding fold data directories)
python evaluate.py --all --dataset synthetic_OC --include_cv_folds

# Generate visualizations from existing results
python utils/plot_metrics.py

# IRT parameter analysis with temporal heatmaps
python analysis/irt_analysis.py --dataset synthetic_OC --analysis_types recovery temporal
```

### Data Generation
```bash
# Standard synthetic dataset (current default)
python data_gen.py --format OC --categories 4 --students 800 --questions 400 --seed 42

# Custom dataset sizes
python data_gen.py --format OC --categories 4 --students 1000 --questions 200 --min_seq 50 --max_seq 200
```

## Ordinal Loss Functions

The system supports specialized loss functions for ordinal regression:

### Available Losses

1. **Cross-Entropy (`ce`)**: Standard baseline loss
2. **QWK Loss (`qwk`)**: Directly optimizes Quadratic Weighted Kappa
3. **EMD Loss (`emd`)**: Earth Mover's Distance for ordinal data
4. **Ordinal CE (`ordinal_ce`)**: Distance-weighted cross-entropy
5. **Combined (`combined`)**: Weighted combination of multiple losses

### Usage Examples

```bash
# Train with QWK loss for direct metric optimization
python train.py --model coral --loss qwk --dataset synthetic_OC

# Use combined loss for balanced optimization
python train.py --model coral --loss combined --ce_weight 0.7 --qwk_weight 0.3

# Complete pipeline with ordinal loss
python main.py --models baseline coral --loss qwk --epochs 40
```

### Tips for Ordinal Losses

- **QWK Loss**: May need lower learning rate (`--lr 0.0005`)
- **Combined Loss**: Start with CE dominant (70% CE, 30% QWK)
- **EMD Loss**: Works well for naturally ordered categories
- **CORAL models**: Best paired with ordinal losses

## Architecture

### Models
- **Baseline GPCM**: DKVMN + IRT parameter extraction + GPCM
- **AKVMN**: Enhanced attention-based DKVMN with learnable parameters
- **CORAL**: COnsistent RAnk Logits for ordinal regression with rank consistency
- **Hybrid CORAL**: Blends CORAL ordinal structure with GPCM formulation

### Core Components
```
core/
├── model.py              # DeepGPCM and AttentionGPCM
├── attention_enhanced.py # Enhanced AKVMN with learnable parameters  
├── memory_networks.py    # DKVMN architecture
├── embeddings.py         # Response embedding strategies
├── layers.py            # Neural layers and IRT extraction
├── coral_layer.py       # CORAL ordinal regression layer
├── coral_gpcm.py        # CORAL-GPCM integration models
└── model_factory.py     # Unified model creation
```

### Ordinal Loss Functions
```
training/
└── ordinal_losses.py    # Specialized losses for ordinal regression
    ├── DifferentiableQWKLoss    # Direct QWK optimization
    ├── OrdinalEMDLoss          # Earth Mover's Distance
    ├── OrdinalCrossEntropyLoss # Distance-weighted CE
    └── CombinedOrdinalLoss     # Weighted combination
```

### Pipeline
```
train.py           # Unified training with CV support
evaluate.py        # Model evaluation with comprehensive metrics
main.py           # Complete pipeline orchestrator
utils/plot_metrics.py  # Visualization generation
analysis/irt_analysis.py # IRT parameter analysis
```

## IRT Analysis

The system extracts Item Response Theory parameters for educational assessment insights:

```bash
# Complete IRT analysis (parameter recovery + temporal analysis)
python analysis/irt_analysis.py --dataset synthetic_OC

# Temporal analysis only
python analysis/irt_analysis.py --dataset synthetic_OC --analysis_types temporal

# Save parameters for external analysis
python analysis/irt_analysis.py --dataset synthetic_OC --save_params
```

### Generated Visualizations
- Parameter recovery plots comparing learned vs true IRT parameters
- Temporal heatmaps showing student ability (θ) evolution over time
- Combined temporal analysis showing α, β parameters → GPCM probabilities
- Standard IRT plots (ICC, TIF, Wright maps)

## Key Features

- **Unified Pipeline**: Single command for complete analysis workflow
- **Enhanced AKVMN**: Learnable ability scale and embedding weights
- **CORAL Integration**: Ordinal regression with rank consistency guarantees
- **Specialized Ordinal Losses**: QWK, EMD, ordinal CE, and combined losses
- **IRT Integration**: Full temporal IRT parameter analysis
- **Comprehensive Metrics**: Categorical accuracy, QWK, ordinal accuracy, MAE
- **Professional Visualizations**: Publication-ready plots with consistent styling
- **Cross-validation**: Automated k-fold CV with best model selection

## Configuration

### Main Arguments
- `--models`: Model types (`baseline`, `akvmn`, `coral`, `hybrid_coral`)
- `--dataset`: Dataset name (default: `synthetic_OC`)
- `--epochs`: Training epochs (default: 30)
- `--cv_folds`: Cross-validation folds (default: 5)
- `--batch_size`: Batch size (default: 32)
- `--device`: Device selection (`cuda`/`cpu`)
- `--action`: Pipeline action (`pipeline`, `train`, `evaluate`)

### Loss Function Arguments
- `--loss`: Loss type (`ce`, `qwk`, `emd`, `ordinal_ce`, `combined`)
- `--ce_weight`: Weight for CE in combined loss (default: 1.0)
- `--qwk_weight`: Weight for QWK in combined loss (default: 0.5)
- `--emd_weight`: Weight for EMD in combined loss (default: 0.0)
- `--coral_weight`: Weight for CORAL loss (default: 0.0)
- `--ordinal_alpha`: Alpha for ordinal CE (default: 1.0)

### Data Formats
- **Ordered Categories (OC)**: Discrete responses {0, 1, 2, ..., K-1}
- **Partial Credit (PC)**: Continuous scores [0, 1]

## Results Structure
```
results/
├── plots/              # Comprehensive visualizations (7 plots)
│   ├── training_metrics.png         # Training curves and convergence
│   ├── test_metrics.png            # Test performance metrics
│   ├── training_vs_test_comparison.png # Training vs test comparison
│   ├── categorical_breakdown.png    # Per-category performance
│   ├── confusion_matrices_test.png  # Model comparison matrices
│   ├── ordinal_distance_distribution_test.png # Error distance analysis
│   └── category_transitions_test.png # Response transition patterns
├── irt/               # IRT analysis results and heatmaps (5 plots)
├── train/             # Training metrics and histories
└── test/              # Test evaluation results with enhanced data
```

## Troubleshooting

**Common Issues:**
- CUDA out of memory → Reduce batch size
- Model loading errors → Check model path  
- Training instability → Adjust learning rate
- Batch evaluation now excludes CV folds by default → Use `--include_cv_folds` if needed
- CV fold evaluation requires corresponding fold data directories

**Performance Expectations:**
- Training: ~10-30 minutes (30 epochs, 5-fold CV, synthetic data)
- Memory: ~2-4GB GPU
- Accuracy: 55-60% categorical accuracy (corrected metrics)
- Sample Count: 41,115 test samples (excluding padding tokens)

---

**Deep-GPCM: Unified Pipeline for Knowledge Tracing**