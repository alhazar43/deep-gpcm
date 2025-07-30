# Deep-GPCM: Knowledge Tracing System

Production-ready Deep Generalized Partial Credit Model for polytomous response prediction.

## Quick Start

```bash
# Complete pipeline (training, evaluation, plotting, IRT analysis)
python main.py --dataset synthetic_OC --epochs 15

# Individual models
python main.py --models baseline --dataset synthetic_OC --epochs 15
python main.py --models akvmn --dataset synthetic_OC --epochs 15
```

## Environment Setup

```bash
# Activate conda environment (REQUIRED)
source ~/anaconda3/etc/profile.d/conda.sh && conda activate vrec-env

# Create directories
mkdir -p logs save_models results/{train,test,valid,plots,irt}
```

## Performance Results

| Model | Categorical Accuracy | Quadratic Weighted Kappa | Ordinal Accuracy | MAE |
|-------|---------------------|-------------------------|------------------|------|
| **AKVMN** | **70.66%** ⭐ | 0.760 | **89.27%** ⭐ | **0.430** ⭐ |
| **Baseline** | 70.46% | **0.761** ⭐ | 89.20% | 0.432 |

*Results from 15 epochs on synthetic_OC dataset*

## Usage

### Complete Pipeline
```bash
# Full pipeline with all phases
python main.py --dataset synthetic_OC --epochs 15

# Training only
python main.py --action train --models baseline akvmn --dataset synthetic_OC

# Evaluation only  
python main.py --action evaluate --models baseline akvmn --dataset synthetic_OC
```

### Individual Components
```bash
# Training with cross-validation
python train.py --model baseline --dataset synthetic_OC --epochs 15 --n_folds 5

# Model evaluation
python evaluate.py --model_path save_models/best_baseline_synthetic_OC.pth

# Generate visualizations
python utils/plot_metrics.py

# IRT parameter analysis
python analysis/irt_analysis.py --dataset synthetic_OC --analysis_types recovery temporal
```

### Data Generation
```bash
# Standard synthetic dataset
python data_gen.py --format OC --categories 4 --students 800 --questions 400 --seed 42
```

## Architecture

### Models
- **Baseline GPCM**: DKVMN + IRT parameter extraction + GPCM
- **AKVMN**: Enhanced attention-based DKVMN with learnable parameters

### Core Components
```
core/
├── model.py            # DeepGPCM and AttentionGPCM
├── attention_enhanced.py # Enhanced AKVMN with learnable parameters  
├── memory_networks.py  # DKVMN architecture
├── embeddings.py       # Response embedding strategies
└── layers.py          # Neural layers and IRT extraction
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
- **IRT Integration**: Full temporal IRT parameter analysis
- **Comprehensive Metrics**: Categorical accuracy, QWK, ordinal accuracy, MAE
- **Professional Visualizations**: Publication-ready plots with consistent styling
- **Cross-validation**: Automated k-fold CV with best model selection

## Configuration

### Main Arguments
- `--models`: Model types (`baseline`, `akvmn`)
- `--dataset`: Dataset name (default: `synthetic_OC`)
- `--epochs`: Training epochs (default: 15)
- `--cv_folds`: Cross-validation folds (default: 5)
- `--device`: Device selection (`cuda`/`cpu`)

### Data Formats
- **Ordered Categories (OC)**: Discrete responses {0, 1, 2, ..., K-1}
- **Partial Credit (PC)**: Continuous scores [0, 1]

## Results Structure
```
results/
├── plots/              # Training and evaluation visualizations
├── irt/               # IRT analysis results and heatmaps  
├── train/             # Training metrics and histories
└── test/              # Test evaluation results
```

## Troubleshooting

**Common Issues:**
- CUDA out of memory → Reduce batch size
- Model loading errors → Check model path
- Training instability → Adjust learning rate

**Performance Expectations:**
- Training: ~5-15 minutes (15 epochs, synthetic data)
- Memory: ~2-4GB GPU
- Accuracy: 60-75% categorical accuracy

---

**Deep-GPCM: Unified Pipeline for Knowledge Tracing**