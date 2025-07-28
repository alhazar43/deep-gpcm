# Deep-GPCM: Unified Knowledge Tracing System

Production-ready Deep Generalized Partial Credit Model for polytomous response prediction with unified pipeline architecture.

## Quick Start

### Complete Pipeline (Recommended)
```bash
# Complete training, evaluation, and visualization pipeline
python main.py --dataset synthetic_OC --epochs 15

# Single model training  
python train.py --model baseline --dataset synthetic_OC --epochs 15 --n_folds 5
python train.py --model deep_integration --dataset synthetic_OC --epochs 15 --n_folds 5

# Model evaluation
python evaluate.py --model_path save_models/best_baseline_synthetic_OC.pth
python evaluate.py --model_path save_models/best_deep_integration_synthetic_OC.pth

# Generate visualizations
python plot_metrics.py --train_results logs/*.json --test_results results/test/*.json
```

### Environment Setup
```bash
# Activate environment
source ~/anaconda3/etc/profile.d/conda.sh && conda activate vrec-env

# Install dependencies (if needed)
pip install torch scikit-learn matplotlib seaborn numpy
```

## Performance Highlights

### Current Results (15 epochs, 5-fold CV)
| Model | Categorical Accuracy | Quadratic Weighted Kappa | Ordinal Accuracy | Parameters |
|-------|---------------------|-------------------------|------------------|------------|
| **Baseline GPCM** | 68.7% | 0.641 | 84.4% | 134K |
| **Deep Integration** | 70.3% | 0.687 | 86.0% | 174K |

*Results from unified pipeline on synthetic_OC dataset*

**Key Improvements**:
- Deep Integration achieves 1.6% higher categorical accuracy
- 4.6pp improvement in Quadratic Weighted Kappa 
- Consistent performance across all ordinal metrics
- Stable training with proper convergence patterns

## Architecture Overview

### Unified Pipeline Design
- **Single Training Script** (`train.py`): Supports both models with k-fold cross-validation
- **Auto-Detection Evaluation** (`evaluate.py`): Automatically detects model type from checkpoints
- **Adaptive Plotting** (`plot_metrics.py`): Generates comprehensive visualizations for any number of models
- **Pipeline Orchestrator** (`main.py`): Coordinates complete workflow execution

### Model Implementations

#### Baseline GPCM
- **Architecture**: DKVMN + GPCM predictor
- **Memory Network**: Dynamic key-value memory for knowledge state tracking
- **Parameters**: 134,055
- **Training**: Stable convergence with cross-entropy loss

#### Deep Integration GPCM  
- **Architecture**: Enhanced DKVMN with multi-head attention and advanced embeddings
- **Innovation**: Proper integration of memory networks with attention mechanisms
- **Parameters**: 174,354  
- **Training**: Improved performance through deep architectural integration

### Core Components
1. **Memory Network**: Dynamic key-value memory system
2. **Embedding Strategy**: Linear decay embedding for polytomous responses
3. **GPCM Predictor**: Multi-category probability prediction
4. **Loss Function**: Cross-entropy loss with comprehensive ordinal metrics

## File Structure

### Core Pipeline Scripts
```
train.py           # Unified training with CV support for both models
evaluate.py        # Auto-detection evaluation with comprehensive metrics  
plot_metrics.py    # Adaptive plotting for variable number of models
main.py           # Complete pipeline orchestrator
```

### Model Implementations
```
models/
├── __init__.py                     # Module initialization
├── baseline.py                     # Baseline DKVMN-GPCM implementation
└── deep_integration_gpcm_proper.py # Deep Integration model
```

### Supporting Infrastructure
```
evaluation/metrics.py    # Comprehensive evaluation metrics
utils/                   # Data utilities and helper functions  
config.py               # Configuration management
model_factory.py        # Model creation utilities
data_gen.py            # Synthetic data generation
requirements.txt       # Package dependencies
```

### Results Organization
```
results/
├── plots/              # Generated visualizations
├── test/               # Test evaluation results
logs/                   # Training logs and histories
save_models/           # Trained model checkpoints
```

## Usage Guide

### 1. Complete Pipeline Execution
```bash
# Train both models, evaluate, and generate plots
python main.py --dataset synthetic_OC --epochs 15

# Train specific models only
python main.py --models baseline --dataset synthetic_OC --epochs 15
python main.py --models deep_integration --dataset synthetic_OC --epochs 15

# Skip training (evaluate existing models only)
python main.py --skip_training --dataset synthetic_OC
```

### 2. Individual Component Usage

#### Training
```bash
# Baseline model with 5-fold CV
python train.py --model baseline --dataset synthetic_OC --epochs 15 --n_folds 5

# Deep Integration model
python train.py --model deep_integration --dataset synthetic_OC --epochs 15 --n_folds 5

# Single-fold training (no CV)
python train.py --model baseline --dataset synthetic_OC --epochs 15 --n_folds 0
```

#### Evaluation
```bash
# Evaluate with auto-detection
python evaluate.py --model_path save_models/best_baseline_synthetic_OC.pth

# Evaluate with specific dataset
python evaluate.py --model_path save_models/best_deep_integration_synthetic_OC.pth --dataset synthetic_OC

# Batch evaluation
python evaluate.py --model_path save_models/best_*_synthetic_OC.pth
```

#### Visualization
```bash
# Generate all plots
python plot_metrics.py --train_results logs/*.json --test_results results/test/*.json

# Specific plot types
python plot_metrics.py --plot_type final --train_results logs/*.json
python plot_metrics.py --plot_type curves --train_results logs/*.json
python plot_metrics.py --plot_type confusion --test_results results/test/*.json
```

### 3. Data Generation
```bash
# Generate synthetic datasets
python data_gen.py --format both --categories 4 --students 800 --questions 50

# Generate for specific format
python data_gen.py --format OC --categories 4 --students 800 --questions 30
```

## Configuration Options

### Command-Line Arguments

#### main.py
- `--models`: Model types to train/evaluate (`baseline`, `deep_integration`)
- `--dataset`: Dataset name (default: `synthetic_OC`)
- `--epochs`: Training epochs (default: 15)
- `--n_folds`: CV folds, 0 for no CV (default: 5)
- `--skip_training`: Skip training phase
- `--skip_evaluation`: Skip evaluation phase
- `--device`: Device selection (`cuda`/`cpu`)
- `--batch_size`: Training batch size

#### train.py
- `--model`: Model type (`baseline`/`deep_integration`)
- `--dataset`: Dataset name
- `--epochs`: Training epochs
- `--n_folds`: Number of CV folds (0 for single-fold)
- `--batch_size`: Batch size
- `--device`: Device selection

#### evaluate.py
- `--model_path`: Path to trained model checkpoint
- `--dataset`: Dataset for evaluation (auto-detected if not specified)
- `--batch_size`: Evaluation batch size
- `--device`: Device selection

#### plot_metrics.py
- `--train_results`: Training result JSON files
- `--test_results`: Test result JSON files  
- `--output_dir`: Output directory for plots
- `--plot_type`: Plot type (`all`/`final`/`curves`/`confusion`)

## Evaluation Metrics

### Comprehensive Assessment
The system provides multiple evaluation metrics for thorough model assessment:

- **Categorical Accuracy**: Standard classification accuracy
- **Quadratic Weighted Kappa (QWK)**: Ordinal agreement measure  
- **Ordinal Accuracy**: Order-preserving prediction accuracy
- **Mean Absolute Error (MAE)**: Prediction distance measure
- **Prediction Consistency**: Multiple prediction method consistency
- **Ordinal Ranking**: Ranking preservation accuracy
- **Distribution Consistency**: Output distribution alignment

### Alternative Prediction Methods
- **Argmax**: Standard maximum probability prediction
- **Cumulative**: Cumulative probability threshold method
- **Expected**: Expected value prediction method

## Data Formats

### Supported Datasets
- **Ordered Categories (OC)**: Discrete responses {0, 1, 2, ..., K-1}
- **Partial Credit (PC)**: Continuous scores in [0, 1] range

### File Structure
```
data/
└── dataset_name/
    ├── dataset_name_train.txt    # Training sequences
    ├── dataset_name_test.txt     # Test sequences  
    └── metadata.json             # Dataset configuration
```

## Advanced Features

### Cross-Validation Support
- Configurable k-fold cross-validation
- Automatic best model selection based on QWK
- Comprehensive fold-wise metrics reporting
- Best model preservation for evaluation

### Auto-Detection
- Model type detection from checkpoints
- Dataset configuration detection from metadata
- Automatic parameter and architecture reconstruction

### Comprehensive Logging
- Training progress with epoch-wise metrics
- Cross-validation summaries
- Model configuration preservation
- Performance tracking and visualization

### Visualization Suite
- Training curves comparison
- Final performance comparison
- Confusion matrices
- Adaptive plotting for any number of models

## Performance Optimization

### Training Efficiency
- GPU acceleration support
- Optimized data loading with batching
- Memory-efficient cross-validation
- Progress tracking and early stopping

### Evaluation Speed
- Batch processing for large datasets
- Efficient metric computation
- Parallel evaluation support
- Caching for repeated evaluations

## Development Notes

### Model Implementation Guidelines
1. All models must inherit from `torch.nn.Module`
2. Implement standard forward pass interface
3. Support both training and evaluation modes
4. Return K-category probability distributions

### Adding New Models
1. Create model file in `models/` directory
2. Import in `train.py` and `evaluate.py`
3. Add to model factory if needed
4. Update documentation

### Testing and Validation
- Use synthetic datasets for rapid prototyping
- Verify performance against established baselines
- Ensure cross-validation stability
- Validate visualization outputs

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Model loading errors**: Ensure correct model path and compatibility
3. **Data format errors**: Verify dataset structure and metadata
4. **Training instability**: Check learning rate and gradient clipping

### Performance Expectations
- **Training Time**: ~5-15 minutes for 15 epochs on synthetic data
- **Memory Usage**: ~2-4GB GPU memory for standard batch sizes
- **Accuracy Range**: 60-75% categorical accuracy on synthetic datasets
- **QWK Range**: 0.4-0.7 for well-performing models

## References

Built upon established knowledge tracing research:
- Dynamic Key-Value Memory Networks (DKVMN)
- Generalized Partial Credit Model (GPCM)
- Deep Item Response Theory frameworks
- Attention mechanisms for educational data

## License

MIT License - see LICENSE file for details.

---

**Deep-GPCM: Unified Pipeline for Production-Ready Knowledge Tracing**