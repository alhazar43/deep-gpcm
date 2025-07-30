# Deep-GPCM: Unified Knowledge Tracing System

Production-ready Deep Generalized Partial Credit Model for polytomous response prediction with unified pipeline architecture.

## Quick Start

### Complete Pipeline (Recommended)
```bash
# Train both baseline models and evaluate with plots
python main.py --dataset synthetic_OC --epochs 15

# Train specific models
python main.py --models baseline --dataset synthetic_OC --epochs 15
python main.py --models akvmn --dataset synthetic_OC --epochs 15

# Train both models
python main.py --models baseline akvmn --dataset synthetic_OC --epochs 15

# Model evaluation (auto-detects model type)
python evaluate.py --model_path save_models/best_baseline_synthetic_OC.pth
python evaluate.py --model_path save_models/best_akvmn_synthetic_OC.pth
```

### Environment Setup
```bash
# Activate conda environment (REQUIRED)
source ~/anaconda3/etc/profile.d/conda.sh && conda activate vrec-env

# Create necessary directories
mkdir -p logs save_models results/{train,test,valid,plots}

# Note: Intel MKL threading issue is automatically fixed in all Python scripts
```

## Performance Highlights

### Current Results (15 epochs, single-fold)
| Model | Categorical Accuracy | Quadratic Weighted Kappa | Ordinal Accuracy | MAE | Parameters |
|-------|---------------------|-------------------------|------------------|------|------------|
| **AKVMN (Enhanced)** | **70.66%** ⭐ | 0.760 | **89.27%** ⭐ | **0.430** ⭐ | 572K |
| **Baseline GPCM** | 70.46% | **0.761** ⭐ | 89.20% | 0.432 | 535K |

*Results from unified pipeline on synthetic_OC dataset*

**Key Findings**:
- AKVMN shows only marginal improvement (0.2%) over baseline
- The added complexity doesn't translate to proportional performance gains
- Enhanced AKVMN includes learnable ability scale and embedding weights from the original implementation

**Key Features**:
- Enhanced AKVMN includes learnable ability scale and embedding weights
- Both models support comprehensive IRT analysis and visualization
- Automatic model type detection in evaluation and analysis scripts
- Integrated plotting in main pipeline - visualizations generated automatically

## Architecture Overview

### Unified Architecture
The Deep-GPCM system follows a clean, unified architecture with consolidated components:

#### Core Components (`core/`)
- **Models** (`model.py`): DeepGPCM (baseline) and AttentionGPCM implementations
- **Enhanced Models** (`attention_enhanced.py`): EnhancedAttentionGPCM with learnable parameters
- **Memory Networks** (`memory_networks.py`): DKVMN dynamic key-value memory architecture
- **Embeddings** (`embeddings.py`): Response embedding strategies (linear_decay, ordered, etc.)
- **Neural Layers** (`layers.py`): All neural network layers including IRT parameter extraction
- **Model Factory** (`model_factory.py`): Centralized model creation with auto-detection

#### Key Features
- **Enhanced AKVMN**: Learnable ability scale and linear decay embedding weights
- **Unified Layers**: All neural components consolidated in `layers.py` for better organization
- **Standardized Scaling**: `ability_scale=1.0` default (2.0 for AKVMN)
- **Modular Design**: Pluggable embedding strategies and memory architectures
- **Clean Imports**: Simplified import structure from core module

### Model Implementations

#### DKVMN-GPCM (Baseline)
- **Architecture**: Modular DKVMN + IRT parameter extraction + GPCM
- **Memory**: Dynamic key-value memory network
- **Parameters**: ~134K (varies by configuration)
- **Features**: Pluggable embeddings, configurable memory size

#### Attention DKVMN-GPCM (AKVMN)
- **Architecture**: Enhanced DKVMN with multi-head attention and iterative refinement
- **Innovation**: Attention-based feature refinement with learnable gates
- **Parameters**: ~175K+ (varies by attention configuration)
- **Features**: Multi-cycle refinement, attention mechanisms

#### Enhanced AKVMN
- **Architecture**: EnhancedAttentionGPCM with learnable IRT parameters
- **Learnable Components**: 
  - Ability scale parameter (initialized at 2.0)
  - Linear decay embedding weights
- **Backward Compatible**: Works with all existing analysis tools

### Key Improvements
1. **Modularity**: Easy to add new models, memory networks, and embedding strategies
2. **Extensibility**: Registry-based model creation and plugin architecture
3. **Type Safety**: Full type hints for better IDE support and maintenance
4. **Research Standards**: Follows PyTorch Lightning and HuggingFace patterns

## File Structure

### Core Pipeline Scripts
```
train.py           # Unified training with CV support for both models
evaluate.py        # Auto-detection evaluation with comprehensive metrics  
plot_metrics.py    # Adaptive plotting for variable number of models
main.py           # Complete pipeline orchestrator
```

### Core Model Components
```
core/
├── __init__.py          # Core module exports
├── model.py            # DeepGPCM and AttentionGPCM implementations
├── memory_networks.py  # DKVMN memory architecture
├── embeddings.py       # Response embedding strategies
└── layers.py          # Neural layers including IRT parameter extraction
```

### Supporting Infrastructure
```
evaluation/metrics.py    # Comprehensive evaluation metrics
utils/                   # Data utilities and helper functions  
config.py               # Configuration management
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
# Train both models and evaluate (no plotting)
python main.py --dataset synthetic_OC --epochs 15

# Train specific models only
python main.py --models baseline --dataset synthetic_OC --epochs 15
python main.py --models akvmn --dataset synthetic_OC --epochs 15

# Train only (no evaluation)
python main.py --action train --models baseline akvmn --dataset synthetic_OC

# Evaluate only (existing models)
python main.py --action evaluate --models baseline akvmn --dataset synthetic_OC

# Optimized training scheme (mixed precision, data loading optimizations)
python main.py --training_scheme optimized --models baseline akvmn --epochs 15
```

### 2. Individual Component Usage

#### Training

##### Optimized Training (Safe Optimizations)
```bash
# Optimized training with mixed precision (baseline model)
python train_optimized.py --model baseline --dataset synthetic_OC --epochs 15

# Enhanced AKVMN with optimized training
python train_optimized.py --model akvmn --dataset synthetic_OC --epochs 15

# With gradient accumulation for larger effective batch size
python train_optimized.py --model akvmn --dataset synthetic_OC --epochs 15 \
    --batch_size 64 --gradient_accumulation_steps 2

# Disable AMP for small models/datasets where overhead exceeds benefits
python train_optimized.py --model baseline --dataset synthetic_OC --epochs 15 --no_amp

# Note: Optimized training preserves model behavior (no parallel processing that breaks temporal dependencies)
```

##### Standard Training
```bash
# Baseline model with 5-fold CV
python train.py --model baseline --dataset synthetic_OC --epochs 15 --n_folds 5

# AKVMN model
python train.py --model akvmn --dataset synthetic_OC --epochs 15 --n_folds 5

# Single-fold training (no CV)
python train.py --model baseline --dataset synthetic_OC --epochs 15 --n_folds 0
```

#### Evaluation
```bash
# Evaluate with auto-detection
python evaluate.py --model_path save_models/best_baseline_synthetic_OC.pth

# Evaluate with specific dataset
python evaluate.py --model_path save_models/best_akvmn_synthetic_OC.pth --dataset synthetic_OC

# Batch evaluation
python evaluate.py --model_path save_models/best_*_synthetic_OC.pth
```

#### Visualization (Separated from Main Pipeline)
```bash
# Generate plots from existing results (no retraining needed)
python utils/plot_metrics.py

# The plotting system automatically:
# - Detects all model variants (baseline, akvmn, optimized versions)
# - Distinguishes between standard and optimized training results
# - Adapts subplot layout based on available metrics
# - Highlights best performing models with gold color and star (★)
# - Shows mean and std for cross-validation results
# - Creates training curves, test comparisons, and train vs test plots
# - Supports both standard and optimized training results

# Note: Plotting is now separate from main.py to allow flexible re-plotting
# without retraining. Run after training/evaluation is complete.
```

#### IRT Parameter Analysis
```bash
# PRIMARY: Unified IRT analysis tool (recommended)
python analysis/irt_analysis.py --dataset synthetic_OC

# The unified tool provides:
# - Automatic model detection for the dataset
# - Temporal parameter extraction with flexible aggregation
# - Parameter recovery analysis with true values
# - Multiple visualization types (recovery, temporal, IRT plots)
# - Comprehensive summary reports

# Advanced usage examples:
# Temporal analysis with average theta
python analysis/irt_analysis.py --dataset synthetic_OC --theta_method average --analysis_types temporal

# Complete analysis with all visualizations
python analysis/irt_analysis.py --dataset synthetic_OC --analysis_types recovery temporal irt_plots

# Extract and save parameters only
python analysis/irt_analysis.py --dataset synthetic_OC --save_params --analysis_types none

# LEGACY: Individual IRT visualization tools (still supported)
# Standard IRT plots for a single model
python analysis/plot_irt.py --model_path save_models/best_baseline_synthetic_OC.pth --plot_type all

# Compare IRT parameters between models  
python analysis/plot_irt.py --compare save_models/best_baseline_synthetic_OC.pth save_models/best_akvmn_synthetic_OC.pth

# Extract and save parameters for external analysis
python analysis/plot_irt.py --model_path save_models/best_baseline_synthetic_OC.pth --save_params baseline_irt.npz

# Generate specific plot types
python analysis/plot_irt.py --model_path save_models/best_akvmn_synthetic_OC.pth --plot_type icc
python analysis/plot_irt.py --model_path save_models/best_baseline_synthetic_OC.pth --plot_type tif
```

### 3. Data Generation
```bash
# Generate standard synthetic dataset
python scripts/data_gen.py --format OC --categories 4 --students 800 --questions 50 --min_seq 10 --max_seq 50

# Generate larger dataset for stable results (current default)
python scripts/data_gen.py --format OC --categories 4 --students 800 --questions 400 --min_seq 100 --max_seq 400 --seed 42
```

## Configuration Options

### Command-Line Arguments

#### main.py
- `--models`: Model types to train/evaluate (`baseline`, `akvmn`)
- `--dataset`: Dataset name (default: `synthetic_OC`)
- `--epochs`: Training epochs (default: 15)
- `--n_folds`: CV folds, 0 for no CV (default: 5)
- `--skip_training`: Skip training phase
- `--skip_evaluation`: Skip evaluation phase
- `--device`: Device selection (`cuda`/`cpu`)
- `--batch_size`: Training batch size

#### train.py
- `--model`: Model type (`baseline`/`akvmn`)
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

#### plot_irt.py
- `--model_path`: Path to trained model checkpoint for IRT analysis
- `--compare`: Compare two models (`MODEL1.pth MODEL2.pth`)
- `--irt_file`: Load IRT parameters from saved file (`.npz`/`.json`)
- `--data_path`: Test data path (default: `data/synthetic_OC/synthetic_oc_test.txt`)
- `--plot_type`: IRT plot type (`all`/`icc`/`iif`/`tif`/`wright`/`dist`)
- `--save_params`: Save extracted IRT parameters to file
- `--output_dir`: Output directory for IRT plots
- `--labels`: Model labels for comparison plots

#### animate_irt.py
- `--model_path`: Path to trained model checkpoint for temporal analysis
- `--data_path`: Test data path (default: `data/synthetic_OC/synthetic_oc_test.txt`)
- `--output_dir`: Output directory for animations (default: `irt_animations`)
- `--max_sequences`: Maximum number of sequences to analyze (default: 50)
- `--sequence_idx`: Specific sequence index for student journey animation (default: 0)
- `--animation_type`: Animation type (`journey`/`distributions`/`heatmap`/`stats`/`all`)

## IRT Parameter Analysis

### Overview
The Deep-GPCM system extracts Item Response Theory (IRT) parameters from trained models, enabling detailed analysis of both student abilities and item characteristics. This functionality provides insights into model behavior and educational assessment quality.

### IRT Parameters
- **θ (theta)**: Student ability parameters - latent knowledge/skill level on scale [-3, 3]
- **α (alpha)**: Item discrimination parameters - how well items differentiate between ability levels
- **β (beta)**: Category threshold parameters - difficulty thresholds for transitioning between adjacent categories (K-1 per item)

### Unified IRT Analysis (Recommended)

The unified IRT analysis tool automatically finds and analyzes all trained models, including enhanced AKVMN:

```bash
# Basic parameter recovery analysis (auto-detects all models)
python analysis/irt_analysis.py --dataset synthetic_OC

# Temporal analysis with average theta
python analysis/irt_analysis.py --dataset synthetic_OC --theta_method average --analysis_types temporal

# Complete analysis with all plots
python analysis/irt_analysis.py --dataset synthetic_OC --analysis_types recovery temporal irt_plots

# Extract and save parameters only
python analysis/irt_analysis.py --dataset synthetic_OC --save_params --analysis_types none
```

The analysis automatically handles:
- Standard baseline models
- Enhanced AKVMN models with learnable parameters
- Optimized training results
- Cross-validation results

### Individual IRT Visualizations

#### Item Characteristic Curves (ICC)
Shows probability of responses across ability levels for individual items:
```bash
python analysis/plot_irt.py --model_path save_models/best_baseline_synthetic_OC.pth --plot_type icc
```

#### Test Information Function (TIF)
Shows total test information across ability levels:
```bash
python analysis/plot_irt.py --model_path save_models/best_akvmn_synthetic_OC.pth --plot_type tif
```

#### Model Comparison
Compare IRT parameters between models:
```bash
python analysis/plot_irt.py --compare save_models/best_baseline_synthetic_OC.pth save_models/best_akvmn_synthetic_OC.pth --labels "Baseline" "AKVMN"
```

### Parameter Extraction and Storage
Extract and save IRT parameters for external analysis:
```bash
# Save as NumPy compressed format
python analysis/plot_irt.py --model_path save_models/best_baseline_synthetic_OC.pth --save_params baseline_irt.npz

# Save as JSON format  
python analysis/plot_irt.py --model_path save_models/best_akvmn_synthetic_OC.pth --save_params akvmn_irt.json
```

### Generated Output Files
The IRT analysis creates organized visualizations:
```
irt_plots/
├── baseline/                          # Baseline model analysis
│   ├── item_characteristic_curves.png
│   ├── item_information_functions.png
│   ├── test_information_function.png
│   ├── wright_map.png
│   └── parameter_distributions.png
├── akvmn/                            # Enhanced AKVMN model analysis (with learnable parameters)
│   └── [same plot types]
└── comparison/                       # Model comparison
    ├── model_comparison.png
    └── parameter_distributions.png
```

### Understanding Dynamic IRT Parameters in Deep-GPCM

Unlike traditional IRT where parameters are static, Deep-GPCM learns **dynamic parameters** that evolve over time:

- **θ (Student Ability)**: Changes as students learn, showing knowledge growth trajectories
- **α (Item Discrimination)**: Adapts based on the DKVMN memory state and student history
- **β (Item Thresholds)**: Difficulty boundaries that adjust to student progress and context

This dynamic nature allows Deep-GPCM to capture the temporal aspects of learning that static IRT models miss.

### Enhanced AKVMN Model

The AKVMN (Attention-based Knowledge Tracing with Value Memory Network) model includes enhanced features with learnable parameters:

- **Learnable Ability Scale**: Dynamically adjusts the scaling of student abilities (initialized at 2.0)
- **Learnable Linear Decay Embedding**: Adaptive decay weights for response embedding that learn from data
- **Backward Compatibility**: The IRT analysis tool automatically detects and handles both standard and enhanced AKVMN models

These enhancements improve the model's ability to capture complex learning patterns while maintaining compatibility with existing analysis tools.

### Temporal IRT Animation Analysis

The system includes advanced temporal animation capabilities that capture the dynamic evolution of IRT parameters as students progress through questions, revealing the true temporal nature of knowledge tracing.

#### Animated Temporal Analysis
```bash
# Generate all temporal animations for a model
python analysis/animate_irt.py --model_path save_models/best_baseline_synthetic_OC.pth --animation_type all

# Individual student learning journey animation  
python analysis/animate_irt.py --model_path save_models/best_akvmn_synthetic_OC.pth --animation_type journey --sequence_idx 0

# Parameter distribution evolution over time
python analysis/animate_irt.py --model_path save_models/best_baseline_synthetic_OC.pth --animation_type distributions

# Ability trajectory heatmaps
python analysis/animate_irt.py --model_path save_models/best_akvmn_synthetic_OC.pth --animation_type heatmap

# Temporal summary statistics
python analysis/animate_irt.py --model_path save_models/best_baseline_synthetic_OC.pth --animation_type stats
```

#### Temporal Animation Types

**Student Learning Journey** - Animated visualization of individual student parameter evolution:
- Real-time student ability (θ) development 
- Item discrimination (α) changes over time
- Threshold parameter (β) evolution
- Response pattern visualization
- Question-by-question progression analysis

**Parameter Distribution Evolution** - Animated histograms showing population-level changes:
- Student ability distribution dynamics
- Discrimination parameter trends
- Threshold parameter shifts
- Statistical measures (mean, variance) over time

**Ability Trajectory Heatmap** - Population view of learning progressions:
- Student ability evolution matrix (students × time)
- Learning trajectory clustering patterns
- Individual vs. population learning rates
- Completion patterns and dropout analysis

**Temporal Summary Statistics** - Aggregate parameter evolution:
- Mean parameter evolution with confidence intervals
- Population statistics by question number
- Sample size changes over time
- Learning curve characterization

#### Generated Animation Files
```
irt_animations/
├── student_journey_seq0_baseline_gpcm.gif      # Individual learning animation
├── student_journey_seq0_akvmn_gpcm.gif         # AKVMN learning journey
├── parameter_distributions_baseline_gpcm.gif   # Population evolution
├── parameter_distributions_akvmn_gpcm.gif      # AKVMN population dynamics  
├── ability_trajectories_baseline_gpcm.png      # Learning trajectory heatmap
├── ability_trajectories_akvmn_gpcm.png         # AKVMN trajectory patterns
├── temporal_summary_stats_baseline_gpcm.png    # Aggregate statistics
└── temporal_summary_stats_akvmn_gpcm.png       # AKVMN summary trends
```

#### Key Insights from Temporal Analysis

**Parameter Type Classification**:
- **θ (Student Ability)**: TEMPORAL - evolves continuously as students progress through questions
- **α (Discrimination)**: STATIC per question - same value for each unique question across all students  
- **β (Thresholds)**: STATIC per question - same threshold array for each unique question across all students

**Critical Findings**:
- **True Ability Range**: Student abilities span much wider ranges than expected:
  - Baseline: [-7.40, 13.79] (not capped at ±2)
  - AKVMN: [-14.50, 11.91] (even wider dynamic range)
- **Learning Trajectory Patterns**: Individual students show distinct temporal ability evolution curves
- **Question-Specific Parameters**: 30 unique questions each have fixed discrimination and threshold values
- **Model Comparison**: AKVMN shows more dynamic ability changes and wider parameter ranges
- **Population Dynamics**: Temporal ability distributions reveal learning progression patterns across the student population

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
- **IRT Parameter Analysis**: Extract and visualize Item Response Theory parameters

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
3. Update documentation

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

## Variational Bayesian GPCM

### Overview
The Deep-GPCM system now includes a Variational Bayesian implementation that incorporates prior distributions for IRT parameters, enabling better parameter recovery and uncertainty quantification.

### Key Features
- **Variational Inference**: Proper Bayesian treatment using ELBO optimization (not simple MAP/regularization)
- **Prior Distributions**: Matching synthetic data generation:
  - θ (student ability) ~ N(0, 1)
  - α (discrimination) ~ LogNormal(0, 0.3)  
  - β (thresholds) ~ Ordered Normal with base difficulty N(0, 1)
- **Uncertainty Quantification**: Full posterior distributions with uncertainty estimates
- **Parameter Recovery**: Improved alignment with ground truth IRT parameters

### Training Bayesian Model
```bash
# Train Variational Bayesian model
python train_bayesian.py --dataset synthetic_OC --epochs 50 --kl_annealing

# Train with custom KL weight
python train_bayesian.py --dataset synthetic_OC --epochs 50 --kl_weight 0.5

# Train with higher learning rate
python train_bayesian.py --dataset synthetic_OC --epochs 50 --learning_rate 0.005
```

### Comparing IRT Parameters
```bash
# Compare all models (baseline, AKVMN, Bayesian) with ground truth
python compare_irt_models.py --dataset synthetic_OC

# Compare specific models
python compare_irt_models.py --baseline_model save_models/best_baseline_synthetic_OC.pth \
                           --bayesian_model save_models/best_bayesian_synthetic_OC.pth \
                           --output_dir irt_comparison

# Generate comparison visualizations
python compare_irt_models.py --dataset synthetic_OC --output_dir irt_comparison
```

### Bayesian Model Architecture
- **Variational Distributions**: 
  - NormalVariational for θ with reparameterization trick
  - LogNormalVariational for α (positive constraint)
  - OrderedNormalVariational for β (ordering constraint)
- **ELBO Loss**: Negative log likelihood + KL divergence with prior
- **KL Annealing**: Optional linear annealing schedule for stable training

### Generated Outputs
```
irt_comparison/
├── all_models_distributions.png      # Parameter distributions across all models
├── baseline_vs_truth_scatter.png     # Baseline vs ground truth scatter plots
├── bayesian_vs_truth_scatter.png     # Bayesian vs ground truth scatter plots
├── akvmn_vs_truth_scatter.png        # AKVMN vs ground truth scatter plots
├── bayesian_uncertainty.png          # Posterior uncertainty visualization
├── comparison_metrics_summary.csv    # Numerical comparison metrics
└── comparison_metrics_summary.txt    # Formatted comparison summary
```

### Key Metrics for IRT Recovery
- **Parameter Correlation**: How well learned parameters correlate with ground truth
- **Distribution Similarity**: Kolmogorov-Smirnov test for distribution matching
- **Wasserstein Distance**: Optimal transport distance between distributions
- **Standardized MSE**: Mean squared error after standardization
- **Posterior Uncertainty**: Standard deviation of posterior distributions

## Deep Bayesian-DKVMN Model

### Architecture Overview

The Deep Bayesian-DKVMN model represents a novel integration that **FULLY UTILIZES BOTH** DKVMN memory operations AND complete IRT computation without bypassing either system:

#### 🧠 **DKVMN Memory Network (Fully Integrated)**
1. **Memory Keys**: Question archetypes with Bayesian IRT parameters
   - Discrimination parameters α ~ LogNormal(μ_α, σ_α²) 
   - Threshold parameters β ~ OrderedNormal(μ_β, σ_β²)
   - Learnable embeddings for similarity computation

2. **Memory Values**: Dynamic student ability belief distributions
   - θ ~ N(μ, σ²) with Bayesian updates
   - Adaptive uncertainty tracking based on evidence
   - Full DKVMN read/write operations

3. **Attention Mechanism**: Hybrid DKVMN + IRT similarity
   - **DKVMN Component**: Embedding-based similarity
   - **IRT Component**: Parameter-based similarity (α, β)
   - **Combined**: Weighted attention using both sources

#### 🎯 **IRT Computation (Complete Implementation)**
1. **Parameter Extraction**: Attention-weighted IRT parameters
   - α (discrimination) weighted by DKVMN attention
   - β (thresholds) weighted by DKVMN attention  
   - θ (abilities) from DKVMN memory read

2. **GPCM Probability**: Full IRT model computation
   - **Complete GPCM formulation** with all parameters
   - **No bypassing** - uses proper IRT mathematics
   - **Integration**: DKVMN attention determines parameter weighting

#### 🔄 **Seamless Integration Process**
1. **Read Phase**: DKVMN attention → ability distributions → θ parameters
2. **Parameter Phase**: Sample α, β from memory keys → attention weighting
3. **IRT Phase**: Full GPCM computation using (θ, α, β) 
4. **Write Phase**: Bayesian memory update using response evidence
5. **ELBO Optimization**: Unified objective combining likelihood + KL

#### ✅ **No Bypassing Guarantee**
- **DKVMN**: Full attention-based memory read/write operations
- **IRT**: Complete GPCM probability computation  
- **Integration**: Both systems contribute to final predictions
- **Evidence**: Memory tracking shows both systems are active

### Training the Deep Bayesian-DKVMN Model

#### Basic Training

```bash
python train_deep_bayesian.py --dataset synthetic_OC
```

#### Advanced Configuration

```bash
python train_deep_bayesian.py \
    --dataset synthetic_OC \
    --epochs 100 \
    --batch_size 8 \
    --learning_rate 0.002 \
    --memory_size 15 \
    --key_dim 30 \
    --value_dim 50 \
    --patience 15 \
    --grad_clip 5.0 \
    --device cuda
```

#### Parameter Details

- `--dataset`: Dataset name (default: synthetic_OC)
- `--epochs`: Training epochs (default: 100)
- `--batch_size`: Batch size (default: 8, optimized for memory)
- `--learning_rate`: Learning rate (default: 0.002)
- `--memory_size`: Memory network size (default: 15)
- `--key_dim`: Memory key dimension (default: 30)
- `--value_dim`: Memory value dimension (default: 50)
- `--patience`: Early stopping patience (default: 15)
- `--grad_clip`: Gradient clipping threshold (default: 5.0)

#### Output Files

Training generates:
- `save_models/best_deep_bayesian_{dataset}.pth`: Best model checkpoint
- `logs/deep_bayesian_training_history_{dataset}_{timestamp}.json`: Training metrics
- Console output with real-time training progress

#### Testing Components

```bash
python test_deep_bayesian.py
```

Validates:
- Model initialization and forward pass
- Probability constraint satisfaction
- ELBO loss computation
- Gradient flow
- Memory operations
- Parameter extraction

### Model Features

#### Bayesian Memory Operations

**Memory Keys (BayesianMemoryKey)**:
- Question embeddings with learnable representations
- Discrimination α parameters with log-normal priors
- Ordered threshold β parameters with normal priors
- KL divergence computation for regularization

**Memory Values (BayesianMemoryValue)**:
- Student ability belief distributions θ ~ N(μ, σ²)
- Bayesian update mechanics with evidence integration
- Adaptive uncertainty tracking

#### Deep Integration Benefits

1. **Parameter Learning**: IRT parameters embedded in memory operations
2. **Uncertainty Quantification**: Full Bayesian treatment of all parameters
3. **Adaptive Memory**: Memory updates based on Bayesian inference
4. **ELBO Optimization**: Principled objective combining likelihood and priors
5. **Interpretability**: Extractable IRT parameters for analysis

#### Training Features

- **KL Annealing**: Gradual introduction of KL regularization (20 epochs warmup)
- **Early Stopping**: Automatic stopping based on test accuracy improvement
- **Gradient Clipping**: Stable training with gradient norm clipping
- **Parameter Recovery**: Automatic IRT parameter extraction and comparison
- **Comprehensive Metrics**: Accuracy, QWK, ordinal accuracy, MAE tracking

### Performance Expectations

Based on recent training (37 epochs):
- **Test Accuracy**: ~55% (best: 54.8%)
- **Quadratic Weighted Kappa**: ~0.24 (best: 0.238)
- **IRT Parameter Recovery**:
  - Beta correlation: 0.598 (approaching target >0.5)
  - Alpha correlation: 0.262
  - Theta correlation: 0.071

The model shows progressive improvement in parameter recovery compared to baseline approaches, with beta parameters showing the strongest correlation with ground truth.

## References

Built upon established knowledge tracing research:
- Dynamic Key-Value Memory Networks (DKVMN)
- Generalized Partial Credit Model (GPCM)
- Deep Item Response Theory frameworks
- Attention mechanisms for educational data
- Variational Bayesian methods for IRT parameter estimation

## License

MIT License - see LICENSE file for details.

---

**Deep-GPCM: Unified Pipeline for Production-Ready Knowledge Tracing**