# Advanced Hyperparameter Optimization for Deep-GPCM

This document describes the modern, adaptive hyperparameter optimization system for Deep-GPCM models, featuring Bayesian optimization, tunable loss weights, and streamlined cross-validation.

## Overview

The new hyperparameter optimization system addresses several limitations of the previous approach:

1. **Streamlined CV Logic**: Uses unified `--n_folds N` parameter instead of redundant `--cv`/`--no_cv` flags
2. **Bayesian Optimization**: Advanced Gaussian Process-based search instead of grid search
3. **Tunable Loss Weights**: Optimizes loss function weights for combined losses
4. **Early Stopping**: Automatic termination of unpromising configurations
5. **Factory Integration**: Seamless integration with existing model factory pattern

## Quick Start

### Basic Hyperparameter Optimization

```bash
# Run Bayesian hyperparameter optimization
python train.py --model deep_gpcm --dataset your_dataset --hyperopt --hyperopt_trials 50

# Specify optimization metric
python train.py --model attn_gpcm_learn --dataset your_dataset --hyperopt \
                --hyperopt_metric quadratic_weighted_kappa --hyperopt_trials 30

# Control cross-validation folds
python train.py --model stable_temporal_attn_gpcm --dataset your_dataset --hyperopt \
                --n_folds 5 --hyperopt_trials 40
```

### Training Modes

```bash
# No cross-validation (single train/test split)
python train.py --model deep_gpcm --dataset your_dataset --n_folds 0

# Single run (same as n_folds 0)
python train.py --model deep_gpcm --dataset your_dataset --n_folds 1

# K-fold cross-validation without hyperparameter optimization
python train.py --model deep_gpcm --dataset your_dataset --n_folds 5

# Bayesian hyperparameter optimization with CV (recommended)
python train.py --model deep_gpcm --dataset your_dataset --hyperopt --n_folds 5
```

## Architecture

### Hyperparameter Search Space

The system automatically creates model-specific search spaces based on:

1. **Factory Registry**: Gets model architecture parameters from `models/factory.py`
2. **Training Parameters**: Learning rate, batch size, weight decay
3. **Loss Weights**: For models with combined loss functions

#### Example Search Space (deep_gpcm)

```python
search_space = [
    HyperparameterSpace('memory_size', 'discrete', (20, 100), default=50),
    HyperparameterSpace('final_fc_dim', 'discrete', (50, 100), default=50),
    HyperparameterSpace('dropout_rate', 'continuous', (0.0, 0.2), default=0.1),
    HyperparameterSpace('lr', 'continuous', (1e-4, 1e-2), log_scale=True, default=1e-3),
    HyperparameterSpace('batch_size', 'discrete', (16, 128), default=64),
    # Loss weights (for combined loss models)
    HyperparameterSpace('ce_weight_logit', 'continuous', (-2.0, 2.0), default=0.0),
    HyperparameterSpace('focal_weight_logit', 'continuous', (-2.0, 2.0), default=0.0),
    # qwk_weight computed to ensure sum = 1.0
]
```

### Bayesian Optimization Process

1. **Random Initialization**: Start with random sampling for exploration
2. **Gaussian Process Fitting**: Model the objective function
3. **Acquisition Function**: Choose next parameters to evaluate
   - Expected Improvement (EI)
   - Probability of Improvement (PI) 
   - Upper Confidence Bound (UCB)
4. **Cross-Validation Evaluation**: Train and validate with sampled parameters
5. **Early Stopping**: Stop if no improvement for patience iterations

### Loss Weight Optimization

For models with combined loss functions, the system optimizes loss weights using:

1. **Logit Parameterization**: Represents weights in unconstrained space
2. **Softmax Normalization**: Ensures weights sum to 1.0
3. **Automatic Detection**: Infers active loss components from factory config

#### Example: Combined Loss Weight Optimization

```python
# Input logits (unconstrained)
ce_weight_logit = 0.5
focal_weight_logit = -0.2
# qwk_weight_logit = 0.0 (reference)

# Softmax transformation
logits = [0.5, -0.2, 0.0]
weights = softmax(logits)  # [0.549, 0.269, 0.182]

# Final loss weights
ce_weight = 0.549
focal_weight = 0.269  
qwk_weight = 0.182
```

## Configuration

### OptimizationConfig

```python
@dataclass
class OptimizationConfig:
    n_trials: int = 50                    # Maximum optimization trials
    n_initial_random: int = 10            # Initial random exploration
    acquisition_function: str = 'ei'      # EI, PI, or UCB
    early_stopping_patience: int = 15     # Stop after N trials without improvement
    cv_folds: int = 3                     # Cross-validation folds
    cv_epochs: int = 10                   # Epochs per CV fold
    final_epochs: int = 30                # Final model training epochs
    metric_to_optimize: str = 'quadratic_weighted_kappa'  # Target metric
    maximize: bool = True                 # Whether to maximize metric
    parallel_workers: int = 1             # Parallel evaluation (future)
    save_intermediate: bool = True        # Save progress periodically
```

### Command Line Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--hyperopt` | flag | False | Enable Bayesian hyperparameter optimization |
| `--hyperopt_trials` | int | 50 | Number of optimization trials |
| `--hyperopt_metric` | str | quadratic_weighted_kappa | Metric to optimize |
| `--n_folds` | int | 5 | CV folds (0=no CV, 1=single, >1=K-fold) |
| `--loss` | str | factory | Loss function override |

## Usage Examples

### 1. Quick Optimization

```python
from optimization.adaptive_hyperopt import create_optimizer, OptimizationConfig

# Create optimizer
config = OptimizationConfig(n_trials=20, cv_folds=3)
optimizer = create_optimizer(
    model_type='deep_gpcm',
    n_questions=50,
    n_cats=4,
    config=config
)

# Run optimization
best_trial = optimizer.optimize(train_data, test_data)
best_params = best_trial.hyperparameters

print(f"Best QWK: {optimizer.best_score:.4f}")
print(f"Best parameters: {best_params}")
```

### 2. Multi-Model Comparison

```bash
# Compare multiple models with same optimization budget
for model in deep_gpcm attn_gpcm_learn stable_temporal_attn_gpcm; do
    python train.py --model $model --dataset your_dataset --hyperopt \
                    --hyperopt_trials 30 --n_folds 3
done
```

### 3. Production Training Pipeline

```bash
#!/bin/bash
# production_training.sh

DATASET="your_production_dataset"
TRIALS=100
FOLDS=5

# Stage 1: Hyperparameter optimization
python train.py --model deep_gpcm --dataset $DATASET \
                --hyperopt --hyperopt_trials $TRIALS --n_folds $FOLDS \
                --epochs 50

# Stage 2: Final validation (model automatically saved with best params)
python evaluate.py --model_path "saved_models/${DATASET}/best_deep_gpcm.pth" \
                   --dataset $DATASET --split test

# Stage 3: IRT analysis
python analysis/irt_analysis.py --model_path "saved_models/${DATASET}/best_deep_gpcm.pth" \
                                --dataset $DATASET
```

## Performance Considerations

### Computational Complexity

- **Grid Search**: O(k^n) where k = values per parameter, n = parameters
- **Bayesian Optimization**: O(t^3) where t = number of trials
- **Memory Usage**: O(t × d) where d = search space dimensionality

### Optimization Strategies

1. **Start Small**: Begin with 20-30 trials for quick feedback
2. **Increase Gradually**: Scale to 50-100 trials for production
3. **Use Early Stopping**: Set reasonable patience (15-20 trials)
4. **Parallel CV**: Use multiple folds efficiently

### Time Estimates

| Configuration | Est. Time | Use Case |
|---------------|-----------|----------|
| 20 trials, 3 folds, 5 epochs | ~30 min | Quick experimentation |
| 50 trials, 3 folds, 10 epochs | ~2 hours | Development |
| 100 trials, 5 folds, 15 epochs | ~6 hours | Production |

## Advanced Features

### Custom Search Spaces

```python
# Extend search space for specific models
from optimization.adaptive_hyperopt import HyperparameterSpace

custom_spaces = [
    HyperparameterSpace('custom_param', 'continuous', (0.1, 1.0)),
    HyperparameterSpace('custom_choice', 'categorical', ['a', 'b', 'c'])
]

optimizer = create_optimizer(model_type='your_model', ...)
optimizer.search_space.extend(custom_spaces)
```

### Multi-Objective Optimization

```python
# Optimize multiple metrics simultaneously
config = OptimizationConfig(
    metric_to_optimize='quadratic_weighted_kappa',  # Primary
    secondary_metrics=['categorical_accuracy', 'ordinal_accuracy']
)
```

### Acquisition Function Selection

```python
# Try different acquisition functions
configs = [
    OptimizationConfig(acquisition_function='ei'),   # Expected Improvement
    OptimizationConfig(acquisition_function='pi'),   # Probability of Improvement  
    OptimizationConfig(acquisition_function='ucb'),  # Upper Confidence Bound
]
```

## Results and Analysis

### Output Files

The system generates comprehensive results:

```
optimization_results/
├── hyperopt_deep_gpcm_final.json          # Complete optimization results
├── hyperopt_deep_gpcm_intermediate_20.json # Intermediate checkpoints
└── ...

saved_models/your_dataset/
├── best_deep_gpcm.pth                      # Best model with metadata
└── ...

results/train/your_dataset/
├── train_deep_gpcm.json                    # Training metrics with hyperopt summary
└── ...
```

### Result Structure

```json
{
  "model_type": "deep_gpcm",
  "config": {...},
  "trials": [
    {
      "trial_id": 0,
      "hyperparameters": {...},
      "metrics": {"quadratic_weighted_kappa": 0.765, ...},
      "cv_scores": [0.743, 0.778, 0.774],
      "training_time": 127.3,
      "converged": true
    }
  ],
  "best_trial": {...},
  "best_score": 0.778
}
```

### Analysis Tools

```python
import json

# Load optimization results
with open('optimization_results/hyperopt_deep_gpcm_final.json') as f:
    results = json.load(f)

# Analyze parameter importance
trials = results['trials']
param_correlations = analyze_parameter_correlations(trials)

# Visualize optimization progress
plot_optimization_history(trials)
plot_parameter_distributions(trials)
```

## Migration from Legacy System

### Old Command Style
```bash
# Legacy approach (deprecated)
python train.py --model deep_gpcm --dataset data --cv --n_folds 5
python train.py --model deep_gpcm --dataset data --no_cv
```

### New Command Style
```bash
# Modern approach
python train.py --model deep_gpcm --dataset data --hyperopt --n_folds 5
python train.py --model deep_gpcm --dataset data --n_folds 0
```

### Compatibility

The system maintains backward compatibility:
- `--cv` flag maps to basic cross-validation
- `--no_cv` flag sets `--n_folds 0`
- Legacy grid search still available via direct factory calls
- All existing model factory configurations preserved

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**
   ```bash
   # Reduce batch size range
   python train.py --model your_model --hyperopt --batch_size 32
   ```

2. **Optimization Not Converging**
   ```bash
   # Increase trials and patience
   python train.py --model your_model --hyperopt --hyperopt_trials 100
   ```

3. **Poor Search Space Coverage**
   ```python
   # Increase initial random trials
   config = OptimizationConfig(n_initial_random=20)
   ```

### Debug Mode

```bash
# Enable detailed logging
export PYTHONPATH=/path/to/deep-gpcm:$PYTHONPATH
python -m logging DEBUG train.py --model your_model --hyperopt
```

## Future Enhancements

- **Multi-GPU Support**: Parallel trial evaluation
- **Advanced Acquisition**: Multi-fidelity and multi-objective
- **Automated Feature Selection**: Include feature engineering in search
- **Distributed Optimization**: Scale across multiple machines
- **Ensemble Methods**: Optimize ensemble weights
- **Transfer Learning**: Warm-start from previous optimizations

---

*For more examples and advanced usage, see `examples/hyperopt_demo.py`*