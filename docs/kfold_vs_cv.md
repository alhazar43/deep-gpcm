# K-Fold Training vs Cross-Validation with Hyperparameter Tuning

## Overview

The Deep-GPCM training system now supports three distinct modes:

1. **Single Training** (`--n_folds 0`)
2. **K-Fold Training** (`--n_folds N` without `--cv`)
3. **Cross-Validation with Hyperparameter Tuning** (`--n_folds N --cv`)

## Single Training (No Folds)

Single training runs once on the original train/test split without any folding.

### Usage
```bash
# Train without any folding
python train.py --model deep_gpcm --dataset synthetic_OC --epochs 30 --n_folds 0

# Or via main.py
python main.py --dataset synthetic_OC --epochs 30 --n_folds 0
```

### Process
1. Use original train/test split (no data combining)
2. Train once with fixed hyperparameters
3. Save single model

### Output
- Model: `saved_models/dataset/best_model.pth`
- Training log: `results/train/dataset/train_model.json`

## K-Fold Training (Default)

K-fold training splits the data into k folds and trains k models with the SAME hyperparameters, then selects the best model based on performance.

### Usage
```bash
# Train with 5-fold training (no hyperparameter tuning)
python train.py --model deep_gpcm --dataset synthetic_OC --epochs 30 --n_folds 5

# Or via main.py
python main.py --dataset synthetic_OC --epochs 30 --n_folds 5
```

### Process
1. Combine train and test data
2. Split into k folds
3. For each fold:
   - Train on k-1 folds
   - Validate on 1 fold
   - Use SAME hyperparameters (lr=0.001, batch_size=64, etc.)
4. Select best fold based on QWK score
5. Save the best fold's model

### Output
- Individual fold models: `saved_models/dataset/model_fold_X.pth`
- Best model: `saved_models/dataset/best_model.pth`
- Fold results: `results/train/dataset/train_model_fold_X.json`
- Summary: `results/validation/dataset/cv_model_summary.json`

## Cross-Validation with Hyperparameter Tuning

True cross-validation performs nested CV with hyperparameter search to find optimal parameters.

### Usage
```bash
# Train with 5-fold CV and hyperparameter tuning
python train.py --model deep_gpcm --dataset synthetic_OC --epochs 30 --n_folds 5 --cv

# Or via main.py
python main.py --dataset synthetic_OC --epochs 30 --n_folds 5 --cv
```

### Process
1. Outer CV loop (k folds for unbiased evaluation)
2. For each outer fold:
   - Inner CV loop (3 folds for hyperparameter selection)
   - Grid search over hyperparameters:
     - Learning rate: [0.001, 0.005, 0.01]
     - Batch size: [32, 64, 128]
     - Model-specific params (memory_size, final_fc_dim, etc.)
   - Select best hyperparameters based on inner CV
   - Train final model with best params on outer fold
3. Report performance across outer folds
4. Save model from best-performing fold

### Hyperparameter Grids

**Base parameters (all models):**
- `lr`: [0.001, 0.005, 0.01]
- `batch_size`: [32, 64, 128]

**Model-specific parameters:**
- **deep_gpcm**: 
  - `final_fc_dim`: [50, 100]
  - `memory_size`: [20, 50]
- **attn_gpcm**:
  - `final_fc_dim`: [50, 100]
  - `memory_size`: [20, 50]
  - `attention_dim`: [64, 128]
- **coral_gpcm_proper**:
  - `final_fc_dim`: [50, 100]
  - `memory_size`: [20, 50]

### Output
- Best model: `saved_models/dataset/best_model.pth` (includes best hyperparameters)
- CV summary: `results/validation/dataset/cv_model_summary.json`
  - Performance metrics (mean ± std across folds)
  - Best hyperparameters per fold
  - Overall best hyperparameters

## Key Differences

| Aspect | Single Training | K-Fold Training | CV with Hyperparameter Tuning |
|--------|-----------------|-----------------|------------------------------|
| Purpose | Quick experiments | Multiple runs for stability | Find optimal hyperparameters |
| Data Split | Original train/test | Combined & resplit | Combined & resplit |
| Hyperparameters | Fixed | Fixed for all folds | Optimized per fold |
| Computational Cost | 1× training time | k × training time | k × m × training time |
| Models Created | 1 | k + best | k + best |
| Result | Single model | Best model from k folds | Optimal hyperparameters + best model |
| Use Case | Development/Production | Model evaluation | Hyperparameter optimization |

## Examples

### Example 1: Quick Model Comparison
```bash
# Compare models with fixed hyperparameters
python main.py --dataset synthetic_OC --epochs 30 --n_folds 5
```

### Example 2: Production Model Training
```bash
# Find optimal hyperparameters for production
python train.py --model coral_gpcm_proper --dataset synthetic_OC --epochs 30 --n_folds 5 --cv
```

### Example 3: Single Training Run
```bash
# No k-fold, just single training
python train.py --model deep_gpcm --dataset synthetic_OC --epochs 30 --n_folds 0
```

## Performance Considerations

- K-fold training: ~5× slower than single training
- CV with tuning: ~50-100× slower than single training (depends on grid size)
- For quick experiments: Use k-fold training
- For final models: Use CV with hyperparameter tuning