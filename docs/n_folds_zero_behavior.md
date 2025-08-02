# Training with --n_folds 0

## Overview

The `--n_folds 0` option enables single training run without any k-fold splitting. This is useful for:

1. Quick experiments and testing
2. Training on the full dataset without cross-validation overhead
3. Production model training when you have separate train/test sets
4. Rapid prototyping and development

## How It Works

When you specify `--n_folds 0`, the training system:

1. **Uses original train/test split**: Unlike k-fold modes that combine train+test data and resplit, `--n_folds 0` preserves the original data split
2. **Trains once**: Single training run with fixed hyperparameters
3. **Saves single model**: Outputs `saved_models/dataset/best_model.pth`
4. **Generates single result**: Creates `results/train/dataset/train_model.json`

## Usage

```bash
# Train on full dataset without folding
python train.py --model deep_gpcm --dataset synthetic_OC --epochs 30 --n_folds 0

# Or via main.py
python main.py --dataset synthetic_OC --epochs 30 --n_folds 0
```

## File Structure

With `--n_folds 0`, the output structure is simplified:

```
saved_models/
└── synthetic_OC/
    └── best_deep_gpcm.pth      # Single model file

results/
└── train/
    └── synthetic_OC/
        └── train_deep_gpcm.json # Single training log
```

## Comparison with K-Fold Modes

| Mode | Data Usage | Models Created | Training Time | Use Case |
|------|------------|----------------|---------------|----------|
| `--n_folds 0` | Original train/test split | 1 | 1× | Quick experiments, production |
| `--n_folds 5` | Combined & resplit | 5 + best | 5× | Model evaluation, stability testing |
| `--n_folds 5 --cv` | Combined & resplit | 5 + best | 50-100× | Hyperparameter optimization |

## Model Loading

Models trained with `--n_folds 0` are loaded the same way as k-fold models:

```bash
# Evaluation
python evaluate.py --model_path saved_models/synthetic_OC/best_deep_gpcm.pth --dataset synthetic_OC

# Or auto-detection
python evaluate.py --dataset synthetic_OC  # Finds best_deep_gpcm.pth automatically
```

## Implementation Details

The condition in `train.py` line 653:
```python
if args.no_cv or args.n_folds == 0:
    print("📈 Single training (no cross-validation)")
    # ... single training logic
```

This ensures that:
- `--n_folds 0` triggers single training mode
- `--no_cv` (deprecated) also triggers single training
- Data is not combined or reshuffled
- Original train/test split is preserved

## Example Output

```
==================== TRAINING DEEP_GPCM ====================
📈 Single training (no cross-validation)

🔍 MODEL CONFIGURATION:
  - Model type: DeepGPCM
  - Parameters: 278,555

🚀 TRAINING: deep_gpcm
Parameters: 278,555
Model type: DeepGPCM
Epoch | Train Loss | Train Acc | Test Acc | QWK | Ord.Acc | MAE | Grad.Norm | LR | Time(s)
    1 |     1.3794 |   0.3012 |  0.3842 |  0.241 | 0.6676 |  1.092 |    0.329 | 1.00e-03 |    5.4
    2 |     1.2901 |   0.4130 |  0.4732 |  0.486 | 0.7502 |  0.870 |    0.301 | 1.00e-03 |    4.4

✅ Training completed! Best QWK: 0.486 at epoch 2
💾 Model saved to: saved_models/synthetic_OC/best_deep_gpcm.pth
📋 Logs saved to: results/train/synthetic_OC/train_deep_gpcm.json
```

## Summary

`--n_folds 0` provides a clean, simple way to train models without the complexity of k-fold validation. It:
- ✅ Trains on full dataset without folding
- ✅ Saves models properly in the new file structure
- ✅ Can be loaded and evaluated normally
- ✅ Preserves original train/test split
- ✅ Runs 5× faster than 5-fold training
- ✅ Ideal for production and quick experiments