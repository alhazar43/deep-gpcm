# CORAL Integration Guide for Deep-GPCM

## Overview

CORAL (COnsistent RAnk Logits) models have been successfully integrated into the Deep-GPCM pipeline as drop-in replacements for existing models. They provide enhanced ordinal regression capabilities while maintaining full compatibility with the existing train-evaluate-plot-IRT analysis pipeline.

## Available CORAL Models

### 1. CORALDeepGPCM (`coral`)
- Extends base DeepGPCM with CORAL ordinal constraints
- Enforces rank consistency through K-1 binary classifiers
- Provides cumulative probability structure
- Maintains IRT parameter extraction for analysis

### 2. HybridCORALGPCM (`hybrid_coral`)
- Combines CORAL structure with explicit GPCM formulation
- Blends CORAL and GPCM predictions (configurable weight)
- Best of both worlds approach

## Usage

### Basic Training
```bash
# Train CORAL model (same as baseline/akvmn)
python train.py --model coral --dataset synthetic_OC --epochs 30

# Train hybrid model
python train.py --model hybrid_coral --dataset synthetic_OC --epochs 30
```

### With Main Pipeline
```bash
# Complete pipeline with CORAL models
python main.py --models baseline coral --dataset synthetic_OC

# All models comparison
python main.py --models baseline akvmn coral hybrid_coral
```

### Advanced Usage with Ordinal Losses
```bash
# Use the specialized training script for ordinal losses
python train_coral.py --model coral --loss qwk --epochs 30

# Combined loss with QWK optimization
python train_coral.py --model coral --loss combined --qwk_weight 0.5
```

## Key Features

### 1. Full Pipeline Compatibility
- ✅ Works with existing train.py
- ✅ Compatible with evaluate.py
- ✅ Supports plot_metrics.py visualizations
- ✅ IRT analysis compatible

### 2. Ordinal Constraints
- Guaranteed rank consistency
- Cumulative probability structure
- Better calibrated predictions
- Improved ordinal accuracy

### 3. Additional Capabilities
- Access CORAL-specific information via `model.get_coral_info()`
- Support for specialized ordinal loss functions
- Maintains IRT parameter extraction

## Implementation Details

### Model Outputs
All CORAL models return the standard 4 outputs:
1. `theta`: Student abilities (batch_size, seq_len)
2. `beta`: Item thresholds (batch_size, seq_len, n_cats-1)
3. `alpha`: Discrimination parameters (batch_size, seq_len)
4. `probs`: Response probabilities (batch_size, seq_len, n_cats)

### CORAL-Specific Information
```python
# After forward pass
if hasattr(model, 'get_coral_info'):
    coral_info = model.get_coral_info()
    # Contains: logits, cumulative_probs, thresholds
```

## Performance Expectations

Based on research:
- **QWK**: Expected 5-10% improvement
- **Ordinal Accuracy**: Expected 3-5% improvement
- **Calibration**: Better confidence calibration
- **Training Time**: ~5% slower due to additional computations

## Configuration Options

### CORALDeepGPCM
- `coral_hidden_dim`: Hidden layer size (default: final_fc_dim)
- `use_coral_thresholds`: Learnable ordinal thresholds (default: True)
- `coral_dropout`: Dropout rate for CORAL layer (default: 0.1)

### HybridCORALGPCM
- `use_coral_structure`: Enable CORAL cumulative structure (default: True)
- `blend_weight`: CORAL vs GPCM blend (0=GPCM, 1=CORAL, default: 0.5)

## Troubleshooting

### Issue: Lower initial performance
- **Solution**: CORAL models may need different learning rates
- Try: `--lr 0.0005` or use learning rate scheduling

### Issue: Want to use ordinal losses
- **Solution**: Use `train_coral.py` for advanced loss functions
- Supports: QWK loss, EMD loss, ordinal cross-entropy

### Issue: Need to compare with baseline
- **Solution**: Use main.py with multiple models
- Example: `python main.py --models baseline coral --epochs 50`

## Next Steps

1. **Experiment with ordinal losses**: Try QWK loss for direct optimization
2. **Tune hyperparameters**: CORAL models may benefit from different settings
3. **Ablation studies**: Compare CORAL variants on different datasets
4. **Combine with attention**: Try CORAL with attention mechanisms

## References

- Original CORAL paper: "Rank consistent ordinal regression for neural networks"
- Implementation inspired by: https://github.com/Raschka-research-group/coral-pytorch