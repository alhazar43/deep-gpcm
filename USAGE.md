# Deep-GPCM Usage Guide

## Basic Training

**Single Model Training**:
```bash
python train.py --model deep_gpcm --dataset synthetic_500_200_4 --epochs 30
```

**Multiple Models Training**:
```bash
python main.py --models deep_gpcm attn_gpcm_learn --dataset synthetic_500_200_4 --epochs 30
```

## Adaptive Hyperparameter Optimization

**Enable Adaptive Optimization (Default)**:
```bash
python train.py --model deep_gpcm --dataset synthetic_500_200_4 --hyperopt --hyperopt_trials 50
```

**Advanced Adaptive Configuration**:
```bash
python train.py --model deep_gpcm --dataset synthetic_500_200_4 \
    --hyperopt --hyperopt_trials 50 \
    --adaptive --adaptive_epochs 5,15,40 \
    --adaptive_arch --adaptive_learning
```

**Disable Adaptive Features**:
```bash
python train.py --model deep_gpcm --dataset synthetic_500_200_4 \
    --hyperopt --no_adaptive
```

## Complete Pipeline

**Full Pipeline with Adaptive Optimization**:
```bash
python main.py --action pipeline --models deep_gpcm attn_gpcm_learn \
    --dataset synthetic_500_200_4 --epochs 30 \
    --hyperopt --hyperopt_trials 30 --adaptive
```

## Key Features

### Adaptive Epoch Allocation
- **Early Phase**: 5 epochs for quick exploration
- **Standard Phase**: 15 epochs for most trials  
- **Full Phase**: 40 epochs for promising configurations

### Expanded Search Space
- **Architectural Parameters**: `key_dim`, `value_dim`, `embed_dim`, `n_heads` (model-dependent)
- **Learning Parameters**: `lr`, `weight_decay`, `batch_size`, `grad_clip`, `label_smoothing`
- **Loss Optimization**: Automatic loss weight optimization

### Fallback System
- Automatically falls back to original optimization if adaptive features fail
- Configurable failure thresholds and safety mechanisms

## Available Models

- `deep_gpcm`: Core Deep-GPCM with DKVMN memory
- `attn_gpcm_learn`: Attention-enhanced with learnable embeddings
- `attn_gpcm_linear`: Attention-enhanced with linear embeddings  
- `stable_temporal_attn_gpcm`: Production-ready temporal attention model

## Common Usage Patterns

**Development/Testing**:
```bash
python train.py --model deep_gpcm --dataset synthetic_500_200_4 \
    --epochs 10 --hyperopt --hyperopt_trials 10
```

**Production Training**:
```bash
python main.py --action pipeline --models deep_gpcm attn_gpcm_learn \
    --dataset your_dataset --epochs 50 \
    --hyperopt --hyperopt_trials 100 --adaptive
```

**Evaluation Only**:
```bash
python main.py --action evaluate --models deep_gpcm \
    --dataset synthetic_500_200_4
```

## Outputs

- **Models**: `results/{dataset}/models/best_{model}.pth`
- **Metrics**: `results/{dataset}/metrics/train_{model}.json`
- **Plots**: `results/{dataset}/plots/` (9 visualization types)
- **IRT Analysis**: `results/{dataset}/irt_plots/`
- **Hyperopt Reports**: `results/hyperopt/{model}_{dataset}_*`
- **Adaptive Reports**: `results/adaptive_hyperopt/adaptive_report_{model}_{dataset}.json`