# Deep-GPCM Loss Module Architecture

## Overview

The unified loss module provides a comprehensive collection of loss functions specifically designed for ordinal regression and knowledge tracing tasks. The architecture combines standard losses with ordinal-specific implementations and supports multi-objective optimization through combined loss functions.

## Architecture Components

### 1. Base Loss Functions

#### WeightedCrossEntropyLoss
- **Purpose**: Handles class imbalance through per-class weights
- **Use Case**: When certain response categories are underrepresented
- **Input**: Logits and optional class weights

#### FocalLoss (Enhanced)
- **Purpose**: Addresses class imbalance by focusing on hard examples
- **Features**:
  - Configurable gamma parameter for focusing strength
  - Per-class alpha weights support
  - Mask support for sequence data
  - Proper device handling
- **Use Case**: Extreme class imbalance or when easy examples dominate training

### 2. Ordinal-Specific Loss Functions

#### DifferentiableQWKLoss
- **Purpose**: Directly optimizes Quadratic Weighted Kappa (QWK)
- **Method**: Soft confusion matrix computation
- **Strengths**: Direct optimization of evaluation metric
- **Considerations**: Can be unstable early in training

#### OrdinalEMDLoss
- **Purpose**: Earth Mover's Distance for ordinal regression
- **Method**: L1 distance between cumulative distributions
- **Strengths**: Natural handling of ordinal relationships
- **Use Case**: When ordinal distance matters significantly

#### OrdinalCrossEntropyLoss
- **Purpose**: Distance-weighted cross-entropy for ordinal data
- **Method**: Weights errors by ordinal distance from true category
- **Strengths**: Simple yet effective ordinal awareness
- **Parameters**: Alpha controls distance penalty strength

### 3. Combined Loss Functions

#### CombinedOrdinalLoss
- **Purpose**: Multi-objective optimization with configurable weights
- **Components**:
  - Cross-Entropy (ce_weight)
  - QWK (qwk_weight)
  - EMD (emd_weight)
  - CORAL (coral_weight)
  - Focal (focal_weight)
- **Features**:
  - Automatic logit/probability detection
  - CORAL integration support
  - Component-wise loss tracking

### 4. Factory Function

#### create_loss_function
- **Purpose**: Unified interface for loss creation
- **Supported Types**:
  - 'ce': Standard cross-entropy
  - 'weighted_ce': Weighted cross-entropy
  - 'focal': Focal loss
  - 'qwk': Differentiable QWK
  - 'emd': Earth Mover's Distance
  - 'ordinal_ce': Ordinal cross-entropy
  - 'combined': Customizable combined loss
  - 'triple_coral': Pre-configured CORAL combination

## Design Principles

### 1. Modularity
- Each loss function is self-contained
- Clear separation between base and ordinal losses
- Easy to add new loss functions

### 2. Flexibility
- Support for both logits and probabilities
- Optional masking for sequence data
- Configurable reduction methods

### 3. Performance
- Pre-computed weight matrices
- Efficient tensor operations
- Proper device handling

### 4. Compatibility
- Backward compatible with existing code
- Seamless integration with CORAL layers
- Standard PyTorch patterns

## Usage Patterns

### Basic Usage
```python
from training.losses import create_loss_function

# Create a focal loss for 5 categories
loss_fn = create_loss_function('focal', n_cats=5, gamma=2.0, alpha=0.25)

# Use in training
loss = loss_fn(logits, targets)
```

### Combined Loss
```python
# Create a combined loss with custom weights
loss_fn = create_loss_function(
    'combined',
    n_cats=5,
    ce_weight=1.0,
    qwk_weight=0.5,
    focal_weight=0.3,
    focal_gamma=2.0
)

# Returns dictionary with component losses
loss_dict = loss_fn(predictions, targets, mask)
total_loss = loss_dict['total_loss']
```

### CORAL Integration
```python
# Triple CORAL configuration
loss_fn = create_loss_function('triple_coral', n_cats=5)

# Pass CORAL info for proper integration
loss_dict = loss_fn(predictions, targets, mask, coral_info=coral_info)
```

## Migration from Separate Files

The unified module combines:
- `losses.py` (basic losses)
- `ordinal_losses.py` (ordinal-specific losses)

All imports should now use:
```python
from training.losses import DifferentiableQWKLoss  # Not ordinal_losses
```

## Best Practices

### 1. Loss Selection
- Start with standard CE for baseline
- Add ordinal losses for ordinal data
- Use combined loss for multi-objective optimization

### 2. Weight Tuning
- Begin with equal weights
- Adjust based on validation performance
- Monitor individual loss components

### 3. Stability
- QWK loss can be unstable early in training
- Consider warmup or delayed activation
- Use combined loss for stability

### 4. Class Imbalance
- Use focal loss for extreme imbalance
- WeightedCE for moderate imbalance
- Combine with ordinal losses for best results

## Future Extensions

1. **Learnable Weights**: Dynamic weight adjustment during training
2. **Additional Ordinal Losses**: Unimodal loss, ordinal hinge loss
3. **Curriculum Learning**: Progressive loss complexity
4. **Uncertainty Estimation**: Losses that encourage calibrated predictions

## Testing

Run comprehensive tests:
```python
from training.losses import test_all_losses
test_all_losses()
```

This validates:
- Forward pass for all losses
- Gradient computation
- Device handling
- Mask support