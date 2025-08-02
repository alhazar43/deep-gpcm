# CORAL Legacy Removal - Implementation Details

## Detailed Line-by-Line Changes

### 1. `/models/factory.py` - Critical Changes

**Remove imports (lines 6, 11-19):**
```python
# DELETE Line 6:
from .implementations.coral_gpcm import HybridCORALGPCM, EnhancedCORALGPCM

# DELETE Lines 11-19:
try:
    from .implementations.adaptive_coral_gpcm import AdaptiveCORALGPCM
except ImportError:
    AdaptiveCORALGPCM = None

try:
    from .implementations.full_adaptive_coral_gpcm import FullAdaptiveCORALGPCM
except ImportError:
    FullAdaptiveCORALGPCM = None
```

**Remove model creation cases (lines 65-74, 86-180):**
```python
# DELETE Lines 65-74 (coral_gpcm case)
# DELETE Lines 86-100 (ecoral_gpcm case)
# DELETE Lines 101-119 (minimal_adaptive_coral_gpcm case)
# DELETE Lines 121-143 (adaptive_coral_gpcm case)
# DELETE Lines 145-179 (full_adaptive_coral_gpcm case)
```

**Update error message (lines 182-184):**
```python
# CHANGE FROM:
raise ValueError(f"Unknown model type: {model_type}. "
                f"Available: deep_gpcm, attn_gpcm, coral_gpcm, coral_gpcm_proper, ecoral_gpcm, "
                f"minimal_adaptive_coral_gpcm, adaptive_coral_gpcm, full_adaptive_coral_gpcm")
# TO:
raise ValueError(f"Unknown model type: {model_type}. "
                f"Available: deep_gpcm, attn_gpcm, coral_gpcm_proper")
```

### 2. `/models/__init__.py` - Clean Imports

**Remove legacy imports (line 16):**
```python
# DELETE Line 16:
from .implementations.coral_gpcm import HybridCORALGPCM, EnhancedCORALGPCM
```

**Update __all__ list (lines 40-41):**
```python
# DELETE Lines 40-41:
'HybridCORALGPCM',
'EnhancedCORALGPCM',
```

### 3. `/models/implementations/__init__.py` - Remove Exports

**Remove legacy imports and exports:**
```python
# DELETE Line 3:
from .coral_gpcm import HybridCORALGPCM, EnhancedCORALGPCM

# DELETE Lines 9-10 from __all__:
'HybridCORALGPCM',
'EnhancedCORALGPCM',
```

### 4. `/train.py` - Simplify Model Handling

**Update model kwargs handling (lines 527-533):**
```python
# CHANGE FROM:
if model_name in ['coral_gpcm', 'ecoral_gpcm', 'adaptive_coral_gpcm']:
    model_kwargs.update({
        'enable_threshold_coupling': args.enable_threshold_coupling or (model_name in ['ecoral_gpcm']),
        'coupling_type': args.coupling_type,
        'gpcm_weight': args.threshold_gpcm_weight,
        'coral_weight': args.threshold_coral_weight
    })

# TO:
# Remove this entire block - coral_gpcm_proper doesn't need special handling
```

### 5. `/evaluate.py` - Update Model Detection

**Update model type detection (lines 681-690):**
```python
# CHANGE FROM:
if 'coral_gpcm_proper' in model_path:
    model_type = 'coral_gpcm_proper'
    print(f"✓ Using CORAL-GPCM proper model")
elif 'coral_gpcm' in model_path:
    model_type = 'coral_gpcm'
    print(f"✓ Using CORAL-GPCM model")
elif 'ecoral' in model_path:
    model_type = 'ecoral_gpcm'
elif 'adaptive_coral' in model_path:
    model_type = 'adaptive_coral_gpcm'

# TO:
if 'coral_gpcm' in model_path:
    model_type = 'coral_gpcm_proper'
    print(f"✓ Using CORAL-GPCM proper model")
```

**Remove legacy model creation cases (lines 202-243):**
```python
# DELETE Lines 202-211 (coral_gpcm case)
# DELETE Lines 212-221 (ecoral_gpcm case)
# DELETE Lines 222-232 (adaptive_coral_gpcm case)
# DELETE Lines 233-243 (full_adaptive_coral_gpcm case)
```

### 6. `/analysis/irt_analysis.py` - Clean Analysis

**Update model type detection (lines 134-139):**
```python
# CHANGE FROM:
if 'coral_gpcm_proper' in model_path or 'CORALGPCM' in str(type(model)):
    model_type = 'coral_gpcm_proper'
    print(f"  Model type: CORAL-GPCM (Proper)")
else:
    # Default to coral_gpcm for backward compatibility
    model_type = 'coral_gpcm'

# TO:
model_type = 'coral_gpcm_proper'
print(f"  Model type: CORAL-GPCM (Proper)")
```

**Remove legacy handling (lines 184-221):**
```python
# DELETE all cases for coral_gpcm, ecoral_gpcm, adaptive_coral_gpcm
# Keep only coral_gpcm_proper handling
```

### 7. File Deletions

```bash
# Delete legacy implementation files
rm /home/steph/dirt-new/deep-gpcm/models/implementations/coral_gpcm.py
rm /home/steph/dirt-new/deep-gpcm/models/implementations/adaptive_coral_gpcm.py
rm /home/steph/dirt-new/deep-gpcm/models/implementations/full_adaptive_coral_gpcm.py

# Delete legacy training scripts
rm /home/steph/dirt-new/deep-gpcm/train_coral_gpcm_combined.py
rm /home/steph/dirt-new/deep-gpcm/train_coral_gpcm_focal.py
```

### 8. Update Configuration

**No changes needed to `/config.py`** - it only references baseline and akvmn models.

### 9. Migration for Saved Models

Create a simple migration script for users with existing models:

```python
# migrate_coral_models.py
import torch

def migrate_coral_model(old_model_path, new_model_path):
    """Migrate old CORAL models to coral_gpcm_proper format."""
    checkpoint = torch.load(old_model_path, map_location='cpu')
    
    # Update model type in config
    if 'model_type' in checkpoint:
        if checkpoint['model_type'] in ['coral_gpcm', 'ecoral_gpcm', 
                                        'adaptive_coral_gpcm', 
                                        'minimal_adaptive_coral_gpcm',
                                        'full_adaptive_coral_gpcm']:
            checkpoint['model_type'] = 'coral_gpcm_proper'
    
    # Save migrated model
    torch.save(checkpoint, new_model_path)
    print(f"Migrated {old_model_path} -> {new_model_path}")
```

## Testing Plan

### 1. Unit Tests
```bash
# Test model creation
python -c "from models import create_model; m = create_model('deep_gpcm', 100, 4)"
python -c "from models import create_model; m = create_model('attn_gpcm', 100, 4)"
python -c "from models import create_model; m = create_model('coral_gpcm_proper', 100, 4)"

# Test that legacy models fail
python -c "from models import create_model; m = create_model('coral_gpcm', 100, 4)"  # Should fail
```

### 2. Training Tests
```bash
# Test training with each model
python train.py --dataset synthetic_OC --models deep_gpcm --epochs 1
python train.py --dataset synthetic_OC --models attn_gpcm --epochs 1
python train.py --dataset synthetic_OC --models coral_gpcm_proper --epochs 1
```

### 3. Evaluation Tests
```bash
# Test evaluation with saved models
python evaluate.py --model_path save_models/deep_gpcm_synthetic_OC.pth
python evaluate.py --model_path save_models/coral_gpcm_proper_synthetic_OC.pth
```

## Verification Checklist

- [ ] All imports updated in __init__.py files
- [ ] Factory function only creates 3 models
- [ ] train.py works with all 3 models
- [ ] evaluate.py correctly detects model types
- [ ] IRT analysis works with coral_gpcm_proper
- [ ] No broken imports in active code
- [ ] Documentation updated
- [ ] Migration script created
- [ ] All tests pass

## Benefits Achieved

1. **Code reduction**: ~2000+ lines removed
2. **Clarity**: Only 3 distinct models with clear purposes
3. **Maintainability**: Easier to debug and enhance
4. **Performance**: Focus optimization on production models
5. **Documentation**: Simpler user guide