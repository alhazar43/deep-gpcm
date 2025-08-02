
# Loss Module Migration Plan

## Phase 1: Preparation (Current)
1. ✓ Created unified loss module (losses_unified.py)
2. ✓ Includes all functionality from both files
3. ✓ Added enhanced focal loss implementation
4. ✓ Maintained backward compatibility

## Phase 2: Testing
1. Run test suite with unified module
2. Verify all loss functions work correctly
3. Check gradient flow and numerical stability

## Phase 3: Migration
1. Backup original files
2. Replace losses.py with losses_unified.py
3. Update all imports automatically
4. Remove ordinal_losses.py

## Phase 4: Cleanup
1. Update __init__.py exports
2. Update documentation
3. Remove backup files after verification

## Import Changes Required:
- `from training.ordinal_losses import X` → `from training.losses import X`
- All loss classes remain the same
- Factory function `create_loss_function` now included in main module

## New Features Added:
1. Enhanced FocalLoss with:
   - Per-class alpha weights support
   - Mask support for sequence data
   - Proper device handling
   
2. Unified factory function:
   - `create_loss_function(loss_type, n_cats, **kwargs)`
   - Supports all loss types including 'focal'
   
3. Comprehensive testing:
   - `test_all_losses()` function included
