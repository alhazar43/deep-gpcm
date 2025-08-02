# CORAL Legacy Removal Plan - Architectural Review

## Executive Summary

This document provides a comprehensive plan for removing legacy CORAL implementations from the Deep-GPCM codebase while preserving the three core models: `deep_gpcm`, `attn_gpcm`, and `coral_gpcm_proper`.

## 1. Legacy Models to Remove

### 1.1 Core Legacy Models
- **HybridCORALGPCM** (`coral_gpcm`) - Original hybrid implementation
- **EnhancedCORALGPCM** (`ecoral_gpcm`) - Enhanced version with threshold coupling
- **AdaptiveCORALGPCM** (`adaptive_coral_gpcm`) - Adaptive blending variant
- **minimal_adaptive_coral_gpcm** - Minimal adaptive implementation
- **full_adaptive_coral_gpcm** - Full adaptive with larger architecture

### 1.2 Model Locations
- `/models/implementations/coral_gpcm.py` - Contains HybridCORALGPCM and EnhancedCORALGPCM
- `/models/implementations/adaptive_coral_gpcm.py` - May exist (conditional import)
- `/models/implementations/full_adaptive_coral_gpcm.py` - May exist (conditional import)

## 2. Dependencies and Impact Analysis

### 2.1 Primary Import Points
1. **`/models/__init__.py`** (Lines 16, 40-41)
   - Imports: HybridCORALGPCM, EnhancedCORALGPCM
   - Exports in __all__

2. **`/models/implementations/__init__.py`** (Lines 3, 9-10)
   - Imports and exports legacy models

3. **`/models/factory.py`** (Lines 6, 11-19, 65-184)
   - Factory function creates all legacy variants
   - Conditional imports for adaptive models

### 2.2 Training Script Dependencies
1. **`/train.py`** (Lines 527-533)
   - Special handling for coral_gpcm variants
   - Threshold coupling configuration

2. **`/evaluate.py`** (Multiple references)
   - Lines 147, 202, 212, 222, 233, 681-690
   - Model type detection and creation

3. **`/analysis/irt_analysis.py`** (Multiple references)
   - Lines 134, 139, 184, 193, 202, 222
   - IRT parameter extraction for different variants

### 2.3 Utility Dependencies
1. **`/utils/irt_utils.py`** (Lines 26, 114)
   - Special handling for coral_gpcm_proper detection

### 2.4 Test and Script Dependencies
- 56 files reference legacy models (mostly in to_remove/ directory)
- Key active scripts:
  - `train_coral_gpcm_combined.py`
  - `train_coral_gpcm_focal.py`
  - `test_coral_gpcm_proper.py`
  - `compare_all_models_final.py`

## 3. Systematic Removal Plan

### Phase 1: Update Model Registry (Critical Path)

#### 3.1 Update `/models/factory.py`
```python
# Remove lines 6, 11-19 (imports)
# Remove lines 65-74, 86-184 (coral_gpcm, ecoral_gpcm, adaptive variants)
# Update error message (line 182-184) to only list: deep_gpcm, attn_gpcm, coral_gpcm_proper
```

#### 3.2 Update `/models/__init__.py`
```python
# Remove line 16: coral_gpcm imports
# Remove lines 40-41 from __all__
# Keep line 17: CORALGPCM import
```

#### 3.3 Update `/models/implementations/__init__.py`
```python
# Remove line 3: coral_gpcm imports
# Remove lines 9-10 from __all__
```

#### 3.4 Delete Legacy Implementation Files
```bash
rm /models/implementations/coral_gpcm.py
rm /models/implementations/adaptive_coral_gpcm.py  # if exists
rm /models/implementations/full_adaptive_coral_gpcm.py  # if exists
```

### Phase 2: Update Training Infrastructure

#### 3.5 Update `/train.py`
```python
# Simplify lines 527-533
# Remove special handling for coral_gpcm, ecoral_gpcm, adaptive_coral_gpcm
# Keep only coral_gpcm_proper handling if needed
```

#### 3.6 Update `/evaluate.py`
```python
# Update model type detection (lines 681-690)
# Remove references to coral_gpcm, ecoral_gpcm, adaptive_coral_gpcm
# Update create_model calls (lines 202, 212, 222, 233)
```

#### 3.7 Update `/analysis/irt_analysis.py`
```python
# Remove handling for coral_gpcm variants (lines 184, 193, 202, 222)
# Keep only coral_gpcm_proper handling
```

### Phase 3: Clean Configuration and Scripts

#### 3.8 Update comparison scripts
- Update `compare_all_models_final.py` to remove legacy model comparisons
- Update any other comparison scripts

#### 3.9 Clean up training scripts
- Remove or update `train_coral_gpcm_combined.py`
- Remove or update `train_coral_gpcm_focal.py`

### Phase 4: Documentation and Testing

#### 3.10 Update documentation
- Remove references to legacy models from all .md files
- Update model documentation to reflect only 3 models

#### 3.11 Update tests
- Keep `test_coral_gpcm_proper.py`
- Remove tests for legacy models

## 4. Verification Checklist

### 4.1 Pre-removal Verification
- [ ] Ensure coral_gpcm_proper has all necessary features
- [ ] Verify no critical functionality is lost
- [ ] Backup current state

### 4.2 Post-removal Verification
- [ ] All imports resolve correctly
- [ ] Factory creates only 3 models
- [ ] Training script works with all 3 models
- [ ] Evaluation script works with all 3 models
- [ ] No broken references in active code
- [ ] Tests pass for remaining models

## 5. Critical Features to Preserve in coral_gpcm_proper

1. **CORAL ordinal structure** - Properly implemented
2. **IRT parameter extraction** - Working correctly
3. **GPCM probability computation** - Functional
4. **Memory network integration** - Present
5. **Adaptive blending** (if needed) - Can be added as option

## 6. Migration Path for Existing Models

For users with saved models:
1. `coral_gpcm` → migrate to `coral_gpcm_proper`
2. `ecoral_gpcm` → migrate to `coral_gpcm_proper`
3. `adaptive_coral_gpcm` → migrate to `coral_gpcm_proper` with adaptive flag

## 7. Risk Assessment

### Low Risk
- Removing unused adaptive variants
- Cleaning up factory function
- Updating documentation

### Medium Risk
- Updating training/evaluation scripts
- Migrating existing saved models

### High Risk
- Breaking existing workflows
- Loss of experimental features

## 8. Recommended Execution Order

1. **Backup current state**
2. **Update factory and imports** (Phase 1)
3. **Test basic functionality**
4. **Update training infrastructure** (Phase 2)
5. **Test training/evaluation**
6. **Clean scripts and docs** (Phase 3-4)
7. **Final verification**

## 9. Rollback Plan

If issues arise:
1. Restore from backup
2. Identify specific breaking changes
3. Create compatibility layer if needed
4. Proceed with more gradual removal

## 10. Benefits of Simplification

1. **Cleaner architecture** - Only 3 well-defined models
2. **Easier maintenance** - Less code duplication
3. **Clear model hierarchy** - Each model has distinct purpose
4. **Better performance** - Focus optimization on core models
5. **Improved documentation** - Simpler to explain and use