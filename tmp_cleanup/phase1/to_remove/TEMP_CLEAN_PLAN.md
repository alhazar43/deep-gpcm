# Deep-GPCM Cleanup Plan

## Overview
This document outlines the complete cleanup plan to streamline the Deep-GPCM codebase to only essential components.

## Core Components to Preserve

### 1. Main Pipeline Files
- `main.py` - Main orchestrator
- `train.py` - Training pipeline  
- `evaluate.py` - Evaluation pipeline
- `utils/plot_metrics.py` - Visualization (already in correct location)
- `analysis/irt_analysis.py` - IRT analysis (already in correct location)

### 2. Core Model Files to Keep
- `core/model.py` - Contains DeepGPCM and AttentionGPCM base classes
- `core/attention_enhanced.py` - Contains EnhancedAttentionGPCM (used as attn_gpcm)
- `core/coral_gpcm.py` - Contains HybridCORALGPCM (used as coral_gpcm) and EnhancedCORALGPCM
- `core/full_adaptive_blender.py` - Contains FullAdaptiveBlender (used by EnhancedCORALGPCM for full_adaptive_coral_gpcm configuration)
- `core/model_factory.py` - Model creation factory

**NOTE**: `full_adaptive_coral_gpcm` is NOT a separate class but a configuration of `EnhancedCORALGPCM` with FullAdaptiveBlender.

### 3. Supporting Core Files
- `core/__init__.py`
- `core/layers.py` - Basic layers used by models
- `core/memory_networks.py` - Memory components
- `core/coral_layer.py` - CORAL layer implementation
- `core/embeddings.py` - Embedding strategies

### 4. Essential Utils
- `utils/__init__.py`
- `utils/data_utils.py` - Data loading utilities
- `utils/metrics.py` - Metric calculations
- `utils/data_gen.py` - Synthetic data generation (to be moved from scripts/)

### 5. Training Components
- `training/__init__.py`
- `training/losses.py` - Basic loss functions
- `training/ordinal_losses.py` - Ordinal loss implementations

### 6. Documentation to Keep (4 files only)
- `README.md`
- `SUMMARY.md`
- `MATH.md`
- `TODO.md`

### 7. Other Essential Files
- `requirements.txt`
- `.gitignore`
- `config.py` (needs update to remove duplicates)

## Files and Directories to Remove

### 1. All Test/Debug/Demo Files (18 files)
```
debug_adaptive.py
debug_blender_training.py
debug_combined_loss.py
debug_data_scale.py
debug_model_integration.py
debug_parameter_analysis.py
debug_real_training.py
debug_simple_adaptive.py
debug_training_step.py
demo_adaptive_blending.py
demonstrate_adaptive_thresholding.py
quick_adaptive_test.py
test_adaptive_coupling.py
test_backward_compatibility.py
test_complete_integration.py
test_gradient_stability.py
test_loss_comparison.py
test_multi_model_train.py
```

### 2. Experimental/Analysis Scripts (not in analysis/)
```
analyze_adaptive_results.py
analyze_real_dataset.py
compare_adaptive_models.py
convert_dataset_format.py
eval_all.py
generate_large_synthetic.py
improved_analysis.py
phase2_final_analysis.py
quick_preliminary_analysis.py
simple_threshold_validation.py
train_adaptive_example.py
train_adaptive_experiment.py
train_coral.py
validate_threshold_coupling.py
```

### 3. Unused Core Model Files
```
core/threshold_blender.py
core/minimal_adaptive_blender.py
core/safe_threshold_blender.py
core/stable_threshold_blender.py
core/threshold_coupling.py
```

### 4. Unused Training Files
```
training/balanced_losses.py
```

### 5. Documentation to Remove (24 files)
```
ADAPTIVE_BLENDING_SUMMARY.md
BETA_EXTRACTION_REPORT.md
CLAUDE.md
GRADIENT_STABILITY_SOLUTION.md
IRT_ANALYSIS_INTEGRATION_SUMMARY.md
MODEL_TYPES_REFERENCE.md
PIPELINE_RESULTS.md
RESTRUCTURING_PLAN.md
docs/ (entire directory)
research/ (entire directory)
.claude/ (entire directory)
.gemini/ (entire directory)
```

### 6. Analysis Files to Remove (not core irt_analysis.py)
```
analysis/CORAL_DESIGN_FLAW_ANALYSIS.md
analysis/animate_irt.py
analysis/beta_extraction_summary.py
analysis/check_beta_composition.py
analysis/coral_threshold_investigation.py
analysis/extract_beta_params.py
analysis/plot_irt.py
analysis/tau_usage_analysis.py
analysis/verify_beta_extraction.py
```

### 7. Empty/Unused Directories
```
models/ (empty directory)
evaluation/ (empty directory)
tests/ (empty directory)
examples/ (entire directory)
scripts/ (entire directory - after moving data_gen.py to utils/)
```

### 8. Results and Saved Models
```
results/ (entire directory - all plots, JSONs, etc.)
save_models/ (entire directory - all saved model files)
```

### 9. Data Backup
```
data/synthetic_OC_large_backup/ (backup directory)
```

### 10. Cache Directories
```
__pycache__/
core/__pycache__/
training/__pycache__/
utils/__pycache__/
analysis/__pycache__/
```

## Files to Move

### 1. scripts/data_gen.py → utils/data_gen.py
- Move the synthetic data generation script to utils directory
- Update any imports in files that use data_gen.py

## Files to Modify

### 1. config.py
- Remove duplicate AKVMN_CONFIG entries (lines 93-100 and 128)
- Update supported model references

### 2. core/model_factory.py
- Verify it handles the 4 core models with consistent naming:
  - `deep_gpcm` → DeepGPCM
  - `attn_gpcm` → EnhancedAttentionGPCM
  - `coral_gpcm` → HybridCORALGPCM
  - `full_adaptive_coral_gpcm` → EnhancedCORALGPCM (with FullAdaptiveBlender)
- The factory already handles these correctly, just verify after cleanup

### 3. main.py
- Update default models list to include all 4 models
- Remove emoji usage if any

### 4. train.py
- Ensure proper model loading for all 4 models
- Check imports are correct after cleanup

### 5. evaluate.py
- Ensure proper model loading for all 4 models
- Check imports are correct after cleanup

### 6. utils/plot_metrics.py
- Ensure it handles all 4 model types correctly

### 7. analysis/irt_analysis.py
- Update imports to include FullAdaptiveCORALGPCM
- Ensure it handles all 4 model types

## Dependencies to Update

### 1. Import Updates Needed
After removing files, these imports need to be checked/updated:
- Remove any imports from deleted experimental modules
- Ensure all core model imports use the correct paths
- Update any references to removed analysis scripts

### 2. Model Name Consistency
Ensure consistent naming across all files:
- `deep_gpcm` (not deep or DeepGPCM in configs)
- `attn_gpcm` (not attn or attention_gpcm)
- `coral_gpcm` (not coral or hybrid_coral)
- `full_adaptive_coral_gpcm` (not full_adaptive or adaptive_coral)

## Verification Steps

### 1. Pre-cleanup Verification
- [ ] Backup the entire directory
- [ ] Ensure git status is clean (commit any pending changes)

### 2. Post-cleanup Verification
- [ ] Run `python main.py --dataset synthetic_OC --models all --epochs 5` to test pipeline
- [ ] Verify each model can be trained individually
- [ ] Check evaluate.py works with saved models
- [ ] Verify plot_metrics.py generates plots
- [ ] Test irt_analysis.py functionality
- [ ] Ensure no broken imports
- [ ] Check all 4 core documents are present and updated
- [ ] Run validation script to verify model factory:
```python
from core.model_factory import create_model

models = {
    'deep_gpcm': 'DeepGPCM',
    'attn_gpcm': 'EnhancedAttentionGPCM', 
    'coral_gpcm': 'HybridCORALGPCM',
    'full_adaptive_coral_gpcm': 'EnhancedCORALGPCM'
}

for model_type, expected_class in models.items():
    model = create_model(model_type, n_questions=50, n_cats=4)
    assert model.__class__.__name__ == expected_class
    print(f"✓ {model_type} correctly creates {expected_class}")
```

### 3. Final Checks
- [ ] No test/debug/demo files remain
- [ ] Only 4 documentation files in root
- [ ] Clean directory structure
- [ ] All experimental code removed
- [ ] Consistent model naming throughout

## Summary Statistics
- **Files to remove**: ~89 files (1 less due to keeping data_gen.py)
- **Directories to remove**: ~15 directories  
- **Files to move**: 1 file (scripts/data_gen.py → utils/data_gen.py)
- **Files to modify**: 7 files
- **Core files to keep**: ~21 files (including moved data_gen.py)
- **Final documentation**: 4 files (README.md, SUMMARY.md, MATH.md, TODO.md)
- **Final models**: 4 models (deep_gpcm, attn_gpcm, coral_gpcm, full_adaptive_coral_gpcm)

## Architecture Review Recommendations

### 1. Model Naming Convention
- **Decision**: Use `model_type` as the consistent identifier
- **Convention**: Underscore for types (e.g., `deep_gpcm`), CamelCase for classes (e.g., `DeepGPCM`)
- **Documentation**: Clearly document that `full_adaptive_coral_gpcm` is a configuration, not a separate class

### 2. Additional Directories to Check
```bash
# Verify these are truly empty before removal
ls -la models/
ls -la evaluation/  
ls -la tests/
```

### 3. Critical Preservation Notes
- The model factory pattern is already correct - no changes needed
- `core/full_adaptive_blender.py` must be kept as it's used by EnhancedCORALGPCM
- All 4 model configurations are already properly handled in evaluate.py and train.py

### 4. Post-Cleanup Documentation Update
Add to TODO.md:
```markdown
## Model Architecture Clarification
- `deep_gpcm`: Uses DeepGPCM class
- `attn_gpcm`: Uses EnhancedAttentionGPCM class
- `coral_gpcm`: Uses HybridCORALGPCM class  
- `full_adaptive_coral_gpcm`: Uses EnhancedCORALGPCM class with FullAdaptiveBlender configuration
```

Update README.md data generation section:
```bash
# Update from:
python scripts/data_gen.py --format OC --categories 4 --students 800 --questions 400

# To:
python utils/data_gen.py --format OC --categories 4 --students 800 --questions 400
```

## Final Approval
✅ **This cleanup plan is APPROVED by both Research Scientist and Systems Architect**
- Preserves all essential functionality
- Maintains architectural integrity
- Creates a clean, production-ready codebase
- Keeps exactly 4 models and 4 documentation files as requested