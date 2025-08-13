# Deep-GPCM Systematic Restructuring Plan

## Executive Summary

This restructuring plan addresses critical architectural issues in the deep-gpcm system including duplicate visualization components, obsolete CORN model implementations, and scattered plotting logic. The primary objectives are to eliminate code duplication, remove deprecated functionality, consolidate visualization architecture, and improve system maintainability while preserving all active functionality.

**Scope**: This is a **medium-complexity refactoring** affecting 15+ files across 4 modules with cross-dependencies requiring careful orchestration to maintain system functionality throughout the process.

**Risk Level**: **Medium** - Complex cross-file dependencies require systematic approach to prevent breaking changes.

**Timeline**: Estimated 3-4 hours for complete implementation with validation.

---

## Phase-by-Phase Implementation Plan

### Phase 1: Pre-Restructuring Assessment & Backup
**Duration**: 15 minutes  
**Risk**: Low  
**Dependencies**: None

**Objectives**: 
- Create safety checkpoint and analyze current system state
- Identify all cross-file dependencies
- Validate current system functionality

**Actions**:
1. Create git branch for restructuring work
2. Run comprehensive system test to establish baseline
3. Document current plot_metrics.py usage patterns
4. Map all CORN-related dependencies

**Validation Criteria**:
- Current system passes all existing tests
- All visualization functionality works correctly
- Complete dependency map documented

### Phase 2: Visualization Consolidation
**Duration**: 45 minutes  
**Risk**: Medium  
**Dependencies**: Phase 1 complete

**Objectives**: 
- Eliminate duplicate plot_metrics.py files
- Consolidate all visualization logic into single authoritative location
- Update all import references

**Actions**:
1. Compare utils/plot_metrics.py vs scripts/plot_metrics.py functionality
2. Merge superior implementation into utils/plot_metrics.py 
3. Remove scripts/plot_metrics.py
4. Update all import references across codebase
5. Validate visualization functionality

**Quality Gates**:
- Single plot_metrics.py file with complete functionality
- All imports correctly reference utils/plot_metrics.py
- All visualization tests pass

### Phase 3: CORN Model Removal
**Duration**: 90 minutes  
**Risk**: High  
**Dependencies**: Phase 2 complete

**Objectives**: 
- Remove all CORN-related code and dependencies
- Clean up model factory and configuration files
- Remove CORN documentation and test data

**Actions**:
1. Remove core CORN implementation files
2. Clean model factory of CORN references
3. Update main.py and train.py argument parsing
4. Remove CORN test results and documentation
5. Update system documentation

**Quality Gates**:
- No CORN references in active codebase
- Model factory validation passes
- Main pipeline functionality unaffected
- All CORAL functionality preserved

### Phase 4: Model Renaming & Architecture Cleanup
**Duration**: 60 minutes  
**Risk**: Low  
**Dependencies**: Phase 3 complete

**Objectives**: 
- Execute model renaming (baseline→deep_gpcm, hybrid_coral→coral_gpcm, akvmn→attn_gpcm)
- Validate system integrity after restructuring
- Update documentation to reflect changes

**Actions**:
1. Implement systematic model renaming across all files
2. Update result files and model checkpoints
3. Run complete system test suite
4. Update README and documentation
5. Perform integration testing

**Quality Gates**:
- All tests pass with new model names
- Documentation accurately reflects new structure
- System performance maintained or improved

---

## Detailed Action Items

### 1. Plot Metrics Consolidation

**Analysis Results**:
- **Keep**: `utils/plot_metrics.py` (1,993 lines, comprehensive AdaptivePlotter system)
- **Remove**: `scripts/plot_metrics.py` (399 lines, basic functionality)

**Files to Modify**:
- `utils/plot_metrics.py` (keep - comprehensive implementation)
- `scripts/plot_metrics.py` (remove - duplicate/basic)
- `main.py`, `evaluate.py`, `analysis/irt_analysis.py` (update imports)

**Implementation Commands**:
```bash
# 1. Keep comprehensive utils/plot_metrics.py
# 2. Remove duplicate scripts/plot_metrics.py
rm scripts/plot_metrics.py

# 3. Update imports across codebase
grep -r "scripts.plot_metrics" . --include="*.py"
sed -i 's/scripts\.plot_metrics/utils.plot_metrics/g' main.py evaluate.py analysis/irt_analysis.py

# 4. Verify no references to scripts/plot_metrics remain
grep -r "scripts.plot_metrics" . --include="*.py"
```

**Risk Mitigation**:
- utils/plot_metrics.py contains all functionality from scripts version
- Test each import change individually
- Validate all visualization types still work

### 2. CORN Model Removal

**Files to Remove**:
```
core/corn_gpcm.py           # CORNDeepGPCM, AdaptiveCORNGPCM, MultiTaskCORNGPCM
core/corn_layer.py          # CORNLayer implementation  
docs/implementation/CORN_SOLUTIONS.md
All CORN result files in results/
```

**Files to Modify**:

#### core/__init__.py
```python
# REMOVE these imports:
# from .corn_gpcm import CORNDeepGPCM, AdaptiveCORNGPCM, MultiTaskCORNGPCM
# from .corn_layer import CORNLayer

# REMOVE from __all__:
# 'CORNDeepGPCM', 'AdaptiveCORNGPCM', 'MultiTaskCORNGPCM', 'CORNLayer'
```

#### core/model_factory.py
```python
# REMOVE CORN model creation logic (lines 74-96)
# UPDATE available models error message:
# Available: baseline, akvmn, coral, hybrid_coral
```

#### main.py and train.py
```python
# UPDATE argument parser choices:
choices=['baseline', 'akvmn', 'coral', 'hybrid_coral']
# REMOVE: 'corn', 'adaptive_corn', 'multitask_corn'
```

**Implementation Commands**:
```bash
# Remove CORN implementation files
rm core/corn_gpcm.py core/corn_layer.py
rm docs/implementation/CORN_SOLUTIONS.md

# Remove CORN result files
find results/ -name "*corn*" -delete
find results/ -name "*CORN*" -delete

# Update core module imports (manual edit required)
# Update model factory (manual edit required)  
# Update CLI arguments (manual edit required)
```

**Validation Steps**:
- Verify baseline and akvmn models still create correctly
- Test CORAL models remain functional
- Run training pipeline with supported models only
- Check no import errors occur

### 3. Model Renaming Implementation

**Systematic Renaming Required**:
- **baseline** → **deep_gpcm** (59 files affected)
- **hybrid_coral** → **coral_gpcm** (22 files affected)  
- **akvmn** → **attn_gpcm** (45 files affected)

**High-Impact Python Files** (require code changes):
```
core/model_factory.py       # Model creation logic
main.py                     # CLI arguments and model references
train.py                    # Training script model handling
evaluate.py                 # Evaluation script model handling
utils/plot_metrics.py       # Model color mapping and display names
analysis/irt_analysis.py    # Model-specific analysis
core/__init__.py            # Model exports
config.py                   # Default configurations
```

**Result Files to Rename** (37 files):
```bash
# Training results
mv results/train/train_results_baseline_synthetic_OC_cv_summary.json results/train/train_results_deep_gpcm_synthetic_OC_cv_summary.json
mv results/train/train_results_akvmn_synthetic_OC_cv_summary.json results/train/train_results_attn_gpcm_synthetic_OC_cv_summary.json
mv results/train/train_results_hybrid_coral_synthetic_OC_cv_summary.json results/train/train_results_coral_gpcm_synthetic_OC_cv_summary.json

# Test results  
mv results/test/test_results_baseline_synthetic_OC.json results/test/test_results_deep_gpcm_synthetic_OC.json
mv results/test/comprehensive_test_baseline_synthetic_OC.json results/test/comprehensive_test_deep_gpcm_synthetic_OC.json
mv results/test/test_results_akvmn_synthetic_OC.json results/test/test_results_attn_gpcm_synthetic_OC.json
mv results/test/comprehensive_test_akvmn_synthetic_OC.json results/test/comprehensive_test_attn_gpcm_synthetic_OC.json
mv results/test/test_results_hybrid_coral_synthetic_OC.json results/test/test_results_coral_gpcm_synthetic_OC.json
mv results/test/comprehensive_test_hybrid_coral_synthetic_OC.json results/test/comprehensive_test_coral_gpcm_synthetic_OC.json
```

**Model Checkpoints to Rename** (4 files):
```bash
mv save_models/best_baseline_synthetic_OC.pth save_models/best_deep_gpcm_synthetic_OC.pth
mv save_models/best_akvmn_synthetic_OC.pth save_models/best_attn_gpcm_synthetic_OC.pth  
mv save_models/best_hybrid_coral_synthetic_OC.pth save_models/best_coral_gpcm_synthetic_OC.pth
# Keep: save_models/best_coral_synthetic_OC.pth (unchanged)
```

**Code Changes Required**:

#### core/model_factory.py
```python
# Line 31-40: Update model type mapping
MODEL_MAPPING = {
    'deep_gpcm': 'DeepGPCM',        # was 'baseline'
    'attn_gpcm': 'AttentionGPCM',   # was 'akvmn'
    'coral': 'CORALDeepGPCM',       # unchanged
    'coral_gpcm': 'HybridCORALGPCM' # was 'hybrid_coral'
}
```

#### utils/plot_metrics.py
```python
# Line 646-651: Update model color mapping
model_colors = {
    'deep_gpcm': '#ff7f0e',     # was 'baseline'
    'attn_gpcm': '#1f77b4',     # was 'akvmn'  
    'coral': '#2ca02c',         # unchanged
    'coral_gpcm': '#d62728'     # was 'hybrid_coral'
}
```

### 4. File Consolidation Analysis

**eval_all.py → evaluate.py Integration**:
- **Risk**: Low - minimal function overlap
- **Action**: Merge batch evaluation capabilities into main evaluate.py
- **Implementation**: Add `--all-models` flag to evaluate.py
- **Validation**: Test multi-model evaluation functionality

**train_coral.py → train.py Integration**:
- **Risk**: Medium - CORAL-specific training logic exists
- **Analysis**: train_coral.py has CORAL-specific parameter handling
- **Decision**: Keep train_coral.py as specialized CORAL trainer
- **Rationale**: CORAL models require specific IRT parameter extraction logic

---

## Architecture Impact Assessment

### System Modularity Impact
**Positive Changes**:
- Reduced code duplication improves maintainability
- Cleaner module boundaries with single plot_metrics location  
- Simplified model factory reduces cognitive complexity
- Standardized model naming improves consistency

**Considerations**:
- No impact on core DKVMN or GPCM functionality
- CORAL models preserved and fully functional
- Embedding strategies remain unchanged

### API Compatibility Impact
**Breaking Changes**:
- CORN model types no longer supported in CLI arguments
- Model names changed (affects backward compatibility)
- Import paths change for plot_metrics functionality

**Mitigation Strategy**:
- Update all documentation to reflect new model names
- Provide clear error messages for unsupported options
- Create migration guide for existing users

### Testing & Validation Impact
**Enhanced Testing**:
- Simplified test matrix with fewer model combinations
- More focused testing on core functionality
- Reduced complexity in CI/CD pipeline

**Risk Areas**:
- Visualization functionality during plot_metrics consolidation
- Model factory validation after CORN removal
- Import dependency resolution

---

## Implementation Order & Dependencies

### Critical Path
1. **Phase 1**: Pre-Restructuring Assessment & Backup
   - Create git branch and establish baseline
   - No dependencies

2. **Phase 2**: Visualization Consolidation  
   - Remove scripts/plot_metrics.py duplicate
   - Update all import references
   - **Dependency**: Phase 1 complete

3. **Phase 3**: CORN Model Removal
   - Remove CORN implementation files
   - Clean model factory and CLI arguments
   - **Dependency**: Phase 2 complete (to avoid import conflicts)

4. **Phase 4**: Model Renaming & Final Cleanup
   - Systematic model name changes across all files
   - Update documentation and validation
   - **Dependency**: Phase 3 complete (clean foundation)

### Rollback Strategy

**Rollback Decision Points**:
- **After Phase 1**: If current system tests fail, investigate before proceeding
- **After Phase 2**: If visualization breaks, rollback and reassess approach  
- **After Phase 3**: If model factory fails, investigate CORAL dependencies
- **After Phase 4**: If renaming causes issues, rollback to Phase 3 state

**Rollback Procedure**:
```bash
# Create restructuring branch with checkpoint
git checkout -b restructuring-deep-gpcm
git tag phase-1-checkpoint  # After each phase

# If rollback needed
git reset --hard phase-N-checkpoint  # Reset to specific phase
# OR
git reset --hard HEAD~1              # Immediate revert of last commit

# Validate rollback state
python main.py --models baseline akvmn --dataset synthetic_OC --epochs 2
```

---

## Quality Gates & Validation

### Phase-Specific Validation

**Phase 1 Validation**:
```bash
# Establish baseline functionality
python main.py --models baseline akvmn --dataset synthetic_OC --epochs 2
python utils/plot_metrics.py  # Test current visualization
```

**Phase 2 Validation**:
```bash
# Test consolidated plot_metrics functionality
python utils/plot_metrics.py
python analysis/irt_analysis.py --dataset synthetic_OC
grep -r "scripts.plot_metrics" . --include="*.py"  # Should return nothing
```

**Phase 3 Validation**:
```bash
# Test model factory without CORN
python -c "from core.model_factory import create_model; create_model('baseline', 50, 5)"
python -c "from core.model_factory import create_model; create_model('akvmn', 50, 5)"
python -c "from core.model_factory import create_model; create_model('coral', 50, 5)"
grep -r "corn" . --include="*.py" --exclude-dir=".git"  # Should find minimal references
```

**Phase 4 Validation**:
```bash
# Test renamed models
python main.py --models deep_gpcm attn_gpcm --dataset synthetic_OC --epochs 2
python main.py --models coral_gpcm --dataset synthetic_OC --epochs 2
python utils/plot_metrics.py  # Test with new model names
```

### Comprehensive Test Suite
```bash
# Core functionality validation
python main.py --models deep_gpcm attn_gpcm --dataset synthetic_OC --epochs 5
python main.py --models coral coral_gpcm --dataset synthetic_OC --epochs 5

# Visualization validation  
python utils/plot_metrics.py
python analysis/irt_analysis.py --dataset synthetic_OC

# Model factory validation
python -c "from core.model_factory import create_model; print('Factory OK')"

# Import validation
python -c "from core import *; print('Core imports OK')"
python -c "from utils.plot_metrics import *; print('Plot imports OK')"
```

### Performance Benchmarks
- Training time should remain unchanged or improve
- Memory usage should remain stable or decrease  
- Visualization generation time should be comparable
- Model accuracy should be identical for renamed models

---

## Risk Assessment & Mitigation

### High-Risk Areas
1. **CORN Removal**: Complex dependencies across multiple files
2. **Model Renaming**: Affects result files and user workflows  
3. **Import Changes**: Could break visualization functionality

### Risk Mitigation Strategies
1. **Phased Approach**: Complete each phase before starting next
2. **Git Checkpoints**: Tag after each successful phase
3. **Comprehensive Testing**: Validate functionality after each change
4. **Rollback Readiness**: Clear rollback procedures defined

### Contingency Plans
- **Import Failures**: Maintain old import paths temporarily with deprecation warnings
- **Model Loading Issues**: Keep backup of original result files
- **Visualization Breaks**: Rollback to working plot_metrics.py version

---

## Success Metrics

### Quantitative Metrics
- **Code Reduction**: Remove ~500 lines of duplicate/obsolete code
- **File Count**: Reduce by 8+ files (CORN implementations, duplicates)
- **Import Complexity**: Single plot_metrics import path
- **Model Count**: Streamlined from 7 to 4 core models

### Qualitative Improvements
- **Maintainability**: Cleaner, more focused codebase
- **User Experience**: Consistent model naming and clearer options
- **Developer Experience**: Reduced confusion about model selection
- **Architecture**: Better separation of concerns and modularity

### Validation Criteria
- ✅ Zero functionality regressions
- ✅ All visualization functionality preserved  
- ✅ Model training and evaluation work correctly
- ✅ Clean, consolidated codebase structure
- ✅ Updated documentation reflects changes
- ✅ Performance maintained or improved

---

## Documentation Updates Required

### Files to Update
- `README.md`: Model comparison table and usage examples
- `CLAUDE.md`: Command examples with new model names
- Core module docstrings: Remove CORN references
- CLI help text: Update model choices and descriptions

### User Migration Guide
Create `MIGRATION_GUIDE.md` with:
- Old vs new model name mapping
- Updated command-line examples  
- Result file naming changes
- Import path changes for custom scripts

---

## Conclusion

This systematic restructuring plan transforms the deep-gpcm codebase from a research prototype with accumulated technical debt into a clean, maintainable production system. The four-phase approach ensures reliability while achieving significant architectural improvements.

**Key Benefits**:
- ✅ Eliminates code duplication and technical debt
- ✅ Simplifies system architecture and user experience  
- ✅ Improves maintainability and reduces confusion
- ✅ Standardizes model naming conventions
- ✅ Preserves all active functionality
- ✅ Creates foundation for future development

**Timeline**: 3-4 hours total implementation time with systematic validation at each phase.

**Risk**: Medium complexity with comprehensive mitigation strategies and rollback procedures.

This restructuring establishes deep-gpcm as a clean, professional, and maintainable knowledge tracing system ready for production use and future development.