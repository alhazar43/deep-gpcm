# Project Cleanup Plan - Procedural Approach

This document outlines a systematic, phased approach to clean up the project by moving (not removing) deprecated, redundant, and temporary files to a safe temporary directory.

## Safety Protocol: MOVE, DON'T DELETE
All cleanup operations will MOVE files to `/tmp_cleanup/` directory to enable easy restoration if issues arise.

## Phase 1: ZERO RISK Removals
**Estimated Space Savings: 70% of total cleanup benefit**
**Risk Level: None - Safe for immediate execution**

### 1.1 Backup Directories (Massive Space Impact)
- `backups/` - Historical experimental data, safe to archive
- Risk: None - these are explicitly backup/archive files

### 1.2 Development Artifacts
- `*.pyc` files and `__pycache__/` directories
- `.pytest_cache/` directories  
- Temporary result files with date stamps
- Empty directories in results structure

### 1.3 Obsolete Documentation
- Duplicate README files
- Outdated migration documentation
- Development notes and temporary markdown files

## Phase 2: LOW RISK Removals  
**Requires minimal validation - test suite should pass**

### 2.1 Questionable Utility Files
Based on deeper analysis, these files appear unused:
- `utils/data_loading.py` - Not imported by core pipeline
- `utils/organize_plots_by_dataset.py` - Standalone script, not integrated
- `utils/plot_metrics_optimized.py` - Unused optimized version
- `utils/prediction_strategies.py` - Part of unused experimental feature

### 2.2 Development Scripts
- Debug and testing scripts not in main pipeline
- One-off analysis scripts
- Experimental configuration files

## Phase 3: MEDIUM RISK Removals
**Requires comprehensive testing - all models must train/evaluate successfully**

### 3.1 Model Implementations (Requires Factory Registration Check)
These need verification before removal:
- Temporal attention model variants
- Experimental model implementations  
- Legacy model versions not in current factory registry

### 3.2 Potentially Unused Analysis Tools
- `utils/irt_utils.py` - Check if used by IRT analysis pipeline
- `utils/monitoring.py` - Appears to be debugging tool
- Specialized analysis modules

## Phase 4: Architectural Improvements
**Structural cleanup and consolidation**

### 4.1 Directory Structure Optimization
Current: `results/{plots,irt_plots,test,train,validation}/dataset/`
Proposed: `results/dataset/{plots,irt_plots,metrics,models}/`

### 4.2 Configuration Consolidation
- Merge scattered config files into hierarchical system
- Standardize import patterns (relative vs absolute)
- Consolidate duplicate functionality

## Execution Protocol

### Pre-Phase Setup
```bash
# Create temporary cleanup directory
mkdir -p /tmp_cleanup/phase{1,2,3,4}

# Create git branch for each phase
git checkout -b cleanup-phase1
```

### Phase Execution Rules
1. **Move, Never Delete**: `mv file /tmp_cleanup/phaseN/`
2. **Test After Each Phase**: Run full test suite
3. **Git Commit Per Phase**: Clear rollback points
4. **Validation Gates**: All models must train/evaluate successfully
5. **Documentation**: Record all moves for potential restoration

### Success Criteria
- **Functional**: All models train and evaluate successfully  
- **Performance**: No regression in training/evaluation times
- **Maintainability**: Cleaner directory structure and reduced cognitive load
- **Rollback Ready**: Easy restoration path for any moved files

## Rollback Procedure
```bash
# If issues arise, restore from cleanup directory:
cp -r /tmp_cleanup/phaseN/* ./
git checkout main
git branch -D cleanup-phaseN
```

## Current Status
- **Phase 1**: Ready for execution - zero risk
- **Phase 2**: Requires basic validation  
- **Phase 3**: Requires comprehensive testing
- **Phase 4**: Architectural planning stage

**Recommendation**: Start with Phase 1 only for maximum safety and immediate 70% cleanup benefit.