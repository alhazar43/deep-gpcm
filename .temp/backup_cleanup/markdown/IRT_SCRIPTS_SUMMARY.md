# IRT Scripts Summary

## Active Scripts

### 1. `irt_analysis.py` (PRIMARY - Unified Tool)
**Purpose**: Comprehensive IRT analysis with all functionality combined
**Features**:
- Automatic model detection
- Temporal parameter extraction
- Flexible aggregation methods (last/average for θ, average/last for α/β)
- Parameter recovery analysis
- Multiple visualization types
- Summary report generation

**Usage**:
```bash
# Basic analysis
python irt_analysis.py --dataset synthetic_OC

# Temporal analysis
python irt_analysis.py --dataset synthetic_OC --theta_method average --analysis_types temporal

# Full analysis with all plots
python irt_analysis.py --dataset synthetic_OC --analysis_types recovery temporal irt_plots
```

### 2. `plot_irt.py` (Legacy - Still Supported)
**Purpose**: Traditional IRT visualizations for single models
**Features**:
- Item Characteristic Curves (ICC)
- Item Information Functions (IIF)
- Test Information Function (TIF)
- Wright Maps
- Parameter distributions
- Model comparisons

**Usage**:
```bash
python plot_irt.py --model_path save_models/best_baseline_synthetic_OC.pth --plot_type all
python plot_irt.py --compare model1.pth model2.pth
```

### 3. `animate_irt.py` (Legacy - Still Supported)
**Purpose**: Animated temporal IRT parameter visualizations
**Features**:
- Student learning journey animations
- Parameter distribution evolution
- Ability trajectory heatmaps
- Temporal summary statistics

**Usage**:
```bash
python animate_irt.py --model_path save_models/best_baseline_synthetic_OC.pth --animation_type all
```

## Archived Scripts (Moved to .temp/old_irt_scripts/)

1. `analyze_irt.py` - Replaced by `irt_analysis.py`
2. `analyze_irt_temporal.py` - Functionality merged into `irt_analysis.py`
3. `extract_irt_parameters.py` - Functionality merged into `irt_analysis.py`
4. `compare_irt_params.py` - Functionality merged into `irt_analysis.py`
5. `summarize_irt_correlations.py` - Functionality merged into `irt_analysis.py`
6. `test_irt_params.py` - Test script, not actively used

## Migration Guide

### Old Command → New Command

```bash
# Extract parameters
OLD: python extract_irt_parameters.py --model_path model.pth --save_path params.json
NEW: python irt_analysis.py --dataset synthetic_OC --save_params --analysis_types none

# Compare with true parameters
OLD: python compare_irt_params.py --extracted_params params.json --dataset synthetic_OC
NEW: python irt_analysis.py --dataset synthetic_OC --analysis_types recovery

# Temporal analysis
OLD: python analyze_irt_temporal.py --dataset synthetic_OC --theta_method last
NEW: python irt_analysis.py --dataset synthetic_OC --theta_method last --analysis_types temporal

# Auto-detect and analyze all models
OLD: python analyze_irt.py --dataset synthetic_OC
NEW: python irt_analysis.py --dataset synthetic_OC
```

## Key Improvements in Unified Tool

1. **Single Entry Point**: One script for all IRT analysis needs
2. **Flexible Analysis**: Choose specific analysis types with `--analysis_types`
3. **Better Organization**: Consistent output structure in results/irt/
4. **Temporal Awareness**: Proper handling of temporal parameters with aggregation options
5. **Comprehensive Reports**: Detailed summary with all metrics and insights