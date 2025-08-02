# Plotting System Updates

## Summary of Changes

### 1. Decimal Formatting (2 decimals)
- All metric values now display with 2 decimal places instead of 3-4
- Format: `{value:.2f}` for single values
- Format: `{mean:.2f}±{std:.2f}` when std is available and > 0
- Handles NaN/invalid std values by setting them to 0

### 2. Consistent Model Coloring
- Uses centralized `get_model_color()` method for all plots
- Predefined colors for known models:
  - `deep_gpcm`: Orange (#ff7f0e)
  - `attn_gpcm`: Blue (#1f77b4)
  - `coral_gpcm_proper`: Pink (#e377c2)
- Dynamic assignment for unknown models using tab10 colormap
- Ensures same model always gets same color across all plots

### 3. Shortened Filenames
Old → New filename mappings:
- `training_metrics.png` → `train.png`
- `test_metrics.png` → `test.png`
- `training_vs_test_comparison.png` → `train_v_test.png`
- `categorical_breakdown.png` → `cat_breakdown.png`
- `confusion_matrices_test.png` → `confusion_test.png`
- `ordinal_distance_distribution_test.png` → `ord_dist_test.png`
- `category_transitions_test.png` → `cat_trans_test.png`
- `roc_curves_per_category_test.png` → `roc_test.png`
- `calibration_curves_test.png` → `calib_test.png`
- `cv_score_comparison.png` → `cv_scores.png`
- `attention_weights_test.png` → `attention_test.png`
- `time_series_performance_test.png` → `time_series_test.png`

### 4. Additional Features
- New directory structure support: `results/[type]/[dataset]/*.json`
- CV score comparison plot when validation summaries available
- Training vs test comparison now uses CV summaries from validation directory
- Proper handling of mean±std with NaN/0 checks

## Usage

```bash
# Generate all plots for a dataset
python utils/plot_metrics.py --dataset synthetic_OC

# Custom results directory
python utils/plot_metrics.py --dataset synthetic_OC --results_dir custom_results
```

## Generated Plots (10 total)
1. `train.png` - Training metrics over epochs with mean±std bands
2. `test.png` - Test evaluation metrics comparison
3. `train_v_test.png` - Side-by-side training vs test comparison
4. `cat_breakdown.png` - Per-category accuracy breakdown
5. `confusion_test.png` - Confusion matrices for each model
6. `ord_dist_test.png` - Ordinal distance distribution
7. `cat_trans_test.png` - Category transition matrices
8. `roc_test.png` - ROC curves per category
9. `calib_test.png` - Calibration curves
10. `cv_scores.png` - Cross-validation score comparison (when available)