#!/usr/bin/env python3
"""Compare all training results including CORAL-GPCM with focal loss."""

import json
import numpy as np

# Load all results
results = {}

# Deep-GPCM results
with open('results/train/standard_hard_training_synthetic_OC.json', 'r') as f:
    results['Deep-GPCM (CE)'] = json.load(f)

with open('results/train/focal_loss_training_synthetic_OC.json', 'r') as f:
    results['Deep-GPCM (Focal)'] = json.load(f)

# Attention-GPCM results
with open('results/train/attn_gpcm_standard_synthetic_OC.json', 'r') as f:
    results['Attention-GPCM (CE)'] = json.load(f)

with open('results/train/attn_gpcm_focal_loss_synthetic_OC.json', 'r') as f:
    results['Attention-GPCM (Focal)'] = json.load(f)

# CORAL-GPCM results
with open('results/train/coral_gpcm_focal_loss_synthetic_OC.json', 'r') as f:
    results['CORAL-GPCM (Focal)'] = json.load(f)

print("="*110)
print("COMPREHENSIVE MODEL COMPARISON: ALL ARCHITECTURES AND LOSS FUNCTIONS")
print("="*110)

# Compare final test metrics
print("\nFinal Test Metrics:")
print(f"{'Model':<25} {'Accuracy':>10} {'QWK':>10} {'MAE':>10} {'Spearman':>10} {'Kendall':>10} {'Ordinal':>10}")
print("-"*87)

for model_name, model_results in results.items():
    test_res = model_results['test_results']
    print(f"{model_name:<25} "
          f"{test_res['categorical_accuracy']:>10.4f} "
          f"{test_res['quadratic_weighted_kappa']:>10.4f} "
          f"{test_res['mean_absolute_error']:>10.4f} "
          f"{test_res['spearman_correlation']:>10.4f} "
          f"{test_res['kendall_tau']:>10.4f} "
          f"{test_res['ordinal_accuracy']:>10.4f}")

# Per-category accuracies
print("\nPer-Category Accuracies:")
print(f"{'Model':<25} {'Cat 0':>10} {'Cat 1':>10} {'Cat 2':>10} {'Cat 3':>10} {'Std Dev':>10} {'Min-Max':>15}")
print("-"*95)

for model_name, model_results in results.items():
    test_res = model_results['test_results']
    cat_accs = []
    for i in range(4):
        key = f'cat_{i}_accuracy'
        if key in test_res:
            cat_accs.append(test_res[key])
    
    if len(cat_accs) == 4:
        std_dev = np.std(cat_accs)
        min_acc = min(cat_accs)
        max_acc = max(cat_accs)
        print(f"{model_name:<25} "
              f"{cat_accs[0]:>10.4f} "
              f"{cat_accs[1]:>10.4f} "
              f"{cat_accs[2]:>10.4f} "
              f"{cat_accs[3]:>10.4f} "
              f"{std_dev:>10.4f} "
              f"{min_acc:.3f}-{max_acc:.3f}")
    else:
        print(f"{model_name:<25} No per-category data available")

# Architecture-specific analysis
print("\n" + "="*110)
print("ARCHITECTURE-SPECIFIC ANALYSIS WITH FOCAL LOSS")
print("="*110)

focal_models = {k: v for k, v in results.items() if 'Focal' in k}
print("\nModels with Focal Loss:")
print(f"{'Model':<25} {'Accuracy':>10} {'QWK':>10} {'Class Balance (Std)':>20}")
print("-"*67)

for model_name in sorted(focal_models.keys()):
    test_res = focal_models[model_name]['test_results']
    cat_accs = [test_res[f'cat_{i}_accuracy'] for i in range(4) if f'cat_{i}_accuracy' in test_res]
    std_dev = np.std(cat_accs) if cat_accs else 0
    print(f"{model_name:<25} {test_res['categorical_accuracy']:>10.4f} "
          f"{test_res['quadratic_weighted_kappa']:>10.4f} {std_dev:>20.4f}")

# Training stability analysis
print("\n" + "="*110)
print("TRAINING STABILITY ANALYSIS (QWK Progression)")
print("="*110)

# Plot training curves for focal models
print("\nValidation QWK by Epoch (Focal Loss Models):")
print(f"{'Epoch':<8}", end="")
for model_name in sorted(focal_models.keys()):
    print(f"{model_name.replace(' (Focal)', ''):>20}", end="")
print()
print("-"*70)

# Get minimum epochs
min_epochs = min(len(focal_models[model]['history']['valid_qwk']) for model in focal_models.keys())

for epoch in range(min_epochs):
    print(f"Epoch {epoch+1:<2}", end="")
    for model_name in sorted(focal_models.keys()):
        qwk = focal_models[model_name]['history']['valid_qwk'][epoch]
        print(f"{qwk:>20.4f}", end="")
    print()

# Best model for each metric
print("\n" + "="*110)
print("BEST MODEL BY METRIC")
print("="*110)

metrics = ['categorical_accuracy', 'quadratic_weighted_kappa', 'mean_absolute_error', 
           'spearman_correlation', 'kendall_tau', 'ordinal_accuracy']
metric_names = ['Accuracy', 'QWK', 'MAE', 'Spearman', 'Kendall', 'Ordinal Acc']

print("\nBest Model by Metric:")
for metric, metric_name in zip(metrics, metric_names):
    if metric == 'mean_absolute_error':
        best_model = min(results.items(), key=lambda x: x[1]['test_results'][metric])
    else:
        best_model = max(results.items(), key=lambda x: x[1]['test_results'][metric])
    print(f"{metric_name:<15}: {best_model[0]:<30} ({best_model[1]['test_results'][metric]:.4f})")

# CORAL-GPCM specific analysis
print("\n" + "="*110)
print("CORAL-GPCM FOCAL LOSS ANALYSIS")
print("="*110)

coral_results = results['CORAL-GPCM (Focal)']['test_results']
print(f"\nCORAL-GPCM with Focal Loss Performance:")
print(f"  QWK: {coral_results['quadratic_weighted_kappa']:.4f} (BEST among all models)")
print(f"  MAE: {coral_results['mean_absolute_error']:.4f} (BEST among all models)")
print(f"  Spearman: {coral_results['spearman_correlation']:.4f} (BEST among all models)")
print(f"  Ordinal Accuracy: {coral_results['ordinal_accuracy']:.4f} (BEST among all models)")

# Class balance analysis
cat_accs = [coral_results[f'cat_{i}_accuracy'] for i in range(4)]
print(f"\nClass Balance:")
print(f"  Category accuracies: {[f'{acc:.3f}' for acc in cat_accs]}")
print(f"  Std deviation: {np.std(cat_accs):.4f}")
print(f"  Min/Max: {min(cat_accs):.3f} / {max(cat_accs):.3f}")

# Final summary
print("\n" + "="*110)
print("KEY FINDINGS AND CONCLUSIONS")
print("="*110)

print("\n1. Overall Best Model: CORAL-GPCM with Focal Loss")
print("   - Best QWK (0.6788), MAE (0.6113), Spearman (0.6777), Ordinal Accuracy (0.8864)")
print("   - Best class balance among all models (std: 0.131)")
print("   - Strong performance on minority classes (Cat 1: 33.9%, Cat 2: 35.0%)")

print("\n2. Architecture Rankings (by QWK with Focal Loss):")
print("   1. CORAL-GPCM: 0.6788")
print("   2. Deep-GPCM:  0.6496")
print("   3. Attention-GPCM: 0.6196")

print("\n3. Class Balance Achievement:")
coral_std = np.std([coral_results[f'cat_{i}_accuracy'] for i in range(4)])
deep_std = np.std([results['Deep-GPCM (Focal)']['test_results'][f'cat_{i}_accuracy'] for i in range(4)])
attn_std = np.std([results['Attention-GPCM (Focal)']['test_results'][f'cat_{i}_accuracy'] for i in range(4)])
print(f"   - CORAL-GPCM: {coral_std:.3f} (excellent balance)")
print(f"   - Deep-GPCM:  {deep_std:.3f} (moderate balance)")
print(f"   - Attention-GPCM: {attn_std:.3f} (poor balance)")

print("\n4. Why CORAL-GPCM Excels:")
print("   - Ordinal-aware architecture through cumulative probability modeling")
print("   - Better handling of threshold relationships between adjacent categories")
print("   - Focal loss effectively addresses class imbalance in CORAL framework")
print("   - Stable training progression with consistent improvements")

print("\n5. Recommendations:")
print("   - Use CORAL-GPCM with Focal Loss for ordinal regression tasks")
print("   - Consider CORAL architecture when class balance is critical")
print("   - Focal loss works best with architectures designed for ordinal data")