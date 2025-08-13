#!/usr/bin/env python3
"""Compare training results between standard and focal loss training."""

import json
import numpy as np

# Load results
with open('results/train/standard_hard_training_synthetic_OC.json', 'r') as f:
    standard_results = json.load(f)

with open('results/train/focal_loss_training_synthetic_OC.json', 'r') as f:
    focal_results = json.load(f)

print("="*80)
print("TRAINING COMPARISON: Standard vs Focal Loss")
print("="*80)

# Compare final test metrics
print("\nFinal Test Metrics:")
print(f"{'Metric':<30} {'Standard':>12} {'Focal Loss':>12} {'Difference':>12}")
print("-"*68)

metrics_to_compare = [
    'categorical_accuracy',
    'quadratic_weighted_kappa',
    'mean_absolute_error',
    'spearman_correlation',
    'kendall_tau',
    'ordinal_accuracy'
]

for metric in metrics_to_compare:
    standard_val = standard_results['test_results'][metric]
    focal_val = focal_results['test_results'][metric]
    diff = focal_val - standard_val
    
    # MAE should be minimized
    if metric == 'mean_absolute_error':
        diff = -diff
        improvement = "↓" if diff > 0 else "↑"
    else:
        improvement = "↑" if diff > 0 else "↓"
    
    print(f"{metric:<30} {standard_val:>12.4f} {focal_val:>12.4f} {diff:>11.4f}{improvement}")

# Compare per-category accuracies (only focal loss has them)
print("\nPer-Category Accuracies (Focal Loss Only):")
print(f"{'Category':<15} {'Focal Loss':>12}")
print("-"*28)

focal_cat_accs = []
for i in range(4):
    key = f'cat_{i}_accuracy'
    if key in focal_results['test_results']:
        focal_val = focal_results['test_results'][key]
        focal_cat_accs.append(focal_val)
        print(f"Category {i:<6} {focal_val:>12.4f}")

# Analyze class balance
if focal_cat_accs:
    print("\nClass Balance Analysis (Focal Loss):")
    focal_std = np.std(focal_cat_accs)
    focal_min = min(focal_cat_accs)
    focal_max = max(focal_cat_accs)
    print(f"Per-category accuracy std dev: {focal_std:.4f}")
    print(f"Min category accuracy: {focal_min:.4f}")
    print(f"Max category accuracy: {focal_max:.4f}")
    print(f"Range: {focal_max - focal_min:.4f}")

# Training progression
print("\nTraining Progression (QWK):")
print(f"{'Epoch':<10} {'Standard':>12} {'Focal Loss':>12}")
print("-"*35)

for i in range(min(len(standard_results['history']['valid_qwk']), 
                  len(focal_results['history']['valid_qwk']))):
    standard_qwk = standard_results['history']['valid_qwk'][i]
    focal_qwk = focal_results['history']['valid_qwk'][i]
    print(f"Epoch {i+1:<4} {standard_qwk:>12.4f} {focal_qwk:>12.4f}")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)

# Calculate overall improvement
qwk_improvement = (focal_results['test_results']['quadratic_weighted_kappa'] - 
                   standard_results['test_results']['quadratic_weighted_kappa'])
acc_improvement = (focal_results['test_results']['categorical_accuracy'] - 
                   standard_results['test_results']['categorical_accuracy'])

print(f"\nQWK Improvement: {qwk_improvement:.4f} ({qwk_improvement/standard_results['test_results']['quadratic_weighted_kappa']*100:.1f}%)")
print(f"Accuracy Improvement: {acc_improvement:.4f} ({acc_improvement/standard_results['test_results']['categorical_accuracy']*100:.1f}%)")

# Key insights
print("\nKey Insights:")

if qwk_improvement > 0:
    print(f"✓ Focal loss improved QWK by {qwk_improvement:.4f} ({qwk_improvement/standard_results['test_results']['quadratic_weighted_kappa']*100:.1f}%)")
else:
    print(f"✗ Focal loss decreased QWK by {abs(qwk_improvement):.4f}")

if acc_improvement > 0:
    print(f"✓ Focal loss improved accuracy by {acc_improvement:.4f}")
else:
    print(f"✗ Focal loss decreased accuracy by {abs(acc_improvement):.4f} ({acc_improvement/standard_results['test_results']['categorical_accuracy']*100:.1f}%)")

# Analyze focal loss class balance
if focal_cat_accs:
    # Check if minority classes (1, 2) have reasonable accuracy
    minority_accs = [focal_cat_accs[1], focal_cat_accs[2]]
    if min(minority_accs) > 0.15:  # 15% threshold for minority classes
        print(f"✓ Focal loss achieved reasonable minority class performance (Cat 1: {focal_cat_accs[1]:.3f}, Cat 2: {focal_cat_accs[2]:.3f})")
    else:
        print(f"✗ Focal loss still struggles with minority classes (Cat 1: {focal_cat_accs[1]:.3f}, Cat 2: {focal_cat_accs[2]:.3f})")
    
    # Check balance
    if focal_std < 0.3:  # Reasonable balance threshold
        print(f"✓ Focal loss achieved relatively balanced per-category performance (std: {focal_std:.3f})")
    else:
        print(f"✗ Focal loss still has imbalanced per-category performance (std: {focal_std:.3f})")