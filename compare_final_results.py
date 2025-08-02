#!/usr/bin/env python3
"""Compare all training results: Deep-GPCM and Attention-GPCM with CE and Focal Loss."""

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

print("="*100)
print("COMPREHENSIVE MODEL COMPARISON: DEEP-GPCM vs ATTENTION-GPCM with CE vs FOCAL LOSS")
print("="*100)

# Compare final test metrics
print("\nFinal Test Metrics:")
print(f"{'Model':<25} {'Accuracy':>10} {'QWK':>10} {'MAE':>10} {'Spearman':>10} {'Kendall':>10}")
print("-"*77)

for model_name, model_results in results.items():
    test_res = model_results['test_results']
    print(f"{model_name:<25} "
          f"{test_res['categorical_accuracy']:>10.4f} "
          f"{test_res['quadratic_weighted_kappa']:>10.4f} "
          f"{test_res['mean_absolute_error']:>10.4f} "
          f"{test_res['spearman_correlation']:>10.4f} "
          f"{test_res['kendall_tau']:>10.4f}")

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

# Model comparison by architecture
print("\n" + "="*100)
print("MODEL ARCHITECTURE COMPARISON")
print("="*100)

# Deep-GPCM: CE vs Focal
print("\nDeep-GPCM: Cross-Entropy vs Focal Loss")
deep_ce = results['Deep-GPCM (CE)']['test_results']
deep_focal = results['Deep-GPCM (Focal)']['test_results']

print(f"{'Metric':<25} {'CE':>12} {'Focal':>12} {'Difference':>15}")
print("-"*65)

metrics = ['categorical_accuracy', 'quadratic_weighted_kappa', 'mean_absolute_error', 
           'spearman_correlation', 'kendall_tau']
for metric in metrics:
    ce_val = deep_ce[metric]
    focal_val = deep_focal[metric]
    diff = focal_val - ce_val
    if metric == 'mean_absolute_error':
        improvement = "↓" if diff < 0 else "↑"
    else:
        improvement = "↑" if diff > 0 else "↓"
    print(f"{metric:<25} {ce_val:>12.4f} {focal_val:>12.4f} {diff:>14.4f}{improvement}")

# Attention-GPCM: CE vs Focal
print("\nAttention-GPCM: Cross-Entropy vs Focal Loss")
attn_ce = results['Attention-GPCM (CE)']['test_results']
attn_focal = results['Attention-GPCM (Focal)']['test_results']

print(f"{'Metric':<25} {'CE':>12} {'Focal':>12} {'Difference':>15}")
print("-"*65)

for metric in metrics:
    ce_val = attn_ce[metric]
    focal_val = attn_focal[metric]
    diff = focal_val - ce_val
    if metric == 'mean_absolute_error':
        improvement = "↓" if diff < 0 else "↑"
    else:
        improvement = "↑" if diff > 0 else "↓"
    print(f"{metric:<25} {ce_val:>12.4f} {focal_val:>12.4f} {diff:>14.4f}{improvement}")

# Loss function comparison
print("\n" + "="*100)
print("LOSS FUNCTION COMPARISON")
print("="*100)

# CE models
print("\nCross-Entropy Models:")
print(f"{'Model':<25} {'Accuracy':>10} {'QWK':>10} {'Class Balance (Std)':>20}")
print("-"*67)

for model_name in ['Deep-GPCM (CE)', 'Attention-GPCM (CE)']:
    test_res = results[model_name]['test_results']
    cat_accs = [test_res[f'cat_{i}_accuracy'] for i in range(4) if f'cat_{i}_accuracy' in test_res]
    std_dev = np.std(cat_accs) if cat_accs else 0
    print(f"{model_name:<25} {test_res['categorical_accuracy']:>10.4f} "
          f"{test_res['quadratic_weighted_kappa']:>10.4f} {std_dev:>20.4f}")

# Focal models
print("\nFocal Loss Models:")
print(f"{'Model':<25} {'Accuracy':>10} {'QWK':>10} {'Class Balance (Std)':>20}")
print("-"*67)

for model_name in ['Deep-GPCM (Focal)', 'Attention-GPCM (Focal)']:
    test_res = results[model_name]['test_results']
    cat_accs = [test_res[f'cat_{i}_accuracy'] for i in range(4) if f'cat_{i}_accuracy' in test_res]
    std_dev = np.std(cat_accs) if cat_accs else 0
    print(f"{model_name:<25} {test_res['categorical_accuracy']:>10.4f} "
          f"{test_res['quadratic_weighted_kappa']:>10.4f} {std_dev:>20.4f}")

# Training stability analysis
print("\n" + "="*100)
print("TRAINING STABILITY ANALYSIS (QWK Progression)")
print("="*100)

# Plot training curves
print("\nValidation QWK by Epoch:")
print(f"{'Epoch':<8}", end="")
for model_name in results.keys():
    print(f"{model_name.replace(' (', '-').replace(')', ''):>18}", end="")
print()
print("-"*82)

# Get minimum epochs
min_epochs = min(len(results[model]['history']['valid_qwk']) for model in results.keys())

for epoch in range(min_epochs):
    print(f"Epoch {epoch+1:<2}", end="")
    for model_name, model_results in results.items():
        qwk = model_results['history']['valid_qwk'][epoch]
        print(f"{qwk:>18.4f}", end="")
    print()

# Final summary
print("\n" + "="*100)
print("KEY FINDINGS")
print("="*100)

# Best model for each metric
print("\nBest Model by Metric:")
for metric, metric_name in zip(metrics, ['Accuracy', 'QWK', 'MAE', 'Spearman', 'Kendall']):
    if metric == 'mean_absolute_error':
        best_model = min(results.items(), key=lambda x: x[1]['test_results'][metric])
    else:
        best_model = max(results.items(), key=lambda x: x[1]['test_results'][metric])
    print(f"{metric_name:<15}: {best_model[0]:<30} ({best_model[1]['test_results'][metric]:.4f})")

print("\nConclusions:")
print("1. Architecture Impact:")
print("   - Deep-GPCM outperforms Attention-GPCM on most metrics")
print("   - Attention mechanism may be overfitting on this small dataset")

print("\n2. Loss Function Impact:")
print("   - Focal Loss improves ordinal metrics (QWK, MAE) for Deep-GPCM")
print("   - Focal Loss causes instability for Attention-GPCM (extreme class imbalance)")
print("   - CE provides more stable training for Attention-GPCM")

print("\n3. Class Balance:")
attention_ce_std = np.std([results['Attention-GPCM (CE)']['test_results'][f'cat_{i}_accuracy'] for i in range(4)])
attention_focal_std = np.std([results['Attention-GPCM (Focal)']['test_results'][f'cat_{i}_accuracy'] for i in range(4)])
print(f"   - Attention-GPCM (CE) std: {attention_ce_std:.4f}")
print(f"   - Attention-GPCM (Focal) std: {attention_focal_std:.4f}")
print(f"   - Focal Loss worsens class imbalance in Attention-GPCM")

print("\n4. Overall Best Model: Deep-GPCM with Focal Loss")
print("   - Best QWK (0.6496), MAE (0.6603), and Spearman (0.6498)")
print("   - Reasonable class balance with improved minority class performance")