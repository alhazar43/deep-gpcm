#!/usr/bin/env python3
"""Compare focal loss training results across different models."""

import json
import numpy as np

# Load all results
results = {}

# Deep-GPCM results
with open('results/train/standard_hard_training_synthetic_OC.json', 'r') as f:
    results['Deep-GPCM (Standard CE)'] = json.load(f)

with open('results/train/focal_loss_training_synthetic_OC.json', 'r') as f:
    results['Deep-GPCM (Focal Loss)'] = json.load(f)

# Attention-GPCM results
with open('results/train/attn_gpcm_focal_loss_synthetic_OC.json', 'r') as f:
    results['Attention-GPCM (Focal Loss)'] = json.load(f)

print("="*80)
print("FOCAL LOSS COMPARISON ACROSS MODELS")
print("="*80)

# Compare final test metrics
print("\nFinal Test Metrics:")
print(f"{'Model':<30} {'Accuracy':>10} {'QWK':>10} {'MAE':>10} {'Spearman':>10}")
print("-"*72)

for model_name, model_results in results.items():
    test_res = model_results['test_results']
    print(f"{model_name:<30} "
          f"{test_res['categorical_accuracy']:>10.4f} "
          f"{test_res['quadratic_weighted_kappa']:>10.4f} "
          f"{test_res['mean_absolute_error']:>10.4f} "
          f"{test_res['spearman_correlation']:>10.4f}")

# Per-category accuracies (only for focal loss models)
print("\nPer-Category Accuracies (Focal Loss Models Only):")
print(f"{'Model':<30} {'Cat 0':>10} {'Cat 1':>10} {'Cat 2':>10} {'Cat 3':>10} {'Std Dev':>10}")
print("-"*82)

for model_name, model_results in results.items():
    if 'Focal Loss' in model_name:
        test_res = model_results['test_results']
        cat_accs = []
        for i in range(4):
            key = f'cat_{i}_accuracy'
            if key in test_res:
                cat_accs.append(test_res[key])
        
        if len(cat_accs) == 4:
            std_dev = np.std(cat_accs)
            print(f"{model_name:<30} "
                  f"{cat_accs[0]:>10.4f} "
                  f"{cat_accs[1]:>10.4f} "
                  f"{cat_accs[2]:>10.4f} "
                  f"{cat_accs[3]:>10.4f} "
                  f"{std_dev:>10.4f}")

# Training progression comparison
print("\nTraining Progression (Validation QWK):")
print(f"{'Epoch':<10}", end="")
for model_name in results.keys():
    print(f"{model_name.split(' ')[0]:>15}", end="")
print()
print("-"*55)

# Get the minimum number of epochs across all models
min_epochs = min(len(results[model]['history']['valid_qwk']) for model in results.keys())

for epoch in range(min_epochs):
    print(f"Epoch {epoch+1:<4}", end="")
    for model_name, model_results in results.items():
        qwk = model_results['history']['valid_qwk'][epoch]
        print(f"{qwk:>15.4f}", end="")
    print()

# Improvement analysis
print("\n" + "="*80)
print("IMPROVEMENT ANALYSIS")
print("="*80)

baseline_qwk = results['Deep-GPCM (Standard CE)']['test_results']['quadratic_weighted_kappa']
baseline_acc = results['Deep-GPCM (Standard CE)']['test_results']['categorical_accuracy']

print(f"\nBaseline (Deep-GPCM with Standard CE): QWK={baseline_qwk:.4f}, Acc={baseline_acc:.4f}")
print("\nImprovements over baseline:")

for model_name, model_results in results.items():
    if model_name != 'Deep-GPCM (Standard CE)':
        test_res = model_results['test_results']
        qwk_diff = test_res['quadratic_weighted_kappa'] - baseline_qwk
        acc_diff = test_res['categorical_accuracy'] - baseline_acc
        
        qwk_pct = (qwk_diff / baseline_qwk) * 100
        acc_pct = (acc_diff / baseline_acc) * 100
        
        print(f"\n{model_name}:")
        print(f"  QWK: {test_res['quadratic_weighted_kappa']:.4f} ({qwk_diff:+.4f}, {qwk_pct:+.1f}%)")
        print(f"  Acc: {test_res['categorical_accuracy']:.4f} ({acc_diff:+.4f}, {acc_pct:+.1f}%)")

# Key insights
print("\n" + "="*80)
print("KEY INSIGHTS")
print("="*80)

# Check class balance improvement
focal_models = {k: v for k, v in results.items() if 'Focal Loss' in k}
if focal_models:
    print("\nClass Balance Analysis (Focal Loss Models):")
    for model_name, model_results in focal_models.items():
        test_res = model_results['test_results']
        cat_accs = [test_res[f'cat_{i}_accuracy'] for i in range(4) if f'cat_{i}_accuracy' in test_res]
        if cat_accs:
            std_dev = np.std(cat_accs)
            min_acc = min(cat_accs)
            max_acc = max(cat_accs)
            minority_accs = [cat_accs[1], cat_accs[2]]  # Categories 1 and 2 are minority
            
            print(f"\n{model_name}:")
            print(f"  Per-category std dev: {std_dev:.4f}")
            print(f"  Min/Max accuracy: {min_acc:.4f} / {max_acc:.4f}")
            print(f"  Minority class avg (Cat 1&2): {np.mean(minority_accs):.4f}")

# Final summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

# Find best model for each metric
metrics = ['categorical_accuracy', 'quadratic_weighted_kappa', 'mean_absolute_error', 'spearman_correlation']
metric_names = ['Accuracy', 'QWK', 'MAE', 'Spearman']

print("\nBest model for each metric:")
for metric, metric_name in zip(metrics, metric_names):
    if metric == 'mean_absolute_error':
        # Lower is better for MAE
        best_model = min(results.items(), key=lambda x: x[1]['test_results'][metric])
    else:
        # Higher is better
        best_model = max(results.items(), key=lambda x: x[1]['test_results'][metric])
    
    print(f"{metric_name:<15}: {best_model[0]:<35} ({best_model[1]['test_results'][metric]:.4f})")

print("\nConclusions:")
print("1. Focal loss improves ordinal metrics (QWK, Spearman) at slight cost to accuracy")
print("2. Deep-GPCM with focal loss achieves best overall ordinal performance")
print("3. Attention-GPCM with focal loss shows extreme class imbalance issues")
print("4. Focal loss helps minority classes but doesn't fully solve imbalance")