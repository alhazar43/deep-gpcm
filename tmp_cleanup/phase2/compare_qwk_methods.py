#!/usr/bin/env python3
"""
Compare QWK scores using different prediction methods.
"""

import os
import sys
import torch
import numpy as np
from sklearn.metrics import cohen_kappa_score

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.predictions import compute_unified_predictions, PredictionConfig
from utils.metrics import compute_metrics_multimethod

def main():
    # Test with synthetic data to show the differences
    print("="*60)
    print("QWK COMPARISON WITH DIFFERENT PREDICTION METHODS")
    print("="*60)
    
    # Create synthetic probability distributions
    np.random.seed(42)
    n_samples = 1000
    n_cats = 4
    
    # Generate realistic probability distributions
    # Mix of confident and uncertain predictions
    probs_list = []
    true_labels = []
    
    for i in range(n_samples):
        true_label = np.random.randint(0, n_cats)
        true_labels.append(true_label)
        
        # Create probability distribution
        if np.random.rand() < 0.7:  # 70% confident predictions
            # Confident: high probability on true label
            probs = np.random.dirichlet(np.ones(n_cats) * 0.5)
            probs[true_label] += 0.5
            probs = probs / probs.sum()
        else:  # 30% uncertain predictions
            # Uncertain: more uniform distribution
            probs = np.random.dirichlet(np.ones(n_cats) * 2.0)
        
        probs_list.append(probs)
    
    # Convert to tensors
    probabilities = torch.tensor(np.array(probs_list), dtype=torch.float32)
    y_true = np.array(true_labels)
    
    print(f"Generated {n_samples} samples with {n_cats} categories")
    
    # Compute all prediction types
    config = PredictionConfig(use_gpu=False)
    predictions = compute_unified_predictions(probabilities, config=config)
    
    # Extract predictions
    hard_pred = predictions['hard'].numpy()
    soft_pred = predictions['soft'].numpy()
    threshold_pred = predictions['threshold'].numpy()
    
    # Calculate QWK for each method
    print("\n" + "-"*60)
    print("PREDICTION STATISTICS")
    print("-"*60)
    
    # 1. Hard predictions
    qwk_hard = cohen_kappa_score(y_true, hard_pred, weights='quadratic')
    acc_hard = (y_true == hard_pred).mean()
    print(f"\n1. Hard Predictions (argmax):")
    print(f"   Distribution - Mean: {hard_pred.mean():.3f}, Std: {hard_pred.std():.3f}")
    print(f"   Accuracy: {acc_hard:.3f}")
    print(f"   QWK: {qwk_hard:.4f}")
    
    # 2. Soft predictions (need to round for QWK)
    soft_rounded = np.round(soft_pred).astype(int)
    soft_rounded = np.clip(soft_rounded, 0, n_cats - 1)
    qwk_soft = cohen_kappa_score(y_true, soft_rounded, weights='quadratic')
    acc_soft = (y_true == soft_rounded).mean()
    mae_soft = np.abs(y_true - soft_pred).mean()
    print(f"\n2. Soft Predictions (expected value):")
    print(f"   Raw - Mean: {soft_pred.mean():.3f}, Std: {soft_pred.std():.3f}")
    print(f"   Rounded - Mean: {soft_rounded.mean():.3f}, Std: {soft_rounded.std():.3f}")
    print(f"   Accuracy (rounded): {acc_soft:.3f}")
    print(f"   MAE (raw): {mae_soft:.3f}")
    print(f"   QWK (rounded): {qwk_soft:.4f}")
    
    # 3. Threshold predictions
    qwk_threshold = cohen_kappa_score(y_true, threshold_pred, weights='quadratic')
    acc_threshold = (y_true == threshold_pred).mean()
    print(f"\n3. Threshold Predictions (cumulative):")
    print(f"   Distribution - Mean: {threshold_pred.mean():.3f}, Std: {threshold_pred.std():.3f}")
    print(f"   Accuracy: {acc_threshold:.3f}")
    print(f"   QWK: {qwk_threshold:.4f}")
    print(f"   Default thresholds: {config.thresholds}")
    
    # Test with different thresholds
    print("\n" + "-"*60)
    print("THRESHOLD SENSITIVITY ANALYSIS")
    print("-"*60)
    
    threshold_sets = [
        ([0.75, 0.5, 0.25], "Default (balanced)"),
        ([0.8, 0.6, 0.4], "Conservative"),
        ([0.7, 0.5, 0.3], "Slightly conservative"),
        ([0.6, 0.4, 0.2], "Liberal"),
    ]
    
    for thresholds, name in threshold_sets:
        config_test = PredictionConfig(use_gpu=False)
        config_test.thresholds = thresholds
        preds_test = compute_unified_predictions(probabilities, config=config_test)
        threshold_pred_test = preds_test['threshold'].numpy()
        qwk_test = cohen_kappa_score(y_true, threshold_pred_test, weights='quadratic')
        acc_test = (y_true == threshold_pred_test).mean()
        print(f"\n{name} {thresholds}:")
        print(f"  Mean: {threshold_pred_test.mean():.3f}, QWK: {qwk_test:.4f}, Acc: {acc_test:.3f}")
    
    # Analyze prediction differences
    print("\n" + "-"*60)
    print("PREDICTION AGREEMENT ANALYSIS")
    print("-"*60)
    
    # Agreement rates
    hard_soft_agree = (hard_pred == soft_rounded).mean()
    hard_threshold_agree = (hard_pred == threshold_pred).mean()
    soft_threshold_agree = (soft_rounded == threshold_pred).mean()
    
    print(f"Hard vs Soft (rounded) agreement: {hard_soft_agree:.1%}")
    print(f"Hard vs Threshold agreement: {hard_threshold_agree:.1%}")
    print(f"Soft (rounded) vs Threshold agreement: {soft_threshold_agree:.1%}")
    
    # Where do they disagree?
    hard_soft_disagree = hard_pred != soft_rounded
    if hard_soft_disagree.any():
        disagree_idx = np.where(hard_soft_disagree)[0][:5]  # First 5 disagreements
        print(f"\nExample disagreements (Hard vs Soft):")
        for idx in disagree_idx:
            print(f"  Sample {idx}: True={y_true[idx]}, Hard={hard_pred[idx]}, "
                  f"Soft={soft_pred[idx]:.2f}→{soft_rounded[idx]}, "
                  f"Probs={probabilities[idx].numpy()}")

if __name__ == "__main__":
    main()