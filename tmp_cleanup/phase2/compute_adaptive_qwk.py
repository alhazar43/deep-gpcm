#!/usr/bin/env python3
"""
Compute QWK and other metrics for Adaptive IRT results
"""

import json
import numpy as np
from utils.metrics import compute_metrics
from sklearn.metrics import cohen_kappa_score
import argparse

def compute_qwk_weights(n_cats):
    """Compute QWK weight matrix"""
    weights = np.zeros((n_cats, n_cats))
    for i in range(n_cats):
        for j in range(n_cats):
            weights[i, j] = (i - j) ** 2 / (n_cats - 1) ** 2
    return weights

def quadratic_weighted_kappa(y_true, y_pred, n_cats=4):
    """Compute quadratic weighted kappa"""
    weights = compute_qwk_weights(n_cats)
    return cohen_kappa_score(y_true, y_pred, weights=weights.flatten())

def compute_adaptive_irt_metrics(results_file):
    """Compute comprehensive metrics for adaptive IRT"""
    
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    predictions = np.array(results['predictions'])
    actual = np.array(results['actual'])
    probabilities = np.array(results['probabilities'])
    
    print(f"Adaptive IRT Results Analysis")
    print("=" * 40)
    print(f"Total predictions: {len(predictions):,}")
    print(f"Sequences evaluated: {results['config']['n_sequences_evaluated']}")
    print()
    
    # Compute metrics using the same system as other models
    n_cats = 4
    metrics = compute_metrics(actual, predictions, probabilities, n_cats=n_cats)
    
    # Display key metrics
    print("Performance Metrics:")
    print("-" * 20)
    print(f"Categorical Accuracy: {metrics['categorical_accuracy']:.3f}")
    print(f"Ordinal Accuracy: {metrics['ordinal_accuracy']:.3f}")
    print(f"Quadratic Weighted Kappa: {metrics['quadratic_weighted_kappa']:.3f}")
    print(f"Mean Absolute Error: {metrics['mean_absolute_error']:.3f}")
    print(f"Kendall Tau: {metrics['kendall_tau']:.3f}")
    print(f"Spearman Correlation: {metrics['spearman_correlation']:.3f}")
    print(f"Cohen Kappa: {metrics['cohen_kappa']:.3f}")
    print(f"Cross Entropy: {metrics['cross_entropy']:.3f}")
    
    # Update results with computed metrics (convert numpy types to native Python)
    for key, value in metrics.items():
        if isinstance(value, (np.integer, np.floating)):
            results['evaluation_results'][key] = float(value)
        else:
            results['evaluation_results'][key] = value
    
    # Save updated results
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nUpdated results saved to: {results_file}")
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description='Compute QWK for Adaptive IRT')
    parser.add_argument('--results_file', 
                       default='./data/synthetic_OC/results/test/adaptive_irt_results.json',
                       help='Path to adaptive IRT results file')
    
    args = parser.parse_args()
    
    metrics = compute_adaptive_irt_metrics(args.results_file)
    
    print(f"\n🎯 Key Result: QWK = {metrics['quadratic_weighted_kappa']:.3f}")

if __name__ == "__main__":
    main()