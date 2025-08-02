#!/usr/bin/env python3
"""
Example script demonstrating the unified prediction system for Deep-GPCM models.
Shows how different prediction methods affect evaluation metrics.
"""

import os
import torch
import argparse
from datetime import datetime

# Example usage commands
usage_examples = """
EXAMPLE USAGE:

1. Basic multi-method evaluation:
   python example_multimethod_eval.py --model_path save_models/best_deep_gpcm_synthetic_OC.pth --use_multimethod_eval

2. With custom thresholds:
   python example_multimethod_eval.py --model_path save_models/best_deep_gpcm_synthetic_OC.pth --use_multimethod_eval --thresholds 0.8 0.6 0.4

3. With adaptive thresholds:
   python example_multimethod_eval.py --model_path save_models/best_deep_gpcm_synthetic_OC.pth --use_multimethod_eval --adaptive_thresholds

4. Compare with standard evaluation:
   python example_multimethod_eval.py --model_path save_models/best_deep_gpcm_synthetic_OC.pth
   python example_multimethod_eval.py --model_path save_models/best_deep_gpcm_synthetic_OC.pth --use_multimethod_eval

5. Batch evaluation with multi-method:
   python evaluate.py --all --use_multimethod_eval
"""

def demonstrate_predictions():
    """Demonstrate different prediction methods on sample data."""
    print("="*60)
    print("UNIFIED PREDICTION SYSTEM DEMONSTRATION")
    print("="*60)
    
    # Create sample probability distributions
    n_samples = 5
    n_cats = 4
    
    # Different types of probability distributions
    probs = torch.tensor([
        [0.7, 0.2, 0.05, 0.05],    # High confidence in category 0
        [0.1, 0.6, 0.2, 0.1],      # High confidence in category 1
        [0.25, 0.25, 0.25, 0.25],  # Uniform (uncertain)
        [0.05, 0.15, 0.6, 0.2],    # High confidence in category 2
        [0.4, 0.3, 0.2, 0.1]       # Decreasing probabilities
    ])
    
    true_labels = torch.tensor([0, 1, 2, 2, 0])
    
    print("\nSample probability distributions:")
    for i in range(n_samples):
        print(f"  Sample {i}: {probs[i].numpy()} (true: {true_labels[i]})")
    
    # Import prediction functions
    from utils.predictions import compute_unified_predictions, PredictionConfig
    
    # Compute predictions
    config = PredictionConfig()
    config.use_gpu = False  # Keep on CPU for demo
    predictions = compute_unified_predictions(probs, config=config)
    
    print("\nPrediction results:")
    print(f"  Hard (argmax):     {predictions['hard'].cpu().numpy()}")
    print(f"  Soft (expected):   {predictions['soft'].cpu().numpy()}")
    print(f"  Threshold:         {predictions['threshold'].cpu().numpy()}")
    
    # Show how different methods affect metrics
    from utils.metrics import compute_metrics_multimethod
    
    metrics = compute_metrics_multimethod(true_labels, predictions, probs, n_cats=n_cats)
    
    print("\nKey metrics by prediction method:")
    key_metrics = ['categorical_accuracy', 'ordinal_accuracy', 'mean_absolute_error', 
                   'quadratic_weighted_kappa', 'spearman_correlation']
    
    for metric in key_metrics:
        if metric in metrics:
            method = metrics.get(f'{metric}_method', 'N/A')
            value = metrics[metric]
            print(f"  {metric:<30} = {value:6.3f} (using {method})")
    
    # Show method comparisons
    if 'method_comparisons' in metrics:
        print("\nMethod agreement rates:")
        for comp, stats in metrics['method_comparisons'].items():
            print(f"  {comp}: {stats:.1%}")

def main():
    """Run the demonstration and show example commands."""
    # Show usage examples
    print(usage_examples)
    
    # Run demonstration
    demonstrate_predictions()
    
    print("\n" + "="*60)
    print("KEY INSIGHTS:")
    print("="*60)
    print("""
1. HARD PREDICTIONS (argmax):
   - Best for: Classification accuracy, confusion matrices
   - Use when: You need discrete category assignments
   - Limitation: Ignores probability distribution information

2. SOFT PREDICTIONS (expected value):
   - Best for: Regression-like metrics (MAE, QWK, correlations)
   - Use when: Ordinal structure matters, continuous assessment needed
   - Benefit: Captures uncertainty and ordinal relationships

3. THRESHOLD PREDICTIONS (cumulative):
   - Best for: Ordinal accuracy, adjacent accuracy
   - Use when: Ordinal thresholds are meaningful
   - Benefit: Respects ordinal structure with discrete outputs

RECOMMENDATION: Use multi-method evaluation to get comprehensive insights!
""")

if __name__ == "__main__":
    main()