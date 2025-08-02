#!/usr/bin/env python3
"""
Demonstrate that threshold prediction with 0.5 threshold is computing the median.
"""

import torch
import numpy as np
import sys
sys.path.append('/home/steph/dirt-new/deep-gpcm')

from utils.predictions import compute_threshold_predictions, categorical_to_cumulative

def test_threshold_as_median():
    """Show that threshold prediction with 0.5 is the median."""
    
    print("="*60)
    print("THRESHOLD PREDICTION = MEDIAN DEMONSTRATION")
    print("="*60)
    
    # Test cases with different probability distributions
    test_cases = [
        ("Uniform", [0.25, 0.25, 0.25, 0.25]),
        ("Skewed left", [0.6, 0.3, 0.08, 0.02]),
        ("Skewed right", [0.02, 0.08, 0.3, 0.6]),
        ("Bimodal", [0.4, 0.1, 0.1, 0.4]),
        ("Center peaked", [0.1, 0.4, 0.4, 0.1]),
        ("Extreme case 1", [0.9, 0.05, 0.03, 0.02]),
        ("Extreme case 2", [0.02, 0.03, 0.05, 0.9]),
    ]
    
    for name, probs in test_cases:
        probs_tensor = torch.tensor([probs], dtype=torch.float32)
        
        # Compute cumulative probabilities P(Y > k)
        cum_probs_gt = categorical_to_cumulative(probs_tensor)
        
        # Also compute P(Y <= k) for clarity
        cum_probs_le = []
        for k in range(4):
            p_le_k = sum(probs[:k+1])
            cum_probs_le.append(p_le_k)
        
        # Compute threshold predictions with different thresholds
        threshold_05 = compute_threshold_predictions(probs_tensor, thresholds=[0.5, 0.5, 0.5])
        
        print(f"\n{name}: {probs}")
        print(f"P(Y > k): {cum_probs_gt[0].numpy()}")
        print(f"P(Y ≤ k): {cum_probs_le}")
        
        # Find median manually
        median_k = -1
        for k in range(4):
            if cum_probs_le[k] >= 0.5:
                median_k = k
                break
        
        print(f"Median (first k where P(Y≤k)≥0.5): {median_k}")
        print(f"Threshold prediction (0.5): {threshold_05[0].item()}")
        
        # Verify they match
        assert threshold_05[0].item() == median_k, f"Mismatch: {threshold_05[0].item()} != {median_k}"
    
    print("\n" + "="*60)
    print("THRESHOLD INTERPRETATION")
    print("="*60)
    
    print("\nThe threshold prediction algorithm:")
    print("1. Converts categorical probs to cumulative P(Y > k)")
    print("2. Finds highest k where P(Y > k) ≥ threshold[k]")
    print("3. With threshold = 0.5, this gives the median:")
    print("   - If P(Y > k) < 0.5, then P(Y ≤ k) ≥ 0.5")
    print("   - So we find the first k where P(Y ≤ k) ≥ 0.5")
    print("   - This is the definition of the median!")
    
    print("\nDifferent threshold interpretations:")
    print("- threshold = 0.5 → median")
    print("- threshold = 0.75 → 25th percentile (conservative)")
    print("- threshold = 0.25 → 75th percentile (optimistic)")
    
    # Show effect of different thresholds
    print("\n" + "="*60)
    print("EFFECT OF DIFFERENT THRESHOLDS")
    print("="*60)
    
    probs = [0.1, 0.2, 0.4, 0.3]
    probs_tensor = torch.tensor([probs], dtype=torch.float32)
    
    threshold_sets = [
        ([0.9, 0.9, 0.9], "10th percentile"),
        ([0.75, 0.75, 0.75], "25th percentile"),
        ([0.5, 0.5, 0.5], "50th percentile (median)"),
        ([0.25, 0.25, 0.25], "75th percentile"),
        ([0.1, 0.1, 0.1], "90th percentile"),
    ]
    
    print(f"\nProbabilities: {probs}")
    cum_probs_gt = categorical_to_cumulative(probs_tensor)
    print(f"P(Y > k): {cum_probs_gt[0].numpy()}")
    
    for thresholds, name in threshold_sets:
        pred = compute_threshold_predictions(probs_tensor, thresholds=thresholds)
        print(f"{name} {thresholds[0]}: predicts category {pred[0].item()}")

if __name__ == "__main__":
    test_threshold_as_median()