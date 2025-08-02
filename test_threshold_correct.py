#!/usr/bin/env python3
"""
Correctly understand threshold prediction algorithm.
"""

import torch
import numpy as np
import sys
sys.path.append('/home/steph/dirt-new/deep-gpcm')

from utils.predictions import compute_threshold_predictions, categorical_to_cumulative

def trace_threshold_algorithm(probs, thresholds):
    """Trace through the threshold prediction algorithm step by step."""
    probs_tensor = torch.tensor([probs], dtype=torch.float32)
    cum_probs_gt = categorical_to_cumulative(probs_tensor)[0].numpy()
    
    print(f"\nProbabilities: {probs}")
    print(f"P(Y > k): {cum_probs_gt}")
    print(f"Thresholds: {thresholds}")
    
    # Simulate the algorithm
    prediction = 0
    print(f"\nStarting with prediction = {prediction}")
    
    # Work backwards from highest to lowest category
    n_cats = len(probs)
    for k in range(n_cats - 2, -1, -1):
        if cum_probs_gt[k] >= thresholds[k]:
            old_pred = prediction
            prediction = k + 1
            print(f"k={k}: P(Y > {k}) = {cum_probs_gt[k]:.3f} >= {thresholds[k]} → prediction = {k+1} (was {old_pred})")
        else:
            print(f"k={k}: P(Y > {k}) = {cum_probs_gt[k]:.3f} < {thresholds[k]} → keep prediction = {prediction}")
    
    # Verify with actual function
    actual = compute_threshold_predictions(probs_tensor, thresholds=thresholds)[0].item()
    print(f"\nFinal prediction: {prediction}")
    print(f"Actual from function: {actual}")
    assert prediction == actual
    
    return prediction

def main():
    print("="*60)
    print("UNDERSTANDING THRESHOLD PREDICTION ALGORITHM")
    print("="*60)
    
    # Test case 1: Skewed right distribution
    print("\nTest 1: Skewed right [0.02, 0.08, 0.3, 0.6]")
    trace_threshold_algorithm([0.02, 0.08, 0.3, 0.6], [0.5, 0.5, 0.5])
    
    # Test case 2: Uniform distribution
    print("\n" + "-"*60)
    print("\nTest 2: Uniform [0.25, 0.25, 0.25, 0.25]")
    trace_threshold_algorithm([0.25, 0.25, 0.25, 0.25], [0.5, 0.5, 0.5])
    
    # Test case 3: Different thresholds
    print("\n" + "-"*60)
    print("\nTest 3: Same distribution, different thresholds")
    probs = [0.1, 0.2, 0.4, 0.3]
    trace_threshold_algorithm(probs, [0.75, 0.5, 0.25])
    trace_threshold_algorithm(probs, [0.9, 0.7, 0.3])
    
    print("\n" + "="*60)
    print("INTERPRETATION")
    print("="*60)
    
    print("\nThe algorithm finds the SMALLEST category k such that:")
    print("P(Y > k-1) < threshold[k-1]")
    print("\nOr equivalently, the SMALLEST k such that:")
    print("P(Y ≤ k) > 1 - threshold[k-1]")
    
    print("\nWith threshold = 0.5:")
    print("- Finds smallest k where P(Y ≤ k) > 0.5")
    print("- This is indeed the median!")
    
    print("\nWith default thresholds [0.75, 0.5, 0.25]:")
    print("- More conservative for low categories (need P(Y>0)≥0.75 to predict >0)")
    print("- Less conservative for high categories (need P(Y>2)≥0.25 to predict 3)")
    
    # Show percentile interpretation
    print("\n" + "="*60)
    print("PERCENTILE INTERPRETATION")
    print("="*60)
    
    probs = [0.1, 0.2, 0.4, 0.3]
    probs_tensor = torch.tensor([probs], dtype=torch.float32)
    
    # Compute cumulative P(Y ≤ k)
    cum_le = np.cumsum(probs)
    print(f"\nProbabilities: {probs}")
    print(f"P(Y ≤ k): {cum_le}")
    
    # Find percentiles
    percentiles = [25, 50, 75]
    for p in percentiles:
        # Find first k where P(Y ≤ k) ≥ p/100
        for k in range(len(probs)):
            if cum_le[k] >= p/100:
                print(f"{p}th percentile: category {k}")
                
                # What threshold gives this?
                # We want smallest k where P(Y > k-1) < threshold
                # Or where P(Y ≤ k) > 1 - threshold
                # So 1 - threshold = p/100
                # threshold = 1 - p/100
                equiv_threshold = 1 - p/100
                pred = compute_threshold_predictions(probs_tensor, 
                                                   thresholds=[equiv_threshold]*3)
                print(f"  Equivalent threshold: {equiv_threshold}")
                print(f"  Threshold prediction: {pred[0].item()}")
                break

if __name__ == "__main__":
    main()