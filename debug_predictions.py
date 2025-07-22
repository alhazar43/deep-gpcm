#!/usr/bin/env python3
"""
Debug script to analyze cumulative prediction behavior in detail.
This investigates why cumulative prediction performs worse than expected.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

def analyze_cumulative_prediction(predictions, targets, method='cumulative', threshold=0.5):
    """Analyze cumulative prediction behavior in detail."""
    
    print(f"\n{'='*80}")
    print(f"DETAILED CUMULATIVE PREDICTION ANALYSIS")
    print(f"Method: {method}, Threshold: {threshold}")
    print(f"{'='*80}")
    
    # Get predictions with different methods
    argmax_preds = torch.argmax(predictions, dim=-1)
    
    # Cumulative method analysis
    cum_probs = torch.cumsum(predictions, dim=-1)
    cumulative_preds = torch.zeros_like(cum_probs[..., 0])
    
    # Track threshold crossing for each sample
    threshold_crossings = []
    
    for k in range(predictions.shape[-1]):
        mask = (cum_probs[..., k] > threshold) & (cumulative_preds == 0)
        cumulative_preds = torch.where(mask, 
                                     torch.tensor(k, dtype=cumulative_preds.dtype, device=cumulative_preds.device),
                                     cumulative_preds)
        
        # Count threshold crossings
        crossings = mask.sum().item()
        threshold_crossings.append(crossings)
        
        print(f"Category {k}: {crossings} predictions cross threshold {threshold}")
    
    # Analyze prediction distributions
    print(f"\nPREDICTION DISTRIBUTION COMPARISON:")
    print(f"Target distribution: {Counter(targets.numpy())}")
    print(f"Argmax distribution: {Counter(argmax_preds.numpy())}")
    print(f"Cumulative distribution: {Counter(cumulative_preds.numpy())}")
    
    # Check cumulative probability behavior
    print(f"\nCUMULATIVE PROBABILITY ANALYSIS:")
    sample_idx = 0
    sample_probs = predictions[sample_idx]
    sample_cum_probs = cum_probs[sample_idx]
    
    print(f"Sample {sample_idx}:")
    print(f"  Raw probabilities: {sample_probs.tolist()}")
    print(f"  Cumulative probabilities: {sample_cum_probs.tolist()}")
    print(f"  Target: {targets[sample_idx].item()}")
    print(f"  Argmax prediction: {argmax_preds[sample_idx].item()}")
    print(f"  Cumulative prediction: {cumulative_preds[sample_idx].item()}")
    
    # Test different thresholds
    print(f"\nTHRESHOLD SENSITIVITY ANALYSIS:")
    thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
    
    for thresh in thresholds:
        test_preds = torch.zeros_like(cum_probs[..., 0])
        for k in range(predictions.shape[-1]):
            mask = (cum_probs[..., k] > thresh) & (test_preds == 0)
            test_preds = torch.where(mask, torch.tensor(k, dtype=test_preds.dtype), test_preds)
        
        accuracy = (test_preds == targets).float().mean().item()
        pred_dist = Counter(test_preds.numpy())
        print(f"  Threshold {thresh}: Accuracy = {accuracy:.3f}, Distribution = {dict(pred_dist)}")
    
    # Analyze where cumulative method fails
    print(f"\nERROR ANALYSIS:")
    argmax_correct = (argmax_preds == targets)
    cumulative_correct = (cumulative_preds == targets)
    
    # Cases where argmax is right but cumulative is wrong
    argmax_wins = argmax_correct & (~cumulative_correct)
    print(f"Argmax correct, Cumulative wrong: {argmax_wins.sum().item()} cases")
    
    # Cases where cumulative is right but argmax is wrong  
    cumulative_wins = (~argmax_correct) & cumulative_correct
    print(f"Cumulative correct, Argmax wrong: {cumulative_wins.sum().item()} cases")
    
    # Analyze specific failure cases
    if argmax_wins.sum() > 0:
        failure_indices = torch.where(argmax_wins)[0][:5]  # First 5 failure cases
        print(f"\nSAMPLE FAILURE CASES (Argmax Right, Cumulative Wrong):")
        
        for i, idx in enumerate(failure_indices):
            idx = idx.item()
            print(f"  Case {i+1} (index {idx}):")
            print(f"    Raw probs: {predictions[idx].tolist()}")
            print(f"    Cum probs: {cum_probs[idx].tolist()}")
            print(f"    Target: {targets[idx].item()}")
            print(f"    Argmax: {argmax_preds[idx].item()}")
            print(f"    Cumulative: {cumulative_preds[idx].item()}")
    
    return {
        'threshold_crossings': threshold_crossings,
        'argmax_preds': argmax_preds,
        'cumulative_preds': cumulative_preds,
        'threshold_analysis': thresholds
    }

if __name__ == "__main__":
    # This would be called from benchmark script with actual data
    print("Run this analysis from the benchmark script")