#!/usr/bin/env python3
"""
Manual test of the three prediction methods to see their differences.
"""

import torch
import numpy as np
from models.model import DeepGpcmModel

def test_prediction_methods():
    """Test all three prediction methods on sample GPCM probabilities."""
    
    # Create a sample model just for testing prediction methods
    model = DeepGpcmModel(n_questions=10, n_cats=4, prediction_method='argmax')
    
    # Create sample GPCM probabilities - design scenarios to show differences
    print("Testing Prediction Methods")
    print("=" * 50)
    
    # Test Case 1: Clear winner (should be same for all methods)
    probs1 = torch.tensor([[0.1, 0.2, 0.6, 0.1]])  # Clear winner at category 2
    print("Test Case 1: Clear winner (category 2)")
    print(f"Probabilities: {probs1[0].numpy()}")
    
    pred_argmax = model.gpcm_predict_argmax(probs1)
    pred_cumulative = model.gpcm_predict_cumulative(probs1)
    pred_expected = model.gpcm_predict_expected(probs1)
    
    print(f"Argmax prediction:     {pred_argmax.item():.0f}")
    print(f"Cumulative prediction: {pred_cumulative.item():.0f}")
    print(f"Expected prediction:   {pred_expected.item():.0f}")
    print()
    
    # Test Case 2: Bimodal distribution (should show differences)
    probs2 = torch.tensor([[0.4, 0.1, 0.1, 0.4]])  # Bimodal: categories 0 and 3
    print("Test Case 2: Bimodal distribution (0 vs 3)")
    print(f"Probabilities: {probs2[0].numpy()}")
    
    pred_argmax = model.gpcm_predict_argmax(probs2)
    pred_cumulative = model.gpcm_predict_cumulative(probs2)
    pred_expected = model.gpcm_predict_expected(probs2)
    
    print(f"Argmax prediction:     {pred_argmax.item():.0f}")
    print(f"Cumulative prediction: {pred_cumulative.item():.0f}")
    print(f"Expected prediction:   {pred_expected.item():.0f}")
    print()
    
    # Test Case 3: Gradual decline (ordinal structure)
    probs3 = torch.tensor([[0.4, 0.3, 0.2, 0.1]])  # Declining probabilities
    print("Test Case 3: Gradual decline (0 > 1 > 2 > 3)")
    print(f"Probabilities: {probs3[0].numpy()}")
    
    pred_argmax = model.gpcm_predict_argmax(probs3)
    pred_cumulative = model.gpcm_predict_cumulative(probs3)
    pred_expected = model.gpcm_predict_expected(probs3)
    
    print(f"Argmax prediction:     {pred_argmax.item():.0f}")
    print(f"Cumulative prediction: {pred_cumulative.item():.0f}")
    print(f"Expected prediction:   {pred_expected.item():.0f}")
    print()
    
    # Test Case 4: Near uniform (should show method differences)
    probs4 = torch.tensor([[0.3, 0.25, 0.25, 0.2]])  # Near uniform with slight bias
    print("Test Case 4: Near uniform distribution")
    print(f"Probabilities: {probs4[0].numpy()}")
    
    pred_argmax = model.gpcm_predict_argmax(probs4)
    pred_cumulative = model.gpcm_predict_cumulative(probs4)
    pred_expected = model.gpcm_predict_expected(probs4)
    
    print(f"Argmax prediction:     {pred_argmax.item():.0f}")
    print(f"Cumulative prediction: {pred_cumulative.item():.0f}")
    print(f"Expected prediction:   {pred_expected.item():.0f}")
    print()
    
    # Show cumulative probabilities to explain cumulative method
    print("Cumulative Probabilities Analysis")
    print("-" * 30)
    for i, probs in enumerate([probs1, probs2, probs3, probs4], 1):
        cum_probs = torch.cumsum(probs, dim=-1)
        print(f"Case {i}: {cum_probs[0].numpy()}")
        # Find where P(Y <= k) > 0.5
        first_above_half = torch.where(cum_probs[0] > 0.5)[0]
        if len(first_above_half) > 0:
            print(f"         First P(Y <= k) > 0.5 at k={first_above_half[0].item()}")
        print()

if __name__ == "__main__":
    test_prediction_methods()