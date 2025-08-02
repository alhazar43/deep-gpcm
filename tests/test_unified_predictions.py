#!/usr/bin/env python3
"""
Test the unified prediction system with edge cases.
"""

import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.predictions import (
    compute_unified_predictions,
    PredictionConfig,
    extract_probabilities_from_model_output
)
from utils.metrics import compute_metrics_multimethod
from utils.monitoring import get_monitor, reset_monitor

def test_basic_predictions():
    """Test basic prediction functionality."""
    print("Testing basic predictions...")
    
    # Create test data
    batch_size = 4
    n_cats = 4
    
    # Normal probabilities
    probs = torch.tensor([
        [0.7, 0.2, 0.05, 0.05],
        [0.1, 0.6, 0.2, 0.1],
        [0.05, 0.15, 0.6, 0.2],
        [0.05, 0.05, 0.1, 0.8]
    ])
    
    targets = torch.tensor([0, 1, 2, 3])
    
    # Compute predictions
    config = PredictionConfig()
    predictions = compute_unified_predictions(probs, config=config)
    
    print(f"Hard predictions: {predictions['hard']}")
    print(f"Soft predictions: {predictions['soft']}")
    print(f"Threshold predictions: {predictions['threshold']}")
    
    # Debug threshold computation
    from utils.predictions import categorical_to_cumulative
    cum_probs = categorical_to_cumulative(probs)
    print(f"\nCumulative probs P(Y > k):\n{cum_probs}")
    print(f"Default thresholds: {torch.linspace(0.75, 0.25, 3)}")
    
    # Compute metrics
    metrics = compute_metrics_multimethod(
        targets,
        predictions,
        probs,
        n_cats=n_cats
    )
    
    print("\nMetrics:")
    for metric, value in metrics.items():
        if not metric.endswith('_method') and not metric.endswith('_error') and isinstance(value, (int, float)):
            method = metrics.get(f'{metric}_method', 'N/A')
            if not np.isnan(value):
                print(f"  {metric}: {value:.4f} (method: {method})")
    
    return True

def test_edge_cases():
    """Test edge cases from test data."""
    print("\n\nTesting edge cases...")
    
    # Load test cases
    test_cases_file = "tests/data/unified_prediction_test_cases.pt"
    if not os.path.exists(test_cases_file):
        print(f"Test cases file not found: {test_cases_file}")
        return False
    
    test_cases = torch.load(test_cases_file)
    single_cases = test_cases['single_predictions']
    
    # Test each edge case
    for case_name, (probs, targets, description) in single_cases.items():
        print(f"\n{case_name}: {description}")
        
        try:
            predictions = compute_unified_predictions(probs)
            
            # Check for NaN or inf
            for method in ['hard', 'soft', 'threshold']:
                pred = predictions[method]
                if torch.isnan(pred).any() or torch.isinf(pred).any():
                    print(f"  WARNING: {method} predictions contain NaN or inf!")
                else:
                    print(f"  {method}: {pred}")
            
            # Compute metrics
            metrics = compute_metrics_multimethod(targets, predictions, probs)
            
            # Show key metrics
            for metric in ['categorical_accuracy', 'ordinal_accuracy', 'mean_absolute_error']:
                if metric in metrics:
                    print(f"  {metric}: {metrics[metric]:.4f}")
                    
        except Exception as e:
            print(f"  ERROR: {e}")
            return False
    
    return True

def test_model_output_extraction():
    """Test extracting probabilities from model output."""
    print("\n\nTesting model output extraction...")
    
    # Simulate model output
    batch_size = 2
    seq_len = 3
    n_cats = 4
    
    student_abilities = torch.randn(batch_size, seq_len)
    item_thresholds = torch.randn(batch_size, seq_len, n_cats - 1)
    discrimination_params = torch.ones(batch_size, seq_len)
    gpcm_probs = torch.rand(batch_size, seq_len, n_cats)
    gpcm_probs = gpcm_probs / gpcm_probs.sum(dim=-1, keepdim=True)
    
    model_output = (student_abilities, item_thresholds, discrimination_params, gpcm_probs)
    
    # Extract probabilities
    extracted_probs = extract_probabilities_from_model_output(model_output)
    
    print(f"Extracted shape: {extracted_probs.shape}")
    print(f"Matches expected: {torch.allclose(extracted_probs, gpcm_probs)}")
    
    return True

def test_monitoring():
    """Test monitoring functionality."""
    print("\n\nTesting monitoring...")
    
    reset_monitor()
    monitor = get_monitor()
    
    # Run some predictions
    probs = torch.rand(100, 4)
    probs = probs / probs.sum(dim=-1, keepdim=True)
    
    for i in range(5):
        predictions = compute_unified_predictions(probs)
    
    # Print summary
    monitor.print_summary()
    
    # Save summary
    filepath = monitor.save_summary("test_monitor.json")
    print(f"\nMonitor summary saved to: {filepath}")
    
    return True

def main():
    """Run all tests."""
    print("="*60)
    print("UNIFIED PREDICTION SYSTEM TESTS")
    print("="*60)
    
    tests = [
        test_basic_predictions,
        test_edge_cases,
        test_model_output_extraction,
        test_monitoring
    ]
    
    passed = 0
    for test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n✓ {test_func.__name__} PASSED")
            else:
                print(f"\n✗ {test_func.__name__} FAILED")
        except Exception as e:
            print(f"\n✗ {test_func.__name__} FAILED with exception: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print(f"Tests passed: {passed}/{len(tests)}")
    print("="*60)
    
    return passed == len(tests)

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)