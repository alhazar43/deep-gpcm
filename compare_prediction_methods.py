#!/usr/bin/env python3
"""
Compare GPCM Prediction Methods

Tests all three prediction methods:
1. Argmax (current) - uses P(Y = k)
2. Cumulative (GPCM-consistent) - uses P(Y ≤ k) 
3. Expected (ordinal regression) - uses E[Y] = Σ k * P(Y = k)

This helps identify which method provides better ordinal consistency.
"""

import os
import json
import pandas as pd
from datetime import datetime
import subprocess
import argparse

def run_experiment(dataset, prediction_method, epochs=10, embedding_strategy='linear_decay'):
    """Run training experiment with specific prediction method."""
    print(f"Running {dataset} with {prediction_method} prediction method...")
    
    cmd = [
        'python', 'train.py',
        '--dataset', dataset,
        '--epochs', str(epochs),
        '--prediction_method', prediction_method,
        '--embedding_strategy', embedding_strategy
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr

def compare_prediction_methods(dataset='synthetic_OC', epochs=10):
    """Compare all three prediction methods on the same dataset."""
    
    print("=" * 60)
    print("GPCM PREDICTION METHOD COMPARISON")
    print("=" * 60)
    print(f"Dataset: {dataset}")
    print(f"Epochs: {epochs}")
    print("")
    
    methods = ['argmax', 'cumulative', 'expected']
    results = {}
    
    for method in methods:
        print(f"Testing {method} method...")
        success, output = run_experiment(dataset, method, epochs)
        
        if success:
            # Try to load the training history
            history_path = f"results/train/training_history_{dataset}.json"
            if os.path.exists(history_path):
                with open(history_path, 'r') as f:
                    history = json.load(f)
                    final_metrics = history[-1]  # Last epoch
                    results[method] = {
                        'categorical_acc': final_metrics['categorical_acc'],
                        'ordinal_acc': final_metrics['ordinal_acc'],
                        'qwk': final_metrics['qwk'],
                        'mae': final_metrics['mae'],
                        'final_loss': final_metrics['valid_loss']
                    }
            print(f"  ✓ {method} completed successfully")
        else:
            print(f"  ✗ {method} failed: {output}")
            results[method] = None
        
        print("")
    
    # Display comparison
    print("RESULTS COMPARISON")
    print("-" * 60)
    print(f"{'Method':<12} {'Cat Acc':<8} {'Ord Acc':<8} {'QWK':<8} {'MAE':<8} {'Loss':<8}")
    print("-" * 60)
    
    for method in methods:
        if results[method]:
            r = results[method]
            print(f"{method:<12} {r['categorical_acc']:<8.3f} {r['ordinal_acc']:<8.3f} "
                  f"{r['qwk']:<8.3f} {r['mae']:<8.3f} {r['final_loss']:<8.3f}")
        else:
            print(f"{method:<12} {'FAILED':<8} {'FAILED':<8} {'FAILED':<8} {'FAILED':<8} {'FAILED':<8}")
    
    print("")
    
    # Analysis
    valid_results = {k: v for k, v in results.items() if v is not None}
    if len(valid_results) > 1:
        print("ANALYSIS")
        print("-" * 30)
        
        # Find best method for each metric
        best_cat = max(valid_results.items(), key=lambda x: x[1]['categorical_acc'])
        best_ord = max(valid_results.items(), key=lambda x: x[1]['ordinal_acc'])
        best_qwk = max(valid_results.items(), key=lambda x: x[1]['qwk'])
        best_mae = min(valid_results.items(), key=lambda x: x[1]['mae'])  # Lower is better
        
        print(f"Best Categorical Accuracy: {best_cat[0]} ({best_cat[1]['categorical_acc']:.3f})")
        print(f"Best Ordinal Accuracy: {best_ord[0]} ({best_ord[1]['ordinal_acc']:.3f})")
        print(f"Best QWK: {best_qwk[0]} ({best_qwk[1]['qwk']:.3f})")
        print(f"Best MAE: {best_mae[0]} ({best_mae[1]['mae']:.3f})")
        
        # Check if cumulative method shows better ordinal consistency
        if 'cumulative' in valid_results and 'argmax' in valid_results:
            cum_ord = valid_results['cumulative']['ordinal_acc']
            arg_ord = valid_results['argmax']['ordinal_acc']
            cum_qwk = valid_results['cumulative']['qwk']
            arg_qwk = valid_results['argmax']['qwk']
            
            print(f"\nCumulative vs Argmax:")
            print(f"  Ordinal Accuracy: {cum_ord:.3f} vs {arg_ord:.3f} "
                  f"({'BETTER' if cum_ord > arg_ord else 'WORSE'})")
            print(f"  QWK: {cum_qwk:.3f} vs {arg_qwk:.3f} "
                  f"({'BETTER' if cum_qwk > arg_qwk else 'WORSE'})")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_path = f"results/comparison/prediction_method_comparison_{timestamp}.json"
    os.makedirs("results/comparison", exist_ok=True)
    
    comparison_data = {
        'dataset': dataset,
        'epochs': epochs,
        'timestamp': timestamp,
        'results': results
    }
    
    with open(results_path, 'w') as f:
        json.dump(comparison_data, f, indent=2)
    
    print(f"\nResults saved to: {results_path}")
    
    return results

def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Compare GPCM prediction methods')
    parser.add_argument('--dataset', type=str, default='synthetic_OC',
                        help='Dataset to test (default: synthetic_OC)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs (default: 10)')
    
    args = parser.parse_args()
    
    compare_prediction_methods(args.dataset, args.epochs)

if __name__ == "__main__":
    main()