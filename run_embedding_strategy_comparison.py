#!/usr/bin/env python3
"""
Run embedding strategy comparison with new prediction accuracy metrics.
Compare ordered, unordered, and linear_decay embedding strategies.
"""

import os
import sys
import json
import subprocess
import argparse
from datetime import datetime
import numpy as np

# Fix MKL threading issue
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

def run_training_experiment(dataset_name, embedding_strategy, epochs=10):
    """Run single training experiment with specified embedding strategy."""
    print(f"\n{'='*60}")
    print(f"Running {embedding_strategy} embedding strategy for {dataset_name}")
    print(f"Epochs: {epochs}")
    print(f"{'='*60}")
    
    cmd = [
        'python', 'train.py',
        '--dataset', dataset_name,
        '--epochs', str(epochs),
        '--embedding_strategy', embedding_strategy,
        '--batch_size', '64',
        '--learning_rate', '0.001',
        '--loss_type', 'ordinal'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: Training failed for {embedding_strategy}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return None
    
    print(f"SUCCESS: {embedding_strategy} training completed")
    return result.stdout

def load_training_results(dataset_name):
    """Load training results from JSON file."""
    results_path = f"results/train/training_history_{dataset_name}.json"
    if not os.path.exists(results_path):
        return None
    
    with open(results_path, 'r') as f:
        return json.load(f)

def save_comparison_results(dataset_name, all_results):
    """Save organized comparison results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create comparison directory
    os.makedirs("results/comparison", exist_ok=True)
    
    # Save detailed results
    comparison_path = f"results/comparison/embedding_strategy_comparison_{dataset_name}_{timestamp}.json"
    with open(comparison_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Comparison results saved to: {comparison_path}")
    
    # Create summary
    summary = create_summary(all_results)
    summary_path = f"results/comparison/embedding_strategy_summary_{dataset_name}_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")
    
    return comparison_path, summary_path

def create_summary(all_results):
    """Create summary of final epoch results."""
    summary = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'description': 'Embedding strategy comparison with new accuracy metrics',
            'epochs': 10
        },
        'results_by_strategy': {}
    }
    
    for strategy, strategy_results in all_results.items():
        if strategy_results:
            final_epoch = strategy_results[-1]  # Last epoch
            summary['results_by_strategy'][strategy] = {
                'final_epoch': final_epoch['epoch'],
                'train_loss': final_epoch['train_loss'],
                'valid_loss': final_epoch['valid_loss'],
                'categorical_acc': final_epoch['categorical_acc'],
                'ordinal_acc': final_epoch['ordinal_acc'],
                'mae': final_epoch['mae'],
                'qwk': final_epoch['qwk'],
                'prediction_consistency_acc': final_epoch['prediction_consistency_acc'],
                'ordinal_ranking_acc': final_epoch['ordinal_ranking_acc'],
                'distribution_consistency': final_epoch['distribution_consistency']
            }
    
    return summary

def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description='Run embedding strategy comparison experiments')
    parser.add_argument('--dataset', type=str, default='synthetic_OC',
                        help='Dataset name (default: synthetic_OC)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for each experiment (default: 10)')
    parser.add_argument('--strategies', nargs='+', 
                        default=['ordered', 'unordered', 'linear_decay', 'adjacent_weighted'],
                        help='Embedding strategies to test (default: all)')
    
    args = parser.parse_args()
    
    print(f"Starting embedding strategy comparison experiment")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs per experiment: {args.epochs}")
    print(f"Embedding strategies: {args.strategies}")
    
    # Run all experiments
    all_results = {}
    experiment_count = 0
    total_experiments = len(args.strategies)
    
    for strategy in args.strategies:
        experiment_count += 1
        print(f"\n>>> Experiment {experiment_count}/{total_experiments} <<<")
        
        # Run training
        output = run_training_experiment(args.dataset, strategy, args.epochs)
        
        if output is not None:
            # Load results immediately (before next experiment overwrites file)
            results = load_training_results(args.dataset)
            if results:
                # Add metadata
                for epoch_data in results:
                    epoch_data['embedding_strategy'] = strategy
                
                all_results[strategy] = results
                print(f"Results collected for {strategy}")
            else:
                print(f"WARNING: Could not load results for {strategy}")
                all_results[strategy] = None
        else:
            all_results[strategy] = None
    
    # Save organized results
    comparison_path, summary_path = save_comparison_results(args.dataset, all_results)
    
    # Print final summary
    print(f"\n{'='*60}")
    print("EMBEDDING STRATEGY COMPARISON COMPLETE")
    print(f"{'='*60}")
    print(f"Total experiments: {total_experiments}")
    print(f"Dataset: {args.dataset}")
    print(f"Results saved to: {comparison_path}")
    print(f"Summary saved to: {summary_path}")
    
    # Show quick performance comparison
    print(f"\n{'='*60}")
    print("QUICK PERFORMANCE SUMMARY - EMBEDDING STRATEGIES")
    print(f"{'='*60}")
    
    summary = create_summary(all_results)
    for strategy, metrics in summary['results_by_strategy'].items():
        if metrics:
            print(f"\n{strategy.upper()} EMBEDDING:")
            print(f"  Categorical Acc: {metrics['categorical_acc']:.3f}")
            print(f"  Ordinal Acc:     {metrics['ordinal_acc']:.3f}")
            print(f"  Pred Consistency: {metrics['prediction_consistency_acc']:.3f} (cumulative method)")
            print(f"  Ordinal Ranking:  {metrics['ordinal_ranking_acc']:.3f}")
            print(f"  MAE:             {metrics['mae']:.3f}")

if __name__ == "__main__":
    main()