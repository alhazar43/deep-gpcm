#!/usr/bin/env python3
"""
Run 10-epoch training comparison across all three prediction methods.
Generates comprehensive analysis with new prediction accuracy metrics.
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

def run_training_experiment(dataset_name, prediction_method, epochs=10, embedding_strategy='linear_decay'):
    """Run single training experiment with specified prediction method."""
    print(f"\n{'='*60}")
    print(f"Running {prediction_method} prediction method for {dataset_name}")
    print(f"Epochs: {epochs}, Embedding: {embedding_strategy}")
    print(f"{'='*60}")
    
    cmd = [
        'python', 'train.py',
        '--dataset', dataset_name,
        '--epochs', str(epochs),
        '--embedding_strategy', embedding_strategy,
        '--prediction_method', prediction_method,
        '--batch_size', '64',
        '--learning_rate', '0.001',
        '--loss_type', 'ordinal'
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: Training failed for {prediction_method}")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return None
    
    print(f"SUCCESS: {prediction_method} training completed")
    return result.stdout

def load_training_results(dataset_name):
    """Load training results from JSON file."""
    results_path = f"results/train/training_history_{dataset_name}.json"
    if not os.path.exists(results_path):
        return None
    
    with open(results_path, 'r') as f:
        return json.load(f)

def run_comparison_analysis():
    """Run comprehensive analysis with new results."""
    print(f"\n{'='*60}")
    print("Running comprehensive analysis with new metrics...")
    print(f"{'='*60}")
    
    cmd = ['python', 'comprehensive_analysis.py']
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ERROR: Analysis failed")
        print(f"STDOUT: {result.stdout}")
        print(f"STDERR: {result.stderr}")
        return False
    
    print("SUCCESS: Comprehensive analysis completed")
    return True

def organize_results(dataset_name, prediction_methods, embedding_strategies):
    """Organize results from all experiments."""
    all_results = {}
    
    for strategy in embedding_strategies:
        strategy_results = {}
        
        for method in prediction_methods:
            # Each experiment overwrites the same file, so we need to collect immediately
            experiment_name = f"{dataset_name}_{strategy}_{method}"
            results = load_training_results(dataset_name)
            
            if results:
                # Add metadata
                for epoch_data in results:
                    epoch_data['prediction_method'] = method
                    epoch_data['embedding_strategy'] = strategy
                
                strategy_results[method] = results
            
        all_results[strategy] = strategy_results
    
    return all_results

def save_comparison_results(dataset_name, all_results):
    """Save organized comparison results."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create comparison directory
    os.makedirs("results/comparison", exist_ok=True)
    
    # Save detailed results
    comparison_path = f"results/comparison/prediction_method_comparison_{dataset_name}_{timestamp}.json"
    with open(comparison_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"Comparison results saved to: {comparison_path}")
    
    # Create summary
    summary = create_summary(all_results)
    summary_path = f"results/comparison/prediction_method_summary_{dataset_name}_{timestamp}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"Summary saved to: {summary_path}")
    
    return comparison_path, summary_path

def create_summary(all_results):
    """Create summary of final epoch results."""
    summary = {
        'experiment_info': {
            'timestamp': datetime.now().isoformat(),
            'description': 'Prediction method comparison with new accuracy metrics',
            'epochs': 10
        },
        'results_by_strategy': {}
    }
    
    for strategy, strategy_results in all_results.items():
        strategy_summary = {}
        
        for method, method_results in strategy_results.items():
            if method_results:
                final_epoch = method_results[-1]  # Last epoch
                strategy_summary[method] = {
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
        
        summary['results_by_strategy'][strategy] = strategy_summary
    
    return summary

def main():
    """Main experiment runner."""
    parser = argparse.ArgumentParser(description='Run prediction method comparison experiments')
    parser.add_argument('--dataset', type=str, default='synthetic_OC',
                        help='Dataset name (default: synthetic_OC)')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of epochs for each experiment (default: 10)')
    parser.add_argument('--strategies', nargs='+', 
                        default=['ordered', 'unordered', 'linear_decay'],
                        help='Embedding strategies to test (default: all)')
    parser.add_argument('--methods', nargs='+',
                        default=['argmax', 'cumulative', 'expected'],
                        help='Prediction methods to test (default: all)')
    
    args = parser.parse_args()
    
    print(f"Starting prediction method comparison experiment")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs per experiment: {args.epochs}")
    print(f"Embedding strategies: {args.strategies}")
    print(f"Prediction methods: {args.methods}")
    
    # Run all experiments
    all_results = {}
    experiment_count = 0
    total_experiments = len(args.strategies) * len(args.methods)
    
    for strategy in args.strategies:
        strategy_results = {}
        
        for method in args.methods:
            experiment_count += 1
            print(f"\n>>> Experiment {experiment_count}/{total_experiments} <<<")
            
            # Run training
            output = run_training_experiment(args.dataset, method, args.epochs, strategy)
            
            if output is not None:
                # Load results immediately (before next experiment overwrites file)
                results = load_training_results(args.dataset)
                if results:
                    # Add metadata
                    for epoch_data in results:
                        epoch_data['prediction_method'] = method
                        epoch_data['embedding_strategy'] = strategy
                    
                    strategy_results[method] = results
                    print(f"Results collected for {strategy}+{method}")
                else:
                    print(f"WARNING: Could not load results for {strategy}+{method}")
                    strategy_results[method] = None
            else:
                strategy_results[method] = None
        
        all_results[strategy] = strategy_results
    
    # Save organized results
    comparison_path, summary_path = save_comparison_results(args.dataset, all_results)
    
    # Run comprehensive analysis
    analysis_success = run_comparison_analysis()
    
    # Print final summary
    print(f"\n{'='*60}")
    print("PREDICTION METHOD COMPARISON COMPLETE")
    print(f"{'='*60}")
    print(f"Total experiments: {total_experiments}")
    print(f"Dataset: {args.dataset}")
    print(f"Results saved to: {comparison_path}")
    print(f"Summary saved to: {summary_path}")
    
    if analysis_success:
        print("Comprehensive analysis plots generated successfully")
        print("Check results/plots/ for visualization files")
    else:
        print("WARNING: Comprehensive analysis failed")
    
    # Show quick performance comparison
    print(f"\n{'='*60}")
    print("QUICK PERFORMANCE SUMMARY - RAW ACCURACY")
    print(f"{'='*60}")
    
    summary = create_summary(all_results)
    for strategy, strategy_results in summary['results_by_strategy'].items():
        print(f"\n{strategy.upper()} EMBEDDING:")
        for method, metrics in strategy_results.items():
            if metrics:
                print(f"  {method:12}: CatAcc={metrics['categorical_acc']:.3f}, "
                      f"OrdAcc={metrics['ordinal_acc']:.3f}, "
                      f"QWK={metrics['qwk']:.3f}")
        
        # Show prediction consistency analysis
        print(f"\n{strategy.upper()} PREDICTION CONSISTENCY:")
        for method, metrics in strategy_results.items():
            if metrics:
                print(f"  {method:12}: PredCons={metrics['prediction_consistency_acc']:.3f} "
                      f"(ordinal training consistency)")

if __name__ == "__main__":
    main()