#!/usr/bin/env python3
"""
Compare detailed metrics across all prediction methods.
"""

import torch
import numpy as np
import sys
import json
sys.path.append('/home/steph/dirt-new/deep-gpcm')

from evaluate import load_simple_data, create_data_loaders, load_trained_model
from utils.predictions import compute_unified_predictions, PredictionConfig
from utils.metrics import compute_metrics_multimethod

def evaluate_with_config(model, test_loader, device, config_name, pred_config=None):
    """Evaluate model with specific prediction configuration."""
    model.eval()
    all_predictions = []
    all_targets = []
    all_masks = []
    
    with torch.no_grad():
        for questions, responses, mask in test_loader:
            questions = questions.to(device)
            responses = responses.to(device)
            mask = mask.to(device)
            
            # Get model output
            _, _, _, gpcm_probs = model(questions, responses)
            
            # Flatten and filter
            probs_flat = gpcm_probs.view(-1, gpcm_probs.size(-1))
            responses_flat = responses.view(-1)
            mask_flat = mask.view(-1)
            
            all_predictions.append(probs_flat.cpu())
            all_targets.append(responses_flat.cpu())
            all_masks.append(mask_flat.cpu())
    
    # Combine
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    # Filter valid
    valid_indices = all_masks.bool()
    all_predictions = all_predictions[valid_indices]
    all_targets = all_targets[valid_indices]
    
    # Compute predictions
    if pred_config is None:
        pred_config = PredictionConfig(use_gpu=False)
    
    unified_preds = compute_unified_predictions(all_predictions, config=pred_config)
    
    # Compute all metrics
    metrics = compute_metrics_multimethod(
        all_targets,
        unified_preds,
        all_predictions,
        n_cats=4,
        metrics_list=[
            'categorical_accuracy',
            'ordinal_accuracy',
            'mean_absolute_error',
            'quadratic_weighted_kappa',
            'cohen_kappa',
            'spearman_correlation',
            'kendall_tau',
            'pearson_correlation',
            'mean_squared_error',
            'cross_entropy'
        ]
    )
    
    return metrics, unified_preds

def main():
    # Load model and data
    model_path = "save_models/best_deep_gpcm_synthetic_OC.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model, _, _ = load_trained_model(model_path, device)
    model.eval()
    
    # Load data
    train_data, test_data, n_questions, n_cats = load_simple_data(
        "data/synthetic_OC/synthetic_oc_train.txt",
        "data/synthetic_OC/synthetic_oc_test.txt"
    )
    _, test_loader = create_data_loaders(train_data, test_data, batch_size=32)
    
    print("="*80)
    print("DETAILED METRICS COMPARISON ACROSS PREDICTION METHODS")
    print("="*80)
    
    # Test different configurations
    configs = [
        ("Default (median [0.5, 0.5, 0.5])", PredictionConfig(use_gpu=False)),
        ("Old defaults [0.75, 0.5, 0.25]", PredictionConfig(use_gpu=False, thresholds=[0.75, 0.5, 0.25])),
        ("Conservative [0.8, 0.6, 0.4]", PredictionConfig(use_gpu=False, thresholds=[0.8, 0.6, 0.4])),
        ("Liberal [0.3, 0.3, 0.3]", PredictionConfig(use_gpu=False, thresholds=[0.3, 0.3, 0.3])),
        ("Adaptive", PredictionConfig(use_gpu=False, adaptive_thresholds=True))
    ]
    
    all_results = {}
    
    for config_name, pred_config in configs:
        print(f"\n{config_name}")
        print("-"*80)
        
        metrics, predictions = evaluate_with_config(model, test_loader, device, config_name, pred_config)
        all_results[config_name] = metrics
        
        # Print prediction statistics
        print(f"\nPrediction Statistics:")
        for method in ['hard', 'soft', 'threshold']:
            if method in predictions:
                pred = predictions[method]
                if method == 'soft':
                    print(f"  {method}: mean={pred.mean():.3f}, std={pred.std():.3f}")
                else:
                    print(f"  {method}: mean={pred.float().mean():.3f}, std={pred.float().std():.3f}")
    
    # Print comparison table
    print("\n" + "="*80)
    print("METRICS COMPARISON TABLE")
    print("="*80)
    
    # Get all metrics
    first_config = list(all_results.values())[0]
    metric_names = [k for k in first_config.keys() if not k.endswith('_method') and not k.endswith('_error') and not k == 'method_comparisons']
    
    # Print header
    print(f"\n{'Metric':<30} {'Method':<10}", end="")
    for config_name in all_results.keys():
        print(f" {config_name[:15]:<15}", end="")
    print()
    print("-"*130)
    
    # Print each metric
    for metric in sorted(metric_names):
        if metric in first_config:
            method = first_config.get(f'{metric}_method', 'N/A')
            print(f"{metric:<30} {method:<10}", end="")
            
            for config_name, results in all_results.items():
                value = results.get(metric, float('nan'))
                if not np.isnan(value):
                    print(f" {value:>15.4f}", end="")
                else:
                    print(f" {'N/A':>15}", end="")
            print()
    
    # Special focus on key metrics by prediction method
    print("\n" + "="*80)
    print("KEY METRICS BY PREDICTION METHOD")
    print("="*80)
    
    key_metrics = {
        'hard': ['categorical_accuracy', 'quadratic_weighted_kappa', 'cohen_kappa'],
        'soft': ['mean_absolute_error', 'mean_squared_error', 'spearman_correlation', 'kendall_tau', 'pearson_correlation'],
        'threshold': ['ordinal_accuracy']
    }
    
    for method, metrics_list in key_metrics.items():
        print(f"\n{method.upper()} Predictions:")
        print("-"*80)
        
        for metric in metrics_list:
            print(f"\n{metric}:")
            for config_name, results in all_results.items():
                value = results.get(metric, float('nan'))
                if not np.isnan(value):
                    print(f"  {config_name:<40} {value:.4f}")

if __name__ == "__main__":
    main()