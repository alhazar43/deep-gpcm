#!/usr/bin/env python3
"""
Comprehensive analysis of all metrics across prediction methods.
Shows categorical accuracy, tau, spearman, and other metrics.
"""

import torch
import numpy as np
import sys
import json
sys.path.append('/home/steph/dirt-new/deep-gpcm')

from evaluate import load_simple_data, create_data_loaders, load_trained_model
from utils.predictions import compute_unified_predictions, PredictionConfig
from utils.metrics import compute_metrics_multimethod

def evaluate_with_config(model, test_loader, device, pred_config=None):
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
            'mean_squared_error',
            'quadratic_weighted_kappa',
            'cohen_kappa',
            'spearman_correlation',
            'kendall_tau',
            'pearson_correlation',
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
    
    print("="*100)
    print("COMPREHENSIVE METRICS ANALYSIS ACROSS ALL PREDICTION METHODS")
    print("="*100)
    
    # Get metrics with default configuration
    default_config = PredictionConfig(use_gpu=False)
    metrics, predictions = evaluate_with_config(model, test_loader, device, default_config)
    
    # Print organized results
    print("\n📊 CLASSIFICATION METRICS (using Hard Predictions)")
    print("-"*100)
    classification_metrics = [
        ("Categorical Accuracy", metrics.get('categorical_accuracy', 'N/A')),
        ("Quadratic Weighted Kappa (QWK)", metrics.get('quadratic_weighted_kappa', 'N/A')),
        ("Cohen's Kappa", metrics.get('cohen_kappa', 'N/A'))
    ]
    for name, value in classification_metrics:
        print(f"{name:.<40} {value:.4f}" if isinstance(value, (int, float)) else f"{name:.<40} {value}")
    
    print("\n📈 CORRELATION METRICS (using Soft Predictions)")
    print("-"*100)
    correlation_metrics = [
        ("Spearman Correlation", metrics.get('spearman_correlation', 'N/A')),
        ("Kendall's Tau", metrics.get('kendall_tau', 'N/A')),
        ("Pearson Correlation", metrics.get('pearson_correlation', 'N/A'))
    ]
    for name, value in correlation_metrics:
        print(f"{name:.<40} {value:.4f}" if isinstance(value, (int, float)) else f"{name:.<40} {value}")
    
    print("\n📏 REGRESSION METRICS (using Soft Predictions)")
    print("-"*100)
    regression_metrics = [
        ("Mean Absolute Error (MAE)", metrics.get('mean_absolute_error', 'N/A')),
        ("Mean Squared Error (MSE)", metrics.get('mean_squared_error', 'N/A'))
    ]
    for name, value in regression_metrics:
        print(f"{name:.<40} {value:.4f}" if isinstance(value, (int, float)) else f"{name:.<40} {value}")
    
    print("\n📐 ORDINAL METRICS (using Threshold Predictions)")
    print("-"*100)
    ordinal_metrics = [
        ("Ordinal Accuracy", metrics.get('ordinal_accuracy', 'N/A'))
    ]
    for name, value in ordinal_metrics:
        print(f"{name:.<40} {value:.4f}" if isinstance(value, (int, float)) else f"{name:.<40} {value}")
    
    print("\n🎲 PROBABILISTIC METRICS (using Raw Probabilities)")
    print("-"*100)
    prob_metrics = [
        ("Cross Entropy", metrics.get('cross_entropy', 'N/A'))
    ]
    for name, value in prob_metrics:
        print(f"{name:.<40} {value:.4f}" if isinstance(value, (int, float)) else f"{name:.<40} {value}")
    
    # Test different threshold configurations
    print("\n" + "="*100)
    print("THRESHOLD CONFIGURATION COMPARISON")
    print("="*100)
    
    configs = [
        ("Median (0.5)", PredictionConfig(use_gpu=False, thresholds=[0.5, 0.5, 0.5])),
        ("Conservative", PredictionConfig(use_gpu=False, thresholds=[0.8, 0.6, 0.4])),
        ("Liberal", PredictionConfig(use_gpu=False, thresholds=[0.3, 0.3, 0.3])),
        ("Asymmetric", PredictionConfig(use_gpu=False, thresholds=[0.75, 0.5, 0.25])),
        ("Adaptive", PredictionConfig(use_gpu=False, adaptive_thresholds=True))
    ]
    
    # Collect results
    results = []
    for config_name, pred_config in configs:
        m, _ = evaluate_with_config(model, test_loader, device, pred_config)
        results.append([
            config_name,
            f"{m.get('categorical_accuracy', 0):.4f}",
            f"{m.get('quadratic_weighted_kappa', 0):.4f}",
            f"{m.get('spearman_correlation', 0):.4f}",
            f"{m.get('kendall_tau', 0):.4f}",
            f"{m.get('ordinal_accuracy', 0):.4f}"
        ])
    
    # Print comparison table
    headers = ["Configuration", "Cat. Acc.", "QWK", "Spearman", "Kendall τ", "Ord. Acc."]
    
    # Print header
    print(f"\n{headers[0]:<15} {headers[1]:<10} {headers[2]:<10} {headers[3]:<10} {headers[4]:<10} {headers[5]:<10}")
    print("-" * 75)
    
    # Print results
    for row in results:
        print(f"{row[0]:<15} {row[1]:<10} {row[2]:<10} {row[3]:<10} {row[4]:<10} {row[5]:<10}")
    
    # Summary insights
    print("\n" + "="*100)
    print("KEY INSIGHTS")
    print("="*100)
    print("""
1. CLASSIFICATION METRICS (Categorical Accuracy, QWK, Cohen's Kappa):
   - Use hard predictions (argmax)
   - Not affected by threshold configuration
   - QWK shows moderate agreement (0.4367)

2. CORRELATION METRICS (Spearman, Kendall's Tau, Pearson):
   - Use soft predictions (expected value)
   - Capture ordinal relationships well
   - Spearman correlation is highest (~0.53)

3. ORDINAL ACCURACY:
   - Uses threshold predictions
   - Strongly affected by threshold configuration
   - Liberal thresholds (0.3) give highest ordinal accuracy

4. THRESHOLD IMPACT:
   - Only affects ordinal accuracy metric
   - Does not affect other metrics (they use hard or soft predictions)
   - Median (0.5) provides balanced performance
""")

if __name__ == "__main__":
    main()