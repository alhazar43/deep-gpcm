#!/usr/bin/env python3
"""
Compare metrics computed with hard vs threshold predictions.
Since both produce discrete labels, we can compute all metrics with both methods.
"""

import torch
import numpy as np
import sys
sys.path.append('/home/steph/dirt-new/deep-gpcm')

from evaluate import load_simple_data, create_data_loaders, load_trained_model
from utils.predictions import compute_unified_predictions, PredictionConfig
from utils.metrics import OrdinalMetrics

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
    
    # Collect predictions
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
    all_targets = all_targets[valid_indices].numpy()
    
    print("="*100)
    print("HARD vs THRESHOLD PREDICTIONS: COMPREHENSIVE COMPARISON")
    print("="*100)
    print("\nSince both methods produce discrete labels, all metrics can be computed with both.\n")
    
    # Test different threshold configurations
    configs = [
        ("Hard (argmax)", None),
        ("Threshold (median 0.5)", PredictionConfig(use_gpu=False, thresholds=[0.5, 0.5, 0.5])),
        ("Threshold (conservative)", PredictionConfig(use_gpu=False, thresholds=[0.8, 0.6, 0.4])),
        ("Threshold (liberal)", PredictionConfig(use_gpu=False, thresholds=[0.3, 0.3, 0.3])),
        ("Threshold (adaptive)", PredictionConfig(use_gpu=False, adaptive_thresholds=True))
    ]
    
    # Metrics to compare
    metrics_to_compare = [
        'categorical_accuracy',
        'quadratic_weighted_kappa',
        'cohen_kappa',
        'ordinal_accuracy',
        'mean_absolute_error',
        'kendall_tau',
        'spearman_correlation'
    ]
    
    # Initialize metric calculator
    ordinal_metrics = OrdinalMetrics(n_cats=4)
    
    # Collect results
    results = {}
    predictions_dict = {}
    
    for config_name, pred_config in configs:
        if config_name.startswith("Hard"):
            # Hard predictions
            y_pred = all_predictions.argmax(dim=-1).numpy()
        else:
            # Threshold predictions
            unified_preds = compute_unified_predictions(all_predictions, config=pred_config)
            y_pred = unified_preds['threshold'].numpy()
        
        predictions_dict[config_name] = y_pred
        results[config_name] = {}
        
        # Compute all metrics
        for metric in metrics_to_compare:
            metric_func = getattr(ordinal_metrics, metric)
            results[config_name][metric] = metric_func(all_targets, y_pred)
    
    # Print comparison table
    print("Metric Comparison Table:")
    print("-" * 100)
    
    # Header
    print(f"{'Metric':<30}", end="")
    for config_name in results.keys():
        print(f"{config_name:<20}", end="")
    print()
    print("-" * 100)
    
    # Results
    for metric in metrics_to_compare:
        print(f"{metric:<30}", end="")
        for config_name in results.keys():
            value = results[config_name][metric]
            print(f"{value:<20.4f}", end="")
        print()
    
    # Analyze differences
    print("\n" + "="*100)
    print("KEY OBSERVATIONS")
    print("="*100)
    
    # Compare hard vs median threshold
    hard_preds = predictions_dict["Hard (argmax)"]
    median_preds = predictions_dict["Threshold (median 0.5)"]
    
    agreement = (hard_preds == median_preds).mean()
    print(f"\n1. Agreement between Hard and Threshold (median 0.5): {agreement:.2%}")
    
    # Where do they differ?
    diff_mask = hard_preds != median_preds
    if diff_mask.sum() > 0:
        print(f"   - They differ on {diff_mask.sum()} out of {len(hard_preds)} samples ({diff_mask.mean():.2%})")
        
        # Analyze the differences
        hard_when_diff = hard_preds[diff_mask]
        thresh_when_diff = median_preds[diff_mask]
        true_when_diff = all_targets[diff_mask]
        
        print(f"   - When they differ:")
        print(f"     - Average hard prediction: {hard_when_diff.mean():.2f}")
        print(f"     - Average threshold prediction: {thresh_when_diff.mean():.2f}")
        print(f"     - Average true label: {true_when_diff.mean():.2f}")
    
    # QWK comparison
    print(f"\n2. QWK Performance:")
    for config_name in results.keys():
        qwk = results[config_name]['quadratic_weighted_kappa']
        print(f"   - {config_name}: {qwk:.4f}")
    
    # Best configuration per metric
    print(f"\n3. Best Configuration per Metric:")
    for metric in metrics_to_compare:
        best_config = max(results.keys(), key=lambda x: results[x][metric])
        best_value = results[best_config][metric]
        print(f"   - {metric}: {best_config} ({best_value:.4f})")
    
    print("\n" + "="*100)
    print("CONCLUSIONS")
    print("="*100)
    print("""
1. Both hard and threshold predictions produce discrete labels suitable for all metrics.

2. QWK (Quadratic Weighted Kappa) as an ordinal metric can benefit from threshold predictions
   that better respect the ordinal structure of the data.

3. Different threshold configurations offer different trade-offs:
   - Conservative thresholds: More cautious predictions, lower scores
   - Liberal thresholds: More confident predictions, may improve ordinal accuracy
   - Median (0.5): Balanced approach, computes the median of the distribution
   - Adaptive: Data-driven thresholds based on the distribution

4. The choice between hard and threshold predictions depends on:
   - Whether ordinal structure matters (use threshold for ordinal-aware predictions)
   - The specific metric being optimized
   - The desired behavior at decision boundaries
""")

if __name__ == "__main__":
    main()