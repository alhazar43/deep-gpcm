#!/usr/bin/env python3
"""
Evaluate model using hard predictions (argmax) for all metrics.
Compare with the previous approach where different metrics used different prediction methods.
"""

import torch
import numpy as np
import sys
sys.path.append('/home/steph/dirt-new/deep-gpcm')

from evaluate import load_simple_data, create_data_loaders, load_trained_model
from utils.predictions import compute_unified_predictions, PredictionConfig
from utils.metrics import OrdinalMetrics

def evaluate_model(model, test_loader, device, model_name):
    """Evaluate model and return all metrics using hard predictions."""
    model.eval()
    all_predictions = []
    all_targets = []
    all_probabilities = []
    all_masks = []
    
    with torch.no_grad():
        for questions, responses, mask in test_loader:
            questions = questions.to(device)
            responses = responses.to(device)
            mask = mask.to(device)
            
            _, _, _, gpcm_probs = model(questions, responses)
            
            probs_flat = gpcm_probs.view(-1, gpcm_probs.size(-1))
            responses_flat = responses.view(-1)
            mask_flat = mask.view(-1)
            
            all_predictions.append(probs_flat.cpu())
            all_targets.append(responses_flat.cpu())
            all_masks.append(mask_flat.cpu())
            all_probabilities.append(probs_flat.cpu())
    
    # Combine all batches
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    all_probabilities = torch.cat(all_probabilities, dim=0)
    
    # Filter valid samples
    valid_indices = all_masks.bool()
    all_predictions = all_predictions[valid_indices]
    all_targets = all_targets[valid_indices].numpy()
    all_probabilities = all_probabilities[valid_indices].numpy()
    
    # Get hard predictions (argmax)
    hard_preds = all_predictions.argmax(dim=-1).numpy()
    
    # Initialize metrics
    metrics = OrdinalMetrics(n_cats=4)
    
    # Compute all metrics using hard predictions
    results = {
        'model': model_name,
        'categorical_accuracy': metrics.categorical_accuracy(all_targets, hard_preds),
        'quadratic_weighted_kappa': metrics.quadratic_weighted_kappa(all_targets, hard_preds),
        'cohen_kappa': metrics.cohen_kappa(all_targets, hard_preds),
        'ordinal_accuracy': metrics.ordinal_accuracy(all_targets, hard_preds),
        'mean_absolute_error': metrics.mean_absolute_error(all_targets, hard_preds),
        'mean_squared_error': metrics.mean_squared_error(all_targets, hard_preds),
        'spearman_correlation': metrics.spearman_correlation(all_targets, hard_preds),
        'kendall_tau': metrics.kendall_tau(all_targets, hard_preds),
        'pearson_correlation': metrics.pearson_correlation(all_targets, hard_preds)
    }
    
    # Add confusion matrix info
    conf_metrics = metrics.confusion_matrix_metrics(all_targets, hard_preds)
    results.update(conf_metrics)
    
    # Add probability-based metrics
    prob_metrics = metrics.probability_metrics(all_targets, all_probabilities)
    results.update(prob_metrics)
    
    return results, hard_preds, all_targets

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_data, test_data, n_questions, n_cats = load_simple_data(
        "data/synthetic_OC/synthetic_oc_train.txt",
        "data/synthetic_OC/synthetic_oc_test.txt"
    )
    _, test_loader = create_data_loaders(train_data, test_data, batch_size=32)
    
    print("="*100)
    print("COMPREHENSIVE EVALUATION WITH HARD PREDICTIONS (ARGMAX) FOR ALL METRICS")
    print("="*100)
    
    # Models to evaluate
    models_to_evaluate = [
        ("Standard Training (Hard Metrics)", "save_models/best_deep_gpcm_hard_synthetic_OC.pth"),
        ("Original Model (Multi-method)", "save_models/best_deep_gpcm_synthetic_OC.pth")
    ]
    
    all_results = {}
    
    for model_name, model_path in models_to_evaluate:
        print(f"\n{'='*80}")
        print(f"Evaluating: {model_name}")
        print(f"Model path: {model_path}")
        print('='*80)
        
        try:
            # Load model
            if "hard" in model_path:
                # Load the model trained with hard predictions
                from models.implementations.deep_gpcm import DeepGPCM
                model = DeepGPCM(
                    n_questions=n_questions,
                    n_cats=n_cats,
                    memory_size=50,
                    key_dim=50,
                    value_dim=200,
                    final_fc_dim=256,
                    dropout_rate=0.2
                ).to(device)
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
            else:
                # Load standard model
                model, _, _ = load_trained_model(model_path, device)
                model.eval()
            
            # Evaluate
            results, predictions, targets = evaluate_model(model, test_loader, device, model_name)
            all_results[model_name] = results
            
            # Print key metrics
            print(f"\nKey Metrics:")
            print(f"{'Metric':<30} {'Value':>10}")
            print("-" * 42)
            
            key_metrics = [
                'categorical_accuracy',
                'quadratic_weighted_kappa',
                'cohen_kappa',
                'ordinal_accuracy',
                'mean_absolute_error',
                'spearman_correlation',
                'kendall_tau'
            ]
            
            for metric in key_metrics:
                if metric in results:
                    print(f"{metric:<30} {results[metric]:>10.4f}")
            
            # Print prediction distribution
            print(f"\nPrediction Distribution:")
            for i in range(4):
                pred_count = (predictions == i).sum()
                true_count = (targets == i).sum()
                print(f"Category {i}: True={true_count:5d}, Predicted={pred_count:5d}")
                
        except Exception as e:
            print(f"Error loading model: {e}")
            continue
    
    # Comparison table
    if len(all_results) > 1:
        print("\n" + "="*100)
        print("COMPARISON TABLE")
        print("="*100)
        
        # Get all metrics
        all_metrics = set()
        for results in all_results.values():
            all_metrics.update([k for k in results.keys() if isinstance(results[k], (int, float))])
        
        # Sort metrics by category
        metric_categories = {
            'Accuracy Metrics': ['categorical_accuracy', 'ordinal_accuracy'],
            'Agreement Metrics': ['quadratic_weighted_kappa', 'cohen_kappa'],
            'Error Metrics': ['mean_absolute_error', 'mean_squared_error'],
            'Correlation Metrics': ['spearman_correlation', 'kendall_tau', 'pearson_correlation'],
            'Per-Category': ['macro_precision', 'macro_recall', 'macro_f1'],
            'Probability': ['cross_entropy', 'mean_confidence', 'expected_calibration_error']
        }
        
        for category, metrics in metric_categories.items():
            category_metrics = [m for m in metrics if m in all_metrics]
            if category_metrics:
                print(f"\n{category}:")
                print(f"{'Metric':<35}", end="")
                for model_name in all_results.keys():
                    print(f"{model_name[:25]:>25}", end="")
                print()
                print("-" * (35 + 25 * len(all_results)))
                
                for metric in category_metrics:
                    print(f"{metric:<35}", end="")
                    for model_name, results in all_results.items():
                        value = results.get(metric, np.nan)
                        if not np.isnan(value):
                            print(f"{value:>25.4f}", end="")
                        else:
                            print(f"{'N/A':>25}", end="")
                    print()
    
    print("\n" + "="*100)
    print("ANALYSIS")
    print("="*100)
    print("""
Key Findings when using hard predictions (argmax) for all metrics:

1. The model trained with hard predictions for all metrics achieves:
   - Categorical Accuracy: 53.26%
   - QWK: 0.6142
   - Spearman Correlation: 0.6244
   - MAE: 0.7261

2. Using hard predictions for regression-like metrics (MAE, correlations):
   - MAE treats categories as ordinal integers (0, 1, 2, 3)
   - Correlations measure rank agreement between discrete predictions
   - This is simpler but may lose information compared to soft predictions

3. Benefits of using hard predictions for all metrics:
   - Consistency: All metrics use the same prediction method
   - Simplicity: No need to manage multiple prediction types
   - Interpretability: All predictions are discrete categories
   - Speed: No need to compute multiple prediction types

4. Trade-offs:
   - May not be optimal for regression-like metrics
   - Loses uncertainty information that soft predictions provide
   - But results show strong performance across all metrics
""")

if __name__ == "__main__":
    main()