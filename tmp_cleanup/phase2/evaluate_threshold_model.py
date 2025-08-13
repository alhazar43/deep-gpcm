#!/usr/bin/env python3
"""
Evaluate the model trained with threshold predictions.
"""

import torch
import numpy as np
import sys
sys.path.append('/home/steph/dirt-new/deep-gpcm')

from evaluate import load_simple_data, create_data_loaders, load_trained_model
from utils.predictions import compute_unified_predictions, PredictionConfig
from utils.metrics import OrdinalMetrics

def main():
    # Load both models for comparison
    standard_model_path = "save_models/best_deep_gpcm_synthetic_OC.pth"
    threshold_model_path = "save_models/best_deep_gpcm_threshold_synthetic_OC.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_data, test_data, n_questions, n_cats = load_simple_data(
        "data/synthetic_OC/synthetic_oc_train.txt",
        "data/synthetic_OC/synthetic_oc_test.txt"
    )
    _, test_loader = create_data_loaders(train_data, test_data, batch_size=32)
    
    print("="*80)
    print("EVALUATION OF THRESHOLD-TRAINED MODEL")
    print("="*80)
    
    # Load threshold-trained model
    print("\n1. Loading threshold-trained model...")
    from models.implementations.deep_gpcm import DeepGPCM
    
    # Create model with same config as training
    threshold_model = DeepGPCM(
        n_questions=n_questions,
        n_cats=n_cats,
        memory_size=50,
        key_dim=50,
        value_dim=200,
        final_fc_dim=256,
        dropout_rate=0.2
    ).to(device)
    
    # Load state dict
    threshold_model.load_state_dict(torch.load(threshold_model_path, map_location=device))
    threshold_model.eval()
    
    # Collect predictions
    all_predictions = []
    all_targets = []
    all_masks = []
    
    with torch.no_grad():
        for questions, responses, mask in test_loader:
            questions = questions.to(device)
            responses = responses.to(device)
            mask = mask.to(device)
            
            _, _, _, gpcm_probs = threshold_model(questions, responses)
            
            probs_flat = gpcm_probs.view(-1, gpcm_probs.size(-1))
            responses_flat = responses.view(-1)
            mask_flat = mask.view(-1)
            
            all_predictions.append(probs_flat.cpu())
            all_targets.append(responses_flat.cpu())
            all_masks.append(mask_flat.cpu())
    
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    valid_indices = all_masks.bool()
    all_predictions = all_predictions[valid_indices]
    all_targets = all_targets[valid_indices].numpy()
    
    # Analyze probability distributions
    print("\nProbability Distribution Analysis:")
    print("-" * 50)
    
    # Average probabilities per category
    avg_probs = all_predictions.mean(dim=0)
    std_probs = all_predictions.std(dim=0)
    
    print(f"Average probabilities: {avg_probs.numpy()}")
    print(f"Std dev probabilities: {std_probs.numpy()}")
    
    # Max probability statistics
    max_probs = all_predictions.max(dim=1)[0]
    print(f"\nMax probability stats: mean={max_probs.mean():.3f}, std={max_probs.std():.3f}")
    
    # Entropy
    epsilon = 1e-15
    probs_safe = torch.clamp(all_predictions, epsilon, 1 - epsilon)
    entropy = -(probs_safe * torch.log(probs_safe)).sum(dim=1)
    print(f"Entropy stats: mean={entropy.mean():.3f}, std={entropy.std():.3f}")
    
    # Get different predictions
    config = PredictionConfig(use_gpu=False, thresholds=[0.5, 0.5, 0.5])
    unified_preds = compute_unified_predictions(all_predictions, config=config)
    
    hard_preds = unified_preds['hard'].numpy()
    soft_preds = unified_preds['soft'].numpy()
    threshold_preds = unified_preds['threshold'].numpy()
    
    # Distribution of predictions
    print("\n" + "="*80)
    print("PREDICTION DISTRIBUTIONS")
    print("="*80)
    
    print("\nCategory distributions:")
    print(f"{'Category':<10} {'True':<10} {'Hard':<10} {'Threshold':<10}")
    print("-" * 40)
    for i in range(4):
        true_count = (all_targets == i).sum()
        hard_count = (hard_preds == i).sum()
        thresh_count = (threshold_preds == i).sum()
        print(f"{i:<10} {true_count:<10} {hard_count:<10} {thresh_count:<10}")
    
    # Compute metrics
    metrics = OrdinalMetrics(n_cats=4)
    
    print("\n" + "="*80)
    print("METRICS COMPARISON")
    print("="*80)
    
    print("\nUsing Hard Predictions:")
    print(f"  Categorical Accuracy: {metrics.categorical_accuracy(all_targets, hard_preds):.4f}")
    print(f"  QWK: {metrics.quadratic_weighted_kappa(all_targets, hard_preds):.4f}")
    print(f"  Spearman: {metrics.spearman_correlation(all_targets, hard_preds):.4f}")
    
    print("\nUsing Threshold Predictions (0.5):")
    print(f"  Categorical Accuracy: {metrics.categorical_accuracy(all_targets, threshold_preds):.4f}")
    print(f"  QWK: {metrics.quadratic_weighted_kappa(all_targets, threshold_preds):.4f}")
    print(f"  Spearman: {metrics.spearman_correlation(all_targets, threshold_preds):.4f}")
    
    print("\nUsing Soft Predictions:")
    print(f"  MAE: {metrics.mean_absolute_error(all_targets, soft_preds):.4f}")
    print(f"  Spearman: {metrics.spearman_correlation(all_targets, soft_preds):.4f}")
    
    # Compare with standard model
    print("\n" + "="*80)
    print("COMPARISON WITH STANDARD MODEL")
    print("="*80)
    
    print("\n2. Loading standard model for comparison...")
    standard_model, _, _ = load_trained_model(standard_model_path, device)
    standard_model.eval()
    
    # Get standard model predictions
    std_predictions = []
    with torch.no_grad():
        for questions, responses, mask in test_loader:
            questions = questions.to(device)
            responses = responses.to(device)
            mask = mask.to(device)
            
            _, _, _, gpcm_probs = standard_model(questions, responses)
            
            probs_flat = gpcm_probs.view(-1, gpcm_probs.size(-1))
            mask_flat = mask.view(-1)
            
            std_predictions.append(probs_flat[mask_flat.bool()].cpu())
    
    std_predictions = torch.cat(std_predictions, dim=0)
    std_unified = compute_unified_predictions(std_predictions, config=config)
    std_hard = std_unified['hard'].numpy()
    
    print("\nStandard Model Metrics:")
    print(f"  Categorical Accuracy: {metrics.categorical_accuracy(all_targets, std_hard):.4f}")
    print(f"  QWK: {metrics.quadratic_weighted_kappa(all_targets, std_hard):.4f}")
    
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    print("""
The threshold-trained model appears to have collapsed to predicting mostly category 1.
This is because training with threshold predictions (median) creates a feedback loop:
1. Initial predictions are uncertain → threshold predictions favor middle categories
2. Model learns to predict middle categories → probabilities become more concentrated
3. This reinforces predicting category 1 (since it's the median for flat distributions)

The standard training approach (using true labels) performs much better.
""")

if __name__ == "__main__":
    main()