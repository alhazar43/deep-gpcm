#!/usr/bin/env python3
"""
Analyze the distribution of predictions to understand why threshold predictions differ from hard predictions.
"""

import torch
import numpy as np
import sys
sys.path.append('/home/steph/dirt-new/deep-gpcm')

from evaluate import load_simple_data, create_data_loaders, load_trained_model
from utils.predictions import compute_unified_predictions, PredictionConfig

def analyze_probability_distribution(probs, name):
    """Analyze a set of probability distributions."""
    print(f"\n{name} Analysis:")
    print("-" * 50)
    
    # Max probabilities
    max_probs = probs.max(dim=1)[0]
    print(f"Max probability stats: mean={max_probs.mean():.3f}, std={max_probs.std():.3f}, min={max_probs.min():.3f}, max={max_probs.max():.3f}")
    
    # Entropy
    epsilon = 1e-15
    probs_safe = torch.clamp(probs, epsilon, 1 - epsilon)
    entropy = -(probs_safe * torch.log(probs_safe)).sum(dim=1)
    print(f"Entropy stats: mean={entropy.mean():.3f}, std={entropy.std():.3f}, min={entropy.min():.3f}, max={entropy.max():.3f}")
    
    # Category probabilities
    for i in range(probs.shape[1]):
        print(f"P(Y={i}): mean={probs[:, i].mean():.3f}, std={probs[:, i].std():.3f}")

def main():
    # Load model and data
    model_path = "save_models/best_deep_gpcm_synthetic_OC.pth"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model, _, _ = load_trained_model(model_path, device)
    model.eval()
    
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
            
            _, _, _, gpcm_probs = model(questions, responses)
            
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
    
    print("="*80)
    print("PREDICTION DISTRIBUTION ANALYSIS")
    print("="*80)
    
    # Analyze overall probability distribution
    analyze_probability_distribution(all_predictions, "Overall Probability Distribution")
    
    # Get different predictions
    hard_preds = all_predictions.argmax(dim=-1)
    config = PredictionConfig(use_gpu=False, thresholds=[0.5, 0.5, 0.5])
    unified_preds = compute_unified_predictions(all_predictions, config=config)
    threshold_preds = unified_preds['threshold']
    soft_preds = unified_preds['soft']
    
    # Compare predictions
    print("\n" + "="*80)
    print("PREDICTION COMPARISON")
    print("="*80)
    
    # Distribution of predictions
    print("\nPrediction Distribution:")
    for i in range(4):
        hard_count = (hard_preds == i).sum().item()
        thresh_count = (threshold_preds == i).sum().item()
        true_count = (all_targets == i).sum()
        print(f"Category {i}: True={true_count:5d}, Hard={hard_count:5d}, Threshold={thresh_count:5d}")
    
    # Where do they differ?
    diff_mask = hard_preds != threshold_preds
    print(f"\nDifferences: {diff_mask.sum().item()} out of {len(hard_preds)} ({diff_mask.float().mean():.2%})")
    
    if diff_mask.sum() > 0:
        # Analyze cases where they differ
        diff_probs = all_predictions[diff_mask]
        print("\nWhen predictions differ:")
        analyze_probability_distribution(diff_probs, "Probability Distribution (Differences Only)")
        
        # Show some examples
        print("\nExample cases where Hard ≠ Threshold:")
        print("-" * 80)
        print("Probs[0]  Probs[1]  Probs[2]  Probs[3]  | Hard  Thresh  True  | CumProb>0  CumProb>1  CumProb>2")
        print("-" * 80)
        
        n_examples = min(20, diff_mask.sum().item())
        diff_indices = torch.where(diff_mask)[0][:n_examples]
        
        for idx in diff_indices:
            probs = all_predictions[idx]
            hard = hard_preds[idx].item()
            thresh = threshold_preds[idx].item()
            true = all_targets[idx]
            
            # Calculate cumulative probabilities P(Y > k)
            cum_probs_gt = []
            for k in range(3):  # k = 0, 1, 2
                p_gt_k = probs[k+1:].sum().item()
                cum_probs_gt.append(p_gt_k)
            
            print(f"{probs[0]:.3f}     {probs[1]:.3f}     {probs[2]:.3f}     {probs[3]:.3f}     | "
                  f"{hard:^5d} {thresh:^7d} {true:^5d} | "
                  f"{cum_probs_gt[0]:.3f}      {cum_probs_gt[1]:.3f}      {cum_probs_gt[2]:.3f}")
    
    # Analyze threshold behavior
    print("\n" + "="*80)
    print("THRESHOLD PREDICTION BEHAVIOR")
    print("="*80)
    print("\nWith threshold=0.5, we predict the smallest k where P(Y > k-1) < 0.5")
    print("This is equivalent to finding the median of the distribution.")
    
    # Show cumulative probability statistics
    print("\nCumulative Probability Statistics P(Y > k):")
    for k in range(3):
        cum_probs = []
        for i in range(len(all_predictions)):
            p_gt_k = all_predictions[i, k+1:].sum().item()
            cum_probs.append(p_gt_k)
        cum_probs = np.array(cum_probs)
        print(f"P(Y > {k}): mean={cum_probs.mean():.3f}, std={cum_probs.std():.3f}, "
              f"median={np.median(cum_probs):.3f}, % < 0.5: {(cum_probs < 0.5).mean():.2%}")

if __name__ == "__main__":
    main()