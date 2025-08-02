#!/usr/bin/env python3
"""
Test QWK (Quadratic Weighted Kappa) with different prediction methods.
"""

import torch
import numpy as np
from sklearn.metrics import cohen_kappa_score
import sys
sys.path.append('/home/steph/dirt-new/deep-gpcm')

from utils.predictions import compute_unified_predictions, PredictionConfig
from utils.metrics import OrdinalMetrics

def test_qwk_with_different_methods():
    """Test QWK calculation with hard, soft (rounded), and threshold predictions."""
    
    # Load a model and get some predictions
    print("Loading model and data...")
    import argparse
    from evaluation_utils import load_model, load_data, create_data_loaders
    
    # Load model
    model_path = "save_models/best_deep_gpcm_synthetic_OC.pth"
    model_name = "deep_gpcm"
    dataset = "synthetic_OC"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create minimal args
    args = argparse.Namespace(
        dataset=dataset,
        batch_size=32,
        device=device
    )
    
    model = load_model(model_name, dataset, model_path, device)
    train_data, test_data = load_data(dataset)
    _, test_loader = create_data_loaders(train_data, test_data, args.batch_size)
    
    # Get predictions from first batch
    model.eval()
    with torch.no_grad():
        questions, responses, mask = next(iter(test_loader))
        questions = questions.to(device)
        responses = responses.to(device)
        mask = mask.to(device)
        
        # Get model output
        _, _, _, gpcm_probs = model(questions, responses)
        
        # Flatten for metrics
        probs_flat = gpcm_probs.view(-1, gpcm_probs.size(-1))
        responses_flat = responses.view(-1)
        mask_flat = mask.view(-1)
        
        # Filter valid positions
        valid_indices = mask_flat.bool()
        probs = probs_flat[valid_indices].cpu()
        targets = responses_flat[valid_indices].cpu()
    
    print(f"\nSample size: {len(targets)}")
    
    # Compute all prediction types
    config = PredictionConfig(use_gpu=False)
    predictions = compute_unified_predictions(probs, config=config)
    
    # Convert to numpy
    y_true = targets.numpy()
    hard_pred = predictions['hard'].numpy()
    soft_pred = predictions['soft'].numpy()
    threshold_pred = predictions['threshold'].numpy()
    
    # Calculate QWK for each method
    print("\n" + "="*60)
    print("QWK COMPARISON")
    print("="*60)
    
    # 1. Hard predictions (argmax)
    qwk_hard = cohen_kappa_score(y_true, hard_pred, weights='quadratic')
    print(f"\n1. Hard Predictions (argmax):")
    print(f"   Mean: {hard_pred.mean():.3f}, Std: {hard_pred.std():.3f}")
    print(f"   QWK: {qwk_hard:.4f}")
    
    # 2. Soft predictions (rounded)
    soft_rounded = np.round(soft_pred).astype(int)
    soft_rounded = np.clip(soft_rounded, 0, 3)  # Ensure valid range
    qwk_soft = cohen_kappa_score(y_true, soft_rounded, weights='quadratic')
    print(f"\n2. Soft Predictions (rounded expected value):")
    print(f"   Raw - Mean: {soft_pred.mean():.3f}, Std: {soft_pred.std():.3f}")
    print(f"   Rounded - Mean: {soft_rounded.mean():.3f}, Std: {soft_rounded.std():.3f}")
    print(f"   QWK: {qwk_soft:.4f}")
    
    # 3. Threshold predictions
    qwk_threshold = cohen_kappa_score(y_true, threshold_pred, weights='quadratic')
    print(f"\n3. Threshold Predictions (cumulative):")
    print(f"   Mean: {threshold_pred.mean():.3f}, Std: {threshold_pred.std():.3f}")
    print(f"   QWK: {qwk_threshold:.4f}")
    
    # Additional analysis
    print("\n" + "="*60)
    print("PREDICTION AGREEMENT")
    print("="*60)
    
    # Agreement between methods
    hard_soft_agree = (hard_pred == soft_rounded).mean()
    hard_threshold_agree = (hard_pred == threshold_pred).mean()
    soft_threshold_agree = (soft_rounded == threshold_pred).mean()
    
    print(f"Hard vs Soft agreement: {hard_soft_agree:.1%}")
    print(f"Hard vs Threshold agreement: {hard_threshold_agree:.1%}")
    print(f"Soft vs Threshold agreement: {soft_threshold_agree:.1%}")
    
    # Distribution analysis
    print("\n" + "="*60)
    print("PREDICTION DISTRIBUTIONS")
    print("="*60)
    
    for method_name, preds in [("Hard", hard_pred), ("Soft (rounded)", soft_rounded), ("Threshold", threshold_pred)]:
        unique, counts = np.unique(preds, return_counts=True)
        print(f"\n{method_name}:")
        for val, count in zip(unique, counts):
            print(f"  Category {val}: {count} ({count/len(preds)*100:.1f}%)")

if __name__ == "__main__":
    test_qwk_with_different_methods()