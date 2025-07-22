#!/usr/bin/env python3
"""
Phase 1 CORAL Integration Benchmarking

This script tests the CORAL integration with DKVMN architecture and compares
performance against the original GPCM approach.
"""

import os
import sys
import torch
import numpy as np
import json
from datetime import datetime

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model import DeepGpcmModel
from models.dkvmn_coral_integration import DeepGpcmCoralModel, create_coral_enhanced_model
from evaluation.metrics import GpcmMetrics
from utils.gpcm_utils import load_gpcm_data, create_gpcm_batch
from train import GpcmDataLoader


def load_model_and_data(dataset_name, device):
    """Load trained model and test data."""
    # Load dataset
    train_path = f"data/{dataset_name}/synthetic_oc_train.txt"
    test_path = f"data/{dataset_name}/synthetic_oc_test.txt"
    
    train_seqs, train_questions, train_responses, n_cats = load_gpcm_data(train_path)
    n_questions = max(max(seq) for seq in train_questions) + 1
    
    test_seqs, test_questions, test_responses, _ = load_gpcm_data(test_path, n_cats)
    
    # Find trained model
    model_dir = "save_models"
    model_files = [f for f in os.listdir(model_dir) if dataset_name.lower() in f.lower() and f.endswith(".pth")]
    if not model_files:
        raise FileNotFoundError(f"No model found for dataset {dataset_name}")
    
    model_path = os.path.join(model_dir, sorted(model_files)[-1])
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    if 'n_questions' in checkpoint and 'n_cats' in checkpoint:
        n_questions = checkpoint['n_questions'] 
        n_cats = checkpoint['n_cats']
    
    original_model = DeepGpcmModel(
        n_questions=n_questions,
        n_cats=n_cats,
        memory_size=50,
        key_dim=50,
        value_dim=200,
        final_fc_dim=50,
        embedding_strategy='ordered'
    ).to(device)
    
    # Load weights
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    original_model.load_state_dict(state_dict)
    original_model.eval()
    
    # Create test loader
    test_loader = GpcmDataLoader(test_questions, test_responses, batch_size=64, shuffle=False)
    
    return original_model, test_loader, n_cats


def extract_predictions(model, data_loader, device, n_cats, use_coral=True):
    """Extract predictions from model."""
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for q_batch, r_batch, mask_batch in data_loader:
            q_batch = q_batch.to(device)
            r_batch = r_batch.to(device)
            mask_batch = mask_batch.to(device)
            
            # Forward pass
            if hasattr(model, 'forward') and 'use_coral' in model.forward.__code__.co_varnames:
                # CORAL-enhanced model
                _, _, _, probs = model(q_batch, r_batch, use_coral=use_coral)
            else:
                # Original model
                _, _, _, probs = model(q_batch, r_batch)
            
            # Extract valid positions
            if mask_batch is not None:
                valid_probs = probs[mask_batch]
                valid_targets = r_batch[mask_batch]
            else:
                valid_probs = probs.view(-1, n_cats)
                valid_targets = r_batch.view(-1)
            
            all_probs.append(valid_probs.cpu())
            all_targets.append(valid_targets.cpu())
    
    return torch.cat(all_probs, dim=0), torch.cat(all_targets, dim=0)


def analyze_rank_consistency(predictions):
    """Analyze rank consistency of predictions."""
    cum_probs = torch.cumsum(predictions, dim=-1)
    violations = 0
    total = 0
    
    for k in range(cum_probs.shape[-1] - 1):
        batch_violations = (cum_probs[:, k] > cum_probs[:, k+1] + 1e-6).sum().item()
        violations += batch_violations
        total += cum_probs.shape[0]
    
    return violations, total


def benchmark_coral_integration(dataset_name="synthetic_OC"):
    """Main benchmarking function."""
    print(f"{'='*80}")
    print(f"CORAL INTEGRATION BENCHMARKING")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*80}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model and data
    print("\n1. Loading model and data...")
    try:
        original_model, test_loader, n_cats = load_model_and_data(dataset_name, device)
        print(f"✅ Model loaded successfully")
        print(f"   Architecture: DKVMN + GPCM")
        print(f"   Categories: {n_cats}")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return
    
    # Create CORAL-enhanced model
    print("\n2. Creating CORAL-enhanced model...")
    try:
        coral_model = create_coral_enhanced_model(original_model)
        coral_model = coral_model.to(device)
        print(f"✅ CORAL model created successfully")
        print(f"   Architecture: DKVMN + IRT + CORAL")
        print(f"   CORAL thresholds initialized: {coral_model.get_coral_thresholds().tolist()}")
    except Exception as e:
        print(f"❌ Error creating CORAL model: {e}")
        return
    
    # Extract predictions
    print("\n3. Extracting predictions...")
    
    # Original GPCM predictions
    print("   Extracting GPCM predictions...")
    gpcm_preds, targets = extract_predictions(original_model, test_loader, device, n_cats)
    print(f"   GPCM predictions: {gpcm_preds.shape}")
    
    # CORAL predictions (untrained)
    print("   Extracting CORAL predictions...")
    coral_preds, _ = extract_predictions(coral_model, test_loader, device, n_cats, use_coral=True)
    print(f"   CORAL predictions: {coral_preds.shape}")
    
    # Benchmark performance
    print(f"\n{'='*80}")
    print("PERFORMANCE COMPARISON: GPCM vs CORAL (UNTRAINED)")
    print(f"{'='*80}")
    
    # GPCM performance
    print("\n🔸 GPCM (Original) Performance:")
    gpcm_results = GpcmMetrics.benchmark_prediction_methods(
        gpcm_preds, targets, n_cats, methods=['argmax', 'cumulative']
    )
    
    for method in ['argmax', 'cumulative']:
        if method in gpcm_results:
            results = gpcm_results[method]
            print(f"   {method.title()} Method:")
            print(f"     Categorical Accuracy: {results['categorical_accuracy']:.3f}")
            print(f"     Ordinal Accuracy:     {results['ordinal_accuracy']:.3f}")
            print(f"     Prediction Consistency: {results['prediction_consistency']:.3f}")
            print(f"     MAE:                  {results['mean_absolute_error']:.3f}")
    
    # CORAL performance (untrained)
    print("\n🔸 CORAL (Untrained) Performance:")
    coral_results = GpcmMetrics.benchmark_prediction_methods(
        coral_preds, targets, n_cats, methods=['argmax', 'cumulative']
    )
    
    for method in ['argmax', 'cumulative']:
        if method in coral_results:
            results = coral_results[method]
            print(f"   {method.title()} Method:")
            print(f"     Categorical Accuracy: {results['categorical_accuracy']:.3f}")
            print(f"     Ordinal Accuracy:     {results['ordinal_accuracy']:.3f}")
            print(f"     Prediction Consistency: {results['prediction_consistency']:.3f}")
            print(f"     MAE:                  {results['mean_absolute_error']:.3f}")
    
    # Rank consistency analysis
    print(f"\n{'='*80}")
    print("RANK CONSISTENCY ANALYSIS")
    print(f"{'='*80}")
    
    gpcm_violations, total = analyze_rank_consistency(gpcm_preds)
    coral_violations, _ = analyze_rank_consistency(coral_preds)
    
    print(f"Rank Consistency Violations:")
    print(f"   GPCM:  {gpcm_violations:,}/{total:,} ({100*gpcm_violations/total:.2f}%)")
    print(f"   CORAL: {coral_violations:,}/{total:,} ({100*coral_violations/total:.2f}%)")
    
    if coral_violations < gpcm_violations:
        improvement = (gpcm_violations - coral_violations) / gpcm_violations * 100
        print(f"   ✅ CORAL improvement: -{improvement:.1f}% violations")
    else:
        degradation = (coral_violations - gpcm_violations) / gpcm_violations * 100 if gpcm_violations > 0 else 0
        print(f"   ⚠️  CORAL degradation: +{degradation:.1f}% violations")
    
    # Probability analysis
    print(f"\n{'='*80}")
    print("PROBABILITY CALIBRATION ANALYSIS")
    print(f"{'='*80}")
    
    def analyze_probabilities(probs, name):
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=-1).mean()
        max_prob_mean = probs.max(dim=-1)[0].mean()
        min_prob_mean = probs.min(dim=-1)[0].mean()
        
        print(f"{name}:")
        print(f"   Mean Entropy:        {entropy:.4f}")
        print(f"   Mean Max Probability: {max_prob_mean:.4f}")
        print(f"   Mean Min Probability: {min_prob_mean:.4f}")
        print(f"   Probability Range:   [{probs.min():.6f}, {probs.max():.6f}]")
    
    analyze_probabilities(gpcm_preds, "GPCM")
    analyze_probabilities(coral_preds, "CORAL (Untrained)")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {
        'experiment_info': {
            'timestamp': timestamp,
            'dataset': dataset_name,
            'description': 'CORAL integration benchmarking vs original GPCM'
        },
        'gpcm_results': gpcm_results,
        'coral_results': coral_results,
        'rank_consistency': {
            'gpcm_violations': gpcm_violations,
            'coral_violations': coral_violations,
            'total_samples': total
        },
        'coral_thresholds': coral_model.get_coral_thresholds().tolist()
    }
    
    os.makedirs("results/benchmarks", exist_ok=True)
    results_path = f"results/benchmarks/coral_integration_benchmark_{dataset_name}_{timestamp}.json"
    
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_path}")
    
    # Summary and recommendations
    print(f"\n{'='*80}")
    print("CORAL INTEGRATION SUMMARY")
    print(f"{'='*80}")
    
    print("✅ CORAL integration successful - model runs without errors")
    print("✅ Rank consistency analysis completed")
    print("✅ Probability calibration analysis completed")
    print("⚠️  CORAL layer is UNTRAINED - performance will improve after training")
    
    print(f"\n📋 NEXT STEPS:")
    print("   1. Train CORAL-enhanced model from scratch")
    print("   2. Compare trained CORAL vs trained GPCM performance")
    print("   3. Analyze educational interpretability preservation")
    print("   4. Implement distance-aware ordinal embeddings")


if __name__ == "__main__":
    benchmark_coral_integration()