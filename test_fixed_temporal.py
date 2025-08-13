#!/usr/bin/env python3
"""
Test script to validate the FixedTemporalAttentionGPCM improvements.

This script compares the fixed model against the original temporal model
and baseline attention model to validate the improvements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import time

# Import model components
from models.implementations.attention_gpcm import AttentionGPCM
from models.implementations.temporal_attention_gpcm import TemporalAttentionGPCM
from models.implementations.fixed_temporal_attention_gpcm import FixedTemporalAttentionGPCM
from models.factory import create_model
from utils.data_loading import load_dataset


def analyze_model_stability(model, questions, responses, targets, mask=None, num_epochs=5):
    """Analyze training stability over multiple epochs."""
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    losses = []
    grad_norms = []
    
    for epoch in range(num_epochs):
        model.zero_grad()
        
        # Forward pass
        outputs = model(questions, responses)
        if isinstance(outputs, tuple):
            logits = outputs[-1]  # Last output is probabilities
        else:
            logits = outputs
        
        # Convert to logits if needed
        if torch.all(logits >= 0) and torch.all(logits <= 1):
            logits = torch.log(logits + 1e-8)
        
        # Loss computation
        batch_size, seq_len = questions.shape
        logits_flat = logits[:, :seq_len, :].reshape(-1, logits.size(-1))
        targets_flat = targets[:, :seq_len].reshape(-1)
        
        # Apply mask if provided
        if mask is not None:
            mask_flat = mask[:, :seq_len].reshape(-1)
            logits_flat = logits_flat[mask_flat]
            targets_flat = targets_flat[mask_flat]
        
        loss = F.cross_entropy(logits_flat, targets_flat)
        
        # Backward pass
        loss.backward()
        
        # Calculate gradient norm
        total_grad_norm = 0
        for param in model.parameters():
            if param.grad is not None:
                total_grad_norm += param.grad.norm().item() ** 2
        total_grad_norm = total_grad_norm ** 0.5
        
        # Update
        optimizer.step()
        
        losses.append(loss.item())
        grad_norms.append(total_grad_norm)
    
    return losses, grad_norms


def compare_batch_size_sensitivity(model_class, n_questions, n_cats, batch_sizes=[8, 16, 32]):
    """Compare model performance across different batch sizes."""
    results = {}
    
    for batch_size in batch_sizes:
        print(f"  Testing batch size {batch_size}...")
        
        # Load data
        train_loader, _, _, _ = load_dataset("synthetic_OC", batch_size=batch_size)
        
        # Get a batch
        for batch in train_loader:
            if len(batch) == 3:
                questions, responses, mask = batch
            else:
                questions, responses = batch[:2]
                mask = None
            targets = responses
            break
        
        # Create model
        model = model_class(n_questions, n_cats)
        
        # Test stability
        losses, grad_norms = analyze_model_stability(model, questions, responses, targets, mask)
        
        # Calculate metrics
        loss_stability = np.std(losses)  # Lower is more stable
        grad_stability = np.std(grad_norms)  # Lower is more stable
        final_loss = losses[-1]
        
        results[batch_size] = {
            'final_loss': final_loss,
            'loss_stability': loss_stability,
            'grad_stability': grad_stability,
            'avg_grad_norm': np.mean(grad_norms)
        }
    
    return results


def main():
    print("=" * 70)
    print("FIXED TEMPORAL ATTENTION GPCM VALIDATION")
    print("=" * 70)
    
    # Load data
    print("Loading data...")
    train_loader, test_loader, n_questions, n_cats = load_dataset("synthetic_OC", batch_size=16)
    
    # Get a test batch
    for batch in train_loader:
        if len(batch) == 3:
            questions, responses, mask = batch
        else:
            questions, responses = batch[:2]
            mask = None
        targets = responses
        break
    
    print(f"Dataset: {n_questions} questions, {n_cats} categories")
    print(f"Batch shape: {questions.shape}")
    
    # Create models for comparison
    print("\nCreating models...")
    
    models = {
        "Baseline AttentionGPCM": create_model("attn_gpcm_linear", n_questions, n_cats),
        "Original TemporalGPCM": TemporalAttentionGPCM(n_questions, n_cats),
        "Fixed TemporalGPCM": create_model("fixed_temporal_attn_gpcm", n_questions, n_cats)
    }
    
    print("\n" + "=" * 70)
    print("SINGLE BATCH COMPARISON")
    print("=" * 70)
    
    # Single batch comparison
    results = {}
    for model_name, model in models.items():
        print(f"\n{model_name}:")
        
        # Analyze stability over 5 epochs
        start_time = time.time()
        losses, grad_norms = analyze_model_stability(model, questions, responses, targets, mask)
        elapsed_time = time.time() - start_time
        
        # Calculate metrics
        initial_loss = losses[0]
        final_loss = losses[-1]
        loss_improvement = initial_loss - final_loss
        loss_stability = np.std(losses)
        grad_stability = np.std(grad_norms)
        avg_grad_norm = np.mean(grad_norms)
        
        print(f"  Initial loss: {initial_loss:.4f}")
        print(f"  Final loss: {final_loss:.4f}")
        print(f"  Loss improvement: {loss_improvement:.4f}")
        print(f"  Loss stability (std): {loss_stability:.4f}")
        print(f"  Avg gradient norm: {avg_grad_norm:.4f}")
        print(f"  Gradient stability (std): {grad_stability:.4f}")
        print(f"  Training time (5 epochs): {elapsed_time:.2f}s")
        
        results[model_name] = {
            'initial_loss': initial_loss,
            'final_loss': final_loss,
            'loss_improvement': loss_improvement,
            'loss_stability': loss_stability,
            'avg_grad_norm': avg_grad_norm,
            'grad_stability': grad_stability,
            'training_time': elapsed_time
        }
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY COMPARISON")
    print("=" * 70)
    
    print(f"{'Model':<25} {'Final Loss':<12} {'Loss Impr':<12} {'Grad Norm':<12} {'Stability':<12}")
    print("-" * 70)
    
    for model_name, result in results.items():
        print(f"{model_name:<25} {result['final_loss']:<12.4f} {result['loss_improvement']:<12.4f} "
              f"{result['avg_grad_norm']:<12.4f} {result['grad_stability']:<12.4f}")
    
    # Analyze improvements
    print("\n" + "=" * 70)
    print("IMPROVEMENT ANALYSIS")
    print("=" * 70)
    
    baseline = results["Baseline AttentionGPCM"]
    original = results["Original TemporalGPCM"] 
    fixed = results["Fixed TemporalGPCM"]
    
    print("Fixed vs Original Temporal:")
    loss_diff = fixed['final_loss'] - original['final_loss']
    grad_diff = fixed['avg_grad_norm'] - original['avg_grad_norm']
    stability_diff = fixed['grad_stability'] - original['grad_stability']
    
    print(f"  Loss difference: {loss_diff:+.4f} ({'Better' if loss_diff < 0 else 'Worse'})")
    print(f"  Gradient norm difference: {grad_diff:+.4f} ({'More stable' if grad_diff < 0 else 'Less stable'})")
    print(f"  Gradient stability difference: {stability_diff:+.4f} ({'More stable' if stability_diff < 0 else 'Less stable'})")
    
    print("\nFixed vs Baseline:")
    loss_diff_base = fixed['final_loss'] - baseline['final_loss']
    grad_diff_base = fixed['avg_grad_norm'] - baseline['avg_grad_norm']
    stability_diff_base = fixed['grad_stability'] - baseline['grad_stability']
    
    print(f"  Loss difference: {loss_diff_base:+.4f} ({'Better' if loss_diff_base < 0 else 'Worse'})")
    print(f"  Gradient norm difference: {grad_diff_base:+.4f} ({'More stable' if grad_diff_base < 0 else 'Less stable'})")
    print(f"  Gradient stability difference: {stability_diff_base:+.4f} ({'More stable' if stability_diff_base < 0 else 'Less stable'})")
    
    # Batch size sensitivity test
    print("\n" + "=" * 70)
    print("BATCH SIZE SENSITIVITY TEST")
    print("=" * 70)
    
    batch_test_models = {
        "Original TemporalGPCM": TemporalAttentionGPCM,
        "Fixed TemporalGPCM": FixedTemporalAttentionGPCM
    }
    
    for model_name, model_class in batch_test_models.items():
        print(f"\n{model_name}:")
        batch_results = compare_batch_size_sensitivity(model_class, n_questions, n_cats)
        
        print(f"{'Batch Size':<12} {'Final Loss':<12} {'Loss Stab':<12} {'Grad Stab':<12}")
        print("-" * 50)
        
        for batch_size, result in batch_results.items():
            print(f"{batch_size:<12} {result['final_loss']:<12.4f} {result['loss_stability']:<12.4f} "
                  f"{result['grad_stability']:<12.4f}")
    
    # Key insights
    print("\n" + "=" * 70)
    print("KEY INSIGHTS")
    print("=" * 70)
    
    if fixed['grad_stability'] < original['grad_stability']:
        print("✅ Fixed model shows improved gradient stability")
    else:
        print("❌ Fixed model gradient stability needs more work")
        
    if fixed['final_loss'] <= original['final_loss']:
        print("✅ Fixed model maintains or improves final performance")
    else:
        print("❌ Fixed model performance degraded")
        
    if fixed['avg_grad_norm'] < original['avg_grad_norm']:
        print("✅ Fixed model has more controlled gradient magnitudes")
    else:
        print("❌ Fixed model gradient magnitudes still high")
    
    print(f"\nOverall: The fixed model {'SUCCESSFULLY' if (fixed['grad_stability'] < original['grad_stability'] and fixed['final_loss'] <= original['final_loss']) else 'PARTIALLY'} addresses the temporal model issues.")


if __name__ == "__main__":
    print("Starting Fixed Temporal Attention GPCM validation...")
    main()
    print("\nValidation complete!")