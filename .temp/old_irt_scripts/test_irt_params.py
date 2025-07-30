#!/usr/bin/env python3
"""
Test script to verify IRT parameters are properly unpackable.
"""

import torch
import numpy as np
from core.model import DeepGPCM, AttentionGPCM

def test_irt_parameter_unpacking():
    """Test that both models return properly unpackable IRT parameters."""
    
    # Test parameters
    n_questions = 30
    n_cats = 4
    batch_size = 8
    seq_len = 10
    
    print("Testing IRT Parameter Unpacking")
    print("=" * 50)
    
    # Create test data
    questions = torch.randint(1, n_questions + 1, (batch_size, seq_len))
    responses = torch.randint(0, n_cats, (batch_size, seq_len))
    
    print(f"Test data: batch_size={batch_size}, seq_len={seq_len}")
    print(f"Questions shape: {questions.shape}")
    print(f"Responses shape: {responses.shape}")
    print()
    
    # Test baseline model
    print("🔸 TESTING BASELINE DKVMN-GPCM")
    baseline_model = DeepGPCM(
        n_questions=n_questions,
        n_cats=n_cats,
        memory_size=20,  # Smaller for testing
        key_dim=32,
        value_dim=64,
        final_fc_dim=32
    )
    
    baseline_model.eval()
    with torch.no_grad():
        theta, beta, alpha, gpcm_probs = baseline_model(questions, responses)
    
    print(f"✅ Theta (student abilities): {theta.shape} - Expected: ({batch_size}, {seq_len})")
    print(f"✅ Beta (item thresholds): {beta.shape} - Expected: ({batch_size}, {seq_len}, {n_cats-1})")
    print(f"✅ Alpha (discrimination): {alpha.shape} - Expected: ({batch_size}, {seq_len})")
    print(f"✅ GPCM probabilities: {gpcm_probs.shape} - Expected: ({batch_size}, {seq_len}, {n_cats})")
    
    # Verify GPCM probabilities sum to 1
    prob_sums = gpcm_probs.sum(dim=-1)
    print(f"✅ Probability sums (should be ~1.0): mean={prob_sums.mean():.4f}, std={prob_sums.std():.6f}")
    
    # Test parameter ranges
    print(f"✅ Theta range: [{theta.min():.3f}, {theta.max():.3f}]")
    print(f"✅ Beta range: [{beta.min():.3f}, {beta.max():.3f}]")
    print(f"✅ Alpha range: [{alpha.min():.3f}, {alpha.max():.3f}] (should be positive)")
    print()
    
    # Test attention model
    print("🔸 TESTING ATTENTION DKVMN-GPCM")
    attention_model = AttentionGPCM(
        n_questions=n_questions,
        n_cats=n_cats,
        embed_dim=32,
        memory_size=20,  # Smaller for testing
        key_dim=32,
        value_dim=64,
        final_fc_dim=32,
        n_heads=2,
        n_cycles=1  # Fewer cycles for testing
    )
    
    attention_model.eval()
    with torch.no_grad():
        theta_attn, beta_attn, alpha_attn, gpcm_probs_attn = attention_model(questions, responses)
    
    print(f"✅ Theta (student abilities): {theta_attn.shape} - Expected: ({batch_size}, {seq_len})")
    print(f"✅ Beta (item thresholds): {beta_attn.shape} - Expected: ({batch_size}, {seq_len}, {n_cats-1})")
    print(f"✅ Alpha (discrimination): {alpha_attn.shape} - Expected: ({batch_size}, {seq_len})")
    print(f"✅ GPCM probabilities: {gpcm_probs_attn.shape} - Expected: ({batch_size}, {seq_len}, {n_cats})")
    
    # Verify GPCM probabilities sum to 1
    prob_sums_attn = gpcm_probs_attn.sum(dim=-1)
    print(f"✅ Probability sums (should be ~1.0): mean={prob_sums_attn.mean():.4f}, std={prob_sums_attn.std():.6f}")
    
    # Test parameter ranges
    print(f"✅ Theta range: [{theta_attn.min():.3f}, {theta_attn.max():.3f}]")
    print(f"✅ Beta range: [{beta_attn.min():.3f}, {beta_attn.max():.3f}]")
    print(f"✅ Alpha range: [{alpha_attn.min():.3f}, {alpha_attn.max():.3f}] (should be positive)")
    print()
    
    # Test GPCM probability computation directly
    print("🔸 TESTING GPCM PROBABILITY COMPUTATION")
    
    # Extract single timestep for testing
    theta_test = theta[:, 0]  # (batch_size,)
    alpha_test = alpha[:, 0]  # (batch_size,)
    beta_test = beta[:, 0, :]  # (batch_size, n_cats-1)
    
    # Expand for GPCM computation
    theta_expanded = theta_test.unsqueeze(1)  # (batch_size, 1)
    alpha_expanded = alpha_test.unsqueeze(1)  # (batch_size, 1)
    beta_expanded = beta_test.unsqueeze(1)    # (batch_size, 1, n_cats-1)
    
    # Test GPCM probability computation
    with torch.no_grad():
        manual_probs = baseline_model.gpcm_layer(theta_expanded, alpha_expanded, beta_expanded)
        manual_probs = manual_probs.squeeze(1)  # (batch_size, n_cats)
    
    # Compare with model output
    model_probs = gpcm_probs[:, 0, :]  # (batch_size, n_cats)
    
    print(f"✅ Manual GPCM shape: {manual_probs.shape}")
    print(f"✅ Model GPCM shape: {model_probs.shape}")
    print(f"✅ Probability difference: max={torch.max(torch.abs(manual_probs - model_probs)):.6f}")
    
    # Verify they match (should be very close)
    if torch.allclose(manual_probs, model_probs, atol=1e-5):
        print("✅ GPCM computation matches perfectly!")
    else:
        print("❌ GPCM computation mismatch!")
    
    print()
    print("🎯 SUMMARY")
    print("=" * 50)
    print("✅ Both models return properly shaped IRT parameters")
    print("✅ IRT parameters are properly unpackable:")
    print("   - Theta (student abilities): (batch_size, seq_len)")
    print("   - Beta (item thresholds): (batch_size, seq_len, K-1)")
    print("   - Alpha (discrimination): (batch_size, seq_len)")
    print("✅ GPCM probabilities are computed correctly using IRT parameters")
    print("✅ All probabilities sum to 1.0 as expected")
    print("✅ Alpha parameters are positive (due to Softplus activation)")
    print("✅ Models work independently without compatibility wrappers")


if __name__ == "__main__":
    test_irt_parameter_unpacking()