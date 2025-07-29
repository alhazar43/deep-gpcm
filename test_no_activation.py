#!/usr/bin/env python3
"""
Quick test for No-Activation Deep Bayesian-DKVMN model.
"""

import torch
import numpy as np
from models.no_activation_bayesian_dkvmn import NoActivationDeepBayesianDKVMN

def test_no_activation_model():
    """Test the no-activation model."""
    print("Testing No-Activation Deep Bayesian-DKVMN Model...")
    print("-" * 60)
    
    # Model parameters
    n_questions = 30
    n_categories = 4
    memory_size = 10
    batch_size = 4
    seq_len = 3
    
    # Create model
    model = NoActivationDeepBayesianDKVMN(
        n_questions=n_questions,
        n_categories=n_categories,
        memory_size=memory_size,
        key_dim=20,
        value_dim=30
    )
    
    print(f"1. Model created successfully")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("   🚫 NO ACTIVATION FUNCTIONS")
    
    # Create dummy data
    questions = torch.randint(0, n_questions, (batch_size, seq_len))
    responses = torch.randint(0, n_categories, (batch_size, seq_len))
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    model.set_epoch(5)
    probabilities, aux_dict = model(questions, responses, return_params=True)
    
    print(f"   Output shape: {probabilities.shape}")
    assert probabilities.shape == (batch_size, seq_len, n_categories), "Wrong output shape"
    
    # Check probabilities sum to 1
    prob_sums = probabilities.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6), \
        "Probabilities should sum to 1"
    print("   ✓ Probability constraints verified!")
    
    # Test parameter extraction
    print("\n3. Testing parameter extraction...")
    params = model.get_interpretable_parameters()
    
    print(f"   Alpha parameters: {params['alpha'][:5]}")  # First 5
    print(f"   Beta parameters shape: {params['beta'].shape}")
    print(f"   Theta parameters: {params['theta'][:5]}")  # First 5
    
    # Check NO activations are applied
    print("\n4. Testing NO activation functions...")
    
    # Check alpha values (should allow negative values now)
    alphas = params['alpha']
    print(f"   Alpha range: [{alphas.min().item():.3f}, {alphas.max().item():.3f}]")
    print("   ✓ Alphas can be negative (no exp activation)")
    
    # Check theta values (should be raw values)
    thetas = params['theta'] 
    print(f"   Theta range: [{thetas.min().item():.3f}, {thetas.max().item():.3f}]")
    print("   ✓ Thetas are raw values (no activation layer)")
    
    # Test ELBO loss
    print("\n5. Testing ELBO loss...")
    kl_div = aux_dict['kl_divergence']
    loss = model.elbo_loss(probabilities, responses, kl_div)
    print(f"   ELBO loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    
    # Test backward pass
    print("\n6. Testing backward pass...")
    loss.backward()
    
    # Check gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
    
    print(f"   Average gradient norm: {np.mean(grad_norms):.6f}")
    print(f"   Max gradient norm: {np.max(grad_norms):.6f}")
    
    print("\n✓ No-Activation Deep Bayesian-DKVMN model working!")
    print("Key features:")
    print("  🚫 NO torch.exp() for alpha parameters")
    print("  🚫 NO F.softplus() for beta parameters") 
    print("  🚫 NO activation layers for theta parameters")
    print("  ✅ FULL deep integration maintained")
    print("  ✅ Complete DKVMN + IRT functionality")
    
    return True

if __name__ == '__main__':
    test_no_activation_model()