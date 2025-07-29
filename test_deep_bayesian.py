#!/usr/bin/env python3
"""
Test script for Deep Bayesian-DKVMN GPCM model.
"""

import torch
import numpy as np
from models.deep_bayesian_dkvmn import DeepBayesianDKVMN

def test_deep_bayesian_model():
    """Test the deep Bayesian-DKVMN model."""
    print("Testing Deep Bayesian-DKVMN Model...")
    print("-" * 50)
    
    # Model parameters
    n_questions = 30
    n_categories = 4
    memory_size = 10
    batch_size = 8
    seq_len = 5
    
    # Create model
    model = DeepBayesianDKVMN(
        n_questions=n_questions,
        n_categories=n_categories,
        memory_size=memory_size,
        key_dim=20,
        value_dim=50
    )
    
    print(f"1. Model created successfully")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy data
    questions = torch.randint(0, n_questions, (batch_size, seq_len))
    responses = torch.randint(0, n_categories, (batch_size, seq_len))
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    model.set_epoch(5)  # Test KL annealing
    probabilities, aux_dict = model(questions, responses, return_params=True)
    
    print(f"   Output shape: {probabilities.shape}")
    print(f"   Expected shape: ({batch_size}, {seq_len}, {n_categories})")
    assert probabilities.shape == (batch_size, seq_len, n_categories), "Wrong output shape"
    
    # Check probabilities sum to 1
    prob_sums = probabilities.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6), \
        "Probabilities should sum to 1"
    print("   Probability constraints verified!")
    
    # Test auxiliary outputs
    print("\n3. Testing auxiliary outputs...")
    assert 'kl_divergence' in aux_dict, "Missing KL divergence"
    print(f"   KL divergence: {aux_dict['kl_divergence'].item():.4f}")
    
    assert 'memory_alphas' in aux_dict, "Missing memory alphas"
    assert 'memory_betas' in aux_dict, "Missing memory betas"
    print(f"   Memory alphas shape: {len(aux_dict['memory_alphas'])}")
    print(f"   Memory betas shape: {len(aux_dict['memory_betas'])}")
    
    # Test ELBO loss computation
    print("\n4. Testing ELBO loss computation...")
    kl_div = aux_dict['kl_divergence']
    loss = model.elbo_loss(probabilities, responses, kl_div)
    print(f"   ELBO loss: {loss.item():.4f}")
    assert loss.item() > 0, "Loss should be positive"
    
    # Test backward pass
    print("\n5. Testing backward pass...")
    loss.backward()
    
    # Check gradients
    grad_norms = []
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norms.append(param.grad.norm().item())
    
    print(f"   Average gradient norm: {np.mean(grad_norms):.6f}")
    print(f"   Max gradient norm: {np.max(grad_norms):.6f}")
    
    # Test parameter extraction
    print("\n6. Testing parameter extraction...")
    interpretable_params = model.get_interpretable_parameters()
    
    print(f"   Alpha parameters: shape {interpretable_params['alpha'].shape}")
    print(f"   Beta parameters: shape {interpretable_params['beta'].shape}")
    print(f"   Theta parameters: shape {interpretable_params['theta'].shape}")
    
    # Test memory operations
    print("\n7. Testing memory operations...")
    # Test individual components
    memory_key = model.memory_keys[0]
    alpha, beta = memory_key.sample_parameters()
    print(f"   Sampled alpha: {alpha.item():.3f}")
    print(f"   Sampled beta shape: {beta.shape}")
    
    memory_value = model.memory_values[0]
    belief_dist = memory_value.get_belief_distribution()
    print(f"   Belief mean: {belief_dist.mean[:3]}")  # First 3 dimensions
    print(f"   Belief std: {belief_dist.stddev[:3]}")   # First 3 dimensions
    
    print("\n✓ Deep Bayesian-DKVMN model working correctly!")
    return True

if __name__ == '__main__':
    test_deep_bayesian_model()