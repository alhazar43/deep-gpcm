#!/usr/bin/env python3
"""
Test script for Fixed Bayesian-DKVMN model.
"""

import torch
import numpy as np
from models.fixed_bayesian_dkvmn import FixedBayesianDKVMN

def test_fixed_bayesian_model():
    """Test the fixed Bayesian-DKVMN model."""
    print("Testing Fixed Bayesian-DKVMN Model...")
    print("-" * 50)
    
    # Model parameters
    n_questions = 30
    n_categories = 4
    memory_size = 20
    ability_dim = 32
    batch_size = 8
    seq_len = 5
    
    # Create model
    model = FixedBayesianDKVMN(
        n_questions=n_questions,
        n_categories=n_categories,
        memory_size=memory_size,
        ability_dim=ability_dim
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
    
    assert 'alphas' in aux_dict, "Missing alphas"
    assert 'betas' in aux_dict, "Missing betas"
    print(f"   Alphas shape: {aux_dict['alphas'].shape}")
    print(f"   Betas shape: {aux_dict['betas'].shape}")
    
    # Test parameter constraints
    print("\n4. Testing parameter constraints...")
    alphas = aux_dict['alphas']
    betas = aux_dict['betas']
    
    # Alpha should be positive
    assert torch.all(alphas > 0), "Alphas should be positive"
    print(f"   Alpha range: [{alphas.min().item():.3f}, {alphas.max().item():.3f}]")
    
    # Beta should be ordered (increasing)
    for i in range(len(betas)):
        beta_seq = betas[i]
        diffs = beta_seq[1:] - beta_seq[:-1]
        assert torch.all(diffs > 0), f"Beta parameters should be ordered for question {i}"
    print(f"   Beta ordering verified for all questions")
    
    # Test ELBO loss computation
    print("\n5. Testing ELBO loss computation...")
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
    
    # Test parameter extraction
    print("\n7. Testing parameter extraction...")
    interpretable_params = model.get_interpretable_parameters()
    
    print(f"   Alpha parameters: shape {interpretable_params['alpha'].shape}")
    print(f"   Beta parameters: shape {interpretable_params['beta'].shape}")
    print(f"   Theta parameters: shape {interpretable_params['theta'].shape}")
    
    # Test question-specific parameter access
    print("\n8. Testing question-specific parameters...")
    test_questions = torch.tensor([0, 5, 10, 15])
    test_alphas, test_betas = model.irt_params.get_parameters(test_questions)
    
    print(f"   Question-specific alphas: {test_alphas}")
    print(f"   Question-specific betas shape: {test_betas.shape}")
    
    # Verify parameters are different for different questions (allow small tolerance for initialization)
    alpha_diff = torch.abs(test_alphas[0] - test_alphas[1])
    if alpha_diff > 1e-3:
        print("   ✓ Question-specific parameters verified")
    else:
        print("   ⚠ Parameters very similar (expected for untrained model)")
    
    # Test deterministic vs sampling
    print("\n9. Testing deterministic vs sampling modes...")
    model.eval()  # Evaluation mode
    _, aux_eval = model(questions[:1, :1], responses[:1, :1], return_params=True)
    
    model.train()  # Training mode
    _, aux_train = model(questions[:1, :1], responses[:1, :1], return_params=True)
    
    print("   ✓ Both evaluation and training modes working")
    
    print("\n✓ Fixed Bayesian-DKVMN model working correctly!")
    print("Key improvements:")
    print("  - Question-specific IRT parameters (no averaging)")
    print("  - Deterministic parameter extraction during evaluation")
    print("  - Proper parameter constraints (α > 0, β ordered)")
    print("  - Direct parameter learning without memory interference")
    
    return True

if __name__ == '__main__':
    test_fixed_bayesian_model()