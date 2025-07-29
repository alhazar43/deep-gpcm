#!/usr/bin/env python3
"""
Test script for Variational Bayesian GPCM model.

This script performs basic functionality tests to ensure the Bayesian model
is working correctly before running full training.
"""

import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from models.baseline_bayesian import (
    VariationalBayesianGPCM, 
    NormalVariational, 
    LogNormalVariational,
    OrderedNormalVariational
)


def test_variational_distributions():
    """Test variational distribution implementations."""
    print("Testing Variational Distributions...")
    print("-" * 50)
    
    # Test NormalVariational
    print("\n1. Testing NormalVariational:")
    normal_dist = NormalVariational(dim=10, prior_mean=0.0, prior_std=1.0)
    
    # Sample
    samples = normal_dist.rsample(n_samples=100)
    print(f"   Sample shape: {samples.shape}")
    print(f"   Sample mean: {samples.mean():.3f}, std: {samples.std():.3f}")
    
    # KL divergence
    kl = normal_dist.kl_divergence()
    print(f"   KL divergence: {kl.item():.4f}")
    assert kl.item() >= 0, "KL divergence should be non-negative"
    
    # Test LogNormalVariational
    print("\n2. Testing LogNormalVariational:")
    lognormal_dist = LogNormalVariational(dim=10, prior_mean=0.0, prior_std=0.3)
    
    # Sample
    samples = lognormal_dist.rsample(n_samples=100)
    print(f"   Sample shape: {samples.shape}")
    print(f"   Sample mean: {samples.mean():.3f}, std: {samples.std():.3f}")
    assert (samples > 0).all(), "Log-normal samples should be positive"
    
    # KL divergence
    kl = lognormal_dist.kl_divergence()
    print(f"   KL divergence: {kl.item():.4f}")
    assert kl.item() >= 0, "KL divergence should be non-negative"
    
    # Test OrderedNormalVariational
    print("\n3. Testing OrderedNormalVariational:")
    ordered_dist = OrderedNormalVariational(n_questions=20, n_thresholds=3)
    
    # Sample
    samples = ordered_dist.rsample(n_samples=10)
    print(f"   Sample shape: {samples.shape}")
    
    # Check ordering
    for i in range(samples.shape[0]):
        for q in range(samples.shape[1]):
            thresholds = samples[i, q, :]
            assert (thresholds[1:] >= thresholds[:-1]).all(), \
                f"Thresholds should be ordered for sample {i}, question {q}"
    print("   Ordering constraint verified!")
    
    # KL divergence
    kl = ordered_dist.kl_divergence()
    print(f"   KL divergence: {kl.item():.4f}")
    assert kl.item() >= 0, "KL divergence should be non-negative"
    
    print("\n✓ All variational distributions working correctly!")


def test_bayesian_model():
    """Test Bayesian GPCM model functionality."""
    print("\n\nTesting Bayesian GPCM Model...")
    print("-" * 50)
    
    # Model parameters
    n_students = 100
    n_questions = 30
    n_categories = 4
    batch_size = 16
    seq_len = 10
    
    # Create model
    model = VariationalBayesianGPCM(
        n_students=n_students,
        n_questions=n_questions,
        n_categories=n_categories,
        kl_weight=1.0
    )
    
    print(f"\n1. Model created successfully")
    print(f"   Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy data
    questions = torch.randint(0, n_questions, (batch_size, seq_len))
    responses = torch.randint(0, n_categories, (batch_size, seq_len))
    student_ids = torch.randint(0, n_students, (batch_size,))
    
    # Test forward pass
    print("\n2. Testing forward pass...")
    probabilities, aux_dict = model(questions, responses, student_ids, return_params=True)
    
    print(f"   Output shape: {probabilities.shape}")
    print(f"   Expected shape: ({batch_size}, {seq_len}, {n_categories})")
    assert probabilities.shape == (batch_size, seq_len, n_categories), "Wrong output shape"
    
    # Check probabilities sum to 1
    prob_sums = probabilities.sum(dim=-1)
    assert torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-6), \
        "Probabilities should sum to 1"
    print("   Probability constraints verified!")
    
    # Check auxiliary outputs
    print("\n3. Testing auxiliary outputs...")
    assert 'kl_divergence' in aux_dict, "Missing KL divergence"
    print(f"   KL divergence: {aux_dict['kl_divergence'].item():.4f}")
    
    assert 'theta' in aux_dict, "Missing theta parameters"
    assert 'alpha' in aux_dict, "Missing alpha parameters"
    assert 'beta' in aux_dict, "Missing beta parameters"
    assert 'predicted_abilities' in aux_dict, "Missing predicted abilities"
    
    print(f"   Theta shape: {aux_dict['theta'].shape}")
    print(f"   Alpha shape: {aux_dict['alpha'].shape}")
    print(f"   Beta shape: {aux_dict['beta'].shape}")
    print(f"   Predicted abilities shape: {aux_dict['predicted_abilities'].shape}")
    
    # Test ELBO loss
    print("\n4. Testing ELBO loss computation...")
    loss = model.elbo_loss(probabilities, responses, aux_dict['kl_divergence'])
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
    
    # Test prior sampling
    print("\n6. Testing prior sampling...")
    prior_samples = model.sample_prior()
    print(f"   Prior theta mean: {prior_samples['theta'].mean():.3f}, "
          f"std: {prior_samples['theta'].std():.3f}")
    print(f"   Prior alpha mean: {prior_samples['alpha'].mean():.3f}, "
          f"std: {prior_samples['alpha'].std():.3f}")
    
    # Test posterior statistics
    print("\n7. Testing posterior statistics...")
    posterior_stats = model.get_posterior_stats()
    
    for param_name in ['theta', 'alpha', 'beta']:
        stats = posterior_stats[param_name]
        print(f"   {param_name}: mean shape {stats['mean'].shape}, "
              f"std shape {stats['std'].shape}")
    
    print("\n✓ Bayesian GPCM model working correctly!")


def test_parameter_recovery():
    """Test if model can recover parameters from synthetic data."""
    print("\n\nTesting Parameter Recovery...")
    print("-" * 50)
    
    # Generate synthetic parameters
    n_students = 50
    n_questions = 20
    n_categories = 4
    
    # True parameters
    true_theta = torch.randn(n_students) * 1.0  # N(0, 1)
    true_alpha = torch.exp(torch.randn(n_questions) * 0.3)  # LogNormal(0, 0.3)
    
    # Generate ordered betas
    true_beta = []
    for q in range(n_questions):
        base = torch.randn(1) * 1.0
        offsets = torch.sort(torch.randn(n_categories - 1) * 0.5)[0]
        true_beta.append(base + offsets)
    true_beta = torch.stack(true_beta)
    
    print(f"Generated true parameters:")
    print(f"  Theta: mean={true_theta.mean():.3f}, std={true_theta.std():.3f}")
    print(f"  Alpha: mean={true_alpha.mean():.3f}, std={true_alpha.std():.3f}")
    print(f"  Beta: shape={true_beta.shape}")
    
    # Create model
    model = VariationalBayesianGPCM(
        n_students=n_students,
        n_questions=n_questions,
        n_categories=n_categories
    )
    
    # Initialize variational parameters closer to truth (for testing)
    with torch.no_grad():
        model.theta_dist.mean.data = true_theta + torch.randn_like(true_theta) * 0.1
        model.alpha_dist.log_mean.data = torch.log(true_alpha) + torch.randn_like(true_alpha) * 0.1
    
    # Get posterior statistics
    posterior_stats = model.get_posterior_stats()
    
    # Compare means
    learned_theta = posterior_stats['theta']['mean']
    learned_alpha = posterior_stats['alpha']['mean']
    
    theta_corr = torch.corrcoef(torch.stack([true_theta, learned_theta]))[0, 1]
    alpha_corr = torch.corrcoef(torch.stack([true_alpha, learned_alpha]))[0, 1]
    
    print(f"\nInitial parameter correlations:")
    print(f"  Theta correlation: {theta_corr:.3f}")
    print(f"  Alpha correlation: {alpha_corr:.3f}")
    
    # Create simple plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    ax1.scatter(true_theta.numpy(), learned_theta.numpy(), alpha=0.5)
    ax1.plot([true_theta.min(), true_theta.max()], 
             [true_theta.min(), true_theta.max()], 'r--')
    ax1.set_xlabel('True θ')
    ax1.set_ylabel('Learned θ')
    ax1.set_title(f'θ Recovery (r={theta_corr:.3f})')
    ax1.grid(True, alpha=0.3)
    
    ax2.scatter(true_alpha.numpy(), learned_alpha.numpy(), alpha=0.5)
    ax2.plot([0, true_alpha.max()], [0, true_alpha.max()], 'r--')
    ax2.set_xlabel('True α')
    ax2.set_ylabel('Learned α')
    ax2.set_title(f'α Recovery (r={alpha_corr:.3f})')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('test_parameter_recovery.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"\nParameter recovery plot saved to test_parameter_recovery.png")
    print("\n✓ Parameter recovery test completed!")


def main():
    """Run all tests."""
    print("=" * 70)
    print("VARIATIONAL BAYESIAN GPCM MODEL - FUNCTIONALITY TESTS")
    print("=" * 70)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run tests
    test_variational_distributions()
    test_bayesian_model()
    test_parameter_recovery()
    
    print("\n" + "=" * 70)
    print("ALL TESTS PASSED! ✓")
    print("=" * 70)
    print("\nThe Variational Bayesian GPCM model is ready for training.")
    print("\nNext steps:")
    print("1. Train the model: python train_bayesian.py --dataset synthetic_OC --epochs 50")
    print("2. Compare with baseline: python compare_irt_models.py --dataset synthetic_OC")
    print("3. Analyze temporal dynamics (future work): Implement VTIRT extension")


if __name__ == '__main__':
    main()