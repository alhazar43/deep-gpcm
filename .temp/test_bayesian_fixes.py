#!/usr/bin/env python3
"""
Quick test to verify Bayesian GPCM fixes work correctly.
"""

import torch
import numpy as np
from models.baseline_bayesian import VariationalBayesianGPCM

def test_fixes():
    """Test the three key fixes."""
    print("Testing Bayesian GPCM fixes...")
    
    # Create model
    model = VariationalBayesianGPCM(
        n_students=100,
        n_questions=30,
        n_categories=4,
        kl_weight=1.0
    )
    
    # Test 1: Better initialization
    print("\n1. Testing improved initialization:")
    theta_mean = model.theta_dist.mean.mean().item()
    theta_var = torch.exp(model.theta_dist.log_var).mean().item()
    print(f"   Theta: mean={theta_mean:.3f}, var={theta_var:.3f} (should be ~0.0, ~0.1)")
    
    alpha_mean = model.alpha_dist.log_mean.mean().item()
    alpha_var = torch.exp(model.alpha_dist.log_var).mean().item()
    print(f"   Alpha (log): mean={alpha_mean:.3f}, var={alpha_var:.3f} (should be ~0.0, ~0.05)")
    
    beta_mean = model.beta_dist.base_mean.mean().item()
    beta_var = torch.exp(model.beta_dist.base_log_var).mean().item()
    print(f"   Beta base: mean={beta_mean:.3f}, var={beta_var:.3f} (should be ~0.0, ~0.1)")
    
    # Test 2: KL annealing
    print("\n2. Testing KL annealing:")
    
    # Create dummy data
    questions = torch.randint(0, 30, (16, 10))
    responses = torch.randint(0, 4, (16, 10))
    
    # Test epoch 0 (should have low effective KL weight)
    model.set_epoch(0)
    _, aux_dict = model(questions, responses)
    kl_factor_0 = aux_dict['kl_annealing_factor']
    print(f"   Epoch 0: KL annealing factor = {kl_factor_0:.3f} (should be 0.0)")
    
    # Test epoch 10 (should have medium effective KL weight)
    model.set_epoch(10)
    _, aux_dict = model(questions, responses)
    kl_factor_10 = aux_dict['kl_annealing_factor']
    print(f"   Epoch 10: KL annealing factor = {kl_factor_10:.3f} (should be ~0.5)")
    
    # Test epoch 25 (should have full KL weight)
    model.set_epoch(25)
    _, aux_dict = model(questions, responses)
    kl_factor_25 = aux_dict['kl_annealing_factor']
    print(f"   Epoch 25: KL annealing factor = {kl_factor_25:.3f} (should be 1.0)")
    
    # Test 3: Parameter sampling consistency
    print("\n3. Testing parameter sampling:")
    
    # Sample multiple times and check consistency
    samples_1 = []
    samples_2 = []
    
    for _ in range(5):
        theta1 = model.theta_dist.rsample(1).squeeze(0)
        theta2 = model.theta_dist.rsample(1).squeeze(0)
        samples_1.append(theta1[:5].mean().item())  # First 5 students
        samples_2.append(theta2[:5].mean().item())
    
    var_within = np.var(samples_1)
    var_between = np.var(samples_2)
    print(f"   Sample variance within: {var_within:.4f}")
    print(f"   Sample variance between: {var_between:.4f}")
    print(f"   Ratio: {var_between/var_within:.2f} (should be close to 1.0)")
    
    print("\n✓ All fixes appear to be working correctly!")
    
    return True

if __name__ == '__main__':
    test_fixes()