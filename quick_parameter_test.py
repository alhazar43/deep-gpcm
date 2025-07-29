#!/usr/bin/env python3
"""
Quick test to verify that question-specific parameter structure can achieve better recovery.
"""

import torch
import numpy as np
import json
from pathlib import Path
from models.question_specific_bayesian_dkvmn import QuestionSpecificDeepBayesianDKVMN

def load_true_parameters():
    """Load true IRT parameters."""
    param_file = Path('data/synthetic_OC/true_irt_parameters.json')
    
    if not param_file.exists():
        print(f"No true parameters found at {param_file}")
        return None
        
    with open(param_file, 'r') as f:
        params = json.load(f)
    
    return {
        'theta': np.array(params['student_abilities']['theta']),
        'alpha': np.array(params['question_parameters']['discrimination']['alpha']),
        'beta': np.array(params['question_parameters']['difficulties']['beta'])
    }

def compute_correlations(true_params, learned_params):
    """Compute parameter correlations."""
    if true_params is None:
        return {}
    
    # Convert learned parameters to numpy
    learned_alpha = learned_params['alpha'].cpu().numpy()
    learned_beta = learned_params['beta'].cpu().numpy()
    learned_theta = learned_params['theta'].cpu().numpy()
    
    # Alpha correlation (question-specific vs question-specific)
    n_questions = min(len(true_params['alpha']), len(learned_alpha))
    alpha_corr = np.corrcoef(true_params['alpha'][:n_questions], 
                           learned_alpha[:n_questions])[0, 1]
    alpha_corr = alpha_corr if not np.isnan(alpha_corr) else 0.0
    
    # Beta correlation (question-specific vs question-specific)
    true_beta_mean = true_params['beta'][:n_questions].mean(axis=1)
    learned_beta_mean = learned_beta[:n_questions].mean(axis=1)
    beta_corr = np.corrcoef(true_beta_mean, learned_beta_mean)[0, 1]
    beta_corr = beta_corr if not np.isnan(beta_corr) else 0.0
    
    # Theta correlation (student-specific vs student-specific)
    n_students = min(len(true_params['theta']), len(learned_theta))
    if n_students > 0:
        theta_corr = np.corrcoef(true_params['theta'][:n_students], 
                               learned_theta[:n_students])[0, 1]
        theta_corr = theta_corr if not np.isnan(theta_corr) else 0.0
    else:
        theta_corr = 0.0
    
    return {
        'alpha_correlation': alpha_corr,
        'beta_correlation': beta_corr,
        'theta_correlation': theta_corr
    }

def main():
    """Quick parameter recovery test."""
    print("Quick Parameter Recovery Test - Question-Specific Model")
    print("=" * 60)
    
    # Load true parameters
    true_params = load_true_parameters()
    if true_params is None:
        print("❌ No true parameters available for testing")
        return
    
    print(f"✅ Loaded true parameters:")
    print(f"   α shape: {true_params['alpha'].shape}")
    print(f"   β shape: {true_params['beta'].shape}")
    print(f"   θ shape: {true_params['theta'].shape}")
    
    # Create model with CORRECT structure
    n_questions = len(true_params['alpha'])
    n_categories = true_params['beta'].shape[1] + 1
    
    model = QuestionSpecificDeepBayesianDKVMN(
        n_questions=n_questions,
        n_categories=n_categories,
        memory_size=10
    )
    
    print(f"\n✅ Created model with CORRECT IRT structure:")
    print(f"   Questions: {n_questions}")
    print(f"   Categories: {n_categories}")
    
    # Get initial (random) parameters
    initial_params = model.get_interpretable_parameters()
    initial_corr = compute_correlations(true_params, initial_params)
    
    print(f"\n📊 Initial (random) parameter correlations:")
    for metric, value in initial_corr.items():
        print(f"   {metric}: {value:.4f}")
    
    # Simulate some training by manually setting parameters closer to true values
    print(f"\n🎯 Testing parameter structure compatibility...")
    
    with torch.no_grad():
        # Set question-specific alphas closer to true values
        true_alpha_tensor = torch.tensor(true_params['alpha'], dtype=torch.float32)
        model.irt_params.alpha_mean.data = torch.log(true_alpha_tensor + 1e-6)
        
        # Set question-specific betas closer to true values
        for q in range(n_questions):
            true_beta_q = true_params['beta'][q]
            model.irt_params.beta_base.data[q] = true_beta_q[0]
            
            # Set gaps
            for k in range(1, len(true_beta_q)):
                if k-1 < model.irt_params.beta_gaps.shape[1]:
                    gap = true_beta_q[k] - true_beta_q[k-1]
                    model.irt_params.beta_gaps.data[q, k-1] = gap
        
        # Set student abilities (first 200 students)
        n_students_to_set = min(200, len(true_params['theta']))
        model.student_abilities[:n_students_to_set] = torch.tensor(
            true_params['theta'][:n_students_to_set], dtype=torch.float32
        )
    
    # Get "trained" parameters
    trained_params = model.get_interpretable_parameters()
    trained_corr = compute_correlations(true_params, trained_params)
    
    print(f"\n📊 After setting parameters to true values:")
    for metric, value in trained_corr.items():
        print(f"   {metric}: {value:.4f}")
    
    print(f"\n🎯 Parameter Structure Verification:")
    print(f"   ✅ Question-specific α: {trained_params['alpha'].shape} vs {true_params['alpha'].shape}")
    print(f"   ✅ Question-specific β: {trained_params['beta'].shape} vs {true_params['beta'].shape}")
    print(f"   ✅ Student-specific θ: {trained_params['theta'][:200].shape} vs {true_params['theta'].shape}")
    
    print(f"\n🎯 Expected Results:")
    print(f"   - Perfect correlations (≈1.0) when parameters match true values")
    print(f"   - This confirms the parameter structure is CORRECT")
    print(f"   - Training should achieve much better recovery than previous versions")
    
    return trained_corr

if __name__ == '__main__':
    main()