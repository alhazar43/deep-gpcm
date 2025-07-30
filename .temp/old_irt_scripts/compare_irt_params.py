#!/usr/bin/env python3
"""
Compare learned IRT parameters with true generated parameters.
"""

import os
import sys
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def load_true_parameters(dataset_path):
    """Load true IRT parameters from synthetic dataset."""
    true_params_path = dataset_path / 'true_irt_parameters.json'
    
    if not true_params_path.exists():
        print(f"Warning: No true parameters found at {true_params_path}")
        return None
    
    with open(true_params_path, 'r') as f:
        true_params = json.load(f)
    
    return true_params


def extract_item_parameters(extracted_params):
    """Extract average item parameters from temporal parameters."""
    questions = extracted_params['question_ids']
    discriminations = extracted_params['item_discriminations']
    thresholds = extracted_params['item_thresholds']
    masks = extracted_params.get('masks', np.ones_like(questions))
    
    n_questions = int(np.max(questions)) + 1
    n_cats = thresholds.shape[-1] + 1
    
    # Calculate average parameters for each question
    item_alphas = np.zeros(n_questions)
    item_betas = np.zeros((n_questions, n_cats - 1))
    item_counts = np.zeros(n_questions)
    
    for i in range(questions.shape[0]):
        for j in range(questions.shape[1]):
            if masks[i, j] > 0 and questions[i, j] > 0:  # Valid position and not padding
                q_id = questions[i, j]
                item_alphas[q_id] += discriminations[i, j]
                item_betas[q_id] += thresholds[i, j]
                item_counts[q_id] += 1
    
    # Average
    for q_id in range(n_questions):
        if item_counts[q_id] > 0:
            item_alphas[q_id] /= item_counts[q_id]
            item_betas[q_id] /= item_counts[q_id]
    
    return item_alphas, item_betas, item_counts


def normalize_parameters(alphas, betas, prior_alpha_mean=1.0, prior_alpha_std=0.5,
                        prior_beta_mean=0.0, prior_beta_std=1.0):
    """Normalize parameters to standard IRT scale."""
    # Normalize discriminations (log-normal prior)
    alphas_norm = np.exp((np.log(alphas + 1e-6) - np.mean(np.log(alphas + 1e-6))) / 
                         np.std(np.log(alphas + 1e-6)) * prior_alpha_std + np.log(prior_alpha_mean))
    
    # Normalize thresholds (normal prior)
    betas_norm = np.zeros_like(betas)
    for k in range(betas.shape[1]):
        betas_norm[:, k] = (betas[:, k] - np.mean(betas[:, k])) / np.std(betas[:, k]) * prior_beta_std + prior_beta_mean
    
    return alphas_norm, betas_norm


def plot_parameter_comparison(true_params, learned_alphas, learned_betas, item_counts):
    """Plot comparison of true vs learned parameters."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Only plot items that were seen in the data
    valid_items = item_counts > 0
    valid_indices = np.where(valid_items)[0]
    
    if true_params:
        true_alphas = np.array(true_params['question_params']['discrimination']['alpha'])
        true_betas = np.array(true_params['question_params']['difficulties']['beta'])
        
        # Normalize parameters
        true_alphas_norm, true_betas_norm = normalize_parameters(true_alphas, true_betas)
        learned_alphas_norm, learned_betas_norm = normalize_parameters(
            learned_alphas[valid_items], learned_betas[valid_items]
        )
        
        # Calculate correlations first
        alpha_corr = np.corrcoef(true_alphas_norm[valid_indices], learned_alphas_norm)[0, 1]
        
        # Plot discrimination comparison
        ax = axes[0, 0]
        ax.scatter(true_alphas_norm[valid_indices], learned_alphas_norm, alpha=0.6)
        ax.plot([0, 3], [0, 3], 'r--', label='Perfect recovery')
        ax.set_xlabel('True α')
        ax.set_ylabel('Learned α')
        ax.set_title(f'Discrimination Parameters\n(r = {alpha_corr:.3f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Calculate beta correlations
        beta_corrs = [np.corrcoef(true_betas_norm[valid_indices, k], learned_betas_norm[:, k])[0, 1] 
                      for k in range(3)]
        
        # Plot threshold comparisons
        for k in range(3):
            ax = axes[(k+1)//2, (k+1)%2]
            ax.scatter(true_betas_norm[valid_indices, k], learned_betas_norm[:, k], alpha=0.6)
            ax.plot([-3, 3], [-3, 3], 'r--', label='Perfect recovery')
            ax.set_xlabel(f'True β_{k}')
            ax.set_ylabel(f'Learned β_{k}')
            ax.set_title(f'Threshold {k} Parameters\n(r = {beta_corrs[k]:.3f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Add overall summary
        fig.suptitle(f'IRT Parameter Recovery\n' +
                     f'Overall Correlations - α: {alpha_corr:.3f}, ' +
                     f'β₀: {beta_corrs[0]:.3f}, β₁: {beta_corrs[1]:.3f}, β₂: {beta_corrs[2]:.3f}',
                     fontsize=14, fontweight='bold')
    else:
        # Just plot distributions if no true parameters
        ax = axes[0, 0]
        ax.hist(learned_alphas[valid_items], bins=30, alpha=0.7)
        ax.set_xlabel('Learned α')
        ax.set_title('Discrimination Distribution')
        ax.grid(True, alpha=0.3)
        
        for k in range(3):
            ax = axes[(k+1)//2, (k+1)%2]
            ax.hist(learned_betas[valid_items, k], bins=30, alpha=0.7)
            ax.set_xlabel(f'Learned β_{k}')
            ax.set_title(f'Threshold {k} Distribution')
            ax.grid(True, alpha=0.3)
            
        fig.suptitle('Learned IRT Parameter Distributions')
    
    plt.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description='Compare learned vs true IRT parameters')
    parser.add_argument('--extracted_params', required=True, help='Path to extracted parameters JSON')
    parser.add_argument('--dataset', default='synthetic_OC', help='Dataset name')
    parser.add_argument('--output', default='results/plots/irt_comparison.png', help='Output plot path')
    
    args = parser.parse_args()
    
    # Load extracted parameters
    print(f"Loading extracted parameters from: {args.extracted_params}")
    with open(args.extracted_params, 'r') as f:
        extracted_params = json.load(f)
    
    # Convert lists back to numpy arrays
    for key in ['student_abilities', 'item_discriminations', 'item_thresholds', 
                'question_ids', 'responses', 'sequence_lengths', 'masks']:
        if key in extracted_params:
            extracted_params[key] = np.array(extracted_params[key])
    
    # Extract average item parameters
    learned_alphas, learned_betas, item_counts = extract_item_parameters(extracted_params)
    
    # Load true parameters
    dataset_path = Path('data') / args.dataset
    true_params = load_true_parameters(dataset_path)
    
    # Create comparison plot
    fig = plot_parameter_comparison(true_params, learned_alphas, learned_betas, item_counts)
    
    # Save plot
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    fig.savefig(args.output, dpi=150, bbox_inches='tight')
    print(f"\nComparison plot saved to: {args.output}")
    
    # Print summary statistics
    print("\nParameter Summary:")
    print(f"  Items seen in data: {np.sum(item_counts > 0)} / {len(item_counts)}")
    print(f"  Avg discrimination: {learned_alphas[item_counts > 0].mean():.3f}")
    print(f"  Avg thresholds: {learned_betas[item_counts > 0].mean(axis=0)}")


if __name__ == "__main__":
    main()