#!/usr/bin/env python3
"""
Compare IRT parameters between baseline, Bayesian, and ground truth models.

This script provides comprehensive comparison of IRT parameters across different models,
visualizing how well each model captures the true parameter distributions.
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from scipy import stats
from typing import Dict, List, Tuple, Optional

from models.baseline import BaselineGPCM
from models.baseline_bayesian import VariationalBayesianGPCM
from models.akvmn_gpcm import AKVMNGPCM
from utils.gpcm_utils import load_gpcm_data, GpcmDataset
from torch.utils.data import DataLoader
from plot_irt import IRTParameterExtractor


def load_ground_truth_params(data_path: Path) -> Optional[Dict[str, np.ndarray]]:
    """Load ground truth IRT parameters from synthetic data."""
    param_file = data_path / 'true_irt_parameters.json'
    
    if not param_file.exists():
        return None
    
    with open(param_file, 'r') as f:
        params = json.load(f)
    
    return {
        'theta': np.array(params['student_abilities']['theta']),
        'alpha': np.array(params['question_parameters']['discrimination']['alpha']),
        'beta': np.array(params['question_parameters']['difficulties']['beta']),
        'metadata': params.get('model_info', {})
    }


def extract_baseline_params(model_path: Path, data_loader: DataLoader, 
                          n_questions: int, n_categories: int) -> Dict[str, np.ndarray]:
    """Extract IRT parameters from baseline model."""
    # Load model
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Detect model type
    if 'attention_weights' in checkpoint['model_state_dict']:
        model = AKVMNGPCM(n_questions=n_questions, n_cats=n_categories)
        model_type = 'akvmn'
    else:
        model = BaselineGPCM(n_questions=n_questions, n_cats=n_categories)
        model_type = 'baseline'
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Extract parameters using existing extractor
    extractor = IRTParameterExtractor(model_path)
    extractor.model = model
    extractor.model_type = model_type
    params = extractor.extract_from_data(data_loader)
    
    return {
        'theta': params['student_abilities'],
        'alpha': params['discrimination_params'],
        'beta': params['item_thresholds'],
        'model_type': model_type
    }


def extract_bayesian_params(model_path: Path) -> Dict[str, np.ndarray]:
    """Extract posterior mean IRT parameters from Bayesian model."""
    checkpoint = torch.load(model_path, map_location='cpu')
    
    # Get posterior statistics
    posterior_stats = checkpoint['posterior_stats']
    
    return {
        'theta': posterior_stats['theta']['mean'].numpy(),
        'alpha': posterior_stats['alpha']['mean'].numpy(),
        'beta': posterior_stats['beta']['mean'].numpy(),
        'theta_std': posterior_stats['theta']['std'].numpy(),
        'alpha_std': posterior_stats['alpha']['std'].numpy(),
        'beta_std': posterior_stats['beta']['std'].numpy(),
        'model_type': 'bayesian'
    }


def compute_distribution_metrics(params1: np.ndarray, params2: np.ndarray, 
                               param_name: str) -> Dict[str, float]:
    """Compute distribution comparison metrics between two parameter sets."""
    metrics = {}
    
    # Basic statistics
    metrics[f'{param_name}_mean_diff'] = abs(params1.mean() - params2.mean())
    metrics[f'{param_name}_std_diff'] = abs(params1.std() - params2.std())
    
    # Correlation (if same length)
    if len(params1) == len(params2):
        correlation = np.corrcoef(params1.flatten(), params2.flatten())[0, 1]
        metrics[f'{param_name}_correlation'] = correlation
        
        # MSE after standardization
        p1_std = (params1 - params1.mean()) / params1.std()
        p2_std = (params2 - params2.mean()) / params2.std()
        metrics[f'{param_name}_standardized_mse'] = np.mean((p1_std - p2_std) ** 2)
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_pval = stats.ks_2samp(params1.flatten(), params2.flatten())
    metrics[f'{param_name}_ks_statistic'] = ks_stat
    metrics[f'{param_name}_ks_pvalue'] = ks_pval
    
    # Wasserstein distance
    wasserstein = stats.wasserstein_distance(params1.flatten(), params2.flatten())
    metrics[f'{param_name}_wasserstein_distance'] = wasserstein
    
    return metrics


def plot_comprehensive_comparison(models_params: Dict[str, Dict[str, np.ndarray]], 
                                save_dir: Path):
    """Create comprehensive comparison plots for all models."""
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup figure
    n_models = len(models_params)
    fig = plt.figure(figsize=(20, 5 * n_models))
    
    # Color palette
    colors = plt.cm.Set1(np.linspace(0, 1, n_models))
    model_names = list(models_params.keys())
    
    # Create subplots grid
    gs = fig.add_gridspec(n_models, 6, hspace=0.3, wspace=0.3)
    
    # For each model, create comparison plots
    for i, (model_name, params) in enumerate(models_params.items()):
        # Theta distribution
        ax = fig.add_subplot(gs[i, 0:2])
        ax.hist(params['theta'], bins=30, alpha=0.7, color=colors[i], 
                density=True, label=model_name)
        
        # Add theoretical distribution for ground truth
        if model_name == 'ground_truth':
            x = np.linspace(-3, 3, 100)
            ax.plot(x, stats.norm.pdf(x, 0, 1), 'k--', alpha=0.5, 
                   label='Theoretical N(0,1)')
        
        ax.set_xlabel('θ (Student Ability)')
        ax.set_ylabel('Density')
        ax.set_title(f'{model_name} - Student Abilities')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        ax.text(0.05, 0.95, f'μ={params["theta"].mean():.3f}\nσ={params["theta"].std():.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Alpha distribution
        ax = fig.add_subplot(gs[i, 2:4])
        ax.hist(params['alpha'], bins=30, alpha=0.7, color=colors[i], 
                density=True, label=model_name)
        
        # Add theoretical distribution for ground truth
        if model_name == 'ground_truth':
            x = np.linspace(0, 3, 100)
            ax.plot(x, stats.lognorm.pdf(x, s=0.3, scale=np.exp(0)), 'k--', 
                   alpha=0.5, label='Theoretical LogN(0,0.3)')
        
        ax.set_xlabel('α (Discrimination)')
        ax.set_ylabel('Density')
        ax.set_title(f'{model_name} - Discrimination Parameters')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        ax.text(0.05, 0.95, f'μ={params["alpha"].mean():.3f}\nσ={params["alpha"].std():.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Beta distribution (mean thresholds)
        ax = fig.add_subplot(gs[i, 4:6])
        beta_mean = params['beta'].mean(axis=1)
        ax.hist(beta_mean, bins=30, alpha=0.7, color=colors[i], 
                density=True, label=model_name)
        
        ax.set_xlabel('β̄ (Mean Threshold)')
        ax.set_ylabel('Density')
        ax.set_title(f'{model_name} - Mean Threshold Parameters')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        ax.text(0.05, 0.95, f'μ={beta_mean.mean():.3f}\nσ={beta_mean.std():.3f}',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.suptitle('IRT Parameter Distributions Across Models', fontsize=16)
    plt.tight_layout()
    plt.savefig(save_dir / 'all_models_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Create pairwise scatter plots
    if 'ground_truth' in models_params:
        true_params = models_params['ground_truth']
        
        for model_name, params in models_params.items():
            if model_name == 'ground_truth':
                continue
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Theta comparison
            ax = axes[0]
            if len(true_params['theta']) == len(params['theta']):
                ax.scatter(true_params['theta'], params['theta'], alpha=0.5, s=20)
                ax.plot([true_params['theta'].min(), true_params['theta'].max()],
                       [true_params['theta'].min(), true_params['theta'].max()], 
                       'r--', label='y=x')
                corr = np.corrcoef(true_params['theta'], params['theta'])[0, 1]
                ax.set_title(f'θ Comparison (r={corr:.3f})')
            else:
                ax.text(0.5, 0.5, 'Different number of students\nCannot create scatter plot',
                       transform=ax.transAxes, ha='center', va='center')
                ax.set_title('θ Comparison')
            
            ax.set_xlabel('True θ')
            ax.set_ylabel(f'{model_name} θ')
            ax.grid(True, alpha=0.3)
            
            # Alpha comparison
            ax = axes[1]
            ax.scatter(true_params['alpha'], params['alpha'], alpha=0.5, s=20)
            ax.plot([0, true_params['alpha'].max()], [0, true_params['alpha'].max()], 
                   'r--', label='y=x')
            corr = np.corrcoef(true_params['alpha'], params['alpha'])[0, 1]
            ax.set_xlabel('True α')
            ax.set_ylabel(f'{model_name} α')
            ax.set_title(f'α Comparison (r={corr:.3f})')
            ax.grid(True, alpha=0.3)
            
            # Beta comparison (mean)
            ax = axes[2]
            true_beta_mean = true_params['beta'].mean(axis=1)
            param_beta_mean = params['beta'].mean(axis=1)
            ax.scatter(true_beta_mean, param_beta_mean, alpha=0.5, s=20)
            ax.plot([true_beta_mean.min(), true_beta_mean.max()],
                   [true_beta_mean.min(), true_beta_mean.max()], 'r--', label='y=x')
            corr = np.corrcoef(true_beta_mean, param_beta_mean)[0, 1]
            ax.set_xlabel('True β̄')
            ax.set_ylabel(f'{model_name} β̄')
            ax.set_title(f'β̄ Comparison (r={corr:.3f})')
            ax.grid(True, alpha=0.3)
            
            plt.suptitle(f'{model_name} vs Ground Truth IRT Parameters', fontsize=14)
            plt.tight_layout()
            plt.savefig(save_dir / f'{model_name}_vs_truth_scatter.png', 
                       dpi=150, bbox_inches='tight')
            plt.close()
    
    # Create uncertainty plots for Bayesian model
    if 'bayesian' in models_params and 'theta_std' in models_params['bayesian']:
        bayesian_params = models_params['bayesian']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Theta uncertainty
        ax = axes[0]
        ax.hist(bayesian_params['theta_std'], bins=30, alpha=0.7, color='purple')
        ax.set_xlabel('Posterior STD')
        ax.set_ylabel('Count')
        ax.set_title(f'θ Uncertainty (mean STD: {bayesian_params["theta_std"].mean():.3f})')
        ax.grid(True, alpha=0.3)
        
        # Alpha uncertainty
        ax = axes[1]
        ax.hist(bayesian_params['alpha_std'], bins=30, alpha=0.7, color='purple')
        ax.set_xlabel('Posterior STD')
        ax.set_ylabel('Count')
        ax.set_title(f'α Uncertainty (mean STD: {bayesian_params["alpha_std"].mean():.3f})')
        ax.grid(True, alpha=0.3)
        
        # Beta uncertainty
        ax = axes[2]
        ax.hist(bayesian_params['beta_std'].flatten(), bins=30, alpha=0.7, color='purple')
        ax.set_xlabel('Posterior STD')
        ax.set_ylabel('Count')
        ax.set_title(f'β Uncertainty (mean STD: {bayesian_params["beta_std"].mean():.3f})')
        ax.grid(True, alpha=0.3)
        
        plt.suptitle('Bayesian Model - Posterior Uncertainty', fontsize=14)
        plt.tight_layout()
        plt.savefig(save_dir / 'bayesian_uncertainty.png', dpi=150, bbox_inches='tight')
        plt.close()


def create_summary_table(metrics_dict: Dict[str, Dict[str, float]], save_path: Path):
    """Create a summary table of all comparison metrics."""
    import pandas as pd
    
    # Convert to DataFrame
    df = pd.DataFrame(metrics_dict).T
    
    # Round values
    df = df.round(4)
    
    # Save as CSV
    df.to_csv(save_path / 'comparison_metrics_summary.csv')
    
    # Create formatted text summary
    with open(save_path / 'comparison_metrics_summary.txt', 'w') as f:
        f.write("IRT Parameter Recovery Comparison Summary\n")
        f.write("=" * 80 + "\n\n")
        
        for comparison, metrics in metrics_dict.items():
            f.write(f"{comparison}:\n")
            f.write("-" * 40 + "\n")
            
            # Group by parameter type
            for param in ['theta', 'alpha', 'beta']:
                param_metrics = {k: v for k, v in metrics.items() if k.startswith(param)}
                if param_metrics:
                    f.write(f"\n  {param.upper()} Parameters:\n")
                    for metric, value in param_metrics.items():
                        f.write(f"    {metric}: {value:.4f}\n")
            
            f.write("\n")
    
    print(f"Summary saved to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Compare IRT parameters across models')
    parser.add_argument('--dataset', type=str, default='synthetic_OC',
                       help='Dataset name')
    parser.add_argument('--baseline_model', type=str, 
                       default='save_models/best_baseline_synthetic_OC.pth',
                       help='Path to baseline model')
    parser.add_argument('--bayesian_model', type=str,
                       default='save_models/best_bayesian_synthetic_OC.pth',
                       help='Path to Bayesian model')
    parser.add_argument('--akvmn_model', type=str,
                       default='save_models/best_akvmn_synthetic_OC.pth',
                       help='Path to AKVMN model')
    parser.add_argument('--output_dir', type=str, default='irt_comparison',
                       help='Output directory for comparison results')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for parameter extraction')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load test data
    data_path = Path('data') / args.dataset
    test_data = load_gpcm_data(data_path / f'synthetic_oc_test.txt')
    
    # Unpack data
    test_sequences, test_questions, test_responses, n_categories = test_data
    
    # Create data loader
    test_dataset = GpcmDataset(test_questions, test_responses)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Get data dimensions
    all_questions = [q for sublist in test_questions for q in sublist]
    n_questions = max(all_questions) + 1
    
    print(f"Dataset: {args.dataset}")
    print(f"Questions: {n_questions}, Categories: {n_categories}")
    
    # Dictionary to store all parameters
    all_params = {}
    
    # Load ground truth parameters (if available)
    true_params = load_ground_truth_params(data_path)
    if true_params is not None:
        all_params['ground_truth'] = true_params
        print("Loaded ground truth IRT parameters")
    
    # Extract baseline parameters
    if Path(args.baseline_model).exists():
        print(f"\nExtracting parameters from baseline model: {args.baseline_model}")
        baseline_params = extract_baseline_params(
            Path(args.baseline_model), test_loader, n_questions, n_categories
        )
        all_params['baseline'] = baseline_params
        print(f"Extracted {baseline_params['model_type']} model parameters")
    
    # Extract Bayesian parameters
    if Path(args.bayesian_model).exists():
        print(f"\nExtracting parameters from Bayesian model: {args.bayesian_model}")
        bayesian_params = extract_bayesian_params(Path(args.bayesian_model))
        all_params['bayesian'] = bayesian_params
        print("Extracted Bayesian model parameters (with uncertainty)")
    
    # Extract AKVMN parameters
    if Path(args.akvmn_model).exists():
        print(f"\nExtracting parameters from AKVMN model: {args.akvmn_model}")
        akvmn_params = extract_baseline_params(
            Path(args.akvmn_model), test_loader, n_questions, n_categories
        )
        all_params['akvmn'] = akvmn_params
        print(f"Extracted {akvmn_params['model_type']} model parameters")
    
    # Compute comparison metrics
    print("\nComputing comparison metrics...")
    comparison_metrics = {}
    
    if 'ground_truth' in all_params:
        # Compare each model with ground truth
        for model_name, params in all_params.items():
            if model_name == 'ground_truth':
                continue
            
            print(f"\nComparing {model_name} with ground truth...")
            metrics = {}
            
            # Theta metrics
            theta_metrics = compute_distribution_metrics(
                all_params['ground_truth']['theta'], 
                params['theta'],
                'theta'
            )
            metrics.update(theta_metrics)
            
            # Alpha metrics
            alpha_metrics = compute_distribution_metrics(
                all_params['ground_truth']['alpha'],
                params['alpha'],
                'alpha'
            )
            metrics.update(alpha_metrics)
            
            # Beta metrics (using mean)
            beta_metrics = compute_distribution_metrics(
                all_params['ground_truth']['beta'].mean(axis=1),
                params['beta'].mean(axis=1),
                'beta'
            )
            metrics.update(beta_metrics)
            
            comparison_metrics[f'{model_name}_vs_truth'] = metrics
    
    # Compare models with each other
    model_names = [k for k in all_params.keys() if k != 'ground_truth']
    for i, model1 in enumerate(model_names):
        for model2 in model_names[i+1:]:
            print(f"\nComparing {model1} with {model2}...")
            metrics = {}
            
            # Only compare parameters that exist in both
            for param_name in ['theta', 'alpha', 'beta']:
                if param_name in all_params[model1] and param_name in all_params[model2]:
                    param1 = all_params[model1][param_name]
                    param2 = all_params[model2][param_name]
                    
                    if param_name == 'beta':
                        param1 = param1.mean(axis=1)
                        param2 = param2.mean(axis=1)
                    
                    param_metrics = compute_distribution_metrics(param1, param2, param_name)
                    metrics.update(param_metrics)
            
            comparison_metrics[f'{model1}_vs_{model2}'] = metrics
    
    # Create visualizations
    print("\nCreating comparison visualizations...")
    plot_comprehensive_comparison(all_params, output_dir)
    
    # Create summary table
    create_summary_table(comparison_metrics, output_dir)
    
    # Print key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS:")
    print("=" * 80)
    
    if 'ground_truth' in all_params:
        for model_name in model_names:
            if f'{model_name}_vs_truth' in comparison_metrics:
                metrics = comparison_metrics[f'{model_name}_vs_truth']
                print(f"\n{model_name.upper()} vs Ground Truth:")
                
                # Find best recovered parameter
                correlations = {
                    'theta': metrics.get('theta_correlation', 0),
                    'alpha': metrics.get('alpha_correlation', 0),
                    'beta': metrics.get('beta_correlation', 0)
                }
                
                best_param = max(correlations, key=correlations.get)
                print(f"  Best recovered: {best_param} (r={correlations[best_param]:.3f})")
                
                # Check distribution similarity
                for param in ['theta', 'alpha', 'beta']:
                    ks_pval = metrics.get(f'{param}_ks_pvalue', 0)
                    if ks_pval > 0.05:
                        print(f"  {param}: Similar distribution (KS p={ks_pval:.3f})")
                    else:
                        print(f"  {param}: Different distribution (KS p={ks_pval:.3f})")
    
    print(f"\nAll results saved to: {output_dir}")


if __name__ == '__main__':
    main()