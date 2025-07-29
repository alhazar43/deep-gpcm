#!/usr/bin/env python3
"""
Training script for Improved Variational Bayesian GPCM model.

This script trains the improved model focusing on better IRT parameter recovery
through architectural improvements rather than naive loss term approaches.
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Tuple, Optional

from models.improved_bayesian_gpcm import ImprovedVariationalBayesianGPCM
from utils.gpcm_utils import load_gpcm_data, GpcmDataset, detect_n_categories
from evaluation.metrics import GpcmMetrics


def load_true_irt_params(data_dir: Path) -> Dict[str, np.ndarray]:
    """Load ground truth IRT parameters from synthetic data generation."""
    param_file = data_dir / 'true_irt_parameters.json'
    
    if not param_file.exists():
        print(f"Warning: No true IRT parameters found at {param_file}")
        return None
    
    with open(param_file, 'r') as f:
        params = json.load(f)
    
    return {
        'theta': np.array(params['student_abilities']['theta']),
        'alpha': np.array(params['question_parameters']['discrimination']['alpha']),
        'beta': np.array(params['question_parameters']['difficulties']['beta'])
    }


def compute_irt_comparison_metrics(true_params: Dict[str, np.ndarray],
                                 learned_stats: Dict[str, Dict[str, torch.Tensor]]) -> Dict[str, float]:
    """Compute metrics comparing true and learned IRT parameters."""
    metrics = {}
    
    # Convert learned parameters to numpy
    learned_theta = learned_stats['theta']['mean'].cpu().numpy()
    learned_alpha = learned_stats['alpha']['mean'].cpu().numpy()
    learned_beta = learned_stats['beta']['mean'].cpu().numpy()
    
    # Handle size mismatches
    n_learned_students = len(learned_theta)
    n_true_students = len(true_params['theta'])
    n_compare_students = min(n_learned_students, n_true_students)
    
    true_theta_subset = true_params['theta'][:n_compare_students]
    learned_theta_subset = learned_theta[:n_compare_students]
    
    # Theta comparison (student abilities)
    if n_compare_students > 1:
        metrics['theta_correlation'] = np.corrcoef(true_theta_subset, learned_theta_subset)[0, 1]
    else:
        metrics['theta_correlation'] = 0.0
    metrics['theta_mse'] = np.mean((true_theta_subset - learned_theta_subset) ** 2)
    
    # Alpha comparison (discrimination)
    metrics['alpha_correlation'] = np.corrcoef(true_params['alpha'], learned_alpha)[0, 1]
    metrics['alpha_mse'] = np.mean((true_params['alpha'] - learned_alpha) ** 2)
    metrics['alpha_mean_ratio'] = learned_alpha.mean() / true_params['alpha'].mean()
    
    # Beta comparison (thresholds) - compare mean per question
    true_beta_mean = true_params['beta'].mean(axis=1)
    learned_beta_mean = learned_beta.mean(axis=1)
    metrics['beta_correlation'] = np.corrcoef(true_beta_mean, learned_beta_mean)[0, 1]
    metrics['beta_mse'] = np.mean((true_params['beta'] - learned_beta) ** 2)
    
    # Distribution comparison using KL divergence approximation
    for param_name, true_vals, learned_vals in [
        ('theta', true_theta_subset, learned_theta_subset),
        ('alpha', true_params['alpha'], learned_alpha)
    ]:
        if len(true_vals) > 5 and len(learned_vals) > 5:  # Need sufficient data for histogram
            bins = np.linspace(min(true_vals.min(), learned_vals.min()),
                              max(true_vals.max(), learned_vals.max()), 15)
            true_hist, _ = np.histogram(true_vals, bins=bins, density=True)
            learned_hist, _ = np.histogram(learned_vals, bins=bins, density=True)
            
            # Add small epsilon to avoid log(0)
            eps = 1e-8
            true_hist = true_hist + eps
            learned_hist = learned_hist + eps
            
            # Normalize
            true_hist = true_hist / true_hist.sum()
            learned_hist = learned_hist / learned_hist.sum()
            
            # KL divergence
            kl_div = np.sum(learned_hist * np.log(learned_hist / true_hist))
            metrics[f'{param_name}_kl_divergence'] = kl_div
        else:
            metrics[f'{param_name}_kl_divergence'] = float('inf')
    
    return metrics


def evaluate(model: ImprovedVariationalBayesianGPCM, data_loader: DataLoader, 
            device: torch.device, n_categories: int) -> Dict[str, float]:
    """Evaluate model with comprehensive metrics."""
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for batch in data_loader:
            questions, responses = batch
            questions = questions.to(device)
            responses = responses.to(device)
            
            probabilities, aux_dict = model(questions, responses)
            
            # Compute loss
            kl_div = aux_dict['kl_divergence']
            loss = model.elbo_loss(probabilities, responses, kl_div)
            total_loss += loss.item()
            
            # Collect predictions
            all_probs.append(probabilities.cpu())
            all_targets.append(responses.cpu())
    
    # Concatenate all predictions
    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute metrics
    metrics_calc = GpcmMetrics()
    metrics = {
        'loss': total_loss / len(data_loader),
        'categorical_accuracy': metrics_calc.categorical_accuracy(all_probs, all_targets),
        'quadratic_weighted_kappa': metrics_calc.quadratic_weighted_kappa(all_probs, all_targets, n_categories),
        'ordinal_accuracy': metrics_calc.ordinal_accuracy(all_probs, all_targets),
        'mean_absolute_error': metrics_calc.mean_absolute_error(all_probs, all_targets)
    }
    
    return metrics


def plot_irt_comparison(true_params: Dict[str, np.ndarray],
                       learned_stats: Dict[str, Dict[str, torch.Tensor]],
                       save_path: Path):
    """Plot comparison between true and learned IRT parameters."""
    learned_theta = learned_stats['theta']['mean'].cpu().numpy()
    learned_alpha = learned_stats['alpha']['mean'].cpu().numpy()
    learned_beta = learned_stats['beta']['mean'].cpu().numpy()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Student abilities (theta)
    n_compare = min(len(true_params['theta']), len(learned_theta))
    if n_compare > 1:
        axes[0, 0].scatter(true_params['theta'][:n_compare], learned_theta[:n_compare], alpha=0.6, s=20)
        axes[0, 0].plot([true_params['theta'][:n_compare].min(), true_params['theta'][:n_compare].max()], 
                       [true_params['theta'][:n_compare].min(), true_params['theta'][:n_compare].max()], 'r--')
        corr = np.corrcoef(true_params['theta'][:n_compare], learned_theta[:n_compare])[0, 1]
        axes[0, 0].set_title(f'Student Abilities (θ)\nCorrelation: {corr:.3f}')
        axes[0, 0].set_xlabel('True θ')
        axes[0, 0].set_ylabel('Learned θ')
    else:
        axes[0, 0].text(0.5, 0.5, 'Insufficient data', ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title('Student Abilities (θ)\nCorrelation: N/A')
    
    # Discrimination parameters (alpha)
    axes[0, 1].scatter(true_params['alpha'], learned_alpha, alpha=0.6, s=20)
    axes[0, 1].plot([true_params['alpha'].min(), true_params['alpha'].max()], 
                   [true_params['alpha'].min(), true_params['alpha'].max()], 'r--')
    corr = np.corrcoef(true_params['alpha'], learned_alpha)[0, 1]
    axes[0, 1].set_title(f'Discrimination (α)\nCorrelation: {corr:.3f}')
    axes[0, 1].set_xlabel('True α')
    axes[0, 1].set_ylabel('Learned α')
    
    # Threshold parameters (beta) - compare means
    true_beta_mean = true_params['beta'].mean(axis=1)
    learned_beta_mean = learned_beta.mean(axis=1)
    axes[0, 2].scatter(true_beta_mean, learned_beta_mean, alpha=0.6, s=20)
    axes[0, 2].plot([true_beta_mean.min(), true_beta_mean.max()], 
                   [true_beta_mean.min(), true_beta_mean.max()], 'r--')
    corr = np.corrcoef(true_beta_mean, learned_beta_mean)[0, 1]
    axes[0, 2].set_title(f'Thresholds (β mean)\nCorrelation: {corr:.3f}')
    axes[0, 2].set_xlabel('True β mean')
    axes[0, 2].set_ylabel('Learned β mean')
    
    # Parameter distributions
    axes[1, 0].hist(true_params['theta'][:n_compare], alpha=0.5, label='True', bins=15, density=True)
    axes[1, 0].hist(learned_theta[:n_compare], alpha=0.5, label='Learned', bins=15, density=True)
    axes[1, 0].set_title('θ Distribution')
    axes[1, 0].legend()
    
    axes[1, 1].hist(true_params['alpha'], alpha=0.5, label='True', bins=15, density=True)
    axes[1, 1].hist(learned_alpha, alpha=0.5, label='Learned', bins=15, density=True)
    axes[1, 1].set_title('α Distribution')
    axes[1, 1].legend()
    
    axes[1, 2].hist(true_beta_mean, alpha=0.5, label='True', bins=15, density=True)
    axes[1, 2].hist(learned_beta_mean, alpha=0.5, label='Learned', bins=15, density=True)
    axes[1, 2].set_title('β Distribution')
    axes[1, 2].legend()
    
    plt.tight_layout()
    plt.savefig(save_path / 'improved_bayesian_irt_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Train Improved Bayesian GPCM')
    parser.add_argument('--dataset', type=str, default='synthetic_OC',
                       help='Dataset name')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--kl_weight', type=float, default=1.0,
                       help='KL divergence weight')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    data_dir = Path(f'data/{args.dataset}')
    
    # Check if data directory exists and find appropriate files
    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset directory {data_dir} not found")
    
    # Look for train and test files
    train_file = None
    test_file = None
    
    for file_path in data_dir.iterdir():
        if 'train' in file_path.name.lower():
            train_file = file_path
        elif 'test' in file_path.name.lower():
            test_file = file_path
    
    if train_file is None or test_file is None:
        raise FileNotFoundError(f"Could not find train/test files in {data_dir}")
    
    # Load data from individual files
    train_sequences, train_questions, train_responses, n_categories = load_gpcm_data(train_file)
    test_sequences, test_questions, test_responses, _ = load_gpcm_data(test_file)
    
    # Calculate dimensions from data
    all_questions = [q for sublist in train_questions + test_questions for q in sublist]
    all_responses = [r for sublist in train_responses + test_responses for r in sublist]
    n_questions = max(all_questions) + 1
    n_students = len(train_questions)  # Number of sequences = number of students
    
    print(f"Dataset: {args.dataset}")
    print(f"Students: {n_students}, Questions: {n_questions}, Categories: {n_categories}")
    print(f"Train samples: {len(train_questions)}, Test samples: {len(test_questions)}")
    
    # Create datasets and loaders
    train_dataset = GpcmDataset(train_questions, train_responses)
    test_dataset = GpcmDataset(test_questions, test_responses)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Create model
    model = ImprovedVariationalBayesianGPCM(
        n_students=n_students,
        n_questions=n_questions, 
        n_categories=n_categories,
        kl_weight=args.kl_weight
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Setup optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    
    # Training history
    history = {
        'train_loss': [], 'train_kl': [], 'train_accuracy': [],
        'test_loss': [], 'test_accuracy': [], 'test_qwk': [], 
        'test_ordinal': [], 'test_mae': []
    }
    
    best_test_acc = 0.0
    save_dir = Path('save_models')
    save_dir.mkdir(exist_ok=True)
    
    print("\nStarting training...")
    for epoch in range(args.epochs):
        model.set_epoch(epoch)
        
        # Training
        model.train()
        train_loss = 0.0
        train_kl = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in train_loader:
            questions, responses = batch
            questions = questions.to(device)
            responses = responses.to(device)
            
            optimizer.zero_grad()
            
            probabilities, aux_dict = model(questions, responses)
            kl_div = aux_dict['kl_divergence']
            
            loss = model.elbo_loss(probabilities, responses, kl_div)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            train_kl += aux_dict['raw_kl_divergence'].item()
            
            # Compute accuracy
            pred_responses = torch.argmax(probabilities, dim=2)
            train_correct += (pred_responses == responses).sum().item()
            train_total += responses.numel()
        
        train_acc = train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        avg_train_kl = train_kl / len(train_loader)
        
        # Validation
        test_metrics = evaluate(model, test_loader, device, n_categories)
        
        # Update history
        history['train_loss'].append(avg_train_loss)
        history['train_kl'].append(avg_train_kl)
        history['train_accuracy'].append(train_acc)
        history['test_loss'].append(test_metrics['loss'])
        history['test_accuracy'].append(test_metrics['categorical_accuracy'])
        history['test_qwk'].append(test_metrics['quadratic_weighted_kappa'])
        history['test_ordinal'].append(test_metrics['ordinal_accuracy'])
        history['test_mae'].append(test_metrics['mean_absolute_error'])
        
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {avg_train_loss:.4f}, KL: {avg_train_kl:.4f}, Acc: {train_acc:.3f}")
        print(f"  Test Loss: {test_metrics['loss']:.4f}, Acc: {test_metrics['categorical_accuracy']:.3f}, "
              f"QWK: {test_metrics['quadratic_weighted_kappa']:.3f}, Ord: {test_metrics['ordinal_accuracy']:.3f}, "
              f"MAE: {test_metrics['mean_absolute_error']:.3f}")
        
        # Save best model
        if test_metrics['categorical_accuracy'] > best_test_acc:
            best_test_acc = test_metrics['categorical_accuracy']
            
            # Get posterior statistics for saving
            posterior_stats = model.get_posterior_stats()
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'test_accuracy': best_test_acc,
                'posterior_stats': posterior_stats,
                'args': args
            }, save_dir / f'best_improved_bayesian_{args.dataset}.pth')
            print(f"  Saved best model to save_models/best_improved_bayesian_{args.dataset}.pth")
    
    print(f"\nTraining complete! Best test accuracy: {best_test_acc:.3f}")
    
    # Load true IRT parameters for comparison
    true_params = load_true_irt_params(data_dir)
    
    if true_params is not None:
        print("\nComparing learned vs true IRT parameters...")
        
        # Load best model
        checkpoint = torch.load(save_dir / f'best_improved_bayesian_{args.dataset}.pth', 
                              map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Get posterior statistics
        posterior_stats = model.get_posterior_stats()
        
        # Compare parameters
        comparison_metrics = compute_irt_comparison_metrics(true_params, posterior_stats)
        
        print("\nIRT Parameter Recovery Metrics:")
        for metric, value in comparison_metrics.items():
            print(f"  {metric}: {value:.4f}")
        
        # Create comparison plots
        plot_dir = Path('irt_plots')
        plot_dir.mkdir(exist_ok=True)
        plot_irt_comparison(true_params, posterior_stats, plot_dir)
        print(f"\nComparison plots saved to {plot_dir}/improved_bayesian_irt_comparison.png")
    
    # Save training history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    logs_dir = Path('logs')
    logs_dir.mkdir(exist_ok=True)
    
    with open(logs_dir / f'improved_bayesian_training_history_{args.dataset}_{timestamp}.json', 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"Training history saved to logs/")


if __name__ == '__main__':
    main()