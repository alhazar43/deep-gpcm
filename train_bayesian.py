#!/usr/bin/env python3
"""
Training script for Variational Bayesian GPCM model with IRT parameter comparison.

This script trains the Bayesian model and compares learned IRT parameters with ground truth
from synthetic data generation.
"""

import argparse
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, Tuple, Optional

from models.baseline_bayesian import VariationalBayesianGPCM
from utils.gpcm_utils import load_gpcm_data, GpcmDataset
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
    
    # Theta comparison (student abilities)
    if len(true_params['theta']) == len(learned_theta):
        # Correlation (invariant to linear transformation)
        metrics['theta_correlation'] = np.corrcoef(true_params['theta'], learned_theta)[0, 1]
        
        # Standardize for MSE comparison
        true_theta_std = (true_params['theta'] - true_params['theta'].mean()) / true_params['theta'].std()
        learned_theta_std = (learned_theta - learned_theta.mean()) / learned_theta.std()
        metrics['theta_mse'] = np.mean((true_theta_std - learned_theta_std) ** 2)
    
    # Alpha comparison (discrimination)
    metrics['alpha_correlation'] = np.corrcoef(true_params['alpha'], learned_alpha)[0, 1]
    metrics['alpha_mse'] = np.mean((true_params['alpha'] - learned_alpha) ** 2)
    metrics['alpha_mean_ratio'] = learned_alpha.mean() / true_params['alpha'].mean()
    
    # Beta comparison (thresholds)
    # Compare mean thresholds per question
    true_beta_mean = true_params['beta'].mean(axis=1)
    learned_beta_mean = learned_beta.mean(axis=1)
    metrics['beta_correlation'] = np.corrcoef(true_beta_mean, learned_beta_mean)[0, 1]
    metrics['beta_mse'] = np.mean((true_params['beta'] - learned_beta) ** 2)
    
    # Distribution comparison using KL divergence approximation
    # Compare histogram distributions
    for param_name, true_vals, learned_vals in [
        ('theta', true_params['theta'], learned_theta),
        ('alpha', true_params['alpha'], learned_alpha)
    ]:
        # Create histograms
        bins = np.linspace(min(true_vals.min(), learned_vals.min()),
                          max(true_vals.max(), learned_vals.max()), 20)
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
    
    return metrics


def plot_irt_comparison(true_params: Dict[str, np.ndarray],
                       learned_stats: Dict[str, Dict[str, torch.Tensor]],
                       save_path: Path):
    """Create comparison plots for true vs learned IRT parameters."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Convert learned parameters
    learned_theta = learned_stats['theta']['mean'].cpu().numpy()
    learned_alpha = learned_stats['alpha']['mean'].cpu().numpy()
    learned_beta = learned_stats['beta']['mean'].cpu().numpy()
    
    # Theta comparison
    ax = axes[0, 0]
    ax.scatter(true_params['theta'], learned_theta, alpha=0.5, s=20)
    ax.plot([true_params['theta'].min(), true_params['theta'].max()],
            [true_params['theta'].min(), true_params['theta'].max()], 'r--', label='y=x')
    ax.set_xlabel('True θ (Student Ability)')
    ax.set_ylabel('Learned θ')
    ax.set_title(f'Student Abilities (r={np.corrcoef(true_params["theta"], learned_theta)[0, 1]:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Alpha comparison
    ax = axes[0, 1]
    ax.scatter(true_params['alpha'], learned_alpha, alpha=0.5, s=20)
    ax.plot([0, true_params['alpha'].max()], [0, true_params['alpha'].max()], 'r--', label='y=x')
    ax.set_xlabel('True α (Discrimination)')
    ax.set_ylabel('Learned α')
    ax.set_title(f'Discrimination Parameters (r={np.corrcoef(true_params["alpha"], learned_alpha)[0, 1]:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Beta comparison (mean per question)
    ax = axes[0, 2]
    true_beta_mean = true_params['beta'].mean(axis=1)
    learned_beta_mean = learned_beta.mean(axis=1)
    ax.scatter(true_beta_mean, learned_beta_mean, alpha=0.5, s=20)
    ax.plot([true_beta_mean.min(), true_beta_mean.max()],
            [true_beta_mean.min(), true_beta_mean.max()], 'r--', label='y=x')
    ax.set_xlabel('True β (Mean Threshold)')
    ax.set_ylabel('Learned β')
    ax.set_title(f'Mean Thresholds (r={np.corrcoef(true_beta_mean, learned_beta_mean)[0, 1]:.3f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Distribution comparisons
    # Theta distributions
    ax = axes[1, 0]
    ax.hist(true_params['theta'], bins=20, alpha=0.5, label='True θ ~ N(0,1)', density=True, color='blue')
    ax.hist(learned_theta, bins=20, alpha=0.5, label='Learned θ', density=True, color='orange')
    ax.set_xlabel('θ (Student Ability)')
    ax.set_ylabel('Density')
    ax.set_title('Student Ability Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Alpha distributions
    ax = axes[1, 1]
    ax.hist(true_params['alpha'], bins=20, alpha=0.5, label='True α ~ LogN(0,0.3)', density=True, color='blue')
    ax.hist(learned_alpha, bins=20, alpha=0.5, label='Learned α', density=True, color='orange')
    ax.set_xlabel('α (Discrimination)')
    ax.set_ylabel('Density')
    ax.set_title('Discrimination Distributions')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Beta spread comparison
    ax = axes[1, 2]
    true_beta_std = true_params['beta'].std(axis=1)
    learned_beta_std = learned_beta.std(axis=1)
    ax.scatter(true_beta_std, learned_beta_std, alpha=0.5, s=20)
    ax.plot([0, true_beta_std.max()], [0, true_beta_std.max()], 'r--', label='y=x')
    ax.set_xlabel('True β Spread (STD)')
    ax.set_ylabel('Learned β Spread')
    ax.set_title('Threshold Spread per Question')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'irt_parameter_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    # Additional plot: Posterior uncertainty
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Theta uncertainty
    ax = axes[0]
    theta_std = learned_stats['theta']['std'].cpu().numpy()
    ax.hist(theta_std, bins=20, alpha=0.7, color='green')
    ax.set_xlabel('Posterior STD')
    ax.set_ylabel('Count')
    ax.set_title(f'Student Ability Uncertainty (mean STD: {theta_std.mean():.3f})')
    ax.grid(True, alpha=0.3)
    
    # Alpha uncertainty
    ax = axes[1]
    alpha_std = learned_stats['alpha']['std'].cpu().numpy()
    ax.hist(alpha_std, bins=20, alpha=0.7, color='green')
    ax.set_xlabel('Posterior STD')
    ax.set_ylabel('Count')
    ax.set_title(f'Discrimination Uncertainty (mean STD: {alpha_std.mean():.3f})')
    ax.grid(True, alpha=0.3)
    
    # Beta uncertainty
    ax = axes[2]
    beta_std = learned_stats['beta']['std'].cpu().numpy()
    ax.hist(beta_std.flatten(), bins=20, alpha=0.7, color='green')
    ax.set_xlabel('Posterior STD')
    ax.set_ylabel('Count')
    ax.set_title(f'Threshold Uncertainty (mean STD: {beta_std.mean():.3f})')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path / 'posterior_uncertainty.png', dpi=150, bbox_inches='tight')
    plt.close()


def train_epoch(model: nn.Module, data_loader: DataLoader, optimizer: optim.Optimizer,
                device: torch.device) -> Tuple[float, float, Dict[str, float]]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_elbo = 0
    total_kl = 0
    n_batches = 0
    
    all_preds = []
    all_targets = []
    
    for questions, responses in data_loader:
        questions = questions.to(device)
        responses = responses.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with variational sampling
        probabilities, aux_dict = model(questions, responses)
        
        # Compute ELBO loss
        kl_div = aux_dict['kl_divergence']
        elbo_loss = model.elbo_loss(probabilities, responses, kl_div)
        
        # Backward pass
        elbo_loss.backward()
        optimizer.step()
        
        # Track losses
        total_loss += elbo_loss.item()
        total_elbo += elbo_loss.item()
        total_kl += kl_div.item()
        n_batches += 1
        
        # Store predictions
        preds = probabilities.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy().flatten())
        all_targets.extend(responses.cpu().numpy().flatten())
    
    # Compute metrics
    correct = np.sum(np.array(all_preds) == np.array(all_targets))
    total = len(all_targets)
    metrics = {'categorical_accuracy': correct / total}
    
    avg_loss = total_loss / n_batches
    avg_kl = total_kl / n_batches
    
    return avg_loss, avg_kl, metrics


def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device,
            n_categories: int) -> Tuple[float, Dict[str, float]]:
    """Evaluate model performance."""
    model.eval()
    total_loss = 0
    n_batches = 0
    
    all_preds = []
    all_targets = []
    all_probs = []
    
    with torch.no_grad():
        for questions, responses in data_loader:
            questions = questions.to(device)
            responses = responses.to(device)
            
            # Forward pass
            probabilities, aux_dict = model(questions, responses)
            
            # Compute loss
            kl_div = aux_dict['kl_divergence']
            elbo_loss = model.elbo_loss(probabilities, responses, kl_div)
            
            total_loss += elbo_loss.item()
            n_batches += 1
            
            # Store predictions
            preds = probabilities.argmax(dim=-1)
            all_preds.extend(preds.cpu().numpy().flatten())
            all_targets.extend(responses.cpu().numpy().flatten())
            all_probs.extend(probabilities.cpu().numpy().reshape(-1, n_categories))
    
    # Compute metrics
    correct = np.sum(np.array(all_preds) == np.array(all_targets))
    total = len(all_targets)
    metrics = {'categorical_accuracy': correct / total}
    
    avg_loss = total_loss / n_batches
    
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description='Train Variational Bayesian GPCM model')
    parser.add_argument('--dataset', type=str, default='synthetic_OC',
                       help='Dataset name')
    parser.add_argument('--epochs', type=int, default=50,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--kl_weight', type=float, default=1.0,
                       help='Weight for KL divergence term')
    parser.add_argument('--kl_annealing', action='store_true',
                       help='Use KL annealing schedule')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--save_dir', type=str, default='save_models',
                       help='Directory to save models')
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory to save logs')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    save_dir = Path(args.save_dir)
    log_dir = Path(args.log_dir)
    save_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    
    # Load data
    data_path = Path('data') / args.dataset
    train_sequences, train_questions, train_responses, n_cats_train = load_gpcm_data(data_path / f'synthetic_oc_train.txt')
    test_sequences, test_questions, test_responses, n_cats_test = load_gpcm_data(data_path / f'synthetic_oc_test.txt')
    
    # Load true IRT parameters (if available)
    true_params = load_true_irt_params(data_path)
    
    # Create datasets
    train_dataset = GpcmDataset(train_questions, train_responses)
    test_dataset = GpcmDataset(test_questions, test_responses)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Get data dimensions
    n_students = max(len(train_questions), len(test_questions))
    all_questions = [q for sublist in train_questions + test_questions for q in sublist]
    all_responses = [r for sublist in train_responses + test_responses for r in sublist]
    n_questions = max(all_questions) + 1
    n_categories = max(all_responses) + 1
    
    print(f"Dataset: {args.dataset}")
    print(f"Students: {n_students}, Questions: {n_questions}, Categories: {n_categories}")
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Create model
    model = VariationalBayesianGPCM(
        n_students=n_students,
        n_questions=n_questions,
        n_categories=n_categories,
        kl_weight=args.kl_weight if not args.kl_annealing else 0.0
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training history
    history = {
        'train_loss': [],
        'train_kl': [],
        'train_accuracy': [],
        'test_loss': [],
        'test_accuracy': [],
        'test_qwk': [],
        'kl_weight': []
    }
    
    # Training loop
    print("\nStarting training...")
    best_test_accuracy = 0
    best_epoch = 0
    
    for epoch in range(args.epochs):
        # KL annealing schedule
        if args.kl_annealing:
            # Linear annealing from 0 to 1 over half the epochs
            kl_weight = min(1.0, epoch / (args.epochs // 2))
            model.kl_weight = kl_weight
            history['kl_weight'].append(kl_weight)
        else:
            history['kl_weight'].append(args.kl_weight)
        
        # Train
        train_loss, train_kl, train_metrics = train_epoch(model, train_loader, optimizer, device)
        
        # Evaluate
        test_loss, test_metrics = evaluate(model, test_loader, device, n_categories)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_kl'].append(train_kl)
        history['train_accuracy'].append(train_metrics['categorical_accuracy'])
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(test_metrics['categorical_accuracy'])
        history['test_qwk'].append(test_metrics.get('quadratic_weighted_kappa', 0.0))
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, KL: {train_kl:.4f}, Acc: {train_metrics['categorical_accuracy']:.3f}")
        print(f"  Test Loss: {test_loss:.4f}, Acc: {test_metrics['categorical_accuracy']:.3f}, QWK: {test_metrics.get('quadratic_weighted_kappa', 0.0):.3f}")
        
        # Save best model
        if test_metrics['categorical_accuracy'] > best_test_accuracy:
            best_test_accuracy = test_metrics['categorical_accuracy']
            best_epoch = epoch + 1
            
            # Get posterior statistics
            posterior_stats = model.get_posterior_stats()
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy': test_metrics['categorical_accuracy'],
                'test_qwk': test_metrics.get('quadratic_weighted_kappa', 0.0),
                'posterior_stats': posterior_stats,
                'args': vars(args)
            }
            
            save_path = save_dir / f'best_bayesian_{args.dataset}.pth'
            torch.save(checkpoint, save_path)
            print(f"  Saved best model to {save_path}")
    
    print(f"\nTraining complete! Best test accuracy: {best_test_accuracy:.3f} at epoch {best_epoch}")
    
    # Load best model for final analysis
    checkpoint = torch.load(save_dir / f'best_bayesian_{args.dataset}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    posterior_stats = checkpoint['posterior_stats']
    
    # Compare with true parameters if available
    if true_params is not None:
        print("\nComparing learned vs true IRT parameters...")
        comparison_metrics = compute_irt_comparison_metrics(true_params, posterior_stats)
        
        print("\nIRT Parameter Recovery Metrics:")
        for metric_name, value in comparison_metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        # Create comparison plots
        plot_dir = Path('irt_plots') / 'bayesian_comparison'
        plot_dir.mkdir(parents=True, exist_ok=True)
        plot_irt_comparison(true_params, posterior_stats, plot_dir)
        print(f"\nComparison plots saved to {plot_dir}")
        
        # Save comparison metrics
        with open(plot_dir / 'comparison_metrics.json', 'w') as f:
            json.dump(comparison_metrics, f, indent=2)
    
    # Save training history
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    history_file = log_dir / f'bayesian_training_history_{args.dataset}_{timestamp}.json'
    
    # Add final metrics to history
    history['final_metrics'] = {
        'best_epoch': best_epoch,
        'best_test_accuracy': best_test_accuracy,
        'best_test_qwk': history['test_qwk'][best_epoch - 1]
    }
    
    if true_params is not None:
        history['irt_comparison'] = comparison_metrics
    
    with open(history_file, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nTraining history saved to {history_file}")
    
    # Plot training curves
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss curves
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train Loss')
    ax.plot(history['test_loss'], label='Test Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('ELBO Loss')
    ax.set_title('Training and Test Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # KL divergence
    ax = axes[0, 1]
    ax.plot(history['train_kl'], label='KL Divergence', color='red')
    ax2 = ax.twinx()
    ax2.plot(history['kl_weight'], label='KL Weight', color='blue', linestyle='--')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('KL Divergence', color='red')
    ax2.set_ylabel('KL Weight', color='blue')
    ax.set_title('KL Divergence and Annealing')
    ax.grid(True, alpha=0.3)
    
    # Accuracy curves
    ax = axes[1, 0]
    ax.plot(history['train_accuracy'], label='Train Accuracy')
    ax.plot(history['test_accuracy'], label='Test Accuracy')
    ax.axhline(y=best_test_accuracy, color='g', linestyle='--', label=f'Best: {best_test_accuracy:.3f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Categorical Accuracy')
    ax.set_title('Training and Test Accuracy')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # QWK curve
    ax = axes[1, 1]
    ax.plot(history['test_qwk'], label='Test QWK', color='purple')
    ax.axhline(y=history['test_qwk'][best_epoch-1], color='g', linestyle='--', 
              label=f'Best: {history["test_qwk"][best_epoch-1]:.3f}')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Quadratic Weighted Kappa')
    ax.set_title('Test QWK Evolution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(log_dir / f'bayesian_training_curves_{args.dataset}_{timestamp}.png', 
                dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Training curves saved to {log_dir}")


if __name__ == '__main__':
    main()