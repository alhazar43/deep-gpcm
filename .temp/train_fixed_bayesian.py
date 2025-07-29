#!/usr/bin/env python3
"""
Training script for Fixed Bayesian-DKVMN with proper IRT parameter recovery.
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
from datetime import datetime
from typing import Dict, Tuple

from models.fixed_bayesian_dkvmn import FixedBayesianDKVMN
from utils.gpcm_utils import load_gpcm_data, GpcmDataset
from evaluation.metrics import GpcmMetrics


def load_true_irt_params(data_dir: Path) -> Dict[str, np.ndarray]:
    """Load ground truth IRT parameters."""
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


def compute_parameter_recovery_metrics(true_params: Dict[str, np.ndarray],
                                     learned_params: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Compute IRT parameter recovery metrics."""
    metrics = {}
    
    if true_params is None:
        print("Warning: No true parameters available for comparison")
        return metrics
    
    # Convert learned parameters to numpy
    learned_alpha = learned_params['alpha'].cpu().numpy()
    learned_beta = learned_params['beta'].cpu().numpy()
    learned_theta = learned_params['theta'].cpu().numpy()
    
    # Alpha comparison (discrimination parameters)
    n_questions = min(len(true_params['alpha']), len(learned_alpha))
    alpha_corr = np.corrcoef(true_params['alpha'][:n_questions], 
                           learned_alpha[:n_questions])[0, 1]
    metrics['alpha_correlation'] = alpha_corr if not np.isnan(alpha_corr) else 0.0
    metrics['alpha_mse'] = np.mean((true_params['alpha'][:n_questions] - 
                                  learned_alpha[:n_questions]) ** 2)
    
    # Beta comparison (threshold parameters)
    true_beta_mean = true_params['beta'][:n_questions].mean(axis=1)
    learned_beta_mean = learned_beta[:n_questions].mean(axis=1)
    beta_corr = np.corrcoef(true_beta_mean, learned_beta_mean)[0, 1]
    metrics['beta_correlation'] = beta_corr if not np.isnan(beta_corr) else 0.0
    metrics['beta_mse'] = np.mean((true_beta_mean - learned_beta_mean) ** 2)
    
    # Theta comparison (student abilities)
    n_students = min(len(true_params['theta']), len(learned_theta))
    theta_corr = np.corrcoef(true_params['theta'][:n_students], 
                           learned_theta[:n_students])[0, 1]
    metrics['theta_correlation'] = theta_corr if not np.isnan(theta_corr) else 0.0
    metrics['theta_mse'] = np.mean((true_params['theta'][:n_students] - 
                                  learned_theta[:n_students]) ** 2)
    
    return metrics


def train_epoch(model: nn.Module, data_loader: DataLoader, optimizer: optim.Optimizer,
                device: torch.device, grad_clip: float = 5.0) -> Tuple[float, float, Dict[str, float]]:
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_kl = 0
    n_batches = 0
    
    all_preds = []
    all_targets = []
    
    for questions, responses in data_loader:
        questions = questions.to(device)
        responses = responses.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        probabilities, aux_dict = model(questions, responses)
        
        # Compute ELBO loss
        kl_div = aux_dict['kl_divergence']
        elbo_loss = model.elbo_loss(probabilities, responses, kl_div)
        
        # Backward pass
        elbo_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        
        total_loss += elbo_loss.item()
        total_kl += aux_dict.get('raw_kl_divergence', kl_div).item()
        n_batches += 1
        
        # Store predictions
        preds = probabilities.argmax(dim=-1)
        all_preds.extend(preds.cpu().numpy().flatten())
        all_targets.extend(responses.cpu().numpy().flatten())
    
    # Compute metrics
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    metrics = {
        'categorical_accuracy': (all_preds == all_targets).mean()
    }
    
    return total_loss / n_batches, total_kl / n_batches, metrics


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
    all_probs_array = np.array(all_probs)
    all_probs_tensor = torch.tensor(all_probs_array)
    all_targets_tensor = torch.tensor(all_targets)
    
    metrics = {
        'categorical_accuracy': GpcmMetrics.categorical_accuracy(all_probs_tensor, all_targets_tensor),
        'quadratic_weighted_kappa': GpcmMetrics.quadratic_weighted_kappa(all_probs_tensor, all_targets_tensor, n_categories),
        'ordinal_accuracy': GpcmMetrics.ordinal_accuracy(all_probs_tensor, all_targets_tensor),
        'mean_absolute_error': GpcmMetrics.mean_absolute_error(all_probs_tensor, all_targets_tensor)
    }
    
    avg_loss = total_loss / n_batches if n_batches > 0 else 0.0
    
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description='Train Fixed Bayesian-DKVMN GPCM model')
    parser.add_argument('--dataset', type=str, default='synthetic_OC',
                       help='Dataset name')
    parser.add_argument('--epochs', type=int, default=100,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.003,
                       help='Learning rate')
    parser.add_argument('--memory_size', type=int, default=20,
                       help='Memory size for student abilities')
    parser.add_argument('--ability_dim', type=int, default=32,
                       help='Ability dimension')
    parser.add_argument('--patience', type=int, default=20,
                       help='Early stopping patience')
    parser.add_argument('--grad_clip', type=float, default=5.0,
                       help='Gradient clipping threshold')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use (cuda/cpu)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories
    save_dir = Path('save_models')
    log_dir = Path('logs')
    save_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    
    # Load data
    data_path = Path('data') / args.dataset
    train_sequences, train_questions, train_responses, n_cats_train = load_gpcm_data(data_path / f'synthetic_oc_train.txt')
    test_sequences, test_questions, test_responses, n_cats_test = load_gpcm_data(data_path / f'synthetic_oc_test.txt')
    
    # Load true IRT parameters
    true_params = load_true_irt_params(data_path)
    
    # Create datasets
    train_dataset = GpcmDataset(train_questions, train_responses)
    test_dataset = GpcmDataset(test_questions, test_responses)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    
    # Get data dimensions
    all_questions = [q for sublist in train_questions + test_questions for q in sublist]
    all_responses = [r for sublist in train_responses + test_responses for r in sublist]
    n_questions = max(all_questions) + 1
    n_categories = max(all_responses) + 1
    
    print(f"Dataset: {args.dataset}")
    print(f"Questions: {n_questions}, Categories: {n_categories}")
    print(f"Train samples: {len(train_dataset)}, Test samples: {len(test_dataset)}")
    
    # Create model
    model = FixedBayesianDKVMN(
        n_questions=n_questions,
        n_categories=n_categories,
        memory_size=args.memory_size,
        ability_dim=args.ability_dim
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
        'test_ordinal_accuracy': [],
        'test_mae': []
    }
    
    # Training loop
    print("\nStarting training...")
    best_test_accuracy = 0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(args.epochs):
        # Update model epoch for KL annealing
        model.set_epoch(epoch)
        
        # Train
        train_loss, train_kl, train_metrics = train_epoch(model, train_loader, optimizer, device, args.grad_clip)
        
        # Evaluate
        test_loss, test_metrics = evaluate(model, test_loader, device, n_categories)
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_kl'].append(train_kl)
        history['train_accuracy'].append(train_metrics['categorical_accuracy'])
        history['test_loss'].append(test_loss)
        history['test_accuracy'].append(test_metrics['categorical_accuracy'])
        history['test_qwk'].append(test_metrics['quadratic_weighted_kappa'])
        history['test_ordinal_accuracy'].append(test_metrics['ordinal_accuracy'])
        history['test_mae'].append(test_metrics['mean_absolute_error'])
        
        # Print progress
        print(f"Epoch {epoch+1}/{args.epochs}")
        print(f"  Train Loss: {train_loss:.4f}, KL: {train_kl:.4f}, Acc: {train_metrics['categorical_accuracy']:.3f}")
        print(f"  Test Loss: {test_loss:.4f}, Acc: {test_metrics['categorical_accuracy']:.3f}, "
              f"QWK: {test_metrics['quadratic_weighted_kappa']:.3f}, "
              f"Ord: {test_metrics['ordinal_accuracy']:.3f}, "
              f"MAE: {test_metrics['mean_absolute_error']:.3f}")
        
        # Save best model and early stopping
        if test_metrics['categorical_accuracy'] > best_test_accuracy:
            best_test_accuracy = test_metrics['categorical_accuracy']
            best_epoch = epoch + 1
            patience_counter = 0
            
            # Extract interpretable parameters
            learned_params = model.get_interpretable_parameters()
            
            checkpoint = {
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_accuracy': test_metrics['categorical_accuracy'],
                'test_qwk': test_metrics['quadratic_weighted_kappa'],
                'learned_params': learned_params,
                'args': vars(args)
            }
            
            save_path = save_dir / f'best_fixed_bayesian_{args.dataset}.pth'
            torch.save(checkpoint, save_path)
            print(f"  Saved best model to {save_path}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
    
    print(f"\nTraining complete! Best test accuracy: {best_test_accuracy:.3f} at epoch {best_epoch}")
    
    # Load best model for final analysis
    checkpoint = torch.load(save_dir / f'best_fixed_bayesian_{args.dataset}.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    learned_params = checkpoint['learned_params']
    
    # Compare with true parameters if available
    if true_params is not None:
        print("\nComparing learned vs true IRT parameters...")
        comparison_metrics = compute_parameter_recovery_metrics(true_params, learned_params)
        
        print("\nIRT Parameter Recovery Metrics:")
        for metric_name, value in comparison_metrics.items():
            print(f"  {metric_name}: {value:.4f}")
        
        # Add to final results
        history['final_metrics'] = {
            'best_epoch': best_epoch,
            'best_test_accuracy': best_test_accuracy,
            'best_test_qwk': test_metrics['quadratic_weighted_kappa']
        }
        history['irt_comparison'] = comparison_metrics
    
    # Save training history
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    history_file = log_dir / f'fixed_bayesian_training_history_{args.dataset}_{timestamp}.json'
    
    # Convert tensors to lists for JSON serialization
    def convert_tensor(obj):
        if isinstance(obj, torch.Tensor):
            return obj.cpu().numpy().tolist()
        return obj
    
    history_serializable = {k: convert_tensor(v) for k, v in history.items()}
    
    with open(history_file, 'w') as f:
        json.dump(history_serializable, f, indent=2)
    
    print(f"\nTraining history saved to {history_file}")


if __name__ == '__main__':
    main()