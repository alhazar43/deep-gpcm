#!/usr/bin/env python3
"""
Standard training for Deep-GPCM using true labels.
All metrics will use hard predictions (argmax) during evaluation.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datetime import datetime

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.implementations.deep_gpcm import DeepGPCM
from utils.metrics import OrdinalMetrics, ensure_results_dirs, save_results

# Reuse functions from train.py
from train import load_simple_data, create_data_loaders


class EarlyStopping:
    """Early stopping utility."""
    def __init__(self, patience=5, verbose=False, delta=0, path='checkpoint.pth'):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, n_epochs):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    total_samples = 0
    predictions = []
    targets = []
    
    for batch_idx, (questions, responses, mask) in enumerate(train_loader):
        questions = questions.to(device)
        responses = responses.to(device)
        mask = mask.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        _, _, _, gpcm_probs = model(questions, responses)
        
        # Standard cross-entropy loss with true labels
        loss = criterion(gpcm_probs.view(-1, gpcm_probs.size(-1)), 
                        responses.view(-1))
        
        # Apply mask to loss
        mask_flat = mask.view(-1)
        loss = (loss * mask_flat).sum() / mask_flat.sum()
        
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item() * mask_flat.sum().item()
        total_samples += mask_flat.sum().item()
        
        # Store predictions and targets for metrics
        with torch.no_grad():
            # Use hard predictions (argmax)
            hard_preds = gpcm_probs.argmax(dim=-1)
            valid_mask = mask_flat.bool()
            predictions.extend(hard_preds.view(-1)[valid_mask].cpu().numpy())
            targets.extend(responses.view(-1)[valid_mask].cpu().numpy())
        
        # Progress update
        if batch_idx % 10 == 0:
            print(f'\rEpoch [{epoch}/{n_epochs}] Batch [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f}', end='')
    
    # Compute metrics
    avg_loss = total_loss / total_samples
    metrics = OrdinalMetrics(n_cats=4)
    
    # Calculate key metrics using hard predictions
    accuracy = metrics.categorical_accuracy(np.array(targets), np.array(predictions))
    qwk = metrics.quadratic_weighted_kappa(np.array(targets), np.array(predictions))
    mae = metrics.mean_absolute_error(np.array(targets), np.array(predictions))
    spearman = metrics.spearman_correlation(np.array(targets), np.array(predictions))
    
    print(f'\nTrain Loss: {avg_loss:.4f}, Acc: {accuracy:.4f}, QWK: {qwk:.4f}, '
          f'MAE: {mae:.4f}, Spearman: {spearman:.4f}')
    
    return avg_loss, accuracy, qwk


def evaluate(model, data_loader, criterion, device, phase='valid'):
    """Evaluate model using hard predictions for all metrics."""
    model.eval()
    total_loss = 0
    total_samples = 0
    all_predictions = []
    all_targets = []
    all_probabilities = []
    
    with torch.no_grad():
        for questions, responses, mask in data_loader:
            questions = questions.to(device)
            responses = responses.to(device)
            mask = mask.to(device)
            
            # Forward pass
            _, _, _, gpcm_probs = model(questions, responses)
            
            # Compute loss
            loss = criterion(gpcm_probs.view(-1, gpcm_probs.size(-1)), 
                           responses.view(-1))
            
            # Apply mask
            mask_flat = mask.view(-1)
            loss = (loss * mask_flat).sum() / mask_flat.sum()
            
            total_loss += loss.item() * mask_flat.sum().item()
            total_samples += mask_flat.sum().item()
            
            # Store results - use hard predictions
            hard_preds = gpcm_probs.argmax(dim=-1)
            valid_mask = mask_flat.bool()
            all_predictions.extend(hard_preds.view(-1)[valid_mask].cpu().numpy())
            all_targets.extend(responses.view(-1)[valid_mask].cpu().numpy())
            all_probabilities.extend(gpcm_probs.view(-1, gpcm_probs.size(-1))[valid_mask].cpu().numpy())
    
    # Compute metrics
    avg_loss = total_loss / total_samples
    metrics = OrdinalMetrics(n_cats=4)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)
    
    # Compute all metrics using hard predictions
    results = {
        'loss': avg_loss,
        'phase': phase,
        'categorical_accuracy': metrics.categorical_accuracy(all_targets, all_predictions),
        'quadratic_weighted_kappa': metrics.quadratic_weighted_kappa(all_targets, all_predictions),
        'cohen_kappa': metrics.cohen_kappa(all_targets, all_predictions),
        'ordinal_accuracy': metrics.ordinal_accuracy(all_targets, all_predictions),
        'mean_absolute_error': metrics.mean_absolute_error(all_targets, all_predictions),
        'mean_squared_error': metrics.mean_squared_error(all_targets, all_predictions),
        'spearman_correlation': metrics.spearman_correlation(all_targets, all_predictions),
        'kendall_tau': metrics.kendall_tau(all_targets, all_predictions),
        'pearson_correlation': metrics.pearson_correlation(all_targets, all_predictions)
    }
    
    # Add probability-based metrics
    prob_metrics = metrics.probability_metrics(all_targets, all_probabilities)
    results.update(prob_metrics)
    
    return avg_loss, results


def main():
    # Configuration
    dataset = 'synthetic_OC'
    n_epochs = 10
    batch_size = 32
    learning_rate = 0.001
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Using device: {device}")
    
    # Ensure results directories exist
    ensure_results_dirs()
    
    # Load data
    print(f"\nLoading {dataset} dataset...")
    train_path = f'data/{dataset}/{dataset.lower()}_train.txt'
    test_path = f'data/{dataset}/{dataset.lower()}_test.txt'
    
    train_data, test_data, n_questions, n_cats = load_simple_data(train_path, test_path)
    print(f"Data loaded: {len(train_data)} train sequences, {len(test_data)} test sequences")
    print(f"Questions: {n_questions}, Categories: {n_cats}")
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(train_data, test_data, batch_size)
    
    # Initialize model
    model = DeepGPCM(
        n_questions=n_questions,
        n_cats=n_cats,
        memory_size=50,
        key_dim=50,
        value_dim=200,
        final_fc_dim=256,
        dropout_rate=0.2
    ).to(device)
    
    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_qwk': [],
        'valid_loss': [], 'valid_acc': [], 'valid_qwk': [],
        'config': {
            'dataset': dataset,
            'n_epochs': n_epochs,
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'n_questions': n_questions,
            'n_cats': n_cats
        }
    }
    
    # Early stopping
    early_stopping = EarlyStopping(patience=10, verbose=True, 
                                 path=f'save_models/best_deep_gpcm_hard_{dataset}.pth')
    
    print("\n" + "="*80)
    print("STANDARD TRAINING WITH HARD PREDICTIONS FOR ALL METRICS")
    print("="*80)
    print("Training with true labels, evaluating all metrics with hard predictions (argmax)")
    print("="*80)
    
    # Training loop
    for epoch in range(1, n_epochs + 1):
        # Train
        train_loss, train_acc, train_qwk = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, n_epochs
        )
        
        # Validate
        valid_loss, valid_results = evaluate(model, test_loader, criterion, device, 'valid')
        
        # Store history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['train_qwk'].append(train_qwk)
        history['valid_loss'].append(valid_loss)
        history['valid_acc'].append(valid_results['categorical_accuracy'])
        history['valid_qwk'].append(valid_results['quadratic_weighted_kappa'])
        
        # Print validation results
        print(f"Valid Loss: {valid_loss:.4f}, "
              f"Acc: {valid_results['categorical_accuracy']:.4f}, "
              f"QWK: {valid_results['quadratic_weighted_kappa']:.4f}, "
              f"MAE: {valid_results['mean_absolute_error']:.4f}, "
              f"Spearman: {valid_results['spearman_correlation']:.4f}")
        
        # Early stopping
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # Load best model
    model.load_state_dict(torch.load(early_stopping.path))
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET (ALL METRICS WITH HARD PREDICTIONS)")
    print("="*80)
    
    test_loss, test_results = evaluate(model, test_loader, criterion, device, 'test')
    
    # Save results
    save_path = save_results(
        {'history': history, 'test_results': test_results},
        f'results/train/standard_hard_training_{dataset}.json'
    )
    
    print(f"\nResults saved to: {save_path}")
    print(f"\n{'Metric':<30} {'Value':>10}")
    print("-" * 42)
    
    # Print all metrics
    for metric, value in sorted(test_results.items()):
        if metric not in ['loss', 'phase'] and isinstance(value, (int, float)):
            print(f"{metric:<30} {value:>10.4f}")


if __name__ == '__main__':
    main()