#!/usr/bin/env python3
"""
Train Deep-GPCM model using threshold predictions (cumulative probability >= 0.5).
This trains the model to predict the median of the probability distribution.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
import argparse
import warnings

# Add project root to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.implementations.deep_gpcm import DeepGPCM
from utils.metrics import OrdinalMetrics, ensure_results_dirs, save_results
from utils.predictions import compute_threshold_predictions, categorical_to_cumulative

# Suppress warnings
warnings.filterwarnings('ignore')


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


def load_simple_data(train_path, test_path):
    """Simple data loading function."""
    def read_data(file_path):
        sequences = []
        with open(file_path, 'r') as f:
            lines = f.readlines()
            i = 0
            while i < len(lines):
                if i + 2 >= len(lines):
                    break
                seq_len = int(lines[i].strip())
                questions = list(map(int, lines[i+1].strip().split(',')))
                responses = list(map(int, lines[i+2].strip().split(',')))
                
                # Ensure lengths match
                questions = questions[:seq_len]
                responses = responses[:seq_len]
                
                sequences.append((questions, responses))
                i += 3
        return sequences
    
    train_data = read_data(train_path)
    test_data = read_data(test_path)
    
    # Get data dimensions
    all_questions = set()
    all_cats = set()
    for seq in train_data + test_data:
        all_questions.update(seq[0])
        all_cats.update(seq[1])
    
    n_questions = max(all_questions) + 1
    n_cats = max(all_cats) + 1
    
    return train_data, test_data, n_questions, n_cats


def create_data_loaders(train_data, test_data, batch_size=32, max_seq_len=200):
    """Create data loaders from sequences."""
    def collate_fn(batch):
        batch_size = len(batch)
        questions = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        responses = torch.zeros(batch_size, max_seq_len, dtype=torch.long)
        mask = torch.zeros(batch_size, max_seq_len, dtype=torch.float)
        
        for i, (q_seq, r_seq) in enumerate(batch):
            seq_len = min(len(q_seq), max_seq_len)
            questions[i, :seq_len] = torch.tensor(q_seq[:seq_len])
            responses[i, :seq_len] = torch.tensor(r_seq[:seq_len])
            mask[i, :seq_len] = 1.0
        
        return questions, responses, mask
    
    train_loader = torch.utils.data.DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    test_loader = torch.utils.data.DataLoader(
        test_data, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )
    
    return train_loader, test_loader


def compute_threshold_label(probs):
    """Compute threshold-based label using cumulative probability >= 0.5.
    
    This finds the median of the distribution: the smallest k where P(Y <= k) >= 0.5
    or equivalently, the smallest k where P(Y > k) < 0.5.
    """
    # Ensure probs are on CPU for numpy operations
    if isinstance(probs, torch.Tensor):
        probs = probs.detach().cpu()
    
    # Convert to cumulative probabilities P(Y > k)
    # For 4 categories, we get P(Y > 0), P(Y > 1), P(Y > 2)
    cum_probs_gt = []
    n_cats = probs.shape[-1]
    
    for k in range(n_cats - 1):
        # P(Y > k) = sum of probabilities for categories > k
        p_gt_k = probs[..., k+1:].sum(dim=-1)
        cum_probs_gt.append(p_gt_k)
    
    cum_probs_gt = torch.stack(cum_probs_gt, dim=-1)
    
    # Find the smallest k where P(Y > k) < 0.5
    # This is equivalent to finding where P(Y <= k) >= 0.5
    predictions = torch.zeros_like(probs[..., 0], dtype=torch.long)
    
    # Work backwards from highest to lowest category
    for k in range(n_cats - 2, -1, -1):
        # If P(Y > k) >= 0.5, then predict > k
        mask_gt_k = cum_probs_gt[..., k] >= 0.5
        predictions = torch.where(mask_gt_k, k + 1, predictions)
    
    return predictions


def train_epoch(model, train_loader, optimizer, criterion, device, epoch, n_epochs):
    """Train for one epoch using threshold predictions."""
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
        
        # Compute threshold predictions for training
        with torch.no_grad():
            # Get threshold-based predictions
            threshold_preds = compute_threshold_label(gpcm_probs)
            # Ensure it's on the same device
            threshold_preds = threshold_preds.to(device)
        
        # Use threshold predictions as targets for training
        # This trains the model to predict the median
        loss = criterion(gpcm_probs.view(-1, gpcm_probs.size(-1)), 
                        threshold_preds.view(-1))
        
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
        valid_mask = mask_flat.bool()
        predictions.extend(threshold_preds.view(-1)[valid_mask].cpu().numpy())
        targets.extend(responses.view(-1)[valid_mask].cpu().numpy())
        
        # Progress update
        if batch_idx % 10 == 0:
            print(f'\rEpoch [{epoch}/{n_epochs}] Batch [{batch_idx}/{len(train_loader)}] '
                  f'Loss: {loss.item():.4f}', end='')
    
    # Compute metrics
    avg_loss = total_loss / total_samples
    metrics = OrdinalMetrics(n_cats=4)
    
    # Calculate key metrics
    accuracy = metrics.categorical_accuracy(np.array(targets), np.array(predictions))
    qwk = metrics.quadratic_weighted_kappa(np.array(targets), np.array(predictions))
    
    print(f'\nTrain Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}, QWK: {qwk:.4f}')
    
    return avg_loss, accuracy, qwk


def evaluate(model, data_loader, criterion, device, phase='valid'):
    """Evaluate model using threshold predictions."""
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
            
            # Compute threshold predictions
            threshold_preds = compute_threshold_label(gpcm_probs)
            # Ensure it's on the same device
            threshold_preds = threshold_preds.to(device)
            
            # Compute loss using threshold predictions
            loss = criterion(gpcm_probs.view(-1, gpcm_probs.size(-1)), 
                           threshold_preds.view(-1))
            
            # Apply mask
            mask_flat = mask.view(-1)
            loss = (loss * mask_flat).sum() / mask_flat.sum()
            
            total_loss += loss.item() * mask_flat.sum().item()
            total_samples += mask_flat.sum().item()
            
            # Store results
            valid_mask = mask_flat.bool()
            all_predictions.extend(threshold_preds.view(-1)[valid_mask].cpu().numpy())
            all_targets.extend(responses.view(-1)[valid_mask].cpu().numpy())
            all_probabilities.extend(gpcm_probs.view(-1, gpcm_probs.size(-1))[valid_mask].cpu().numpy())
    
    # Compute metrics
    avg_loss = total_loss / total_samples
    metrics = OrdinalMetrics(n_cats=4)
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_targets = np.array(all_targets)
    all_probabilities = np.array(all_probabilities)
    
    # Compute all metrics
    results = metrics.compute_all_metrics(all_targets, all_predictions, all_probabilities)
    results['loss'] = avg_loss
    results['phase'] = phase
    
    return avg_loss, results


def main():
    parser = argparse.ArgumentParser(description='Train Deep-GPCM with threshold predictions')
    parser.add_argument('--dataset', type=str, default='synthetic_OC',
                       choices=['synthetic_OC', 'synthetic_PC'],
                       help='Dataset to use')
    parser.add_argument('--n_epochs', type=int, default=10,
                       help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension')
    parser.add_argument('--n_layers', type=int, default=2,
                       help='Number of layers')
    parser.add_argument('--dropout', type=float, default=0.2,
                       help='Dropout rate')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    
    args = parser.parse_args()
    
    # Ensure results directories exist
    ensure_results_dirs()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data
    print(f"\nLoading {args.dataset} dataset...")
    train_path = f'data/{args.dataset}/{args.dataset.lower()}_train.txt'
    test_path = f'data/{args.dataset}/{args.dataset.lower()}_test.txt'
    
    # Load data
    train_data, test_data, n_questions, n_cats = load_simple_data(train_path, test_path)
    print(f"Data loaded: {len(train_data)} train sequences, {len(test_data)} test sequences")
    print(f"Questions: {n_questions}, Categories: {n_cats}")
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(train_data, test_data, args.batch_size)
    
    # Initialize model
    model = DeepGPCM(
        n_questions=n_questions,
        n_cats=n_cats,
        memory_size=50,
        key_dim=50,
        value_dim=200,
        final_fc_dim=args.hidden_dim,
        dropout_rate=args.dropout
    ).to(device)
    
    print(f"\nModel initialized with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss(reduction='none')
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
    
    # Training history
    history = {
        'train_loss': [], 'train_acc': [], 'train_qwk': [],
        'valid_loss': [], 'valid_acc': [], 'valid_qwk': [],
        'config': vars(args)
    }
    
    # Early stopping
    early_stopping = EarlyStopping(patience=10, verbose=True, 
                                 path=f'save_models/best_deep_gpcm_threshold_{args.dataset}.pth')
    
    print("\n" + "="*80)
    print("TRAINING WITH THRESHOLD PREDICTIONS (MEDIAN)")
    print("="*80)
    print("Using cumulative probability >= 0.5 for all categories")
    print("This trains the model to predict the median of the distribution")
    print("="*80)
    
    # Training loop
    for epoch in range(1, args.n_epochs + 1):
        # Train
        train_loss, train_acc, train_qwk = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, args.n_epochs
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
              f"Accuracy: {valid_results['categorical_accuracy']:.4f}, "
              f"QWK: {valid_results['quadratic_weighted_kappa']:.4f}, "
              f"Ordinal Acc: {valid_results['ordinal_accuracy']:.4f}")
        
        # Early stopping
        early_stopping(valid_loss, model)
        if early_stopping.early_stop:
            print("Early stopping triggered")
            break
    
    # Load best model
    model.load_state_dict(torch.load(early_stopping.path))
    
    # Final evaluation
    print("\n" + "="*80)
    print("FINAL EVALUATION ON TEST SET")
    print("="*80)
    
    test_loss, test_results = evaluate(model, test_loader, criterion, device, 'test')
    
    # Save results
    save_path = save_results(
        {'history': history, 'test_results': test_results},
        f'results/train/threshold_training_{args.dataset}.json'
    )
    
    print(f"\nResults saved to: {save_path}")
    print(f"\nFinal Test Metrics:")
    print(f"  Loss: {test_loss:.4f}")
    print(f"  Categorical Accuracy: {test_results['categorical_accuracy']:.4f}")
    print(f"  QWK: {test_results['quadratic_weighted_kappa']:.4f}")
    print(f"  Ordinal Accuracy: {test_results['ordinal_accuracy']:.4f}")
    print(f"  Spearman Correlation: {test_results['spearman_correlation']:.4f}")
    print(f"  Kendall Tau: {test_results['kendall_tau']:.4f}")


if __name__ == '__main__':
    main()