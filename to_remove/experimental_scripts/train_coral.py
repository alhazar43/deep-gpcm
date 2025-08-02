#!/usr/bin/env python3
"""
Training script with CORAL support for Deep-GPCM Models
Extends the base training script to support CORAL models and ordinal losses.
"""

import os

# Fix Intel MKL threading issue
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import torch
import json
import time
import argparse
import numpy as np
from datetime import datetime
from sklearn.model_selection import KFold

from utils.metrics import compute_metrics, save_results, ensure_results_dirs
from models.factory import create_model
from models.coral_gpcm import CORALDeepGPCM, HybridCORALGPCM
from models.coral_layer import CORALCompatibleLoss
from training.ordinal_losses import (
    DifferentiableQWKLoss, OrdinalEMDLoss, 
    OrdinalCrossEntropyLoss, CombinedOrdinalLoss
)
import torch.utils.data as data_utils
import torch.nn as nn
import torch.optim as optim


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
    
    # Extract statistics
    all_questions = set()
    all_responses = set()
    
    for questions, responses in train_data + test_data:
        all_questions.update(questions)
        all_responses.update(responses)
    
    n_questions = max(all_questions)
    n_cats = len(all_responses)
    
    print(f"Dataset statistics:")
    print(f"  Questions: {n_questions}")
    print(f"  Categories: {n_cats} ({sorted(all_responses)})")
    print(f"  Train sequences: {len(train_data)}")
    print(f"  Test sequences: {len(test_data)}")
    
    return train_data, test_data, n_questions, n_cats


def collate_fn(batch, max_len=None):
    """Custom collate function with proper padding."""
    if max_len is None:
        max_len = max(len(questions) for questions, _ in batch)
    
    questions_padded = []
    responses_padded = []
    masks = []
    
    for questions, responses in batch:
        q_len = len(questions)
        
        # Pad sequences
        q_padded = questions + [0] * (max_len - q_len)
        r_padded = responses + [0] * (max_len - q_len)
        
        # Create mask (True for valid positions, False for padding)
        mask = [True] * q_len + [False] * (max_len - q_len)
        
        questions_padded.append(q_padded)
        responses_padded.append(r_padded)
        masks.append(mask)
    
    return (torch.tensor(questions_padded), 
            torch.tensor(responses_padded), 
            torch.tensor(masks, dtype=torch.bool))


def create_loss_function(loss_type, n_cats, **kwargs):
    """Create loss function based on type."""
    if loss_type == 'ce':
        return nn.CrossEntropyLoss()
    elif loss_type == 'qwk':
        return DifferentiableQWKLoss(n_cats)
    elif loss_type == 'emd':
        return OrdinalEMDLoss(n_cats)
    elif loss_type == 'ordinal_ce':
        return OrdinalCrossEntropyLoss(n_cats, alpha=1.0)
    elif loss_type == 'coral':
        return CORALCompatibleLoss(n_cats)
    elif loss_type == 'combined':
        return CombinedOrdinalLoss(
            n_cats,
            ce_weight=kwargs.get('ce_weight', 1.0),
            qwk_weight=kwargs.get('qwk_weight', 0.5),
            emd_weight=kwargs.get('emd_weight', 0.0),
            coral_weight=kwargs.get('coral_weight', 0.0)
        )
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")


def train_epoch(model, train_loader, criterion, optimizer, device, is_coral_model=False):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (questions, responses, mask) in enumerate(train_loader):
        questions = questions.to(device)
        responses = responses.to(device)
        mask = mask.to(device)
        
        # Forward pass
        outputs = model(questions, responses)
        
        # Handle different model outputs
        if len(outputs) == 4:
            theta, beta, alpha, probs = outputs
        else:
            raise ValueError(f"Unexpected model output shape: {len(outputs)}")
        
        # Compute loss
        if is_coral_model and hasattr(model, 'get_coral_info'):
            # For CORAL models, use CORAL-compatible loss
            coral_info = model.get_coral_info()
            if isinstance(criterion, CORALCompatibleLoss):
                loss = criterion((probs, coral_info), responses, mask)
            elif isinstance(criterion, CombinedOrdinalLoss):
                # Pass CORAL info to combined loss
                loss_dict = criterion(probs, responses, mask, coral_info)
                loss = loss_dict['total_loss']
            else:
                # Use standard loss on probabilities
                probs_flat = probs.view(-1, probs.size(-1))
                responses_flat = responses.view(-1)
                mask_flat = mask.view(-1).bool()
                
                valid_probs = probs_flat[mask_flat]
                valid_responses = responses_flat[mask_flat]
                
                if isinstance(criterion, nn.CrossEntropyLoss):
                    # Convert probs to logits for CE loss
                    valid_logits = torch.log(valid_probs + 1e-8)
                    loss = criterion(valid_logits, valid_responses)
                else:
                    loss = criterion(valid_probs, valid_responses)
        else:
            # Standard loss computation
            probs_flat = probs.view(-1, probs.size(-1))
            responses_flat = responses.view(-1)
            mask_flat = mask.view(-1).bool()
            
            valid_probs = probs_flat[mask_flat]
            valid_responses = responses_flat[mask_flat]
            
            if isinstance(criterion, nn.CrossEntropyLoss):
                valid_logits = torch.log(valid_probs + 1e-8)
                loss = criterion(valid_logits, valid_responses)
            else:
                loss = criterion(valid_probs, valid_responses)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Calculate accuracy
        pred = probs.argmax(dim=-1)
        valid_pred = pred[mask]
        valid_true = responses[mask]
        correct += (valid_pred == valid_true).sum().item()
        total += mask.sum().item()
        
        total_loss += loss.item()
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct / total if total > 0 else 0
    
    return avg_loss, accuracy


def evaluate(model, test_loader, device, n_cats):
    """Evaluate model on test set."""
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for questions, responses, mask in test_loader:
            questions = questions.to(device)
            responses = responses.to(device)
            mask = mask.to(device)
            
            outputs = model(questions, responses)
            
            if len(outputs) == 4:
                _, _, _, probs = outputs
            else:
                raise ValueError(f"Unexpected model output shape: {len(outputs)}")
            
            pred = probs.argmax(dim=-1)
            
            # Collect predictions and targets
            valid_pred = pred[mask].cpu()
            valid_true = responses[mask].cpu()
            
            all_preds.extend(valid_pred.tolist())
            all_targets.extend(valid_true.tolist())
    
    # Compute metrics
    metrics = compute_metrics(all_targets, all_preds, n_cats)
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description='Train Deep-GPCM models with CORAL support')
    
    # Model selection
    parser.add_argument('--model', type=str, default='baseline',
                       choices=['baseline', 'akvmn', 'coral', 'hybrid_coral'],
                       help='Model type to train')
    
    # Data arguments
    parser.add_argument('--dataset', type=str, default='synthetic_OC',
                       help='Dataset name')
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Directory containing data files')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--n_folds', type=int, default=0,
                       help='Number of CV folds (0 = no CV)')
    
    # Loss function arguments
    parser.add_argument('--loss', type=str, default='ce',
                       choices=['ce', 'qwk', 'emd', 'ordinal_ce', 'coral', 'combined'],
                       help='Loss function type')
    parser.add_argument('--ce_weight', type=float, default=1.0,
                       help='Weight for CE loss in combined loss')
    parser.add_argument('--qwk_weight', type=float, default=0.5,
                       help='Weight for QWK loss in combined loss')
    parser.add_argument('--emd_weight', type=float, default=0.0,
                       help='Weight for EMD loss in combined loss')
    parser.add_argument('--coral_weight', type=float, default=0.0,
                       help='Weight for CORAL loss in combined loss')
    
    # CORAL-specific arguments
    parser.add_argument('--coral_hidden_dim', type=int, default=None,
                       help='Hidden dimension for CORAL layer')
    parser.add_argument('--coral_dropout', type=float, default=0.1,
                       help='Dropout rate for CORAL layer')
    parser.add_argument('--blend_weight', type=float, default=0.5,
                       help='Blend weight for hybrid CORAL model')
    
    # Other arguments
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                       help='Device to use')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Ensure directories exist
    ensure_results_dirs()
    
    # Load data
    train_path = os.path.join(args.data_dir, args.dataset, f"{args.dataset.lower()}_train.txt")
    test_path = os.path.join(args.data_dir, args.dataset, f"{args.dataset.lower()}_test.txt")
    
    train_data, test_data, n_questions, n_cats = load_simple_data(train_path, test_path)
    
    # Create model
    model_kwargs = {
        'coral_hidden_dim': args.coral_hidden_dim,
        'coral_dropout': args.coral_dropout,
        'blend_weight': args.blend_weight
    }
    
    model = create_model(args.model, n_questions, n_cats, **model_kwargs)
    model = model.to(args.device)
    
    # Check if it's a CORAL model
    is_coral_model = isinstance(model, (CORALDeepGPCM, HybridCORALGPCM))
    
    # Create loss function
    if is_coral_model and args.loss == 'ce':
        # Automatically use CORAL loss for CORAL models if CE is selected
        print("Note: Using CORAL-compatible loss for CORAL model")
        args.loss = 'coral'
    
    loss_kwargs = {
        'ce_weight': args.ce_weight,
        'qwk_weight': args.qwk_weight,
        'emd_weight': args.emd_weight,
        'coral_weight': args.coral_weight if is_coral_model else 0.0
    }
    
    criterion = create_loss_function(args.loss, n_cats, **loss_kwargs)
    
    # Create data loaders
    train_dataset = list(zip(train_data, [None] * len(train_data)))
    train_dataset = [(q[0], q[1]) for (q, _), _ in train_dataset]
    
    test_dataset = list(zip(test_data, [None] * len(test_data)))
    test_dataset = [(q[0], q[1]) for (q, _), _ in test_dataset]
    
    train_loader = data_utils.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True,
        collate_fn=collate_fn
    )
    
    test_loader = data_utils.DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=3)
    
    # Training loop
    print(f"\nTraining {args.model} model with {args.loss} loss")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model type: {type(model).__name__}")
    print("Epoch | Train Loss | Train Acc | Test Acc | QWK | Ord.Acc | MAE | Time(s)")
    print("-" * 85)
    
    best_qwk = -1.0
    best_epoch = 0
    training_history = []
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, args.device, is_coral_model
        )
        
        # Evaluate
        test_metrics = evaluate(model, test_loader, args.device, n_cats)
        
        # Update scheduler
        scheduler.step(test_metrics['quadratic_weighted_kappa'])
        
        # Save best model
        if test_metrics['quadratic_weighted_kappa'] > best_qwk:
            best_qwk = test_metrics['quadratic_weighted_kappa']
            best_epoch = epoch
            
            # Save model
            save_path = f"save_models/best_{args.model}_{args.dataset}_coral.pth"
            torch.save({
                'model_state_dict': model.state_dict(),
                'model_config': {
                    'model_type': args.model,
                    'n_questions': n_questions,
                    'n_cats': n_cats,
                    'loss_type': args.loss
                },
                'best_qwk': best_qwk,
                'epoch': epoch
            }, save_path)
        
        # Record history
        epoch_results = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            **test_metrics,
            'lr': optimizer.param_groups[0]['lr']
        }
        training_history.append(epoch_results)
        
        # Print progress
        elapsed = time.time() - start_time
        print(f"{epoch:5d} | {train_loss:10.4f} | {train_acc:9.4f} | "
              f"{test_metrics['categorical_accuracy']:8.4f} | "
              f"{test_metrics['quadratic_weighted_kappa']:6.4f} | "
              f"{test_metrics['ordinal_accuracy']:7.4f} | "
              f"{test_metrics['mean_absolute_error']:5.3f} | "
              f"{elapsed:7.1f}")
    
    print(f"\nBest QWK: {best_qwk:.4f} at epoch {best_epoch}")
    
    # Save training history
    save_results(
        training_history, 
        f'train_results_{args.model}_{args.dataset}_coral',
        'train'
    )
    
    print("\nTraining completed!")


if __name__ == "__main__":
    main()