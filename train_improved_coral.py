#!/usr/bin/env python3
"""
Training Script for Improved CORAL Integration with Proper Loss Function

Key improvements:
1. Uses IRT parameters → CORAL instead of bypassing GPCM framework
2. Uses CORN loss for proper CORAL threshold learning
3. Maintains educational interpretability through IRT parameters
"""

import os
import sys
import torch
import torch.optim as optim
import numpy as np
import json
import logging
import argparse
from datetime import datetime
from tqdm import tqdm

# Add path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.model import DeepGpcmModel
from models.improved_coral_integration import (
    create_improved_coral_enhanced_model, 
    ImprovedCoralTrainer
)
from evaluation.metrics import GpcmMetrics
from utils.gpcm_utils import load_gpcm_data, create_gpcm_batch, OrdinalLoss
from train import GpcmDataLoader, setup_logging


def evaluate_improved_coral_model(model, data_loader, trainer, device, n_cats, use_coral=True):
    """Enhanced evaluation for improved CORAL model."""
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for q_batch, r_batch, mask_batch in data_loader:
            q_batch = q_batch.to(device)
            r_batch = r_batch.to(device)
            mask_batch = mask_batch.to(device)
            
            # Forward pass with improved CORAL
            _, _, _, probs, coral_logits = model(q_batch, r_batch, use_coral=use_coral)
            
            # Compute loss only on valid positions
            if mask_batch is not None:
                valid_probs = probs[mask_batch]
                valid_targets = r_batch[mask_batch]
                if coral_logits is not None:
                    valid_coral_logits = coral_logits[mask_batch]
                else:
                    valid_coral_logits = None
            else:
                valid_probs = probs.view(-1, n_cats)
                valid_targets = r_batch.view(-1)
                if coral_logits is not None:
                    valid_coral_logits = coral_logits.view(-1, n_cats - 1)
                else:
                    valid_coral_logits = None
            
            # Use appropriate loss function
            loss = trainer.compute_loss(
                valid_probs.unsqueeze(1), 
                valid_targets.unsqueeze(1),
                valid_coral_logits.unsqueeze(1) if valid_coral_logits is not None else None,
                use_coral=use_coral
            )
            total_loss += loss.item()
            
            # Store for metrics
            all_probs.append(valid_probs.cpu())
            all_targets.append(valid_targets.cpu())
    
    # Compute metrics
    all_probs = torch.cat(all_probs, dim=0).unsqueeze(1)  # Add seq_len dim
    all_targets = torch.cat(all_targets, dim=0).unsqueeze(1)
    
    metrics = GpcmMetrics()
    prediction_method = 'cumulative'
    
    results = {
        'loss': total_loss / len(data_loader),
        'categorical_acc': metrics.categorical_accuracy(all_probs, all_targets, method=prediction_method),
        'ordinal_acc': metrics.ordinal_accuracy(all_probs, all_targets, method=prediction_method),
        'mae': metrics.mean_absolute_error(all_probs, all_targets, method=prediction_method),
        'qwk': metrics.quadratic_weighted_kappa(all_probs, all_targets, n_cats, method=prediction_method),
        'prediction_consistency_acc': metrics.prediction_consistency_accuracy(all_probs, all_targets, method=prediction_method),
        'ordinal_ranking_acc': metrics.ordinal_ranking_accuracy(all_probs, all_targets),
        'distribution_consistency': metrics.distribution_consistency_score(all_probs, all_targets)
    }
    
    # Skip per-category accuracy to avoid indexing issues
    # per_cat_acc = metrics.per_category_accuracy(all_probs, all_targets, n_cats, method=prediction_method)
    # results.update(per_cat_acc)
    
    return results


def train_epoch_improved_coral(model, train_loader, optimizer, trainer, device, n_cats, use_coral=True):
    """Enhanced training epoch for improved CORAL model."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for q_batch, r_batch, mask_batch in tqdm(train_loader, desc="Training"):
        q_batch = q_batch.to(device)
        r_batch = r_batch.to(device)
        mask_batch = mask_batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass with improved CORAL
        _, _, _, probs, coral_logits = model(q_batch, r_batch, use_coral=use_coral)
        
        # Compute loss only on valid positions
        if mask_batch is not None:
            valid_probs = probs[mask_batch]
            valid_targets = r_batch[mask_batch]
            if coral_logits is not None:
                valid_coral_logits = coral_logits[mask_batch]
            else:
                valid_coral_logits = None
        else:
            valid_probs = probs.view(-1, n_cats)
            valid_targets = r_batch.view(-1)
            if coral_logits is not None:
                valid_coral_logits = coral_logits.view(-1, n_cats - 1)
            else:
                valid_coral_logits = None
        
        # Use appropriate loss function
        loss = trainer.compute_loss(
            valid_probs.unsqueeze(1), 
            valid_targets.unsqueeze(1),
            valid_coral_logits.unsqueeze(1) if valid_coral_logits is not None else None,
            use_coral=use_coral
        )
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def train_improved_coral_model(dataset_name, n_epochs=15, batch_size=64, learning_rate=0.001,
                               embedding_strategy='linear_decay', device=None):
    """
    Train improved CORAL-enhanced model with proper loss function.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print(f"TRAINING IMPROVED CORAL MODEL")
    print(f"Dataset: {dataset_name}")
    print(f"Embedding Strategy: {embedding_strategy}")
    print(f"Epochs: {n_epochs}")
    print(f"Using CORN Loss for CORAL threshold learning")
    print(f"{'='*80}")
    
    # Load data
    train_path = f"data/{dataset_name}/synthetic_oc_train.txt"
    test_path = f"data/{dataset_name}/synthetic_oc_test.txt"
    
    print("Loading data...")
    train_seqs, train_questions, train_responses, n_cats = load_gpcm_data(train_path)
    test_seqs, test_questions, test_responses, _ = load_gpcm_data(test_path, n_cats)
    
    # Create data loaders
    train_loader = GpcmDataLoader(train_questions, train_responses, batch_size, shuffle=True)
    test_loader = GpcmDataLoader(test_questions, test_responses, batch_size, shuffle=False)
    
    # Determine n_questions
    all_questions = []
    for q_seq in train_questions + test_questions:
        all_questions.extend(q_seq)
    n_questions = max(all_questions)
    
    print(f"Data loaded: {len(train_seqs)} train, {len(test_seqs)} test")
    print(f"Questions: {n_questions}, Categories: {n_cats}")
    
    # Create base model
    base_model = DeepGpcmModel(
        n_questions=n_questions,
        n_cats=n_cats,
        memory_size=50,
        key_dim=50,
        value_dim=200,
        final_fc_dim=50,
        embedding_strategy=embedding_strategy
    ).to(device)
    
    # Create improved CORAL model
    model = create_improved_coral_enhanced_model(base_model)
    model = model.to(device)
    
    print(f"Created improved CORAL-enhanced model")
    print(f"Architecture: IRT parameters → CORAL features → Rank-consistent probabilities")
    
    # Create trainer with proper loss functions
    trainer = ImprovedCoralTrainer(model, n_cats, device)
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.7)
    
    # Training tracking
    training_history = []
    best_valid_acc = 0.0
    
    print(f"\nStarting training with improved CORAL integration...")
    
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        
        # Train
        train_loss = train_epoch_improved_coral(model, train_loader, optimizer, trainer, device, n_cats, use_coral=True)
        
        # Evaluate
        train_metrics = evaluate_improved_coral_model(model, train_loader, trainer, device, n_cats, use_coral=True)
        test_metrics = evaluate_improved_coral_model(model, test_loader, trainer, device, n_cats, use_coral=True)
        
        # Learning rate scheduling
        scheduler.step(test_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log progress
        print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_metrics['loss']:.4f}")
        print(f"Test Metrics - Cat Acc: {test_metrics['categorical_acc']:.3f}, "
              f"Ord Acc: {test_metrics['ordinal_acc']:.3f}, "
              f"Pred Cons: {test_metrics['prediction_consistency_acc']:.3f}, "
              f"MAE: {test_metrics['mae']:.3f}")
        
        # Save best model
        if test_metrics['categorical_acc'] > best_valid_acc:
            best_valid_acc = test_metrics['categorical_acc']
            model_filename = f"best_model_{dataset_name}_improved_coral_{embedding_strategy}.pth"
            model_path = os.path.join("save_models", model_filename)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_valid_acc': best_valid_acc,
                'n_cats': n_cats,
                'n_questions': n_questions,
                'embedding_strategy': embedding_strategy,
                'model_type': 'improved_coral'
            }, model_path)
            print(f"Saved best model: {model_filename}")
        
        # Store training history
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'valid_loss': test_metrics['loss'],
            'learning_rate': current_lr,
            'embedding_strategy': embedding_strategy,
            'model_type': 'improved_coral',
            **train_metrics,
            **{f'valid_{k}': v for k, v in test_metrics.items()}
        }
        training_history.append(epoch_data)
    
    # Save training history
    os.makedirs("results/train", exist_ok=True)
    history_filename = f"training_history_{dataset_name}_improved_coral_{embedding_strategy}.json"
    history_path = os.path.join("results/train", history_filename)
    
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_valid_acc:.3f}")
    print(f"Training history saved: {history_filename}")
    
    return model, training_history


def compare_coral_implementations(dataset_name="synthetic_OC", epochs=15):
    """
    Compare original CORAL vs improved CORAL implementation.
    """
    print(f"\n{'='*100}")
    print(f"COMPARING CORAL IMPLEMENTATIONS")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*100}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    embedding_strategies = ['linear_decay', 'adjacent_weighted']
    results = {}
    
    for strategy in embedding_strategies:
        print(f"\n>>> Testing {strategy} embedding <<<")
        
        try:
            # Train improved CORAL model
            model, history = train_improved_coral_model(
                dataset_name=dataset_name,
                n_epochs=epochs,
                embedding_strategy=strategy,
                device=device
            )
            
            results[f"improved_coral_{strategy}"] = {
                'final_metrics': history[-1],
                'best_epoch': max(history, key=lambda x: x['categorical_acc']),
                'model_type': 'improved_coral',
                'embedding_strategy': strategy
            }
            
            print(f"✅ Improved CORAL with {strategy} completed successfully")
            
        except Exception as e:
            print(f"❌ Error training improved CORAL with {strategy}: {e}")
            continue
    
    # Print comparison results
    print(f"\n{'='*80}")
    print(f"IMPROVED CORAL RESULTS SUMMARY")
    print(f"{'='*80}")
    
    print(f"\nFinal Performance:")
    print(f"{'Strategy':<20} {'Cat_Acc':<8} {'Ord_Acc':<8} {'Pred_Cons':<10} {'MAE':<8}")
    print("-" * 60)
    
    for key, result in results.items():
        final = result['final_metrics']
        print(f"{key:<20} {final['categorical_acc']:<8.3f} {final['ordinal_acc']:<8.3f} "
              f"{final['prediction_consistency_acc']:<10.3f} {final['mae']:<8.3f}")
    
    return results


def main():
    """Main function for improved CORAL training."""
    parser = argparse.ArgumentParser(description='Train improved CORAL-enhanced models')
    parser.add_argument('--dataset', type=str, default='synthetic_OC', help='Dataset name')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    parser.add_argument('--strategy', type=str, default='linear_decay', 
                       choices=['ordered', 'unordered', 'linear_decay', 'adjacent_weighted'],
                       help='Embedding strategy')
    parser.add_argument('--compare', action='store_true', help='Compare implementations')
    
    args = parser.parse_args()
    
    if args.compare:
        # Run comparison
        compare_coral_implementations(args.dataset, args.epochs)
    else:
        # Train single model
        train_improved_coral_model(
            dataset_name=args.dataset,
            n_epochs=args.epochs,
            embedding_strategy=args.strategy
        )


if __name__ == "__main__":
    main()