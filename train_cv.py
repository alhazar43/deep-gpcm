#!/usr/bin/env python3
"""
Cross-Validation Training Script for Deep-GPCM Model

Implements 5-fold cross-validation following deep-2pl patterns.
"""

import os
import torch
import torch.optim as optim
import numpy as np
import json
import logging
import argparse
from datetime import datetime
from tqdm import tqdm
from sklearn.model_selection import KFold

from models.model import DeepGpcmModel
from utils.gpcm_utils import (
    OrdinalLoss, GpcmMetrics, load_gpcm_data, create_gpcm_batch,
    CrossEntropyLossWrapper, MSELossWrapper
)
from train import GpcmDataLoader, train_epoch, evaluate_model, setup_logging


def create_cv_folds(questions, responses, n_folds=5, random_state=42):
    """Create cross-validation folds."""
    n_samples = len(questions)
    indices = np.arange(n_samples)
    
    kfold = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    folds = []
    
    for train_idx, valid_idx in kfold.split(indices):
        train_questions = [questions[i] for i in train_idx]
        train_responses = [responses[i] for i in train_idx]
        valid_questions = [questions[i] for i in valid_idx]
        valid_responses = [responses[i] for i in valid_idx]
        
        folds.append({
            'train_questions': train_questions,
            'train_responses': train_responses,
            'valid_questions': valid_questions,
            'valid_responses': valid_responses,
            'train_idx': train_idx.tolist(),
            'valid_idx': valid_idx.tolist()
        })
    
    return folds


def train_single_fold(fold_data, fold_idx, n_questions, n_cats, n_epochs, batch_size, 
                     learning_rate, loss_type, memory_size, device, logger):
    """Train model on a single fold."""
    
    logger.info(f"Training fold {fold_idx + 1}/5")
    
    # Create data loaders
    train_loader = GpcmDataLoader(
        fold_data['train_questions'], 
        fold_data['train_responses'], 
        batch_size, 
        shuffle=True
    )
    valid_loader = GpcmDataLoader(
        fold_data['valid_questions'], 
        fold_data['valid_responses'], 
        batch_size, 
        shuffle=False
    )
    
    # Create model
    model = DeepGpcmModel(
        n_questions=n_questions,
        n_cats=n_cats,
        memory_size=memory_size,
        key_dim=50,
        value_dim=200,
        final_fc_dim=50
    ).to(device)
    
    # Create loss function
    if loss_type == 'ordinal':
        loss_fn = OrdinalLoss(n_cats)
    elif loss_type == 'crossentropy':
        loss_fn = CrossEntropyLossWrapper()
    elif loss_type == 'mse':
        loss_fn = MSELossWrapper()
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")
    
    # Create optimizer and scheduler
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Metrics
    metrics = GpcmMetrics()
    
    # Training loop
    best_valid_acc = 0.0
    fold_history = []
    
    for epoch in range(n_epochs):
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, n_cats)
        
        # Validation
        valid_results = evaluate_model(model, valid_loader, loss_fn, metrics, device, n_cats)
        
        # Update learning rate
        scheduler.step(valid_results['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log progress (less verbose during CV)
        if epoch % 5 == 0 or epoch == n_epochs - 1:
            logger.info(f"  Fold {fold_idx + 1}, Epoch {epoch + 1}: "
                       f"Train Loss: {train_loss:.4f}, "
                       f"Valid Acc: {valid_results['categorical_acc']:.4f}")
        
        # Save fold history
        epoch_data = {
            'fold': fold_idx + 1,
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'valid_loss': valid_results['loss'],
            'categorical_acc': valid_results['categorical_acc'],
            'ordinal_acc': valid_results['ordinal_acc'],
            'mae': valid_results['mae'],
            'qwk': valid_results['qwk'],
            'learning_rate': current_lr
        }
        fold_history.append(epoch_data)
        
        # Track best model for this fold
        if valid_results['categorical_acc'] > best_valid_acc:
            best_valid_acc = valid_results['categorical_acc']
    
    # Final validation results for this fold
    final_results = evaluate_model(model, valid_loader, loss_fn, metrics, device, n_cats)
    
    fold_results = {
        'fold': fold_idx + 1,
        'best_valid_acc': best_valid_acc,
        'final_results': final_results,
        'history': fold_history
    }
    
    return fold_results, model


def train_cross_validation(dataset_name, n_folds=5, n_epochs=30, batch_size=64, 
                          learning_rate=0.001, loss_type='ordinal', n_cats=None, 
                          memory_size=50, device=None):
    """Main cross-validation training function."""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger = setup_logging(dataset_name)
    logger.info(f"Starting {n_folds}-fold CV training on {dataset_name}")
    logger.info(f"Device: {device}, Categories: {n_cats}, Loss: {loss_type}")
    
    # Create directories
    os.makedirs("save_models", exist_ok=True)
    os.makedirs("results/cv", exist_ok=True)
    
    # Load all data for cross-validation
    train_path = f"data/{dataset_name}/synthetic_oc_train.txt"
    test_path = f"data/{dataset_name}/synthetic_oc_test.txt"
    
    logger.info("Loading data...")
    train_seqs, train_questions, train_responses, detected_cats = load_gpcm_data(train_path)
    test_seqs, test_questions, test_responses, _ = load_gpcm_data(test_path, detected_cats)
    
    if n_cats is None:
        n_cats = detected_cats
    
    logger.info(f"Train data: {len(train_seqs)} sequences")
    logger.info(f"Test data: {len(test_seqs)} sequences")
    logger.info(f"Detected {n_cats} categories")
    
    # Combine train and test for CV (following research practice)
    all_questions = train_questions + test_questions
    all_responses = train_responses + test_responses
    
    # Determine n_questions
    flat_questions = []
    for q_seq in all_questions:
        flat_questions.extend(q_seq)
    n_questions = max(flat_questions)
    
    logger.info(f"Total sequences for CV: {len(all_questions)}")
    logger.info(f"Unique questions: {n_questions}")
    
    # Create CV folds
    logger.info("Creating cross-validation folds...")
    cv_folds = create_cv_folds(all_questions, all_responses, n_folds)
    
    # Train each fold
    all_fold_results = []
    all_models = []
    
    for fold_idx, fold_data in enumerate(cv_folds):
        logger.info(f"\nFold {fold_idx + 1}/{n_folds}")
        logger.info(f"  Train: {len(fold_data['train_questions'])} sequences")
        logger.info(f"  Valid: {len(fold_data['valid_questions'])} sequences")
        
        fold_results, model = train_single_fold(
            fold_data, fold_idx, n_questions, n_cats, n_epochs, batch_size,
            learning_rate, loss_type, memory_size, device, logger
        )
        
        all_fold_results.append(fold_results)
        all_models.append(model)
        
        logger.info(f"  Fold {fold_idx + 1} completed: "
                   f"Best Acc: {fold_results['best_valid_acc']:.4f}")
    
    # Compute cross-validation statistics
    cv_stats = compute_cv_statistics(all_fold_results, logger)
    
    # Save results
    cv_results = {
        'dataset': dataset_name,
        'n_folds': n_folds,
        'n_epochs': n_epochs,
        'hyperparameters': {
            'batch_size': batch_size,
            'learning_rate': learning_rate,
            'loss_type': loss_type,
            'n_cats': n_cats,
            'memory_size': memory_size
        },
        'fold_results': all_fold_results,
        'cv_statistics': cv_stats,
        'timestamp': datetime.now().isoformat()
    }
    
    results_path = f"results/cv/cv_results_{dataset_name}_{n_folds}fold.json"
    with open(results_path, 'w') as f:
        json.dump(cv_results, f, indent=2)
    
    logger.info(f"\nCross-validation results saved to: {results_path}")
    
    return cv_results


def compute_cv_statistics(fold_results, logger):
    """Compute cross-validation statistics."""
    metrics = ['categorical_acc', 'ordinal_acc', 'mae', 'qwk']
    cv_stats = {}
    
    for metric in metrics:
        values = [fold['final_results'][metric] for fold in fold_results]
        cv_stats[metric] = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'values': values
        }
    
    # Print summary
    logger.info("\n" + "="*50)
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info("="*50)
    
    for metric in metrics:
        stats = cv_stats[metric]
        logger.info(f"{metric.replace('_', ' ').title():<20}: "
                   f"{stats['mean']:.4f} ± {stats['std']:.4f} "
                   f"(range: {stats['min']:.4f}-{stats['max']:.4f})")
    
    return cv_stats


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Cross-validation training for Deep-GPCM')
    parser.add_argument('--dataset', type=str, default='synthetic_OC',
                        help='Dataset name (default: synthetic_OC)')
    parser.add_argument('--n_folds', type=int, default=5,
                        help='Number of CV folds (default: 5)')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of epochs (default: 30)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size (default: 64)')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate (default: 0.001)')
    parser.add_argument('--loss_type', type=str, default='ordinal',
                        choices=['ordinal', 'crossentropy', 'mse'],
                        help='Loss function type (default: ordinal)')
    parser.add_argument('--n_cats', type=int, default=None,
                        help='Number of categories (auto-detect if None)')
    parser.add_argument('--memory_size', type=int, default=50,
                        help='Memory size for DKVMN (default: 50)')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Run cross-validation
    results = train_cross_validation(
        dataset_name=args.dataset,
        n_folds=args.n_folds,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        loss_type=args.loss_type,
        n_cats=args.n_cats,
        memory_size=args.memory_size
    )
    
    return results


if __name__ == "__main__":
    main()