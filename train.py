#!/usr/bin/env python3
"""
Training Script for Deep-GPCM Model

Supports GPCM training with ordinal loss and multi-category evaluation.
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

from models.model import DeepGpcmModel
from utils.gpcm_utils import (
    OrdinalLoss, GpcmMetrics, load_gpcm_data, create_gpcm_batch,
    CrossEntropyLossWrapper, MSELossWrapper
)


def setup_logging(dataset_name, fold_idx=None):
    """Setup logging configuration."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fold_str = f"_fold{fold_idx}" if fold_idx is not None else ""
    log_filename = f"logs/train_{dataset_name}{fold_str}_{timestamp}.log"
    
    os.makedirs("logs", exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    
    return logging.getLogger(__name__)


class GpcmDataLoader:
    """Simple data loader for GPCM data."""
    
    def __init__(self, questions, responses, batch_size=32, shuffle=True):
        self.questions = questions
        self.responses = responses
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.n_samples = len(questions)
        
        self.indices = list(range(self.n_samples))
        if self.shuffle:
            np.random.shuffle(self.indices)
        
        self.current_idx = 0
    
    def __iter__(self):
        if self.shuffle:
            np.random.shuffle(self.indices)
        self.current_idx = 0
        return self
    
    def __next__(self):
        if self.current_idx >= self.n_samples:
            raise StopIteration
        
        # Get batch indices
        end_idx = min(self.current_idx + self.batch_size, self.n_samples)
        batch_indices = self.indices[self.current_idx:end_idx]
        
        # Create batch
        batch_questions = [self.questions[i] for i in batch_indices]
        batch_responses = [self.responses[i] for i in batch_indices]
        
        q_batch, r_batch, mask_batch = create_gpcm_batch(batch_questions, batch_responses)
        
        self.current_idx = end_idx
        
        return q_batch, r_batch, mask_batch
    
    def __len__(self):
        return (self.n_samples + self.batch_size - 1) // self.batch_size


def evaluate_model(model, data_loader, loss_fn, metrics, device, n_cats):
    """Evaluate model on dataset."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for q_batch, r_batch, mask_batch in data_loader:
            q_batch = q_batch.to(device)
            r_batch = r_batch.to(device)
            mask_batch = mask_batch.to(device)
            
            # Forward pass
            _, _, _, _, gpcm_probs = model(q_batch, r_batch)
            
            # Compute loss only on valid positions
            if mask_batch is not None:
                valid_probs = gpcm_probs[mask_batch]
                valid_targets = r_batch[mask_batch]
            else:
                valid_probs = gpcm_probs.view(-1, n_cats)
                valid_targets = r_batch.view(-1)
            
            loss = loss_fn(valid_probs.unsqueeze(1), valid_targets.unsqueeze(1))
            total_loss += loss.item()
            
            # Store for metrics
            all_predictions.append(valid_probs.cpu())
            all_targets.append(valid_targets.cpu())
    
    # Compute metrics
    all_predictions = torch.cat(all_predictions, dim=0).unsqueeze(1)  # Add seq_len dim
    all_targets = torch.cat(all_targets, dim=0).unsqueeze(1)
    
    results = {
        'loss': total_loss / len(data_loader),
        'categorical_acc': metrics.categorical_accuracy(all_predictions, all_targets),
        'ordinal_acc': metrics.ordinal_accuracy(all_predictions, all_targets),
        'mae': metrics.mean_absolute_error(all_predictions, all_targets),
        'qwk': metrics.quadratic_weighted_kappa(all_predictions, all_targets, n_cats)
    }
    
    per_cat_acc = metrics.per_category_accuracy(all_predictions, all_targets, n_cats)
    results.update(per_cat_acc)
    
    return results


def train_epoch(model, train_loader, optimizer, loss_fn, device, n_cats):
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for q_batch, r_batch, mask_batch in tqdm(train_loader, desc="Training"):
        q_batch = q_batch.to(device)
        r_batch = r_batch.to(device) 
        mask_batch = mask_batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        _, _, _, _, gpcm_probs = model(q_batch, r_batch)
        
        # Compute loss only on valid positions
        if mask_batch is not None:
            valid_probs = gpcm_probs[mask_batch]
            valid_targets = r_batch[mask_batch]
        else:
            valid_probs = gpcm_probs.view(-1, n_cats)
            valid_targets = r_batch.view(-1)
        
        loss = loss_fn(valid_probs.unsqueeze(1), valid_targets.unsqueeze(1))
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def train_model(dataset_name, n_epochs=30, batch_size=64, learning_rate=0.001, 
                loss_type='ordinal', n_cats=4, memory_size=50, device=None,
                embedding_strategy='linear_decay'):
    """Main training function."""
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    logger = setup_logging(dataset_name)
    logger.info(f"Training Deep-GPCM on {dataset_name}")
    logger.info(f"Device: {device}, Categories: {n_cats}, Loss: {loss_type}, Embedding: {embedding_strategy}")
    
    # Create directories
    os.makedirs("save_models", exist_ok=True)
    os.makedirs("results/train", exist_ok=True)
    os.makedirs("results/valid", exist_ok=True)
    
    # Load data
    train_path = f"data/{dataset_name}/synthetic_oc_train.txt"
    test_path = f"data/{dataset_name}/synthetic_oc_test.txt"
    
    logger.info("Loading training data...")
    train_seqs, train_questions, train_responses, detected_cats = load_gpcm_data(train_path)
    if n_cats is None:
        n_cats = detected_cats
    logger.info(f"Train data: {len(train_seqs)} sequences, {n_cats} categories")
    
    logger.info("Loading test data...")
    test_seqs, test_questions, test_responses, _ = load_gpcm_data(test_path, n_cats)
    logger.info(f"Test data: {len(test_seqs)} sequences")
    
    # Create data loaders
    train_loader = GpcmDataLoader(train_questions, train_responses, batch_size, shuffle=True)
    test_loader = GpcmDataLoader(test_questions, test_responses, batch_size, shuffle=False)
    
    # Determine n_questions from data
    all_questions = []
    for q_seq in train_questions + test_questions:
        all_questions.extend(q_seq)
    n_questions = max(all_questions)
    
    logger.info(f"Detected {n_questions} unique questions")
    
    # Create model
    model = DeepGpcmModel(
        n_questions=n_questions,
        n_cats=n_cats,
        memory_size=memory_size,
        key_dim=50,
        value_dim=200,
        final_fc_dim=50,
        embedding_strategy=embedding_strategy
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
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Metrics
    metrics = GpcmMetrics()
    
    logger.info("Starting training...")
    
    best_valid_acc = 0.0
    training_history = []
    
    for epoch in range(n_epochs):
        logger.info(f"Epoch {epoch + 1}/{n_epochs}")
        
        # Training
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn, device, n_cats)
        
        # Validation (using test set for now)
        valid_results = evaluate_model(model, test_loader, loss_fn, metrics, device, n_cats)
        
        # Update learning rate
        scheduler.step(valid_results['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log results
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Valid Loss: {valid_results['loss']:.4f}, Cat Acc: {valid_results['categorical_acc']:.4f}")
        logger.info(f"Ord Acc: {valid_results['ordinal_acc']:.4f}, MAE: {valid_results['mae']:.4f}, QWK: {valid_results['qwk']:.4f}")
        logger.info(f"Learning Rate: {current_lr:.6f}")
        
        # Save training history
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'valid_loss': valid_results['loss'],
            'categorical_acc': valid_results['categorical_acc'],
            'ordinal_acc': valid_results['ordinal_acc'],
            'mae': valid_results['mae'],
            'qwk': valid_results['qwk'],
            'learning_rate': current_lr
        }
        training_history.append(epoch_data)
        
        # Save best model
        if valid_results['categorical_acc'] > best_valid_acc:
            best_valid_acc = valid_results['categorical_acc']
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_valid_acc': best_valid_acc,
                'n_cats': n_cats,
                'n_questions': n_questions
            }, f"save_models/best_model_{dataset_name}.pth")
            logger.info(f"New best model saved! Valid Acc: {best_valid_acc:.4f}")
    
    # Save training history
    with open(f"results/train/training_history_{dataset_name}.json", 'w') as f:
        json.dump(training_history, f, indent=2)
    
    logger.info(f"Training completed! Best valid accuracy: {best_valid_acc:.4f}")
    
    return training_history, best_valid_acc


def main():
    """Main function with command line interface."""
    parser = argparse.ArgumentParser(description='Train Deep-GPCM Model')
    parser.add_argument('--dataset', type=str, default='synthetic_OC',
                        help='Dataset name (default: synthetic_OC)')
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
    parser.add_argument('--embedding_strategy', type=str, default='linear_decay',
                        choices=['ordered', 'unordered', 'linear_decay'],
                        help='Embedding strategy (default: linear_decay)')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Train model
    train_model(
        dataset_name=args.dataset,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        loss_type=args.loss_type,
        n_cats=args.n_cats,
        memory_size=args.memory_size,
        embedding_strategy=args.embedding_strategy
    )


if __name__ == "__main__":
    main()