"""
Unified Training Script for Deep-GPCM
Supports baseline and AKVMN models with unified interface.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import argparse
import json
import time
from datetime import datetime
from typing import Dict, Any, Tuple

from config import get_model_config, get_preset_configs
from model_factory import create_model, print_model_summary
from utils.gpcm_utils import load_gpcm_data
from utils.data_utils import UnifiedDataLoader
from utils.loss_utils import create_loss_function
from evaluation.metrics import GpcmMetrics


def train_epoch(model: nn.Module, train_loader: UnifiedDataLoader, 
                optimizer: torch.optim.Optimizer, loss_fn: nn.Module,
                device: torch.device, n_cats: int) -> Tuple[float, float]:
    """Train model for one epoch."""
    model.train()
    total_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    
    for q_batch, r_batch, mask_batch in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        _, _, _, gpcm_probs = model(q_batch, r_batch)
        
        # Compute loss
        if mask_batch is not None:
            valid_probs = gpcm_probs[mask_batch]
            valid_targets = r_batch[mask_batch]
        else:
            valid_probs = gpcm_probs.view(-1, n_cats)
            valid_targets = r_batch.view(-1)
        
        if hasattr(loss_fn, '__class__') and 'Ordinal' in loss_fn.__class__.__name__:
            loss = loss_fn(valid_probs, valid_targets)
        else:
            loss = loss_fn(valid_probs.unsqueeze(1), valid_targets.unsqueeze(1))
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        predictions = torch.argmax(valid_probs, dim=-1)
        correct_predictions += (predictions == valid_targets).sum().item()
        total_predictions += valid_targets.size(0)
    
    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / total_predictions
    
    return avg_loss, accuracy


def evaluate_model(model: nn.Module, test_loader: UnifiedDataLoader,
                  loss_fn: nn.Module, metrics: GpcmMetrics,
                  device: torch.device, n_cats: int) -> Dict[str, float]:
    """Evaluate model performance."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    inference_times = []
    
    with torch.no_grad():
        for q_batch, r_batch, mask_batch in test_loader:
            # Time inference
            start_time = time.time()
            _, _, _, gpcm_probs = model(q_batch, r_batch)
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Compute loss
            if mask_batch is not None:
                valid_probs = gpcm_probs[mask_batch]
                valid_targets = r_batch[mask_batch]
            else:
                valid_probs = gpcm_probs.view(-1, n_cats)
                valid_targets = r_batch.view(-1)
            
            if hasattr(loss_fn, '__class__') and 'Ordinal' in loss_fn.__class__.__name__:
                loss = loss_fn(valid_probs, valid_targets)
            else:
                loss = loss_fn(valid_probs.unsqueeze(1), valid_targets.unsqueeze(1))
            
            total_loss += loss.item()
            
            # Collect predictions and targets
            all_predictions.append(valid_probs.cpu())
            all_targets.extend(valid_targets.cpu().numpy())
    
    # Compute comprehensive metrics
    all_probs = torch.cat(all_predictions, dim=0)
    targets_tensor = torch.tensor(all_targets, dtype=torch.long)
    
    eval_metrics = metrics.benchmark_prediction_methods(
        all_probs, targets_tensor, n_cats
    )
    
    eval_metrics.update({
        'test_loss': total_loss / len(test_loader),
        'avg_inference_time': np.mean(inference_times)
    })
    
    return eval_metrics


def save_model_and_results(model: nn.Module, config: Any, results: Dict[str, Any],
                          save_dir: str, dataset_name: str):
    """Save model checkpoint and training results."""
    os.makedirs(save_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(save_dir, f"best_{config.model_type}_{dataset_name}.pth")
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config.__dict__,
        'results': results
    }, model_path)
    
    # Save results
    results_path = os.path.join(save_dir, f"results_{config.model_type}_{dataset_name}.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    return model_path, results_path


def train_model(config, dataset_name: str, device: torch.device) -> Dict[str, Any]:
    """
    Train a model with given configuration.
    
    Args:
        config: Model configuration
        dataset_name: Name of the dataset
        device: Computing device
        
    Returns:
        Training results dictionary
    """
    print(f"\n=== Training {config.model_type.upper()} on {dataset_name} ===")
    
    # Load data
    train_path = f"data/{dataset_name}/{dataset_name.lower()}_train.txt"
    test_path = f"data/{dataset_name}/{dataset_name.lower()}_test.txt"
    
    train_seqs, train_questions, train_responses, n_cats = load_gpcm_data(train_path)
    test_seqs, test_questions, test_responses, _ = load_gpcm_data(test_path, n_cats)
    
    # Determine n_questions
    all_questions = []
    for q_seq in train_questions + test_questions:
        all_questions.extend(q_seq)
    n_questions = max(all_questions)
    
    # Update config
    config.n_cats = n_cats
    config.n_questions = n_questions
    
    print(f"Data: {len(train_seqs)} train, {len(test_seqs)} test")
    print(f"Questions: {n_questions}, Categories: {n_cats}")
    
    # Create model
    model = create_model(config, n_questions, device)
    print_model_summary(model, config)
    
    # Create data loaders
    train_loader = UnifiedDataLoader(train_questions, train_responses, 
                                   config.batch_size, shuffle=True, device=device)
    test_loader = UnifiedDataLoader(test_questions, test_responses,
                                  config.batch_size, shuffle=False, device=device)
    
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    loss_fn = create_loss_function(config.loss_type, n_cats)
    metrics = GpcmMetrics()
    
    # Training loop
    best_accuracy = 0.0
    training_history = []
    
    print(f"\nStarting training for {config.epochs} epochs...")
    
    for epoch in range(config.epochs):
        # Training
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, loss_fn, device, n_cats
        )
        
        # Evaluation
        eval_results = evaluate_model(
            model, test_loader, loss_fn, metrics, device, n_cats
        )
        
        # Track best model
        current_accuracy = eval_results['argmax']['categorical_accuracy']
        if current_accuracy > best_accuracy:
            best_accuracy = current_accuracy
            best_results = eval_results.copy()
        
        # Store epoch results
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            **eval_results
        }
        training_history.append(epoch_data)
        
        # Print progress
        if epoch % 5 == 0 or epoch == config.epochs - 1:
            print(f"Epoch {epoch+1:3d}/{config.epochs}: "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Test Acc: {current_accuracy:.4f}")
    
    # Compile final results
    final_results = {
        'model_type': config.model_type,
        'dataset': dataset_name,
        'config': config.__dict__,
        'training_history': training_history,
        'best_metrics': best_results,
        'final_accuracy': best_accuracy
    }
    
    # Save model and results
    model_path, results_path = save_model_and_results(
        model, config, final_results, config.save_dir, dataset_name
    )
    
    print(f"\n✅ Training completed!")
    print(f"Best accuracy: {best_accuracy:.4f}")
    print(f"Model saved: {model_path}")
    print(f"Results saved: {results_path}")
    
    return final_results


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Unified Deep-GPCM Training')
    parser.add_argument('--model', type=str, default='baseline',
                        choices=['baseline', 'akvmn'],
                        help='Model type to train')
    parser.add_argument('--dataset', type=str, default='synthetic_OC',
                        help='Dataset name')
    parser.add_argument('--epochs', type=int, default=30,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--device', type=str, default='auto',
                        help='Device to use (auto, cpu, cuda)')
    parser.add_argument('--config', type=str, default=None,
                        help='Path to custom config file')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Load or create configuration
    if args.config:
        from config import load_config
        config = load_config(args.config)
    else:
        preset_configs = get_preset_configs()
        if args.model in preset_configs:
            config = preset_configs[args.model]
        else:
            config = get_model_config(args.model)
    
    # Override config with command line arguments
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.learning_rate
    config.dataset_name = args.dataset
    
    # Train model
    results = train_model(config, args.dataset, device)
    
    print(f"\n🎯 Final Results Summary:")
    print(f"Model: {config.model_type.upper()}")
    print(f"Dataset: {args.dataset}")
    print(f"Final Accuracy: {results['final_accuracy']:.4f}")


if __name__ == "__main__":
    main()