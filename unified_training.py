#!/usr/bin/env python3
"""
Unified Training Script for Baseline vs AKVMN Comparison
Ensures both models use identical training pipeline, loss, metrics, and output format.
"""

import os
import torch
import torch.nn as nn
import numpy as np
import json
import time
from datetime import datetime
from typing import Dict, Any, Tuple, List

from config import get_model_config, get_preset_configs
from model_factory import create_model, print_model_summary
from utils.gpcm_utils import load_gpcm_data
from utils.data_utils import UnifiedDataLoader
from utils.loss_utils import create_loss_function
from evaluation.metrics import GpcmMetrics


def train_epoch(model: nn.Module, train_loader: UnifiedDataLoader, 
                optimizer: torch.optim.Optimizer, loss_fn: nn.Module,
                device: torch.device, n_cats: int) -> Tuple[float, float]:
    """Train model for one epoch with unified interface."""
    model.train()
    total_loss = 0.0
    total_uncertainty_loss = 0.0
    n_batches = 0
    
    for q_batch, r_batch, mask_batch in train_loader:
        optimizer.zero_grad()
        
        # Forward pass - both models return same format
        outputs = model(q_batch, r_batch)
        
        # Handle different model output formats
        if len(outputs) == 4:
            _, _, _, gpcm_probs = outputs
        else:
            gpcm_probs = outputs[-1]  # Last output is always probabilities
        
        # Compute loss with unified interface
        if mask_batch is not None:
            valid_probs = gpcm_probs[mask_batch]
            valid_targets = r_batch[mask_batch]
        else:
            valid_probs = gpcm_probs.view(-1, n_cats)
            valid_targets = r_batch.view(-1)
        
        # Unified loss computation
        loss = loss_fn(valid_probs, valid_targets)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches, total_uncertainty_loss / n_batches


def evaluate_epoch(model: nn.Module, val_loader: UnifiedDataLoader, 
                   loss_fn: nn.Module, device: torch.device, n_cats: int) -> Dict[str, float]:
    """Evaluate model for one epoch with unified metrics."""
    model.eval()
    total_loss = 0.0
    total_uncertainty_loss = 0.0
    all_probs = []
    all_targets = []
    n_batches = 0
    
    with torch.no_grad():
        for q_batch, r_batch, mask_batch in val_loader:
            # Forward pass
            outputs = model(q_batch, r_batch)
            
            # Handle different model output formats
            if len(outputs) == 4:
                _, _, _, gpcm_probs = outputs
            else:
                gpcm_probs = outputs[-1]  # Last output is always probabilities
            
            # Compute loss
            if mask_batch is not None:
                valid_probs = gpcm_probs[mask_batch]
                valid_targets = r_batch[mask_batch]
            else:
                valid_probs = gpcm_probs.view(-1, n_cats)
                valid_targets = r_batch.view(-1)
            
            loss = loss_fn(valid_probs, valid_targets)
            
            total_loss += loss.item()
            n_batches += 1
            
            # Collect for metrics computation
            all_probs.append(valid_probs.cpu())
            all_targets.append(valid_targets.cpu())
    
    # Concatenate all predictions and targets
    all_probs = torch.cat(all_probs, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Compute unified metrics using existing GpcmMetrics
    metrics = {}
    metrics['loss'] = total_loss / n_batches
    metrics['uncertainty_loss'] = total_uncertainty_loss / n_batches
    
    # Use argmax prediction method for consistency
    prediction_method = 'argmax'
    
    metrics['categorical_acc'] = GpcmMetrics.categorical_accuracy(all_probs, all_targets, method=prediction_method)
    metrics['ordinal_acc'] = GpcmMetrics.ordinal_accuracy(all_probs, all_targets, method=prediction_method)
    metrics['mae'] = GpcmMetrics.mean_absolute_error(all_probs, all_targets, method=prediction_method)
    metrics['qwk'] = GpcmMetrics.quadratic_weighted_kappa(all_probs, all_targets, n_cats, method=prediction_method)
    
    return metrics


def train_model_unified(model_config: Dict[str, Any], epochs: int = 30) -> Tuple[nn.Module, List[Dict[str, float]]]:
    """
    Train a model with unified pipeline.
    
    Args:
        model_config: Model configuration dictionary
        epochs: Number of training epochs
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    
    print(f"=== Training {model_config['model_type'].upper()} Model ===")
    
    # Device setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data with correct interface
    dataset_name = model_config['dataset_name']
    base_name = dataset_name.lower()
    train_path = f"data/{dataset_name}/{base_name}_train.txt"
    val_path = f"data/{dataset_name}/{base_name}_test.txt"  # Using test as validation for now
    
    # Load train data
    train_seqs, train_questions, train_responses, n_cats = load_gpcm_data(train_path)
    val_seqs, val_questions, val_responses, _ = load_gpcm_data(val_path, n_cats)
    
    # Update n_questions if not specified
    all_questions = []
    for q_seq in train_questions + val_questions:
        all_questions.extend(q_seq)
    
    if model_config.get('n_questions') is None:
        model_config['n_questions'] = max(all_questions) + 1
    
    # Update n_cats
    model_config['n_cats'] = n_cats
    
    # Create data loaders with correct format
    train_loader = UnifiedDataLoader(
        train_questions, train_responses, 
        batch_size=model_config['batch_size'], shuffle=True, device=device
    )
    val_loader = UnifiedDataLoader(
        val_questions, val_responses,
        batch_size=model_config['batch_size'], shuffle=False, device=device
    )
    
    # Create model with unified interface
    # Convert dict config to BaseConfig if needed
    if isinstance(model_config, dict):
        from config import get_model_config
        model_type = model_config.pop('model_type')  # Remove to avoid duplicate
        config_obj = get_model_config(model_type, **model_config)
        model_config['model_type'] = model_type  # Restore for later use
    else:
        config_obj = model_config
    
    model = create_model(config_obj, model_config['n_questions'], device)
    
    # Print model summary
    print(f"Model: {model_config['model_type']}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create unified loss function
    loss_fn = create_loss_function(model_config['loss_type'], model_config['n_cats'])
    loss_fn = loss_fn.to(device)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=model_config['learning_rate'])
    
    # Training loop with unified metrics
    training_history = []
    best_qwk = -1.0
    best_model_state = None
    
    print(f"Starting training for {epochs} epochs...")
    
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        
        # Train epoch
        train_loss, train_uncertainty_loss = train_epoch(
            model, train_loader, optimizer, loss_fn, device, model_config['n_cats']
        )
        
        # Validate epoch
        val_metrics = evaluate_epoch(
            model, val_loader, loss_fn, device, model_config['n_cats']
        )
        
        epoch_time = time.time() - start_time
        
        # Create unified epoch record
        epoch_record = {
            'epoch': epoch,
            'train_loss': train_loss,
            'train_uncertainty_loss': train_uncertainty_loss,
            'valid_loss': val_metrics['loss'],
            'categorical_acc': val_metrics['categorical_acc'],
            'ordinal_acc': val_metrics['ordinal_acc'],
            'mae': val_metrics['mae'],
            'qwk': val_metrics['qwk'],
            'learning_rate': model_config['learning_rate'],
            'epoch_time': epoch_time
        }
        
        training_history.append(epoch_record)
        
        # Save best model
        if val_metrics['qwk'] > best_qwk:
            best_qwk = val_metrics['qwk']
            best_model_state = model.state_dict().copy()
        
        # Print progress
        if epoch % 5 == 0 or epoch == 1:
            print(f"Epoch {epoch:2d}/{epochs}: "
                  f"Loss: {train_loss:.4f}, "
                  f"Cat Acc: {val_metrics['categorical_acc']:.3f}, "
                  f"Ord Acc: {val_metrics['ordinal_acc']:.3f}, "
                  f"QWK: {val_metrics['qwk']:.3f}, "
                  f"MAE: {val_metrics['mae']:.3f} "
                  f"({epoch_time:.1f}s)")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    print(f"Training completed. Best QWK: {best_qwk:.3f}")
    
    return model, training_history


def save_training_results(model_type: str, dataset_name: str, training_history: List[Dict], 
                         model: nn.Module, model_config: Dict[str, Any]) -> str:
    """Save training results with unified format."""
    
    # Create results directory
    os.makedirs("results/train", exist_ok=True)
    
    # Save training history with unified filename format
    history_file = f"results/train/training_history_{model_type}_{dataset_name}.json"
    
    with open(history_file, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    # Save model
    os.makedirs("save_models", exist_ok=True)
    model_file = f"save_models/best_{model_type}_{dataset_name}.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': model_config,
        'training_history': training_history
    }, model_file)
    
    print(f"✅ Saved {model_type} training history: {history_file}")
    print(f"✅ Saved {model_type} model: {model_file}")
    
    return history_file


def compare_models(epochs: int = 30, dataset_name: str = "synthetic_OC"):
    """
    Train both baseline and AKVMN models with identical pipeline and compare results.
    
    Args:
        epochs: Number of training epochs for both models
        dataset_name: Dataset to use for training
    """
    
    print("=== Unified Model Comparison Training ===")
    print(f"Epochs: {epochs}")
    print(f"Dataset: {dataset_name}")
    print(f"Unified loss: CrossEntropy")
    print(f"Unified metrics: categorical_acc, ordinal_acc, qwk, mae")
    
    # Get preset configurations
    configs = get_preset_configs()
    
    # Baseline configuration - convert to dict for modification
    baseline_config = {
        field.name: getattr(configs['baseline'], field.name)
        for field in configs['baseline'].__dataclass_fields__.values()
    }
    baseline_config['dataset_name'] = dataset_name
    baseline_config['epochs'] = epochs
    baseline_config['model_type'] = 'baseline'
    
    # AKVMN configuration - convert to dict for modification
    akvmn_config = {
        field.name: getattr(configs['akvmn'], field.name)
        for field in configs['akvmn'].__dataclass_fields__.values()
    }
    akvmn_config['dataset_name'] = dataset_name
    akvmn_config['epochs'] = epochs
    akvmn_config['model_type'] = 'akvmn'
    
    results = {}
    
    # Train baseline model
    print("\n" + "="*50)
    baseline_model, baseline_history = train_model_unified(baseline_config, epochs)
    baseline_file = save_training_results('baseline', dataset_name, baseline_history, baseline_model, baseline_config)
    results['baseline'] = {
        'history': baseline_history,
        'file': baseline_file,
        'final_metrics': baseline_history[-1]
    }
    
    # Train AKVMN model
    print("\n" + "="*50)
    akvmn_model, akvmn_history = train_model_unified(akvmn_config, epochs)
    akvmn_file = save_training_results('akvmn', dataset_name, akvmn_history, akvmn_model, akvmn_config)
    results['akvmn'] = {
        'history': akvmn_history,
        'file': akvmn_file,
        'final_metrics': akvmn_history[-1]
    }
    
    # Print comparison summary
    print("\n" + "="*50)
    print("=== Final Comparison Summary ===")
    
    for model_name, result in results.items():
        final = result['final_metrics']
        print(f"\n{model_name.upper()} Final Performance:")
        print(f"  Categorical Accuracy: {final['categorical_acc']:.3f}")
        print(f"  Ordinal Accuracy: {final['ordinal_acc']:.3f}")
        print(f"  QWK: {final['qwk']:.3f}")
        print(f"  MAE: {final['mae']:.3f}")
        print(f"  Training file: {result['file']}")
    
    # Calculate improvements
    if 'baseline' in results and 'akvmn' in results:
        baseline_final = results['baseline']['final_metrics']
        akvmn_final = results['akvmn']['final_metrics']
        
        print(f"\n🏆 AKVMN vs Baseline Improvements:")
        print(f"  Categorical Acc: {akvmn_final['categorical_acc'] - baseline_final['categorical_acc']:+.3f}")
        print(f"  Ordinal Acc: {akvmn_final['ordinal_acc'] - baseline_final['ordinal_acc']:+.3f}")
        print(f"  QWK: {akvmn_final['qwk'] - baseline_final['qwk']:+.3f}")
        print(f"  MAE: {akvmn_final['mae'] - baseline_final['mae']:+.3f} (lower is better)")
    
    return results


def main():
    """Main function for unified training comparison."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Unified Deep-GPCM Model Training')
    parser.add_argument('--epochs', type=int, default=20, help='Number of training epochs')
    parser.add_argument('--dataset', type=str, default='synthetic_OC', help='Dataset name')
    parser.add_argument('--models', nargs='+', choices=['baseline', 'akvmn', 'both'], 
                       default=['both'], help='Models to train')
    
    args = parser.parse_args()
    
    if 'both' in args.models:
        # Train both models with comparison
        results = compare_models(epochs=args.epochs, dataset_name=args.dataset)
    else:
        # Train individual models
        configs = get_preset_configs()
        
        for model_type in args.models:
            if model_type in ['baseline', 'akvmn']:
                config = {
                    field.name: getattr(configs[model_type], field.name)
                    for field in configs[model_type].__dataclass_fields__.values()
                }
                config['dataset_name'] = args.dataset
                config['epochs'] = args.epochs
                config['model_type'] = model_type
                
                model, history = train_model_unified(config, args.epochs)
                save_training_results(model_type, args.dataset, history, model, config)


if __name__ == "__main__":
    main()