#!/usr/bin/env python3
"""
Enhanced Training Script for Deep-GPCM with CORAL Integration

Supports both original GPCM and CORAL-enhanced training with embedding strategy comparison.
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
from models.dkvmn_coral_integration import DeepGpcmCoralModel, create_coral_enhanced_model
from models.coral_layer import CoralLoss
from evaluation.metrics import GpcmMetrics
from utils.gpcm_utils import (
    OrdinalLoss, load_gpcm_data, create_gpcm_batch,
    CrossEntropyLossWrapper, MSELossWrapper
)
from train import GpcmDataLoader, setup_logging


def evaluate_model_enhanced(model, data_loader, loss_fn, metrics, device, n_cats, use_coral=None):
    """Enhanced evaluation supporting both GPCM and CORAL models."""
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for q_batch, r_batch, mask_batch in data_loader:
            q_batch = q_batch.to(device)
            r_batch = r_batch.to(device)
            mask_batch = mask_batch.to(device)
            
            # Forward pass - handle both model types
            if hasattr(model, 'forward') and 'use_coral' in model.forward.__code__.co_varnames and use_coral is not None:
                # CORAL-enhanced model
                _, _, _, probs = model(q_batch, r_batch, use_coral=use_coral)
            else:
                # Original model
                _, _, _, probs = model(q_batch, r_batch)
            
            # Compute loss only on valid positions
            if mask_batch is not None:
                valid_probs = probs[mask_batch]
                valid_targets = r_batch[mask_batch]
            else:
                valid_probs = probs.view(-1, n_cats)
                valid_targets = r_batch.view(-1)
            
            loss = loss_fn(valid_probs.unsqueeze(1), valid_targets.unsqueeze(1))
            total_loss += loss.item()
            
            # Store for metrics
            all_probs.append(valid_probs.cpu())
            all_targets.append(valid_targets.cpu())
    
    # Compute metrics
    all_probs = torch.cat(all_probs, dim=0).unsqueeze(1)  # Add seq_len dim
    all_targets = torch.cat(all_targets, dim=0).unsqueeze(1)
    
    # Use cumulative prediction method for proper evaluation
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
    
    per_cat_acc = metrics.per_category_accuracy(all_probs, all_targets, n_cats, method=prediction_method)
    results.update(per_cat_acc)
    
    return results


def train_epoch_enhanced(model, train_loader, optimizer, loss_fn, device, n_cats, use_coral=None):
    """Enhanced training epoch supporting both GPCM and CORAL models."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for q_batch, r_batch, mask_batch in tqdm(train_loader, desc="Training"):
        q_batch = q_batch.to(device)
        r_batch = r_batch.to(device)
        mask_batch = mask_batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass - handle both model types
        if hasattr(model, 'forward') and 'use_coral' in model.forward.__code__.co_varnames and use_coral is not None:
            # CORAL-enhanced model
            _, _, _, probs = model(q_batch, r_batch, use_coral=use_coral)
        else:
            # Original model
            _, _, _, probs = model(q_batch, r_batch)
        
        # Compute loss only on valid positions
        if mask_batch is not None:
            valid_probs = probs[mask_batch]
            valid_targets = r_batch[mask_batch]
        else:
            valid_probs = probs.view(-1, n_cats)
            valid_targets = r_batch.view(-1)
        
        loss = loss_fn(valid_probs.unsqueeze(1), valid_targets.unsqueeze(1))
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def train_coral_enhanced_model(dataset_name, n_epochs=10, batch_size=64, learning_rate=0.001,
                              embedding_strategy='ordered', model_type='coral', device=None):
    """
    Train CORAL-enhanced model with comprehensive benchmarking.
    
    Args:
        dataset_name: Dataset to train on
        n_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate
        embedding_strategy: Embedding strategy to use
        model_type: 'gpcm' for original, 'coral' for CORAL-enhanced
        device: Device to train on
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print(f"TRAINING {model_type.upper()} MODEL")
    print(f"Dataset: {dataset_name}")
    print(f"Embedding Strategy: {embedding_strategy}")
    print(f"Epochs: {n_epochs}")
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
    
    # Create model
    base_model = DeepGpcmModel(
        n_questions=n_questions,
        n_cats=n_cats,
        memory_size=50,
        key_dim=50,
        value_dim=200,
        final_fc_dim=50,
        embedding_strategy=embedding_strategy
    ).to(device)
    
    if model_type == 'coral':
        model = create_coral_enhanced_model(base_model)
        model = model.to(device)
        use_coral = True
        print(f"Created CORAL-enhanced model")
        print(f"CORAL thresholds: {model.get_coral_thresholds().tolist()}")
    else:
        model = base_model
        use_coral = None
        print(f"Created original GPCM model")
    
    # Create loss function and optimizer
    loss_fn = OrdinalLoss(n_cats)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.7)
    
    # Training metrics
    metrics = GpcmMetrics()
    training_history = []
    best_valid_acc = 0.0
    
    print(f"\nStarting training...")
    
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        
        # Train
        train_loss = train_epoch_enhanced(model, train_loader, optimizer, loss_fn, device, n_cats, use_coral)
        
        # Evaluate
        train_metrics = evaluate_model_enhanced(model, train_loader, loss_fn, metrics, device, n_cats, use_coral)
        test_metrics = evaluate_model_enhanced(model, test_loader, loss_fn, metrics, device, n_cats, use_coral)
        
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
            model_filename = f"best_model_{dataset_name}_{model_type}_{embedding_strategy}.pth"
            model_path = os.path.join("save_models", model_filename)
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch,
                'best_valid_acc': best_valid_acc,
                'n_cats': n_cats,
                'n_questions': n_questions,
                'embedding_strategy': embedding_strategy,
                'model_type': model_type
            }, model_path)
            print(f"Saved best model: {model_filename}")
        
        # Store training history
        epoch_data = {
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'valid_loss': test_metrics['loss'],
            'learning_rate': current_lr,
            'embedding_strategy': embedding_strategy,
            'model_type': model_type,
            **train_metrics,
            **{f'valid_{k}': v for k, v in test_metrics.items()}
        }
        training_history.append(epoch_data)
    
    # Save training history
    os.makedirs("results/train", exist_ok=True)
    history_filename = f"training_history_{dataset_name}_{model_type}_{embedding_strategy}.json"
    history_path = os.path.join("results/train", history_filename)
    
    with open(history_path, 'w') as f:
        json.dump(training_history, f, indent=2)
    
    print(f"\nTraining complete!")
    print(f"Best validation accuracy: {best_valid_acc:.3f}")
    print(f"Training history saved: {history_filename}")
    
    return model, training_history


def run_comprehensive_coral_embedding_comparison(dataset_name="synthetic_OC", epochs=10):
    """
    Run comprehensive comparison of GPCM vs CORAL across all embedding strategies.
    """
    print(f"\n{'='*100}")
    print(f"COMPREHENSIVE CORAL vs GPCM EMBEDDING STRATEGY COMPARISON")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*100}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Embedding strategies to test
    embedding_strategies = ['ordered', 'unordered', 'linear_decay', 'adjacent_weighted']
    model_types = ['gpcm', 'coral']
    
    # Store all results
    all_results = {}
    training_results = {}
    
    total_experiments = len(embedding_strategies) * len(model_types)
    experiment_count = 0
    
    for embedding_strategy in embedding_strategies:
        for model_type in model_types:
            experiment_count += 1
            print(f"\n>>> Experiment {experiment_count}/{total_experiments} <<<")
            print(f"Training {model_type.upper()} with {embedding_strategy} embedding")
            
            try:
                # Train model
                model, history = train_coral_enhanced_model(
                    dataset_name=dataset_name,
                    n_epochs=epochs,
                    embedding_strategy=embedding_strategy,
                    model_type=model_type,
                    device=device
                )
                
                # Store results
                key = f"{model_type}_{embedding_strategy}"
                all_results[key] = history
                training_results[key] = {
                    'final_metrics': history[-1],
                    'best_epoch': max(history, key=lambda x: x['categorical_acc']),
                    'model_type': model_type,
                    'embedding_strategy': embedding_strategy
                }
                
                print(f"✅ {key} training completed successfully")
                
            except Exception as e:
                print(f"❌ Error training {key}: {e}")
                continue
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs("results/comparison", exist_ok=True)
    
    # Save detailed results
    detailed_path = f"results/comparison/coral_gpcm_embedding_comparison_{dataset_name}_{timestamp}.json"
    with open(detailed_path, 'w') as f:
        json.dump({
            'experiment_info': {
                'timestamp': timestamp,
                'dataset': dataset_name,
                'epochs': epochs,
                'total_experiments': total_experiments,
                'description': 'Comprehensive CORAL vs GPCM embedding strategy comparison'
            },
            'detailed_results': all_results,
            'summary_results': training_results
        }, f, indent=2)
    
    # Create performance summary
    print(f"\n{'='*100}")
    print(f"FINAL RESULTS SUMMARY")
    print(f"{'='*100}")
    
    # Performance comparison table
    print(f"\nFinal Performance Comparison:")
    print(f"{'Model_Strategy':<25} {'Cat_Acc':<8} {'Ord_Acc':<8} {'Pred_Cons':<10} {'MAE':<8} {'Ord_Rank':<10}")
    print("-" * 80)
    
    for key, result in training_results.items():
        final = result['final_metrics']
        print(f"{key:<25} {final['categorical_acc']:<8.3f} {final['ordinal_acc']:<8.3f} "
              f"{final['prediction_consistency_acc']:<10.3f} {final['mae']:<8.3f} "
              f"{final['ordinal_ranking_acc']:<10.3f}")
    
    # Find best performers
    print(f"\n🏆 BEST PERFORMERS:")
    
    metrics_to_check = ['categorical_acc', 'ordinal_acc', 'prediction_consistency_acc', 'ordinal_ranking_acc']
    
    for metric in metrics_to_check:
        best_key = max(training_results.keys(), 
                      key=lambda x: training_results[x]['final_metrics'][metric])
        best_value = training_results[best_key]['final_metrics'][metric]
        print(f"   {metric}: {best_key} ({best_value:.3f})")
    
    print(f"\n💾 Results saved to: {detailed_path}")
    
    return all_results, training_results


def main():
    """Main function for comprehensive CORAL vs GPCM comparison."""
    parser = argparse.ArgumentParser(description='Train and compare CORAL-enhanced vs GPCM models')
    parser.add_argument('--dataset', type=str, default='synthetic_OC', help='Dataset name')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--single', type=str, help='Train single model: format "model_type,embedding_strategy"')
    
    args = parser.parse_args()
    
    if args.single:
        # Train single model
        try:
            model_type, embedding_strategy = args.single.split(',')
            train_coral_enhanced_model(
                dataset_name=args.dataset,
                n_epochs=args.epochs,
                embedding_strategy=embedding_strategy,
                model_type=model_type
            )
        except ValueError:
            print("Error: --single format should be 'model_type,embedding_strategy'")
            print("Example: --single coral,ordered")
    else:
        # Run comprehensive comparison
        run_comprehensive_coral_embedding_comparison(args.dataset, args.epochs)


if __name__ == "__main__":
    main()