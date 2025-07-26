#!/usr/bin/env python3
"""
Comprehensive Benchmark: GPCM vs CORAL vs CORN
Evaluates all three approaches on the same 7 metrics for fair comparison.
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
from models.dkvmn_coral_integration import create_coral_enhanced_model
from models.dkvmn_corn_integration import create_corn_enhanced_model
from models.corn_layer import CornLoss
from evaluation.metrics import GpcmMetrics
from utils.gpcm_utils import load_gpcm_data, create_gpcm_batch, OrdinalLoss
from train import GpcmDataLoader, setup_logging
from train_corn_enhanced import train_corn_enhanced_model

def evaluate_model_comprehensive(model, data_loader, loss_fn, metrics, device, n_cats, 
                                 model_type="gpcm", trainer=None):
    """Comprehensive evaluation supporting all three model types."""
    model.eval()
    total_loss = 0.0
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for q_batch, r_batch, mask_batch in data_loader:
            q_batch = q_batch.to(device)
            r_batch = r_batch.to(device)
            mask_batch = mask_batch.to(device)
            
            # Forward pass based on model type
            if model_type == "corn":
                _, _, _, probs, logits = model(q_batch, r_batch, use_corn=True)
            elif model_type == "coral":
                _, _, _, probs = model(q_batch, r_batch, use_coral=True)
                logits = None
            else:
                _, _, _, probs = model(q_batch, r_batch)
                logits = None
            
            # Compute loss only on valid positions
            if mask_batch is not None:
                valid_probs = probs[mask_batch]
                valid_targets = r_batch[mask_batch]
                if logits is not None:
                    valid_logits = logits[mask_batch]
                else:
                    valid_logits = None
            else:
                valid_probs = probs.view(-1, n_cats)
                valid_targets = r_batch.view(-1)
                if logits is not None:
                    valid_logits = logits.view(-1, n_cats - 1)
                else:
                    valid_logits = None
            
            # Compute loss
            if model_type == "corn":
                loss = loss_fn(valid_logits, valid_targets)
            else:
                loss = loss_fn(valid_probs.unsqueeze(1), valid_targets.unsqueeze(1))
            
            total_loss += loss.item()
            
            # Store for metrics
            all_probs.append(valid_probs.cpu())
            all_targets.append(valid_targets.cpu())
    
    # Compute metrics
    all_probs = torch.cat(all_probs, dim=0).unsqueeze(1)  # Add seq_len dim
    all_targets = torch.cat(all_targets, dim=0).unsqueeze(1)
    
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
    
    return results


def train_epoch_comprehensive(model, train_loader, optimizer, loss_fn, device, n_cats, 
                             model_type="gpcm", trainer=None):
    """Comprehensive training epoch supporting all three model types."""
    model.train()
    total_loss = 0.0
    n_batches = 0
    
    for q_batch, r_batch, mask_batch in tqdm(train_loader, desc=f"Training {model_type}"):
        q_batch = q_batch.to(device)
        r_batch = r_batch.to(device)
        mask_batch = mask_batch.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass based on model type
        if model_type == "corn":
            _, _, _, probs, logits = model(q_batch, r_batch, use_corn=True)
        elif model_type == "coral":
            _, _, _, probs = model(q_batch, r_batch, use_coral=True)
            logits = None
        else:
            _, _, _, probs = model(q_batch, r_batch)
            logits = None
        
        # Compute loss only on valid positions
        if mask_batch is not None:
            valid_probs = probs[mask_batch]
            valid_targets = r_batch[mask_batch]
            if logits is not None:
                valid_logits = logits[mask_batch]
            else:
                valid_logits = None
        else:
            valid_probs = probs.view(-1, n_cats)
            valid_targets = r_batch.view(-1)
            if logits is not None:
                valid_logits = logits.view(-1, n_cats - 1)
            else:
                valid_logits = None
        
        # Compute loss
        if model_type == "corn":
            loss = loss_fn(valid_logits, valid_targets)
        else:
            loss = loss_fn(valid_probs.unsqueeze(1), valid_targets.unsqueeze(1))
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
    
    return total_loss / n_batches


def train_single_model(dataset_name, embedding_strategy, model_type, n_epochs=15, device=None):
    """Train a single model (GPCM, CORAL, or CORN)."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n{'='*80}")
    print(f"TRAINING {model_type.upper()} MODEL")
    print(f"Dataset: {dataset_name}")
    print(f"Embedding Strategy: {embedding_strategy}")
    print(f"{'='*80}")
    
    # Load data
    train_path = f"data/{dataset_name}/synthetic_oc_train.txt"
    test_path = f"data/{dataset_name}/synthetic_oc_test.txt"
    
    train_seqs, train_questions, train_responses, n_cats = load_gpcm_data(train_path)
    test_seqs, test_questions, test_responses, _ = load_gpcm_data(test_path, n_cats)
    
    # Create data loaders
    batch_size = 64
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
    
    # Create appropriate model based on type
    if model_type == "gpcm":
        model = base_model
        loss_fn = OrdinalLoss(n_cats)
        print("Created GPCM model")
    elif model_type == "coral":
        model = create_coral_enhanced_model(base_model).to(device)
        loss_fn = OrdinalLoss(n_cats)
        print("Created CORAL model")
    elif model_type == "corn":
        model = create_corn_enhanced_model(base_model).to(device)
        loss_fn = CornLoss(n_cats)
        print("Created CORN model")
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Create optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=3, factor=0.7)
    
    # Training metrics
    metrics = GpcmMetrics()
    training_history = []
    best_valid_acc = 0.0
    
    os.makedirs("save_models", exist_ok=True)
    print(f"Starting training...")
    
    for epoch in range(n_epochs):
        print(f"\nEpoch {epoch+1}/{n_epochs}")
        
        # Train
        train_loss = train_epoch_comprehensive(
            model, train_loader, optimizer, loss_fn, device, n_cats, model_type
        )
        
        # Evaluate
        train_metrics = evaluate_model_comprehensive(
            model, train_loader, loss_fn, metrics, device, n_cats, model_type
        )
        test_metrics = evaluate_model_comprehensive(
            model, test_loader, loss_fn, metrics, device, n_cats, model_type
        )
        
        # Learning rate scheduling
        scheduler.step(test_metrics['loss'])
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log progress
        print(f"Train Loss: {train_loss:.4f}, Test Loss: {test_metrics['loss']:.4f}")
        print(f"Test Metrics - Cat Acc: {test_metrics['categorical_acc']:.3f}, "
              f"Ord Acc: {test_metrics['ordinal_acc']:.3f}, "
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
    
    print(f"Training complete! Best validation accuracy: {best_valid_acc:.3f}")
    
    return {
        'model_type': model_type,
        'embedding_strategy': embedding_strategy,
        'final_metrics': training_history[-1],
        'best_epoch': max(training_history, key=lambda x: x['categorical_acc']),
        'training_history': training_history
    }


def run_comprehensive_benchmark(dataset_name="synthetic_OC", epochs=15):
    """
    Run comprehensive benchmark: GPCM vs CORAL vs CORN.
    """
    print(f"\n{'='*100}")
    print(f"COMPREHENSIVE 3-WAY BENCHMARK")
    print(f"GPCM vs CORAL vs CORN")
    print(f"Dataset: {dataset_name}")
    print(f"{'='*100}")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test strategies (focus on best performers)
    embedding_strategies = ['linear_decay', 'adjacent_weighted']
    model_types = ['gpcm', 'coral', 'corn']
    
    # Store all results
    all_results = {}
    
    total_experiments = len(embedding_strategies) * len(model_types)
    experiment_count = 0
    
    for embedding_strategy in embedding_strategies:
        for model_type in model_types:
            experiment_count += 1
            print(f"\n>>> Experiment {experiment_count}/{total_experiments} <<<")
            print(f"Training {model_type.upper()} with {embedding_strategy} embedding")
            
            try:
                # Train model
                result = train_single_model(
                    dataset_name=dataset_name,
                    embedding_strategy=embedding_strategy,
                    model_type=model_type,
                    n_epochs=epochs,
                    device=device
                )
                
                # Store results
                key = f"{model_type}_{embedding_strategy}"
                all_results[key] = result
                
                print(f"✅ {key} training completed successfully")
                
            except Exception as e:
                print(f"❌ Error training {key}: {e}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save comprehensive results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    os.makedirs("results/comparison", exist_ok=True)
    
    # Save detailed results
    detailed_path = f"results/comparison/comprehensive_3way_benchmark_{dataset_name}_{timestamp}.json"
    with open(detailed_path, 'w') as f:
        json.dump({
            'experiment_info': {
                'timestamp': timestamp,
                'dataset': dataset_name,
                'epochs': epochs,
                'total_experiments': total_experiments,
                'description': 'Comprehensive 3-way benchmark: GPCM vs CORAL vs CORN'
            },
            'results': all_results
        }, f, indent=2)
    
    # Print comprehensive results
    print(f"\n{'='*100}")
    print(f"COMPREHENSIVE BENCHMARK RESULTS")
    print(f"{'='*100}")
    
    print(f"\nFinal Performance Comparison (All 7 Metrics):")
    print(f"{'Model_Strategy':<30} {'Cat_Acc':<8} {'Ord_Acc':<8} {'Pred_Cons':<10} {'MAE':<8} {'QWK':<8} {'Ord_Rank':<10} {'Dist_Cons':<10}")
    print("-" * 120)
    
    for key, result in all_results.items():
        final = result['final_metrics']
        print(f"{key:<30} {final['categorical_acc']:<8.3f} {final['ordinal_acc']:<8.3f} "
              f"{final['prediction_consistency_acc']:<10.3f} {final['mae']:<8.3f} "
              f"{final['qwk']:<8.3f} {final['ordinal_ranking_acc']:<10.3f} "
              f"{final['distribution_consistency']:<10.3f}")
    
    # Find best performers by model type
    print(f"\n🏆 BEST PERFORMERS BY MODEL TYPE:")
    
    for model_type in model_types:
        model_results = {k: v for k, v in all_results.items() if v['model_type'] == model_type}
        if model_results:
            best_key = max(model_results.keys(), 
                          key=lambda x: model_results[x]['final_metrics']['categorical_acc'])
            best_acc = model_results[best_key]['final_metrics']['categorical_acc']
            print(f"   {model_type.upper()}: {best_key} ({best_acc:.3f})")
    
    print(f"\n💾 Results saved to: {detailed_path}")
    
    return all_results


def main():
    """Main function for comprehensive 3-way benchmark."""
    parser = argparse.ArgumentParser(description='Comprehensive 3-way benchmark')
    parser.add_argument('--dataset', type=str, default='synthetic_OC', help='Dataset name')
    parser.add_argument('--epochs', type=int, default=15, help='Number of epochs')
    
    args = parser.parse_args()
    
    # Run comprehensive benchmark
    run_comprehensive_benchmark(args.dataset, args.epochs)


if __name__ == "__main__":
    main()