#!/usr/bin/env python3
"""
Standalone Baseline Training Script
Uses historically proven optimal configuration for baseline model.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import json
from datetime import datetime
from typing import Dict, Any

from config import get_preset_configs
from model_factory import create_model
from utils.gpcm_utils import load_gpcm_data
from utils.data_utils import UnifiedDataLoader
from evaluation.metrics import GpcmMetrics


def create_optimal_loss_function(n_cats=4, device='cpu'):
    """Create the historically optimal loss function for baseline."""
    # From conversation history: Cross-Entropy achieved 55.0% accuracy
    return nn.CrossEntropyLoss().to(device)


def train_epoch(model, train_loader, optimizer, criterion, device, n_cats=4):
    """Train for one epoch with proper loss handling."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    
    for questions, responses, mask in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        read_content, mastery_level, logits, probs = model(questions, responses)
        
        # Flatten for loss computation
        probs_flat = probs.view(-1, n_cats)
        responses_flat = responses.view(-1)
        
        # Compute loss
        loss = criterion(probs_flat, responses_flat)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        predicted = probs_flat.argmax(dim=-1)
        correct += (predicted == responses_flat).sum().item()
        total += responses_flat.numel()
    
    return total_loss / len(train_loader), correct / total


def evaluate_epoch(model, test_loader, criterion, device, n_cats=4):
    """Evaluate model for one epoch with comprehensive metrics."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    inference_times = []
    
    with torch.no_grad():
        for questions, responses, mask in test_loader:
            start_time = time.time()
            read_content, mastery_level, logits, probs = model(questions, responses)
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            inference_times.append(inference_time)
            
            # Flatten for evaluation
            probs_flat = probs.view(-1, n_cats)
            responses_flat = responses.view(-1)
            
            # Compute loss
            loss = criterion(probs_flat, responses_flat)
            total_loss += loss.item()
            
            all_predictions.append(probs_flat.cpu())
            all_targets.append(responses_flat.cpu())
    
    # Combine all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate comprehensive metrics using the working pattern from train.py
    metrics_calc = GpcmMetrics()
    eval_metrics = metrics_calc.benchmark_prediction_methods(all_predictions, all_targets, n_cats)
    
    # Extract specific metrics for compatibility (use argmax method)
    argmax_metrics = eval_metrics['argmax']
    categorical_acc = argmax_metrics['categorical_accuracy'] 
    ordinal_acc = argmax_metrics['ordinal_accuracy']
    qwk = argmax_metrics['quadratic_weighted_kappa']
    mae = argmax_metrics['mean_absolute_error']
    pred_consistency = argmax_metrics['prediction_consistency']
    ordinal_ranking = argmax_metrics['ordinal_ranking']
    dist_consistency = argmax_metrics['distribution_consistency']
    avg_inference_time = sum(inference_times) / len(inference_times)
    
    return {
        'loss': total_loss / len(test_loader),
        'categorical_accuracy': categorical_acc,
        'ordinal_accuracy': ordinal_acc,
        'qwk': qwk,
        'mae': mae,
        'prediction_consistency': pred_consistency,
        'ordinal_ranking': ordinal_ranking,
        'distribution_consistency': dist_consistency,
        'avg_inference_time': avg_inference_time
    }


def main():
    """Train baseline model with optimal historical configuration."""
    print("🏋️ STANDALONE BASELINE TRAINING")
    print("Using historically proven optimal configuration")
    print("="*60)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data (following working train.py pattern)
    print("📊 Loading data...")
    dataset_name = "synthetic_OC"
    train_path = f"data/{dataset_name}/{dataset_name.lower()}_train.txt"
    test_path = f"data/{dataset_name}/{dataset_name.lower()}_test.txt"
    
    train_seqs, train_questions, train_responses, n_cats = load_gpcm_data(train_path)
    test_seqs, test_questions, test_responses, _ = load_gpcm_data(test_path, n_cats)
    
    # Create data loaders
    train_loader = UnifiedDataLoader(train_questions, train_responses, batch_size=32, shuffle=True, device=device)
    test_loader = UnifiedDataLoader(test_questions, test_responses, batch_size=32, shuffle=False, device=device)
    
    # Get number of questions
    all_questions = []
    for seq in train_questions + test_questions:
        all_questions.extend(seq)
    n_questions = max(all_questions)
    
    print(f"Data: {len(train_questions)} train, {len(test_questions)} test")
    print(f"Questions: {n_questions}, Categories: 4")
    
    # Create baseline model with optimal configuration
    print("\n🏗️ Creating baseline model...")
    config = get_preset_configs()['baseline']
    config.n_questions = n_questions
    config.loss_type = 'crossentropy'  # Historically optimal
    config.learning_rate = 0.001
    config.epochs = 30  # Full training
    
    model = create_model(config, n_questions, device)
    param_count = sum(p.numel() for p in model.parameters())
    print(f"Model created: {param_count:,} parameters")
    
    # Create optimal loss function and optimizer
    criterion = create_optimal_loss_function(n_cats=4, device=device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    print(f"\n🚀 Training for {config.epochs} epochs...")
    print("Epoch | Train Loss | Train Acc | Test Acc | QWK | Ord.Acc | Time(ms)")
    print("-" * 70)
    
    best_metrics = None
    best_acc = 0.0
    training_history = []
    
    for epoch in range(config.epochs):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Evaluate
        test_metrics = evaluate_epoch(model, test_loader, criterion, device)
        
        # Print progress
        if epoch % 5 == 0 or epoch == config.epochs - 1:
            print(f"{epoch+1:5d} | {train_loss:10.4f} | {train_acc:9.4f} | "
                  f"{test_metrics['categorical_accuracy']:8.4f} | "
                  f"{test_metrics['qwk']:7.3f} | "
                  f"{test_metrics['ordinal_accuracy']:7.4f} | "
                  f"{test_metrics['avg_inference_time']:7.1f}")
        
        # Track best performance
        if test_metrics['categorical_accuracy'] > best_acc:
            best_acc = test_metrics['categorical_accuracy']
            best_metrics = test_metrics.copy()
            
            # Save best model
            os.makedirs('save_models', exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': config.__dict__,
                'metrics': best_metrics,
                'epoch': epoch + 1
            }, 'save_models/best_baseline_standalone_synthetic_OC.pth')
        
        # Record training history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'test_metrics': test_metrics
        })
    
    print("=" * 70)
    print("✅ Training completed!")
    print(f"Best Categorical Accuracy: {best_metrics['categorical_accuracy']:.4f}")
    print(f"Best QWK: {best_metrics['qwk']:.3f}")
    print(f"Best Ordinal Accuracy: {best_metrics['ordinal_accuracy']:.4f}")
    
    # Save complete results
    results = {
        'model_type': 'baseline_standalone',
        'dataset': 'synthetic_OC',
        'config': config.__dict__,
        'training_history': training_history,
        'best_metrics': best_metrics,
        'final_accuracy': best_metrics['categorical_accuracy'],
        'parameter_count': param_count,
        'training_completed': datetime.now().isoformat()
    }
    
    results_file = 'save_models/results_baseline_standalone_synthetic_OC.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {results_file}")
    
    # Compare with expected historical performance
    print(f"\n📊 PERFORMANCE COMPARISON:")
    print(f"Historical Expected: ~55.0% categorical accuracy")
    print(f"Current Achieved: {best_metrics['categorical_accuracy']:.1%}")
    
    if best_metrics['categorical_accuracy'] >= 0.50:
        print("🏆 SUCCESS: Performance matches historical expectations!")
    else:
        print("⚠️ WARNING: Performance below historical expectations")
        print("Consider increasing epochs or checking data quality")
    
    return results


if __name__ == "__main__":
    main()