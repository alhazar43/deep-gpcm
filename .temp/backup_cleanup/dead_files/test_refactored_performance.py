#!/usr/bin/env python3
"""
Performance Test Script for Refactored Deep-GPCM Models
Tests the refactored modular architecture against original performance benchmarks.
"""

import os
import torch
import json
import time
import argparse
import numpy as np
from datetime import datetime
from sklearn.model_selection import KFold

from core.model import DKVMNGPCM as BaselineGPCM, AttentionDKVMNGPCM as AKVMNGPCM
from evaluation.metrics import GpcmMetrics
import torch.utils.data as data_utils
import torch.nn as nn
import torch.optim as optim


def load_simple_data(train_path, test_path):
    """Load data in the same format as original."""
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
                
                questions = questions[:seq_len]
                responses = responses[:seq_len]
                
                sequences.append((questions, responses))
                i += 3
        return sequences
    
    train_data = read_data(train_path)
    test_data = read_data(test_path)
    
    # Find number of questions and categories
    all_questions = []
    all_responses = []
    for q, r in train_data + test_data:
        all_questions.extend(q)
        all_responses.extend(r)
    
    n_questions = max(all_questions) + 1
    n_cats = max(all_responses) + 1
    
    return train_data, test_data, n_questions, n_cats


def pad_sequence_batch(batch):
    """Collate function for padding sequences."""
    questions_batch, responses_batch = zip(*batch)
    max_len = max(len(seq) for seq in questions_batch)
    
    questions_padded = []
    responses_padded = []
    masks = []
    
    for q, r in zip(questions_batch, responses_batch):
        q_len = len(q)
        q_pad = q + [0] * (max_len - q_len)
        r_pad = r + [0] * (max_len - q_len)
        mask = [1] * q_len + [0] * (max_len - q_len)
        
        questions_padded.append(q_pad)
        responses_padded.append(r_pad)
        masks.append(mask)
    
    return (torch.tensor(questions_padded), 
            torch.tensor(responses_padded), 
            torch.tensor(masks))


def create_data_loaders(train_data, test_data, batch_size=32):
    """Create data loaders."""
    class SequenceDataset(data_utils.Dataset):
        def __init__(self, data):
            self.data = data
        
        def __len__(self):
            return len(self.data)
        
        def __getitem__(self, idx):
            return self.data[idx]
    
    train_dataset = SequenceDataset(train_data)
    test_dataset = SequenceDataset(test_data)
    
    train_loader = data_utils.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=pad_sequence_batch
    )
    test_loader = data_utils.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_sequence_batch
    )
    
    return train_loader, test_loader


def create_model(model_type, n_questions, n_cats, device):
    """Create model with exact same parameters as original."""
    if model_type == 'baseline':
        model = BaselineGPCM(
            n_questions=n_questions,
            n_cats=n_cats,
            memory_size=50,
            key_dim=50,
            value_dim=200,
            final_fc_dim=50,
            embedding_strategy="linear_decay"
        )
    elif model_type == 'akvmn':
        model = AKVMNGPCM(
            n_questions=n_questions,
            n_cats=n_cats,
            embed_dim=64,
            memory_size=50,
            key_dim=50,
            value_dim=200,
            final_fc_dim=50,
            n_heads=4,
            n_cycles=2,
            embedding_strategy="linear_decay"
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


def train_refactored_model(model, train_loader, test_loader, device, epochs, model_name):
    """Train refactored model with exact same settings as original."""
    print(f"\n🚀 TESTING REFACTORED MODEL: {model_name}")
    print("=" * 80)
    
    # Same weight initialization as original
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.kaiming_normal_(m.weight)
    
    model.apply(init_weights)
    
    # Same optimizer settings as original
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=3)
    
    training_history = []
    best_qwk = -1.0
    best_epoch = 0
    best_model_state = None
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model type: {type(model).__name__}")
    print("Epoch | Train Loss | Train Acc | Test Acc | QWK | Ord.Acc | MAE | Grad.Norm | LR | Time(s)")
    print("-" * 95)
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Training phase
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        grad_norms = []
        
        for batch_idx, (questions, responses, mask) in enumerate(train_loader):
            questions = questions.to(device)
            responses = responses.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass - refactored models
            student_abilities, item_thresholds, discrimination_params, gpcm_probs = model(questions, responses)
            
            # Same loss computation as original
            probs_flat = gpcm_probs.view(-1, gpcm_probs.size(-1))
            responses_flat = responses.view(-1)
            
            loss = criterion(probs_flat, responses_flat)
            
            if torch.isnan(loss):
                print(f"🚨 WARNING: NaN loss in epoch {epoch+1}, batch {batch_idx}")
                continue
            
            loss.backward()
            
            # Same gradient clipping as original
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_norms.append(grad_norm.item())
            
            optimizer.step()
            
            total_loss += loss.item()
            predicted = probs_flat.argmax(dim=-1)
            correct += (predicted == responses_flat).sum().item()
            total += responses_flat.numel()
        
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
        
        # Evaluation phase - same as original
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for questions, responses, mask in test_loader:
                questions = questions.to(device)
                responses = responses.to(device)
                
                student_abilities, item_thresholds, discrimination_params, gpcm_probs = model(questions, responses)
                probs_flat = gpcm_probs.view(-1, gpcm_probs.size(-1))
                responses_flat = responses.view(-1)
                
                all_predictions.append(probs_flat.cpu())
                all_targets.append(responses_flat.cpu())
        
        # Same metrics computation as original
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics_calc = GpcmMetrics()
        eval_metrics = metrics_calc.benchmark_prediction_methods(all_predictions, all_targets, all_predictions.size(-1))
        
        # Extract same metrics as original
        argmax_metrics = eval_metrics['argmax']
        test_acc = argmax_metrics['categorical_accuracy']
        qwk = argmax_metrics['quadratic_weighted_kappa']
        ordinal_acc = argmax_metrics['ordinal_accuracy']
        mae = argmax_metrics['mean_absolute_error']
        
        # Same scheduler update as original
        scheduler.step(qwk)
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - start_time
        
        print(f"{epoch+1:5d} | {train_loss:10.4f} | {train_acc:8.4f} | {test_acc:7.4f} | "
              f"{qwk:6.3f} | {ordinal_acc:6.4f} | {mae:6.3f} | {avg_grad_norm:8.3f} | {current_lr:.2e} | {epoch_time:6.1f}")
        
        # Same best model selection as original
        if qwk > best_qwk:
            best_qwk = qwk
            best_epoch = epoch + 1
            best_model_state = model.state_dict().copy()
        
        # Record training history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'qwk': qwk,
            'ordinal_accuracy': ordinal_acc,
            'mae': mae,
            'gradient_norm': avg_grad_norm,
            'learning_rate': current_lr
        })
        
        # Same early stopping as original
        if avg_grad_norm > 100:
            print(f"\n🚨 STOPPING: Gradient explosion detected (norm: {avg_grad_norm:.1f})")
            break
    
    print(f"\n✅ Training completed! Best QWK: {best_qwk:.3f} at epoch {best_epoch}")
    
    # Save best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
        
        # Save model with detailed config
        model_path = f"save_models/refactored_{model_name}_synthetic_OC.pth"
        os.makedirs("save_models", exist_ok=True)
        
        torch.save({
            'model_state_dict': best_model_state,
            'model_name': model_name,
            'model_type': model_name,
            'n_questions': n_questions,
            'n_cats': n_cats,
            'dataset': 'synthetic_OC',
            'config': {
                'model_type': model_name,
                'n_questions': n_questions,
                'n_cats': n_cats,
                'dataset': 'synthetic_OC',
                'refactored': True
            },
            'training_metrics': {
                'categorical_accuracy': training_history[best_epoch-1]['test_accuracy'],
                'ordinal_accuracy': training_history[best_epoch-1]['ordinal_accuracy'],
                'quadratic_weighted_kappa': best_qwk,
                'mean_absolute_error': training_history[best_epoch-1]['mae'],
                'train_accuracy': training_history[best_epoch-1]['train_accuracy'],
                'train_loss': training_history[best_epoch-1]['train_loss'],
                'best_epoch': best_epoch,
                'total_epochs': len(training_history),
                'parameters': sum(p.numel() for p in model.parameters()),
                'model_type': type(model).__name__
            }
        }, model_path)
        
        print(f"💾 Refactored model saved to: {model_path}")
    
    return {
        'best_qwk': best_qwk,
        'best_epoch': best_epoch,
        'training_history': training_history,
        'final_metrics': eval_metrics
    }


def main():
    parser = argparse.ArgumentParser(description='Test refactored Deep-GPCM performance')
    parser.add_argument('--model', choices=['baseline', 'akvmn'], default='baseline', 
                       help='Model type to test')
    parser.add_argument('--epochs', type=int, default=30, 
                       help='Number of epochs (use 30 for full test)')
    parser.add_argument('--dataset', default='synthetic_OC', 
                       help='Dataset name')
    parser.add_argument('--device', default='cuda', 
                       help='Device to use')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("REFACTORED DEEP-GPCM PERFORMANCE TEST")
    print("=" * 80)
    print(f"Model: {args.model}")
    print(f"Dataset: {args.dataset}")
    print(f"Epochs: {args.epochs}")
    print(f"Device: {args.device}")
    print()
    
    # Load data
    train_path = f"data/{args.dataset}/{args.dataset.lower()}_train.txt"
    test_path = f"data/{args.dataset}/{args.dataset.lower()}_test.txt"
    
    train_data, test_data, n_questions, n_cats = load_simple_data(train_path, test_path)
    train_loader, test_loader = create_data_loaders(train_data, test_data, batch_size=32)
    
    print(f"📊 Data loaded: {len(train_data)} train, {len(test_data)} test")
    print(f"Questions: {n_questions}, Categories: {n_cats}")
    
    # Create and test refactored model
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model = create_model(args.model, n_questions, n_cats, device)
    
    # Train with exact same configuration as working original
    results = train_refactored_model(
        model, train_loader, test_loader, device, args.epochs, args.model
    )
    
    print(f"\n" + "=" * 80)
    print("REFACTORED MODEL PERFORMANCE SUMMARY")
    print("=" * 80)
    print(f"🎯 Best QWK: {results['best_qwk']:.3f} (Target: >0.65)")
    print(f"📊 Best Epoch: {results['best_epoch']}")
    
    # Compare against known benchmarks
    target_qwk = 0.65  # Based on original working results
    if results['best_qwk'] >= target_qwk:
        print(f"✅ SUCCESS: Refactored model achieves target performance!")
    else:
        print(f"❌ ISSUE: Refactored model underperforms (QWK {results['best_qwk']:.3f} < {target_qwk})")
    
    # Save results for analysis
    os.makedirs("logs", exist_ok=True)
    results_path = f"logs/refactored_test_{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    with open(results_path, 'w') as f:
        # Convert any tensors to lists for JSON serialization
        serializable_results = {
            'model_type': args.model,
            'epochs': args.epochs,
            'dataset': args.dataset,
            'best_qwk': results['best_qwk'],
            'best_epoch': results['best_epoch'],
            'training_history': results['training_history'],
            'test_timestamp': datetime.now().isoformat(),
            'refactored': True
        }
        json.dump(serializable_results, f, indent=2)
    
    print(f"📋 Results saved to: {results_path}")
    print("\n✅ Refactored performance test completed!")


if __name__ == "__main__":
    main()