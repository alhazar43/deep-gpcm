#!/usr/bin/env python3
"""
Proper Deep Integration Training Script
Uses actual GPCM probability computation for fair comparison with baseline.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import time
import json
from datetime import datetime
import numpy as np

from models.deep_integration_gpcm_proper import ProperDeepIntegrationGPCM
from evaluation.metrics import GpcmMetrics
import torch.utils.data as data_utils


def pad_sequence_batch(batch):
    """Collate function for padding sequences."""
    questions_batch, responses_batch = zip(*batch)
    
    # Find max length in batch
    max_len = max(len(seq) for seq in questions_batch)
    
    # Pad sequences
    questions_padded = []
    responses_padded = []
    masks = []
    
    for q, r in zip(questions_batch, responses_batch):
        q_len = len(q)
        # Pad questions and responses
        q_pad = q + [0] * (max_len - q_len)
        r_pad = r + [0] * (max_len - q_len)
        mask = [1] * q_len + [0] * (max_len - q_len)
        
        questions_padded.append(q_pad)
        responses_padded.append(r_pad)
        masks.append(mask)
    
    return (torch.tensor(questions_padded), 
            torch.tensor(responses_padded), 
            torch.tensor(masks))


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
    
    # Find number of questions and categories
    all_questions = []
    all_responses = []
    for q, r in train_data + test_data:
        all_questions.extend(q)
        all_responses.extend(r)
    
    n_questions = max(all_questions) + 1
    n_cats = max(all_responses) + 1
    
    return train_data, test_data, n_questions, n_cats


def train_epoch(model, train_loader, optimizer, criterion, device, n_cats=4):
    """Train for one epoch with gradient monitoring."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    grad_norms = []
    
    for batch_idx, (questions, responses, mask) in enumerate(train_loader):
        questions = questions.to(device)
        responses = responses.to(device)
        mask = mask.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass - note different output format from baseline
        student_abilities, item_thresholds, discrimination_params, gpcm_probs = model(questions, responses)
        
        # Flatten for loss computation
        probs_flat = gpcm_probs.view(-1, n_cats)
        responses_flat = responses.view(-1)
        
        # Compute loss using GPCM probabilities
        loss = criterion(probs_flat, responses_flat)
        
        # Check for NaN
        if torch.isnan(loss):
            print(f"🚨 WARNING: NaN loss detected in batch {batch_idx}, skipping")
            continue
        
        loss.backward()
        
        # Monitor gradient norms
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        grad_norms.append(grad_norm.item())
        
        # Warning for large gradients
        if grad_norm > 5.0:
            print(f"⚠️  Large gradient norm: {grad_norm:.3f} in batch {batch_idx}")
        
        optimizer.step()
        
        total_loss += loss.item()
        predicted = probs_flat.argmax(dim=-1)
        correct += (predicted == responses_flat).sum().item()
        total += responses_flat.numel()
    
    avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
    return total_loss / len(train_loader), correct / total, avg_grad_norm


def evaluate_epoch(model, test_loader, criterion, device, n_cats=4):
    """Evaluate model with comprehensive metrics."""
    model.eval()
    total_loss = 0.0
    all_predictions = []
    all_targets = []
    inference_times = []
    
    with torch.no_grad():
        for questions, responses, mask in test_loader:
            questions = questions.to(device)
            responses = responses.to(device) 
            mask = mask.to(device)
            
            start_time = time.time()
            student_abilities, item_thresholds, discrimination_params, gpcm_probs = model(questions, responses)
            inference_time = (time.time() - start_time) * 1000
            inference_times.append(inference_time)
            
            # Flatten for evaluation
            probs_flat = gpcm_probs.view(-1, n_cats)
            responses_flat = responses.view(-1)
            
            # Compute loss
            loss = criterion(probs_flat, responses_flat)
            total_loss += loss.item()
            
            all_predictions.append(probs_flat.cpu())
            all_targets.append(responses_flat.cpu())
    
    # Combine all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate comprehensive metrics
    metrics_calc = GpcmMetrics()
    eval_metrics = metrics_calc.benchmark_prediction_methods(all_predictions, all_targets, n_cats)
    
    # Extract argmax metrics for compatibility
    argmax_metrics = eval_metrics['argmax']
    categorical_acc = argmax_metrics['categorical_accuracy']
    ordinal_acc = argmax_metrics['ordinal_accuracy']
    qwk = argmax_metrics['quadratic_weighted_kappa']
    mae = argmax_metrics['mean_absolute_error']
    
    avg_inference_time = np.mean(inference_times)
    
    return total_loss / len(test_loader), categorical_acc, ordinal_acc, qwk, mae, avg_inference_time


def main():
    print("🚀 PROPER DEEP INTEGRATION TRAINING")
    print("Testing with actual GPCM probability computation")
    print("=" * 70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("📊 Loading data...")
    dataset_name = "synthetic_OC"  # Start with same dataset as "fixed" version
    train_path = f"data/{dataset_name}/{dataset_name.lower()}_train.txt"
    test_path = f"data/{dataset_name}/{dataset_name.lower()}_test.txt"
    
    try:
        train_data, test_data, n_questions, n_cats = load_simple_data(train_path, test_path)
        print(f"Data: {len(train_data)} train, {len(test_data)} test")
        print(f"Questions: {n_questions}, Categories: {n_cats}")
    except FileNotFoundError:
        print(f"Dataset not found. Generating synthetic data...")
        os.system("python data_gen.py --format OC --categories 4 --students 800 --questions 50")
        train_data, test_data, n_questions, n_cats = load_simple_data(train_path, test_path)
    
    # Create simple dataset class for variable length sequences
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
        train_dataset, batch_size=32, shuffle=True, collate_fn=pad_sequence_batch
    )
    test_loader = data_utils.DataLoader(
        test_dataset, batch_size=32, shuffle=False, collate_fn=pad_sequence_batch
    )
    
    # Create model
    print("\\n🏗️ Creating Proper Deep Integration model...")
    model = ProperDeepIntegrationGPCM(
        n_questions=n_questions,
        n_cats=n_cats,
        embed_dim=64,
        memory_size=50,
        key_dim=50,
        value_dim=200,
        final_fc_dim=50,
        n_heads=4,
        n_cycles=2,  # Start with 2 cycles for stability
        embedding_strategy="linear_decay"
    ).to(device)
    
    model_info = model.get_model_info()
    print(f"Model created: {model_info['parameters']:,} parameters")
    print(f"Architecture: {model_info['architecture']}")
    print(f"Refinement cycles: {model_info['cycles']}")
    print(f"Expected: {model_info['expected_improvement']}")
    
    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=3)
    
    # Training loop
    print("\\n🚀 Training for 30 epochs...")
    print("Epoch | Train Loss | Train Acc | Test Acc | QWK | Ord.Acc | Grad.Norm | Time(ms)")
    print("-" * 80)
    
    best_qwk = 0.0
    best_metrics = {}
    training_history = []
    
    for epoch in range(30):  # Same as baseline for fair comparison
        start_time = time.time()
        
        # Enable anomaly detection for debugging
        if epoch == 0:
            torch.autograd.set_detect_anomaly(True)
        
        # Train
        train_loss, train_acc, grad_norm = train_epoch(model, train_loader, optimizer, criterion, device, n_cats)
        
        # Evaluate
        test_loss, test_acc, ordinal_acc, qwk, mae, inference_time = evaluate_epoch(model, test_loader, criterion, device, n_cats)
        
        # Update learning rate
        scheduler.step(qwk)
        
        epoch_time = (time.time() - start_time) * 1000
        
        print(f"{epoch+1:5d} | {train_loss:10.4f} | {train_acc:8.4f} | {test_acc:7.4f} | "
              f"{qwk:6.3f} | {ordinal_acc:6.4f} | {grad_norm:8.3f} | {inference_time:7.1f}")
        
        # Save best model
        if qwk > best_qwk:
            best_qwk = qwk
            best_metrics = {
                'categorical_accuracy': test_acc,
                'ordinal_accuracy': ordinal_acc,
                'quadratic_weighted_kappa': qwk,
                'mean_absolute_error': mae,
                'train_accuracy': train_acc,
                'train_loss': train_loss,
                'test_loss': test_loss,
                'inference_time_ms': inference_time
            }
            
            torch.save({
                'model_state_dict': model.state_dict(),
                'config': model_info,
                'metrics': best_metrics,
                'epoch': epoch + 1
            }, f'save_models/best_deep_integration_proper_{dataset_name}.pth')
        
        # Record training history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'ordinal_accuracy': ordinal_acc,
            'qwk': qwk,
            'mae': mae,
            'gradient_norm': grad_norm,
            'inference_time_ms': inference_time,
            'learning_rate': optimizer.param_groups[0]['lr']
        })
        
        # Early stopping if gradients explode
        if grad_norm > 100:
            print(f"\\n🚨 Gradient explosion detected (norm: {grad_norm:.1f}), stopping training")
            break
    
    print("=" * 80)
    print("✅ Training completed!")
    print(f"Best Categorical Accuracy: {best_metrics['categorical_accuracy']:.4f}")
    print(f"Best QWK: {best_metrics['quadratic_weighted_kappa']:.3f}")
    print(f"Best Ordinal Accuracy: {best_metrics['ordinal_accuracy']:.4f}")
    
    # Save results
    results = {
        'model_type': 'deep_integration_gpcm_proper',
        'dataset': dataset_name,
        'config': model_info,
        'training_history': training_history,
        'best_metrics': best_metrics,
        'training_completed': True,
        'timestamp': datetime.now().isoformat()
    }
    
    results_file = f'save_models/results_deep_integration_proper_{dataset_name}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"💾 Results saved to: {results_file}")
    
    # Compare with baseline (if available)
    baseline_file = f'save_models/results_baseline_standalone_{dataset_name}.json'
    if os.path.exists(baseline_file):
        print("\\n📊 COMPARISON WITH BASELINE:")
        with open(baseline_file, 'r') as f:
            baseline_results = json.load(f)
        
        baseline_acc = baseline_results['best_metrics']['categorical_accuracy']
        baseline_qwk = baseline_results['best_metrics']['quadratic_weighted_kappa']
        
        acc_improvement = best_metrics['categorical_accuracy'] - baseline_acc
        qwk_improvement = best_metrics['quadratic_weighted_kappa'] - baseline_qwk
        
        print(f"Baseline:           {baseline_acc:.1%} categorical, {baseline_qwk:.3f} QWK")
        print(f"Deep Integration:   {best_metrics['categorical_accuracy']:.1%} categorical, {best_metrics['quadratic_weighted_kappa']:.3f} QWK")
        print(f"Improvement:        {acc_improvement:+.1%} categorical, {qwk_improvement:+.3f} QWK")
        
        # Sanity check
        if best_metrics['categorical_accuracy'] > 0.95:
            print("\\n⚠️  WARNING: Accuracy > 95% is suspiciously high!")
            print("This suggests the model might be overfitting or the dataset is too simple.")
            print("Consider testing on a more challenging dataset (STATICS, assist2015).")
        elif acc_improvement > 0.2:
            print("\\n⚠️  WARNING: >20% improvement is unusually large!")
            print("Double-check that both models are using the same GPCM computation.")
        else:
            print("\\n🎯 REALISTIC IMPROVEMENT: Results look reasonable!")
    
    print("\\n🔍 NEXT STEPS:")
    print("1. Test on larger dataset: python train_deep_integration_proper.py --dataset STATICS")
    print("2. Compare gradient norms and training stability")
    print("3. Analyze what the iterative refinement is actually learning")


if __name__ == "__main__":
    main()