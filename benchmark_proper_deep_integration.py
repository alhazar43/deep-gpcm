#!/usr/bin/env python3
"""
Benchmark Proper Deep Integration Model
Adapts existing benchmark and plot scripts to test the realistic Deep Integration model.
"""

import os
import torch
import json
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

from models.deep_integration_gpcm_proper import ProperDeepIntegrationGPCM
from models.baseline import BaselineGPCM
from evaluation.metrics import GpcmMetrics
import torch.utils.data as data_utils
import torch.nn as nn
import torch.optim as optim


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


def train_and_evaluate_model(model, train_loader, test_loader, device, epochs=10, model_name="model"):
    """Train and evaluate a model with comprehensive metrics tracking."""
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    training_history = []
    best_qwk = 0.0
    best_metrics = {}
    
    print(f"\\n🚀 Training {model_name} for {epochs} epochs...")
    print("Epoch | Train Loss | Train Acc | Test Acc | QWK | Ord.Acc | Time(s)")
    print("-" * 70)
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for questions, responses, mask in train_loader:
            questions = questions.to(device)
            responses = responses.to(device)
            
            optimizer.zero_grad()
            
            # Handle different model output formats
            if 'BaselineGPCM' in str(type(model)):  # Baseline model
                student_abilities, item_thresholds, discrimination_params, gpcm_probs = model(questions, responses)
                probs_flat = gpcm_probs.view(-1, gpcm_probs.size(-1))
            else:  # Deep Integration model
                student_abilities, item_thresholds, discrimination_params, gpcm_probs = model(questions, responses)
                probs_flat = gpcm_probs.view(-1, gpcm_probs.size(-1))
            
            responses_flat = responses.view(-1)
            
            # Compute loss
            loss = criterion(probs_flat, responses_flat)
            loss.backward()
            
            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            predicted = probs_flat.argmax(dim=-1)
            correct += (predicted == responses_flat).sum().item()
            total += responses_flat.numel()
        
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        
        # Evaluate
        model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for questions, responses, mask in test_loader:
                questions = questions.to(device)
                responses = responses.to(device)
                
                # Handle different model output formats
                if 'BaselineGPCM' in str(type(model)):  # Baseline model
                    student_abilities, item_thresholds, discrimination_params, gpcm_probs = model(questions, responses)
                    probs_flat = gpcm_probs.view(-1, gpcm_probs.size(-1))
                else:  # Deep Integration model
                    student_abilities, item_thresholds, discrimination_params, gpcm_probs = model(questions, responses)
                    probs_flat = gpcm_probs.view(-1, gpcm_probs.size(-1))
                
                responses_flat = responses.view(-1)
                
                all_predictions.append(probs_flat.cpu())
                all_targets.append(responses_flat.cpu())
        
        # Combine predictions and compute metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        metrics_calc = GpcmMetrics()
        eval_metrics = metrics_calc.benchmark_prediction_methods(all_predictions, all_targets, all_predictions.size(-1))
        
        # Extract argmax metrics
        argmax_metrics = eval_metrics['argmax']
        test_acc = argmax_metrics['categorical_accuracy']
        qwk = argmax_metrics['quadratic_weighted_kappa']
        ordinal_acc = argmax_metrics['ordinal_accuracy']
        mae = argmax_metrics['mean_absolute_error']
        
        epoch_time = time.time() - start_time
        
        print(f"{epoch+1:5d} | {train_loss:10.4f} | {train_acc:8.4f} | {test_acc:7.4f} | "
              f"{qwk:6.3f} | {ordinal_acc:6.4f} | {epoch_time:6.1f}")
        
        # Save best metrics
        if qwk > best_qwk:
            best_qwk = qwk
            best_metrics = {
                'categorical_accuracy': test_acc,
                'ordinal_accuracy': ordinal_acc,
                'quadratic_weighted_kappa': qwk,
                'mean_absolute_error': mae,
                'train_accuracy': train_acc,
                'train_loss': train_loss
            }
        
        # Record training history
        training_history.append({
            'epoch': epoch + 1,
            'train_loss': train_loss,
            'train_accuracy': train_acc,
            'test_accuracy': test_acc,
            'qwk': qwk,
            'ordinal_accuracy': ordinal_acc,
            'mae': mae
        })
    
    print(f"Best {model_name} QWK: {best_qwk:.3f}")
    return best_metrics, training_history


def plot_training_comparison(baseline_history, deep_integration_history):
    """Plot training comparison between baseline and Deep Integration."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Training Comparison: Baseline vs Proper Deep Integration', fontsize=16)
    
    metrics = [
        ('test_accuracy', 'Test Accuracy'),
        ('qwk', 'Quadratic Weighted Kappa'),
        ('ordinal_accuracy', 'Ordinal Accuracy'),
        ('mae', 'Mean Absolute Error'),
        ('train_loss', 'Training Loss'),
        ('train_accuracy', 'Train Accuracy')
    ]
    
    for idx, (metric, title) in enumerate(metrics):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Extract data
        baseline_epochs = [h['epoch'] for h in baseline_history]
        baseline_values = [h[metric] for h in baseline_history]
        
        deep_epochs = [h['epoch'] for h in deep_integration_history]
        deep_values = [h[metric] for h in deep_integration_history]
        
        # Plot
        ax.plot(baseline_epochs, baseline_values, 'b-', label='Baseline', linewidth=2)
        ax.plot(deep_epochs, deep_values, 'r-', label='Deep Integration', linewidth=2)
        
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('results/plots/proper_deep_integration_comparison.png', dpi=300, bbox_inches='tight')
    print("📊 Saved training comparison plot to: results/plots/proper_deep_integration_comparison.png")
    return fig


def main():
    print("🚀 COMPREHENSIVE BENCHMARK: Baseline vs Proper Deep Integration")
    print("=" * 70)
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load data
    print("\\n📊 Loading data...")
    dataset_name = "synthetic_OC"
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
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(train_data, test_data)
    
    # Create models
    print("\\n🏗️ Creating models...")
    
    # Baseline model
    baseline_model = BaselineGPCM(
        n_questions=n_questions,
        n_cats=n_cats,
        memory_size=50,
        key_dim=50,
        value_dim=200,
        final_fc_dim=50
    ).to(device)
    baseline_params = sum(p.numel() for p in baseline_model.parameters())
    print(f"Baseline: {baseline_params:,} parameters")
    
    # Proper Deep Integration model
    deep_integration_model = ProperDeepIntegrationGPCM(
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
    ).to(device)
    deep_params = sum(p.numel() for p in deep_integration_model.parameters())
    print(f"Deep Integration: {deep_params:,} parameters")
    
    # Train and evaluate both models
    epochs = 15  # Reasonable number for comparison
    
    # Train baseline
    baseline_metrics, baseline_history = train_and_evaluate_model(
        baseline_model, train_loader, test_loader, device, epochs, "Baseline"
    )
    
    # Train Deep Integration  
    deep_metrics, deep_history = train_and_evaluate_model(
        deep_integration_model, train_loader, test_loader, device, epochs, "Deep Integration"
    )
    
    # Compare results
    print("\\n📊 FINAL COMPARISON:")
    print("-" * 70)
    print(f"{'Metric':<25} {'Baseline':<12} {'Deep Integration':<15} {'Improvement':<12}")
    print("-" * 70)
    
    metrics_to_compare = [
        ('categorical_accuracy', 'Categorical Accuracy'),
        ('quadratic_weighted_kappa', 'QWK'),
        ('ordinal_accuracy', 'Ordinal Accuracy'),
        ('mean_absolute_error', 'MAE')
    ]
    
    realistic_improvements = []
    
    for metric_key, metric_name in metrics_to_compare:
        baseline_val = baseline_metrics[metric_key]
        deep_val = deep_metrics[metric_key]
        
        if metric_key == 'mean_absolute_error':  # Lower is better
            improvement = baseline_val - deep_val
            improvement_pct = (improvement / baseline_val) * 100
        else:  # Higher is better
            improvement = deep_val - baseline_val
            improvement_pct = (improvement / baseline_val) * 100
        
        realistic_improvements.append(improvement_pct)
        
        print(f"{metric_name:<25} {baseline_val:<12.3f} {deep_val:<15.3f} {improvement_pct:+7.1f}%")
    
    # Create directories if needed
    os.makedirs('results/plots', exist_ok=True)
    
    # Plot comparison
    plot_training_comparison(baseline_history, deep_history)
    
    # Save results
    results = {
        'timestamp': datetime.now().isoformat(),
        'dataset': dataset_name,
        'baseline': {
            'metrics': baseline_metrics,
            'history': baseline_history,
            'parameters': baseline_params
        },
        'deep_integration': {
            'metrics': deep_metrics,
            'history': deep_history,
            'parameters': deep_params
        },
        'comparison': {
            'improvements_percent': dict(zip([m[0] for m in metrics_to_compare], realistic_improvements))
        }
    }
    
    results_file = f'results/proper_deep_integration_benchmark_{dataset_name}.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\\n💾 Results saved to: {results_file}")
    
    # Sanity check
    avg_improvement = np.mean([abs(x) for x in realistic_improvements])
    if avg_improvement > 20:
        print("\\n⚠️  WARNING: >20% average improvement is suspicious!")
        print("Double-check that both models are using the same GPCM computation.")
    elif max([abs(x) for x in realistic_improvements]) < 1:
        print("\\n😐 Minimal improvement detected - this is actually realistic!")
        print("Deep learning improvements are often small but meaningful.")
    else:
        print("\\n🎯 REALISTIC IMPROVEMENT: Results look reasonable!")
        print(f"Average improvement: {avg_improvement:.1f}%")
    
    print("\\n✅ Benchmark completed!")


if __name__ == "__main__":
    main()