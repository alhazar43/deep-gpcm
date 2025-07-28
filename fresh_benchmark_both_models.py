#!/usr/bin/env python3
"""
Fresh Benchmark for Both Models
Tests baseline and proper Deep Integration from scratch to ensure nothing is broken.
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


def fresh_train_and_evaluate(model, train_loader, test_loader, device, epochs=20, model_name="model"):
    """Fresh training from scratch with comprehensive logging."""
    print(f"\\n🚀 FRESH TRAINING: {model_name}")
    print("-" * 60)
    
    # Fresh model initialization - ensure we start clean
    def init_weights(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal_(m.weight)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            torch.nn.init.kaiming_normal_(m.weight)
    
    model.apply(init_weights)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.8, patience=3)
    
    training_history = []
    best_qwk = -1.0  # Start with -1 to ensure we save something
    best_metrics = {}
    
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Model type: {type(model).__name__}")
    print("Epoch | Train Loss | Train Acc | Test Acc | QWK | Ord.Acc | Grad.Norm | LR | Time(s)")
    print("-" * 85)
    
    for epoch in range(epochs):
        start_time = time.time()
        
        # Train
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        grad_norms = []
        
        for batch_idx, (questions, responses, mask) in enumerate(train_loader):
            questions = questions.to(device)
            responses = responses.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass - both models have same output format now
            student_abilities, item_thresholds, discrimination_params, gpcm_probs = model(questions, responses)
            
            # Flatten for loss computation
            probs_flat = gpcm_probs.view(-1, gpcm_probs.size(-1))
            responses_flat = responses.view(-1)
            
            # Compute loss
            loss = criterion(probs_flat, responses_flat)
            
            # Check for NaN
            if torch.isnan(loss):
                print(f"🚨 WARNING: NaN loss in epoch {epoch+1}, batch {batch_idx}")
                continue
            
            loss.backward()
            
            # Monitor gradient norms
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_norms.append(grad_norm.item())
            
            # Check for gradient explosion
            if grad_norm > 10.0:
                print(f"⚠️  Large gradient: {grad_norm:.2f} in epoch {epoch+1}, batch {batch_idx}")
            
            optimizer.step()
            
            total_loss += loss.item()
            predicted = probs_flat.argmax(dim=-1)
            correct += (predicted == responses_flat).sum().item()
            total += responses_flat.numel()
        
        train_loss = total_loss / len(train_loader)
        train_acc = correct / total
        avg_grad_norm = np.mean(grad_norms) if grad_norms else 0.0
        
        # Evaluate
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
        
        # Combine predictions and compute metrics
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        # Calculate comprehensive metrics
        metrics_calc = GpcmMetrics()
        eval_metrics = metrics_calc.benchmark_prediction_methods(all_predictions, all_targets, all_predictions.size(-1))
        
        # Extract argmax metrics
        argmax_metrics = eval_metrics['argmax']
        test_acc = argmax_metrics['categorical_accuracy']
        qwk = argmax_metrics['quadratic_weighted_kappa']
        ordinal_acc = argmax_metrics['ordinal_accuracy']
        mae = argmax_metrics['mean_absolute_error']
        
        # Update learning rate
        scheduler.step(qwk)
        current_lr = optimizer.param_groups[0]['lr']
        
        epoch_time = time.time() - start_time
        
        print(f"{epoch+1:5d} | {train_loss:10.4f} | {train_acc:8.4f} | {test_acc:7.4f} | "
              f"{qwk:6.3f} | {ordinal_acc:6.4f} | {avg_grad_norm:8.3f} | {current_lr:.2e} | {epoch_time:6.1f}")
        
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
                'epoch': epoch + 1,
                'gradient_norm': avg_grad_norm,
                'learning_rate': current_lr
            }
            
            # Save model checkpoint
            checkpoint_dir = f'fresh_results/checkpoints'
            os.makedirs(checkpoint_dir, exist_ok=True)
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'epoch': epoch + 1,
                'best_qwk': best_qwk,
                'metrics': best_metrics
            }, f'{checkpoint_dir}/best_{model_name.lower().replace(" ", "_")}.pth')
        
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
        
        # Early stopping on gradient explosion
        if avg_grad_norm > 100:
            print(f"\\n🚨 STOPPING: Gradient explosion detected (norm: {avg_grad_norm:.1f})")
            break
    
    print(f"\\n✅ {model_name} training completed!")
    print(f"Best QWK: {best_qwk:.3f} at epoch {best_metrics.get('epoch', 'unknown')}")
    
    return best_metrics, training_history


def create_fresh_comparison_plot(baseline_history, deep_history, save_path):
    """Create fresh comparison plot."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Fresh Benchmark: Baseline vs Deep Integration (Clean Training)', fontsize=16)
    
    metrics = [
        ('test_accuracy', 'Test Accuracy'),
        ('qwk', 'Quadratic Weighted Kappa'),
        ('ordinal_accuracy', 'Ordinal Accuracy'),
        ('mae', 'Mean Absolute Error'),
        ('train_loss', 'Training Loss'),
        ('gradient_norm', 'Gradient Norm')
    ]
    
    for idx, (metric, title) in enumerate(metrics):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        # Extract data
        baseline_epochs = [h['epoch'] for h in baseline_history]
        baseline_values = [h[metric] for h in baseline_history]
        
        deep_epochs = [h['epoch'] for h in deep_history]
        deep_values = [h[metric] for h in deep_history]
        
        # Plot
        ax.plot(baseline_epochs, baseline_values, 'b-', label='Baseline', linewidth=2, marker='o', markersize=3)
        ax.plot(deep_epochs, deep_values, 'r-', label='Deep Integration', linewidth=2, marker='s', markersize=3)
        
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add final values as text
        if baseline_values and deep_values:
            final_baseline = baseline_values[-1]
            final_deep = deep_values[-1]
            if metric != 'mae' and metric != 'train_loss' and metric != 'gradient_norm':  # Higher is better
                improvement = ((final_deep - final_baseline) / final_baseline) * 100
            else:  # Lower is better
                improvement = ((final_baseline - final_deep) / final_baseline) * 100
            
            ax.text(0.02, 0.98, f'Δ: {improvement:+.1f}%', 
                   transform=ax.transAxes, fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"📊 Fresh comparison plot saved to: {save_path}")
    return fig


def main():
    print("🚀 FRESH BENCHMARK: Testing Both Models from Scratch")
    print("=" * 70)
    print("This benchmark ensures no existing functionality is broken")
    print()
    
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Create fresh results directory
    os.makedirs('fresh_results/plots', exist_ok=True)
    os.makedirs('fresh_results/data', exist_ok=True)
    
    # Load data
    print("\\n📊 Loading data...")
    dataset_name = "synthetic_OC"
    train_path = f"data/{dataset_name}/{dataset_name.lower()}_train.txt"
    test_path = f"data/{dataset_name}/{dataset_name.lower()}_test.txt"
    
    try:
        train_data, test_data, n_questions, n_cats = load_simple_data(train_path, test_path)
        print(f"Data loaded: {len(train_data)} train, {len(test_data)} test")
        print(f"Questions: {n_questions}, Categories: {n_cats}")
    except FileNotFoundError:
        print(f"Dataset not found. Generating synthetic data...")
        os.system("python data_gen.py --format OC --categories 4 --students 800 --questions 50")
        train_data, test_data, n_questions, n_cats = load_simple_data(train_path, test_path)
    
    # Create data loaders
    train_loader, test_loader = create_data_loaders(train_data, test_data, batch_size=32)
    
    print(f"\\n🧪 TRAINING CONFIGURATION:")
    print(f"  Dataset: {dataset_name}")
    print(f"  Epochs: 20")
    print(f"  Batch size: 32")
    print(f"  Learning rate: 0.001")
    print(f"  Weight decay: 1e-5")
    print(f"  Loss: CrossEntropyLoss")
    print(f"  Optimizer: Adam")
    
    # Test 1: Fresh Baseline Model
    print("\\n" + "="*70)
    print("TEST 1: Fresh Baseline Model")
    print("="*70)
    
    baseline_model = BaselineGPCM(
        n_questions=n_questions,
        n_cats=n_cats,
        memory_size=50,
        key_dim=50,
        value_dim=200,
        final_fc_dim=50
    ).to(device)
    
    baseline_metrics, baseline_history = fresh_train_and_evaluate(
        baseline_model, train_loader, test_loader, device, epochs=20, model_name="Baseline"
    )
    
    # Test 2: Fresh Deep Integration Model
    print("\\n" + "="*70)
    print("TEST 2: Fresh Deep Integration Model")
    print("="*70)
    
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
    
    deep_metrics, deep_history = fresh_train_and_evaluate(
        deep_integration_model, train_loader, test_loader, device, epochs=20, model_name="Deep Integration"
    )
    
    # Comparison Analysis
    print("\\n" + "="*70)
    print("FRESH BENCHMARK RESULTS COMPARISON")
    print("="*70)
    
    print(f"{'Metric':<25} {'Baseline':<12} {'Deep Integration':<15} {'Improvement':<12} {'Status':<10}")
    print("-" * 80)
    
    metrics_to_compare = [
        ('categorical_accuracy', 'Categorical Accuracy'),
        ('quadratic_weighted_kappa', 'QWK'),
        ('ordinal_accuracy', 'Ordinal Accuracy'),
        ('mean_absolute_error', 'MAE')
    ]
    
    all_improvements = []
    
    for metric_key, metric_name in metrics_to_compare:
        baseline_val = baseline_metrics[metric_key]
        deep_val = deep_metrics[metric_key]
        
        if metric_key == 'mean_absolute_error':  # Lower is better
            improvement = baseline_val - deep_val
            improvement_pct = (improvement / baseline_val) * 100 if baseline_val != 0 else 0
        else:  # Higher is better
            improvement = deep_val - baseline_val
            improvement_pct = (improvement / baseline_val) * 100 if baseline_val != 0 else 0
        
        all_improvements.append(improvement_pct)
        
        # Status assessment
        if abs(improvement_pct) < 0.1:
            status = "Same"
        elif improvement_pct > 0:
            status = "Better" if improvement_pct < 20 else "Suspicious"
        else:
            status = "Worse"
        
        print(f"{metric_name:<25} {baseline_val:<12.3f} {deep_val:<15.3f} {improvement_pct:+7.1f}% {status:<10}")
    
    # Create plots
    plot_path = 'fresh_results/plots/fresh_benchmark_comparison.png'
    create_fresh_comparison_plot(baseline_history, deep_history, plot_path)
    
    # Save detailed results
    fresh_results = {
        'timestamp': datetime.now().isoformat(),
        'dataset': dataset_name,
        'training_config': {
            'epochs': 20,
            'batch_size': 32,
            'learning_rate': 0.001,
            'weight_decay': 1e-5,
            'loss_function': 'CrossEntropyLoss',
            'optimizer': 'Adam'
        },
        'baseline': {
            'final_metrics': baseline_metrics,
            'training_history': baseline_history,
            'parameters': sum(p.numel() for p in baseline_model.parameters()),
            'model_type': 'BaselineGPCM'
        },
        'deep_integration': {
            'final_metrics': deep_metrics,
            'training_history': deep_history,
            'parameters': sum(p.numel() for p in deep_integration_model.parameters()),
            'model_type': 'ProperDeepIntegrationGPCM'
        },
        'comparison': {
            'improvements_percent': dict(zip([m[0] for m in metrics_to_compare], all_improvements)),
            'avg_improvement': np.mean([abs(x) for x in all_improvements]),
            'max_improvement': max([abs(x) for x in all_improvements])
        }
    }
    
    results_file = f'fresh_results/data/fresh_benchmark_results.json'
    with open(results_file, 'w') as f:
        json.dump(fresh_results, f, indent=2)
    
    print(f"\\n💾 Detailed results saved to: {results_file}")
    
    # Final Assessment
    avg_improvement = fresh_results['comparison']['avg_improvement']
    max_improvement = fresh_results['comparison']['max_improvement']
    
    print("\\n🔍 ASSESSMENT:")
    if max_improvement > 50:
        print("🚨 SUSPICIOUS: >50% improvement suggests a bug or unfair comparison")
    elif avg_improvement > 20:
        print("⚠️  QUESTIONABLE: >20% average improvement is unusually high")
    elif avg_improvement < 1:
        print("😐 MINIMAL: <1% improvement - models are essentially equivalent")
    else:
        print("✅ REALISTIC: 1-20% improvements are reasonable for architectural changes")
    
    print(f"   Average improvement: {avg_improvement:.1f}%")
    print(f"   Maximum improvement: {max_improvement:.1f}%")
    
    # Training stability check
    final_baseline_grad = baseline_history[-1]['gradient_norm']
    final_deep_grad = deep_history[-1]['gradient_norm']
    
    print("\\n🔧 TRAINING STABILITY:")
    if final_baseline_grad > 10 or final_deep_grad > 10:
        print("⚠️  High gradient norms detected - potential training instability")
    else:
        print("✅ Both models trained stably (gradient norms < 10)")
    
    print(f"   Baseline final gradient norm: {final_baseline_grad:.3f}")
    print(f"   Deep Integration final gradient norm: {final_deep_grad:.3f}")
    
    print("\\n✅ FRESH BENCHMARK COMPLETED!")
    print("Both models have been tested from scratch with identical conditions.")


if __name__ == "__main__":
    main()