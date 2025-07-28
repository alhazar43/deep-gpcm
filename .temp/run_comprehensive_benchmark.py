#!/usr/bin/env python3
"""
Comprehensive Benchmark Script for Deep-GPCM
Runs all models and evaluates with 7 comprehensive metrics.
"""

import os
import torch
import time
import json
from datetime import datetime

from config import get_preset_configs
from model_factory import create_model
from utils.gpcm_utils import load_gpcm_data
from utils.data_utils import UnifiedDataLoader
from utils.loss_utils import create_loss_function
from evaluation.metrics import GpcmMetrics
import torch.optim as optim


def train_model_simple(model, train_loader, test_loader, config, device, epochs=5):
    """Simple training function."""
    criterion = create_loss_function(config.loss_type, n_cats=config.n_cats).to(device)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    
    best_acc = 0.0
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for questions, responses, mask in train_loader:
            questions, responses = questions.to(device), responses.to(device)
            
            optimizer.zero_grad()
            read_content, mastery_level, logits, probs = model(questions, responses)
            
            loss = criterion(probs, responses)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            predicted = probs.argmax(dim=-1)
            correct += (predicted == responses).sum().item()
            total += responses.numel()
        
        # Evaluation
        model.eval()
        test_correct = 0
        test_total = 0
        
        with torch.no_grad():
            for questions, responses, mask in test_loader:
                questions, responses = questions.to(device), responses.to(device)
                read_content, mastery_level, logits, probs = model(questions, responses)
                
                predicted = probs.argmax(dim=-1)
                test_correct += (predicted == responses).sum().item()
                test_total += responses.numel()
        
        test_acc = test_correct / test_total
        best_acc = max(best_acc, test_acc)
        
        if epoch == 0 or epoch == epochs - 1:
            print(f"Epoch {epoch+1:3d}/{epochs}: Train Acc: {correct/total:.4f}, Test Acc: {test_acc:.4f}")
    
    return best_acc


def evaluate_comprehensive(model, test_loader, device, n_cats=4):
    """Comprehensive evaluation with all 7 metrics."""
    model.eval()
    
    all_predictions = []
    all_targets = []
    inference_times = []
    
    with torch.no_grad():
        for questions, responses, mask in test_loader:
            questions, responses = questions.to(device), responses.to(device)
            
            start_time = time.time()
            read_content, mastery_level, logits, probs = model(questions, responses)
            inference_time = (time.time() - start_time) * 1000  # Convert to ms
            inference_times.append(inference_time)
            
            # Flatten for evaluation
            probs_flat = probs.view(-1, n_cats)
            responses_flat = responses.view(-1)
            
            all_predictions.append(probs_flat.cpu())
            all_targets.append(responses_flat.cpu())
    
    # Combine all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    # Calculate comprehensive metrics
    metrics = GpcmMetrics(n_cats=n_cats)
    
    # 1. Categorical Accuracy
    predicted_classes = all_predictions.argmax(dim=-1)
    categorical_acc = (predicted_classes == all_targets).float().mean().item()
    
    # 2. Ordinal Accuracy (perfect ordering)
    ordinal_acc = metrics.ordinal_accuracy(all_predictions, all_targets)
    
    # 3. Quadratic Weighted Kappa (QWK)
    qwk = metrics.quadratic_weighted_kappa(all_predictions, all_targets)
    
    # 4. Mean Absolute Error (MAE)
    mae = metrics.mean_absolute_error(all_predictions, all_targets)
    
    # 5. Prediction Consistency
    pred_consistency = metrics.prediction_consistency(all_predictions, all_targets)
    
    # 6. Ordinal Ranking Metric
    ordinal_ranking = metrics.ordinal_ranking_metric(all_predictions, all_targets)
    
    # 7. Distribution Consistency
    dist_consistency = metrics.distribution_consistency(all_predictions, all_targets)
    
    # Average inference time
    avg_inference_time = sum(inference_times) / len(inference_times)
    
    return {
        'categorical_acc': categorical_acc,
        'ordinal_acc': ordinal_acc,
        'qwk': qwk,
        'mae': mae,
        'prediction_consistency': pred_consistency,
        'ordinal_ranking': ordinal_ranking,
        'distribution_consistency': dist_consistency,
        'avg_inference_time': avg_inference_time
    }


def main():
    """Run comprehensive benchmark."""
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    dataset_name = "data/synthetic_OC/synthetic_oc"
    epochs = 10  # Longer training for better results
    
    print(f"🏁 COMPREHENSIVE BENCHMARK WITH ALL 7 METRICS")
    print(f"Dataset: {dataset_name}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print("="*60)
    
    # Load data
    print("📊 Loading data...")
    train_questions, train_responses, test_questions, test_responses = load_gpcm_data(dataset_name)
    
    # Create data loaders
    train_loader = UnifiedDataLoader(train_questions, train_responses, batch_size=32, shuffle=True, device=device)
    test_loader = UnifiedDataLoader(test_questions, test_responses, batch_size=32, shuffle=False, device=device)
    
    # Get number of questions
    n_questions = max(max(max(seq) for seq in train_questions), max(max(seq) for seq in test_questions))
    
    print(f"Data: {len(train_questions)} train, {len(test_questions)} test")
    print(f"Questions: {n_questions}, Categories: 4")
    
    # Get configurations
    configs = get_preset_configs()
    model_types = ['baseline', 'akvmn', 'deep_integration']
    
    results = {}
    
    for model_type in model_types:
        print(f"\n🚀 Benchmarking {model_type.upper()}...")
        
        try:
            # Create model
            config = configs[model_type]
            config.n_questions = n_questions
            model = create_model(config, n_questions, device)
            
            # Print model summary
            param_count = sum(p.numel() for p in model.parameters())
            print(f"Parameters: {param_count:,}")
            
            # Train model
            print("🏋️ Training...")
            best_acc = train_model_simple(model, train_loader, test_loader, config, device, epochs)
            
            # Comprehensive evaluation
            print("📊 Comprehensive evaluation...")
            metrics = evaluate_comprehensive(model, test_loader, device)
            
            # Store results
            results[model_type] = {
                'parameters': param_count,
                'best_accuracy': best_acc,
                'metrics': metrics,
                'config': {
                    'model_type': model_type,
                    'embed_dim': getattr(config, 'embed_dim', None),
                    'n_cycles': getattr(config, 'n_cycles', None)
                }
            }
            
            print(f"✅ {model_type.upper()} completed")
            print(f"   Categorical Acc: {metrics['categorical_acc']:.4f}")
            print(f"   Ordinal Acc: {metrics['ordinal_acc']:.4f}")
            print(f"   QWK: {metrics['qwk']:.4f}")
            print(f"   Parameters: {param_count:,}")
            
        except Exception as e:
            print(f"❌ {model_type.upper()} failed: {e}")
            import traceback
            traceback.print_exc()
    
    # Print comprehensive summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE BENCHMARK RESULTS")
    print(f"{'='*80}")
    
    # Print header
    print(f"{'Model':<15} {'Params':<10} {'Cat.Acc':<8} {'Ord.Acc':<8} {'QWK':<8} {'MAE':<8} {'Pred.Cons':<10} {'Ord.Rank':<9} {'Dist.Cons':<9} {'Time(ms)':<8}")
    print("-" * 80)
    
    # Print results for each model
    for model_name, result in results.items():
        metrics = result['metrics']
        print(f"{model_name.title():<15} "
              f"{result['parameters']:<10,} "
              f"{metrics['categorical_acc']:<8.4f} "
              f"{metrics['ordinal_acc']:<8.4f} "
              f"{metrics['qwk']:<8.3f} "
              f"{metrics['mae']:<8.3f} "
              f"{metrics['prediction_consistency']:<10.3f} "
              f"{metrics['ordinal_ranking']:<9.3f} "
              f"{metrics['distribution_consistency']:<9.3f} "
              f"{metrics['avg_inference_time']:<8.1f}")
    
    # Save results
    os.makedirs('results/benchmark', exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f'results/benchmark/comprehensive_benchmark_{timestamp}.json'
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    # Find best model
    best_model = max(results.keys(), key=lambda x: results[x]['metrics']['categorical_acc'])
    best_acc = results[best_model]['metrics']['categorical_acc']
    
    print(f"🏆 Best Model: {best_model.title()} ({best_acc:.4f} categorical accuracy)")
    
    return results


if __name__ == "__main__":
    main()