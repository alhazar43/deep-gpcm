#!/usr/bin/env python3
"""
Extract Comprehensive Results for All Models
Creates a summary with all 7 metrics from saved training results.
"""

import json
import os
from datetime import datetime


def extract_best_metrics(results_file):
    """Extract best metrics from training results."""
    with open(results_file, 'r') as f:
        data = json.load(f)
    
    # Get best metrics using argmax method (most comparable to our historical results)
    best = data['best_metrics']['argmax']
    
    return {
        'categorical_accuracy': best['categorical_accuracy'],
        'ordinal_accuracy': best['ordinal_accuracy'], 
        'qwk': best['quadratic_weighted_kappa'],
        'mae': best['mean_absolute_error'],
        'prediction_consistency': best['prediction_consistency'],
        'ordinal_ranking': best['ordinal_ranking'],
        'distribution_consistency': best['distribution_consistency'],
        'avg_inference_time': data['best_metrics']['avg_inference_time'],
        'parameters': None  # Will be filled from config
    }


def main():
    """Extract comprehensive results."""
    print("📊 EXTRACTING COMPREHENSIVE RESULTS")
    print("="*50)
    
    # Model results to extract
    model_files = {
        'baseline': 'save_models/results_baseline_synthetic_OC.json',
        'akvmn': 'save_models/results_akvmn_synthetic_OC.json'
    }
    
    results = {}
    
    for model_name, file_path in model_files.items():
        if os.path.exists(file_path):
            print(f"📁 Processing {model_name}...")
            try:
                metrics = extract_best_metrics(file_path)
                
                # Extract parameter count from config
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    
                # Calculate parameters based on config (approximate)
                config = data['config']
                if model_name == 'baseline':
                    metrics['parameters'] = 133205  # From model summary
                elif model_name == 'akvmn': 
                    metrics['parameters'] = 153254  # From model summary
                
                results[model_name] = metrics
                print(f"✅ {model_name}: {metrics['categorical_accuracy']:.4f} categorical accuracy")
                
            except Exception as e:
                print(f"❌ Failed to process {model_name}: {e}")
        else:
            print(f"⚠️  {file_path} not found")
    
    # Now train deep_integration manually using the working AKVMN model as base
    print(f"\n🚀 Training Deep Integration manually...")
    
    import torch
    from config import get_preset_configs
    from model_factory import create_model
    from utils.gpcm_utils import load_gpcm_data
    from utils.data_utils import UnifiedDataLoader
    from utils.loss_utils import create_loss_function
    from evaluation.metrics import GpcmMetrics
    import torch.optim as optim
    import time
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load data
        train_questions, train_responses, test_questions, test_responses = load_gpcm_data("data/synthetic_OC/synthetic_oc")
        train_loader = UnifiedDataLoader(train_questions, train_responses, batch_size=32, shuffle=True, device=device)
        test_loader = UnifiedDataLoader(test_questions, test_responses, batch_size=32, shuffle=False, device=device)
        n_questions = max(max(max(seq) for seq in train_questions), max(max(seq) for seq in test_questions))
        
        # Create deep integration model
        config = get_preset_configs()['deep_integration']
        config.n_questions = n_questions
        model = create_model(config, n_questions, device)
        
        # Train for 3 epochs
        criterion = create_loss_function(config.loss_type, n_cats=config.n_cats).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
        
        best_metrics = None
        
        for epoch in range(3):
            # Train
            model.train()
            total_loss = 0.0
            correct = 0
            total = 0
            
            for questions, responses, mask in train_loader:
                optimizer.zero_grad()
                read_content, mastery_level, logits, probs = model(questions, responses)
                
                loss = criterion(probs, responses)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                predicted = probs.argmax(dim=-1)
                correct += (predicted == responses).sum().item()
                total += responses.numel()
            
            # Evaluate
            model.eval()
            all_predictions = []
            all_targets = []
            inference_times = []
            
            with torch.no_grad():
                for questions, responses, mask in test_loader:
                    start_time = time.time()
                    read_content, mastery_level, logits, probs = model(questions, responses)
                    inference_time = (time.time() - start_time) * 1000
                    inference_times.append(inference_time)
                    
                    probs_flat = probs.view(-1, 4)
                    responses_flat = responses.view(-1)
                    
                    all_predictions.append(probs_flat.cpu())
                    all_targets.append(responses_flat.cpu())
            
            # Combine predictions
            all_predictions = torch.cat(all_predictions, dim=0)
            all_targets = torch.cat(all_targets, dim=0)
            
            # Calculate metrics
            metrics_calc = GpcmMetrics(n_cats=4)
            
            predicted_classes = all_predictions.argmax(dim=-1)
            categorical_acc = (predicted_classes == all_targets).float().mean().item()
            ordinal_acc = metrics_calc.ordinal_accuracy(all_predictions, all_targets)
            qwk = metrics_calc.quadratic_weighted_kappa(all_predictions, all_targets)
            mae = metrics_calc.mean_absolute_error(all_predictions, all_targets)
            pred_consistency = metrics_calc.prediction_consistency(all_predictions, all_targets)
            ordinal_ranking = metrics_calc.ordinal_ranking_metric(all_predictions, all_targets)
            dist_consistency = metrics_calc.distribution_consistency(all_predictions, all_targets)
            avg_inference_time = sum(inference_times) / len(inference_times)
            
            current_metrics = {
                'categorical_accuracy': categorical_acc,
                'ordinal_accuracy': ordinal_acc,
                'qwk': qwk,
                'mae': mae,
                'prediction_consistency': pred_consistency,
                'ordinal_ranking': ordinal_ranking,
                'distribution_consistency': dist_consistency,
                'avg_inference_time': avg_inference_time,
                'parameters': sum(p.numel() for p in model.parameters())
            }
            
            if best_metrics is None or categorical_acc > best_metrics['categorical_accuracy']:
                best_metrics = current_metrics
            
            print(f"Epoch {epoch+1:3d}/3: Train Acc: {correct/total:.4f}, Test Acc: {categorical_acc:.4f}")
        
        results['deep_integration'] = best_metrics
        print(f"✅ deep_integration: {best_metrics['categorical_accuracy']:.4f} categorical accuracy")
        
    except Exception as e:
        print(f"❌ Failed to train deep_integration: {e}")
        import traceback
        traceback.print_exc()
    
    # Print comprehensive summary
    print(f"\n{'='*80}")
    print("COMPREHENSIVE BENCHMARK RESULTS - ALL 7 METRICS")
    print(f"{'='*80}")
    
    # Print header
    print(f"{'Model':<15} {'Params':<10} {'Cat.Acc':<8} {'Ord.Acc':<8} {'QWK':<8} {'MAE':<8} {'Pred.Cons':<10} {'Ord.Rank':<9} {'Dist.Cons':<9} {'Time(ms)':<8}")
    print("-" * 100)
    
    # Print results for each model
    for model_name, metrics in results.items():
        print(f"{model_name.title():<15} "
              f"{metrics['parameters']:<10,} "
              f"{metrics['categorical_accuracy']:<8.4f} "
              f"{metrics['ordinal_accuracy']:<8.4f} "
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
    if results:
        best_model = max(results.keys(), key=lambda x: results[x]['categorical_accuracy'])
        best_acc = results[best_model]['categorical_accuracy']
        print(f"🏆 Best Model: {best_model.title()} ({best_acc:.4f} categorical accuracy)")
    
    return results


if __name__ == "__main__":
    main()