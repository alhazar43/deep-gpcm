#!/usr/bin/env python3
"""
Unified Evaluation Script for Deep-GPCM Models
Tests any trained model on comprehensive metrics with auto-detection.
"""

import os

# Fix Intel MKL threading issue
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import torch
import json
import time
import argparse
import numpy as np
from datetime import datetime

from core.model import DeepGPCM, AttentionGPCM
from core.attention_enhanced import EnhancedAttentionGPCM
from utils.metrics import compute_metrics, save_results
import torch.utils.data as data_utils
import torch.nn as nn


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
        mask = [True] * q_len + [False] * (max_len - q_len)
        
        questions_padded.append(q_pad)
        responses_padded.append(r_pad)
        masks.append(mask)
    
    return (torch.tensor(questions_padded), 
            torch.tensor(responses_padded), 
            torch.tensor(masks, dtype=torch.bool))


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


def load_trained_model(model_path, device):
    """Load trained model with auto-detection."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    print(f"📂 Loading model from: {model_path}")
    
    # Load checkpoint
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")
    
    # Extract config
    if 'config' not in checkpoint:
        raise ValueError("Checkpoint missing config information")
    
    config = checkpoint['config']
    model_type = config['model_type']
    n_questions = config['n_questions']
    n_cats = config['n_cats']
    
    print(f"🔍 Auto-detected model: {model_type}")
    print(f"📊 Dataset config: {n_questions} questions, {n_cats} categories")
    
    # Create model based on type
    if model_type == 'deep_gpcm':
        model = DeepGPCM(
            n_questions=n_questions,
            n_cats=n_cats,
            memory_size=50,
            key_dim=50,
            value_dim=200,
            final_fc_dim=50
        )
    elif model_type == 'attn_gpcm':
        # Use enhanced version with learnable parameters
        model = EnhancedAttentionGPCM(
            n_questions=n_questions,
            n_cats=n_cats,
            embed_dim=64,
            memory_size=50,
            key_dim=50,
            value_dim=200,
            final_fc_dim=50,
            n_heads=4,
            n_cycles=2,
            embedding_strategy="linear_decay",
            ability_scale=2.0  # Default for old models
        )
    elif model_type == 'coral_gpcm':
        from core.coral_gpcm import HybridCORALGPCM
        model = HybridCORALGPCM(
            n_questions=n_questions,
            n_cats=n_cats,
            memory_size=50,
            key_dim=50,
            value_dim=200,
            final_fc_dim=50
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    # Load model state
    try:
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
    except Exception as e:
        raise RuntimeError(f"Failed to load model state: {e}")
    
    print(f"✅ Model loaded successfully")
    print(f"🔧 Parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Extract training metrics if available
    training_metrics = checkpoint.get('metrics', {})
    
    return model, config, training_metrics


def evaluate_model(model, test_loader, device):
    """Comprehensive model evaluation."""
    print(f"\\n🧪 EVALUATING MODEL")
    print("-" * 50)
    
    model.eval()
    all_predictions = []
    all_targets = []
    all_masks = []
    inference_times = []
    
    # Advanced data collection for plotting
    all_sequences = []
    all_responses = []
    attention_weights = []
    
    print(f"Processing {len(test_loader)} batches...")
    
    with torch.no_grad():
        for batch_idx, (questions, responses, mask) in enumerate(test_loader):
            questions = questions.to(device)
            responses = responses.to(device)
            mask = mask.to(device)
            
            # Time inference
            start_time = time.time()
            
            # Forward pass
            student_abilities, item_thresholds, discrimination_params, gpcm_probs = model(questions, responses)
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Collect sequence-level data for advanced analysis
            batch_size, seq_len = questions.shape
            for i in range(batch_size):
                seq_mask = mask[i].cpu().numpy()
                valid_len = int(seq_mask.sum())
                
                if valid_len > 0:
                    # Store sequence data
                    all_sequences.append(questions[i, :valid_len].cpu().numpy())
                    all_responses.append(responses[i, :valid_len].cpu().numpy())
            
            # Collect attention weights if available
            if hasattr(model, 'attention_weights') and model.attention_weights is not None:
                attention_weights.append(model.attention_weights.cpu().numpy())
            elif hasattr(model, 'memory') and hasattr(model.memory, 'attention_weights'):
                if model.memory.attention_weights is not None:
                    attention_weights.append(model.memory.attention_weights.cpu().numpy())
            
            # Collect predictions and targets (flatten but keep mask for filtering)
            probs_flat = gpcm_probs.view(-1, gpcm_probs.size(-1))
            responses_flat = responses.view(-1)
            mask_flat = mask.view(-1)
            
            all_predictions.append(probs_flat.cpu())
            all_targets.append(responses_flat.cpu())
            all_masks.append(mask_flat.cpu())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # Combine all predictions, targets, and masks
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    all_masks = torch.cat(all_masks, dim=0)
    
    # Filter out padding tokens using mask
    valid_indices = all_masks.bool()
    all_predictions = all_predictions[valid_indices]
    all_targets = all_targets[valid_indices]
    
    print(f"📊 Total samples: {len(all_targets)}")
    import numpy as np
    print(f"⏱️  Average inference time: {np.mean(inference_times)*1000:.2f}ms per batch")
    
    # Calculate comprehensive metrics using simplified system
    print("\\n🔬 Computing comprehensive metrics...")
    n_cats = all_predictions.size(-1)
    
    # Get predictions
    y_pred = all_predictions.argmax(dim=-1)
    
    # Compute comprehensive metrics
    eval_metrics = compute_metrics(all_targets, y_pred, all_predictions, n_cats=n_cats)
    
    results = eval_metrics.copy()
    
    # Add inference performance metrics
    results['performance'] = {
        'avg_inference_time_ms': np.mean(inference_times) * 1000,
        'total_samples': len(all_targets),
        'samples_per_second': len(all_targets) / sum(inference_times)
    }
    
    # Add confusion matrix (already computed in metrics)
    if 'confusion_matrix' not in results:
        confusion_matrix = torch.zeros(n_cats, n_cats, dtype=torch.int)
        for target, pred in zip(all_targets, y_pred):
            confusion_matrix[target, pred] += 1
        results['confusion_matrix'] = confusion_matrix.numpy().tolist()
    
    # Add advanced visualization data that plot_metrics.py expects
    # Ordinal distances for distance distribution plots
    ordinal_distances = torch.abs(all_targets - y_pred)
    results['ordinal_distances'] = ordinal_distances.numpy().tolist()
    
    # Store probability predictions for ROC and calibration analysis
    results['probability_predictions'] = all_predictions.numpy().tolist()
    results['probabilities'] = all_predictions.numpy().tolist()  # For plot_metrics.py compatibility
    results['actual'] = all_targets.numpy().tolist()  # For plot_metrics.py compatibility
    
    # Transition matrix calculation
    if all_responses:
        transition_matrix = _calculate_transition_matrix(all_responses, n_cats)
        results['transition_matrix'] = transition_matrix.tolist()
    
    # Sequence data for time series analysis
    if all_sequences and all_responses:
        results['sequences'] = all_sequences
        results['responses'] = all_responses
    
    # Attention weights if available
    if attention_weights:
        try:
            stacked_attention = np.stack(attention_weights)
            avg_attention = np.mean(stacked_attention, axis=0)
            results['attention_weights'] = avg_attention.tolist()
        except:
            pass  # Skip if attention weights have inconsistent shapes
    
    return results


def _calculate_transition_matrix(sequences, n_cats=4):
    """Calculate transition matrix from response sequences."""
    import numpy as np
    transitions = np.zeros((n_cats, n_cats), dtype=int)
    
    for sequence in sequences:
        for i in range(len(sequence) - 1):
            curr_resp = int(sequence[i])
            next_resp = int(sequence[i + 1])
            if 0 <= curr_resp < n_cats and 0 <= next_resp < n_cats:
                transitions[curr_resp, next_resp] += 1
    
    return transitions


def print_evaluation_summary(results, model_name):
    """Print formatted evaluation summary."""
    print(f"\\n{'='*60}")
    print(f"EVALUATION SUMMARY: {model_name}")
    print(f"{'='*60}")
    
    # Main metrics
    print(f"\\n📈 CORE METRICS:")
    core_metrics = ['categorical_accuracy', 'ordinal_accuracy', 'quadratic_weighted_kappa', 'mean_absolute_error']
    for metric in core_metrics:
        if metric in results:
            print(f"  {metric.replace('_', ' ').title():<25}: {results[metric]:.4f}")
    
    # Additional ordinal metrics
    ordinal_metrics = ['kendall_tau', 'spearman_correlation', 'cohen_kappa', 'ordinal_loss']
    ordinal_present = [m for m in ordinal_metrics if m in results]
    if ordinal_present:
        print(f"\\n🔄 ORDINAL METRICS:")
        for metric in ordinal_present:
            print(f"  {metric.replace('_', ' ').title():<25}: {results[metric]:.4f}")
    
    # Category breakdown
    cat_metrics = [k for k in results.keys() if k.startswith('cat_') and k.endswith('_accuracy')]
    if cat_metrics:
        print(f"\\n📊 CATEGORY BREAKDOWN:")
        for metric in sorted(cat_metrics):
            print(f"  {metric.replace('_', ' ').title():<25}: {results[metric]:.4f}")
    
    # Performance metrics
    if 'performance' in results:
        perf = results['performance']
        print(f"\\n⚡ PERFORMANCE METRICS:")
        print(f"  Average inference time:   {perf['avg_inference_time_ms']:.2f} ms/batch")
        print(f"  Samples per second:       {perf['samples_per_second']:.1f}")
        print(f"  Total samples evaluated:  {perf['total_samples']:,}")


def find_trained_models(models_dir: str = "save_models") -> dict:
    """Find all trained model files for batch evaluation."""
    from pathlib import Path
    
    models_dir = Path(models_dir)
    if not models_dir.exists():
        print(f"❌ Models directory not found: {models_dir}")
        return {}
    
    model_files = {}
    for model_file in models_dir.glob("*.pth"):
        model_name = model_file.stem
        # Expected format: best_modeltype_dataset.pth
        parts = model_name.split('_')
        if len(parts) >= 3 and parts[0] == 'best':
            # Handle the three main model types
            if len(parts) >= 3:
                if parts[1] == 'deep' and parts[2] == 'gpcm':
                    model_type = 'deep'
                    dataset = '_'.join(parts[3:])
                elif parts[1] == 'attn' and parts[2] == 'gpcm':
                    model_type = 'attn'
                    dataset = '_'.join(parts[3:])
                elif parts[1] == 'coral' and parts[2] == 'gpcm':
                    model_type = 'coral'
                    dataset = '_'.join(parts[3:])
                else:
                    # Fallback for other patterns
                    model_type = parts[1]
                    dataset = '_'.join(parts[2:])
            
            if dataset not in model_files:
                model_files[dataset] = {}
            model_files[dataset][model_type] = str(model_file)
    
    return model_files


def generate_evaluation_summary(results_dir: str = "results/test") -> dict:
    """Generate summary of all evaluation results."""
    from pathlib import Path
    
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return {}
    
    summary = {}
    for result_file in results_dir.glob("test_results_*.json"):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            model_type = data['config'].get('model_type', 'unknown')
            dataset = data['config'].get('dataset', 'unknown')
            
            if dataset not in summary:
                summary[dataset] = {}
            
            # Handle both old and new JSON structures
            if 'evaluation_results' in data:
                eval_results = data['evaluation_results']
                metrics = {
                    'categorical_accuracy': eval_results.get('categorical_accuracy', 0),
                    'ordinal_accuracy': eval_results.get('ordinal_accuracy', 0),
                    'quadratic_weighted_kappa': eval_results.get('quadratic_weighted_kappa', 0),
                    'mean_absolute_error': eval_results.get('mean_absolute_error', 0),
                    'total_samples': eval_results.get('performance', {}).get('total_samples', 0),
                    'timestamp': data.get('timestamp', 'unknown')
                }
            else:
                # New flattened structure
                metrics = {
                    'categorical_accuracy': data.get('categorical_accuracy', 0),
                    'ordinal_accuracy': data.get('ordinal_accuracy', 0),
                    'quadratic_weighted_kappa': data.get('quadratic_weighted_kappa', 0),
                    'mean_absolute_error': data.get('mean_absolute_error', 0),
                    'total_samples': data.get('performance', {}).get('total_samples', 0),
                    'timestamp': data.get('timestamp', 'unknown')
                }
            
            summary[dataset][model_type] = metrics
            
        except Exception as e:
            print(f"⚠️  Could not process {result_file}: {e}")
    
    return summary


def print_evaluation_summary(summary: dict):
    """Print formatted evaluation summary."""
    print(f"\n{'='*80}")
    print("EVALUATION SUMMARY")
    print(f"{'='*80}")
    
    for dataset, models in summary.items():
        print(f"\n📊 DATASET: {dataset}")
        print("-" * 50)
        
        if not models:
            print("  No evaluation results found")
            continue
        
        print(f"{'Model':<12} {'Cat.Acc':<8} {'Ord.Acc':<8} {'QWK':<8} {'MAE':<8} {'Samples':<8}")
        print("-" * 64)
        
        sorted_models = sorted(models.items(), 
                             key=lambda x: x[1].get('categorical_accuracy', 0), 
                             reverse=True)
        
        for model_type, metrics in sorted_models:
            cat_acc = metrics.get('categorical_accuracy', 0)
            ord_acc = metrics.get('ordinal_accuracy', 0)
            qwk = metrics.get('quadratic_weighted_kappa', 0)
            mae = metrics.get('mean_absolute_error', 0)
            samples = metrics.get('total_samples', 0)
            
            is_best = model_type == sorted_models[0][0]
            prefix = "🏆" if is_best else "  "
            
            print(f"{prefix}{model_type:<10} {cat_acc:.4f}   {ord_acc:.4f}   {qwk:.4f}   {mae:.4f}   {samples:>6,}")


def batch_evaluate_models(dataset_filter=None, model_filter=None, batch_size=32, device=None, regenerate_plots=False, include_cv_folds=False):
    """Batch evaluate all available models."""
    available_models = find_trained_models()
    
    if not available_models:
        print("❌ No trained models found in save_models/")
        return
    
    # Filter out CV fold models by default (they lack corresponding data directories)
    if not include_cv_folds:
        main_models = {}
        for dataset, models in available_models.items():
            if '_fold_' not in dataset:
                main_models[dataset] = models
        available_models = main_models
    
    # Apply dataset filter
    if dataset_filter:
        if dataset_filter in available_models:
            available_models = {dataset_filter: available_models[dataset_filter]}
        else:
            print(f"❌ Dataset '{dataset_filter}' not found in main models")
            return
    
    print(f"🔍 Found models for {len(available_models)} datasets:")
    for dataset, models in available_models.items():
        model_list = ', '.join(models.keys())
        print(f"  {dataset}: {model_list}")
    
    # Device setup
    if device:
        device_obj = torch.device(device)
    else:
        device_obj = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"\n🧪 BATCH EVALUATION")
    print(f"Device: {device_obj}, Batch size: {batch_size}")
    print("-" * 50)
    
    total_evaluations = 0
    successful_evaluations = 0
    
    for dataset, models in available_models.items():
        print(f"\n📊 Evaluating dataset: {dataset}")
        
        for model_type, model_path in models.items():
            if model_filter and model_type not in model_filter:
                continue
            
            total_evaluations += 1
            print(f"🧪 Evaluating: {model_type}")
            
            try:
                # Load and evaluate model
                model, config, training_metrics = load_trained_model(model_path, device_obj)
                train_path = f"data/{dataset}/{dataset.lower()}_train.txt"
                test_path = f"data/{dataset}/{dataset.lower()}_test.txt"
                
                train_data, test_data, n_questions, n_cats = load_simple_data(train_path, test_path)
                test_questions = [seq[0] for seq in test_data]
                test_responses = [seq[1] for seq in test_data]
                
                from utils.data_utils import UnifiedDataLoader
                test_loader = UnifiedDataLoader(test_questions, test_responses, 
                                              batch_size=batch_size, shuffle=False, device=device_obj)
                
                results = evaluate_model(model, test_loader, device_obj)
                results['config'] = config
                results['training_metrics'] = training_metrics
                
                # Save results
                result_file = f"results/test/test_results_{model_type}_{dataset}.json"
                from utils.metrics import save_results
                save_results(results, result_file)
                
                print_evaluation_summary({dataset: {model_type: {
                    'categorical_accuracy': results.get('categorical_accuracy', 0),
                    'ordinal_accuracy': results.get('ordinal_accuracy', 0),
                    'quadratic_weighted_kappa': results.get('quadratic_weighted_kappa', 0),
                    'mean_absolute_error': results.get('mean_absolute_error', 0),
                    'total_samples': results.get('performance', {}).get('total_samples', 0)
                }}})
                
                successful_evaluations += 1
                print(f"✅ Success: {model_type}")
                
            except Exception as e:
                print(f"❌ Failed: {model_type} - {e}")
    
    print(f"\n✅ BATCH EVALUATION COMPLETED")
    print(f"Total: {total_evaluations}, Successful: {successful_evaluations}, Failed: {total_evaluations - successful_evaluations}")
    
    if successful_evaluations > 0:
        print("\n📋 Final Summary:")
        summary = generate_evaluation_summary()
        print_evaluation_summary(summary)
        
        if regenerate_plots:
            print("\n🎨 Regenerating plots...")
            try:
                from utils.plot_metrics import plot_all_results
                generated_plots = plot_all_results()
                print(f"Generated {len(generated_plots)} plots")
            except Exception as e:
                print(f"⚠️  Plot generation failed: {e}")


def main():
    parser = argparse.ArgumentParser(description='Unified Deep-GPCM Model Evaluation')
    parser.add_argument('--model_path', help='Path to trained model (required for single evaluation)')
    parser.add_argument('--dataset', default='synthetic_OC', help='Dataset name for evaluation')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--device', default=None, help='Device (cuda/cpu)')
    
    # Batch evaluation options
    parser.add_argument('--all', action='store_true', help='Evaluate all available models (main models only)')
    parser.add_argument('--models', nargs='+', help='Specific models to evaluate (for batch mode)')
    parser.add_argument('--include_cv_folds', action='store_true', help='Include CV fold models in batch evaluation')
    parser.add_argument('--summary_only', action='store_true', help='Only show summary of existing results')
    parser.add_argument('--regenerate_plots', action='store_true', help='Regenerate all plots after evaluation')
    
    args = parser.parse_args()
    
    print("=" * 80)
    print("UNIFIED DEEP-GPCM MODEL EVALUATION")
    print("=" * 80)
    
    # Create results directory
    from utils.metrics import ensure_results_dirs
    ensure_results_dirs()
    
    # Handle batch evaluation modes
    if args.summary_only:
        print("📋 Generating summary from existing results...")
        summary = generate_evaluation_summary()
        print_evaluation_summary(summary)
        return
    
    if args.all:
        print("🚀 BATCH EVALUATION MODE")
        batch_evaluate_models(
            dataset_filter=args.dataset if args.dataset != 'synthetic_OC' else None,
            model_filter=args.models,
            batch_size=args.batch_size,
            device=args.device,
            regenerate_plots=args.regenerate_plots,
            include_cv_folds=args.include_cv_folds
        )
        return
    
    # Single model evaluation
    if not args.model_path:
        print("❌ --model_path is required for single model evaluation")
        print("Use --all for batch evaluation or --summary_only for results summary")
        return
    
    # Device setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Model path: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    try:
        # Load trained model
        model, config, training_metrics = load_trained_model(args.model_path, device)
        
        # Load test data
        train_path = f"data/{args.dataset}/{args.dataset.lower()}_train.txt"
        test_path = f"data/{args.dataset}/{args.dataset.lower()}_test.txt"
        
        try:
            train_data, test_data, n_questions, n_cats = load_simple_data(train_path, test_path)
            print(f"\\n📊 Test data loaded: {len(test_data)} sequences")
            
            # Verify data compatibility
            if n_questions != config['n_questions'] or n_cats != config['n_cats']:
                print(f"⚠️  WARNING: Data dimensions mismatch!")
                print(f"   Model expects: {config['n_questions']} questions, {config['n_cats']} categories")
                print(f"   Data contains: {n_questions} questions, {n_cats} categories")
                print(f"   Using model dimensions for evaluation...")
        except FileNotFoundError:
            print(f"❌ Dataset {args.dataset} not found at {test_path}")
            return
        
        # Create test data loader
        _, test_loader = create_data_loaders(train_data, test_data, args.batch_size)
        
        # Evaluate model
        results = evaluate_model(model, test_loader, device)
        
        # Print summary
        model_name = config['model_type']  
        # Create summary in expected format for single model
        summary = {args.dataset: {model_name: results}}
        print_evaluation_summary(summary)
        
        # Save detailed results
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'model_path': args.model_path,
            'config': config,
            'dataset': args.dataset,
            'evaluation_config': {
                'batch_size': args.batch_size,
                'device': str(device)
            },
            'training_metrics': training_metrics,
            'evaluation_results': results
        }
        
        # Save using simplified system
        model_name = config['model_type']
        dataset_name = config.get('dataset', args.dataset)
        
        filename = f"test_results_{model_name}_{dataset_name}.json"
        
        output_path = save_results(
            output_data, f"results/test/{filename}"
        )
        
        print(f"\\n💾 Results saved to: {output_path}")
        print("\\n✅ Evaluation completed successfully!")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        return 1
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())