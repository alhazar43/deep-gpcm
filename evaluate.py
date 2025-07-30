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
    if model_type == 'baseline':
        model = DeepGPCM(
            n_questions=n_questions,
            n_cats=n_cats,
            memory_size=50,
            key_dim=50,
            value_dim=200,
            final_fc_dim=50
        )
    elif model_type == 'akvmn':
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
    inference_times = []
    
    print(f"Processing {len(test_loader)} batches...")
    
    with torch.no_grad():
        for batch_idx, (questions, responses, mask) in enumerate(test_loader):
            questions = questions.to(device)
            responses = responses.to(device)
            
            # Time inference
            start_time = time.time()
            
            # Forward pass
            student_abilities, item_thresholds, discrimination_params, gpcm_probs = model(questions, responses)
            
            inference_time = time.time() - start_time
            inference_times.append(inference_time)
            
            # Collect predictions and targets
            probs_flat = gpcm_probs.view(-1, gpcm_probs.size(-1))
            responses_flat = responses.view(-1)
            
            all_predictions.append(probs_flat.cpu())
            all_targets.append(responses_flat.cpu())
            
            if (batch_idx + 1) % 10 == 0:
                print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")
    
    # Combine all predictions and targets
    all_predictions = torch.cat(all_predictions, dim=0)
    all_targets = torch.cat(all_targets, dim=0)
    
    print(f"📊 Total samples: {len(all_targets)}")
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
    
    return results


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


def main():
    parser = argparse.ArgumentParser(description='Unified Deep-GPCM Model Evaluation')
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--dataset', default='synthetic_OC', help='Dataset name for evaluation')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for evaluation')
    parser.add_argument('--device', default=None, help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print("UNIFIED DEEP-GPCM MODEL EVALUATION")
    print("=" * 80)
    print(f"Model path: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Device: {device}")
    print(f"Batch size: {args.batch_size}")
    print()
    
    # Create results directory
    from utils.metrics import ensure_results_dirs
    ensure_results_dirs()
    
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
        print_evaluation_summary(results, model_name)
        
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