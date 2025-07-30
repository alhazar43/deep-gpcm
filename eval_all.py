#!/usr/bin/env python3
"""
Comprehensive Evaluation Script for Advanced Deep-GPCM Visualizations
Extends existing evaluation to collect all data needed for advanced plots.
"""

import os
# Fix Intel MKL threading issue
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import torch
import torch.nn.functional as F
import json
import time
import argparse
import numpy as np
from datetime import datetime
from collections import defaultdict
from sklearn.metrics import confusion_matrix

# Import existing functionality
from evaluate import (
    load_trained_model, load_simple_data, create_data_loaders, 
    pad_sequence_batch, print_evaluation_summary
)
from utils.metrics import compute_metrics, save_results, ensure_results_dirs, OrdinalMetrics


class AdvancedEvaluator:
    """Extended evaluator that collects data for all advanced visualizations."""
    
    def __init__(self, n_cats=4):
        self.n_cats = n_cats
        self.ordinal_metrics = OrdinalMetrics(n_cats)
        
    def evaluate_model_comprehensive(self, model, test_loader, device, collect_sequences=True):
        """Comprehensive evaluation collecting all visualization data."""
        print(f"\n🧪 COMPREHENSIVE EVALUATION WITH ADVANCED DATA COLLECTION")
        print("-" * 70)
        
        model.eval()
        
        # Storage for all data types
        all_predictions = []
        all_probabilities = []
        all_targets = []
        all_sequences = []
        all_responses = []
        sequence_predictions = []
        sequence_targets = []
        inference_times = []
        attention_weights = []  # Store attention weights from DKVMN
        
        # For sequence-level performance analysis
        accuracy_by_position = defaultdict(list)
        
        print(f"Processing {len(test_loader)} batches...")
        
        # Hook to capture attention weights from DKVMN
        captured_attention = []
        def attention_hook(module, input, output):
            if hasattr(module, 'correlation_weight') and 'MemoryHeadGroup' in str(module.__class__):
                # This is a memory head, capture the correlation weights if they exist
                if len(input) >= 2:  # embedded_query_vector, key_memory_matrix
                    query_vector, key_matrix = input[0], input[1]
                    if query_vector is not None and key_matrix is not None:
                        corr_weights = F.softmax(torch.matmul(query_vector, key_matrix.t()), dim=1)
                        captured_attention.append(corr_weights.detach().cpu().numpy())
        
        # Register hooks for attention capture (try to find memory components)
        hooks = []
        for name, module in model.named_modules():
            if 'memory' in name.lower() or 'head' in name.lower():
                if hasattr(module, 'correlation_weight'):
                    hook = module.register_forward_hook(attention_hook)
                    hooks.append(hook)
        
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
                
                # Get predictions
                batch_predictions = gpcm_probs.argmax(dim=-1)
                
                # Collect batch data
                batch_size, seq_len = questions.shape
                
                for i in range(batch_size):
                    # Get valid sequence length
                    valid_mask = mask[i].bool()
                    valid_len = valid_mask.sum().item()
                    
                    if valid_len > 0:
                        # Sequence-level data
                        seq_questions = questions[i, :valid_len].cpu().numpy()
                        seq_responses = responses[i, :valid_len].cpu().numpy()
                        seq_predictions = batch_predictions[i, :valid_len].cpu().numpy()
                        seq_probabilities = gpcm_probs[i, :valid_len].cpu().numpy()
                        
                        # Store sequences for transition analysis
                        if collect_sequences:
                            all_sequences.append(seq_questions.tolist())
                            all_responses.append(seq_responses.tolist())
                            sequence_predictions.append(seq_predictions.tolist())
                            sequence_targets.append(seq_responses.tolist())
                        
                        # Position-based accuracy analysis
                        for pos in range(min(valid_len, 20)):  # Limit to first 20 positions
                            if pos < len(seq_responses) and pos < len(seq_predictions):
                                is_correct = seq_responses[pos] == seq_predictions[pos]
                                accuracy_by_position[pos].append(is_correct)
                        
                        # Flatten for overall metrics
                        all_targets.extend(seq_responses)
                        all_predictions.extend(seq_predictions)
                        all_probabilities.extend(seq_probabilities)
                
                # Try to extract attention weights if available
                if hasattr(model, 'attention_weights') and model.attention_weights is not None:
                    attention_weights.append(model.attention_weights.cpu().numpy())
                elif hasattr(model, 'memory') and hasattr(model.memory, 'attention_weights'):
                    if model.memory.attention_weights is not None:
                        attention_weights.append(model.memory.attention_weights.cpu().numpy())
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"  Processed {batch_idx + 1}/{len(test_loader)} batches")
        
        # Convert to numpy arrays
        all_targets = np.array(all_targets)
        all_predictions = np.array(all_predictions)
        all_probabilities = np.array(all_probabilities)
        
        print(f"📊 Total samples: {len(all_targets)}")
        print(f"⏱️  Average inference time: {np.mean(inference_times)*1000:.2f}ms per batch")
        
        # Calculate comprehensive metrics
        print("\n🔬 Computing comprehensive metrics...")
        eval_metrics = compute_metrics(all_targets, all_predictions, all_probabilities, n_cats=self.n_cats)
        
        # Calculate additional data for visualizations
        print("🎨 Generating advanced visualization data...")
        
        # 1. Confusion Matrix
        conf_matrix = confusion_matrix(all_targets, all_predictions, labels=list(range(self.n_cats)))
        
        # 2. Ordinal distances
        ordinal_distances = np.abs(all_targets - all_predictions)
        
        # 3. Category transition matrix
        transition_matrix = self._calculate_transition_matrix(all_responses) if collect_sequences else None
        
        # 4. Position-based accuracy
        position_accuracy = {}
        for pos, accuracies in accuracy_by_position.items():
            if accuracies:
                position_accuracy[pos] = np.mean(accuracies)
        
        # 5. Average attention weights
        avg_attention = None
        if attention_weights:
            try:
                # Stack and average attention weights
                stacked_attention = np.stack(attention_weights)
                avg_attention = np.mean(stacked_attention, axis=0)
            except:
                print("⚠️  Could not process attention weights")
        
        # Compile comprehensive results
        results = eval_metrics.copy()
        
        # Add visualization data
        results['confusion_matrix'] = conf_matrix.tolist()
        results['ordinal_distances'] = ordinal_distances.tolist()
        results['probabilities'] = all_probabilities.tolist()
        results['predictions'] = all_predictions.tolist()
        results['actual'] = all_targets.tolist()
        
        if transition_matrix is not None:
            results['transition_matrix'] = transition_matrix.tolist()
        
        if collect_sequences:
            results['sequences'] = all_sequences[:100]  # Limit size
            results['sequence_responses'] = all_responses[:100]
            results['sequence_predictions'] = sequence_predictions[:100]
        
        results['accuracy_by_position'] = position_accuracy
        
        if avg_attention is not None:
            results['attention_weights'] = avg_attention.tolist()
        
        # Performance metrics
        results['performance'] = {
            'avg_inference_time_ms': np.mean(inference_times) * 1000,
            'total_samples': len(all_targets),
            'samples_per_second': len(all_targets) / sum(inference_times),
            'total_sequences': len(all_sequences) if collect_sequences else 0
        }
        
        # Cleanup hooks
        for hook in hooks:
            hook.remove()
        
        return results
    
    def _calculate_transition_matrix(self, sequences):
        """Calculate category transition matrix from response sequences."""
        transitions = np.zeros((self.n_cats, self.n_cats))
        
        for seq_responses in sequences:
            for i in range(len(seq_responses) - 1):
                current_cat = seq_responses[i]
                next_cat = seq_responses[i + 1]
                if 0 <= current_cat < self.n_cats and 0 <= next_cat < self.n_cats:
                    transitions[current_cat, next_cat] += 1
        
        return transitions


def evaluate_all_models(model_paths, dataset='synthetic_OC', batch_size=32, device=None):
    """Evaluate multiple models with comprehensive data collection."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("=" * 80)
    print("COMPREHENSIVE DEEP-GPCM MODEL EVALUATION")
    print("=" * 80)
    print(f"Models: {len(model_paths)}")
    print(f"Dataset: {dataset}")
    print(f"Device: {device}")
    print(f"Batch size: {batch_size}")
    print()
    
    # Ensure results directories exist
    ensure_results_dirs()
    
    # Load test data once
    train_path = f"data/{dataset}/{dataset.lower()}_train.txt"
    test_path = f"data/{dataset}/{dataset.lower()}_test.txt"
    
    try:
        train_data, test_data, n_questions, n_cats = load_simple_data(train_path, test_path)
        print(f"📊 Test data loaded: {len(test_data)} sequences")
    except FileNotFoundError:
        print(f"❌ Dataset {dataset} not found")
        return []
    
    # Create test data loader
    _, test_loader = create_data_loaders(train_data, test_data, batch_size)
    
    # Initialize evaluator
    evaluator = AdvancedEvaluator(n_cats)
    
    results = []
    
    for model_path in model_paths:
        print(f"\n{'='*60}")
        print(f"EVALUATING: {model_path}")
        print(f"{'='*60}")
        
        try:
            # Load model
            model, config, training_metrics = load_trained_model(model_path, device)
            
            # Verify compatibility
            if n_questions != config['n_questions'] or n_cats != config['n_cats']:
                print(f"⚠️  WARNING: Data dimensions mismatch!")
                print(f"   Model expects: {config['n_questions']} questions, {config['n_cats']} categories")
                print(f"   Data contains: {n_questions} questions, {n_cats} categories")
            
            # Comprehensive evaluation
            eval_results = evaluator.evaluate_model_comprehensive(model, test_loader, device)
            
            # Print summary
            model_name = config['model_type']
            print_evaluation_summary(eval_results, model_name)
            
            # Prepare output data
            output_data = {
                'timestamp': datetime.now().isoformat(),
                'model_path': model_path,
                'config': config,
                'dataset': dataset,
                'evaluation_config': {
                    'batch_size': batch_size,
                    'device': str(device),
                    'comprehensive': True
                },
                'training_metrics': training_metrics,
                'evaluation_results': eval_results
            }
            
            # Save results
            model_name = config['model_type']
            dataset_name = config.get('dataset', dataset)
            filename = f"comprehensive_test_{model_name}_{dataset_name}.json"
            
            output_path = save_results(output_data, f"results/test/{filename}")
            print(f"💾 Comprehensive results saved to: {output_path}")
            
            results.append(output_data)
            
        except Exception as e:
            print(f"❌ Error evaluating {model_path}: {e}")
            continue
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Comprehensive Deep-GPCM Model Evaluation')
    parser.add_argument('--model_paths', nargs='+', required=True, 
                       help='Paths to trained models')
    parser.add_argument('--dataset', default='synthetic_OC', 
                       help='Dataset name for evaluation')
    parser.add_argument('--batch_size', type=int, default=32, 
                       help='Batch size for evaluation')
    parser.add_argument('--device', default=None, 
                       help='Device (cuda/cpu)')
    parser.add_argument('--generate_plots', action='store_true',
                       help='Generate plots after evaluation')
    
    args = parser.parse_args()
    
    # Device setup
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Evaluate all models
    results = evaluate_all_models(
        args.model_paths, 
        args.dataset, 
        args.batch_size, 
        device
    )
    
    if results:
        print(f"\n✅ Successfully evaluated {len(results)} models!")
        
        # Generate plots if requested
        if args.generate_plots:
            print("\n🎨 Generating comprehensive visualizations...")
            try:
                from utils.plot_metrics import plot_all_results
                generated_plots = plot_all_results()
                print(f"Generated {len(generated_plots)} plots!")
            except ImportError:
                print("❌ Could not import plotting utilities")
            except Exception as e:
                print(f"❌ Error generating plots: {e}")
    else:
        print("❌ No models were successfully evaluated")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())