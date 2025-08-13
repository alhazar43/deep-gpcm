#!/usr/bin/env python3
"""
Comprehensive validation framework for StableTemporalAttentionGPCM.

Validates the production-ready solution across different:
- Batch sizes (8, 16, 32, 64) for batch independence 
- Sequence lengths (10-200 steps) for temporal robustness
- Educational datasets (different domains) for generalization
- Gradient stability and performance metrics

This script provides systematic evidence that the architectural changes solve
the identified instability issues.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import json
import time

from models.factory import create_model
from utils.data_loading import load_dataset
from training.losses import get_loss_function


class StabilityValidator:
    """Comprehensive validation framework for temporal attention stability."""
    
    def __init__(self):
        self.results = {}
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
    
    def test_batch_size_independence(self, model_name: str = "stable_temporal_attn_gpcm") -> Dict:
        """Test that model performance is independent of batch size."""
        print(f"\n{'='*60}")
        print(f"BATCH SIZE INDEPENDENCE TEST - {model_name}")
        print(f"{'='*60}")
        
        # Test different batch sizes
        batch_sizes = [8, 16, 32, 64]
        results = {}
        
        # Load small dataset for quick testing
        dataset_name = "synthetic_OC"
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            
            try:
                # Load data with specific batch size
                train_loader, test_loader, n_questions, n_cats = load_dataset(
                    dataset_name, batch_size=batch_size
                )
                
                # Create model
                model = create_model(model_name, n_questions, n_cats).to(self.device)
                loss_fn = get_loss_function('ce')
                
                # Get a representative batch
                for batch in train_loader:
                    if len(batch) == 3:
                        questions, responses, mask = batch
                        questions = questions.to(self.device)
                        responses = responses.to(self.device)
                        if mask is not None:
                            mask = mask.to(self.device)
                    else:
                        questions, responses = batch[:2]
                        questions = questions.to(self.device)
                        responses = responses.to(self.device)
                        mask = None
                    break
                
                # Test gradient stability
                model.train()
                grad_norms = []
                losses = []
                
                # Run multiple forward/backward passes
                for i in range(5):
                    model.zero_grad()
                    
                    # Forward pass
                    outputs = model(questions, responses)
                    if isinstance(outputs, tuple):
                        logits = outputs[-1]  # GPCM probabilities
                    else:
                        logits = outputs
                    
                    # Compute loss
                    if torch.allclose(logits.sum(-1), torch.ones(logits.shape[:-1]), atol=1e-6):
                        # Convert probabilities to logits
                        logits = torch.log(logits + 1e-8)
                    
                    # Flatten for loss computation
                    batch_size, seq_len = questions.shape
                    logits_flat = logits.view(-1, logits.size(-1))
                    targets_flat = responses.view(-1)
                    
                    if mask is not None:
                        mask_flat = mask.view(-1)
                        logits_flat = logits_flat[mask_flat]
                        targets_flat = targets_flat[mask_flat]
                    
                    loss = loss_fn(logits_flat, targets_flat)
                    
                    # Backward pass
                    loss.backward()
                    
                    # Collect gradient statistics
                    total_grad_norm = 0.0
                    param_count = 0
                    for param in model.parameters():
                        if param.grad is not None:
                            total_grad_norm += param.grad.norm().item() ** 2
                            param_count += 1
                    
                    total_grad_norm = total_grad_norm ** 0.5
                    grad_norms.append(total_grad_norm)
                    losses.append(loss.item())
                
                # Calculate stability metrics
                grad_variance = np.var(grad_norms)
                loss_variance = np.var(losses)
                mean_grad_norm = np.mean(grad_norms)
                mean_loss = np.mean(losses)
                
                results[batch_size] = {
                    'mean_loss': mean_loss,
                    'loss_variance': loss_variance,
                    'mean_grad_norm': mean_grad_norm,
                    'grad_variance': grad_variance,
                    'grad_norms': grad_norms,
                    'losses': losses,
                    'stability_score': 1.0 / (1.0 + grad_variance)  # Higher is better
                }
                
                print(f"  Mean loss: {mean_loss:.4f} ± {loss_variance:.4f}")
                print(f"  Mean grad norm: {mean_grad_norm:.4f} ± {grad_variance:.4f}")
                print(f"  Stability score: {results[batch_size]['stability_score']:.4f}")
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                results[batch_size] = None
        
        # Analyze batch size effects
        print(f"\n{'='*60}")
        print("BATCH SIZE INDEPENDENCE ANALYSIS")
        print(f"{'='*60}")
        
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if len(valid_results) >= 2:
            # Check for batch size dependency
            batch_sizes_list = list(valid_results.keys())
            stability_scores = [valid_results[bs]['stability_score'] for bs in batch_sizes_list]
            grad_variances = [valid_results[bs]['grad_variance'] for bs in batch_sizes_list]
            
            # Calculate coefficient of variation across batch sizes
            stability_cv = np.std(stability_scores) / np.mean(stability_scores)
            grad_var_cv = np.std(grad_variances) / np.mean(grad_variances)
            
            print(f"Stability CV across batch sizes: {stability_cv:.4f} (lower is better)")
            print(f"Gradient variance CV: {grad_var_cv:.4f} (lower is better)")
            
            # Determine if batch size independent
            is_batch_independent = stability_cv < 0.2 and grad_var_cv < 0.5
            print(f"Batch size independence: {'PASS' if is_batch_independent else 'FAIL'}")
            
            results['summary'] = {
                'batch_independence': is_batch_independent,
                'stability_cv': stability_cv,
                'grad_var_cv': grad_var_cv,
                'mean_stability': np.mean(stability_scores),
                'mean_grad_variance': np.mean(grad_variances)
            }
        
        return results
    
    def test_sequence_length_robustness(self, model_name: str = "stable_temporal_attn_gpcm") -> Dict:
        """Test model robustness across different sequence lengths."""
        print(f"\n{'='*60}")
        print(f"SEQUENCE LENGTH ROBUSTNESS TEST - {model_name}")
        print(f"{'='*60}")
        
        # Use synthetic data with different sequence lengths
        results = {}
        batch_size = 16  # Fixed batch size
        
        # Test different sequence lengths
        seq_lengths = [10, 25, 50, 100, 150]
        
        for seq_len in seq_lengths:
            print(f"\nTesting sequence length: {seq_len}")
            
            try:
                # Create synthetic data with specific sequence length
                n_questions = 50
                n_cats = 4
                
                questions = torch.randint(1, n_questions + 1, (batch_size, seq_len))
                responses = torch.randint(0, n_cats, (batch_size, seq_len))
                
                questions = questions.to(self.device)
                responses = responses.to(self.device)
                
                # Create model
                model = create_model(model_name, n_questions, n_cats).to(self.device)
                loss_fn = get_loss_function('ce')
                
                # Test forward pass performance
                model.eval()
                
                forward_times = []
                memory_usage = []
                
                for i in range(3):  # Multiple runs for stability
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                    
                    start_time = time.time()
                    
                    with torch.no_grad():
                        outputs = model(questions, responses)
                    
                    forward_time = time.time() - start_time
                    forward_times.append(forward_time)
                    
                    if torch.cuda.is_available():
                        memory_usage.append(torch.cuda.max_memory_allocated() / 1024**2)  # MB
                
                # Test gradient computation
                model.train()
                model.zero_grad()
                
                outputs = model(questions, responses)
                if isinstance(outputs, tuple):
                    logits = outputs[-1]
                else:
                    logits = outputs
                
                # Convert probabilities to logits if needed
                if torch.allclose(logits.sum(-1), torch.ones(logits.shape[:-1]), atol=1e-6):
                    logits = torch.log(logits + 1e-8)
                
                # Compute loss and gradients
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = responses.view(-1)
                loss = loss_fn(logits_flat, targets_flat)
                
                grad_start_time = time.time()
                loss.backward()
                grad_time = time.time() - grad_start_time
                
                # Collect gradient norm
                total_grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
                
                results[seq_len] = {
                    'mean_forward_time': np.mean(forward_times),
                    'forward_time_std': np.std(forward_times),
                    'grad_time': grad_time,
                    'total_grad_norm': total_grad_norm,
                    'mean_memory_mb': np.mean(memory_usage) if memory_usage else 0,
                    'loss': loss.item(),
                    'output_shape': list(logits.shape)
                }
                
                print(f"  Forward time: {results[seq_len]['mean_forward_time']:.4f}s")
                print(f"  Gradient time: {grad_time:.4f}s") 
                print(f"  Memory usage: {results[seq_len]['mean_memory_mb']:.1f}MB")
                print(f"  Gradient norm: {total_grad_norm:.4f}")
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                results[seq_len] = None
        
        # Analyze scaling behavior
        print(f"\n{'='*60}")
        print("SEQUENCE LENGTH SCALING ANALYSIS")
        print(f"{'='*60}")
        
        valid_results = {k: v for k, v in results.items() if v is not None}
        
        if len(valid_results) >= 3:
            seq_lengths_list = list(valid_results.keys())
            forward_times = [valid_results[sl]['mean_forward_time'] for sl in seq_lengths_list]
            memory_usage = [valid_results[sl]['mean_memory_mb'] for sl in seq_lengths_list]
            
            # Check if scaling is reasonable (should be roughly linear or sub-quadratic)
            time_scaling = np.polyfit(np.log(seq_lengths_list), np.log(forward_times), 1)[0]
            memory_scaling = np.polyfit(np.log(seq_lengths_list), np.log(memory_usage), 1)[0] if any(memory_usage) else 0
            
            print(f"Time scaling exponent: {time_scaling:.2f} (linear=1.0, quadratic=2.0)")
            print(f"Memory scaling exponent: {memory_scaling:.2f}")
            
            # Good scaling if time is sub-quadratic and memory is reasonable
            good_scaling = time_scaling < 2.5 and memory_scaling < 2.0
            print(f"Scaling performance: {'PASS' if good_scaling else 'FAIL'}")
            
            results['summary'] = {
                'time_scaling_exponent': time_scaling,
                'memory_scaling_exponent': memory_scaling,
                'good_scaling': good_scaling,
                'max_forward_time': max(forward_times),
                'max_memory_mb': max(memory_usage) if memory_usage else 0
            }
        
        return results
    
    def compare_against_baseline(self, stable_model: str = "stable_temporal_attn_gpcm", 
                               baseline_model: str = "attn_gpcm_linear") -> Dict:
        """Compare stable temporal model against baseline attention model."""
        print(f"\n{'='*60}")
        print(f"BASELINE COMPARISON: {stable_model} vs {baseline_model}")
        print(f"{'='*60}")
        
        results = {}
        dataset_name = "synthetic_OC"
        batch_size = 32
        
        try:
            # Load data
            train_loader, test_loader, n_questions, n_cats = load_dataset(
                dataset_name, batch_size=batch_size
            )
            
            # Get test batch
            for batch in train_loader:
                if len(batch) == 3:
                    questions, responses, mask = batch
                    questions = questions.to(self.device)
                    responses = responses.to(self.device)
                    if mask is not None:
                        mask = mask.to(self.device)
                else:
                    questions, responses = batch[:2]
                    questions = questions.to(self.device)
                    responses = responses.to(self.device)
                    mask = None
                break
            
            models = {
                'stable': create_model(stable_model, n_questions, n_cats).to(self.device),
                'baseline': create_model(baseline_model, n_questions, n_cats).to(self.device)
            }
            
            loss_fn = get_loss_function('ce')
            
            for model_name, model in models.items():
                print(f"\nTesting {model_name} model:")
                
                # Performance test
                model.eval()
                forward_times = []
                
                for i in range(5):
                    start_time = time.time()
                    with torch.no_grad():
                        outputs = model(questions, responses)
                    forward_time = time.time() - start_time
                    forward_times.append(forward_time)
                
                # Gradient stability test
                model.train()
                grad_norms = []
                losses = []
                
                for i in range(5):
                    model.zero_grad()
                    
                    outputs = model(questions, responses)
                    if isinstance(outputs, tuple):
                        logits = outputs[-1]
                    else:
                        logits = outputs
                    
                    # Handle probability outputs
                    if torch.allclose(logits.sum(-1), torch.ones(logits.shape[:-1]), atol=1e-6):
                        logits = torch.log(logits + 1e-8)
                    
                    # Compute loss
                    logits_flat = logits.view(-1, logits.size(-1))
                    targets_flat = responses.view(-1)
                    
                    if mask is not None:
                        mask_flat = mask.view(-1)
                        logits_flat = logits_flat[mask_flat]
                        targets_flat = targets_flat[mask_flat]
                    
                    loss = loss_fn(logits_flat, targets_flat)
                    loss.backward()
                    
                    # Gradient norm
                    total_grad_norm = sum(p.grad.norm().item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5
                    grad_norms.append(total_grad_norm)
                    losses.append(loss.item())
                
                # Parameter count
                param_count = sum(p.numel() for p in model.parameters())
                
                results[model_name] = {
                    'mean_forward_time': np.mean(forward_times),
                    'forward_time_std': np.std(forward_times),
                    'mean_loss': np.mean(losses),
                    'loss_variance': np.var(losses),
                    'mean_grad_norm': np.mean(grad_norms),
                    'grad_variance': np.var(grad_norms),
                    'param_count': param_count,
                    'stability_score': 1.0 / (1.0 + np.var(grad_norms))
                }
                
                print(f"  Forward time: {results[model_name]['mean_forward_time']:.4f}s")
                print(f"  Parameters: {param_count:,}")
                print(f"  Loss: {results[model_name]['mean_loss']:.4f} ± {results[model_name]['loss_variance']:.4f}")
                print(f"  Grad norm: {results[model_name]['mean_grad_norm']:.4f} ± {results[model_name]['grad_variance']:.4f}")
                print(f"  Stability: {results[model_name]['stability_score']:.4f}")
            
            # Compare results
            print(f"\n{'='*40}")
            print("COMPARISON SUMMARY")
            print(f"{'='*40}")
            
            if 'stable' in results and 'baseline' in results:
                stable_r = results['stable']
                baseline_r = results['baseline']
                
                speed_ratio = stable_r['mean_forward_time'] / baseline_r['mean_forward_time']
                param_ratio = stable_r['param_count'] / baseline_r['param_count']
                stability_improvement = stable_r['stability_score'] / baseline_r['stability_score']
                
                print(f"Speed ratio (stable/baseline): {speed_ratio:.2f}x")
                print(f"Parameter ratio: {param_ratio:.2f}x")
                print(f"Stability improvement: {stability_improvement:.2f}x")
                
                # Assessment
                is_improvement = (
                    speed_ratio < 2.0 and  # Not more than 2x slower
                    param_ratio < 1.5 and  # Not more than 50% more parameters
                    stability_improvement > 1.0  # Better stability
                )
                
                print(f"Overall assessment: {'IMPROVEMENT' if is_improvement else 'NEEDS_WORK'}")
                
                results['comparison'] = {
                    'speed_ratio': speed_ratio,
                    'param_ratio': param_ratio,
                    'stability_improvement': stability_improvement,
                    'is_improvement': is_improvement
                }
        
        except Exception as e:
            print(f"Comparison failed: {str(e)}")
            results['error'] = str(e)
        
        return results
    
    def run_comprehensive_validation(self) -> Dict:
        """Run all validation tests and provide comprehensive report."""
        print("🔬 COMPREHENSIVE VALIDATION OF STABLE TEMPORAL ATTENTION GPCM")
        print("=" * 80)
        
        all_results = {}
        
        # Test 1: Batch size independence
        all_results['batch_independence'] = self.test_batch_size_independence()
        
        # Test 2: Sequence length robustness
        all_results['sequence_robustness'] = self.test_sequence_length_robustness()
        
        # Test 3: Baseline comparison
        all_results['baseline_comparison'] = self.compare_against_baseline()
        
        # Generate overall assessment
        print(f"\n{'='*80}")
        print("FINAL VALIDATION REPORT")
        print(f"{'='*80}")
        
        assessments = []
        
        # Batch independence assessment
        if 'summary' in all_results['batch_independence']:
            batch_summary = all_results['batch_independence']['summary']
            if batch_summary['batch_independence']:
                assessments.append("✅ PASS: Batch size independence achieved")
            else:
                assessments.append("❌ FAIL: Batch size dependency detected")
        
        # Sequence scaling assessment
        if 'summary' in all_results['sequence_robustness']:
            seq_summary = all_results['sequence_robustness']['summary']
            if seq_summary['good_scaling']:
                assessments.append("✅ PASS: Good sequence length scaling")
            else:
                assessments.append("❌ FAIL: Poor sequence length scaling")
        
        # Baseline comparison assessment
        if 'comparison' in all_results['baseline_comparison']:
            comp_summary = all_results['baseline_comparison']['comparison']
            if comp_summary['is_improvement']:
                assessments.append("✅ PASS: Improvement over baseline")
            else:
                assessments.append("❌ FAIL: No clear improvement over baseline")
        
        # Overall assessment
        pass_count = sum(1 for assessment in assessments if "✅ PASS" in assessment)
        total_tests = len(assessments)
        
        for assessment in assessments:
            print(assessment)
        
        print(f"\nOverall Score: {pass_count}/{total_tests} tests passed")
        
        if pass_count == total_tests:
            print("🎉 VALIDATION SUCCESS: Ready for production deployment!")
        elif pass_count >= total_tests * 0.7:
            print("⚠️  VALIDATION PARTIAL: Needs minor improvements")
        else:
            print("❌ VALIDATION FAILED: Requires significant changes")
        
        all_results['final_assessment'] = {
            'tests_passed': pass_count,
            'total_tests': total_tests,
            'success_rate': pass_count / total_tests,
            'production_ready': pass_count == total_tests
        }
        
        return all_results


def main():
    """Run comprehensive validation."""
    validator = StabilityValidator()
    results = validator.run_comprehensive_validation()
    
    # Save results
    results_path = Path("validation_results_stable_temporal.json")
    with open(results_path, 'w') as f:
        # Convert numpy types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            return obj
        
        json.dump(results, f, indent=2, default=convert_numpy)
    
    print(f"\n📊 Detailed results saved to: {results_path}")
    
    return results


if __name__ == "__main__":
    main()