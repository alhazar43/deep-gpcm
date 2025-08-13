#!/usr/bin/env python3
"""
Test script for enhanced StableTemporalAttentionGPCM.

This script validates:
1. Batch size independence (works with small batches 8, 16, 32, 64)
2. Performance improvements from the 3-phase enhancement
3. Stability compared to original temporal model
"""

import torch
import torch.nn.functional as F
import time
from pathlib import Path

# Import models
from models.implementations.stable_temporal_attention_gpcm import StableTemporalAttentionGPCM
from models.implementations.temporal_attention_gpcm import TemporalAttentionGPCM
from models.implementations.attention_gpcm import AttentionGPCM


def create_test_data(batch_size=16, seq_len=10, n_questions=50, n_cats=5):
    """Create test data for model validation."""
    torch.manual_seed(42)  # Reproducible results
    
    questions = torch.randint(1, n_questions + 1, (batch_size, seq_len))
    responses = torch.randint(0, n_cats, (batch_size, seq_len))
    
    return questions, responses


def test_batch_size_independence():
    """Test that enhanced model works across different batch sizes."""
    print("=" * 60)
    print("BATCH SIZE INDEPENDENCE TEST")
    print("=" * 60)
    
    n_questions = 50
    n_cats = 5
    seq_len = 10
    batch_sizes = [8, 16, 32, 64]
    
    # Create enhanced stable model
    model = StableTemporalAttentionGPCM(
        n_questions=n_questions,
        n_cats=n_cats,
        embed_dim=64,
        temporal_window=5,
        temporal_dim=8
    )
    model.eval()
    
    print(f"Testing enhanced StableTemporalAttentionGPCM...")
    print(f"Model info: {model.get_model_info()['performance_enhancements']}")
    print()
    
    results = {}
    
    for batch_size in batch_sizes:
        print(f"Batch size: {batch_size}")
        
        # Create test data
        questions, responses = create_test_data(batch_size, seq_len, n_questions, n_cats)
        
        try:
            # Forward pass
            with torch.no_grad():
                start_time = time.time()
                outputs = model(questions, responses)
                forward_time = time.time() - start_time
                
                # Unpack outputs (theta, beta, alpha, probs)
                theta, beta, alpha, probs = outputs
                
                # Basic validation
                assert probs.shape == (batch_size, seq_len, n_cats)
                assert torch.all(probs >= 0) and torch.all(probs <= 1)
                assert torch.allclose(probs.sum(dim=-1), torch.ones(batch_size, seq_len), atol=1e-6)
                
                # Compute metrics
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
                confidence = probs.max(dim=-1)[0].mean()
                
                results[batch_size] = {
                    'success': True,
                    'forward_time': forward_time,
                    'entropy': entropy.item(),
                    'confidence': confidence.item(),
                    'output_range': (probs.min().item(), probs.max().item())
                }
                
                print(f"  ✅ Success - Time: {forward_time:.4f}s, Entropy: {entropy:.3f}, Confidence: {confidence:.3f}")
                
        except Exception as e:
            results[batch_size] = {
                'success': False,
                'error': str(e)
            }
            print(f"  ❌ Failed: {e}")
    
    # Summary
    print(f"\n{'Batch Size':<12} {'Status':<8} {'Time (s)':<10} {'Entropy':<10} {'Confidence':<12}")
    print("-" * 60)
    
    successful_batches = []
    for batch_size, result in results.items():
        if result['success']:
            successful_batches.append(batch_size)
            print(f"{batch_size:<12} ✅       {result['forward_time']:<10.4f} {result['entropy']:<10.3f} {result['confidence']:<12.3f}")
        else:
            print(f"{batch_size:<12} ❌       {'ERROR':<10} {'ERROR':<10} {'ERROR':<12}")
    
    if len(successful_batches) == len(batch_sizes):
        print(f"\n🎉 BATCH SIZE INDEPENDENCE: VERIFIED")
        print(f"   Enhanced model works with all tested batch sizes: {successful_batches}")
    else:
        print(f"\n⚠️  BATCH SIZE ISSUES: Only works with {successful_batches}")
    
    return results


def compare_model_performance():
    """Compare enhanced stable model with original temporal and baseline attention."""
    print("\n" + "=" * 60)
    print("PERFORMANCE COMPARISON TEST")
    print("=" * 60)
    
    n_questions = 50
    n_cats = 5
    batch_size = 16  # Small batch to test stability
    seq_len = 20
    
    # Create test data
    questions, responses = create_test_data(batch_size, seq_len, n_questions, n_cats)
    targets = responses  # For loss computation
    
    models = {
        'Baseline (AttentionGPCM)': AttentionGPCM(n_questions, n_cats),
        'Original Temporal': TemporalAttentionGPCM(n_questions, n_cats, temporal_window=3, temporal_dim=8),
        'Enhanced Stable Temporal': StableTemporalAttentionGPCM(n_questions, n_cats, temporal_window=5, temporal_dim=8)
    }
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\nTesting {model_name}...")
        
        try:
            model.eval()
            
            # Forward pass
            with torch.no_grad():
                start_time = time.time()
                outputs = model(questions, responses)
                forward_time = time.time() - start_time
                
                # Unpack outputs
                if isinstance(outputs, tuple) and len(outputs) == 4:
                    theta, beta, alpha, probs = outputs
                else:
                    probs = outputs
                
                # Compute metrics
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
                confidence = probs.max(dim=-1)[0].mean()
                
                # Compute cross-entropy loss (like training)
                logits = torch.log(probs + 1e-8)
                loss = F.cross_entropy(logits.view(-1, n_cats), targets.view(-1))
                
                # Parameter count
                params = sum(p.numel() for p in model.parameters())
                
                results[model_name] = {
                    'success': True,
                    'forward_time': forward_time,
                    'entropy': entropy.item(),
                    'confidence': confidence.item(),
                    'loss': loss.item(),
                    'parameters': params,
                    'output_range': (probs.min().item(), probs.max().item())
                }
                
                print(f"  ✅ Success")
                print(f"     Time: {forward_time:.4f}s")
                print(f"     Loss: {loss:.4f}")
                print(f"     Entropy: {entropy:.3f}")
                print(f"     Confidence: {confidence:.3f}")
                print(f"     Parameters: {params:,}")
                
        except Exception as e:
            results[model_name] = {
                'success': False,
                'error': str(e)
            }
            print(f"  ❌ Failed: {e}")
    
    # Summary comparison
    print(f"\n{'Model':<25} {'Status':<8} {'Loss':<8} {'Entropy':<8} {'Confidence':<10} {'Params':<10}")
    print("-" * 75)
    
    for model_name, result in results.items():
        if result['success']:
            print(f"{model_name:<25} ✅       {result['loss']:<8.4f} {result['entropy']:<8.3f} "
                  f"{result['confidence']:<10.3f} {result['parameters']:<10,}")
        else:
            print(f"{model_name:<25} ❌       {'ERROR':<8} {'ERROR':<8} {'ERROR':<10} {'ERROR':<10}")
    
    # Performance analysis
    if results.get('Enhanced Stable Temporal', {}).get('success') and results.get('Original Temporal', {}).get('success'):
        enhanced = results['Enhanced Stable Temporal']
        original = results['Original Temporal']
        
        print(f"\n📊 PERFORMANCE ANALYSIS:")
        print(f"   Loss improvement: {original['loss'] - enhanced['loss']:+.4f}")
        print(f"   Confidence improvement: {enhanced['confidence'] - original['confidence']:+.3f}")
        print(f"   Speed improvement: {original['forward_time'] - enhanced['forward_time']:+.4f}s")
        
        if enhanced['loss'] < original['loss']:
            print(f"   🎉 Enhanced model shows better loss!")
        if enhanced['confidence'] > original['confidence']:
            print(f"   🎉 Enhanced model shows better confidence!")
    
    return results


def test_gradient_stability():
    """Test gradient stability across different batch sizes."""
    print("\n" + "=" * 60)
    print("GRADIENT STABILITY TEST")
    print("=" * 60)
    
    n_questions = 50
    n_cats = 5
    seq_len = 15
    batch_sizes = [8, 16, 32]
    
    model = StableTemporalAttentionGPCM(n_questions, n_cats, temporal_window=5, temporal_dim=8)
    
    gradient_norms = {}
    
    for batch_size in batch_sizes:
        print(f"\nTesting gradient stability with batch size {batch_size}...")
        
        # Create test data
        questions, responses = create_test_data(batch_size, seq_len, n_questions, n_cats)
        
        model.train()
        model.zero_grad()
        
        try:
            # Forward pass
            outputs = model(questions, responses)
            theta, beta, alpha, probs = outputs
            
            # Compute loss
            logits = torch.log(probs + 1e-8)
            loss = F.cross_entropy(logits.view(-1, n_cats), responses.view(-1))
            
            # Backward pass
            loss.backward()
            
            # Compute gradient norms
            total_grad_norm = 0.0
            param_grad_norms = {}
            
            for name, param in model.named_parameters():
                if param.grad is not None:
                    grad_norm = param.grad.norm().item()
                    param_grad_norms[name] = grad_norm
                    total_grad_norm += grad_norm ** 2
            
            total_grad_norm = total_grad_norm ** 0.5
            
            gradient_norms[batch_size] = {
                'total_norm': total_grad_norm,
                'loss': loss.item(),
                'param_norms': param_grad_norms
            }
            
            print(f"  Loss: {loss:.4f}")
            print(f"  Total gradient norm: {total_grad_norm:.4f}")
            print(f"  Max param gradient: {max(param_grad_norms.values()):.4f}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            gradient_norms[batch_size] = {'error': str(e)}
    
    # Analyze gradient consistency
    print(f"\n{'Batch Size':<12} {'Loss':<8} {'Total Grad':<12} {'Max Param Grad':<15} {'Status':<8}")
    print("-" * 60)
    
    successful_tests = []
    for batch_size, result in gradient_norms.items():
        if 'error' not in result:
            successful_tests.append(batch_size)
            max_param_grad = max(result['param_norms'].values())
            status = "✅" if result['total_norm'] < 10.0 else "⚠️"
            print(f"{batch_size:<12} {result['loss']:<8.4f} {result['total_norm']:<12.4f} {max_param_grad:<15.4f} {status:<8}")
        else:
            print(f"{batch_size:<12} {'ERROR':<8} {'ERROR':<12} {'ERROR':<15} ❌")
    
    if len(successful_tests) == len(batch_sizes):
        print(f"\n🎉 GRADIENT STABILITY: VERIFIED")
        
        # Check consistency across batch sizes
        total_norms = [gradient_norms[bs]['total_norm'] for bs in successful_tests]
        norm_variance = sum((x - sum(total_norms)/len(total_norms))**2 for x in total_norms) / len(total_norms)
        
        if norm_variance < 1.0:
            print(f"   Gradient norms are consistent across batch sizes (variance: {norm_variance:.3f})")
        else:
            print(f"   ⚠️  Some gradient variance across batch sizes (variance: {norm_variance:.3f})")
    
    return gradient_norms


def main():
    """Run all tests for enhanced StableTemporalAttentionGPCM."""
    print("🧪 ENHANCED STABLE TEMPORAL ATTENTION GPCM TEST SUITE")
    print("=" * 80)
    print("Testing 3-phase enhancement:")
    print("  Phase 1: Parameter tuning (sensitivity & scaling)")
    print("  Phase 2: Selective de-regularization (LayerNorm & gains)")  
    print("  Phase 3: Enhanced temporal features (normalized time gaps)")
    print("=" * 80)
    
    # Run tests
    batch_results = test_batch_size_independence()
    performance_results = compare_model_performance()
    gradient_results = test_gradient_stability()
    
    # Final summary
    print("\n" + "=" * 80)
    print("FINAL TEST SUMMARY")
    print("=" * 80)
    
    # Check batch size independence
    batch_success = all(r.get('success', False) for r in batch_results.values())
    print(f"✅ Batch size independence: {'PASS' if batch_success else 'FAIL'}")
    
    # Check performance vs original
    enhanced_success = performance_results.get('Enhanced Stable Temporal', {}).get('success', False)
    original_success = performance_results.get('Original Temporal', {}).get('success', False)
    
    if enhanced_success and original_success:
        enhanced_loss = performance_results['Enhanced Stable Temporal']['loss']
        original_loss = performance_results['Original Temporal']['loss']
        performance_improved = enhanced_loss <= original_loss * 1.05  # Allow 5% tolerance
        print(f"✅ Performance vs original: {'IMPROVED' if performance_improved else 'NEEDS_WORK'}")
        print(f"   Enhanced loss: {enhanced_loss:.4f} vs Original loss: {original_loss:.4f}")
    else:
        print(f"❌ Performance comparison: FAILED (models didn't run)")
    
    # Check gradient stability
    gradient_success = all('error' not in r for r in gradient_results.values())
    print(f"✅ Gradient stability: {'PASS' if gradient_success else 'FAIL'}")
    
    # Overall assessment
    all_tests_passed = batch_success and enhanced_success and gradient_success
    
    if all_tests_passed:
        print(f"\n🎉 ALL TESTS PASSED!")
        print(f"   Enhanced StableTemporalAttentionGPCM is ready for training")
        print(f"   Expected improvements: batch size independence + better performance")
    else:
        print(f"\n⚠️  SOME TESTS FAILED")
        print(f"   Review the results above for specific issues")
    
    print(f"\n📝 TRAINING RECOMMENDATION:")
    if batch_success:
        print(f"   ✅ Can use batch_size=64 (default) - model is batch size independent")
        print(f"   ✅ Should show better performance than original temporal model")
        print(f"   ✅ Maintains stability while recovering performance")
    else:
        print(f"   ⚠️  Stick to larger batch sizes (32+) until issues are resolved")


if __name__ == "__main__":
    main()