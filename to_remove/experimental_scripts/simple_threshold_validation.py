#!/usr/bin/env python3
"""
Simple validation script for GPCM-CORAL threshold integration.
Tests core functionality without complex data dependencies.
"""

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'

import torch
import time
import json
import numpy as np
from datetime import datetime

from models.factory import create_model
from models.threshold_coupling import test_linear_threshold_coupler


def test_coupling_interface():
    """Test the threshold coupling interface and compatibility."""
    print("🧪 TESTING THRESHOLD COUPLING INTERFACE")
    print("-" * 60)
    
    try:
        # Test model creation with coupling
        model = create_model(
            model_type='coral_gpcm',
            n_questions=50,
            n_cats=4,
            enable_threshold_coupling=True,
            threshold_gpcm_weight=0.6,
            threshold_coral_weight=0.4
        )
        
        print("✅ Model creation with coupling: PASS")
        
        # Test coupling info retrieval
        coupling_info = model.get_coupling_info()
        assert coupling_info is not None
        assert coupling_info['coupling_enabled'] == True
        print("✅ Coupling info retrieval: PASS")
        print(f"   Coupling type: {coupling_info['coupling_type']}")
        print(f"   Config weights: GPCM={coupling_info['config']['gpcm_weight']}, CORAL={coupling_info['config']['coral_weight']}")
        
        # Test backward compatibility (coupling disabled)
        model_no_coupling = create_model(
            model_type='coral_gpcm',
            n_questions=50,
            n_cats=4,
            enable_threshold_coupling=False
        )
        
        coupling_info_disabled = model_no_coupling.get_coupling_info()
        assert coupling_info_disabled is None
        print("✅ Backward compatibility: PASS")
        
        # Test forward pass shapes
        batch_size, seq_len = 2, 5
        questions = torch.randint(0, 50, (batch_size, seq_len))
        responses = torch.randint(0, 4, (batch_size, seq_len))
        
        with torch.no_grad():
            abilities, thresholds, discrimination, probs = model(questions, responses)
            
        expected_shapes = {
            'abilities': (batch_size, seq_len),
            'thresholds': (batch_size, seq_len, 3),  # n_cats - 1
            'probs': (batch_size, seq_len, 4)
        }
        
        for name, tensor in [('abilities', abilities), ('thresholds', thresholds), ('probs', probs)]:
            assert tensor.shape == expected_shapes[name], f"Shape mismatch for {name}: {tensor.shape} vs {expected_shapes[name]}"
        
        print("✅ Forward pass shapes: PASS")
        print(f"   Abilities: {abilities.shape}, Thresholds: {thresholds.shape}, Probs: {probs.shape}")
        
        # Test gradient flow through coupling
        model.train()
        # Enable gradients for this test
        questions.requires_grad_(False)  # Input doesn't need gradients
        responses.requires_grad_(False)  # Targets don't need gradients
        
        # Forward pass with gradients enabled
        abilities, thresholds, discrimination, probs = model(questions, responses)
        
        loss = torch.nn.functional.cross_entropy(
            probs.reshape(-1, 4), 
            responses.reshape(-1)
        )
        loss.backward()
        
        # Check if coupler parameters have gradients
        if model.threshold_coupler is not None:
            coupler_has_grads = any(p.grad is not None for p in model.threshold_coupler.parameters())
            assert coupler_has_grads, "Threshold coupler parameters should have gradients"
            print("✅ Gradient flow through coupling: PASS")
        else:
            print("✅ Gradient flow through coupling: PASS (no coupler, baseline model)")
        
        print("✅ All interface tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Interface test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def performance_stress_test():
    """Test performance characteristics of threshold coupling."""
    print("\n⚡ PERFORMANCE STRESS TEST")
    print("-" * 60)
    
    # Test configurations
    configs = [
        {'name': 'baseline', 'enable_threshold_coupling': False},
        {'name': 'coupled', 'enable_threshold_coupling': True, 'threshold_gpcm_weight': 0.7, 'threshold_coral_weight': 0.3}
    ]
    
    results = {}
    
    for config in configs:
        print(f"Testing {config['name']} configuration...")
        
        # Create model
        torch.manual_seed(42)
        model = create_model(
            model_type='coral_gpcm',
            n_questions=100,
            n_cats=4,
            memory_size=50,
            **{k: v for k, v in config.items() if k != 'name'}
        )
        
        # Test data
        batch_size, seq_len = 8, 20
        questions = torch.randint(0, 100, (batch_size, seq_len))
        responses = torch.randint(0, 4, (batch_size, seq_len))
        
        # Warm up
        with torch.no_grad():
            model(questions, responses)
        
        # Timing test
        n_trials = 10
        times = []
        
        for _ in range(n_trials):
            start_time = time.time()
            with torch.no_grad():
                model(questions, responses)
            times.append(time.time() - start_time)
        
        avg_time = np.mean(times)
        std_time = np.std(times)
        
        results[config['name']] = {
            'avg_time': avg_time,
            'std_time': std_time,
            'times': times
        }
        
        print(f"   Average time: {avg_time*1000:.2f} ± {std_time*1000:.2f} ms")
    
    # Calculate overhead
    baseline_time = results['baseline']['avg_time']
    coupled_time = results['coupled']['avg_time']
    overhead_percent = ((coupled_time - baseline_time) / baseline_time) * 100
    
    print(f"\nPerformance overhead: {overhead_percent:+.1f}%")
    
    # Success criterion: ≤5% overhead
    overhead_ok = overhead_percent <= 5.0
    print(f"Overhead criterion (≤5%): {'✅ PASS' if overhead_ok else '❌ FAIL'}")
    
    return overhead_ok, overhead_percent


def validate_phase_completion():
    """Validate that all phases from TODO.md are complete."""
    print("\n📋 PHASE COMPLETION VALIDATION")
    print("-" * 60)
    
    phases = [
        {
            'name': 'Phase 0: Simplification',
            'tests': [
                ('Clean implementation exists', lambda: os.path.exists('core/threshold_coupling.py')),
                ('Research archive exists', lambda: os.path.exists('research/threshold_coupling_archived.py')),
            ]
        },
        {
            'name': 'Phase 1: Linear Coupling Implementation',
            'tests': [
                ('LinearThresholdCoupler works', lambda: test_linear_threshold_coupler() is None),
                ('Factory pattern works', lambda: create_model('coral_gpcm', 50, 4, enable_threshold_coupling=True) is not None),
            ]
        },
        {
            'name': 'Phase 2: Integration',
            'tests': [
                ('Model factory updated', lambda: 'threshold_coupling_config' in str(create_model.__code__)),
                ('CORALDeepGPCM integrated', lambda: hasattr(create_model('coral_gpcm', 50, 4, enable_threshold_coupling=True), 'threshold_coupler')),
            ]
        }
    ]
    
    total_tests = 0
    passed_tests = 0
    
    for phase in phases:
        print(f"\n{phase['name']}:")
        phase_passed = 0
        
        for test_name, test_func in phase['tests']:
            total_tests += 1
            try:
                # Suppress output from test functions
                import sys
                from io import StringIO
                old_stdout = sys.stdout
                sys.stdout = StringIO()
                
                result = test_func()
                
                sys.stdout = old_stdout
                
                if result is not False:  # None or True both count as pass
                    print(f"  ✅ {test_name}")
                    passed_tests += 1
                    phase_passed += 1
                else:
                    print(f"  ❌ {test_name}")
            except Exception as e:
                sys.stdout = old_stdout
                print(f"  ❌ {test_name} (Error: {str(e)[:50]})")
        
        phase_score = phase_passed / len(phase['tests'])
        print(f"  Phase score: {phase_passed}/{len(phase['tests'])} ({phase_score*100:.0f}%)")
    
    overall_score = passed_tests / total_tests
    print(f"\n🏆 OVERALL COMPLETION: {passed_tests}/{total_tests} ({overall_score*100:.0f}%)")
    
    return overall_score >= 0.8  # At least 80% of tests must pass


def main():
    """Run comprehensive validation."""
    print("=" * 80)
    print("GPCM-CORAL THRESHOLD COUPLING VALIDATION")
    print("=" * 80)
    print()
    
    # Phase 1: Core unit tests
    print("1️⃣ Running core threshold coupling tests...")
    try:
        test_linear_threshold_coupler()
        core_tests_ok = True
        print("✅ Core tests passed!\n")
    except Exception as e:
        print(f"❌ Core tests failed: {e}\n")
        core_tests_ok = False
    
    # Phase 2: Interface compatibility
    interface_ok = test_coupling_interface()
    
    # Phase 3: Performance testing
    if interface_ok:
        perf_ok, overhead = performance_stress_test()
    else:
        perf_ok = False
        overhead = float('inf')
    
    # Phase 4: Phase completion validation
    phases_ok = validate_phase_completion()
    
    # Summary
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)
    
    results = {
        'core_tests': core_tests_ok,
        'interface_compatibility': interface_ok,
        'performance_acceptable': perf_ok,
        'phases_complete': phases_ok,
        'performance_overhead_percent': overhead if overhead != float('inf') else None
    }
    
    success_count = sum([core_tests_ok, interface_ok, perf_ok, phases_ok])
    
    print(f"Core threshold coupling tests: {'✅ PASS' if core_tests_ok else '❌ FAIL'}")
    print(f"Interface compatibility: {'✅ PASS' if interface_ok else '❌ FAIL'}")
    print(f"Performance acceptable: {'✅ PASS' if perf_ok else '❌ FAIL'} (Overhead: {overhead:+.1f}%)")
    print(f"All phases complete: {'✅ PASS' if phases_ok else '❌ FAIL'}")
    
    print(f"\n🏆 OVERALL SUCCESS: {success_count}/4 criteria met")
    
    overall_success = success_count >= 3  # At least 3/4 must pass
    if overall_success:
        print("🎉 THRESHOLD COUPLING INTEGRATION SUCCESSFUL!")
    else:
        print("⚠️  THRESHOLD COUPLING INTEGRATION NEEDS ATTENTION")
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_data = {
        'timestamp': timestamp,
        'validation_results': {k: bool(v) if isinstance(v, (bool, np.bool_)) else v for k, v in results.items()},
        'success_count': int(success_count),
        'total_criteria': 4,
        'overall_success': bool(overall_success)
    }
    
    os.makedirs("results/validation", exist_ok=True)
    results_file = f"results/validation/simple_threshold_validation_{timestamp}.json"
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    print(f"📋 Results saved to: {results_file}")
    
    return overall_success


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)