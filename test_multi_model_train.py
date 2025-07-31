#!/usr/bin/env python3
"""
Test script for multi-model training functionality
"""

import subprocess
import sys

def test_single_model():
    """Test single model training (backward compatibility)"""
    print("Testing single model training...")
    cmd = [
        sys.executable, "train.py",
        "--model", "deep_gpcm",
        "--dataset", "synthetic_OC",
        "--epochs", "2",
        "--no_cv"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        print(f"Single model test: {'✅ PASSED' if result.returncode == 0 else '❌ FAILED'}")
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ Single model test: TIMEOUT")
        return False

def test_multi_model():
    """Test multi-model training"""
    print("Testing multi-model training...")
    cmd = [
        sys.executable, "train.py", 
        "--models", "deep_gpcm", "attn_gpcm",
        "--dataset", "synthetic_OC",
        "--epochs", "2", 
        "--no_cv"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        print(f"Multi-model test: {'✅ PASSED' if result.returncode == 0 else '❌ FAILED'}")
        if result.returncode != 0:
            print(f"Error: {result.stderr}")
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("❌ Multi-model test: TIMEOUT")
        return False

def test_argument_validation():
    """Test argument validation"""
    print("Testing argument validation...")
    
    # Test both --model and --models (should fail)
    cmd = [
        sys.executable, "train.py",
        "--model", "deep_gpcm",
        "--models", "attn_gpcm", 
        "--dataset", "synthetic_OC"
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        validation_works = result.returncode != 0 and "Cannot specify both" in result.stderr
        print(f"Validation test: {'✅ PASSED' if validation_works else '❌ FAILED'}")
        return validation_works
    except subprocess.TimeoutExpired:
        print("❌ Validation test: TIMEOUT") 
        return False

if __name__ == "__main__":
    print("=" * 60)
    print("MULTI-MODEL TRAINING TEST SUITE")
    print("=" * 60)
    
    tests = [
        test_argument_validation,
        test_single_model,
        test_multi_model
    ]
    
    results = []
    for test in tests:
        results.append(test())
        print()
    
    print("=" * 60)
    print(f"RESULTS: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)
    
    if all(results):
        print("🎉 All tests passed! Multi-model training is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")