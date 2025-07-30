"""
Testing Utilities for Deep-GPCM
Consolidated testing functions from various debug files.
"""

import torch
import numpy as np
import time
from typing import Dict, Any, List, Tuple

from .data_utils import create_synthetic_data, UnifiedDataLoader
from .loss_utils import create_loss_function, compare_loss_functions


def test_model_forward_pass(model: torch.nn.Module, n_questions: int, n_cats: int, 
                          batch_size: int = 4, seq_len: int = 10) -> Dict[str, Any]:
    """
    Test model forward pass with synthetic data.
    
    Args:
        model: Model to test
        n_questions: Number of questions
        n_cats: Number of categories
        batch_size: Batch size for testing
        seq_len: Sequence length for testing
        
    Returns:
        Test results dictionary
    """
    print(f"🧪 Testing {model.__class__.__name__} forward pass...")
    
    # Create test data
    device = next(model.parameters()).device
    questions = torch.randint(1, n_questions + 1, (batch_size, seq_len), device=device)
    responses = torch.randint(0, n_cats, (batch_size, seq_len), device=device)
    
    results = {
        'model_name': model.__class__.__name__,
        'input_shape': f"Questions: {questions.shape}, Responses: {responses.shape}",
        'success': False,
        'error': None,
        'output_shapes': {},
        'inference_time': None,
        'memory_usage': None
    }
    
    try:
        # Test forward pass
        model.eval()
        
        # Time inference
        start_time = time.time()
        
        with torch.no_grad():
            outputs = model(questions, responses)
        
        inference_time = time.time() - start_time
        
        # Analyze outputs
        if isinstance(outputs, tuple):
            output_shapes = {f'output_{i}': list(out.shape) for i, out in enumerate(outputs)}
            gpcm_probs = outputs[-1]  # Last output should be GPCM probabilities
        else:
            output_shapes = {'output': list(outputs.shape)}
            gpcm_probs = outputs
        
        # Validate GPCM probabilities
        if gpcm_probs.shape[-1] != n_cats:
            raise ValueError(f"Expected {n_cats} categories, got {gpcm_probs.shape[-1]}")
        
        # Check if probabilities sum to 1
        prob_sums = torch.sum(gpcm_probs, dim=-1)
        if not torch.allclose(prob_sums, torch.ones_like(prob_sums), atol=1e-5):
            print(f"⚠️  Warning: Probabilities don't sum to 1 (range: {prob_sums.min():.4f}-{prob_sums.max():.4f})")
        
        # Memory usage (approximate)
        if torch.cuda.is_available() and device.type == 'cuda':
            memory_usage = torch.cuda.memory_allocated(device) / 1024**2  # MB
        else:
            memory_usage = None
        
        results.update({
            'success': True,
            'output_shapes': output_shapes,
            'inference_time': inference_time,
            'memory_usage': memory_usage,
            'prob_shape': list(gpcm_probs.shape),
            'prob_range': [float(gpcm_probs.min()), float(gpcm_probs.max())],
            'prob_sum_range': [float(prob_sums.min()), float(prob_sums.max())]
        })
        
        print(f"✅ Forward pass successful!")
        print(f"   Inference time: {inference_time*1000:.2f}ms")
        print(f"   Output shapes: {output_shapes}")
        
    except Exception as e:
        results['error'] = str(e)
        print(f"❌ Forward pass failed: {e}")
    
    return results


def test_model_training_step(model: torch.nn.Module, n_questions: int, n_cats: int,
                           loss_type: str = 'crossentropy', batch_size: int = 4,
                           seq_len: int = 10) -> Dict[str, Any]:
    """
    Test model training step.
    
    Args:
        model: Model to test
        n_questions: Number of questions
        n_cats: Number of categories
        loss_type: Type of loss function
        batch_size: Batch size for testing
        seq_len: Sequence length for testing
        
    Returns:
        Training test results
    """
    print(f"🧪 Testing {model.__class__.__name__} training step...")
    
    device = next(model.parameters()).device
    
    # Create test data
    questions = torch.randint(1, n_questions + 1, (batch_size, seq_len), device=device)
    responses = torch.randint(0, n_cats, (batch_size, seq_len), device=device)
    
    # Create optimizer and loss function
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = create_loss_function(loss_type, n_cats)
    
    results = {
        'model_name': model.__class__.__name__,
        'success': False,
        'error': None,
        'initial_loss': None,
        'final_loss': None,
        'gradient_norms': None,
        'parameter_updates': None
    }
    
    try:
        model.train()
        
        # Get initial parameters for comparison
        initial_params = {name: param.clone() for name, param in model.named_parameters()}
        
        # Forward pass
        outputs = model(questions, responses)
        gpcm_probs = outputs[-1] if isinstance(outputs, tuple) else outputs
        
        # Compute loss
        targets = responses.view(-1)
        probs = gpcm_probs.view(-1, n_cats)
        
        initial_loss = loss_fn(probs, targets)
        
        # Backward pass
        optimizer.zero_grad()
        initial_loss.backward()
        
        # Compute gradient norms
        grad_norms = {}
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norms[name] = float(param.grad.norm())
        
        # Update parameters
        optimizer.step()
        
        # Forward pass again to check loss change
        outputs = model(questions, responses)
        gpcm_probs = outputs[-1] if isinstance(outputs, tuple) else outputs
        probs = gpcm_probs.view(-1, n_cats)
        final_loss = loss_fn(probs, targets)
        
        # Check parameter updates
        param_updates = {}
        for name, param in model.named_parameters():
            if name in initial_params:
                update_norm = float((param - initial_params[name]).norm())
                param_updates[name] = update_norm
        
        results.update({
            'success': True,
            'initial_loss': float(initial_loss),
            'final_loss': float(final_loss),
            'loss_change': float(final_loss - initial_loss),
            'gradient_norms': grad_norms,
            'parameter_updates': param_updates,
            'avg_grad_norm': np.mean(list(grad_norms.values())) if grad_norms else 0.0,
            'avg_param_update': np.mean(list(param_updates.values())) if param_updates else 0.0
        })
        
        print(f"✅ Training step successful!")
        print(f"   Initial loss: {initial_loss:.4f}")
        print(f"   Final loss: {final_loss:.4f}")
        print(f"   Loss change: {final_loss - initial_loss:+.4f}")
        print(f"   Avg gradient norm: {results['avg_grad_norm']:.6f}")
        
    except Exception as e:
        results['error'] = str(e)
        print(f"❌ Training step failed: {e}")
    
    return results


def compare_model_performance(models: Dict[str, torch.nn.Module], n_questions: int, 
                            n_cats: int, n_epochs: int = 5, batch_size: int = 16) -> Dict[str, Any]:
    """
    Compare performance of multiple models.
    
    Args:
        models: Dictionary of model_name -> model
        n_questions: Number of questions
        n_cats: Number of categories
        n_epochs: Number of training epochs
        batch_size: Batch size
        
    Returns:
        Comparison results
    """
    print(f"🏁 Comparing {len(models)} models...")
    
    # Create test dataset
    questions, responses = create_synthetic_data(
        n_sequences=100, n_questions=n_questions, n_cats=n_cats
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    results = {}
    
    for model_name, model in models.items():
        print(f"\n📊 Testing {model_name}...")
        
        model = model.to(device)
        
        # Test forward pass
        forward_results = test_model_forward_pass(model, n_questions, n_cats)
        
        # Test training
        training_results = test_model_training_step(model, n_questions, n_cats)
        
        # Quick training run
        model.train()
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        loss_fn = create_loss_function('crossentropy', n_cats)
        
        train_loader = UnifiedDataLoader(questions, responses, batch_size, device=device)
        
        losses = []
        start_time = time.time()
        
        try:
            for epoch in range(n_epochs):
                epoch_loss = 0.0
                for q_batch, r_batch, mask_batch in train_loader:
                    optimizer.zero_grad()
                    
                    outputs = model(q_batch, r_batch)
                    gpcm_probs = outputs[-1] if isinstance(outputs, tuple) else outputs
                    
                    targets = r_batch.view(-1)
                    probs = gpcm_probs.view(-1, n_cats)
                    
                    loss = loss_fn(probs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                losses.append(epoch_loss / len(train_loader))
            
            training_time = time.time() - start_time
            training_success = True
            
        except Exception as e:
            training_time = None
            training_success = False
            print(f"❌ Training failed for {model_name}: {e}")
        
        # Compile results
        results[model_name] = {
            'forward_pass': forward_results,
            'training_step': training_results,
            'quick_training': {
                'success': training_success,
                'epochs': n_epochs,
                'losses': losses if training_success else None,
                'final_loss': losses[-1] if training_success and losses else None,
                'training_time': training_time,
                'parameters': sum(p.numel() for p in model.parameters())
            }
        }
    
    return results


def run_comprehensive_tests(models: Dict[str, torch.nn.Module], n_questions: int = 50, 
                          n_cats: int = 4) -> Dict[str, Any]:
    """
    Run comprehensive test suite on models.
    
    Args:
        models: Dictionary of model_name -> model
        n_questions: Number of questions
        n_cats: Number of categories
        
    Returns:
        Comprehensive test results
    """
    print("🧪 COMPREHENSIVE MODEL TESTING")
    print("=" * 50)
    
    all_results = {}
    
    # Individual model tests
    for model_name, model in models.items():
        print(f"\n🔬 Testing {model_name}...")
        
        # Forward pass test
        forward_results = test_model_forward_pass(model, n_questions, n_cats)
        
        # Training step test
        training_results = test_model_training_step(model, n_questions, n_cats)
        
        all_results[model_name] = {
            'forward_pass': forward_results,
            'training_step': training_results
        }
    
    # Comparative tests
    if len(models) > 1:
        print(f"\n🏁 Running comparative tests...")
        comparison_results = compare_model_performance(models, n_questions, n_cats)
        all_results['comparison'] = comparison_results
    
    # Summary
    print(f"\n📊 TEST SUMMARY")
    print("-" * 30)
    
    for model_name in models.keys():
        forward_success = all_results[model_name]['forward_pass']['success']
        training_success = all_results[model_name]['training_step']['success']
        
        status = "✅" if forward_success and training_success else "❌"
        print(f"{status} {model_name}: Forward={forward_success}, Training={training_success}")
    
    return all_results


# Benchmarking utilities (consolidated from benchmark files)
def quick_benchmark_models(model_configs: Dict[str, Any], dataset_name: str = "synthetic_OC",
                          epochs: int = 5) -> Dict[str, Any]:
    """
    Quick benchmark for model comparison.
    
    Args:
        model_configs: Dictionary of model configurations
        dataset_name: Dataset to use
        epochs: Number of epochs
        
    Returns:
        Benchmark results
    """
    print(f"⚡ Quick benchmark: {len(model_configs)} models, {epochs} epochs")
    
    from ..config_unified import get_preset_configs
    from ..model_factory_unified import create_model
    from ..utils.gpcm_utils import load_gpcm_data
    
    # Use synthetic data if dataset not found
    try:
        train_path = f"data/{dataset_name}/{dataset_name.lower()}_train.txt"
        _, train_questions, train_responses, n_cats = load_gpcm_data(train_path)
        n_questions = max([max(seq) for seq in train_questions])
    except:
        print("📋 Using synthetic data for benchmark")
        train_questions, train_responses = create_synthetic_data(
            n_sequences=50, n_questions=20, n_cats=4
        )
        n_questions, n_cats = 20, 4
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    results = {}
    
    for model_name, config in model_configs.items():
        print(f"🚀 Benchmarking {model_name}...")
        
        try:
            # Create model
            model = create_model(config, n_questions, device)
            
            # Quick training
            model.train()
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
            loss_fn = create_loss_function('crossentropy', n_cats)
            
            train_loader = UnifiedDataLoader(train_questions, train_responses, 16, device=device)
            
            start_time = time.time()
            final_loss = None
            
            for epoch in range(epochs):
                epoch_loss = 0.0
                for q_batch, r_batch, _ in train_loader:
                    optimizer.zero_grad()
                    
                    outputs = model(q_batch, r_batch)
                    gpcm_probs = outputs[-1] if isinstance(outputs, tuple) else outputs
                    
                    targets = r_batch.view(-1)
                    probs = gpcm_probs.view(-1, n_cats)
                    
                    loss = loss_fn(probs, targets)
                    loss.backward()
                    optimizer.step()
                    
                    epoch_loss += loss.item()
                
                final_loss = epoch_loss / len(train_loader)
            
            training_time = time.time() - start_time
            
            results[model_name] = {
                'success': True,
                'final_loss': final_loss,
                'training_time': training_time,
                'parameters': sum(p.numel() for p in model.parameters()),
                'epochs': epochs
            }
            
            print(f"✅ {model_name}: Loss={final_loss:.4f}, Time={training_time:.1f}s, Params={results[model_name]['parameters']:,}")
            
        except Exception as e:
            results[model_name] = {
                'success': False,
                'error': str(e)
            }
            print(f"❌ {model_name} failed: {e}")
    
    return results


if __name__ == "__main__":
    print("🧪 Testing utilities module...")
    
    # Test data utilities
    from .data_utils import test_data_loader, test_data_validation
    test_data_loader()
    test_data_validation()
    
    # Test loss utilities
    from .loss_utils import test_loss_functions
    test_loss_functions()
    
    print("✅ All utility tests passed!")