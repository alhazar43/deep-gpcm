#!/usr/bin/env python3
"""
Test parameter counts for baseline and AKVMN models.
"""

import sys
import importlib.util

def load_torch_like_module():
    """Create a torch-like module for parameter counting."""
    class Parameter:
        def __init__(self, tensor_shape):
            if isinstance(tensor_shape, tuple):
                self.numel_value = 1
                for dim in tensor_shape:
                    self.numel_value *= dim
            else:
                self.numel_value = 1
        
        def numel(self):
            return self.numel_value
    
    class TensorLike:
        def __init__(self, *shape):
            self.shape = shape
            self.numel_value = 1
            for dim in shape:
                self.numel_value *= dim
        
        def numel(self):
            return self.numel_value
        
        def unsqueeze(self, *args):
            return self
        
        def expand(self, *args):
            return self
        
        def clone(self):
            return self
        
        def squeeze(self, *args):
            return self
    
    class Module:
        def __init__(self):
            self._modules = {}
            self._parameters = {}
        
        def parameters(self):
            params = []
            for param in self._parameters.values():
                params.append(param)
            for module in self._modules.values():
                if hasattr(module, 'parameters'):
                    params.extend(module.parameters())
            return params
        
        def add_parameter(self, name, param):
            self._parameters[name] = param
        
        def add_module(self, name, module):
            self._modules[name] = module
    
    class Linear(Module):
        def __init__(self, in_features, out_features, bias=True):
            super().__init__()
            self.weight = Parameter((out_features, in_features))
            self.add_parameter('weight', self.weight)
            if bias:
                self.bias = Parameter((out_features,))
                self.add_parameter('bias', self.bias)
    
    class Embedding(Module):
        def __init__(self, num_embeddings, embedding_dim, padding_idx=None):
            super().__init__()
            self.weight = Parameter((num_embeddings, embedding_dim))
            self.add_parameter('weight', self.weight)
    
    class Sequential(Module):
        def __init__(self, *modules):
            super().__init__()
            for i, module in enumerate(modules):
                self.add_module(str(i), module)
    
    class Dropout(Module):
        def __init__(self, p=0.5):
            super().__init__()
    
    class ReLU(Module):
        def __init__(self):
            super().__init__()
    
    class Tanh(Module):
        def __init__(self):
            super().__init__()
    
    class Softplus(Module):
        def __init__(self):
            super().__init__()
    
    # Create mock torch module
    mock_torch = type('torch', (), {
        'nn': type('nn', (), {
            'Module': Module,
            'Linear': Linear,
            'Embedding': Embedding,
            'Sequential': Sequential,
            'Dropout': Dropout,
            'ReLU': ReLU,
            'Tanh': Tanh,
            'Softplus': Softplus,
            'Parameter': Parameter,
        }),
        'randn': lambda *shape: TensorLike(*shape),
        'zeros': lambda *shape: TensorLike(*shape),
        'arange': lambda n, device=None: TensorLike(n),
        'clamp': lambda x, min=None, max=None: x,
        'cat': lambda tensors, dim=-1: TensorLike(10),  # Mock concatenation
        'matmul': lambda a, b: TensorLike(10, 10),  # Mock matrix multiplication
        'bmm': lambda a, b: TensorLike(10, 10, 10),  # Mock batch matrix multiplication
    })
    
    # Add functional
    mock_torch.nn.functional = type('functional', (), {
        'one_hot': lambda x, num_classes: TensorLike(10, num_classes),
        'softmax': lambda x, dim=-1: x,
        'sigmoid': lambda x: x,
        'tanh': lambda x: x,
        'cosine_similarity': lambda a, b, dim=-1: TensorLike(10),
    })
    
    # Alias F
    mock_torch.F = mock_torch.nn.functional
    
    return mock_torch

def count_baseline_parameters():
    """Count parameters in baseline model."""
    # Load torch mock
    torch = load_torch_like_module()
    
    # Add torch to sys.modules temporarily
    sys.modules['torch'] = torch
    sys.modules['torch.nn'] = torch.nn
    sys.modules['torch.nn.functional'] = torch.nn.functional
    
    try:
        # Import baseline
        from models.baseline import BaselineGPCM
        
        # Create model with default parameters (matching synthetic dataset)
        model = BaselineGPCM(n_questions=50, n_cats=4)
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        
        print(f"Baseline Model:")
        print(f"  Parameters: {param_count:,}")
        print(f"  Target: 130,655")
        print(f"  Match: {'✅' if abs(param_count - 130655) < 1000 else '❌'}")
        
        return param_count
        
    except Exception as e:
        print(f"Error testing baseline: {e}")
        import traceback
        traceback.print_exc()
        return 0
    finally:
        # Clean up sys.modules
        if 'torch' in sys.modules:
            del sys.modules['torch']
        if 'torch.nn' in sys.modules:
            del sys.modules['torch.nn']
        if 'torch.nn.functional' in sys.modules:
            del sys.modules['torch.nn.functional']

def count_akvmn_parameters():
    """Count parameters in AKVMN model."""
    # Load torch mock
    torch = load_torch_like_module()
    
    # Add torch to sys.modules temporarily
    sys.modules['torch'] = torch
    sys.modules['torch.nn'] = torch.nn
    sys.modules['torch.nn.functional'] = torch.nn.functional
    
    try:
        # Import AKVMN
        from models.akvmn import AKVMNGPCM
        
        # Create model with default parameters (matching synthetic dataset)
        model = AKVMNGPCM(n_questions=50, n_cats=4)
        
        # Count parameters
        param_count = sum(p.numel() for p in model.parameters())
        
        print(f"\nAKVMN Model:")
        print(f"  Parameters: {param_count:,}")
        print(f"  Target: 171,217")
        print(f"  Match: {'✅' if abs(param_count - 171217) < 1000 else '❌'}")
        
        # Get model info
        model_info = model.get_model_info()
        print(f"  Architecture: {model_info['architecture']}")
        print(f"  Features: {len(model_info['features'])}")
        
        return param_count
        
    except Exception as e:
        print(f"Error testing AKVMN: {e}")
        import traceback
        traceback.print_exc()
        return 0
    finally:
        # Clean up sys.modules
        if 'torch' in sys.modules:
            del sys.modules['torch']
        if 'torch.nn' in sys.modules:
            del sys.modules['torch.nn']
        if 'torch.nn.functional' in sys.modules:
            del sys.modules['torch.nn.functional']

def main():
    """Main function."""
    print("=== Parameter Count Analysis ===")
    
    baseline_params = count_baseline_parameters()
    akvmn_params = count_akvmn_parameters()
    
    print(f"\n=== Summary ===")
    print(f"Baseline: {baseline_params:,} parameters")
    print(f"AKVMN: {akvmn_params:,} parameters")
    
    if akvmn_params > 0 and baseline_params > 0:
        diff = akvmn_params - baseline_params
        percent_diff = (diff / baseline_params) * 100
        print(f"Difference: {diff:,} ({percent_diff:+.1f}%)")
        
        # Historical comparison
        target_baseline = 130655
        target_akvmn = 171217
        target_diff = target_akvmn - target_baseline
        
        print(f"\nHistorical Targets:")
        print(f"Baseline: {target_baseline:,}")
        print(f"AKVMN: {target_akvmn:,}")
        print(f"Target difference: {target_diff:,}")

if __name__ == "__main__":
    main()