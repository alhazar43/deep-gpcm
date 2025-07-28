"""
Unified Model Factory for Deep-GPCM
Creates baseline and AKVMN models from configuration.
"""

import torch
from typing import Union

from config import BaseConfig, AKVMNConfig, DeepIntegrationConfig
from models.baseline import create_baseline_gpcm
from models.akvmn import create_akvmn_gpcm
from models.deep_integration_simplified import create_simplified_deep_integration_gpcm


def create_model(config: BaseConfig, n_questions: int, device: torch.device) -> torch.nn.Module:
    """
    Factory function to create models based on configuration.
    
    Args:
        config: Model configuration
        n_questions: Number of questions (detected from data)
        device: Target device for model
        
    Returns:
        Configured model instance
    """
    # Common parameters
    common_params = {
        'n_questions': n_questions,
        'n_cats': config.n_cats,
        'memory_size': config.memory_size,
        'key_dim': config.key_dim,
        'value_dim': config.value_dim,
        'final_fc_dim': config.final_fc_dim,
        'embedding_strategy': config.embedding_strategy,
        'prediction_method': config.prediction_method
    }
    
    # Create model based on type
    if config.model_type == "akvmn":
        # AKVMN has same base parameters but doesn't use embed_dim, n_cycles, adaptive_cycles in constructor
        # The historical Deep Integration model worked with just the common parameters
        model = create_akvmn_gpcm(**common_params)
    elif config.model_type == "deep_integration":
        # Deep Integration needs additional parameters
        deep_params = {
            **common_params,
            'embed_dim': config.embed_dim,
            'n_cycles': config.n_cycles
        }
        model = create_simplified_deep_integration_gpcm(**deep_params)
    else:  # baseline
        model = create_baseline_gpcm(**common_params)
    
    return model.to(device)


def count_parameters(model: torch.nn.Module) -> dict:
    """
    Count model parameters for analysis.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with parameter counts
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    return {
        'total_parameters': total_params,
        'trainable_parameters': trainable_params,
        'non_trainable_parameters': total_params - trainable_params
    }


def get_model_info(model: torch.nn.Module, config: BaseConfig) -> dict:
    """
    Get comprehensive model information.
    
    Args:
        model: PyTorch model
        config: Model configuration
        
    Returns:
        Dictionary with model information
    """
    param_info = count_parameters(model)
    
    # Get model-specific info if available
    if hasattr(model, 'get_model_info'):
        model_specific_info = model.get_model_info()
    else:
        model_specific_info = {
            "name": config.model_type,
            "type": config.model_type,
            "architecture": "DKVMN-GPCM"
        }
    
    model_info = {
        **model_specific_info,
        'embedding_strategy': config.embedding_strategy,
        'prediction_method': config.prediction_method,
        'parameters': param_info,
        'config': {
            'n_cats': config.n_cats,
            'memory_size': config.memory_size,
            'key_dim': config.key_dim,
            'value_dim': config.value_dim,
            'final_fc_dim': config.final_fc_dim
        }
    }
    
    return model_info


def print_model_summary(model: torch.nn.Module, config: BaseConfig):
    """
    Print a formatted model summary.
    
    Args:
        model: PyTorch model
        config: Model configuration
    """
    info = get_model_info(model, config)
    
    print("=== Model Summary ===")
    print(f"Name: {info.get('name', 'Unknown')}")
    print(f"Type: {info['type'].upper()}")
    print(f"Architecture: {info.get('architecture', 'DKVMN-GPCM')}")
    print(f"Parameters: {info['parameters']['total_parameters']:,}")
    print(f"Embedding: {info['embedding_strategy']}")
    
    if 'features' in info:
        print(f"Features: {', '.join(info['features'])}")
    
    if config.model_type == "akvmn":
        print(f"AKVMN: {config.embed_dim}d, {config.n_cycles} cycles")
    elif config.model_type == "deep_integration":
        print(f"Deep Integration: {config.embed_dim}d, {config.n_cycles} cycles")
    
    print("=" * 21)