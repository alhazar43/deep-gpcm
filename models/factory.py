from typing import Union, Optional, Dict, Any
import torch.nn as nn

from .implementations.deep_gpcm import DeepGPCM
from .implementations.attention_gpcm import AttentionGPCM, EnhancedAttentionGPCM
# Legacy CORAL imports removed - only coral_gpcm_proper remains
from .implementations.coral_gpcm_proper import CORALGPCM
# Legacy adaptive imports removed

# Model metadata registry with colors and display names
MODEL_REGISTRY = {
    'deep_gpcm': {
        'class': DeepGPCM,
        'color': '#ff7f0e',  # Orange
        'display_name': 'Deep-GPCM',
        'description': 'Deep learning GPCM with DKVMN memory'
    },
    'attn_gpcm': {
        'class': EnhancedAttentionGPCM,
        'color': '#1f77b4',  # Blue
        'display_name': 'Attention-GPCM',
        'description': 'Attention-enhanced GPCM with multi-head attention'
    },
    'coral_gpcm_proper': {
        'class': CORALGPCM,
        'color': '#e377c2',  # Pink
        'display_name': 'CORAL-GPCM',
        'description': 'CORAL-enhanced GPCM with ordinal regression'
    }
}

def create_model(model_type, n_questions, n_cats, **kwargs):
    """Create model based on type.
    
    Args:
        model_type: Model type name from MODEL_REGISTRY
        n_questions: Number of questions
        n_cats: Number of response categories
        **kwargs: Additional model-specific parameters
        
    Returns:
        Model instance with metadata attributes
    """
    if model_type not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model type: {model_type}. Available: {available}")
    
    model_info = MODEL_REGISTRY[model_type]
    model_class = model_info['class']
    
    # Common parameters
    common_params = {
        'n_questions': n_questions,
        'n_cats': n_cats,
        'memory_size': kwargs.get('memory_size', 50),
        'key_dim': kwargs.get('key_dim', 50),
        'value_dim': kwargs.get('value_dim', 200),
        'final_fc_dim': kwargs.get('final_fc_dim', 50),
        'embedding_strategy': kwargs.get('embedding_strategy', 'linear_decay'),
        'dropout_rate': kwargs.get('dropout_rate', 0.0)
    }
    
    # Model-specific creation
    if model_type == 'deep_gpcm':
        model = model_class(**common_params)
    
    elif model_type == 'attn_gpcm':
        # Use enhanced version with learnable parameters
        model = model_class(
            n_questions=n_questions,
            n_cats=n_cats,
            embed_dim=kwargs.get('embed_dim', 64),
            memory_size=common_params['memory_size'],
            key_dim=common_params['key_dim'],
            value_dim=common_params['value_dim'],
            final_fc_dim=common_params['final_fc_dim'],
            n_heads=kwargs.get('n_heads', 4),
            n_cycles=kwargs.get('n_cycles', 2),
            embedding_strategy=common_params['embedding_strategy'],
            ability_scale=kwargs.get('ability_scale', 2.0),
            dropout_rate=kwargs.get('dropout_rate', 0.1)
        )
    
    elif model_type == 'coral_gpcm_proper':
        # Proper CORAL-GPCM with correct IRT-CORAL separation
        model = model_class(
            **common_params,
            ability_scale=kwargs.get('ability_scale', 1.0),
            use_discrimination=kwargs.get('use_discrimination', True),
            coral_dropout=kwargs.get('coral_dropout', 0.1),
            use_adaptive_blending=kwargs.get('use_adaptive_blending', True),
            blend_weight=kwargs.get('blend_weight', 0.5)
        )
    
    # Attach metadata to model instance
    model.model_type = model_type
    model.model_color = model_info['color']
    model.display_name = model_info['display_name']
    model.description = model_info['description']
    
    return model


def get_model_metadata(model_type: str) -> Dict[str, Any]:
    """Get metadata for a model type.
    
    Args:
        model_type: Model type name
        
    Returns:
        Dictionary with model metadata (color, display_name, description)
    """
    if model_type not in MODEL_REGISTRY:
        return {
            'color': '#808080',  # Gray for unknown models
            'display_name': model_type.replace('_', ' ').title(),
            'description': 'Unknown model'
        }
    return {k: v for k, v in MODEL_REGISTRY[model_type].items() if k != 'class'}


def get_model_color(model_type: str) -> str:
    """Get color for a model type.
    
    Args:
        model_type: Model type name
        
    Returns:
        Hex color string
    """
    return get_model_metadata(model_type)['color']


def get_all_model_types() -> list:
    """Get list of all available model types."""
    return list(MODEL_REGISTRY.keys())