from typing import Union, Optional, Dict, Any
import torch.nn as nn

from .implementations.deep_gpcm import DeepGPCM
from .implementations.attention_gpcm import AttentionGPCM, EnhancedAttentionGPCM
# Ordinal Attention GPCM - complete standalone implementation with temperature suppression
from .implementations.ordinal_attention_gpcm import OrdinalAttentionGPCM
# Legacy adaptive imports removed

# Enhanced model registry with configuration management
MODEL_REGISTRY = {
    'deep_gpcm': {
        'class': DeepGPCM,
        'color': '#ff7f0e',  # Orange
        'display_name': 'Deep-GPCM',
        'description': 'Deep learning GPCM with DKVMN memory',
        'default_params': {
            'memory_size': 50,        # Optimized from adaptive hyperopt
            'key_dim': 64,           # Optimized from adaptive hyperopt
            'value_dim': 128,         # Optimized from adaptive hyperopt
            'final_fc_dim': 50,       # Optimized from adaptive hyperopt
            'embedding_strategy': 'linear_decay',
            'dropout_rate': 0.05       # Optimized from adaptive hyperopt
        },
        'hyperparameter_grid': {
            'memory_size': [20, 50, 100],
            'final_fc_dim': [50, 100],
            'dropout_rate': [0.0, 0.1]
        },
        # 'loss_config': {
        #     'type': 'focal',
        #     'focal_gamma': 2.0,
        #     'focal_alpha': 1.0
        # }
        'loss_config': {
            'type': 'combined',
            'ce_weight': 0.0,        # Optimized from adaptive hyperopt
            'focal_weight': 0.2,     # Reduced to make room for weighted ordinal
            'qwk_weight': 0.2,       # Optimized from adaptive hyperopt
            'weighted_ordinal_weight': 0.6,  # NEW: Simple weighted ordinal for class balance
            'ordinal_penalty': 0.5   # Mild penalty for ordinal distance mistakes
        },
        'training_config': {
            'lr': 0.001,              # Standard default
            'batch_size': 64          # Optimized batch size only
        }
    },
    'attn_gpcm_learn': {
        'class': EnhancedAttentionGPCM,
        'color': "#1100FF",  # Blue
        'display_name': 'Attention-GPCM-Learned',
        'description': 'Attention-enhanced GPCM with multi-head attention and Learned embeddings',
        'default_params': {
            'memory_size': 50,        # Optimized from adaptive hyperopt
            'key_dim': 64,           # Optimized from adaptive hyperopt
            'value_dim': 128,         # Optimized from adaptive hyperopt
            'final_fc_dim': 50,       # Optimized from adaptive hyperopt
            'embedding_strategy': 'linear_decay',
            'dropout_rate': 0.05,      # Optimized from adaptive hyperopt
            'embed_dim': 64,          # Optimized from adaptive hyperopt
            'n_heads': 8,             # Optimized from adaptive hyperopt
            'n_cycles': 3,            # Optimized from adaptive hyperopt
            'ability_scale': 1.0
        },
        'hyperparameter_grid': {
            'memory_size': [20, 50, 100],
            'final_fc_dim': [50, 100],
            'dropout_rate': [0.0, 0.1, 0.2],
            'embed_dim': [32, 64, 128],
            'n_heads': [2, 4, 8],
            'n_cycles': [1, 2, 3]
        },
        'loss_config': {
            'type': 'combined',
            'ce_weight': 0.0,        # Reduced to make room for weighted ordinal
            'focal_weight': 0.2,     # Optimized from adaptive hyperopt
            'qwk_weight': 0.2,       # Optimized from adaptive hyperopt
            'weighted_ordinal_weight': 0.6,  # NEW: Simple weighted ordinal for class balance
            'ordinal_penalty': 0.5   # Mild penalty for ordinal distance mistakes
        },
        'training_config': {
            'lr': 0.001,              # Standard default
            'batch_size': 64,         # Optimized batch size only
            'grad_clip': 1.5          # Moderate gradient clipping for research-based approach
        }
    },
    # 'attn_gpcm_linear': {
    #     'class': EnhancedAttentionGPCM,
    #     'color': '#2ca02c',  # Green
    #     'display_name': 'Attention-GPCM-Learned',
    #     'description': 'Attention-enhanced GPCM with multi-head attention and Learned embeddings',
    #     'default_params': {
    #         'memory_size': 50,        # Optimized from adaptive hyperopt
    #         'key_dim': 64,           # Optimized from adaptive hyperopt
    #         'value_dim': 128,         # Optimized from adaptive hyperopt
    #         'final_fc_dim': 50,       # Optimized from adaptive hyperopt
    #         'embedding_strategy': 'linear_decay',
    #         'dropout_rate': 0.05,      # Optimized from adaptive hyperopt
    #         'embed_dim': 64,          # Optimized from adaptive hyperopt
    #         'n_heads': 8,             # Optimized from adaptive hyperopt
    #         'n_cycles': 3,            # Optimized from adaptive hyperopt
    #         'ability_scale': 1.0,
    #         'use_learnable_embedding': False,
    #     },
    #     'hyperparameter_grid': {
    #         'memory_size': [20, 50, 100],
    #         'final_fc_dim': [50, 100],
    #         'dropout_rate': [0.0, 0.1, 0.2],
    #         'embed_dim': [32, 64, 128],
    #         'n_heads': [2, 4, 8],
    #         'n_cycles': [1, 2, 3]
    #     },
    #     'loss_config': {
    #         'type': 'combined',
    #         'ce_weight': 0.0,        # Reduced to make room for weighted ordinal
    #         'focal_weight': 0.6,     # Optimized from adaptive hyperopt
    #         'qwk_weight': 0.2,       # Optimized from adaptive hyperopt
    #         'weighted_ordinal_weight': 0.2,  # NEW: Simple weighted ordinal for class balance
    #         'ordinal_penalty': 0.5   # Mild penalty for ordinal distance mistakes
    #     },
    #     'training_config': {
    #         'lr': 0.001,              # Standard default
    #         'batch_size': 64          # Optimized batch size only
    #     }
    #     # 'loss_config': {
    #     #     'type': 'focal',
    #     #     'focal_gamma': 2.0,
    #     #     'focal_alpha': 1.0
    #     # }
    # },

    'attn_gpcm_linear': {
        'class': OrdinalAttentionGPCM,
        'color': '#d62728',  # red
        'display_name': 'Ordinal-Attention-GPCM',
        'description': 'Enhanced ordinal embedding with adaptive weight suppression to reduce adjacent category interference',
        'default_params': {
            # IDENTICAL base parameters to attn_gpcm_linear
            'memory_size': 50,
            'key_dim': 64,
            'value_dim': 128,
            'final_fc_dim': 50,
            'dropout_rate': 0.05,
            'embed_dim': 64,
            'n_heads': 8,
            'n_cycles': 3,
            'ability_scale': 1.0,
            'use_learnable_embedding': False,
            # NEW: Adaptive suppression parameters
            'suppression_mode': 'temperature',  # 'temperature', 'confidence', 'attention', 'none'
            'temperature_init': 1.0  # Initial temperature for weight sharpening
        },
        'hyperparameter_grid': {
            # IDENTICAL base hyperparameter grid to attn_gpcm_linear
            'memory_size': [20, 50, 100],
            'final_fc_dim': [50, 100],
            'dropout_rate': [0.0, 0.1, 0.2],
            'embed_dim': [32, 64, 128],
            'n_heads': [2, 4, 8],
            'n_cycles': [1, 2, 3],
            # NEW: Suppression parameters for hyperopt
            'suppression_mode': ['temperature', 'none'],  # Start conservative
            'temperature_init': [1.5, 2.0, 3.0, 4.0]  # Temperature range
        },
        'loss_config': {
            # IDENTICAL loss config to attn_gpcm_linear
            'type': 'combined',
            'ce_weight': 0.0,
            'focal_weight': 0.2,
            'qwk_weight': 0.2,
            'weighted_ordinal_weight': 0.6,
            'ordinal_penalty': 0.5
        },
        'training_config': {
            # IDENTICAL training config to attn_gpcm_linear
            'lr': 0.001,
            'batch_size': 64,
            'grad_clip': 5.0  # Adaptive gradient clipping will increase automatically if needed
        }
    },
    # 'coral_prob': {
    #     'class': CORALGPCM,
    #     'color': '#e377c2',  # Pink
    #     'display_name': 'CORAL_GPCM-PROB',
    #     'description': 'CORAL-enhanced GPCM with ordinal regression',
    #     'default_params': {
    #         'memory_size': 50,
    #         'key_dim': 50,
    #         'value_dim': 200,
    #         'final_fc_dim': 50,
    #         'embedding_strategy': 'linear_decay',
    #         'dropout_rate': 0.0,
    #         'ability_scale': 1.0,
    #         'use_discrimination': True,
    #         'coral_dropout': 0.1,
    #         'use_adaptive_blending': True,
    #         'blend_weight': 0.5
    #     },
    #     'hyperparameter_grid': {
    #         'memory_size': [20, 50, 100],
    #         'final_fc_dim': [50, 100],
    #         'coral_dropout': [0.0, 0.1, 0.2],
    #         'blend_weight': [0.3, 0.5, 0.7]
    #     },
    #     'loss_config': {
    #         'type': 'combined',
    #         'focal_weight': 0.4,
    #         'qwk_weight': 0.2,
    #         'coral_weight': 0.4,
    #         'ce_weight': 0.0
    #     }
    # },
    # 'coral_thresh': {
    #     'class': CORALGPCMFixed,
    #     'color': '#ff0000',  # Red
    #     'display_name': 'CORAL-GPCM-TAU',
    #     'description': 'CORAL-GPCM with combined beta and tau parameters',
    #     'default_params': {
    #         'memory_size': 50,
    #         'key_dim': 50,
    #         'value_dim': 200,
    #         'final_fc_dim': 50,
    #         'embedding_strategy': 'linear_decay',
    #         'dropout_rate': 0.0,
    #         'ability_scale': 1.0,
    #         'use_discrimination': True,
    #         'coral_dropout': 0.1
    #     },
    #     'hyperparameter_grid': {
    #         'memory_size': [20, 50, 100],
    #         'final_fc_dim': [50, 100],
    #         'coral_dropout': [0.0, 0.1, 0.2]
    #     },
    #     'loss_config': {
    #         'type': 'combined',
    #         'focal_weight': 0.4,
    #         'qwk_weight': 0.2,
    #         'coral_weight': 0.4,
    #         'ce_weight': 0.0
    #     }
    # },
    # 'temporal_attn_gpcm': {
    #     'class': TemporalAttentionGPCM,
    #     'color': '#9467bd',  # Purple
    #     'display_name': 'Temporal-Attention-GPCM',
    #     'description': 'Advanced attention GPCM with positional encoding and temporal features',
    #     'default_params': {
    #         'memory_size': 50,
    #         'key_dim': 50,
    #         'value_dim': 200,
    #         'final_fc_dim': 50,
    #         'embedding_strategy': 'linear_decay',
    #         'dropout_rate': 0.1,
    #         'embed_dim': 64,
    #         'n_heads': 4,
    #         'n_cycles': 2,
    #         'ability_scale': 1.0,
    #         'max_seq_len': 1000,
    #         'temporal_window': 3,
    #         'temporal_dim': 8
    #     },
    #     'hyperparameter_grid': {
    #         'memory_size': [20, 50, 100],
    #         'final_fc_dim': [50, 100],
    #         'dropout_rate': [0.0, 0.1, 0.2],
    #         'embed_dim': [32, 64, 128],
    #         'n_heads': [2, 4, 8],
    #         'n_cycles': [1, 2, 3],
    #         'temporal_window': [2, 3, 5],
    #         'temporal_dim': [4, 8, 16]
    #     },
    #     'loss_config': {
    #         'type': 'combined',
    #         'ce_weight': 0.6,
    #         'qwk_weight': 0.2,
    #         'focal_weight': 0.2
    #     }
    # },
    # 'fixed_temporal_attn_gpcm': {
    #     'class': FixedTemporalAttentionGPCM,
    #     'color': '#8c564b',  # Brown
    #     'display_name': 'Fixed-Temporal-Attention-GPCM',
    #     'description': 'Stabilized temporal attention GPCM with adaptive positional encoding and improved gradient flow',
    #     'default_params': {
    #         'memory_size': 50,
    #         'key_dim': 50,
    #         'value_dim': 200,
    #         'final_fc_dim': 50,
    #         'embedding_strategy': 'linear_decay',
    #         'dropout_rate': 0.1,
    #         'embed_dim': 64,
    #         'n_heads': 4,
    #         'n_cycles': 2,
    #         'ability_scale': 1.0,
    #         'max_seq_len': 1000,
    #         'temporal_window': 3,
    #         'temporal_dim': 8
    #     },
    #     'hyperparameter_grid': {
    #         'memory_size': [20, 50, 100],
    #         'final_fc_dim': [50, 100],
    #         'dropout_rate': [0.0, 0.1, 0.2],
    #         'embed_dim': [32, 64, 128],
    #         'n_heads': [2, 4, 8],
    #         'n_cycles': [1, 2, 3],
    #         'temporal_window': [2, 3, 5],
    #         'temporal_dim': [4, 8, 16]
    #     },
    #     'loss_config': {
    #         'type': 'combined',
    #         'ce_weight': 0.6,
    #         'qwk_weight': 0.2,
    #         'focal_weight': 0.2
    #     }
    # },
    # 'stable_temporal_attn_gpcm': {
    #     'class': StableTemporalAttentionGPCM,
    #     'color': '#17becf',  # Cyan
    #     'display_name': 'Stable-Temporal-Attention-GPCM',
    #     'description': 'Production-ready temporal GPCM with relative attention and no positional encoding conflicts',
    #     'default_params': {
    #         'memory_size': 20,
    #         'key_dim': 50,
    #         'value_dim': 200,
    #         'final_fc_dim': 100,
    #         'embedding_strategy': 'linear_decay',
    #         'dropout_rate': 0.05,
    #         'embed_dim': 64,
    #         'n_heads': 4,
    #         'n_cycles': 2,
    #         'ability_scale': 1.0,
    #         'temporal_window': 5
    #     },
    #     'hyperparameter_grid': {
    #         'memory_size': [20, 50, 100],
    #         'final_fc_dim': [50, 100],
    #         'dropout_rate': [0.0, 0.1, 0.2],
    #         'embed_dim': [32, 64, 128],
    #         'n_heads': [2, 4, 8],
    #         'n_cycles': [1, 2, 3],
    #         'temporal_window': [3, 5, 7]
    #     },
    #     'loss_config': {
    #         'type': 'combined',
    #         'ce_weight': 0.2,
    #         'qwk_weight': 0.2,
    #         'focal_weight': 0.6
    #     }
    # },
    # 'stable_temporal_attn_gpcm': {
    #     'class': StableTemporalAttentionGPCM,
    #     'color': '#17becf',  # Cyan - fresh stable model
    #     'display_name': 'Stable-Temporal-Attention-GPCM',
    #     'description': 'Production-ready temporal GPCM with no positional encoding conflicts and batch size independence',
    #     'default_params': {
    #         'memory_size': 50,
    #         'key_dim': 50,
    #         'value_dim': 200,
    #         'final_fc_dim': 50,
    #         'embedding_strategy': 'linear_decay',
    #         'dropout_rate': 0.1,
    #         'embed_dim': 64,
    #         'n_heads': 4,
    #         'n_cycles': 2,
    #         'ability_scale': 1.0,
    #         'max_seq_len': 1000,
    #         'temporal_window': 5,  # Larger window for relative attention
    #         'temporal_dim': 8
    #     },
    #     'hyperparameter_grid': {
    #         'memory_size': [20, 50, 100],
    #         'final_fc_dim': [50, 100],
    #         'dropout_rate': [0.0, 0.1, 0.2],
    #         'embed_dim': [32, 64, 128],
    #         'n_heads': [2, 4, 8],
    #         'n_cycles': [1, 2, 3],
    #         'temporal_window': [3, 5, 7],  # Relative attention window
    #         'temporal_dim': [4, 8, 16]
    #     },
    #     'loss_config': {
    #         'type': 'combined',
    #         'ce_weight': 0.4,
    #         'qwk_weight': 0.2,
    #         'focal_weight': 0.4
    #     }
    # }
    # Legacy model name aliases for backward compatibility
    # 'coral_gpcm_proper': {
    #     'class': CORALGPCM,
    #     'color': '#e377c2',  # Pink
    #     'display_name': 'CORAL_GPCM-PROPER',
    #     'description': 'CORAL-enhanced GPCM with ordinal regression (legacy name)',
    #     'default_params': {
    #         'memory_size': 50,
    #         'key_dim': 50,
    #         'value_dim': 200,
    #         'final_fc_dim': 50,
    #         'embedding_strategy': 'linear_decay',
    #         'dropout_rate': 0.0,
    #         'ability_scale': 1.0,
    #         'use_discrimination': True,
    #         'coral_dropout': 0.1,
    #         'use_adaptive_blending': True,
    #         'blend_weight': 0.5
    #     },
    #     'hyperparameter_grid': {
    #         'memory_size': [20, 50, 100],
    #         'final_fc_dim': [50, 100],
    #         'coral_dropout': [0.0, 0.1, 0.2],
    #         'blend_weight': [0.3, 0.5, 0.7]
    #     },
    #     'loss_config': {
    #         'type': 'ce'  # Original used CE loss
    #     }
    # },
    # 'coral_gpcm_fixed': {
    #     'class': CORALGPCMFixed,
    #     'color': '#ff0000',  # Red
    #     'display_name': 'CORAL-GPCM-FIXED',
    #     'description': 'CORAL-GPCM with combined beta and tau parameters (legacy name)',
    #     'default_params': {
    #         'memory_size': 50,
    #         'key_dim': 50,
    #         'value_dim': 200,
    #         'final_fc_dim': 50,
    #         'embedding_strategy': 'linear_decay',
    #         'dropout_rate': 0.0,
    #         'ability_scale': 1.0,
    #         'use_discrimination': True,
    #         'coral_dropout': 0.1
    #     },
    #     'hyperparameter_grid': {
    #         'memory_size': [20, 50, 100],
    #         'final_fc_dim': [50, 100],
    #         'coral_dropout': [0.0, 0.1, 0.2]
    #     },
    #     'loss_config': {
    #         'type': 'ce'  # Original used CE loss
    #     }
    # }
}

def create_model(model_type, n_questions, n_cats, **kwargs):
    """Create model based on type using registry configuration.
    
    Args:
        model_type: Model type name from MODEL_REGISTRY
        n_questions: Number of questions
        n_cats: Number of response categories
        **kwargs: Parameter overrides for model-specific parameters
        
    Returns:
        Model instance with metadata attributes
    """
    if model_type not in MODEL_REGISTRY:
        available = ', '.join(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model type: {model_type}. Available: {available}")
    
    model_info = MODEL_REGISTRY[model_type]
    model_class = model_info['class']
    
    # Get default parameters from registry and merge with overrides
    default_params = get_model_default_params(model_type)
    
    # Base parameters that all models need
    model_params = {
        'n_questions': n_questions,
        'n_cats': n_cats,
    }
    
    # Add default parameters from registry
    model_params.update(default_params)
    
    # Override with any explicitly provided kwargs
    model_params.update(kwargs)
    
    # Create model instance using the unified parameter set
    model = model_class(**model_params)
    
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


# Enhanced factory functions for dynamic configuration management
def get_model_config(model_type: str) -> Dict[str, Any]:
    """Get full model configuration including default parameters.
    
    Args:
        model_type: Model type name
        
    Returns:
        Dictionary with complete model configuration
    """
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Model type '{model_type}' not found in registry")
    
    config = MODEL_REGISTRY[model_type].copy()
    
    # Handle parameter variants with parent relationships (for future use)
    if 'parent' in config:
        parent_config = MODEL_REGISTRY[config['parent']]
        # Merge parent default_params with variant params
        merged_params = parent_config.get('default_params', {}).copy()
        merged_params.update(config.get('default_params', {}))
        config['default_params'] = merged_params
        
        # Inherit loss_config from parent if not specified
        if 'loss_config' not in config and 'loss_config' in parent_config:
            config['loss_config'] = parent_config['loss_config']
    
    return config


def get_model_default_params(model_type: str) -> Dict[str, Any]:
    """Get default parameters for a model type.
    
    Args:
        model_type: Model type name
        
    Returns:
        Dictionary with default parameters
    """
    config = get_model_config(model_type)
    return config.get('default_params', {})


def get_model_loss_config(model_type: str) -> Dict[str, Any]:
    """Get loss configuration for a model type.
    
    Args:
        model_type: Model type name
        
    Returns:
        Dictionary with loss configuration
    """
    config = get_model_config(model_type)
    return config.get('loss_config', {})


def get_model_training_config(model_type: str) -> Dict[str, Any]:
    """Get training configuration for a model type.
    
    Args:
        model_type: Model type name
        
    Returns:
        Dictionary with training configuration (lr, weight_decay, batch_size, grad_clip, label_smoothing)
    """
    config = get_model_config(model_type)
    return config.get('training_config', {})


def get_model_hyperparameter_grid(model_type: str) -> Dict[str, list]:
    """Get hyperparameter grid for a model type.
    
    Args:
        model_type: Model type name
        
    Returns:
        Dictionary with hyperparameter search space
    """
    config = get_model_config(model_type)
    return config.get('hyperparameter_grid', {})


def get_model_variants(base_model: str) -> list:
    """Get parameter variants of a base model.
    
    Args:
        base_model: Base model type name
        
    Returns:
        List of variant model names including the base model
    """
    variants = [base_model] if base_model in MODEL_REGISTRY else []
    
    # Find variants that have this as parent
    for model_name, config in MODEL_REGISTRY.items():
        if config.get('parent') == base_model:
            variants.append(model_name)
    
    return variants


def create_model_with_config(model_type: str, n_questions: int, n_cats: int, 
                           config_overrides: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
    """Create model with configuration-based parameters.
    
    Args:
        model_type: Model type name from MODEL_REGISTRY
        n_questions: Number of questions
        n_cats: Number of response categories
        config_overrides: Dictionary of parameter overrides
        **kwargs: Additional parameter overrides
        
    Returns:
        Model instance with metadata attributes
    """
    # Get default parameters from registry
    default_params = get_model_default_params(model_type)
    
    # Apply config overrides
    if config_overrides:
        default_params.update(config_overrides)
    
    # Apply kwargs overrides (highest priority)
    default_params.update(kwargs)
    
    # Create model using enhanced parameters
    return create_model(model_type, n_questions, n_cats, **default_params)


def validate_model_type(model_type: str) -> bool:
    """Validate that a model type exists in the registry.
    
    Args:
        model_type: Model type name to validate
        
    Returns:
        True if model exists, False otherwise
    """
    return model_type in MODEL_REGISTRY


def get_models_by_capability(capability: str) -> list:
    """Get models that have a specific capability.
    
    Args:
        capability: Capability to search for (e.g., 'attention', 'coral')
        
    Returns:
        List of model names that have the specified capability
    """
    matching_models = []
    capability_lower = capability.lower()
    
    for model_name, config in MODEL_REGISTRY.items():
        # Check model name
        if capability_lower in model_name.lower():
            matching_models.append(model_name)
        # Check description
        elif capability_lower in config.get('description', '').lower():
            matching_models.append(model_name)
    
    return matching_models


def get_model_type_from_path(model_path) -> str:
    """
    Extract model type from model path.
    
    Args:
        model_path: Path to model file
        
    Returns:
        Inferred model type
    """
    from pathlib import Path
    
    path = Path(model_path)
    filename = path.stem  # Get filename without extension
    
    # Remove common prefixes
    if filename.startswith('best_'):
        filename = filename[5:]
    
    # Check if it matches any registered model type
    for model_type in MODEL_REGISTRY.keys():
        if model_type in filename:
            return model_type
    
    # Default fallback
    return 'deep_gpcm'