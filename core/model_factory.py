"""Model factory for creating Deep-GPCM models."""

from .model import DeepGPCM, AttentionGPCM
from .attention_enhanced import EnhancedAttentionGPCM
from .coral_gpcm import CORALDeepGPCM, HybridCORALGPCM
from .threshold_coupling import ThresholdCouplingConfig


def create_model(model_type, n_questions, n_cats, **kwargs):
    """Create model based on type.
    
    Args:
        model_type: 'deep_gpcm', 'attn_gpcm', 'coral', 'coral_gpcm'
        n_questions: Number of questions
        n_cats: Number of response categories
        **kwargs: Additional model-specific parameters
        
    Returns:
        Model instance
    """
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
    
    if model_type == 'deep_gpcm':
        model = DeepGPCM(**common_params)
    
    elif model_type == 'attn_gpcm':
        # Use enhanced version with learnable parameters
        model = EnhancedAttentionGPCM(
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
    
    elif model_type == 'coral':
        # CORAL-enhanced Deep GPCM
        # Create threshold coupling config if specified
        threshold_config = None
        if kwargs.get('enable_threshold_coupling', False):
            threshold_config = ThresholdCouplingConfig(
                enabled=True,
                coupling_type=kwargs.get('coupling_type', 'linear'),
                gpcm_weight=kwargs.get('gpcm_weight', 0.7),
                coral_weight=kwargs.get('coral_weight', 0.3)
            )
        
        model = CORALDeepGPCM(
            **common_params,
            ability_scale=kwargs.get('ability_scale', 1.0),
            use_discrimination=kwargs.get('use_discrimination', True),
            coral_hidden_dim=kwargs.get('coral_hidden_dim', None),
            use_coral_thresholds=kwargs.get('use_coral_thresholds', True),
            coral_dropout=kwargs.get('coral_dropout', 0.1),
            threshold_coupling_config=threshold_config
        )
    
    elif model_type == 'coral_gpcm':
        # For coral_gpcm, use CORALDeepGPCM with threshold coupling by default
        # Create threshold coupling config (enabled by default for coral_gpcm)
        threshold_config = ThresholdCouplingConfig(
            enabled=kwargs.get('enable_threshold_coupling', True),  # Default enabled
            coupling_type=kwargs.get('coupling_type', 'linear'),
            gpcm_weight=kwargs.get('gpcm_weight', 0.7),
            coral_weight=kwargs.get('coral_weight', 0.3)
        )
        
        model = CORALDeepGPCM(
            **common_params,
            ability_scale=kwargs.get('ability_scale', 1.0),
            use_discrimination=kwargs.get('use_discrimination', True),
            coral_hidden_dim=kwargs.get('coral_hidden_dim', None),
            use_coral_thresholds=kwargs.get('use_coral_thresholds', True),
            coral_dropout=kwargs.get('coral_dropout', 0.1),
            threshold_coupling_config=threshold_config
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: deep_gpcm, attn_gpcm, coral, coral_gpcm")
    
    return model