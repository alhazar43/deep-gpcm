"""Model factory for creating Deep-GPCM models."""

from .model import DeepGPCM, AttentionGPCM
from .attention_enhanced import EnhancedAttentionGPCM
from .coral_gpcm import HybridCORALGPCM, EnhancedCORALGPCM, ThresholdCouplingConfig


def create_model(model_type, n_questions, n_cats, **kwargs):
    """Create model based on type.
    
    Args:
        model_type: 'deep_gpcm', 'attn_gpcm', 'coral_gpcm', 'ecoral_gpcm'
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
    
    elif model_type == 'coral_gpcm':
        # Hybrid CORAL-GPCM model 
        model = HybridCORALGPCM(
            **common_params,
            ability_scale=kwargs.get('ability_scale', 1.0),
            use_discrimination=kwargs.get('use_discrimination', True),
            use_coral_structure=kwargs.get('use_coral_structure', True),
            blend_weight=kwargs.get('blend_weight', 0.5)
        )
    
    elif model_type == 'ecoral_gpcm':
        # Enhanced CORAL-GPCM model WITH threshold coupling (no adaptive blending)
        model = EnhancedCORALGPCM(
            **common_params,
            ability_scale=kwargs.get('ability_scale', 1.0),
            use_discrimination=kwargs.get('use_discrimination', True),
            enable_threshold_coupling=kwargs.get('enable_threshold_coupling', True),
            coupling_type=kwargs.get('coupling_type', 'linear'),
            gpcm_weight=kwargs.get('gpcm_weight', 0.7),
            coral_weight=kwargs.get('coral_weight', 0.3),
            # Adaptive blending disabled by default for backward compatibility
            enable_adaptive_blending=False,
            blend_weight=kwargs.get('blend_weight', 0.5)
        )
    
    elif model_type == 'minimal_adaptive_coral_gpcm':
        # Enhanced CORAL-GPCM model WITH minimal adaptive threshold-distance blending
        model = EnhancedCORALGPCM(
            **common_params,
            ability_scale=kwargs.get('ability_scale', 1.0),
            use_discrimination=kwargs.get('use_discrimination', True),
            # *** THRESHOLD COUPLING DISABLED TO PREVENT CONFLICT WITH ADAPTIVE BLENDING ***
            enable_threshold_coupling=False,  # Disabled to prevent interaction with adaptive blending
            coupling_type=kwargs.get('coupling_type', 'linear'),
            gpcm_weight=kwargs.get('gpcm_weight', 0.7),
            coral_weight=kwargs.get('coral_weight', 0.3),
            # *** MINIMAL ADAPTIVE BLENDING ENABLED ***
            enable_adaptive_blending=True,
            use_full_blender=False,  # Use minimal blender for maximum stability
            blend_weight=kwargs.get('blend_weight', 0.5),  # Fallback weight
            range_sensitivity_init=kwargs.get('range_sensitivity_init', 0.01),  # Ultra-conservative for gradient stability
            distance_sensitivity_init=kwargs.get('distance_sensitivity_init', 0.01),  # Ultra-conservative for gradient stability
            baseline_bias_init=kwargs.get('baseline_bias_init', 0.0)
        )
    
    elif model_type == 'adaptive_coral_gpcm':
        # Enhanced CORAL-GPCM model WITH full adaptive threshold-distance blending (standard architecture)
        model = EnhancedCORALGPCM(
            **common_params,
            ability_scale=kwargs.get('ability_scale', 1.0),
            use_discrimination=kwargs.get('use_discrimination', True),
            # *** THRESHOLD COUPLING DISABLED TO PREVENT CONFLICT WITH ADAPTIVE BLENDING ***
            enable_threshold_coupling=False,  # Disabled to prevent interaction with adaptive blending
            coupling_type=kwargs.get('coupling_type', 'linear'),
            gpcm_weight=kwargs.get('gpcm_weight', 0.7),
            coral_weight=kwargs.get('coral_weight', 0.3),
            # *** FULL ADAPTIVE BLENDING ENABLED ***
            enable_adaptive_blending=True,
            use_full_blender=kwargs.get('use_full_blender', True),  # Use full blender by default
            blend_weight=kwargs.get('blend_weight', 0.5),  # Fallback weight
            range_sensitivity_init=kwargs.get('range_sensitivity_init', 0.1),  # Full blender conservative defaults
            distance_sensitivity_init=kwargs.get('distance_sensitivity_init', 0.2),  # Full blender conservative defaults
            baseline_bias_init=kwargs.get('baseline_bias_init', 0.0),
            # BGT stability parameters for full blender
            use_bgt_transforms=kwargs.get('use_bgt_transforms', True),
            gradient_clipping=kwargs.get('gradient_clipping', 0.5),
            parameter_bounds=kwargs.get('parameter_bounds', True)
        )
    
    elif model_type == 'full_adaptive_coral_gpcm':
        # Enhanced CORAL-GPCM model WITH full adaptive threshold-distance blending and larger architecture
        # Use larger architecture for ~400k parameters
        large_params = {
            'n_questions': n_questions,
            'n_cats': n_cats,
            'memory_size': kwargs.get('memory_size', 100),  # Increased from 50
            'key_dim': kwargs.get('key_dim', 100),  # Increased from 50
            'value_dim': kwargs.get('value_dim', 400),  # Increased from 200
            'final_fc_dim': kwargs.get('final_fc_dim', 100),  # Increased from 50
            'embedding_strategy': kwargs.get('embedding_strategy', 'linear_decay'),
            'dropout_rate': kwargs.get('dropout_rate', 0.0)
        }
        
        model = EnhancedCORALGPCM(
            **large_params,
            ability_scale=kwargs.get('ability_scale', 1.0),
            use_discrimination=kwargs.get('use_discrimination', True),
            # *** THRESHOLD COUPLING DISABLED TO PREVENT CONFLICT WITH ADAPTIVE BLENDING ***
            enable_threshold_coupling=False,  # Disabled to prevent interaction with adaptive blending
            coupling_type=kwargs.get('coupling_type', 'linear'),
            gpcm_weight=kwargs.get('gpcm_weight', 0.7),
            coral_weight=kwargs.get('coral_weight', 0.3),
            # *** FULL ADAPTIVE BLENDING ENABLED WITH BGT STABILITY ***
            enable_adaptive_blending=True,
            use_full_blender=True,  # Use full blender with learnable parameters
            blend_weight=kwargs.get('blend_weight', 0.5),  # Fallback weight
            range_sensitivity_init=kwargs.get('range_sensitivity_init', 0.1),  # Conservative defaults
            distance_sensitivity_init=kwargs.get('distance_sensitivity_init', 0.2),  # Conservative defaults
            baseline_bias_init=kwargs.get('baseline_bias_init', 0.0),
            # BGT stability parameters
            use_bgt_transforms=kwargs.get('use_bgt_transforms', True),
            gradient_clipping=kwargs.get('gradient_clipping', 0.5),  # More aggressive clipping
            parameter_bounds=kwargs.get('parameter_bounds', True)
        )
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. "
                        f"Available: deep_gpcm, attn_gpcm, coral_gpcm, ecoral_gpcm, "
                        f"minimal_adaptive_coral_gpcm, adaptive_coral_gpcm, full_adaptive_coral_gpcm")
    
    return model