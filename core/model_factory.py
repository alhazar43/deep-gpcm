"""Model factory for creating Deep-GPCM models."""

from .model import DeepGPCM, AttentionGPCM
from .attention_enhanced import EnhancedAttentionGPCM


def create_model(model_type, n_questions, n_cats):
    """Create model based on type.
    
    Args:
        model_type: 'baseline' or 'akvmn'
        n_questions: Number of questions
        n_cats: Number of response categories
        
    Returns:
        Model instance
    """
    if model_type == 'baseline':
        model = DeepGPCM(
            n_questions=n_questions,
            n_cats=n_cats,
            memory_size=50,
            key_dim=50,
            value_dim=200,
            final_fc_dim=50
        )
    elif model_type == 'akvmn':
        # Use enhanced version with learnable parameters
        model = EnhancedAttentionGPCM(
            n_questions=n_questions,
            n_cats=n_cats,
            embed_dim=64,
            memory_size=50,
            key_dim=50,
            value_dim=200,
            final_fc_dim=50,
            n_heads=4,
            n_cycles=2,
            embedding_strategy="linear_decay",
            ability_scale=2.0  # Start with old AKVMN default
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model