"""Model factory and registry for knowledge tracing models."""

from typing import Dict, Type, Any
from core.base_model import BaseKnowledgeTracingModel

MODEL_REGISTRY: Dict[str, Type[BaseKnowledgeTracingModel]] = {}


def register_model(name: str):
    """Decorator to register models in the registry."""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def create_model(model_type: str, **kwargs) -> BaseKnowledgeTracingModel:
    """Factory function to create models from registry.
    
    Args:
        model_type: Name of the model to create
        **kwargs: Model-specific arguments
        
    Returns:
        Instantiated model
        
    Raises:
        ValueError: If model_type is not in registry
    """
    if model_type not in MODEL_REGISTRY:
        available_models = list(MODEL_REGISTRY.keys())
        raise ValueError(f"Unknown model: {model_type}. Available models: {available_models}")
    
    return MODEL_REGISTRY[model_type](**kwargs)


def list_models() -> Dict[str, Type[BaseKnowledgeTracingModel]]:
    """List all registered models."""
    return MODEL_REGISTRY.copy()


def get_model_info(model_type: str) -> Dict[str, Any]:
    """Get information about a registered model."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_type}")
    
    model_cls = MODEL_REGISTRY[model_type]
    return {
        "name": model_type,
        "class": model_cls.__name__,
        "module": model_cls.__module__,
        "docstring": model_cls.__doc__
    }