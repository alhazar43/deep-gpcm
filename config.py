"""
Unified Configuration System for Deep-GPCM
Clean, modular configuration for baseline and AKVMN models.
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
import json
import os


@dataclass
class BaseConfig:
    """Base configuration for all models."""
    # Model identification
    model_type: str = "baseline"  # baseline, akvmn
    dataset_name: str = "synthetic_OC"
    
    # Data parameters
    n_questions: Optional[int] = None  # Auto-detected from data
    n_cats: int = 4
    
    # Core GPCM parameters
    memory_size: int = 50
    key_dim: int = 50
    value_dim: int = 200
    final_fc_dim: int = 50
    embedding_strategy: str = "linear_decay"
    prediction_method: str = "cumulative"
    
    # Training parameters
    batch_size: int = 32
    learning_rate: float = 0.001
    epochs: int = 30
    loss_type: str = "crossentropy"
    
    # System parameters
    device: str = "auto"
    save_dir: str = "save_models"
    results_dir: str = "results"


@dataclass
class AKVMNConfig(BaseConfig):
    """AKVMN-specific configuration."""
    model_type: str = "akvmn"
    
    # AKVMN-specific parameters
    embed_dim: int = 64
    n_cycles: int = 2
    adaptive_cycles: bool = False


@dataclass
class DeepIntegrationConfig(BaseConfig):
    """Deep Integration specific configuration."""
    model_type: str = "deep_integration"
    
    # Deep Integration parameters
    embed_dim: int = 64
    n_cycles: int = 2


def get_model_config(model_type: str, **kwargs) -> BaseConfig:
    """
    Create model configuration.
    
    Args:
        model_type: Type of model ("baseline", "akvmn", "deep_integration")
        **kwargs: Additional configuration parameters
        
    Returns:
        Configuration instance
    """
    if model_type == "akvmn":
        return AKVMNConfig(**kwargs)
    elif model_type == "deep_integration":
        return DeepIntegrationConfig(**kwargs)
    else:
        return BaseConfig(model_type="baseline", **kwargs)


def get_preset_configs() -> Dict[str, BaseConfig]:
    """Get preset configurations for different models."""
    return {
        "baseline": BaseConfig(
            model_type="baseline",
            embedding_strategy="linear_decay",
            loss_type="crossentropy",
            epochs=30
        ),
        
        "akvmn": AKVMNConfig(
            model_type="akvmn",
            embed_dim=64,
            n_cycles=2,
            adaptive_cycles=False,
            embedding_strategy="linear_decay",
            loss_type="crossentropy",
            epochs=30
        ),
        
        "deep_integration": DeepIntegrationConfig(
            model_type="deep_integration",
            embed_dim=64,
            n_cycles=2,
            embedding_strategy="linear_decay",
            loss_type="crossentropy",
            epochs=30
        )
    }


def load_config(config_path: str) -> BaseConfig:
    """Load configuration from JSON file."""
    with open(config_path, 'r') as f:
        config_dict = json.load(f)
    
    model_type = config_dict.get("model_type", "baseline")
    return get_model_config(model_type, **config_dict)


def save_config(config: BaseConfig, config_path: str):
    """Save configuration to JSON file."""
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    
    with open(config_path, 'w') as f:
        config_dict = {
            field.name: getattr(config, field.name)
            for field in config.__dataclass_fields__.values()
        }
        json.dump(config_dict, f, indent=2)


# Preset configurations
BASELINE_CONFIG = get_preset_configs()["baseline"]
AKVMN_CONFIG = get_preset_configs()["akvmn"]
DEEP_INTEGRATION_CONFIG = get_preset_configs()["deep_integration"]