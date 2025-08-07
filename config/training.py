"""
Training-specific configuration classes.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path

from .base import BaseConfig, LossConfig, ValidationConfig, PathConfig


@dataclass
class TrainingConfig(BaseConfig):
    """Training-specific configuration with factory integration."""
    
    # Training parameters
    epochs: int = 30
    batch_size: int = 64
    lr: float = 0.001
    
    # Cross-validation
    n_folds: int = 5
    cv: bool = False  # Enable hyperparameter optimization
    
    # Loss configuration
    loss_config: LossConfig = field(default_factory=LossConfig)
    loss_config_override: bool = False  # User explicitly set loss config
    
    # Validation
    validation_config: ValidationConfig = field(default_factory=ValidationConfig)
    
    # Paths
    path_config: Optional[PathConfig] = None
    
    def __post_init__(self):
        super().__post_init__()
        
        # Initialize path config
        if self.path_config is None:
            self.path_config = PathConfig(dataset=self.dataset, model=self.model)
        
        # Apply factory loss config if not overridden
        if not self.loss_config_override:
            self.loss_config = LossConfig.from_factory(self.model)
    
    def get_training_args(self) -> List[str]:
        """Generate training arguments for subprocess calls."""
        args = [
            '--model', self.model,
            '--dataset', self.dataset,
            '--epochs', str(self.epochs),
            '--batch_size', str(self.batch_size),
            '--lr', str(self.lr),
            '--n_folds', str(self.n_folds),
            '--seed', str(self.seed)
        ]
        
        if self.device:
            args.extend(['--device', self.device])
        
        if self.cv:
            args.append('--cv')
        
        # Add loss configuration only if not factory default
        if not self.loss_config.is_factory_default(self.model):
            args.extend(self.loss_config.to_args())
        
        return args
    
    def get_model_params(self) -> Dict[str, Any]:
        """Get model parameters with factory defaults applied."""
        base_params = {
            'n_questions': None,  # Will be set from data
            'n_cats': None,       # Will be set from data
        }
        
        # Apply factory defaults
        factory_params = self.apply_factory_defaults()
        base_params.update(factory_params)
        
        return base_params
    
    def summary(self) -> str:
        """Generate configuration summary."""
        lines = [
            f"Training Configuration for {self.model}:",
            f"  Dataset: {self.dataset}",
            f"  Epochs: {self.epochs}",
            f"  Batch size: {self.batch_size}",
            f"  Learning rate: {self.lr}",
            f"  Device: {self.device}",
            f"  Cross-validation: {'Yes' if self.cv else f'{self.n_folds}-fold training'}",
            f"  Loss: {self.loss_config.loss_type}",
        ]
        
        if self.loss_config.loss_type in ['combined', 'triple_coral']:
            lines.append(f"    Weights - CE: {self.loss_config.ce_weight}, "
                        f"Focal: {self.loss_config.focal_weight}, "
                        f"QWK: {self.loss_config.qwk_weight}, "
                        f"CORAL: {self.loss_config.coral_weight}")
        
        return '\n'.join(lines)


@dataclass
class HyperparameterConfig:
    """Hyperparameter optimization configuration."""
    
    # Optimization strategy
    strategy: str = 'bayesian'  # 'grid', 'random', 'bayesian'
    n_trials: int = 50
    timeout: Optional[int] = None  # seconds
    
    # Search space (will be merged with factory defaults)
    search_space: Optional[Dict[str, Any]] = None
    
    # Optimization targets
    primary_metric: str = 'qwk'
    minimize_primary: bool = False
    
    # Multi-objective optimization
    secondary_metrics: Optional[List[str]] = None
    metric_weights: Optional[Dict[str, float]] = None
    
    # Pruning
    enable_pruning: bool = True
    pruning_strategy: str = 'median'  # 'median', 'percentile'
    
    def __post_init__(self):
        if self.search_space is None:
            self.search_space = {}
        
        if self.secondary_metrics is None:
            self.secondary_metrics = ['training_time', 'model_size']
        
        if self.metric_weights is None:
            self.metric_weights = {
                self.primary_metric: 1.0,
                'training_time': -0.1,  # Prefer faster training
                'model_size': -0.05     # Prefer smaller models
            }


@dataclass 
class ExperimentConfig:
    """Experiment tracking and reproducibility configuration."""
    
    # Experiment metadata
    experiment_name: Optional[str] = None
    tags: Optional[List[str]] = None
    description: Optional[str] = None
    
    # Reproducibility
    deterministic: bool = True
    benchmark_mode: bool = False  # Enable cudnn benchmark
    
    # Logging
    log_level: str = 'INFO'
    log_metrics_every: int = 1
    save_checkpoints: bool = True
    checkpoint_every: int = 10
    
    # Tracking
    track_gradients: bool = False
    track_weights: bool = False
    track_activations: bool = False
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []
        
        if self.experiment_name is None:
            import datetime
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            self.experiment_name = f"experiment_{timestamp}"


@dataclass
class MultiModelConfig:
    """Configuration for training multiple models."""
    
    models: List[str] = field(default_factory=lambda: ['deep_gpcm', 'attn_gpcm', 'coral_gpcm_proper', 'coral_gpcm_fixed'])
    dataset: str = 'synthetic_OC'
    
    # Training settings (applied to all models)
    epochs: int = 30
    cv: bool = False
    
    # Parallel execution
    parallel: bool = False
    max_workers: int = 2
    
    # Model-specific overrides
    model_overrides: Optional[Dict[str, Dict[str, Any]]] = None
    
    def __post_init__(self):
        if self.model_overrides is None:
            self.model_overrides = {}
        
        # Validate all models
        from models.factory import get_all_model_types
        available = get_all_model_types()
        invalid_models = [m for m in self.models if m not in available]
        if invalid_models:
            raise ValueError(f"Invalid models: {invalid_models}. Available: {available}")
    
    def get_training_configs(self) -> List[TrainingConfig]:
        """Generate individual training configs for each model."""
        configs = []
        
        for model in self.models:
            # Base configuration
            config = TrainingConfig(
                model=model,
                dataset=self.dataset,
                epochs=self.epochs,
                cv=self.cv
            )
            
            # Apply model-specific overrides
            if model in self.model_overrides:
                overrides = self.model_overrides[model]
                for key, value in overrides.items():
                    if hasattr(config, key):
                        setattr(config, key, value)
            
            configs.append(config)
        
        return configs