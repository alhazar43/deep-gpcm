"""
Base configuration classes with factory integration.
"""

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List
from pathlib import Path
import torch

from models.factory import (
    get_all_model_types, validate_model_type, 
    get_model_loss_config, get_model_default_params
)


@dataclass
class LossConfig:
    """Unified loss configuration with factory integration."""
    loss_type: str = 'ce'
    ce_weight: float = 1.0
    focal_weight: float = 0.0
    focal_gamma: float = 2.0
    focal_alpha: float = 1.0
    qwk_weight: float = 0.5
    coral_weight: float = 0.0
    ordinal_weight: float = 0.0
    
    @classmethod
    def from_factory(cls, model_type: str) -> 'LossConfig':
        """Create loss config from factory registry."""
        if not validate_model_type(model_type):
            raise ValueError(f"Invalid model type: {model_type}")
        
        factory_config = get_model_loss_config(model_type)
        if not factory_config:
            return cls()  # Default configuration
        
        # Map factory config to our fields
        kwargs = {}
        if 'type' in factory_config:
            kwargs['loss_type'] = factory_config['type']
        
        # Map weight parameters
        for key, value in factory_config.items():
            if key.endswith('_weight') and hasattr(cls, key):
                kwargs[key] = value
            elif key in ['focal_gamma', 'focal_alpha']:
                kwargs[key] = value
                
        return cls(**kwargs)
    
    def to_args(self) -> List[str]:
        """Convert to command line arguments."""
        args = ['--loss', self.loss_type]
        
        if self.loss_type in ['combined']:
            if self.ce_weight != 1.0:
                args.extend(['--ce_weight', str(self.ce_weight)])
            if self.focal_weight != 0.0:
                args.extend(['--focal_weight', str(self.focal_weight)])
            if self.qwk_weight != 0.5:
                args.extend(['--qwk_weight', str(self.qwk_weight)])
            if self.coral_weight != 0.0:
                args.extend(['--coral_weight', str(self.coral_weight)])
            if self.ordinal_weight != 0.0:
                args.extend(['--ordinal_weight', str(self.ordinal_weight)])
        
        if self.loss_type == 'focal':
            if self.focal_gamma != 2.0:
                args.extend(['--focal_gamma', str(self.focal_gamma)])
            if self.focal_alpha != 1.0:
                args.extend(['--focal_alpha', str(self.focal_alpha)])
        
        if self.loss_type == 'ordinal_ce':
            if self.ordinal_weight != 0.0:
                args.extend(['--ordinal_weight', str(self.ordinal_weight)])
        
        return args
    
    def is_factory_default(self, model_type: str) -> bool:
        """Check if this config matches factory defaults."""
        factory_config = self.from_factory(model_type)
        return self == factory_config


@dataclass
class BaseConfig:
    """Base configuration with factory integration and validation."""
    model: str
    dataset: str = 'synthetic_OC'
    device: Optional[str] = None
    seed: int = 42
    
    def __post_init__(self):
        self.validate()
        self._set_device()
    
    def validate(self):
        """Validate configuration parameters."""
        if not validate_model_type(self.model):
            available = ', '.join(get_all_model_types())
            raise ValueError(f"Invalid model '{self.model}'. Available: {available}")
        
        if self.seed < 0:
            raise ValueError("Seed must be non-negative")
    
    def _set_device(self):
        """Set device if not specified."""
        if self.device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def get_model_defaults(self) -> Dict[str, Any]:
        """Get factory defaults for this model."""
        return get_model_default_params(self.model)
    
    def apply_factory_defaults(self, **overrides) -> Dict[str, Any]:
        """Apply factory defaults with optional overrides."""
        defaults = self.get_model_defaults()
        defaults.update(overrides)
        return defaults


@dataclass
class PathConfig:
    """Path management configuration."""
    dataset: str
    model: str
    base_dir: str = '.'
    
    def __post_init__(self):
        self.base_path = Path(self.base_dir)
        self.data_path = self.base_path / 'data' / self.dataset
        self.results_path = self.base_path / 'results'
        self.models_path = self.base_path / 'saved_models' / self.dataset
        
        # Ensure directories exist
        self.results_path.mkdir(parents=True, exist_ok=True)
        self.models_path.mkdir(parents=True, exist_ok=True)
    
    @property
    def model_save_path(self) -> Path:
        """Path for saving trained models."""
        return self.models_path / f'best_{self.model}.pth'
    
    @property
    def train_results_path(self) -> Path:
        """Path for training results."""
        return self.results_path / 'train' / self.dataset / f'train_{self.model}.json'
    
    @property 
    def test_results_path(self) -> Path:
        """Path for test results."""
        return self.results_path / 'test' / self.dataset / f'test_{self.model}.json'


@dataclass
class ValidationConfig:
    """Validation and quality control configuration."""
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 0.001
    monitor_metric: str = 'qwk'
    
    # Gradient monitoring
    gradient_clip_value: Optional[float] = 1.0
    gradient_monitoring: bool = True
    explosion_threshold: float = 10.0
    
    # Model validation
    validate_every: int = 1
    save_best_only: bool = True
    
    def should_stop_early(self, history: List[float]) -> bool:
        """Determine if training should stop early."""
        if not self.early_stopping or len(history) < self.patience:
            return False
        
        recent_scores = history[-self.patience:]
        best_recent = max(recent_scores)
        improvement = best_recent - recent_scores[0]
        
        return improvement < self.min_delta