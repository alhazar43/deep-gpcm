"""
Optimizer creation utilities for Deep-GPCM training.
"""

import torch
import torch.optim as optim
from typing import Dict, Any, Iterator


def create_optimizer(parameters: Iterator, optimizer_type: str = 'adam', 
                    lr: float = 0.001, **kwargs) -> torch.optim.Optimizer:
    """
    Create optimizer with standard configurations.
    
    Args:
        parameters: Model parameters
        optimizer_type: Type of optimizer ('adam', 'adamw', 'sgd', 'rmsprop')
        lr: Learning rate
        **kwargs: Additional optimizer arguments
        
    Returns:
        Configured optimizer
    """
    
    optimizer_type = optimizer_type.lower()
    
    if optimizer_type == 'adam':
        return optim.Adam(
            parameters,
            lr=lr,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0.0)
        )
    
    elif optimizer_type == 'adamw':
        return optim.AdamW(
            parameters,
            lr=lr,
            betas=kwargs.get('betas', (0.9, 0.999)),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0.01)
        )
    
    elif optimizer_type == 'sgd':
        return optim.SGD(
            parameters,
            lr=lr,
            momentum=kwargs.get('momentum', 0.9),
            weight_decay=kwargs.get('weight_decay', 0.0),
            nesterov=kwargs.get('nesterov', False)
        )
    
    elif optimizer_type == 'rmsprop':
        return optim.RMSprop(
            parameters,
            lr=lr,
            alpha=kwargs.get('alpha', 0.99),
            eps=kwargs.get('eps', 1e-8),
            weight_decay=kwargs.get('weight_decay', 0.0),
            momentum=kwargs.get('momentum', 0.0)
        )
    
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def get_optimizer_config(optimizer_type: str) -> Dict[str, Any]:
    """Get default configuration for optimizer type."""
    
    configs = {
        'adam': {
            'lr': 0.001,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0.0
        },
        'adamw': {
            'lr': 0.001,
            'betas': (0.9, 0.999),
            'eps': 1e-8,
            'weight_decay': 0.01
        },
        'sgd': {
            'lr': 0.01,
            'momentum': 0.9,
            'weight_decay': 0.0,
            'nesterov': False
        },
        'rmsprop': {
            'lr': 0.001,
            'alpha': 0.99,
            'eps': 1e-8,
            'weight_decay': 0.0,
            'momentum': 0.0
        }
    }
    
    return configs.get(optimizer_type.lower(), configs['adam'])


def create_optimizer_with_config(model: torch.nn.Module, config: Dict[str, Any]) -> torch.optim.Optimizer:
    """Create optimizer using configuration dictionary."""
    
    optimizer_type = config.pop('type', 'adam')
    return create_optimizer(model.parameters(), optimizer_type, **config)