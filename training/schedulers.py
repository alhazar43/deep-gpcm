"""
Learning rate scheduler utilities for Deep-GPCM training.
"""

import torch
import torch.optim.lr_scheduler as lr_scheduler
from typing import Dict, Any, Optional


def create_scheduler(optimizer: torch.optim.Optimizer, scheduler_type: str = 'reduce_on_plateau',
                    **kwargs) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """
    Create learning rate scheduler.
    
    Args:
        optimizer: PyTorch optimizer
        scheduler_type: Type of scheduler ('reduce_on_plateau', 'cosine', 'step', 'exponential', 'none')
        **kwargs: Additional scheduler arguments
        
    Returns:
        Configured scheduler or None
    """
    
    scheduler_type = scheduler_type.lower()
    
    if scheduler_type == 'none' or scheduler_type is None:
        return None
    
    elif scheduler_type == 'reduce_on_plateau':
        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=kwargs.get('mode', 'max'),  # 'max' for metrics like QWK where higher is better
            factor=kwargs.get('factor', 0.5),
            patience=kwargs.get('patience', 10),
            threshold=kwargs.get('threshold', 1e-4),
            min_lr=kwargs.get('min_lr', 1e-7),
            verbose=kwargs.get('verbose', True)
        )
    
    elif scheduler_type == 'cosine':
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=kwargs.get('T_max', 50),
            eta_min=kwargs.get('eta_min', 0),
            last_epoch=kwargs.get('last_epoch', -1)
        )
    
    elif scheduler_type == 'step':
        return lr_scheduler.StepLR(
            optimizer,
            step_size=kwargs.get('step_size', 30),
            gamma=kwargs.get('gamma', 0.1),
            last_epoch=kwargs.get('last_epoch', -1)
        )
    
    elif scheduler_type == 'multistep':
        return lr_scheduler.MultiStepLR(
            optimizer,
            milestones=kwargs.get('milestones', [30, 60, 90]),
            gamma=kwargs.get('gamma', 0.1),
            last_epoch=kwargs.get('last_epoch', -1)
        )
    
    elif scheduler_type == 'exponential':
        return lr_scheduler.ExponentialLR(
            optimizer,
            gamma=kwargs.get('gamma', 0.95),
            last_epoch=kwargs.get('last_epoch', -1)
        )
    
    elif scheduler_type == 'cosine_restarts':
        return lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer,
            T_0=kwargs.get('T_0', 10),
            T_mult=kwargs.get('T_mult', 1),
            eta_min=kwargs.get('eta_min', 0),
            last_epoch=kwargs.get('last_epoch', -1)
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def get_scheduler_config(scheduler_type: str) -> Dict[str, Any]:
    """Get default configuration for scheduler type."""
    
    configs = {
        'reduce_on_plateau': {
            'mode': 'max',
            'factor': 0.5,
            'patience': 10,
            'threshold': 1e-4,
            'min_lr': 1e-7,
            'verbose': True
        },
        'cosine': {
            'T_max': 50,
            'eta_min': 0
        },
        'step': {
            'step_size': 30,
            'gamma': 0.1
        },
        'multistep': {
            'milestones': [30, 60, 90],
            'gamma': 0.1
        },
        'exponential': {
            'gamma': 0.95
        },
        'cosine_restarts': {
            'T_0': 10,
            'T_mult': 1,
            'eta_min': 0
        }
    }
    
    return configs.get(scheduler_type.lower(), configs['reduce_on_plateau'])


def create_scheduler_with_config(optimizer: torch.optim.Optimizer, 
                                config: Dict[str, Any]) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    """Create scheduler using configuration dictionary."""
    
    scheduler_type = config.pop('type', 'reduce_on_plateau')
    return create_scheduler(optimizer, scheduler_type, **config)