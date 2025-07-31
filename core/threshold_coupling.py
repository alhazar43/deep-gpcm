"""
Clean threshold coupling mechanisms for integrating GPCM β thresholds 
with CORAL ordinal thresholds.

This module provides a minimal, clean implementation following the 
simplification principles outlined in the project TODO.md.
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from dataclasses import dataclass


@dataclass
class ThresholdCouplingConfig:
    """Simple configuration for threshold coupling."""
    enabled: bool = False
    coupling_type: str = "linear"
    gpcm_weight: float = 0.7
    coral_weight: float = 0.3


class ThresholdCoupler(ABC):
    """Base class for threshold coupling mechanisms."""
    
    @abstractmethod
    def couple(self, 
               gpcm_thresholds: torch.Tensor,
               coral_thresholds: torch.Tensor, 
               student_ability: torch.Tensor,
               item_discrimination: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Couple GPCM and CORAL thresholds.
        
        Args:
            gpcm_thresholds: GPCM β thresholds, shape (batch_size, seq_len, n_thresholds)
            coral_thresholds: CORAL τ thresholds, shape (n_thresholds,)
            student_ability: Student abilities θ, shape (batch_size, seq_len)
            item_discrimination: Item discrimination α, shape (batch_size, seq_len)
            
        Returns:
            unified_thresholds: Coupled thresholds, shape (batch_size, seq_len, n_thresholds)
        """
        pass


class LinearThresholdCoupler(nn.Module, ThresholdCoupler):
    """
    Simple linear coupling of GPCM and CORAL thresholds.
    
    Uses learnable weights for weighted combination:
    unified = gpcm_weight * gpcm_thresholds + coral_weight * coral_thresholds
    """
    
    def __init__(self, n_thresholds: int, gpcm_weight: float = 0.7, coral_weight: float = 0.3):
        """
        Initialize linear threshold coupler.
        
        Args:
            n_thresholds: Number of thresholds (n_cats - 1)
            gpcm_weight: Initial weight for GPCM thresholds
            coral_weight: Initial weight for CORAL thresholds
        """
        super().__init__()
        self.n_thresholds = n_thresholds
        
        # Learnable weights with proper initialization
        self.gpcm_weight = nn.Parameter(torch.tensor(gpcm_weight))
        self.coral_weight = nn.Parameter(torch.tensor(coral_weight))
        
        # Initialize weights to sum to 1.0
        self._normalize_weights()
    
    def _normalize_weights(self):
        """Ensure weights sum to 1.0 during initialization."""
        with torch.no_grad():
            total = self.gpcm_weight + self.coral_weight
            self.gpcm_weight.div_(total)
            self.coral_weight.div_(total)
    
    def couple(self, 
               gpcm_thresholds: torch.Tensor,
               coral_thresholds: torch.Tensor, 
               student_ability: torch.Tensor,
               item_discrimination: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Apply linear coupling.
        
        Args:
            gpcm_thresholds: Shape (batch_size, seq_len, n_thresholds)
            coral_thresholds: Shape (n_thresholds,)
            student_ability: Shape (batch_size, seq_len) - unused in linear coupling
            item_discrimination: Shape (batch_size, seq_len) - unused in linear coupling
            
        Returns:
            unified_thresholds: Shape (batch_size, seq_len, n_thresholds)
        """
        # Validate input shapes
        batch_size, seq_len, n_thresh = gpcm_thresholds.shape
        assert n_thresh == self.n_thresholds, f"Expected {self.n_thresholds} thresholds, got {n_thresh}"
        assert coral_thresholds.shape == (self.n_thresholds,), f"CORAL thresholds shape mismatch: {coral_thresholds.shape}"
        
        # Expand CORAL thresholds to match GPCM dimensions
        coral_expanded = coral_thresholds.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1)
        
        # Apply learnable weighted combination
        unified_thresholds = (
            self.gpcm_weight * gpcm_thresholds + 
            self.coral_weight * coral_expanded
        )
        
        return unified_thresholds
    
    def get_coupling_info(self) -> Dict[str, float]:
        """Get current coupling weights for diagnostics."""
        return {
            'gpcm_weight': self.gpcm_weight.item(),
            'coral_weight': self.coral_weight.item(),
            'weight_sum': (self.gpcm_weight + self.coral_weight).item()
        }


class ThresholdCouplerFactory:
    """Factory for creating threshold couplers based on configuration."""
    
    @staticmethod
    def create_coupler(config: ThresholdCouplingConfig, n_thresholds: int) -> Optional[ThresholdCoupler]:
        """
        Create threshold coupler based on configuration.
        
        Args:
            config: Threshold coupling configuration
            n_thresholds: Number of thresholds
            
        Returns:
            ThresholdCoupler instance or None if disabled
        """
        if not config.enabled:
            return None
        
        if config.coupling_type == "linear":
            return LinearThresholdCoupler(
                n_thresholds=n_thresholds,
                gpcm_weight=config.gpcm_weight,
                coral_weight=config.coral_weight
            )
        else:
            raise ValueError(f"Unknown coupling type: {config.coupling_type}. Available: linear")


def test_linear_threshold_coupler():
    """Test the LinearThresholdCoupler implementation."""
    print("Testing LinearThresholdCoupler")
    print("=" * 40)
    
    # Test parameters
    batch_size, seq_len, n_thresholds = 4, 8, 3
    
    # Create sample data
    gpcm_thresholds = torch.randn(batch_size, seq_len, n_thresholds)
    coral_thresholds = torch.randn(n_thresholds)
    student_ability = torch.randn(batch_size, seq_len)
    
    # Create coupler
    coupler = LinearThresholdCoupler(n_thresholds, gpcm_weight=0.6, coral_weight=0.4)
    
    # Test coupling
    unified = coupler.couple(gpcm_thresholds, coral_thresholds, student_ability)
    
    # Validate results
    assert unified.shape == gpcm_thresholds.shape, f"Shape mismatch: {unified.shape} vs {gpcm_thresholds.shape}"
    assert torch.isfinite(unified).all(), "Non-finite values in output"
    
    # Test coupling info
    info = coupler.get_coupling_info()
    print(f"Coupling weights: GPCM={info['gpcm_weight']:.3f}, CORAL={info['coral_weight']:.3f}")
    print(f"Weight sum: {info['weight_sum']:.3f}")
    
    print("✅ LinearThresholdCoupler test passed")
    
    # Test factory
    config = ThresholdCouplingConfig(enabled=True, coupling_type="linear")
    factory_coupler = ThresholdCouplerFactory.create_coupler(config, n_thresholds)
    assert factory_coupler is not None, "Factory failed to create coupler"
    
    # Test disabled config
    disabled_config = ThresholdCouplingConfig(enabled=False)
    disabled_coupler = ThresholdCouplerFactory.create_coupler(disabled_config, n_thresholds)
    assert disabled_coupler is None, "Factory should return None for disabled config"
    
    print("✅ ThresholdCouplerFactory test passed")
    print("✅ All tests completed successfully!")


if __name__ == "__main__":
    test_linear_threshold_coupler()