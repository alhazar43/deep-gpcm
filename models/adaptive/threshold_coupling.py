from dataclasses import dataclass
from typing import Literal, Optional
import torch
import torch.nn as nn


@dataclass
class ThresholdCouplingConfig:
    """Simple configuration for threshold coupling."""
    enabled: bool = False
    coupling_type: str = "linear"
    gpcm_weight: float = 0.7
    coral_weight: float = 0.3

