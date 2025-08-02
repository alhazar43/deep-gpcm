"""
Deep-GPCM Models Package

Organized structure:
- base/: Base classes and interfaces
- implementations/: Concrete model implementations  
- components/: Reusable model components
- adaptive/: Adaptive and experimental features
- factory.py: Model creation utilities
"""

# Import all public classes for backward compatibility
from .base.base_model import BaseKnowledgeTracingModel
from .implementations.deep_gpcm import DeepGPCM
from .implementations.attention_gpcm import AttentionGPCM, EnhancedAttentionGPCM
# Legacy CORAL imports removed
from .implementations.coral_gpcm_proper import CORALGPCM
from .components.memory_networks import DKVMN, MemoryNetwork, MemoryHeadGroup
from .components.embeddings import (
    create_embedding_strategy,
    OrderedEmbedding,
    UnorderedEmbedding,
    LinearDecayEmbedding,
    AdjacentWeightedEmbedding
)
from .components.irt_layers import IRTParameterExtractor, GPCMProbabilityLayer
from .components.attention_layers import AttentionRefinementModule, EmbeddingProjection
# Legacy component imports removed - only needed for coral_gpcm_proper internals
from .factory import create_model

__all__ = [
    # Base
    'BaseKnowledgeTracingModel',
    # Models
    'DeepGPCM',
    'AttentionGPCM',
    'EnhancedAttentionGPCM',
    'CORALGPCM',  # coral_gpcm_proper - the only CORAL implementation
    # Components
    'DKVMN',
    'MemoryNetwork',
    'MemoryHeadGroup',
    'create_embedding_strategy',
    'IRTParameterExtractor',
    'GPCMProbabilityLayer',
    'AttentionRefinementModule',
    'EmbeddingProjection',
    # Legacy components removed
    # Factory
    'create_model',
]
