# Deep-GPCM Codebase Refactoring Plan

## Current Issues
1. **Duplicated Code**: DKVMN implementation exists in baseline.py, imported by akvmn_gpcm.py
2. **Poor Naming**: "baseline" doesn't describe DKVMN-GPCM functionality  
3. **Lack of Modularity**: Models aren't easily extensible, embedding strategies scattered
4. **Mixed Concerns**: IRT logic mixed with neural architecture
5. **Unclear Organization**: Loss functions split across utils files

## Proposed Architecture

### 1. Core Components (`core/`)
```
core/
├── __init__.py
├── memory_networks.py      # DKVMN, future memory architectures
├── embeddings.py          # All embedding strategies  
├── irt_layers.py          # IRT parameter extraction layers
└── base_model.py          # Abstract base class for all models
```

### 2. Models (`models/`)
```
models/
├── __init__.py
├── dkvmn_gpcm.py         # Current "baseline" → clearer name
├── attention_dkvmn_gpcm.py # Current "akvmn_gpcm" → descriptive name
└── model_factory.py       # Model creation utilities
```

### 3. Training Components (`training/`)
```
training/
├── __init__.py
├── losses.py              # All loss functions (OrdinalLoss, CrossEntropy, etc.)
├── optimizers.py          # Custom optimizers if needed
└── trainers.py            # Training loop abstractions
```

### 4. Data (`data/`)
```
data/
├── __init__.py
├── datasets.py            # Dataset classes
├── loaders.py             # Data loading utilities
└── generators.py          # Synthetic data generation
```

### 5. Evaluation (`evaluation/`)
```
evaluation/
├── __init__.py
├── metrics.py             # Keep existing metrics
└── irt_analysis.py        # IRT parameter analysis utilities
```

### 6. Experiments (`experiments/`)
```
experiments/
├── train.py               # Training script
├── evaluate.py            # Evaluation script
├── analyze_irt.py         # IRT analysis script
└── configs/               # Configuration files
```

### 7. Visualization (`visualization/`)
```
visualization/
├── __init__.py
├── performance_plots.py   # Training curves, metrics
├── irt_plots.py          # IRT parameter visualizations
└── temporal_analysis.py   # Temporal IRT animations
```

## File Renaming Map

### Models
| Current Name | New Name | Rationale |
|-------------|----------|-----------|
| `models/baseline.py` | `models/dkvmn_gpcm.py` | Describes actual architecture |
| `models/akvmn_gpcm.py` | `models/attention_dkvmn_gpcm.py` | Clear about attention enhancement |

### Scripts  
| Current Name | New Name | Location Change |
|-------------|----------|-----------------|
| `train.py` | `experiments/train.py` | Group experiments |
| `evaluate.py` | `experiments/evaluate.py` | Group experiments |
| `plot_irt.py` | `visualization/irt_plots.py` | Group visualizations |
| `animate_irt.py` | `visualization/temporal_analysis.py` | Clearer purpose |
| `plot_metrics.py` | `visualization/performance_plots.py` | Clearer purpose |
| `data_gen.py` | `data/generators.py` | Group data utilities |

### Utilities
| Current Name | New Name | Reason |
|-------------|----------|--------|
| `utils/gpcm_utils.py` | Split → `training/losses.py` + `core/irt_layers.py` | Separate concerns |
| `utils/loss_utils.py` | Merge → `training/losses.py` | Consolidate losses |
| `utils/data_utils.py` | `data/loaders.py` | Clearer purpose |

## Detailed Refactoring Steps

### Phase 1: Extract Core Components
1. **Create `core/memory_networks.py`**
   - Move DKVMN class from baseline.py
   - Create abstract MemoryNetwork base class
   - Ensure backward compatibility

2. **Create `core/embeddings.py`**
   - Extract embedding functions from baseline.py
   - Create EmbeddingStrategy abstract class
   - Implement: OrderedEmbedding, LinearDecayEmbedding, etc.

3. **Create `core/irt_layers.py`**
   - Extract IRT parameter networks (ability, discrimination, threshold)
   - Create reusable IRT components

4. **Create `core/base_model.py`**
   ```python
   class BaseKnowledgeTracingModel(nn.Module):
       """Abstract base for all KT models."""
       def forward(self, questions, responses):
           raise NotImplementedError
       
       def extract_irt_parameters(self, ...):
           """Optional IRT parameter extraction."""
           pass
   ```

### Phase 2: Reorganize Models
1. **Rename and refactor models/**
   - `baseline.py` → `dkvmn_gpcm.py`
   - `akvmn_gpcm.py` → `attention_dkvmn_gpcm.py`
   - Remove duplicate code, import from core/

2. **Create model factory**
   ```python
   def create_model(model_type: str, **kwargs):
       """Factory for model creation."""
       models = {
           'dkvmn_gpcm': DKVMNGPCM,
           'attention_dkvmn_gpcm': AttentionDKVMNGPCM,
       }
       return models[model_type](**kwargs)
   ```

### Phase 3: Consolidate Training Components
1. **Merge loss functions**
   - Move OrdinalLoss from utils/gpcm_utils.py
   - Consolidate with utils/loss_utils.py
   - Create training/losses.py

2. **Create training abstractions**
   - Extract training loop logic
   - Create Trainer class for different training strategies

### Phase 4: Update Scripts
1. **Update imports in all scripts**
   - Maintain backward compatibility during transition
   - Use model factory instead of direct imports

2. **Update configuration handling**
   - Create YAML/JSON configs for experiments
   - Centralize hyperparameters

## Implementation Order

### Week 1: Core Extraction (No Breaking Changes)
- [ ] Create core/ directory structure
- [ ] Extract DKVMN to core/memory_networks.py
- [ ] Extract embeddings to core/embeddings.py
- [ ] Create base model class
- [ ] Add compatibility imports in original files

### Week 2: Model Refactoring
- [ ] Create new model files with clear names
- [ ] Implement model factory
- [ ] Update model inheritance
- [ ] Test performance parity

### Week 3: Training & Data Organization
- [ ] Consolidate loss functions
- [ ] Create training abstractions
- [ ] Reorganize data utilities
- [ ] Update experiment scripts

### Week 4: Testing & Migration
- [ ] Comprehensive testing
- [ ] Performance validation
- [ ] Update documentation
- [ ] Create migration guide

## Performance Validation Checkpoints

After each phase, validate:
1. Model outputs remain identical (numerical precision)
2. Training convergence matches original
3. Final metrics match baseline:
   - Baseline GPCM: 69.9% accuracy, 0.628 QWK
   - Attention DKVMN: 71.8% accuracy, 0.696 QWK

## Deep Learning Research Code Standards

Following established patterns from:
- **PyTorch Lightning**: Modular trainers, clear separation of concerns
- **HuggingFace Transformers**: Model registry, configuration management
- **FAIR Research**: Abstract base classes, factory patterns
- **Stanford NLP**: Clear module organization, extensible architectures

### Key Principles:
1. **Single Responsibility**: Each module has one clear purpose
2. **Dependency Injection**: Models receive components, not create them
3. **Configuration-Driven**: Hyperparameters in configs, not code
4. **Type Hints**: Full typing for better IDE support and clarity
5. **Registry Pattern**: Central registration for models, losses, etc.

## Benefits of New Architecture

1. **Modularity**: Easy to add new memory networks, embeddings, or models
2. **Clarity**: File names match functionality
3. **Reusability**: Shared components reduce duplication
4. **Extensibility**: Clear interfaces for new implementations
5. **Maintainability**: Separated concerns, cleaner code
6. **Research-Ready**: Follows DL research code standards

## Code Examples: New Modular Structure

### Example 1: Core Components
```python
# core/base_model.py
from abc import ABC, abstractmethod
import torch.nn as nn

class BaseKnowledgeTracingModel(nn.Module, ABC):
    """Base class for all knowledge tracing models."""
    
    @abstractmethod
    def forward(self, questions, responses):
        """Forward pass."""
        pass
    
    def get_model_info(self):
        """Model information."""
        return {
            "name": self.__class__.__name__,
            "parameters": sum(p.numel() for p in self.parameters())
        }

# core/embeddings.py
class EmbeddingStrategy(ABC):
    """Base class for embedding strategies."""
    
    @abstractmethod
    def embed(self, questions, responses, n_questions, n_cats):
        pass

class LinearDecayEmbedding(EmbeddingStrategy):
    """Linear decay embedding with triangular weights."""
    
    def embed(self, questions, responses, n_questions, n_cats):
        # Implementation from baseline.py
        ...
```

### Example 2: Model Implementation
```python
# models/dkvmn_gpcm.py
from core.base_model import BaseKnowledgeTracingModel
from core.memory_networks import DKVMN
from core.embeddings import LinearDecayEmbedding
from core.irt_layers import IRTParameterExtractor

class DKVMNGPCM(BaseKnowledgeTracingModel):
    """DKVMN with GPCM for polytomous responses."""
    
    def __init__(self, config):
        super().__init__()
        self.memory = DKVMN(config.memory_size, config.key_dim, config.value_dim)
        self.embedding = LinearDecayEmbedding()
        self.irt_extractor = IRTParameterExtractor(config)
        
    def forward(self, questions, responses):
        # Clean implementation using components
        embedded = self.embedding.embed(questions, responses, self.n_questions, self.n_cats)
        memory_output = self.memory(embedded)
        theta, alpha, beta = self.irt_extractor(memory_output)
        return self.compute_gpcm_probability(theta, alpha, beta)
```

### Example 3: Model Registry
```python
# models/model_factory.py
from typing import Dict, Type
from core.base_model import BaseKnowledgeTracingModel

MODEL_REGISTRY: Dict[str, Type[BaseKnowledgeTracingModel]] = {}

def register_model(name: str):
    """Decorator to register models."""
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator

def create_model(model_type: str, **kwargs) -> BaseKnowledgeTracingModel:
    """Create model from registry."""
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_type}")
    return MODEL_REGISTRY[model_type](**kwargs)

# Usage
@register_model("dkvmn_gpcm")
class DKVMNGPCM(BaseKnowledgeTracingModel):
    ...
```

### Example 4: Adding a New Model
```python
# models/transformer_gpcm.py
@register_model("transformer_gpcm")
class TransformerGPCM(BaseKnowledgeTracingModel):
    """Transformer-based knowledge tracing with GPCM."""
    
    def __init__(self, config):
        super().__init__()
        self.transformer = nn.TransformerEncoder(...)
        self.embedding = LinearDecayEmbedding()  # Reuse existing
        self.irt_extractor = IRTParameterExtractor(config)  # Reuse existing
        
    def forward(self, questions, responses):
        embedded = self.embedding.embed(questions, responses, self.n_questions, self.n_cats)
        transformer_output = self.transformer(embedded)
        theta, alpha, beta = self.irt_extractor(transformer_output)
        return self.compute_gpcm_probability(theta, alpha, beta)

# Automatically available in training:
python experiments/train.py --model transformer_gpcm
```

## Migration Strategy

1. **Parallel Implementation**: Keep old structure while building new
2. **Compatibility Layer**: Temporary imports for smooth transition
3. **Gradual Migration**: Update scripts one by one
4. **Validation Suite**: Ensure identical behavior throughout

## Success Criteria

- [ ] All tests pass with new structure
- [ ] Performance metrics unchanged
- [ ] Code coverage maintained/improved
- [ ] Documentation updated
- [ ] New model can be added in <30 minutes