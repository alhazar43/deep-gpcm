# Deep-GPCM Architecture Documentation

## System Overview

The Deep-GPCM system implements three core models for educational data analysis using Item Response Theory (IRT) and deep learning. The architecture follows a layered modular design with shared components, factory-based model creation, and unified training/evaluation pipelines.

## Core Models

### 1. DeepGPCM (`deep_gpcm`)
**Class**: `models.implementations.deep_gpcm.DeepGPCM`  
**Description**: Baseline model combining DKVMN memory networks with GPCM probability computation.

**Architecture**:
```
Input → Embedding → DKVMN Memory → Summary Network → IRT Layers → GPCM Probabilities
```

**Key Components**:
- **Embedding Strategy**: Linear decay embedding with triangular weights
- **Memory Network**: DKVMN with read/write operations
- **IRT Parameter Extraction**: Theta (ability), alpha (discrimination), beta (thresholds)
- **GPCM Layer**: Generalized partial credit model probability computation

### 2. EnhancedAttentionGPCM (`attn_gpcm_learn`)
**Class**: `models.implementations.attention_gpcm.EnhancedAttentionGPCM`  
**Description**: Attention-enhanced model with learnable embeddings and multi-head refinement.

**Architecture**:
```
Input → Embedding → Projection → Attention Refinement → DKVMN Memory → Summary → IRT → GPCM
```

**Key Enhancements**:
- **Attention Refinement**: Multi-cycle attention with fusion and gating
- **Embedding Projection**: Fixed-dimension projection for attention compatibility
- **Learnable Features**: Enhanced parameter learning through attention mechanisms

### 3. OrdinalAttentionGPCM (`attn_gpcm_linear`)
**Class**: `models.implementations.ordinal_attention_gpcm.OrdinalAttentionGPCM`  
**Description**: Standalone implementation with ordinal embedding and adaptive suppression.

**Architecture**:
```
Input → Fixed Linear Decay → Temperature Suppression → Attention → Memory → IRT → GPCM
```

**Key Features**:
- **Ordinal Embedding**: Enhanced triangular weights with suppression mechanisms
- **Temperature Control**: Adaptive weight sharpening to reduce adjacent category interference
- **Standalone Design**: Complete reimplementation for ordinal response modeling

## Architectural Layers

### Layer 1: Factory and Registry (`models/factory.py`)

**Model Registry**:
- Central model configuration with metadata, hyperparameters, and loss configurations
- Factory pattern for model instantiation with parameter validation
- Automatic configuration management and inheritance

**Key Functions**:
- `create_model()`: Main factory function with parameter merging
- `get_model_config()`: Configuration retrieval with inheritance support
- `validate_model_type()`: Model type validation against registry

### Layer 2: Base Model Architecture (`models/base/`)

**BaseKnowledgeTracingModel** (`models/base/base_model.py`):
```python
class BaseKnowledgeTracingModel(nn.Module, ABC):
    @abstractmethod
    def forward(self, questions, responses) -> Tuple[torch.Tensor, ...]
```

**Common Interface**:
- Abstract forward method with standardized input/output
- Model metadata management
- IRT parameter extraction interface

### Layer 3: Component Library (`models/components/`)

#### Memory Networks (`memory_networks.py`)
**DKVMN Implementation**:
```
Key Memory (Static) ←→ Value Memory (Dynamic)
        ↓                    ↓
   Correlation Weight    Read/Write Heads
        ↓                    ↓
   Memory Attention ←→ Memory Updates
```

**Key Classes**:
- `DKVMN`: Main memory network with key-value separation
- `MemoryHeadGroup`: Read/write operations with erase-add mechanism
- `MemoryNetwork`: Abstract base for memory architectures

#### Embedding Strategies (`embeddings.py`)
**Strategy Pattern**:
```python
class EmbeddingStrategy(ABC):
    @abstractmethod
    def embed(self, q_data, r_data, n_questions, n_cats) -> torch.Tensor
    
    @property
    @abstractmethod
    def output_dim(self) -> int
```

**Implementations**:
- `LinearDecayEmbedding`: Triangular weight decay for ordinal responses
- `OrderedEmbedding`: Correctness + score components
- `UnorderedEmbedding`: One-hot categorical encoding

#### Attention Mechanisms (`attention_layers.py`)
**AttentionRefinementModule**:
```
Input Embeddings
       ↓
Multi-Head Attention (n_cycles)
       ↓
Feature Fusion + Gating
       ↓
Layer Normalization
       ↓
Refined Embeddings
```

**Key Features**:
- Iterative refinement with configurable cycles
- Residual connections with gating mechanisms
- Multi-head attention with fusion layers

#### IRT Parameter Extraction (`irt_layers.py`)
**IRTParameterExtractor**:
```
Neural Features → Ability Network → θ (Student Ability)
                ↓
Question Features → Threshold Network → β (Item Difficulty)
                ↓
Combined Features → Discrimination Network → α (Discrimination)
```

**GPCMProbabilityLayer**:
- Implements GPCM probability computation: P(X=k) = exp(sum)/sum(exp)
- Handles ordinal response categories with cumulative logits

### Layer 4: Training Pipeline (`training/`)

#### Loss Functions (`losses.py`)
**Unified Loss Architecture**:
```python
def create_loss_function(loss_type, loss_config, class_weights=None):
    # Supports: 'ce', 'focal', 'combined', 'ordinal_ce'
    # Combined loss: CE + Focal + QWK + Weighted Ordinal
```

**Loss Components**:
- Cross-entropy with class weighting
- Focal loss for imbalanced data
- Quadratic weighted kappa (QWK) for ordinal correlation
- Weighted ordinal loss for adjacent category penalties

#### Training Configuration (`config/`)
**Configuration Hierarchy**:
```
BaseConfig
    ├── TrainingConfig (epochs, batch_size, lr)
    ├── LossConfig (loss_type, weights)
    ├── ValidationConfig (early_stopping, monitoring)
    └── PathConfig (data, models, results)
```

**Factory Integration**:
- Automatic parameter loading from model registry
- Configuration validation and inheritance
- Command-line argument generation

### Layer 5: Data and Evaluation Pipeline

#### Data Processing (`data/loaders.py`)
**Data Flow**:
```
Raw Data → Sequence Parsing → Tensor Conversion → DataLoader → Model Input
```

**Features**:
- Automatic format detection and validation
- Cross-validation splitting with stratification
- Batch processing with padding and masking

#### Metrics and Evaluation (`utils/metrics.py`)
**Comprehensive Metrics**:
- Classification: Accuracy, precision, recall, F1-score
- Ordinal: QWK, ordinal accuracy, adjacent accuracy
- Probabilistic: AUC, log-likelihood, calibration metrics
- IRT-specific: Parameter recovery, ability estimation

## Dependency Analysis

### Import Chain Analysis

**Core Dependencies**:
```
models/factory.py
├── models/implementations/deep_gpcm.py
│   ├── models/base/base_model.py
│   ├── models/components/memory_networks.py
│   ├── models/components/embeddings.py
│   └── models/components/irt_layers.py
├── models/implementations/attention_gpcm.py
│   ├── models/implementations/deep_gpcm.py (inheritance)
│   └── models/components/attention_layers.py
└── models/implementations/ordinal_attention_gpcm.py
    ├── models/base/base_model.py
    ├── models/components/memory_networks.py
    ├── models/components/embeddings.py (custom implementation)
    ├── models/components/irt_layers.py
    └── models/components/attention_layers.py
```

**Training Dependencies**:
```
train.py
├── models/factory.py (model creation)
├── training/losses.py (loss functions)
├── utils/metrics.py (evaluation)
├── config/training.py (configuration)
└── data/loaders.py (data processing)
```

### Component Relationships

**DeepGPCM Composition**:
- Inherits: `BaseKnowledgeTracingModel`
- Composes: `DKVMN`, `IRTParameterExtractor`, `GPCMProbabilityLayer`
- Uses: `LinearDecayEmbedding` (via factory)

**EnhancedAttentionGPCM Composition**:
- Inherits: `DeepGPCM` (complete inheritance)
- Extends: `AttentionRefinementModule`, `EmbeddingProjection`
- Overrides: `create_embeddings()`, `process_embeddings()`

**OrdinalAttentionGPCM Composition**:
- Inherits: `BaseKnowledgeTracingModel` (standalone)
- Composes: Custom `FixedLinearDecayEmbedding`, `DKVMN`, `AttentionRefinementModule`
- Implements: Temperature suppression and ordinal-aware attention

## Data Flow Architecture

### Training Pipeline
```
1. Configuration Loading
   ├── Factory registry lookup
   ├── Parameter validation
   └── Loss configuration

2. Data Processing
   ├── Dataset loading and parsing
   ├── Cross-validation splitting
   └── DataLoader creation

3. Model Creation
   ├── Factory instantiation
   ├── Parameter injection
   └── Component initialization

4. Training Loop
   ├── Forward pass: questions, responses → probabilities
   ├── Loss computation: combined loss with multiple components
   ├── Backward pass: gradient computation and clipping
   └── Optimization: parameter updates with scheduling

5. Validation and Monitoring
   ├── Metric computation: QWK, accuracy, AUC
   ├── Early stopping: patience-based monitoring
   └── Model checkpointing: best model saving
```

### Inference Pipeline
```
1. Model Loading
   ├── Model type detection from path
   ├── Factory reconstruction
   └── State dict loading

2. Data Preprocessing
   ├── Sequence formatting
   ├── Tensor conversion
   └── Batch preparation

3. Forward Pass
   ├── Embedding creation
   ├── Memory operations
   ├── IRT parameter extraction
   └── GPCM probability computation

4. Post-processing
   ├── Probability extraction
   ├── Prediction generation
   └── Metric computation
```

## Configuration Management

### Factory Pattern Implementation
```python
MODEL_REGISTRY = {
    'deep_gpcm': {
        'class': DeepGPCM,
        'default_params': {...},
        'loss_config': {...},
        'training_config': {...}
    }
}
```

**Features**:
- Centralized configuration with inheritance
- Parameter validation and merging
- Automatic hyperparameter grid generation
- Loss configuration integration

### Runtime Configuration
**Parameter Priority** (highest to lowest):
1. Explicit kwargs in `create_model()`
2. Configuration overrides
3. Factory default parameters
4. Model class defaults

## Performance and Scalability

### Memory Management
- Batch-wise memory initialization for DKVMN
- Gradient accumulation for large sequences
- Automatic mixed precision support

### Optimization Features
- Adaptive learning rate scheduling
- Gradient clipping for stability
- Early stopping with patience monitoring
- Hyperparameter optimization integration

### Monitoring and Diagnostics
- Comprehensive metric tracking
- Model parameter analysis
- Training curve visualization
- IRT parameter recovery validation

## Extension Points

### Adding New Models
1. Implement `BaseKnowledgeTracingModel`
2. Add to `MODEL_REGISTRY` in factory
3. Configure default parameters and loss settings
4. Implement model-specific components if needed

### Custom Components
1. Extend abstract base classes (`EmbeddingStrategy`, `MemoryNetwork`)
2. Implement required abstract methods
3. Register in factory or inject via parameters

### New Loss Functions
1. Add to `create_loss_function()` in `training/losses.py`
2. Update model registry configurations
3. Implement gradient-compatible computation

This architecture provides a robust, extensible framework for educational data modeling with clear separation of concerns, factory-based configuration management, and comprehensive evaluation capabilities.