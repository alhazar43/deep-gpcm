# Deep-GPCM System Architecture

## Overview

The Deep-GPCM (Generalized Partial Credit Model) system is a production-ready knowledge tracing framework that combines deep learning memory networks with Item Response Theory (IRT) for polytomous response prediction. The system implements multiple model architectures including attention mechanisms, ordinal regression (CORAL), temporal analysis capabilities, and advanced adaptive hyperparameter optimization.

## System Architecture

### Core Components

```
deep-gpcm/
├── models/                    # Model implementations and components
├── training/                  # Training orchestration and loss functions
├── utils/                     # Utilities and metrics
├── config/                    # Configuration management
├── data/                      # Dataset management
├── optimization/              # Adaptive hyperparameter optimization
├── results/                   # Output organization
└── [train.py, evaluate.py, main.py]  # Entry points
```

### Data Flow Architecture

```
Data Input → Embedding → Memory Networks → IRT Parameters → GPCM → Predictions
     ↓            ↓            ↓              ↓           ↓          ↓
   Loaders   Strategies    DKVMN/Attention  θ,α,β    Probabilities  Metrics
```

## Model Architecture

### Base Architecture: Deep-GPCM

The core Deep-GPCM model follows a sophisticated multi-stage architecture:

#### 1. Embedding Layer
- **Component**: `models/components/embeddings.py`
- **Strategies**: Linear decay, learned embeddings, one-hot
- **Purpose**: Convert discrete question-response pairs into continuous representations
- **Output**: Embedded sequences for memory network processing

#### 2. Memory Network (DKVMN)
- **Component**: `models/components/memory_networks.py`
- **Architecture**: Dynamic Key-Value Memory Network
- **Key Components**:
  - Static key memory matrix (learned)
  - Dynamic value memory (per-batch initialization)
  - Read/write heads with correlation-based attention
  - Erase-add memory update mechanism

```python
class DKVMN:
    key_memory_matrix: [memory_size, key_dim]      # Static knowledge concepts
    value_memory_matrix: [batch_size, memory_size, value_dim]  # Dynamic states
    read_head: correlation-weighted retrieval
    write_head: erase-add memory updates
```

#### 3. IRT Parameter Extraction
- **Component**: `models/components/irt_layers.py`
- **Parameters Extracted**:
  - Student abilities (θ): Current knowledge state
  - Item discrimination (α): Question difficulty gradients
  - Item thresholds (β): Category boundary parameters
- **Architecture**: Multi-layer neural networks with proper scaling

#### 4. GPCM Probability Layer
- **Component**: `models/components/irt_layers.py`
- **Function**: Converts IRT parameters to category probabilities
- **Formula**: Implements proper GPCM probability computation with numerical stability

### Enhanced Architectures

#### Attention-Enhanced GPCM
- **Implementation**: `models/implementations/attention_gpcm.py`
- **Features**:
  - Multi-head attention mechanisms
  - Positional encoding for temporal patterns
  - Attention refinement cycles
  - Learnable vs linear embedding strategies

#### CORAL-Enhanced GPCM
- **Implementation**: `models/implementations/coral_gpcm_proper.py`
- **Features**:
  - Ordinal regression with CORAL loss
  - Proper IRT-CORAL architectural separation
  - Combined loss optimization
  - Threshold coupling mechanisms

#### Temporal Attention GPCM
- **Implementation**: `models/implementations/stable_temporal_attention_gpcm.py`
- **Features**:
  - Relative positional attention
  - Temporal window mechanisms
  - Stabilized gradient flow
  - Production-ready temporal modeling

## Pipeline Architecture

### Training Pipeline

#### 1. Unified Argument System
- **Component**: `utils/args.py`
- **Features**: Backward-compatible argument parsing with validation
- **Modes**: Single training, K-fold CV, hyperparameter optimization

#### 2. Training Orchestration
- **Entry Point**: `train.py`
- **Workflow**:
  ```
  Data Loading → Model Creation → Loss Configuration → Training Loop → Validation → Model Saving
  ```

#### 3. Cross-Validation System
- **Outer Loop**: Model evaluation with train/test splits
- **Inner Loop**: Hyperparameter optimization (when enabled)
- **Metrics**: Comprehensive evaluation with QWK optimization

#### 4. Adaptive Hyperparameter Optimization
- **Components**: `optimization/enhanced_adaptive_hyperopt.py`, `optimization/adaptive_scheduler.py`
- **Method**: Enhanced Bayesian optimization with adaptive epoch allocation
- **Features**:
  - Model-aware parameter search (12-15 parameters vs original 5)
  - 3-phase adaptive epoch allocation (5→15→40 epochs)
  - Intelligent early stopping and convergence detection
  - Fallback safety system with automatic degradation
  - AI-generated analysis and recommendations
  - Parameter importance ranking and interaction analysis

### Evaluation Pipeline

#### 1. Model Loading and Validation
- **Entry Point**: `evaluate.py`
- **Auto-detection**: Intelligent model type inference from paths
- **Compatibility**: Backward compatibility with legacy model formats

#### 2. Comprehensive Metrics System
- **Component**: `utils/metrics.py`
- **Metrics**:
  - Categorical accuracy
  - Ordinal accuracy (adjacent category tolerance)
  - Quadratic Weighted Kappa (QWK)
  - Mean Absolute Error (MAE)
  - Kendall's Tau, Spearman correlation
  - Cohen's Kappa, Cross-entropy

#### 3. Unified Prediction System
- **Component**: `utils/predictions.py`
- **Methods**: Argmax, threshold-based, adaptive threshold
- **Configuration**: Flexible prediction strategy selection

## Technical Implementation

### Model Factory System

#### Dynamic Model Registry
- **Component**: `models/factory.py`
- **Features**:
  - Centralized model configuration
  - Default parameter management
  - Loss function configuration
  - Hyperparameter search spaces
  - Model validation and metadata

```python
MODEL_REGISTRY = {
    'deep_gpcm': {
        'class': DeepGPCM,
        'default_params': {...},
        'hyperparameter_grid': {...},
        'loss_config': {...}
    }
}
```

#### Model Creation Pattern
```python
def create_model(model_type, n_questions, n_cats, **kwargs):
    # 1. Validate model type
    # 2. Load configuration
    # 3. Merge parameters
    # 4. Instantiate model
    # 5. Attach metadata
```

### Configuration Management

#### Hierarchical Configuration System
- **Base Config**: `config/base.py` - Common parameters
- **Training Config**: `config/training.py` - Training-specific settings
- **Evaluation Config**: `config/evaluation.py` - Evaluation settings
- **Pipeline Config**: `config/pipeline.py` - End-to-end orchestration

#### Configuration Builder Pattern
```python
@dataclass
class TrainingConfig:
    model: str
    dataset: str
    epochs: int = 30
    n_folds: int = 5
    # ... validation and post-processing
```

### Loss Function Architecture

#### Unified Loss System
- **Component**: `training/losses.py`
- **Supported Losses**:
  - Cross-entropy (ce)
  - Focal loss (focal)
  - Quadratic Weighted Kappa (qwk)
  - CORAL ordinal loss (coral)
  - Combined multi-objective (combined)

#### Loss Configuration Pattern
```python
def create_loss_function(loss_type, n_cats, **kwargs):
    # 1. Parse loss type
    # 2. Create appropriate loss instance
    # 3. Configure parameters
    # 4. Return callable loss function
```

### Data Management

#### Dataset-Centric Organization
```
results/
└── {dataset}/
    ├── metrics/          # Training and evaluation results
    ├── models/           # Saved model checkpoints
    ├── plots/           # Visualization outputs
    └── irt_plots/       # IRT analysis visualizations
```

#### Path Management System
- **Component**: `utils/path_utils.py`
- **Features**:
  - Dual structure support (legacy/new)
  - Intelligent path resolution
  - Automatic directory creation
  - Backward compatibility

### Memory and Performance Optimization

#### Memory Network Efficiency
- **Batch-wise value memory initialization**: Prevents GPU memory explosions
- **Correlation-based attention**: Efficient O(memory_size) operations
- **Gradient clipping**: Prevents gradient explosion in deep architectures

#### Training Optimizations
- **Dynamic batch sizing**: Adapts to available GPU memory
- **Learning rate scheduling**: ReduceLROnPlateau with QWK monitoring
- **Early stopping**: Gradient norm and performance-based stopping

## Integration Points

### External Tool Integration

#### Hyperparameter Optimization
- **Library**: Optuna with Gaussian Process samplers
- **Integration**: Seamless pipeline integration with visualization
- **Analysis**: Parameter importance and convergence analysis

#### Visualization Pipeline
- **Component**: `utils/plot_metrics.py`
- **Outputs**: Training curves, ROC analysis, confusion matrices, calibration plots
- **IRT Analysis**: Temporal parameter evolution, probability heatmaps

### Extensibility Architecture

#### Model Extension Pattern
1. Inherit from `BaseKnowledgeTracingModel`
2. Implement required methods: `forward`, `get_logits`
3. Register in `MODEL_REGISTRY`
4. Add loss configuration
5. Define hyperparameter grid

#### Component Modularity
- **Embedding strategies**: Pluggable via strategy pattern
- **Memory networks**: Abstract base class for new architectures
- **Attention mechanisms**: Composable attention components
- **Loss functions**: Modular loss registry system

## Performance Characteristics

### Scalability Profile
- **Model Sizes**: 194K - 279K parameters
- **Memory Usage**: ~2-4GB GPU memory for typical datasets
- **Training Time**: 4-9 minutes (small), 58-85 minutes (medium), 108-153 minutes (large)
- **Throughput**: Batch processing with dynamic sizing

### Quality Assurance
- **Testing Framework**: Comprehensive unit and integration tests
- **Validation Pipeline**: Multi-metric evaluation with statistical testing
- **Backward Compatibility**: Legacy format support with migration utilities
- **Error Handling**: Graceful degradation and recovery mechanisms

## Deployment Architecture

### Production Readiness
- **Containerization**: Docker-compatible with conda environment
- **Monitoring**: Training progress tracking and resource monitoring
- **Logging**: Comprehensive logging with structured output
- **Configuration**: Environment-based configuration management

### Maintenance and Operations
- **Cleanup System**: Automated result cleanup with backup mechanisms
- **Migration Tools**: Legacy format migration utilities
- **Documentation**: Comprehensive API documentation and usage examples
- **Version Control**: Git-based version management with semantic versioning

## Adaptive Hyperparameter Optimization Architecture

### Multi-Stage Intelligence Stack
```
User Request
     ↓
┌─────────────────────────────────────┐
│ Adaptive Configuration Layer        │
│ • Model-aware parameter detection   │
│ • Search space expansion           │
│ • Resource allocation planning     │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│ Enhanced Bayesian Optimizer         │
│ • TPE sampler with adaptive epochs  │
│ • Trial result analysis            │
│ • Performance pattern recognition  │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│ Adaptive Scheduler                  │
│ • Dynamic epoch allocation         │
│ • Early stopping logic            │
│ • Fallback safety system          │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│ Model Training & Evaluation         │
│ • Cross-validation execution       │
│ • Metric collection               │
│ • Performance validation          │
└─────────────────────────────────────┘
     ↓
┌─────────────────────────────────────┐
│ Analysis & Recommendation Engine    │
│ • Parameter importance ranking     │
│ • Performance insight generation   │
│ • Next-step recommendations       │
└─────────────────────────────────────┘
```

### Enhanced Search Space Architecture
```
ExpandedSearchSpace
├── Model Detection → Model-Aware Parameter Filter
├── Base Parameters (Original 5)
│   ├── memory_size, final_fc_dim, dropout_rate
│   └── ce_weight_logit, focal_weight_logit
├── Architectural Parameters
│   ├── Common: key_dim, value_dim
│   ├── Attention: embed_dim, n_heads, n_cycles  
│   └── Advanced: num_layers, hidden_dim
└── Learning Parameters
    ├── Optimization: lr, weight_decay
    ├── Training: batch_size
    └── Regularization: grad_clip, label_smoothing
```

### Adaptive Epoch Allocation System
```
Trial Evaluation Pipeline:
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Phase 1     │    │ Phase 2     │    │ Phase 3     │
│ 5 epochs    │───▶│ 15 epochs   │───▶│ 40 epochs   │
│ Quick scan  │    │ Standard    │    │ Full eval   │
│ All trials  │    │ Promising   │    │ Best only   │
└─────────────┘    └─────────────┘    └─────────────┘
```

### Parameter Flow Architecture
```
Model Name → Parameter Filter → Search Space → Optimization → Results
    ↓              ↓               ↓             ↓          ↓
deep_gpcm → Core+Common → 12 params → TPE → Best Config
attn_*    → +Attention  → 15 params → TPE → Best Config
temporal  → +Advanced   → 18 params → TPE → Best Config
```

### Safety and Fallback Architecture
```
Performance Monitor
├── Trial Success Rate Tracking
├── Consecutive Failure Detection
├── Performance Degradation Analysis
└── Automatic Fallback Trigger
    └── Revert to Original 5-Parameter Search
```

### Key Implementation Classes

#### AdaptiveScheduler
- **Purpose**: Manages intelligent epoch allocation strategy
- **Features**: 3-phase allocation, early stopping, performance tracking
- **Location**: `optimization/adaptive_scheduler.py`

#### ExpandedSearchSpace
- **Purpose**: Model-aware parameter space generation
- **Features**: Automatic parameter filtering, architecture detection
- **Extensibility**: Easy addition of new parameter types

#### EnhancedAdaptiveHyperopt
- **Purpose**: Main optimization coordinator
- **Features**: Trial management, fallback handling, AI analysis generation
- **Output**: Comprehensive reports with actionable insights

### Integration Points
- **Unified Parser**: Seamlessly integrates with existing train.py/main.py
- **Model Factory**: Automatic parameter compatibility detection
- **Results Pipeline**: Enhanced output with adaptive analysis reports
- **Backward Compatibility**: Original optimization available via --no_adaptive

This architecture provides a robust, scalable, and maintainable foundation for knowledge tracing research and production deployments, with clear separation of concerns, extensive configurability, and state-of-the-art adaptive hyperparameter optimization.