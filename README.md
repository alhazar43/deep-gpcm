# Deep-GPCM: Production-Ready Knowledge Tracing System

**Version 2.0.0** - Complete system restoration with validated enhancements

Deep-GPCM is a production-ready neural knowledge tracing system for polytomous responses, featuring optimal loss functions, transformer enhancements, and uncertainty quantification.

## 🎉 **Major Release Highlights**

**✅ System Restored**: Fixed critical baseline issues (18.7% → 54.0-54.6% accuracy)  
**✅ Transformer Enhancement**: +1.5% consistent performance improvement (Production Ready)  
**✅ Uncertainty Quantification**: Complete Bayesian framework (Optimization Ready)  
**✅ Clean Codebase**: Streamlined production-ready architecture

## 🏆 **Performance Achievements**

| Component | Status | Accuracy | vs Baseline | Production Ready |
|-----------|--------|----------|-------------|------------------|
| **Fixed Baseline** | ✅ Stable | 54.0-54.6% | Reference | ✅ Yes |
| **Simplified Transformer** | ✅ Validated | 55.4-55.9% | +1.5% | ✅ Yes |
| **Bayesian GPCM** | ⚠️ Functional | 52.2-52.8% | -3.3% | ⚠️ Needs optimization |

## 🚀 **Quick Start (Production)**

### Installation
```bash
# Clone and install dependencies
git clone <repository>
cd deep-gpcm
pip install -r requirements.txt
```

### Basic Usage - Fixed Baseline
```python
from models.model import DeepGpcmModel

# Create production-ready baseline model
model = DeepGpcmModel(
    n_questions=50,           # Number of questions in domain
    n_cats=4,                 # Response categories {0,1,2,3}
    memory_size=40,           # DKVMN memory size
    key_dim=40,               # Memory key dimension
    value_dim=120,            # Memory value dimension
    final_fc_dim=40,          # Final layer dimension
    embedding_strategy='linear_decay'  # Optimal strategy
)

# Expected performance: 54.0-54.6% categorical accuracy
```

### Enhanced Usage - Transformer Integration
```python
from models.simplified_transformer import SimplifiedTransformerGPCM

# Create enhanced model with transformer
base_model = DeepGpcmModel(...)  # As above
transformer_model = SimplifiedTransformerGPCM(
    base_model,
    d_model=128,              # Transformer dimension
    nhead=8,                  # Attention heads
    num_layers=2,             # Transformer layers
    dropout=0.1               # Regularization
)

# Expected performance: 55.4-55.9% accuracy (+1.5% improvement)
```

### Advanced Usage - Uncertainty Quantification
```python
from models.bayesian_gpcm import BayesianGPCM

# Create model with uncertainty estimates
bayesian_model = BayesianGPCM(
    base_model,
    n_concepts=8,             # Knowledge concepts
    state_dim=12,             # Knowledge state dimension
    n_mc_samples=10,          # Monte Carlo samples
    kl_weight=0.001           # Optimized KL weight
)

# Get predictions with uncertainty
predictions = bayesian_model.predict_with_uncertainty(q_data, r_data)
# Returns: mean, std, confidence intervals, epistemic/aleatoric uncertainty
```

## 🔧 **Training**

### Basic Training
```bash
# Train baseline model
python train.py --dataset synthetic_OC --embedding_strategy linear_decay --epochs 30

# Cross-validation training
python train_cv.py --dataset synthetic_OC --n_folds 5 --epochs 20
```

### Model Evaluation
```bash
# Evaluate trained model
python evaluate.py --model_path save_models/best_model.pth --dataset synthetic_OC
```

### Data Generation
```bash
# Generate synthetic training data
python data_gen.py --format OC --categories 4 --students 800 --questions 50

# Generate larger dataset for analysis
python data_gen.py --format OC --categories 4 --students 1000 --output_dir data/large
```

## 📊 **Architecture Overview**

### Core Architecture
```
Input: (questions, responses) 
  ↓
Linear Decay Embedding (Optimal Strategy)
  ↓  
DKVMN Memory Network (Fixed Implementation)
  ↓
GPCM Parameter Prediction (θ, α, β)
  ↓
K-category Response Probabilities
```

### Enhancement Options
1. **Simplified Transformer**: Direct sequence modeling for +1.5% improvement
2. **Bayesian Framework**: Uncertainty quantification with variational inference
3. **Advanced Losses**: Cross-Entropy (best overall) or Focal Loss (best ordinal)

## 📁 **Project Structure**

```
deep-gpcm/
├── models/
│   ├── model.py                    # Fixed baseline GPCM
│   ├── memory.py                   # Restored DKVMN memory
│   ├── simplified_transformer.py  # Production transformer (+1.5%)
│   ├── bayesian_gpcm.py           # Uncertainty quantification
│   └── advanced_losses.py         # Optimal loss functions
├── data/                           # Synthetic datasets (OC/PC formats)
├── utils/
│   └── gpcm_utils.py              # Core utilities and metrics
├── evaluation/
│   └── metrics.py                 # Comprehensive evaluation framework
├── train.py                       # Single-fold training
├── train_cv.py                    # Cross-validation training
├── evaluate.py                    # Model evaluation
├── data_gen.py                    # Synthetic data generation
└── Documentation/
    ├── PRODUCTION_DEPLOYMENT_GUIDE.md  # Complete deployment guide
    ├── TECHNICAL_IMPLEMENTATION_LOG.md # Technical details
    └── PHASE_2_COMPLETION_REPORT.md    # Executive summary
```

## 🎯 **Configuration Guidelines**

### Model Sizing by Domain
```python
# Small Domain (≤25 questions, ≤1000 students)
small_config = {
    'memory_size': 20, 'key_dim': 20, 'value_dim': 60,
    'transformer': {'d_model': 64, 'nhead': 4, 'num_layers': 1}
}

# Medium Domain (25-100 questions, 1000-10000 students)
medium_config = {
    'memory_size': 40, 'key_dim': 40, 'value_dim': 120,
    'transformer': {'d_model': 128, 'nhead': 8, 'num_layers': 2}
}

# Large Domain (100+ questions, 10000+ students)
large_config = {
    'memory_size': 80, 'key_dim': 80, 'value_dim': 240,
    'transformer': {'d_model': 256, 'nhead': 8, 'num_layers': 3}
}
```

### Training Configuration
```python
# Optimizer setup
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = torch.nn.CrossEntropyLoss()  # Or create_loss_function('focal', num_classes=4, gamma=2.0)

# Multi-timestep training (recommended)
for t in range(min(4, probs.size(1))):
    target = r_data[:, t]
    logits = probs[:, t, :]
    loss += criterion(logits, target)
loss = loss / min(4, probs.size(1))
```

## 📈 **Data Formats**

### Ordered Categories (OC) - Recommended
```
48                              # sequence length
26,9,25,18,6,29,...            # question IDs
2,0,2,0,3,3,...                # response categories {0,1,2,3}
```

### Partial Credit (PC) - Alternative
```
48                              # sequence length  
26,9,25,18,6,29,...            # question IDs
0.667,0.000,0.667,0.000,...    # normalized scores [0,1]
```

## 🔍 **Evaluation Metrics**

The system provides comprehensive evaluation including:

- **Categorical Accuracy**: Standard classification accuracy
- **Ordinal Accuracy**: Order-preserving accuracy for educational assessment
- **Mean Absolute Error**: Distance-based error measurement
- **Quadratic Weighted Kappa**: Agreement measure for ordinal data
- **Prediction Consistency**: Training/inference alignment validation
- **Uncertainty Metrics**: Epistemic and aleatoric uncertainty (Bayesian model)

## 🛠️ **Troubleshooting**

### Common Issues

**Issue**: NaN losses during training
```python
# Solution: Ensure proper gradient clipping
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

**Issue**: Poor performance vs benchmarks
```python
# Check: Use linear_decay embedding (optimal)
model = DeepGpcmModel(..., embedding_strategy='linear_decay')

# Check: Multi-timestep loss computation
# Use 3-4 timesteps, not single timestep
```

**Issue**: Memory errors in old codebase
```python
# Fixed in v2.0.0: Update memory calls
# OLD: memory.attention(embed, ability)
# NEW: memory.attention(embed)
```

## 📋 **Migration from Previous Versions**

### From v1.x to v2.0.0
1. **Update Memory Calls**: All `attention()` and `write()` calls have new signatures
2. **Use Simplified Transformer**: Replace complex transformer with `SimplifiedTransformerGPCM`
3. **Validate Performance**: Ensure 54.0%+ accuracy restored
4. **Clean Dependencies**: Remove broken components (Phase 2.2, 2.3)

See `PRODUCTION_DEPLOYMENT_GUIDE.md` for detailed migration instructions.

## 🎯 **Roadmap**

### Phase 3 (Ready for Implementation)
- **Multi-Task Learning**: Joint optimization for difficulty estimation, learning trajectories
- **Advanced Training**: Curriculum learning, data augmentation, regularization
- **Bayesian Optimization**: Parameter tuning for performance recovery

### Research Applications
- **Educational Assessment**: Polytomous response modeling in online learning
- **Adaptive Testing**: Uncertainty-guided question selection
- **Learning Analytics**: Student knowledge state tracking and intervention triggers

## 📚 **Documentation**

- **PRODUCTION_DEPLOYMENT_GUIDE.md**: Complete deployment instructions and configurations
- **TECHNICAL_IMPLEMENTATION_LOG.md**: Detailed technical implementation record
- **PHASE_2_COMPLETION_REPORT.md**: Executive summary of system restoration
- **CHANGELOG.md**: Complete version history and breaking changes

## ⚙️ **Requirements**

```
torch>=1.9.0
numpy>=1.20.0
scikit-learn>=0.24.0
matplotlib>=3.0.0
seaborn>=0.11.0
pandas>=1.3.0
tqdm>=4.60.0
```

## 📄 **Citation**

```bibtex
@software{deep_gpcm_2025,
  title={Deep-GPCM: Production-Ready Knowledge Tracing for Polytomous Responses},
  version={2.0.0},
  year={2025},
  note={Transformer-enhanced DKVMN with uncertainty quantification}
}
```

---

**Deep-GPCM v2.0.0**: Restored, enhanced, and production-ready for educational assessment applications.