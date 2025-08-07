# Deep-GPCM Model Architecture Summary

## Core Models Overview

This document summarizes the 4 architecturally distinct models in the Deep-GPCM framework, their configurations, parameters, and mathematical formulations.

## 1. DeepGPCM (Baseline)
**File**: `models/implementations/deep_gpcm.py`  
**Class**: `DeepGPCM`  
**Model Key**: `deep_gpcm`  
**Color**: `#ff7f0e` (Orange)

### Architecture
- **Foundation**: DKVMN memory network + IRT parameter extraction
- **Mathematical Model**: Pure GPCM with neural parameter extraction
- **Parameters**: ~151K
- **Innovation**: Temporal adaptation of classical IRT through neural memory

### Configuration
```python
DeepGPCM(
    n_questions=n_questions,
    n_cats=n_cats,
    memory_size=50,
    key_dim=50,
    value_dim=200,
    final_fc_dim=50,
    embedding_strategy="linear_decay",
    ability_scale=1.0,
    use_discrimination=True,
    dropout_rate=0.0
)
```

### Loss Configuration
- **Loss Function**: Focal Loss
- **Mathematical Focus**: Addresses class imbalance in ordinal categories

### Mathematical Formulation
```
Z_{t,k} = Σ_{j=0}^k α_t(θ_t - β_{t,j})
P(Y_t = k | θ_t, α_t, β_t) = exp(Z_{t,k}) / Σ_{c=0}^{K-1} exp(Z_{t,c})

Where:
- θ_t ∈ [-3, 3]: Student ability (tanh-bounded)
- α_t ∈ [0.1, 3.0]: Item discrimination (softplus + ε)  
- β_{t,k}: Monotonic thresholds via cumulative sum
```

## 2. EnhancedAttentionGPCM (Attention-Enhanced)
**File**: `models/implementations/attention_gpcm.py`  
**Class**: `EnhancedAttentionGPCM`  
**Model Key**: `attn_gpcm`  
**Color**: `#1f77b4` (Blue)

### Architecture
- **Foundation**: DeepGPCM + Multi-head attention refinement
- **Enhancement**: Iterative attention cycles on embeddings before IRT extraction
- **Parameters**: ~302K (2x baseline due to attention layers)
- **Innovation**: Attention-based embedding refinement for ordinal sequences

### Configuration
```python
EnhancedAttentionGPCM(
    n_questions=n_questions,
    n_cats=n_cats,
    embed_dim=64,
    memory_size=50,
    key_dim=50,
    value_dim=200,
    final_fc_dim=50,
    n_heads=4,
    n_cycles=2,
    embedding_strategy="linear_decay",
    ability_scale=2.0,
    dropout_rate=0.1
)
```

### Loss Configuration
- **Loss Function**: Cross-Entropy Loss
- **Focus**: Standard classification approach

### Mathematical Formulation
```
embed_refined = AttentionRefinement(embed_base, n_cycles=2)
refined_t = gate_t ⊙ fusion(embed_t, attn(embed_t)) + (1 - gate_t) ⊙ embed_t

IRT Parameters: θ_t, α_t, β_t = IRTExtractor(embed_refined)
P(Y_t = k) = GPCM(θ_t, α_t, β_t)
```

## 3. CORALGPCM (Probability Blending)
**File**: `models/implementations/coral_gpcm_proper.py`  
**Class**: `CORALGPCM`  
**Model Key**: `coral_gpcm_proper`  
**Color**: `#e377c2` (Pink)

### Architecture
- **Foundation**: Dual-branch architecture with adaptive probability blending
- **IRT Branch**: Standard GPCM computation with β thresholds
- **CORAL Branch**: Ordinal regression with separate τ thresholds  
- **Integration**: Weighted mixture of probability distributions
- **Parameters**: ~154K
- **Innovation**: Ensemble-style combination of IRT and ordinal approaches

### Configuration
```python
CORALGPCM(
    n_questions=n_questions,
    n_cats=n_cats,
    memory_size=50,
    key_dim=50,
    value_dim=200,
    final_fc_dim=50,
    embedding_strategy="linear_decay",
    ability_scale=1.0,
    use_discrimination=True,
    coral_dropout=0.1,
    use_adaptive_blending=True,
    blend_weight=0.5
)
```

### Loss Configuration
- **Loss Function**: Combined Loss
  - Focal Loss: 40% weight
  - QWK Loss: 20% weight  
  - CORAL Loss: 40% weight
- **Focus**: Multi-objective optimization across IRT and ordinal metrics

### Mathematical Formulation
```
# IRT Branch
P_GPCM = GPCM(θ_t, α_t, β_t)

# CORAL Branch  
P(Y > k) = σ(α_t(θ_t - τ_k))  // Uses separate τ thresholds
P_CORAL = CumulativeToCategorical(P(Y > k))

# Adaptive Blending
λ = AdaptiveBlender(β_t, τ_k)  // Geometry-based blending
P_final = λ·P_GPCM + (1-λ)·P_CORAL
```

## 4. CORALGPCMFixed (Parameter Combination)
**File**: `models/implementations/coral_gpcm_fixed.py`  
**Class**: `CORALGPCMFixed`  
**Model Key**: `coral_gpcm_fixed`  
**Color**: `#ff0000` (Red)

### Architecture
- **Foundation**: Unified computation with combined threshold parameters
- **Parameter Fusion**: β (item-specific) + τ (global ordinal) at threshold level
- **Single Computation**: Direct GPCM with fused thresholds
- **Parameters**: ~154K  
- **Innovation**: Parameter-level fusion for ordinal-aware thresholds

### Configuration
```python
CORALGPCMFixed(
    n_questions=n_questions,
    n_cats=n_cats,
    memory_size=50,
    key_dim=50,
    value_dim=200,
    final_fc_dim=50,
    embedding_strategy="linear_decay",
    ability_scale=1.0,
    use_discrimination=True,
    coral_dropout=0.1
)
```

### Loss Configuration
- **Loss Function**: Combined Loss
  - Focal Loss: 40% weight
  - QWK Loss: 20% weight
  - CORAL Loss: 40% weight
- **Focus**: Same multi-objective approach as CORAL Proper

### Mathematical Formulation
```
# Parameter Combination
β_combined = β_t + τ_k  // Fuse item-specific and ordinal thresholds

# Single GPCM Computation
P_final = GPCM(θ_t, α_t, β_combined)

Where:
- β_t: Item-specific thresholds (temporal, per-item)
- τ_k: Global ordinal thresholds (ordered: τ_1 ≤ τ_2 ≤ τ_3)
```

## Removed Models (Configuration Variants)

### 1. TestGPCM → CORALGPCMFixed
- **Reason**: Identical architecture, only difference was CE loss vs Combined loss
- **Solution**: Use CORALGPCMFixed with `--loss ce` flag in training

### 2. AttentionGPCMNew → EnhancedAttentionGPCM  
- **Reason**: Identical architecture, only difference was combined loss weights
- **Solution**: Use EnhancedAttentionGPCM with `--loss combined --ce_weight 0.6 --qwk_weight 0.2 --focal_weight 0.2`

### 3. ModularAttentionGPCM → Removed
- **Reason**: Over-engineered with worse empirical performance
- **Performance**: QWK 0.682 vs 0.689 (EnhancedAttentionGPCM)
- **Complexity**: Significantly higher implementation and parameter complexity

## Model Comparison Matrix

| Model | Parameters | Innovation Type | Loss Function | Performance Focus |
|-------|------------|----------------|---------------|-------------------|
| DeepGPCM | ~151K | Neural-IRT Integration | Focal | Baseline Reference |
| EnhancedAttentionGPCM | ~302K | Attention Refinement | Cross-Entropy | Embedding Enhancement |
| CORALGPCM | ~154K | Probability Blending | Combined | IRT-Ordinal Ensemble |
| CORALGPCMFixed | ~154K | Parameter Fusion | Combined | Unified Ordinal Thresholds |

## Training Commands

### Individual Model Training
```bash
# Baseline
python train.py --model deep_gpcm --dataset DATASET --loss focal

# Attention Enhanced  
python train.py --model attn_gpcm --dataset DATASET --loss ce

# CORAL Probability Blending
python train.py --model coral_gpcm_proper --dataset DATASET --loss combined \
    --focal_weight 0.4 --qwk_weight 0.2 --coral_weight 0.4

# CORAL Parameter Fusion
python train.py --model coral_gpcm_fixed --dataset DATASET --loss combined \
    --focal_weight 0.4 --qwk_weight 0.2 --coral_weight 0.4
```

### Complete Pipeline
```bash
# Train all 4 core models
python main.py --models deep_gpcm attn_gpcm coral_gpcm_proper coral_gpcm_fixed \
    --dataset DATASET --epochs 30 --n_folds 5
```

## Research Contributions

1. **Neural-IRT Integration**: Temporal adaptation of psychometric models through memory networks
2. **Attention-Enhanced Embeddings**: Multi-head attention refinement for ordinal sequences  
3. **Dual-Branch Ordinal Integration**: Probability-level blending of IRT and CORAL approaches
4. **Parameter-Level Fusion**: Threshold-level combination for unified ordinal computation

## Mathematical Validation

All models maintain:
- **Probability Normalization**: Σ_k P(Y_t = k) = 1
- **Monotonic Thresholds**: β_{t,k} < β_{t,k+1} (enforced via cumulative sum)
- **Bounded Parameters**: θ ∈ [-3,3], α ≥ 0.1
- **Gradient Stability**: BGT framework for numerical stability

## Performance Benchmarks

Based on synthetic_OC dataset (30 epochs, 5-fold CV):
- **DeepGPCM**: QWK ~0.67 (baseline reference)
- **EnhancedAttentionGPCM**: QWK ~0.69 (+3% improvement)
- **CORALGPCM**: QWK ~0.69 (probability blending)
- **CORALGPCMFixed**: QWK ~0.69 (parameter fusion)

Performance may vary by dataset characteristics and ordinal structure complexity.