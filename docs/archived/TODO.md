# Deep-GPCM Performance Enhancement TODO

## Executive Summary

Current Deep-GPCM shows moderate performance (QWK: 0.696, Ordinal Acc: 86.1%) with poor parameter recovery (θ correlation: -0.03). This document outlines research directions for improving ordinal accuracy and QWK through Bayesian approaches and state-of-the-art ordinal regression methods.

## Core Issues Identified

1. **Parameter Recovery Problem**: Temporal model parameters don't correlate with static IRT parameters
2. **Shallow Integration**: Attention mechanism operates independently of memory dynamics
3. **Missing Ordinal Constraints**: Loss functions don't explicitly enforce ordinal relationships
4. **Uncertainty Quantification**: No confidence measures for predictions

## Research Directions

### 1. Bayesian Enhancement (Priority: HIGH)

#### 1.1 Variational Ability Estimation
- [ ] Replace point estimates with distributions: θ ~ N(μ_θ, σ_θ²)
- [ ] Implement reparameterization trick for differentiable sampling
- [ ] Add KL regularization term to loss function
- [ ] Test β-VAE approach with annealing schedule

**Implementation Steps:**
```python
# Modify DeepGPCM to include variational components
class VariationalDeepGPCM(nn.Module):
    def __init__(self, ...):
        self.ability_mu = nn.Linear(memory_dim, 1)
        self.ability_log_var = nn.Linear(memory_dim, 1)
```

#### 1.2 Temporal Consistency Regularization
- [ ] Add smoothness constraint for ability evolution
- [ ] Implement Markovian prior: θ_t+1 ~ N(θ_t, σ_evolution²)
- [ ] Test different evolution noise levels
- [ ] Validate on synthetic data with known trajectories

### 2. SOTA Ordinal Regression Integration (Priority: HIGH)

#### 2.1 CORAL Layer Implementation
- [ ] Replace final GPCM layer with CORAL framework
- [ ] Implement rank-consistent logits transformation
- [ ] Add binary classifier ensemble for ordinal prediction
- [ ] Compare with current GPCM performance

**Key Features:**
- Guaranteed rank monotonicity
- Consistent confidence scores
- Architecture-agnostic design

#### 2.2 CORN Framework Testing
- [ ] Implement conditional ordinal regression
- [ ] Use chain rule for probability computation
- [ ] Test weight-sharing vs independent parameters
- [ ] Benchmark against CORAL

### 3. Advanced Loss Functions (Priority: MEDIUM)

#### 3.1 Ordinal-Aware Losses
- [ ] Implement Earth Mover's Distance (EMD) loss
- [ ] Add ordinal margin penalties
- [ ] Test weighted ordinal cross-entropy
- [ ] Implement QWK as differentiable loss

**Proposed Loss:**
```python
def ordinal_qwk_loss(pred_probs, true_labels, n_categories=4):
    # Convert to confusion matrix
    # Compute QWK
    # Return -QWK as loss (to maximize)
```

#### 3.2 Uncertainty-Weighted Loss
- [ ] Weight losses by prediction uncertainty
- [ ] Add confidence calibration term
- [ ] Implement focal loss variant for ordinal data
- [ ] Test on imbalanced datasets

### 4. Architectural Improvements (Priority: MEDIUM)

#### 4.1 Co-evolutionary Attention-Memory
- [ ] Integrate attention within memory update loop
- [ ] Add bidirectional information flow
- [ ] Implement gated fusion mechanism
- [ ] Test computational efficiency

#### 4.2 Hierarchical Processing
- [ ] Add multi-scale temporal attention
- [ ] Implement progressive refinement
- [ ] Test different aggregation strategies
- [ ] Validate on long sequences

### 5. Numerical Stability (Priority: LOW)

#### 5.1 Stable GPCM Computation
- [ ] Implement log-space operations
- [ ] Use logsumexp for normalization
- [ ] Add gradient clipping
- [ ] Test on edge cases

### 6. Evaluation Framework Enhancement

#### 6.1 Comprehensive Metrics
- [ ] Add calibration metrics (ECE, MCE)
- [ ] Implement ordinal correlation measures
- [ ] Track uncertainty quality metrics
- [ ] Create visualization dashboard

#### 6.2 Ablation Studies
- [ ] Test each component independently
- [ ] Measure contribution to QWK improvement
- [ ] Analyze failure cases
- [ ] Document best practices

## Implementation Roadmap

### Phase 1: Quick Wins (1-2 weeks)
1. Add CORAL layer to existing models
2. Implement ordinal-aware loss functions
3. Add numerical stability improvements
4. Benchmark on current datasets

### Phase 2: Bayesian Enhancement (2-3 weeks)
1. Implement variational ability estimation
2. Add temporal consistency regularization
3. Test uncertainty quantification
4. Validate parameter recovery improvement

### Phase 3: Architecture Refinement (3-4 weeks)
1. Develop co-evolutionary attention-memory
2. Add hierarchical processing
3. Optimize computational efficiency
4. Conduct comprehensive ablations

### Phase 4: Integration & Validation (1-2 weeks)
1. Combine best-performing components
2. Fine-tune hyperparameters
3. Run full benchmark suite
4. Document findings and best practices

## Expected Outcomes

### Performance Targets
- QWK: 0.75+ (from 0.696)
- Ordinal Accuracy: 90%+ (from 86.1%)
- Parameter Recovery: θ correlation > 0.5 (from -0.03)

### Research Contributions
1. Novel Bayesian DKVMN for ordinal knowledge tracing
2. CORAL/CORN integration with memory networks
3. Uncertainty-aware ordinal prediction framework
4. Comprehensive benchmark on educational datasets

## Technical Considerations

### Without True IRT Parameters
Since real-world datasets lack true IRT parameters:
1. Use self-supervised objectives (temporal consistency)
2. Leverage ordinal structure as supervision signal
3. Implement contrastive learning between similar abilities
4. Use curriculum learning with difficulty progression

### Computational Efficiency
- Maintain current inference speed
- Use efficient attention mechanisms
- Implement gradient checkpointing if needed
- Profile and optimize bottlenecks

## Collaboration Points

### With Systems Architect
- [ ] Design efficient CORAL integration
- [ ] Optimize memory usage for Bayesian components
- [ ] Implement distributed training if needed
- [ ] Create modular architecture for easy ablation

### With Domain Experts
- [ ] Validate educational relevance of improvements
- [ ] Test on diverse ordinal datasets
- [ ] Ensure interpretability of parameters
- [ ] Document practical deployment guidelines

## References

1. CORAL: https://github.com/Raschka-research-group/coral-pytorch
2. dlordinal: https://github.com/ayrna/dlordinal
3. Variational IRT: Local vtirt implementation
4. Bayesian Deep Learning: VIBO approaches

## Next Steps

1. **Immediate**: Implement CORAL layer and test
2. **This Week**: Add variational ability estimation
3. **Next Week**: Develop ordinal-aware losses
4. **Following**: Architecture improvements and ablations

---
*Last Updated: [Current Date]*
*Status: Research Planning Phase*