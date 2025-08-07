# Deep-GPCM TODO: Critical Issues and Implementation Roadmap

## CRITICAL PRIORITY: CORAL Design Flaw Fix

### 🚨 BLOCKING ISSUE: Incorrect Parameter Usage

**Status**: CRITICAL - Invalidates all current CORAL research

**Problem**: CORAL uses β (beta) parameters instead of τ (tau) thresholds, making CORAL and GPCM computations identical.

**Evidence**:
- CORAL parameters (τ): `[0., 0., 0.]` (all zeros - unused)
- GPCM parameters (β): `[-0.8234, -0.0156, 0.7453]` (actively learned)
- Both systems use identical β parameters for computation

**Required Actions**:

#### Task 1: Fix CORAL Parameter Extraction (Priority: CRITICAL)
- 📋 **Implement τ extraction from CORAL logits**: Convert `coral_logits` to usable threshold parameters
- 📋 **Update CORAL computation pathway**: Use τ parameters instead of β in `coral_probs` calculation
- ✅ **Preserve GPCM computation**: Keep existing IRT extractor for GPCM baseline
- 📋 **Files to modify**:
  - `models/model.py` (Lines ~180-200): Core CORAL computation logic
  - `models/coral_layer.py`: CORAL layer parameter usage
  - `analysis/extract_beta_params.py`: Extract both β and τ parameters

#### Task 2: Verification and Validation (Priority: CRITICAL)
- 📋 **Parameter usage audit**: Confirm τ parameters are extracted and used for CORAL
- 📋 **Computational verification**: Validate CORAL and GPCM produce different results
- 📋 **Training validation**: Ensure adaptive blending provides genuine benefit
- 📋 **Re-run all benchmarks**: Invalidate current results and generate corrected comparisons

#### Task 3: Architecture Testing (Priority: HIGH)
- [ ] **τ parameter learning**: Test that ordinal thresholds are properly optimized
- [ ] **Threshold coupling validation**: Confirm coupling/decoupling mechanisms work with correct parameters
- [ ] **Gradient flow analysis**: Verify gradient flow through corrected CORAL pathway

## HIGH PRIORITY: Adaptive Blending System Completion

### Task 4: BGT Framework Integration (Priority: HIGH)

**Status**: ✅ BGT mathematics proven stable, 🚧 forward pass integration needs debugging

**Completed**:
- ✅ BGT mathematical framework developed and validated
- ✅ `StableThresholdDistanceBlender` with comprehensive testing (4/4 tests pass)
- ✅ Gradient explosion prevention (reduced >20,000 to <10 gradient norms)
- ✅ Component-level validation (all BGT transforms stable)

**Outstanding Issues**:
- 🚧 **Resolve model-level integration**: BGT components work but full model training unstable
- 🚧 **Debug gradient coupling**: Investigate memory network interaction with adaptive blending
- 📋 **Implement gradient isolation**: Apply `item_betas.detach()` solution for memory decoupling
- 📋 **Validate training stability**: Ensure consistent training with adaptive blending enabled

**Action Items**:
- 🚧 **Complete MinimalAdaptiveBlender integration**: Deploy gradient isolation solution
- ❌ **Test FullAdaptiveBlender training**: BLOCKED by CORAL design flaw
- ❌ **Production deployment**: BLOCKED until CORAL fix implemented

### Task 5: Multi-Dataset Validation (Priority: MEDIUM)

**Requirements**:
- [ ] **Test across datasets**: Validate adaptive blending on multiple educational datasets
- [ ] **Performance consistency**: Ensure improvements generalize beyond synthetic data
- [ ] **Computational overhead**: Verify <15% training overhead target met
- [ ] **Memory efficiency**: Confirm reasonable GPU memory usage

## MEDIUM PRIORITY: System Enhancements

### Task 6: IRT Analysis Integration (Priority: MEDIUM)

**Status**: System functional but needs CORAL parameter fix integration

**Action Items**:
- [ ] **Update parameter extraction**: Integrate corrected β/τ parameter extraction
- [ ] **Enhanced temporal analysis**: Improve temporal parameter visualization
- [ ] **Parameter recovery metrics**: Develop better correlation measures for temporal parameters
- [ ] **Comparative analysis**: Enable side-by-side CORAL vs GPCM parameter comparison

### Task 7: Loss Function Optimization (Priority: MEDIUM)

**Current Status**: QWK and combined losses implemented

**Enhancement Opportunities**:
- [ ] **Ordinal loss tuning**: Optimize QWK loss hyperparameters for better convergence
- [ ] **Combined loss balancing**: Find optimal weights for multi-objective optimization
- [ ] **EMD loss integration**: Complete Earth Mover's Distance loss implementation
- [ ] **Loss scheduling**: Implement dynamic loss weight scheduling during training

### Task 8: Architecture Improvements (Priority: MEDIUM)

**Attention Mechanisms**:
- [ ] **Attention-CORAL integration**: Combine attention refinement with CORAL ordinal structure
- [ ] **Memory capacity scaling**: Optimize memory size for different dataset scales
- [ ] **Embedding strategy optimization**: Compare and optimize response embedding approaches

**Model Efficiency**:
- [ ] **Parameter reduction**: Investigate model compression techniques
- [ ] **Training acceleration**: Implement gradient checkpointing and mixed precision
- [ ] **Inference optimization**: Optimize models for deployment scenarios

## LOWER PRIORITY: Research and Validation

### Task 9: Temporal Parameter Research (Priority: LOW)

**Research Questions**:
- [ ] **Temporal vs static IRT**: Develop theoretical framework for temporal parameter interpretation
- [ ] **Learning trajectory modeling**: Implement explicit learning curve modeling
- [ ] **Parameter evolution patterns**: Analyze common temporal parameter evolution patterns
- [ ] **Curriculum learning integration**: Use temporal patterns for curriculum design

### Task 10: Bayesian Extensions (Priority: LOW)

**Proposed Enhancements**:
- [ ] **Variational parameter estimation**: Implement Bayesian parameter uncertainty
- [ ] **Hierarchical modeling**: Add student and item population modeling
- [ ] **Uncertainty quantification**: Provide confidence intervals for predictions
- [ ] **Active learning integration**: Use uncertainty for intelligent question selection

### Task 11: Visualization and Interpretability (Priority: LOW)

**Dashboard Development**:
- [ ] **Interactive parameter visualization**: Real-time parameter evolution display
- [ ] **Student trajectory analysis**: Individual learning journey visualization
- [ ] **Item analysis tools**: Item characteristic curve interactive plots
- [ ] **Model comparison interface**: Side-by-side model performance comparison

## Implementation Timeline

### Phase 1: Critical Fixes (Week 1-2)
**Priority**: CRITICAL
- [ ] Complete CORAL design flaw fix
- [ ] Validate corrected parameter usage
- [ ] Re-run benchmark comparisons
- [ ] Update all documentation

### Phase 2: Adaptive Blending (Week 3-4)  
**Priority**: HIGH
- [ ] Finalize BGT integration debugging
- [ ] Deploy stable adaptive blending
- [ ] Multi-dataset validation
- [ ] Performance optimization

### Phase 3: System Enhancement (Week 5-8)
**Priority**: MEDIUM
- [ ] IRT analysis improvements
- [ ] Loss function optimization
- [ ] Architecture enhancements
- [ ] Production hardening

### Phase 4: Research Extensions (Month 3+)
**Priority**: LOW
- [ ] Temporal parameter research
- [ ] Bayesian extensions
- [ ] Advanced visualization
- [ ] Publication preparation

## Success Criteria

### Phase 1 Success Metrics
- 📋 **CORAL parameters**: τ ≠ β demonstrated with different learned values
- 📋 **Performance differentiation**: CORAL vs GPCM show measurable differences
- 📋 **Benchmark validity**: All results regenerated with corrected implementation
- 🚧 **Documentation complete**: Partially updated with status indicators

### Phase 2 Success Metrics
- [ ] **Training stability**: Adaptive models train without gradient explosion
- [ ] **Performance improvement**: >10% QWK improvement with full adaptive blending
- [ ] **Resource efficiency**: <15% computational overhead
- [ ] **Production readiness**: Models deployable in main pipeline

### Overall Project Success
- [ ] **Research validity**: All CORAL claims backed by corrected implementation
- [ ] **Practical impact**: Demonstrable improvements in educational assessment
- [ ] **Scientific contribution**: Novel adaptive blending and temporal IRT insights
- [ ] **Community adoption**: System used by educational data mining researchers

## Risk Mitigation

### Technical Risks
- **CORAL fix complexity**: May require extensive architecture changes
  - *Mitigation*: Incremental implementation with rollback capability
- **BGT integration stability**: Complex mathematical transforms may be fragile
  - *Mitigation*: Comprehensive testing and gradual integration
- **Performance regression**: Fixes may reduce current performance
  - *Mitigation*: A/B testing and performance monitoring

### Project Risks
- **Timeline pressure**: Critical fixes may delay other enhancements
  - *Mitigation*: Prioritize critical issues and defer non-essential features
- **Research validity**: Corrected results may not support original claims
  - *Mitigation*: Prepare for honest reporting of corrected findings
- **Resource constraints**: Multiple complex tasks competing for attention  
  - *Mitigation*: Clear prioritization and incremental delivery

---

**Next Action**: Begin Task 1 (CORAL parameter extraction fix) immediately - this blocks all other CORAL-related work and invalidates current research claims.