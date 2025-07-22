# Deep-GPCM TODO: Critical Performance Improvements

## Current Performance Issues Analysis

**Performance Results from Latest Experiments:**
- **Categorical Accuracy**: ~50% (essentially random performance)
- **Ordinal Accuracy**: ~77% (reasonable but could be much better)  
- **Prediction Consistency**: ~37% (**CRITICAL ISSUE** - severe train/inference mismatch)
- **Ordinal Ranking**: ~63% (decent but suboptimal)
- **MAE**: ~0.83 (acceptable for 4-category problem)

**Root Cause Analysis:**
1. **Training/Inference Mismatch**: Model trained with `OrdinalLoss` but predictions made with `argmax` instead of cumulative methods
2. **Lack of Rank Consistency**: No rank-monotonicity guarantees in current architecture
3. **Suboptimal Embedding Strategies**: Current embeddings don't preserve ordinal structure effectively
4. **Missing SOTA Techniques**: Not leveraging cutting-edge ordinal classification methods

## Priority 1: CRITICAL FIXES (Immediate)

### 1.1 Fix Training/Inference Alignment ⚠️ **URGENT**
**Problem**: 37% prediction consistency indicates severe mismatch between training objective and prediction method.

**Actions:**
- [ ] **Change default prediction from `argmax` to `cumulative`** in evaluation pipeline
- [ ] Implement proper cumulative probability thresholding (P(Y≤k) > 0.5)
- [ ] Add calibrated cumulative prediction method that aligns with `OrdinalLoss`
- [ ] Run A/B test: argmax vs cumulative vs expected value predictions
- [ ] **Target**: Improve prediction consistency from 37% to >70%

### 1.2 Implement CORAL Framework ⚠️ **HIGH PRIORITY**
**Problem**: No rank consistency guarantees leading to ordinal violations in predictions.

**Actions:**
- [ ] Implement CORAL (COnsistent RAnk Logits) output layer
- [ ] Replace current softmax with CORAL's cumulative link approach
- [ ] Add CORAL loss function: `L = -Σ log P(Y>τk | x)^{I(y>k)} * P(Y≤τk | x)^{I(y≤k)}`
- [ ] Ensure rank-monotonic predictions: P(Y≤1|x) ≤ P(Y≤2|x) ≤ ... ≤ P(Y≤K|x)
- [ ] **Target**: Achieve mathematically guaranteed rank consistency

### 1.3 Advanced Ordinal Embeddings ⚠️ **HIGH PRIORITY**
**Problem**: Current embeddings don't capture ordinal structure effectively.

**Actions:**
- [ ] Implement **Ordinal Binary Decomposition (OBD)** embeddings
- [ ] Add **distance-aware** embeddings: `embed(k) ∝ exp(-α|k-target|)`
- [ ] Create **hierarchical ordinal** embeddings with category relationships
- [ ] Implement **learnable threshold** embeddings for GPCM thresholds
- [ ] **Target**: Improve categorical accuracy from 50% to >70%

## Priority 2: SOTA Integration (Next Phase)

### 2.1 CORN (Conditional Ordinal Regression) Implementation
**Theory**: Achieves rank consistency through conditional probability training.

**Actions:**
- [ ] Implement CORN training scheme with conditional datasets
- [ ] Add unconditional probability recovery through chain rule
- [ ] Compare CORN vs CORAL performance on ordinal metrics
- [ ] **Target**: Best-in-class ordinal ranking performance (>80%)

### 2.2 Deep-IRT Integration for Educational Relevance
**Theory**: Combine deep learning with psychometric interpretability.

**Actions:**
- [ ] Extract interpretable IRT parameters (θ, α, β) from DKVMN states
- [ ] Implement IRT-guided loss function with ability/difficulty constraints
- [ ] Add temporal IRT parameter tracking for knowledge evolution
- [ ] **Target**: Educationally interpretable model with competitive performance

### 2.3 Probability Calibration and Uncertainty Quantification
**Problem**: Poor probability calibration affects cumulative prediction methods.

**Actions:**
- [ ] Implement Platt scaling for probability calibration
- [ ] Add Monte Carlo dropout for uncertainty estimation
- [ ] Implement temperature scaling for ordinal probabilities
- [ ] Add calibration metrics (ECE, MCE) to evaluation pipeline
- [ ] **Target**: Well-calibrated probabilities for reliable predictions

## Priority 3: Architecture Enhancements

### 3.1 Ordinal-Aware Memory Architecture
**Actions:**
- [ ] Modify DKVMN memory to store ordinal relationships
- [ ] Implement ordinal attention mechanisms
- [ ] Add memory-based ordinal constraint enforcement
- [ ] **Target**: Memory that preserves and leverages ordinal structure

### 3.2 Multi-Scale Ordinal Features
**Actions:**
- [ ] Implement coarse-to-fine ordinal prediction (binary → tertiary → full)
- [ ] Add ordinal feature fusion from multiple scales
- [ ] Create ordinal ensemble methods with different granularities
- [ ] **Target**: Robust predictions across ordinal scales

## Priority 4: Comprehensive Evaluation Framework

### 4.1 Enhanced Metrics Suite
**Actions:**
- [ ] Add **Ordinal Classification Index (OCI)** - comprehensive ordinal metric
- [ ] Implement **Rank Consistency Rate** - percentage of rank-monotonic predictions
- [ ] Add **Ordinal Confidence Calibration** metrics
- [ ] Create **Educational Impact Score** combining accuracy + interpretability
- [ ] **Target**: Comprehensive evaluation matching educational needs

### 4.2 Benchmark Against SOTA Methods
**Actions:**
- [ ] Compare against traditional ordinal regression baselines
- [ ] Benchmark against recent CORAL/CORN implementations
- [ ] Evaluate against classical IRT models (2PL, GPCM in R)
- [ ] **Target**: Demonstrate superior performance vs established methods

## Implementation Priorities

**Week 1-2: Critical Fixes**
1. Fix prediction method alignment (cumulative vs argmax)
2. Implement basic CORAL framework
3. Quick ordinal embedding improvements

**Week 3-4: SOTA Integration**
1. Full CORAL/CORN implementation
2. Advanced ordinal embeddings
3. Probability calibration

**Week 5-6: Validation & Optimization**
1. Comprehensive benchmarking
2. Hyperparameter optimization
3. Educational validation studies

## Success Criteria

**Minimum Acceptable Performance:**
- Categorical Accuracy: >70% (up from ~50%)
- Prediction Consistency: >70% (up from ~37%)
- Ordinal Ranking: >80% (up from ~63%)
- Rank Consistency Rate: >95% (new metric)

**Excellence Targets:**
- Categorical Accuracy: >80%
- Prediction Consistency: >85%
- Ordinal Ranking: >90%
- Educational Interpretability: Validated by domain experts

## Research Questions to Address

1. **Which ordinal technique is most effective for IRT knowledge tracing?** (CORAL vs CORN vs OBD)
2. **How does prediction method affect educational decision-making?** (argmax vs cumulative impact)
3. **What is the optimal balance between accuracy and interpretability?** (Deep-IRT trade-offs)
4. **How do ordinal constraints improve knowledge tracing over time?** (temporal consistency analysis)

## Long-term Vision

Transform Deep-GPCM from a research prototype with fundamental issues into a production-ready, educationally-valid ordinal knowledge tracing system that:
- Achieves SOTA performance on ordinal educational prediction
- Provides interpretable IRT-compatible outputs
- Demonstrates clear educational impact through proper ordinal handling
- Serves as reference implementation for ordinal educational AI

---

**Note**: The current ~50% categorical accuracy and ~37% prediction consistency represent fundamental algorithmic issues that must be resolved before any other improvements. This TODO prioritizes critical fixes that will provide immediate, substantial performance gains.