# Deep-GPCM Mathematical Foundations

## Overview

This document provides the complete mathematical formulation for the Deep-GPCM system, including DKVMN memory networks, IRT parameter extraction, GPCM probability computation, CORAL ordinal regression, and adaptive blending mechanisms.

## Core Architecture Mathematical Flow

```
Input → Embeddings → DKVMN → IRT Parameters → GPCM/CORAL → Adaptive Blending → Output
```

## 1. DKVMN Memory Network Mathematics

### 1.1 Memory Architecture

**Memory Matrices:**
- **Key Memory**: `M_k ∈ ℝ^(N×d_k)` - Static concept representations
- **Value Memory**: `M_v ∈ ℝ^(N×d_v)` - Dynamic mastery states

Where:
- `N`: Memory size (number of concepts)
- `d_k`: Key dimension 
- `d_v`: Value dimension

### 1.2 Attention Mechanism

**Attention Weights:**
```
w_t^(i) = softmax(q_t^T M_k^(i))
```

**Read Operation:**
```
r_t = Σ_{i=1}^N w_t^(i) M_v^(i)
```

Where:
- `q_t ∈ ℝ^{d_k}`: Question embedding at time t
- `r_t ∈ ℝ^{d_v}`: Read content vector

### 1.3 Memory Update

**Erase Operation:**
```
M_v^(i)_{new} = M_v^(i)_{old} ⊙ (1 - w_t^(i) e_t^T)
```

**Add Operation:**
```
M_v^(i)_{final} = M_v^(i)_{new} + w_t^(i) a_t^T
```

Where:
- `e_t ∈ ℝ^{d_v}`: Erase vector (what to forget)
- `a_t ∈ ℝ^{d_v}`: Add vector (what to remember)
- `⊙`: Element-wise multiplication

## 2. IRT Parameter Extraction

### 2.1 Summary Vector Creation

**Memory-Question Integration:**
```
s_t = tanh(W_s [r_t; q_t] + b_s)
```

Where:
- `s_t ∈ ℝ^{d_s}`: Summary vector
- `W_s ∈ ℝ^{d_s×(d_v+d_k)}`: Summary weight matrix
- `[r_t; q_t]`: Concatenation of read content and question embedding

### 2.2 IRT Parameter Computation

**Student Ability (θ):**
```
θ_t = tanh(W_θ s_t + b_θ)
```

**Item Discrimination (α):**
```
α_t = softplus(W_α [s_t; q_t] + b_α) + ε_α
```

**Item Thresholds (β):**
```
β_{t,k} = Σ_{j=0}^k softplus(W_β^{(j)} [s_t; q_t] + b_β^{(j)})
```

Where:
- `θ_t ∈ ℝ`: Student ability at time t
- `α_t ∈ ℝ`: Item discrimination at time t
- `β_{t,k} ∈ ℝ`: Cumulative threshold for category k
- `ε_α = 0.1`: Minimum discrimination to ensure positivity
- `softplus(x) = log(1 + exp(x))`: Ensures monotonic thresholds

## 3. GPCM Probability Computation

### 3.1 Generalized Partial Credit Model

**Cumulative Logits:**
```
Z_{t,k} = Σ_{j=0}^k α_t(θ_t - β_{t,j})
```

**Category Probabilities:**
```
P(Y_t = k | θ_t, α_t, β_t) = exp(Z_{t,k}) / Σ_{c=0}^{K-1} exp(Z_{t,c})
```

Where:
- `Y_t`: Response at time t
- `K`: Number of categories (4 for Deep-GPCM)
- `Z_{t,0} = 0` by convention

### 3.2 Mathematical Properties

**Monotonicity Constraint:**
```
β_{t,0} < β_{t,1} < β_{t,2} < ... < β_{t,K-2}
```

**Probability Normalization:**
```
Σ_{k=0}^{K-1} P(Y_t = k | θ_t, α_t, β_t) = 1
```

## 4. CORAL Ordinal Regression

### 4.1 CORAL Framework

**Ordinal Thresholds (τ):**
```
τ = [τ_0, τ_1, τ_2, ..., τ_{K-2}]
```

**Cumulative Logits:**
```
f_k = h(x) - τ_k
```

Where:
- `h(x)`: Neural network output (scalar)
- `τ_k`: Learnable ordinal threshold

### 4.2 CORAL Probability Computation

**Binary Probabilities:**
```
P(Y > k) = σ(f_k) = σ(h(x) - τ_k)
```

**Category Probabilities:**
```
P(Y = 0) = 1 - P(Y > 0)
P(Y = k) = P(Y > k-1) - P(Y > k)  for k ∈ {1, ..., K-2}
P(Y = K-1) = P(Y > K-2)
```

Where `σ(x) = 1/(1 + exp(-x))` is the sigmoid function.

### 4.3 CRITICAL ISSUE: Current Implementation Flaw

**Intended CORAL Computation:**
```
P_CORAL(Y_t = k) = CORAL_structure(α_t(θ_t - τ_k))
```

**Actual (Incorrect) Implementation:**
```
P_CORAL(Y_t = k) = CORAL_structure(α_t(θ_t - β_{t,k}))
```

**Problem**: CORAL uses GPCM β parameters instead of dedicated τ parameters, making both systems identical.

## 5. Adaptive Blending Mathematics

### 5.1 Threshold Distance Analysis

**Semantic Threshold Alignment:**
```
d_{i,j} = |τ_i - β_{j,i}|  ∀i ∈ {0, 1, 2}
```

Where:
- `τ_i`: CORAL threshold for boundary i
- `β_{j,i}`: GPCM threshold for item j, boundary i

**Distance Statistics:**
```
d_min = min_i(d_{i,j})
d_avg = (1/K) Σ_i d_{i,j}
d_spread = std(d_{0,j}, d_{1,j}, d_{2,j})
```

### 5.2 Geometric Analysis

**Range Divergence:**
```
R_GPCM = max_i(β_{j,i}) - min_i(β_{j,i})
R_CORAL = max_i(τ_i) - min_i(τ_i)
ΔR = |R_CORAL - R_GPCM| / (R_CORAL + R_GPCM + ε)
```

**Threshold Correlation:**
```
ρ = (Σ_i β_{j,i} τ_i) / (||β_j|| ||τ|| + ε)
```

### 5.3 BGT (Bounded Geometric Transform) Framework

**Problem**: Original operations cause gradient explosion:
```
log(1 + x) → ∞  as x → ∞
x/(1 + x) → singularities near x = -1
```

**BGT Solutions:**
```
log_BGT(x) = 2 tanh(clamp(x, 0, 10) / 2)        ∈ [0, 2]
div_BGT(x) = σ(clamp(x, -10, 10))               ∈ [0, 1]
corr_BGT(x) = 0.5(1 + tanh(clamp(x, -3, 3)))   ∈ [0, 1]
```

### 5.4 Adaptive Weight Computation

**BGT-Stabilized Weight Formula:**
```
w_{j,k} = σ(clamp(
    λ_R · log_BGT(ΔR_j) + 
    λ_D · div_BGT(d_min,j) + 
    λ_C · corr_BGT(ρ_j) + 
    λ_S · exp(-clamp(d_spread,j, 0, 5)) + 
    b_0,
    -3, 3
))
```

Where:
- `λ_R ∈ [0.1, 2.0]`: Range sensitivity (learnable)
- `λ_D ∈ [0.5, 3.0]`: Distance sensitivity (learnable)
- `λ_C = 0.3`: Correlation weight (fixed)
- `λ_S = 0.1`: Spread penalty weight (fixed)
- `b_0 ∈ [-1.0, 1.0]`: Baseline bias (learnable)

### 5.5 Final Probability Blending

**Category-Specific Blending:**
```
P_final(Y_t = k) = (1 - w_{t,k}) P_GPCM(Y_t = k) + w_{t,k} P_CORAL(Y_t = k)
```

**Normalization:**
```
P_final(Y_t = k) ← P_final(Y_t = k) / Σ_{c=0}^{K-1} P_final(Y_t = c)
```

## 6. Loss Functions

### 6.1 Cross-Entropy Loss

**Standard Classification Loss:**
```
ℒ_CE = -Σ_t Σ_k y_{t,k} log(P_final(Y_t = k))
```

### 6.2 Quadratic Weighted Kappa Loss

**QWK Differentiable Approximation:**
```
W_{i,j} = (i - j)² / (K - 1)²
```

**Soft Confusion Matrix:**
```
C_{i,j} = Σ_t δ(y_t = i) P_final(Y_t = j)
```

**QWK Computation:**
```
p_o = Σ_{i,j} C_{i,j}(1 - W_{i,j}) / Σ_{i,j} C_{i,j}
p_e = Σ_{i,j} (Σ_k C_{i,k})(Σ_k C_{k,j})(1 - W_{i,j}) / (Σ_{i,j} C_{i,j})²
QWK = (p_o - p_e) / (1 - p_e + ε)
```

**QWK Loss:**
```
ℒ_QWK = -QWK
```

### 6.3 CORAL Ordinal Loss

**Cumulative Link Loss:**
```
ℒ_CORAL = -Σ_t Σ_{k=0}^{K-2} [y_t > k] log(σ(f_{t,k})) + [y_t ≤ k] log(1 - σ(f_{t,k}))
```

### 6.4 Combined Loss

**Weighted Combination:**
```
ℒ_total = λ_CE ℒ_CE + λ_QWK ℒ_QWK + λ_CORAL ℒ_CORAL
```

With typical weights: `λ_CE = 0.5`, `λ_QWK = 0.3`, `λ_CORAL = 0.5`

## 7. Gradient Flow Analysis

### 7.1 Critical Gradient Paths

**Memory Updates:**
```
∂ℒ/∂M_v^(i) = Σ_t w_t^(i) ∂ℒ/∂r_t
```

**IRT Parameters:**
```
∂ℒ/∂θ_t = α_t Σ_k (P_final(Y_t = k) - y_{t,k}) g_k
∂ℒ/∂α_t = Σ_k (P_final(Y_t = k) - y_{t,k}) (θ_t - β_{t,k}) g_k
∂ℒ/∂β_{t,k} = -α_t Σ_{j≥k} (P_final(Y_t = j) - y_{t,j}) g_j
```

Where `g_k` represents the GPCM gradient contribution for category k.

### 7.2 Gradient Isolation for Adaptive Blending

**Problem**: Gradient coupling between adaptive weights and memory updates:
```
∂ℒ/∂w_{t,k} ∝ ∂d_min/∂β_{t,k} → ∂ℒ/∂M_v
```

**Solution**: Strategic gradient detachment:
```
w_{t,k} = f(β_{t,k}.detach(), τ_k, θ_t, α_t)
```

This breaks the gradient flow while preserving adaptive behavior.

## 8. Numerical Stability Considerations

### 8.1 Stability Constraints

**Parameter Bounds:**
```
θ_t ∈ [-3, 3]      (tanh activation)
α_t ∈ [0.1, 3.0]   (softplus + ε)
β_{t,k} ∈ ℝ       (monotonic via cumsum)
τ_k ∈ ℝ           (learnable thresholds)
```

**Probability Clamping:**
```
P(Y_t = k) ∈ [ε, 1-ε]  where ε = 1e-7
```

### 8.2 BGT Stability Guarantees

**Bounded Outputs:**
```
log_BGT(x) ∈ [0, 2]
div_BGT(x) ∈ [0, 1]  
corr_BGT(x) ∈ [0, 1]
```

**Bounded Gradients:**
```
|∇ log_BGT(x)| ≤ 1
|∇ div_BGT(x)| ≤ 0.25
|∇ corr_BGT(x)| ≤ 0.5
```

## 9. Critical Mathematical Issues

### 9.1 ❌ CORAL Design Flaw (BLOCKING)

**Problem**: Current CORAL implementation uses β parameters instead of τ thresholds, making CORAL and GPCM computations mathematically identical.

**Current (INCORRECT) Implementation**:
```
P_CORAL(Y_t = k) = σ(α_t(θ_t - β_{t,k}))  // Uses β parameters
P_GPCM(Y_t = k) = σ(α_t(θ_t - β_{t,k}))   // Also uses β parameters
```

**Correct Mathematical Formulation**:
```
P_CORAL(Y_t = k) = σ(α_t(θ_t - τ_k))      // Should use τ thresholds  
P_GPCM(Y_t = k) = σ(α_t(θ_t - β_{t,k}))   // Correctly uses β parameters
```

**Evidence from Parameter Extraction**:
- CORAL τ parameters: `[0., 0., 0.]` (all zeros, unused)
- GPCM β parameters: `[-0.8234, -0.0156, 0.7453]` (actively learned)
- Both systems use identical β values in computation

**Mathematical Impact**:
```
P_CORAL ≡ P_GPCM  ⟹  Adaptive Blending = Identity Transform
w_{t,k} P_CORAL + (1-w_{t,k}) P_GPCM = P_GPCM  (no benefit)
```

**Status**: 📋 Fix required before any CORAL-related research is valid.

### 9.2 Validated Performance Metrics

**Current Verified Results** (from `/results/train/`):

| Model | Categorical Acc. | QWK | Ordinal Acc. | Status |
|-------|-----------------|-----|--------------|--------|
| **Deep-GPCM** | **53.5%** (±1.3%) | **0.643** (±0.016) | **83.1%** (±0.7%) | ✅ VALID |
| **Full Adaptive CORAL-GPCM** | **53.7%** (±1.1%) | **0.681** (±0.012) | **87.4%** (±0.4%) | ❌ INVALID |

**Critical Note**: CORAL results are invalid due to β/τ parameter confusion.

## 10. Model Complexity Analysis

### 10.1 Parameter Counts

**DeepGPCM:**
- Memory: `N × (d_k + d_v)` = 50 × (50 + 200) = 12,500
- Embeddings: `(Q + 1) × d_k` = 401 × 50 = 20,050
- IRT extraction: `~6,000` parameters
- GPCM value transform: `1,600 × 200` = 320,000
- **Total**: ~448,555 parameters

**CORAL Extension:**
- CORAL projection: `d_v × (K-1)` = 200 × 3 = 600
- Threshold parameters: `K-1` = 3
- **Additional**: ~603 parameters

**Adaptive Blending:**
- Range sensitivity: 1
- Distance sensitivity: 1  
- Baseline bias: 1
- **Additional**: 3 parameters

### 9.2 Computational Complexity

**Forward Pass:** `O(N × d_v + Q × d_k + T × K)`
**Memory Update:** `O(T × N × d_v)`
**IRT Extraction:** `O(T × d_s × K)`
**Adaptive Blending:** `O(T × K²)`

Where:
- `T`: Sequence length
- `N`: Memory size
- `Q`: Number of questions
- `K`: Number of categories

## 10. Mathematical Validation Properties

### 10.1 Probability Axioms

**Non-negativity:**
```
P(Y_t = k) ≥ 0  ∀k, t
```

**Normalization:**
```
Σ_{k=0}^{K-1} P(Y_t = k) = 1  ∀t
```

**Monotonicity (CORAL):**
```
P(Y > 0) ≥ P(Y > 1) ≥ ... ≥ P(Y > K-2) ≥ 0
```

### 10.2 Gradient Properties

**Gradient Bounds:**
```
||∇_θ ℒ|| ≤ C_θ
||∇_α ℒ|| ≤ C_α  
||∇_β ℒ|| ≤ C_β
```

With BGT framework ensuring `C_* < ∞`.

### 10.3 Convergence Properties

**Loss Monotonicity:**
```
ℒ(t+1) ≤ ℒ(t) + δ
```

Where `δ → 0` as training progresses.

---

## Critical Mathematical Issues

### Issue 1: CORAL Parameter Usage (CRITICAL)

**Current Implementation:**
```
P_CORAL(Y_t = k) = CORAL_structure(α_t(θ_t - β_{t,k}))  ❌
```

**Correct Implementation:**
```
P_CORAL(Y_t = k) = CORAL_structure(α_t(θ_t - τ_k))      ✅
```

### Issue 2: Temporal vs Static Parameters

**Neural Model (Temporal):**
```
θ_t, α_t, β_{t,k} = f(memory_state_t, question_t)
```

**Classical IRT (Static):**
```
θ_i, α_j, β_{j,k} = constants
```

This fundamental difference explains poor parameter recovery correlations.

---

**Mathematical Framework Status**: Complete foundation with identified critical issues requiring architectural fixes.