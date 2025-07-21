# Deep-GPCM Implementation Plan

## Phase 1: Core GPCM Components (High Priority)

### 1.1 GPCM Embedding Strategies ⏳ In Progress
**File**: `models/gpcm_embeddings.py`

#### Strategy 1: Ordered Embedding (R^(2Q)) - RECOMMENDED
```python
def ordered_embedding(self, q_data, r_data, n_categories):
    """Most intuitive for partial credit"""
    low_component = ((n_categories - 1 - r_data) / (n_categories - 1)).unsqueeze(-1) * q_data
    high_component = (r_data / (n_categories - 1)).unsqueeze(-1) * q_data
    return torch.cat([low_component, high_component], dim=-1)
```

#### Strategy 2: Unordered Embedding (R^(KQ))
For MCQ-style categorical responses

#### Strategy 3: Linear Decay Embedding (R^(KQ))
Triangular weights around actual response

#### Strategy 4: Adjacent Weighted Embedding (R^(KQ))
Focus on actual + adjacent categories

### 1.2 GPCM Predictor ⭐ Next Priority
**File**: `models/gpcm_predictor.py`

- IRT parameter generation (θ, α, β thresholds)
- GPCM probability calculation
- Integration with DKVMN memory

### 1.3 Ordinal Loss Function ⭐ Next Priority
**File**: `models/losses.py`

```python
def ordinal_loss(predictions, targets, n_categories):
    """
    L = -Σ Σ Σ [I(y≤k)log(P(Y≤k)) + I(y>k)log(1-P(Y≤k))]
    """
```

### 1.4 Data Loader Extension 🔄 Medium Priority
**File**: `data/dataloader.py`

- Auto-detect K categories from responses
- Handle both PC and OC formats
- Metadata extraction and validation

## Phase 2: Training & Evaluation (Medium Priority)

### 2.1 Training Pipeline 🔄 Medium Priority
**File**: `train.py`

- GPCM-aware training loop
- Multi-metric tracking
- Model checkpointing

### 2.2 Evaluation Metrics 🔄 Medium Priority
**File**: `evaluate.py`

- Categorical accuracy (exact match)
- Ordinal accuracy (within 1 category) 
- Mean Absolute Error (MAE)
- Quadratic Weighted Kappa
- Per-category F1 scores

### 2.3 Model Configuration 🔄 Medium Priority
**File**: `utils/config.py`

- GPCM-specific configurations
- Embedding strategy selection
- Loss function options

## Phase 3: Advanced Features (Low Priority)

### 3.1 Visualization 📊 Future
- Polytomous response heatmaps
- Category probability evolution
- IRT parameter distributions

### 3.2 Model Comparison 📈 Future
- Baseline comparisons
- Binary vs GPCM performance
- Embedding strategy evaluation

### 3.3 Documentation 📝 Future
- API documentation
- Tutorial notebooks
- Usage examples

## Implementation Status

- [x] ✅ Project setup and structure
- [x] ✅ Synthetic data generation (OC & PC formats)
- [x] ✅ Data migration to deep-gpcm
- [ ] 🔄 GPCM embedding layer (Phase 1.1) 
- [ ] ⭐ GPCM predictor (Phase 1.2)
- [ ] ⭐ Ordinal loss function (Phase 1.3)
- [ ] 🔄 Data loader extension (Phase 1.4)
- [ ] 🔄 Training pipeline (Phase 2.1)
- [ ] 🔄 Evaluation metrics (Phase 2.2)

## Next Immediate Steps

1. **Implement ordered embedding strategy** (most intuitive)
2. **Create GPCM predictor with probability calculation**
3. **Implement ordinal loss function**
4. **Test with synthetic_OC dataset**

## Data Available

- ✅ `data/synthetic_OC/`: 4-category ordered responses {0,1,2,3}
- ✅ `data/synthetic_PC/`: Partial credit scores {0.000,0.333,0.667,1.000}
- ✅ True IRT parameters for validation

## Target Metrics

- **Categorical Accuracy**: Exact category prediction
- **Ordinal Accuracy**: Within ±1 category
- **QWK**: Quadratic Weighted Kappa for ordinal data

Ready for Phase 1 implementation!