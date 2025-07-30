# AKVMN Model Comparison Analysis

## Overview
Comparing the performance drop between the old AKVMN implementation (`deep-gpcm-old/models/akvmn_gpcm.py`) and the new AttentionGPCM (`deep-gpcm/core/model.py`).

## Key Architectural Differences

### 1. Model Naming and Identity
- **Old**: `AKVMNGPCM` with model_name = "akvmn_gpcm"
- **New**: `AttentionGPCM` with model_name = "attention_dkvmn_gpcm"
- This might affect model loading/evaluation if names don't match

### 2. Parameter Initialization

#### Old AKVMN:
```python
# Ability scaling parameter
self.ability_scale = nn.Parameter(torch.tensor(2.0))  # Learnable parameter

# Embedding initialization
if embedding_strategy == "linear_decay":
    self.decay_weights = nn.Parameter(torch.ones(n_cats))  # Additional learnable weights
```

#### New AttentionGPCM:
```python
# Ability scale passed to IRTParameterExtractor
ability_scale: float = 1.0  # Fixed value, not learnable

# No decay_weights parameter for linear_decay strategy
```

### 3. Embedding Strategy Implementation

#### Old AKVMN (linear_decay):
```python
# Has learnable decay weights
self.decay_weights = nn.Parameter(torch.ones(n_cats))
self.gpcm_embed = nn.Linear(n_cats, embed_dim)

# In forward:
r_onehot = F.one_hot(responses, num_classes=self.n_cats).float()
decay_weights = F.softmax(self.decay_weights, dim=0)  # Learnable weights
gpcm_embed = self.gpcm_embed(r_onehot * decay_weights)
```

#### New AttentionGPCM:
- Inherits from DeepGPCM which uses the standard embedding strategies
- No learnable decay_weights parameter
- Linear decay is computed with fixed triangular weights

### 4. Architecture Components

#### Old AKVMN:
- Direct implementation with all components in one class
- Explicit GPCM probability computation method
- Clear separation of IRT parameter networks

#### New AttentionGPCM:
- Inherits from DeepGPCM base class
- Adds AttentionRefinementModule on top
- Projects embeddings to fixed dimension before refinement
- Uses modular design with separate layers

### 5. Key Missing Elements in New Implementation

1. **Learnable ability_scale**: Old model had this as a learnable parameter, new has fixed value
2. **Learnable decay_weights**: For linear_decay strategy, old had learnable weights
3. **Default Parameters**: Different defaults might affect performance:
   - Old: `embed_dim=64` (dedicated parameter)
   - New: Uses embedding strategy output_dim, then projects to `embed_dim`

### 6. Processing Flow Differences

#### Old AKVMN:
```python
# Direct embedding creation with learnable components
gpcm_embed = self._create_gpcm_embedding(questions, responses)
# Then iterative refinement
refined = self._iterative_refinement(features, question_embeds)
```

#### New AttentionGPCM:
```python
# Base embedding → Projection → Refinement
base_embeds = super().create_embeddings(questions, responses)
projected_embeds = self.embedding_projection(base_embeds)
refined_embeds = self.attention_refinement(projected_embeds)
```

## Potential Causes of Performance Drop

1. **Fixed vs Learnable Parameters**:
   - `ability_scale` is no longer learnable (2.0 → 1.0 fixed)
   - `decay_weights` for linear_decay no longer learnable

2. **Different Initialization**:
   - Old model had careful initialization for all components
   - New model might have different initialization patterns

3. **Embedding Dimension Handling**:
   - Old: Direct control over embed_dim
   - New: Projects from strategy output_dim to embed_dim (extra transformation)

4. **Model Name Mismatch**:
   - If evaluation expects "akvmn" but model reports "attention_dkvmn_gpcm"

## Recommendations for Fix

1. **Make ability_scale learnable again**:
   - Add as parameter in AttentionGPCM
   - Pass to IRTParameterExtractor differently

2. **Add learnable decay_weights for linear_decay**:
   - Override create_embeddings in AttentionGPCM
   - Add learnable weights like old model

3. **Verify initialization**:
   - Ensure same initialization schemes
   - Check parameter ranges

4. **Test with same defaults**:
   - Use ability_scale=2.0
   - Verify embed_dim handling

5. **Fix model naming**:
   - Ensure consistent naming for evaluation