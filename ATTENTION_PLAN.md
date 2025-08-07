# Modular Attention Architecture for Ordinal Knowledge Tracing

## Executive Summary

Current AttentionGPCM uses generic self-attention refinement that treats ordinal responses as independent categories. This document presents a modular, extensible architecture for ordinal-aware attention mechanisms, designed for maintainability, testability, and improved QWK/ordinal accuracy metrics.

## Problem Analysis

### Current Limitations

1. **No Ordinal Structure Awareness**
   - Treats response categories (0, 1, 2, 3) as unrelated
   - Missing understanding that |2-3| < |0-3| (ordinal distance matters)

2. **Equal Treatment of All Prediction Errors**
   - Predicting 2 when answer is 3 (small error) treated same as predicting 0 when answer is 3 (large error)
   - Critical for QWK metric which penalizes based on distance

3. **No Response Pattern Learning**
   - Misses temporal patterns like:
     - Improving: 0 → 1 → 2 → 3
     - Declining: 3 → 2 → 1 → 0
     - Near-miss: Consistently ±1 from correct

4. **Generic Refinement Objective**
   - Optimizes for general attention quality
   - Not aligned with ordinal prediction objectives

## Modular Architecture Design

### Core Design Principles

1. **Separation of Concerns**: Each attention mechanism is a self-contained module
2. **Standardized Interfaces**: All mechanisms implement a common interface
3. **Composability**: Mechanisms can be combined and chained flexibly
4. **Configuration-Driven**: Behavior controlled through configuration, not code changes
5. **Testability**: Each component can be tested in isolation

### Base Classes and Interfaces

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn

@dataclass
class AttentionContext:
    """Standardized context for attention mechanisms."""
    embeddings: torch.Tensor      # [batch, seq_len, embed_dim]
    responses: torch.Tensor       # [batch, seq_len] - ordinal values
    mask: Optional[torch.Tensor]  # [batch, seq_len] - padding mask
    metadata: Dict[str, any]      # Additional context (e.g., timestamps, KC info)

class AttentionMechanism(ABC, nn.Module):
    """Base class for all attention mechanisms."""
    
    def __init__(self, config: 'AttentionConfig'):
        super().__init__()
        self.config = config
        self.embed_dim = config.embed_dim
        
    @abstractmethod
    def forward(self, context: AttentionContext) -> torch.Tensor:
        """Apply attention mechanism to input context."""
        pass
    
    def get_attention_weights(self, context: AttentionContext) -> Optional[torch.Tensor]:
        """Optional: Return attention weights for visualization."""
        return None

@dataclass
class AttentionConfig:
    """Configuration for attention mechanisms."""
    embed_dim: int = 512
    n_heads: int = 8
    n_cats: int = 4
    dropout: float = 0.1
    # Mechanism-specific configs
    distance_penalty: float = 0.5
    window_size: int = 3
    correct_threshold: int = 2
```

### Plugin Registry System

```python
class AttentionRegistry:
    """Registry for dynamically loading attention mechanisms."""
    
    _mechanisms: Dict[str, type] = {}
    
    @classmethod
    def register(cls, name: str):
        """Decorator to register attention mechanisms."""
        def decorator(mechanism_class):
            if not issubclass(mechanism_class, AttentionMechanism):
                raise ValueError(f"{mechanism_class} must inherit from AttentionMechanism")
            cls._mechanisms[name] = mechanism_class
            return mechanism_class
        return decorator
    
    @classmethod
    def create(cls, name: str, config: AttentionConfig) -> AttentionMechanism:
        """Create an attention mechanism by name."""
        if name not in cls._mechanisms:
            raise ValueError(f"Unknown mechanism: {name}")
        return cls._mechanisms[name](config)
    
    @classmethod
    def list_available(cls) -> List[str]:
        """List all registered mechanisms."""
        return list(cls._mechanisms.keys())
```

### Composable Pipeline Architecture

```python
class AttentionPipeline(nn.Module):
    """Compose multiple attention mechanisms into a pipeline."""
    
    def __init__(self, mechanisms: List[AttentionMechanism], fusion_method: str = 'concat'):
        super().__init__()
        self.mechanisms = nn.ModuleList(mechanisms)
        self.fusion_method = fusion_method
        
        if fusion_method == 'concat':
            total_dim = sum(m.embed_dim for m in mechanisms)
            self.fusion_layer = nn.Linear(total_dim, mechanisms[0].embed_dim)
        elif fusion_method == 'weighted':
            self.weights = nn.Parameter(torch.ones(len(mechanisms)) / len(mechanisms))
        
    def forward(self, context: AttentionContext) -> torch.Tensor:
        outputs = []
        for mechanism in self.mechanisms:
            output = mechanism(context)
            outputs.append(output)
        
        if self.fusion_method == 'concat':
            concatenated = torch.cat(outputs, dim=-1)
            return self.fusion_layer(concatenated)
        elif self.fusion_method == 'weighted':
            weights = torch.softmax(self.weights, dim=0)
            return sum(w * out for w, out in zip(weights, outputs))
        elif self.fusion_method == 'mean':
            return torch.stack(outputs).mean(dim=0)
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
```

## Proposed Improvements

### 1. Ordinal-Aware Attention Scoring

```python
@AttentionRegistry.register('ordinal_aware')
class OrdinalAwareAttention(AttentionMechanism):
    """Attention mechanism that considers ordinal distance between responses."""
    
    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        self.multihead_attn = nn.MultiheadAttention(
            config.embed_dim, 
            config.n_heads,
            dropout=config.dropout
        )
        self.distance_penalty = nn.Parameter(torch.tensor(config.distance_penalty))
        
    def forward(self, context: AttentionContext) -> torch.Tensor:
        embeddings = context.embeddings
        responses = context.responses
        batch_size, seq_len, _ = embeddings.shape
        
        # Transpose for multihead attention (seq_len first)
        embeddings_t = embeddings.transpose(0, 1)
        
        # Standard attention
        attn_output, attn_weights = self.multihead_attn(
            embeddings_t, embeddings_t, embeddings_t,
            key_padding_mask=context.mask
        )
        
        # Apply ordinal distance weighting
        if responses is not None:
            resp_distances = torch.abs(
                responses.unsqueeze(-1) - responses.unsqueeze(-2)
            )
            ordinal_weights = torch.exp(-self.distance_penalty * resp_distances)
            
            # Reshape attention weights and apply ordinal weighting
            attn_weights_reshaped = attn_weights.view(
                batch_size, seq_len, seq_len
            )
            attn_weights_reshaped = attn_weights_reshaped * ordinal_weights
            
            # Re-normalize
            attn_weights_reshaped = attn_weights_reshaped / (
                attn_weights_reshaped.sum(dim=-1, keepdim=True) + 1e-9
            )
            
            # Apply weighted attention to values
            values = embeddings
            output = torch.matmul(attn_weights_reshaped, values)
        else:
            output = attn_output.transpose(0, 1)
        
        return output
    
    def get_attention_weights(self, context: AttentionContext) -> torch.Tensor:
        """Return attention weights for visualization."""
        with torch.no_grad():
            embeddings_t = context.embeddings.transpose(0, 1)
            _, attn_weights = self.multihead_attn(
                embeddings_t, embeddings_t, embeddings_t,
                key_padding_mask=context.mask
            )
        return attn_weights
```

**Benefits**: Questions with similar response levels attend more to each other

### 2. Response-Conditioned Refinement

```python
@AttentionRegistry.register('response_conditioned')
class ResponseConditionedRefinement(AttentionMechanism):
    """Different refinement strategies based on response level."""
    
    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        n_cats = config.n_cats
        embed_dim = config.embed_dim
        
        # Separate refinement paths for each response category
        self.response_gates = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim) for _ in range(n_cats)
        ])
        
        # Response-specific transformations
        self.response_transforms = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout),
                nn.Linear(embed_dim, embed_dim)
            ) for _ in range(n_cats)
        ])
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, context: AttentionContext) -> torch.Tensor:
        embeddings = context.embeddings
        responses = context.responses
        batch_size, seq_len, embed_dim = embeddings.shape
        
        # Initialize output with input (residual connection)
        refined = embeddings.clone()
        
        for cat in range(self.config.n_cats):
            # Create mask for current category
            cat_mask = (responses == cat)
            
            if cat_mask.any():
                # Get embeddings for this category
                cat_embeds = embeddings[cat_mask]
                
                # Response-specific transformation
                transformed = self.response_transforms[cat](cat_embeds)
                
                # Response-specific gating
                gate = torch.sigmoid(self.response_gates[cat](cat_embeds))
                
                # Apply gated transformation
                refined[cat_mask] = gate * transformed + (1 - gate) * cat_embeds
        
        # Apply layer normalization
        refined = self.layer_norm(refined)
        
        return refined
```

**Benefits**: Allows model to learn different refinement strategies for different mastery levels

### 3. Ordinal Pattern Attention

```python
@AttentionRegistry.register('ordinal_pattern')
class OrdinalPatternAttention(AttentionMechanism):
    """Detect and leverage ordinal patterns in response sequences."""
    
    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        embed_dim = config.embed_dim
        n_cats = config.n_cats
        window_size = config.window_size
        
        # Pattern detection via temporal convolution
        self.pattern_conv = nn.Conv1d(
            in_channels=embed_dim + n_cats,
            out_channels=embed_dim,
            kernel_size=window_size,
            padding=window_size//2
        )
        
        # Pattern-based attention
        self.pattern_attention = nn.MultiheadAttention(
            embed_dim, 
            num_heads=config.n_heads,
            dropout=config.dropout
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(embed_dim)
        
    def forward(self, context: AttentionContext) -> torch.Tensor:
        embeddings = context.embeddings
        responses = context.responses
        batch_size, seq_len, embed_dim = embeddings.shape
        
        # Concatenate response information
        resp_onehot = F.one_hot(responses, num_classes=self.config.n_cats).float()
        combined = torch.cat([embeddings, resp_onehot], dim=-1)
        
        # Extract temporal patterns via convolution
        patterns = self.pattern_conv(combined.transpose(1, 2)).transpose(1, 2)
        patterns = F.relu(patterns)
        
        # Transpose for multihead attention
        embeddings_t = embeddings.transpose(0, 1)
        patterns_t = patterns.transpose(0, 1)
        
        # Apply pattern-based attention
        refined, attn_weights = self.pattern_attention(
            embeddings_t, patterns_t, patterns_t,
            key_padding_mask=context.mask
        )
        
        # Transpose back and add residual
        refined = refined.transpose(0, 1) + embeddings
        refined = self.layer_norm(refined)
        
        return refined
    
    def get_attention_weights(self, context: AttentionContext) -> torch.Tensor:
        """Return pattern attention weights."""
        with torch.no_grad():
            # Recompute patterns
            embeddings = context.embeddings
            responses = context.responses
            resp_onehot = F.one_hot(responses, num_classes=self.config.n_cats).float()
            combined = torch.cat([embeddings, resp_onehot], dim=-1)
            patterns = self.pattern_conv(combined.transpose(1, 2)).transpose(1, 2)
            patterns = F.relu(patterns)
            
            # Get attention weights
            embeddings_t = embeddings.transpose(0, 1)
            patterns_t = patterns.transpose(0, 1)
            _, attn_weights = self.pattern_attention(
                embeddings_t, patterns_t, patterns_t,
                key_padding_mask=context.mask
            )
        return attn_weights
```

**Benefits**: Captures improvement/decline trends and response sequences

### 4. QWK-Aligned Attention

```python
@AttentionRegistry.register('qwk_aligned')
class QWKAlignedAttention(AttentionMechanism):
    """Attention mechanism aligned with Quadratic Weighted Kappa metric."""
    
    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        embed_dim = config.embed_dim
        n_heads = config.n_heads
        
        # Multi-head projections
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads
        assert self.head_dim * n_heads == embed_dim, "embed_dim must be divisible by n_heads"
        
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Dropout
        self.attn_dropout = nn.Dropout(config.dropout)
        
        # QWK weight matrix
        self.register_buffer('qwk_weights', self._create_qwk_matrix(config.n_cats))
        
    def _create_qwk_matrix(self, n_cats):
        """Create quadratic weight matrix for QWK."""
        weights = torch.zeros(n_cats, n_cats)
        for i in range(n_cats):
            for j in range(n_cats):
                weights[i, j] = 1 - ((i - j) ** 2) / ((n_cats - 1) ** 2)
        return weights
    
    def forward(self, context: AttentionContext) -> torch.Tensor:
        embeddings = context.embeddings
        responses = context.responses
        batch_size, seq_len, embed_dim = embeddings.shape
        
        # Multi-head projections
        Q = self.q_proj(embeddings).view(batch_size, seq_len, self.n_heads, self.head_dim)
        K = self.k_proj(embeddings).view(batch_size, seq_len, self.n_heads, self.head_dim)
        V = self.v_proj(embeddings).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Transpose for attention computation
        Q = Q.transpose(1, 2)  # [batch, n_heads, seq_len, head_dim]
        K = K.transpose(1, 2)
        V = V.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Apply QWK-based weighting
        resp_i = responses.unsqueeze(2).expand(-1, -1, seq_len)
        resp_j = responses.unsqueeze(1).expand(-1, seq_len, -1)
        qwk_mask = self.qwk_weights[resp_i, resp_j]
        
        # Expand QWK mask for all heads
        qwk_mask = qwk_mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
        
        # Combine attention scores with QWK weights
        scores = scores * qwk_mask
        
        # Apply mask if provided
        if context.mask is not None:
            mask = context.mask.unsqueeze(1).unsqueeze(2)
            scores = scores.masked_fill(mask, -1e9)
        
        # Apply softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        
        # Compute output
        output = torch.matmul(attn_weights, V)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, embed_dim)
        output = self.out_proj(output)
        
        return output
    
    def get_attention_weights(self, context: AttentionContext) -> torch.Tensor:
        """Return QWK-weighted attention scores."""
        with torch.no_grad():
            embeddings = context.embeddings
            responses = context.responses
            batch_size, seq_len, _ = embeddings.shape
            
            Q = self.q_proj(embeddings).view(batch_size, seq_len, self.n_heads, self.head_dim)
            K = self.k_proj(embeddings).view(batch_size, seq_len, self.n_heads, self.head_dim)
            
            Q = Q.transpose(1, 2)
            K = K.transpose(1, 2)
            
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
            
            # Apply QWK weighting
            resp_i = responses.unsqueeze(2).expand(-1, -1, seq_len)
            resp_j = responses.unsqueeze(1).expand(-1, seq_len, -1)
            qwk_mask = self.qwk_weights[resp_i, resp_j]
            qwk_mask = qwk_mask.unsqueeze(1).expand(-1, self.n_heads, -1, -1)
            
            scores = scores * qwk_mask
            attn_weights = F.softmax(scores, dim=-1)
            
        return attn_weights.mean(dim=1)  # Average over heads
```

**Benefits**: Attention weights directly consider QWK scoring for better metric alignment

### 5. Hierarchical Ordinal Attention

```python
@AttentionRegistry.register('hierarchical_ordinal')
class HierarchicalOrdinalAttention(AttentionMechanism):
    """Two-level attention: binary (correct/incorrect) then ordinal refinement."""
    
    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        embed_dim = config.embed_dim
        n_heads = config.n_heads
        
        # Level 1: Binary distinction
        self.binary_attention = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=config.dropout
        )
        self.binary_norm = nn.LayerNorm(embed_dim)
        self.binary_dropout = nn.Dropout(config.dropout)
        
        # Level 2: Fine-grained ordinal
        self.ordinal_attention = nn.MultiheadAttention(
            embed_dim, n_heads, dropout=config.dropout
        )
        self.ordinal_norm = nn.LayerNorm(embed_dim)
        self.ordinal_dropout = nn.Dropout(config.dropout)
        
        # Learnable threshold for "correct"
        self.correct_threshold = nn.Parameter(
            torch.tensor(float(config.correct_threshold))
        )
        
    def forward(self, context: AttentionContext) -> torch.Tensor:
        embeddings = context.embeddings
        responses = context.responses
        
        # Transpose for multihead attention
        embeddings_t = embeddings.transpose(0, 1)
        
        # Level 1: Binary attention (correct vs incorrect)
        binary_labels = (responses >= self.correct_threshold).float()
        
        # Apply binary-aware attention
        binary_refined, binary_weights = self.binary_attention(
            embeddings_t, embeddings_t, embeddings_t,
            key_padding_mask=context.mask
        )
        
        # Transpose back and apply residual + norm
        binary_refined = binary_refined.transpose(0, 1)
        binary_refined = self.binary_dropout(binary_refined)
        binary_refined = self.binary_norm(embeddings + binary_refined)
        
        # Level 2: Ordinal refinement
        binary_refined_t = binary_refined.transpose(0, 1)
        ordinal_refined, ordinal_weights = self.ordinal_attention(
            binary_refined_t, binary_refined_t, binary_refined_t,
            key_padding_mask=context.mask
        )
        
        # Transpose back and apply residual + norm
        ordinal_refined = ordinal_refined.transpose(0, 1)
        ordinal_refined = self.ordinal_dropout(ordinal_refined)
        ordinal_refined = self.ordinal_norm(binary_refined + ordinal_refined)
        
        return ordinal_refined
    
    def get_attention_weights(self, context: AttentionContext) -> Dict[str, torch.Tensor]:
        """Return both binary and ordinal attention weights."""
        with torch.no_grad():
            embeddings_t = context.embeddings.transpose(0, 1)
            
            # Binary attention weights
            _, binary_weights = self.binary_attention(
                embeddings_t, embeddings_t, embeddings_t,
                key_padding_mask=context.mask
            )
            
            # Compute binary refined for ordinal attention
            binary_refined, _ = self.binary_attention(
                embeddings_t, embeddings_t, embeddings_t,
                key_padding_mask=context.mask
            )
            binary_refined = binary_refined.transpose(0, 1)
            binary_refined = self.binary_norm(context.embeddings + binary_refined)
            binary_refined_t = binary_refined.transpose(0, 1)
            
            # Ordinal attention weights
            _, ordinal_weights = self.ordinal_attention(
                binary_refined_t, binary_refined_t, binary_refined_t,
                key_padding_mask=context.mask
            )
            
        return {
            'binary': binary_weights,
            'ordinal': ordinal_weights
        }
```

**Benefits**: Hierarchical processing matches how we think about partial credit

## Integrated Architecture

```python
class ModularAttentionGPCM(AttentionGPCM):
    """Enhanced AttentionGPCM with modular ordinal-aware attention mechanisms."""
    
    def __init__(self, 
                 n_questions, n_responses=4, embed_dim=512,
                 attention_config: Optional[AttentionConfig] = None,
                 attention_mechanisms: Optional[List[str]] = None,
                 fusion_method: str = 'concat',
                 **kwargs):
        super().__init__(n_questions, n_responses, embed_dim, **kwargs)
        
        # Default configuration
        if attention_config is None:
            attention_config = AttentionConfig(
                embed_dim=embed_dim,
                n_cats=n_responses,
                n_heads=8,
                dropout=0.1
            )
        
        # Default mechanisms if not specified
        if attention_mechanisms is None:
            attention_mechanisms = [
                'ordinal_aware',
                'response_conditioned',
                'ordinal_pattern',
                'qwk_aligned'
            ]
        
        # Create attention pipeline
        mechanisms = []
        for mech_name in attention_mechanisms:
            mechanism = AttentionRegistry.create(mech_name, attention_config)
            mechanisms.append(mechanism)
        
        self.attention_pipeline = AttentionPipeline(
            mechanisms, 
            fusion_method=fusion_method
        )
        
        # Store config for later use
        self.attention_config = attention_config
        self._context_cache = {}
    
    def create_embeddings(self, questions, responses):
        """Override to create attention context."""
        embeddings = super().create_embeddings(questions, responses)
        
        # Create attention context
        context = AttentionContext(
            embeddings=embeddings,
            responses=responses,
            mask=None,  # Can be added based on padding
            metadata={
                'questions': questions,
                'batch_size': questions.size(0),
                'seq_len': questions.size(1)
            }
        )
        
        # Cache context for process_embeddings
        self._context_cache[id(embeddings)] = context
        
        return embeddings
    
    def process_embeddings(self, gpcm_embeds, q_embeds):
        """Apply modular attention refinement."""
        # Retrieve context
        context = self._context_cache.get(id(gpcm_embeds))
        
        if context is None:
            # Fallback to original if context not available
            return super().process_embeddings(gpcm_embeds, q_embeds)
        
        # Apply attention pipeline
        refined = self.attention_pipeline(context)
        
        # Clean up context cache
        del self._context_cache[id(gpcm_embeds)]
        
        return refined
    
    def get_attention_weights(self) -> Dict[str, torch.Tensor]:
        """Get attention weights from all mechanisms for analysis."""
        weights = {}
        for i, mechanism in enumerate(self.attention_pipeline.mechanisms):
            name = type(mechanism).__name__
            if hasattr(self, '_last_context'):
                mech_weights = mechanism.get_attention_weights(self._last_context)
                if mech_weights is not None:
                    weights[f"{name}_{i}"] = mech_weights
        return weights

# Factory function for easy creation
def create_attention_gpcm(n_questions, n_responses=4, 
                         attention_type='modular',
                         mechanisms=None,
                         **kwargs):
    """Factory function to create AttentionGPCM variants."""
    if attention_type == 'modular':
        return ModularAttentionGPCM(
            n_questions, n_responses,
            attention_mechanisms=mechanisms,
            **kwargs
        )
    elif attention_type == 'basic':
        return AttentionGPCM(n_questions, n_responses, **kwargs)
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")
```

## Implementation Strategy

### Phase 1: Individual Components (Week 1)
1. Implement and test each attention component separately
2. Validate on synthetic ordinal data
3. Measure impact on QWK and ordinal accuracy

### Phase 2: Integration (Week 2)
1. Combine components into integrated architecture
2. Experiment with different combinations
3. Hyperparameter tuning

### Phase 3: Evaluation (Week 3)
1. Full evaluation on all datasets
2. Ablation studies
3. Comparison with baseline AttentionGPCM

## Expected Improvements

1. **QWK Score**: +0.05 to +0.10 improvement
2. **Ordinal Accuracy**: +3% to +5% improvement
3. **Near-Miss Predictions**: Significant reduction in large errors (|pred - true| > 2)

## Evaluation Metrics

1. **Primary Metrics**:
   - Quadratic Weighted Kappa (QWK)
   - Ordinal Accuracy (exact match)
   - Mean Absolute Error (MAE)

2. **Secondary Metrics**:
   - Near-miss rate (|pred - true| = 1)
   - Large error rate (|pred - true| > 2)
   - Per-category F1 scores

## Testing Strategy

### Unit Testing

```python
import pytest
import torch
from attention_mechanisms import AttentionRegistry, AttentionConfig, AttentionContext

class TestAttentionMechanisms:
    """Unit tests for individual attention mechanisms."""
    
    @pytest.fixture
    def sample_context(self):
        """Create sample attention context."""
        batch_size, seq_len, embed_dim = 2, 10, 64
        return AttentionContext(
            embeddings=torch.randn(batch_size, seq_len, embed_dim),
            responses=torch.randint(0, 4, (batch_size, seq_len)),
            mask=None,
            metadata={}
        )
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return AttentionConfig(embed_dim=64, n_heads=4, n_cats=4)
    
    @pytest.mark.parametrize("mechanism_name", AttentionRegistry.list_available())
    def test_mechanism_forward(self, mechanism_name, config, sample_context):
        """Test forward pass of each mechanism."""
        mechanism = AttentionRegistry.create(mechanism_name, config)
        output = mechanism(sample_context)
        
        # Check output shape
        assert output.shape == sample_context.embeddings.shape
        
        # Check no NaN values
        assert not torch.isnan(output).any()
    
    def test_attention_weights(self, config, sample_context):
        """Test attention weight extraction."""
        mechanism = AttentionRegistry.create('qwk_aligned', config)
        weights = mechanism.get_attention_weights(sample_context)
        
        if weights is not None:
            # Check weight properties
            assert weights.min() >= 0
            assert weights.max() <= 1
            # Check weights sum to 1 along attention dimension
            assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))
```

### Integration Testing

```python
class TestModularAttentionGPCM:
    """Integration tests for the complete system."""
    
    def test_pipeline_integration(self):
        """Test complete attention pipeline."""
        model = ModularAttentionGPCM(
            n_questions=100,
            n_responses=4,
            attention_mechanisms=['ordinal_aware', 'qwk_aligned'],
            fusion_method='weighted'
        )
        
        # Test data
        questions = torch.randint(0, 100, (8, 20))
        responses = torch.randint(0, 4, (8, 20))
        
        # Forward pass
        output = model(questions, responses)
        
        # Check output validity
        assert output.shape == (8, 20, 4)
        assert not torch.isnan(output).any()
    
    def test_mechanism_composition(self):
        """Test different mechanism combinations."""
        configs = [
            (['ordinal_aware'], 'mean'),
            (['response_conditioned', 'ordinal_pattern'], 'concat'),
            (['qwk_aligned', 'hierarchical_ordinal'], 'weighted'),
        ]
        
        for mechanisms, fusion in configs:
            model = create_attention_gpcm(
                n_questions=100,
                attention_type='modular',
                mechanisms=mechanisms,
                fusion_method=fusion
            )
            
            # Verify model creation
            assert len(model.attention_pipeline.mechanisms) == len(mechanisms)
```

## Performance Optimization

### Caching Strategies

```python
class CachedAttentionMechanism(AttentionMechanism):
    """Base class with caching support for attention mechanisms."""
    
    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        self._cache = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def _get_cache_key(self, context: AttentionContext) -> str:
        """Generate cache key from context."""
        # Use tensor shapes and response patterns as key
        key_parts = [
            str(context.embeddings.shape),
            str(context.responses.unique().tolist()),
            str(context.mask is not None)
        ]
        return "_".join(key_parts)
    
    def forward(self, context: AttentionContext) -> torch.Tensor:
        """Forward with caching."""
        if not self.training:  # Only cache during inference
            cache_key = self._get_cache_key(context)
            
            if cache_key in self._cache:
                self._cache_hits += 1
                return self._cache[cache_key].clone()
            
            self._cache_misses += 1
            output = self._forward_impl(context)
            self._cache[cache_key] = output.clone()
            return output
        
        return self._forward_impl(context)
    
    def _forward_impl(self, context: AttentionContext) -> torch.Tensor:
        """Actual forward implementation to be overridden."""
        raise NotImplementedError
    
    def clear_cache(self):
        """Clear the cache."""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0
```

### Optimization Techniques

```python
# 1. Gradient Checkpointing for Memory Efficiency
class MemoryEfficientPipeline(AttentionPipeline):
    """Pipeline with gradient checkpointing."""
    
    def forward(self, context: AttentionContext) -> torch.Tensor:
        outputs = []
        for mechanism in self.mechanisms:
            # Use gradient checkpointing for each mechanism
            output = torch.utils.checkpoint.checkpoint(
                mechanism, context
            )
            outputs.append(output)
        
        return self._fuse_outputs(outputs)

# 2. Mixed Precision Training
def create_optimized_model(**kwargs):
    """Create model with optimization settings."""
    model = ModularAttentionGPCM(**kwargs)
    
    # Enable mixed precision
    model = model.half()  # Convert to fp16
    
    # Use torch.compile for additional optimization (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        model = torch.compile(model)
    
    return model

# 3. Batch Processing Optimization
class BatchOptimizedAttention(AttentionMechanism):
    """Optimized for large batch processing."""
    
    def forward(self, context: AttentionContext) -> torch.Tensor:
        # Process in smaller chunks to avoid memory issues
        batch_size = context.embeddings.size(0)
        chunk_size = 32
        
        if batch_size > chunk_size:
            outputs = []
            for i in range(0, batch_size, chunk_size):
                chunk_context = AttentionContext(
                    embeddings=context.embeddings[i:i+chunk_size],
                    responses=context.responses[i:i+chunk_size],
                    mask=context.mask[i:i+chunk_size] if context.mask is not None else None,
                    metadata=context.metadata
                )
                outputs.append(self._process_chunk(chunk_context))
            
            return torch.cat(outputs, dim=0)
        
        return self._process_chunk(context)
```

## Extension Guide

### Adding New Attention Mechanisms

```python
# Step 1: Define your mechanism
@AttentionRegistry.register('my_custom_attention')
class MyCustomAttention(AttentionMechanism):
    """Custom attention mechanism for specific use case."""
    
    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        # Add your components
        self.custom_layer = nn.Linear(config.embed_dim, config.embed_dim)
        
    def forward(self, context: AttentionContext) -> torch.Tensor:
        embeddings = context.embeddings
        
        # Your custom attention logic
        attended = self.custom_layer(embeddings)
        
        # Must return tensor of same shape as input embeddings
        return attended

# Step 2: Use in model
model = ModularAttentionGPCM(
    n_questions=100,
    attention_mechanisms=['my_custom_attention', 'ordinal_aware']
)
```

### Creating Composite Mechanisms

```python
class CompositeAttention(AttentionMechanism):
    """Combine multiple mechanisms with custom logic."""
    
    def __init__(self, config: AttentionConfig):
        super().__init__(config)
        
        # Create sub-mechanisms
        self.ordinal = OrdinalAwareAttention(config)
        self.pattern = OrdinalPatternAttention(config)
        
        # Fusion layer
        self.gate = nn.Sequential(
            nn.Linear(config.embed_dim * 2, config.embed_dim),
            nn.Sigmoid()
        )
    
    def forward(self, context: AttentionContext) -> torch.Tensor:
        # Apply sub-mechanisms
        ordinal_out = self.ordinal(context)
        pattern_out = self.pattern(context)
        
        # Custom fusion logic
        concat = torch.cat([ordinal_out, pattern_out], dim=-1)
        gate_weights = self.gate(concat)
        
        # Gated combination
        output = gate_weights * ordinal_out + (1 - gate_weights) * pattern_out
        
        return output
```

## Configuration Management

### YAML Configuration

```yaml
# config/attention_config.yaml
attention:
  type: modular
  mechanisms:
    - ordinal_aware
    - response_conditioned
    - qwk_aligned
  
  fusion_method: weighted
  
  config:
    embed_dim: 512
    n_heads: 8
    n_cats: 4
    dropout: 0.1
    distance_penalty: 0.5
    window_size: 3
    correct_threshold: 2

model:
  n_questions: 1000
  n_responses: 4
  embed_dim: 512
  
training:
  learning_rate: 0.001
  batch_size: 32
  epochs: 100
  gradient_checkpointing: true
  mixed_precision: true
```

### Configuration Loading

```python
import yaml
from dataclasses import asdict

class ConfigLoader:
    """Load and validate configuration."""
    
    @staticmethod
    def load_config(config_path: str) -> Dict:
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    @staticmethod
    def create_attention_config(config_dict: Dict) -> AttentionConfig:
        """Create AttentionConfig from dictionary."""
        attention_params = config_dict.get('attention', {}).get('config', {})
        return AttentionConfig(**attention_params)
    
    @staticmethod
    def create_model_from_config(config_path: str) -> ModularAttentionGPCM:
        """Create model from configuration file."""
        config = ConfigLoader.load_config(config_path)
        
        # Extract model parameters
        model_params = config.get('model', {})
        attention_params = config.get('attention', {})
        
        # Create attention config
        attention_config = ConfigLoader.create_attention_config(config)
        
        # Create model
        model = ModularAttentionGPCM(
            n_questions=model_params['n_questions'],
            n_responses=model_params.get('n_responses', 4),
            embed_dim=model_params.get('embed_dim', 512),
            attention_config=attention_config,
            attention_mechanisms=attention_params.get('mechanisms'),
            fusion_method=attention_params.get('fusion_method', 'concat')
        )
        
        return model

# Usage
model = ConfigLoader.create_model_from_config('config/attention_config.yaml')
```

## Migration Strategy

### Phase 1: Parallel Implementation (Week 1-2)
1. **Keep existing AttentionGPCM unchanged**
2. **Implement modular components alongside**
3. **Create compatibility layer**

```python
class CompatibilityWrapper(nn.Module):
    """Wrap modular attention to work with existing code."""
    
    def __init__(self, legacy_model, modular_model):
        super().__init__()
        self.legacy = legacy_model
        self.modular = modular_model
        self.use_modular = False
    
    def forward(self, questions, responses):
        if self.use_modular:
            return self.modular(questions, responses)
        return self.legacy(questions, responses)
    
    def enable_modular(self):
        """Switch to modular implementation."""
        self.use_modular = True
        # Copy weights if needed
        self._migrate_weights()
```

### Phase 2: Gradual Migration (Week 3-4)
1. **Add feature flags for modular components**
2. **A/B test modular vs legacy**
3. **Monitor performance metrics**

```python
# Feature flag configuration
FEATURE_FLAGS = {
    'use_modular_attention': False,
    'modular_mechanisms': ['ordinal_aware'],
    'enable_caching': True,
    'gradual_rollout_percentage': 0.1  # 10% of training runs
}

def create_model_with_flags(n_questions, **kwargs):
    """Create model based on feature flags."""
    if FEATURE_FLAGS['use_modular_attention']:
        return ModularAttentionGPCM(
            n_questions,
            attention_mechanisms=FEATURE_FLAGS['modular_mechanisms'],
            **kwargs
        )
    return AttentionGPCM(n_questions, **kwargs)
```

### Phase 3: Full Migration (Week 5-6)
1. **Replace legacy code with modular**
2. **Update all dependent code**
3. **Deprecate old implementation**

```python
# Deprecation warnings
import warnings

class AttentionGPCM(nn.Module):
    def __init__(self, *args, **kwargs):
        warnings.warn(
            "AttentionGPCM is deprecated. Use ModularAttentionGPCM instead.",
            DeprecationWarning,
            stacklevel=2
        )
        # Redirect to modular implementation
        self._impl = ModularAttentionGPCM(*args, **kwargs)
    
    def forward(self, *args, **kwargs):
        return self._impl(*args, **kwargs)
```

### Migration Checklist
- [ ] Implement all modular components
- [ ] Create comprehensive test suite
- [ ] Benchmark performance vs legacy
- [ ] Document API changes
- [ ] Update training scripts
- [ ] Migrate saved model checkpoints
- [ ] Update evaluation pipelines
- [ ] Remove legacy code

## Risk Mitigation

1. **Overfitting**: Use dropout and regularization in attention layers
2. **Computational Cost**: Profile and optimize expensive operations
3. **Training Instability**: Careful initialization and gradient clipping
4. **Backward Compatibility**: Maintain legacy interface during migration
5. **Performance Regression**: Continuous benchmarking and monitoring

## References

1. SAKT: A Self-Attentive Model for Knowledge Tracing
2. AKT: Context-Aware Attentive Knowledge Tracing
3. Ordinal Regression with Multiple Output CNN
4. Learning to Rank with Attentive Media