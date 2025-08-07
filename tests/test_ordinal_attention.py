"""
Unit tests for ordinal-aware attention mechanisms.
"""

import torch
import torch.nn as nn
import pytest
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.components.ordinal_attention import (
    OrdinalAwareSelfAttention,
    ResponseConditionedAttention,
    OrdinalPatternAttention,
    QWKAlignedAttention,
    HierarchicalOrdinalAttention,
    AttentionRegistry,
    OrdinalAttentionPipeline
)
from models.components.ordinal_attention_integration import (
    OrdinalAwareAttentionRefinement,
    create_ordinal_attention
)


class TestAttentionRegistry:
    """Test the attention registry system."""
    
    def test_registry_basic(self):
        """Test basic registry functionality."""
        # Check registered types
        available = AttentionRegistry.list_available()
        assert "ordinal_aware" in available
        assert "response_conditioned" in available
        assert "ordinal_pattern" in available
        assert "qwk_aligned" in available
        assert "hierarchical_ordinal" in available
        
        # Get attention classes
        ordinal_cls = AttentionRegistry.get("ordinal_aware")
        assert ordinal_cls == OrdinalAwareSelfAttention
        
        response_cls = AttentionRegistry.get("response_conditioned")
        assert response_cls == ResponseConditionedAttention
    
    def test_registry_error(self):
        """Test registry error handling."""
        with pytest.raises(ValueError):
            AttentionRegistry.get("nonexistent")


class TestOrdinalAwareSelfAttention:
    """Test ordinal-aware self-attention mechanism."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.batch_size = 2
        self.seq_len = 5
        self.embed_dim = 64
        self.n_cats = 4
        self.n_heads = 8
        
        self.attention = OrdinalAwareSelfAttention(
            embed_dim=self.embed_dim,
            n_cats=self.n_cats,
            n_heads=self.n_heads,
            dropout=0.0  # No dropout for testing
        )
    
    def test_forward_shape(self):
        """Test output shapes."""
        # Create inputs
        query = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        key = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        value = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        responses = torch.randint(0, self.n_cats, (self.batch_size, self.seq_len))
        
        # Forward pass
        output = self.attention(query, key, value, responses=responses)
        
        # Check shape
        assert output.shape == (self.batch_size, self.seq_len, self.embed_dim)
    
    def test_ordinal_distance_computation(self):
        """Test ordinal distance calculation."""
        # Create test responses
        responses = torch.tensor([[0, 1, 2, 3], [3, 2, 1, 0]])
        
        # Compute distances
        distances = self.attention.compute_ordinal_distances(responses)
        
        # Check shape
        assert distances.shape == (2, 4, 4)
        
        # Check specific distances
        assert distances[0, 0, 0].item() == 0.0  # Same response
        assert distances[0, 0, 3].item() == 1.0  # Max distance (0 to 3)
        assert abs(distances[0, 0, 1].item() - 1/3) < 1e-5  # Distance 1
        
        # Check symmetry
        assert torch.allclose(distances[0, 0, 1], distances[0, 1, 0])
    
    def test_mask_application(self):
        """Test attention masking."""
        query = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        key = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        value = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        
        # Create mask (attend only to past positions)
        mask = torch.tril(torch.ones(self.seq_len, self.seq_len))
        mask = mask.unsqueeze(0).expand(self.batch_size, -1, -1)
        
        # Forward with mask
        output = self.attention(query, key, value, mask=mask)
        
        # Check output shape
        assert output.shape == (self.batch_size, self.seq_len, self.embed_dim)
    
    def test_config(self):
        """Test configuration retrieval."""
        config = self.attention.get_config()
        
        assert config["type"] == "ordinal_aware"
        assert config["embed_dim"] == self.embed_dim
        assert config["n_cats"] == self.n_cats
        assert config["n_heads"] == self.n_heads


class TestResponseConditionedAttention:
    """Test response-conditioned attention mechanism."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.batch_size = 2
        self.seq_len = 5
        self.embed_dim = 64
        self.n_cats = 4
        self.n_heads = 8
        
        self.attention = ResponseConditionedAttention(
            embed_dim=self.embed_dim,
            n_cats=self.n_cats,
            n_heads=self.n_heads,
            dropout=0.0
        )
    
    def test_forward_shape(self):
        """Test output shapes."""
        query = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        key = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        value = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        responses = torch.randint(0, self.n_cats, (self.batch_size, self.seq_len))
        
        output = self.attention(query, key, value, responses=responses)
        
        assert output.shape == (self.batch_size, self.seq_len, self.embed_dim)
    
    def test_response_modulation(self):
        """Test that different responses produce different outputs."""
        query = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        key = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        value = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        
        # Test with different response patterns
        responses1 = torch.zeros(self.batch_size, self.seq_len, dtype=torch.long)
        responses2 = torch.ones(self.batch_size, self.seq_len, dtype=torch.long)
        
        output1 = self.attention(query, key, value, responses=responses1)
        output2 = self.attention(query, key, value, responses=responses2)
        
        # Outputs should be different
        assert not torch.allclose(output1, output2, atol=1e-5)
    
    def test_without_responses(self):
        """Test behavior without response information."""
        query = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        key = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        value = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        
        # Should work without responses
        output = self.attention(query, key, value)
        assert output.shape == (self.batch_size, self.seq_len, self.embed_dim)


class TestOrdinalAttentionPipeline:
    """Test attention pipeline composition."""
    
    def test_pipeline_composition(self):
        """Test composing multiple attention mechanisms."""
        embed_dim = 64
        n_cats = 4
        
        # Create two mechanisms
        ordinal_attn = OrdinalAwareSelfAttention(embed_dim, n_cats)
        response_attn = ResponseConditionedAttention(embed_dim, n_cats)
        
        # Create pipeline
        pipeline = OrdinalAttentionPipeline([ordinal_attn, response_attn])
        
        # Test forward
        batch_size, seq_len = 2, 5
        query = torch.randn(batch_size, seq_len, embed_dim)
        key = torch.randn(batch_size, seq_len, embed_dim)
        value = torch.randn(batch_size, seq_len, embed_dim)
        responses = torch.randint(0, n_cats, (batch_size, seq_len))
        
        output = pipeline(query, key, value, responses=responses)
        assert output.shape == (batch_size, seq_len, embed_dim)


class TestOrdinalAwareAttentionRefinement:
    """Test the integrated attention refinement module."""
    
    def test_legacy_compatibility(self):
        """Test backward compatibility with legacy mode."""
        # Create with legacy mode
        refinement = OrdinalAwareAttentionRefinement(
            embed_dim=64,
            n_heads=4,
            n_cycles=2,
            dropout_rate=0.0,
            n_cats=4,
            use_legacy=True
        )
        
        # Should have legacy module
        assert hasattr(refinement, 'legacy_module')
        
        # Test forward
        embeddings = torch.randn(2, 5, 64)
        output = refinement(embeddings)
        assert output.shape == embeddings.shape
    
    def test_ordinal_mode(self):
        """Test ordinal-aware mode."""
        refinement = OrdinalAwareAttentionRefinement(
            embed_dim=64,
            n_heads=4,
            n_cycles=2,
            dropout_rate=0.0,
            n_cats=4,
            attention_types=["ordinal_aware"],
            use_legacy=False
        )
        
        # Test forward with responses
        embeddings = torch.randn(2, 5, 64)
        responses = torch.randint(0, 4, (2, 5))
        
        output = refinement(embeddings, responses)
        assert output.shape == embeddings.shape
    
    def test_multi_mechanism(self):
        """Test with multiple attention mechanisms."""
        refinement = OrdinalAwareAttentionRefinement(
            embed_dim=64,
            n_heads=4,
            n_cycles=1,  # Single cycle for testing
            dropout_rate=0.0,
            n_cats=4,
            attention_types=["ordinal_aware", "response_conditioned"],
            use_legacy=False
        )
        
        embeddings = torch.randn(2, 5, 64)
        responses = torch.randint(0, 4, (2, 5))
        
        output = refinement(embeddings, responses)
        assert output.shape == embeddings.shape


class TestFactoryFunction:
    """Test the factory function for creating attention mechanisms."""
    
    def test_create_ordinal_attention(self):
        """Test creating attention mechanisms via factory."""
        # Create ordinal-aware attention
        ordinal_attn = create_ordinal_attention(
            "ordinal_aware",
            embed_dim=64,
            n_cats=4,
            n_heads=8,
            dropout=0.1,
            distance_penalty=0.2
        )
        
        assert isinstance(ordinal_attn, OrdinalAwareSelfAttention)
        config = ordinal_attn.get_config()
        assert config["distance_penalty"] == 0.2
        
        # Create response-conditioned attention
        response_attn = create_ordinal_attention(
            "response_conditioned",
            embed_dim=64,
            n_cats=4,
            n_heads=8,
            dropout=0.1
        )
        
        assert isinstance(response_attn, ResponseConditionedAttention)


class TestOrdinalPatternAttention:
    """Test ordinal pattern attention mechanism."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.batch_size = 2
        self.seq_len = 5
        self.embed_dim = 64
        self.n_cats = 4
        self.n_heads = 8
        self.pattern_size = 3
        
        self.attention = OrdinalPatternAttention(
            embed_dim=self.embed_dim,
            n_cats=self.n_cats,
            n_heads=self.n_heads,
            dropout=0.0,
            pattern_size=self.pattern_size
        )
    
    def test_forward_shape(self):
        """Test output shapes."""
        query = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        key = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        value = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        responses = torch.randint(0, self.n_cats, (self.batch_size, self.seq_len))
        
        output = self.attention(query, key, value, responses=responses)
        
        assert output.shape == (self.batch_size, self.seq_len, self.embed_dim)
    
    def test_pattern_extraction(self):
        """Test pattern extraction."""
        responses = torch.tensor([[0, 1, 2, 3, 2], [3, 2, 1, 0, 1]])
        patterns = self.attention.extract_patterns(responses)
        
        assert patterns.shape == responses.shape
        # Check that patterns are valid indices
        max_pattern = self.n_cats ** self.pattern_size
        assert patterns.max() < max_pattern
        assert patterns.min() >= 0


class TestQWKAlignedAttention:
    """Test QWK-aligned attention mechanism."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.batch_size = 2
        self.seq_len = 5
        self.embed_dim = 64
        self.n_cats = 4
        self.n_heads = 8
        
        self.attention = QWKAlignedAttention(
            embed_dim=self.embed_dim,
            n_cats=self.n_cats,
            n_heads=self.n_heads,
            dropout=0.0
        )
    
    def test_qwk_weights(self):
        """Test QWK weight matrix creation."""
        weights = self.attention._create_qwk_weights()
        
        # Check shape
        assert weights.shape == (self.n_cats, self.n_cats)
        
        # Check diagonal is 1
        assert torch.allclose(weights.diag(), torch.ones(self.n_cats))
        
        # Check symmetry
        assert torch.allclose(weights, weights.T)
        
        # Check specific values for 4 categories
        assert weights[0, 0].item() == 1.0
        assert abs(weights[0, 1].item() - 8/9) < 1e-5  # 1 - (1/3)^2
        assert abs(weights[0, 2].item() - 5/9) < 1e-5  # 1 - (2/3)^2
        assert weights[0, 3].item() == 0.0  # 1 - (3/3)^2
    
    def test_forward_shape(self):
        """Test output shapes."""
        query = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        key = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        value = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        responses = torch.randint(0, self.n_cats, (self.batch_size, self.seq_len))
        
        output = self.attention(query, key, value, responses=responses)
        
        assert output.shape == (self.batch_size, self.seq_len, self.embed_dim)


class TestHierarchicalOrdinalAttention:
    """Test hierarchical ordinal attention mechanism."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.batch_size = 2
        self.seq_len = 5
        self.embed_dim = 64
        self.n_cats = 4
        self.n_heads = 8
        self.n_levels = 2
        
        self.attention = HierarchicalOrdinalAttention(
            embed_dim=self.embed_dim,
            n_cats=self.n_cats,
            n_heads=self.n_heads,
            dropout=0.0,
            n_levels=self.n_levels
        )
    
    def test_hierarchy_creation(self):
        """Test hierarchy creation."""
        hierarchy = self.attention._create_hierarchy()
        
        assert len(hierarchy) == self.n_levels
        
        # Level 1: binary split
        assert len(hierarchy[0]) == 2
        assert hierarchy[0][0] == [0, 1]  # Low
        assert hierarchy[0][1] == [2, 3]  # High
        
        # Level 2: individual
        assert len(hierarchy[1]) == self.n_cats
        for i in range(self.n_cats):
            assert hierarchy[1][i] == [i]
    
    def test_forward_shape(self):
        """Test output shapes."""
        query = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        key = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        value = torch.randn(self.batch_size, self.seq_len, self.embed_dim)
        responses = torch.randint(0, self.n_cats, (self.batch_size, self.seq_len))
        
        output = self.attention(query, key, value, responses=responses)
        
        assert output.shape == (self.batch_size, self.seq_len, self.embed_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])