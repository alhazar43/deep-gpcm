#!/usr/bin/env python3
"""
Debug script for investigating TemporalAttentionGPCM early performance issues.

This script creates diagnostic versions of the model to identify why temporal features
cause poor early QWK performance and why larger batch sizes are needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Optional, Tuple, List
from pathlib import Path

# Import model components
from models.implementations.temporal_attention_gpcm import TemporalAttentionGPCM
from models.implementations.attention_gpcm import AttentionGPCM
from models.factory import create_model
from utils.data_loading import load_dataset

class DiagnosticTemporalGPCM(TemporalAttentionGPCM):
    """Diagnostic version with controllable temporal features."""
    
    def __init__(self, *args, enable_temporal=True, enable_positional=True, 
                 temporal_noise_std=0.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.enable_temporal = enable_temporal
        self.enable_positional = enable_positional  
        self.temporal_noise_std = temporal_noise_std
        
    def create_embeddings(self, questions: torch.Tensor, responses: torch.Tensor) -> torch.Tensor:
        """Create embeddings with optional positional encoding."""
        # Get base embeddings (projected to embed_dim)
        base_embeds = super(TemporalAttentionGPCM, self).create_embeddings(questions, responses)
        
        if self.enable_positional:
            # Add positional encoding
            position_encoded = self.positional_encoding(base_embeds)
        else:
            position_encoded = base_embeds
            
        return position_encoded
    
    def process_embeddings(self, gpcm_embeds: torch.Tensor, q_embeds: torch.Tensor,
                          responses: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply attention refinement with optional temporal features."""
        batch_size, seq_len = gpcm_embeds.shape[:2]
        
        # Apply standard attention refinement
        if self.use_ordinal_attention and responses is not None:
            refined_embeds = self.attention_refinement(gpcm_embeds, responses)
        else:
            refined_embeds = self.attention_refinement(gpcm_embeds)
        
        if not self.enable_temporal:
            # Skip temporal processing entirely
            return refined_embeds
            
        # Add temporal features to each timestep
        enhanced_embeds = []
        for t in range(seq_len):
            if responses is not None and hasattr(self, '_current_questions'):
                temporal_features_t = self.temporal_extractor.extract_features(
                    self._current_questions, responses, t
                )
                
                # Add noise for diagnostic purposes
                if self.temporal_noise_std > 0:
                    noise = torch.randn_like(temporal_features_t) * self.temporal_noise_std
                    temporal_features_t = temporal_features_t + noise
                    
            else:
                # Create zero temporal features
                temporal_features_t = torch.zeros(batch_size, self.temporal_dim, device=gpcm_embeds.device)
            
            # Fuse attention-refined embedding with temporal features
            enhanced_embed_t = self.feature_fusion(
                refined_embeds[:, t, :],  # [B, embed_dim]
                temporal_features_t       # [B, temporal_dim] 
            )
            enhanced_embeds.append(enhanced_embed_t)
        
        # Stack enhanced embeddings
        enhanced_embeds = torch.stack(enhanced_embeds, dim=1)
        return enhanced_embeds


def analyze_gradient_flow(model, questions, responses, targets, mask=None):
    """Analyze gradient magnitudes across model components."""
    model.train()
    model.zero_grad()
    
    # Forward pass
    outputs = model(questions, responses)
    
    # Handle different output formats
    if isinstance(outputs, tuple):
        # For GPCM models that return (theta, beta, alpha, probs)
        logits = outputs[-1]  # Last output is probabilities
    else:
        logits = outputs
    
    # Convert probabilities to logits if needed (for cross entropy)
    if torch.all(logits >= 0) and torch.all(logits <= 1) and torch.allclose(logits.sum(-1), torch.ones(logits.shape[:-1]), atol=1e-6):
        # This looks like probabilities, convert to logits
        logits = torch.log(logits + 1e-8)
    
    # Flatten tensors for loss computation, handling mismatched batch dimensions
    batch_size, seq_len = questions.shape
    
    # Debug tensor shapes
    print(f"  DEBUG: logits shape: {logits.shape}, questions shape: {questions.shape}")
    
    # Handle different logits dimensions
    if len(logits.shape) == 2:
        # Logits is [batch * seq, categories]
        logits_flat = logits
        targets_flat = targets.view(-1)
    else:
        # Logits is [batch, seq, categories] 
        logits_flat = logits[:, :seq_len, :].reshape(-1, logits.size(-1))  # Match sequence length
        targets_flat = targets[:, :seq_len].reshape(-1)  # Match sequence length
    
    # Apply mask if provided
    if mask is not None:
        mask_flat = mask[:, :seq_len].reshape(-1)
        logits_flat = logits_flat[mask_flat]
        targets_flat = targets_flat[mask_flat]
    
    loss = F.cross_entropy(logits_flat, targets_flat)
    
    # Backward pass
    loss.backward()
    
    # Collect gradient norms
    gradient_norms = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            gradient_norms[name] = param.grad.norm().item()
        else:
            gradient_norms[name] = 0.0
    
    return gradient_norms, loss.item()


def analyze_activation_stats(model, questions, responses):
    """Analyze activation statistics during forward pass."""
    model.eval()
    
    activation_stats = {}
    
    # Hook to capture activations
    def capture_activations(name):
        def hook(module, input, output):
            if isinstance(output, torch.Tensor):
                activation_stats[name] = {
                    'mean': output.mean().item(),
                    'std': output.std().item(),
                    'max': output.max().item(),
                    'min': output.min().item(),
                    'norm': output.norm().item()
                }
        return hook
    
    # Register hooks
    hooks = []
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.MultiheadAttention)):
            hooks.append(module.register_forward_hook(capture_activations(name)))
    
    # Forward pass
    with torch.no_grad():
        _ = model(questions, responses)
    
    # Remove hooks
    for hook in hooks:
        hook.remove()
    
    return activation_stats


def compare_model_behaviors():
    """Compare temporal vs non-temporal model behaviors."""
    print("=" * 60)
    print("TEMPORAL ATTENTION GPCM DIAGNOSTIC ANALYSIS")
    print("=" * 60)
    
    # Load synthetic_OC data for quick testing
    data_path = Path("data/synthetic_OC")
    if not data_path.exists():
        print(f"Data path {data_path} not found. Skipping analysis.")
        return
    
    try:
        # Load data
        print("Loading data...")
        train_loader, test_loader, n_questions, n_cats = load_dataset(
            "synthetic_OC", batch_size=16
        )
        
        # Get a small batch for analysis
        for batch in train_loader:
            if len(batch) == 3:
                questions, responses, mask = batch
            else:
                questions, responses = batch[:2]
                mask = None
            targets = responses  # For diagnostic purposes
            break
        
        print(f"Dataset: {n_questions} questions, {n_cats} categories")
        print(f"Batch shape: {questions.shape}")
        
        # Create diagnostic models
        print("\nCreating diagnostic models...")
        
        # 1. Baseline AttentionGPCM (no temporal)
        baseline_model = create_model("attn_gpcm_linear", n_questions, n_cats)
        
        # 2. Full TemporalAttentionGPCM 
        full_temporal = DiagnosticTemporalGPCM(
            n_questions, n_cats, enable_temporal=True, enable_positional=True
        )
        
        # 3. Only positional (no temporal features)
        positional_only = DiagnosticTemporalGPCM(
            n_questions, n_cats, enable_temporal=False, enable_positional=True
        )
        
        # 4. Only temporal (no positional)
        temporal_only = DiagnosticTemporalGPCM(
            n_questions, n_cats, enable_temporal=True, enable_positional=False
        )
        
        # 5. Neither (pure attention)
        pure_attention = DiagnosticTemporalGPCM(
            n_questions, n_cats, enable_temporal=False, enable_positional=False
        )
        
        models = {
            "Baseline (AttentionGPCM)": baseline_model,
            "Full Temporal": full_temporal,
            "Positional Only": positional_only, 
            "Temporal Only": temporal_only,
            "Pure Attention": pure_attention
        }
        
        print("\nAnalyzing model behaviors...")
        print("-" * 60)
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\n{model_name}:")
            
            try:
                # Forward pass analysis
                model.eval()
                with torch.no_grad():
                    if hasattr(model, 'forward'):
                        outputs = model(questions, responses)
                        if isinstance(outputs, tuple):
                            logits = outputs[-1]  # Last output is probabilities
                        else:
                            logits = outputs
                    else:
                        print(f"  ERROR: Model has no forward method")
                        continue
                
                # Calculate basic metrics
                probs = F.softmax(logits, dim=-1)
                entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
                confidence = probs.max(dim=-1)[0].mean()
                
                print(f"  Output shape: {logits.shape}")
                print(f"  Mean entropy: {entropy:.4f}")
                print(f"  Mean confidence: {confidence:.4f}")
                print(f"  Logit range: [{logits.min():.3f}, {logits.max():.3f}]")
                
                # Gradient analysis
                grad_norms, loss = analyze_gradient_flow(model, questions, responses, targets, mask)
                total_grad_norm = sum(grad_norms.values())
                
                print(f"  Loss: {loss:.4f}")
                print(f"  Total grad norm: {total_grad_norm:.4f}")
                
                # Find largest gradient components
                sorted_grads = sorted(grad_norms.items(), key=lambda x: x[1], reverse=True)[:3]
                print(f"  Top gradients:")
                for name, norm in sorted_grads:
                    print(f"    {name}: {norm:.4f}")
                
                # Activation analysis
                activation_stats = analyze_activation_stats(model, questions, responses)
                
                # Focus on key components
                key_components = ['temporal_extractor', 'feature_fusion', 'positional_encoding',
                                'attention_refinement', 'gpcm_value_embed']
                
                print(f"  Key activation stats:")
                for comp in key_components:
                    matching = [name for name in activation_stats.keys() if comp in name]
                    if matching:
                        stats = activation_stats[matching[0]]
                        print(f"    {comp}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")
                
                results[model_name] = {
                    'entropy': entropy.item(),
                    'confidence': confidence.item(), 
                    'loss': loss,
                    'total_grad_norm': total_grad_norm,
                    'logit_range': (logits.min().item(), logits.max().item())
                }
                
            except Exception as e:
                print(f"  ERROR: {str(e)}")
                results[model_name] = None
        
        # Summary comparison
        print("\n" + "=" * 60)
        print("SUMMARY COMPARISON")
        print("=" * 60)
        
        print(f"{'Model':<20} {'Loss':<8} {'Entropy':<8} {'Confidence':<10} {'Grad Norm':<10}")
        print("-" * 60)
        
        for model_name, result in results.items():
            if result:
                print(f"{model_name:<20} {result['loss']:<8.4f} {result['entropy']:<8.4f} "
                      f"{result['confidence']:<10.4f} {result['total_grad_norm']:<10.4f}")
        
        # Analysis insights
        print("\n" + "=" * 60)
        print("DIAGNOSTIC INSIGHTS")
        print("=" * 60)
        
        if results.get("Baseline (AttentionGPCM)") and results.get("Full Temporal"):
            baseline = results["Baseline (AttentionGPCM)"]
            temporal = results["Full Temporal"]
            
            loss_diff = temporal['loss'] - baseline['loss']
            grad_diff = temporal['total_grad_norm'] - baseline['total_grad_norm']
            
            print(f"1. Loss difference (Temporal - Baseline): {loss_diff:+.4f}")
            print(f"2. Gradient norm difference: {grad_diff:+.4f}")
            
            if loss_diff > 0.1:
                print("   → Temporal features significantly increase loss")
            if grad_diff > 1.0:
                print("   → Temporal features cause much larger gradients")
            
        # Component analysis
        if results.get("Positional Only") and results.get("Temporal Only"):
            pos_only = results["Positional Only"] 
            temp_only = results["Temporal Only"]
            
            print(f"3. Positional encoding impact: {pos_only['loss']:.4f}")
            print(f"4. Temporal features impact: {temp_only['loss']:.4f}")
            
            if temp_only['loss'] > pos_only['loss']:
                print("   → Temporal features are the primary issue, not positional encoding")
            else:
                print("   → Positional encoding is the primary issue, not temporal features")
        
    except Exception as e:
        print(f"Analysis failed: {str(e)}")
        import traceback
        traceback.print_exc()


def test_batch_size_sensitivity():
    """Test how different batch sizes affect temporal model stability."""
    print("\n" + "=" * 60)
    print("BATCH SIZE SENSITIVITY ANALYSIS")
    print("=" * 60)
    
    try:
        # Load data with different batch sizes
        batch_sizes = [8, 16, 32, 64]
        results = {}
        
        for batch_size in batch_sizes:
            print(f"\nTesting batch size: {batch_size}")
            
            # Load data
            train_loader, _, n_questions, n_cats = load_dataset(
                "synthetic_OC", batch_size=batch_size
            )
            
            # Get a batch
            for batch in train_loader:
                if len(batch) == 3:
                    questions, responses, mask = batch
                else:
                    questions, responses = batch[:2]
                    mask = None
                targets = responses
                break
            
            # Create temporal model
            model = DiagnosticTemporalGPCM(n_questions, n_cats)
            
            # Analyze gradients
            grad_norms, loss = analyze_gradient_flow(model, questions, responses, targets, mask)
            total_grad_norm = sum(grad_norms.values())
            
            # Calculate gradient variance across batch
            model.train()
            individual_grads = []
            
            for i in range(min(batch_size, 8)):  # Sample up to 8 individual items
                model.zero_grad()
                outputs = model(questions[i:i+1], responses[i:i+1])
                
                # Handle different output formats
                if isinstance(outputs, tuple):
                    logits = outputs[-1]  # Last output is probabilities
                else:
                    logits = outputs
                
                # Convert probabilities to logits if needed
                if torch.all(logits >= 0) and torch.all(logits <= 1) and torch.allclose(logits.sum(-1), torch.ones(logits.shape[:-1]), atol=1e-6):
                    logits = torch.log(logits + 1e-8)
                
                # Flatten properly for single item
                batch_size_i, seq_len_i = questions[i:i+1].shape
                logits_flat = logits[:, :seq_len_i, :].reshape(-1, logits.size(-1))
                targets_flat = targets[i:i+1, :seq_len_i].reshape(-1)
                
                loss_i = F.cross_entropy(logits_flat, targets_flat)
                loss_i.backward()
                
                grad_norm_i = sum(p.grad.norm().item() for p in model.parameters() if p.grad is not None)
                individual_grads.append(grad_norm_i)
            
            grad_variance = np.var(individual_grads) if individual_grads else 0.0
            
            results[batch_size] = {
                'loss': loss,
                'total_grad_norm': total_grad_norm,
                'grad_variance': grad_variance,
                'individual_grads': individual_grads
            }
            
            print(f"  Loss: {loss:.4f}")
            print(f"  Total grad norm: {total_grad_norm:.4f}")
            print(f"  Gradient variance: {grad_variance:.4f}")
            print(f"  Individual grad norms: {[f'{g:.3f}' for g in individual_grads[:4]]}...")
        
        # Analyze batch size effects
        print(f"\n{'Batch Size':<12} {'Loss':<8} {'Grad Norm':<12} {'Grad Variance':<15}")
        print("-" * 50)
        
        for batch_size, result in results.items():
            print(f"{batch_size:<12} {result['loss']:<8.4f} {result['total_grad_norm']:<12.4f} "
                  f"{result['grad_variance']:<15.4f}")
        
        # Find trends
        batch_sizes_list = list(results.keys())
        grad_norms_list = [results[bs]['total_grad_norm'] for bs in batch_sizes_list]
        grad_vars_list = [results[bs]['grad_variance'] for bs in batch_sizes_list]
        
        if len(grad_norms_list) > 1:
            if grad_norms_list[-1] < grad_norms_list[0] * 0.5:
                print("\n→ Gradient norms decrease significantly with larger batch sizes")
            if grad_vars_list[-1] < grad_vars_list[0] * 0.5:
                print("→ Gradient variance decreases with larger batch sizes")
                print("  This suggests temporal features create noisy gradients that need averaging")
        
    except Exception as e:
        print(f"Batch size analysis failed: {str(e)}")


if __name__ == "__main__":
    print("Starting TemporalAttentionGPCM performance investigation...")
    compare_model_behaviors()
    test_batch_size_sensitivity()
    print("\nDiagnostic analysis complete!")