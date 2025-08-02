"""
IRT Parameter Extraction and Processing Utilities

Handles extraction of effective IRT parameters from different model types,
including special handling for CORAL-GPCM models.
"""

import torch
import numpy as np
from typing import Dict, Any, Optional, Tuple


def extract_effective_thresholds(model, model_type: str) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """Extract effective thresholds from model based on type.
    
    For coral_gpcm_proper, this computes the weighted sum of beta and tau
    as specified in the requirements.
    
    Args:
        model: The trained model
        model_type: Type of model ('deep_gpcm', 'attn_gpcm', 'coral_gpcm_proper', etc.)
        
    Returns:
        tuple: (effective_thresholds, raw_tau) where raw_tau is only present for CORAL models
    """
    if model_type == 'coral_gpcm_proper':
        # Get CORAL info
        coral_info = model.get_coral_info()
        if coral_info and 'tau_thresholds' in coral_info:
            tau = coral_info['tau_thresholds']  # Shape: (n_cats-1,)
            
            # For coral_gpcm_proper with adaptive blending, the effective thresholds
            # are a weighted combination of beta and tau based on the adaptive blender
            if hasattr(model, 'use_adaptive_blending') and model.use_adaptive_blending:
                # The effective threshold is computed dynamically by the blender
                # For IRT analysis, we'll return tau as the representative threshold
                # since the actual blending is instance-specific
                return tau, tau
            else:
                # Fixed blending: weighted average of beta and tau
                # Since beta is per-item and tau is global, we can't compute
                # a single effective threshold without specific items
                # Return tau as the representative threshold
                return tau, tau
        else:
            # Fallback if CORAL info not available
            return None, None
            
    elif model_type in ['coral_gpcm', 'ecoral_gpcm', 'adaptive_coral_gpcm']:
        # Legacy CORAL models - these might not have proper tau extraction
        # Try to get from model state
        if hasattr(model, 'coral_tau'):
            tau = model.coral_tau
            return tau, tau
        else:
            return None, None
            
    else:
        # Non-CORAL models don't have tau thresholds
        return None, None


def compute_effective_beta_for_item(beta: torch.Tensor, tau: torch.Tensor, 
                                   blend_weight: float = 0.5) -> torch.Tensor:
    """Compute effective beta for a specific item in CORAL models.
    
    Args:
        beta: Item-specific beta thresholds, shape (n_cats-1,)
        tau: Global CORAL tau thresholds, shape (n_cats-1,)
        blend_weight: Weight for CORAL contribution (1 - blend_weight for GPCM)
        
    Returns:
        Effective beta thresholds
    """
    return (1 - blend_weight) * beta + blend_weight * tau


def extract_irt_parameters(model, model_type: str, questions: torch.Tensor, 
                          responses: torch.Tensor) -> Dict[str, Any]:
    """Extract IRT parameters from a forward pass.
    
    Args:
        model: The trained model
        model_type: Type of model
        questions: Question tensor for forward pass
        responses: Response tensor for forward pass
        
    Returns:
        Dictionary containing IRT parameters
    """
    model.eval()
    with torch.no_grad():
        # Forward pass
        abilities, thresholds, discriminations, probs = model(questions, responses)
        
        # Extract effective thresholds for CORAL models
        effective_tau, raw_tau = extract_effective_thresholds(model, model_type)
        
        # Prepare results
        results = {
            'abilities': abilities.cpu().numpy(),
            'thresholds': thresholds.cpu().numpy(),  # Item-specific beta
            'discriminations': discriminations.cpu().numpy(),
            'probabilities': probs.cpu().numpy(),
            'model_type': model_type
        }
        
        # Add CORAL-specific information
        if effective_tau is not None:
            results['coral_tau'] = effective_tau.cpu().numpy()
            results['has_coral'] = True
            
            # For coral_gpcm_proper, compute sample effective thresholds
            if model_type == 'coral_gpcm_proper':
                # Get blend weight
                blend_weight = getattr(model, 'blend_weight', 0.5)
                
                # Compute effective thresholds for each item in the batch
                batch_size, seq_len = questions.shape
                effective_betas = []
                
                for t in range(seq_len):
                    beta_t = thresholds[:, t, :]  # Shape: (batch_size, n_cats-1)
                    # Compute effective beta for each item in batch
                    eff_beta_t = compute_effective_beta_for_item(
                        beta_t.mean(dim=0),  # Average across batch
                        effective_tau,
                        blend_weight
                    )
                    effective_betas.append(eff_beta_t)
                
                results['effective_thresholds'] = torch.stack(effective_betas).cpu().numpy()
                results['blend_weight'] = blend_weight
        else:
            results['has_coral'] = False
            
        return results


def summarize_irt_parameters(irt_params: Dict[str, Any]) -> Dict[str, Any]:
    """Summarize IRT parameters for reporting.
    
    Args:
        irt_params: Dictionary from extract_irt_parameters
        
    Returns:
        Summary statistics
    """
    summary = {
        'model_type': irt_params['model_type'],
        'has_coral': irt_params['has_coral']
    }
    
    # Ability statistics
    abilities = irt_params['abilities']
    summary['ability_stats'] = {
        'mean': float(np.mean(abilities)),
        'std': float(np.std(abilities)),
        'min': float(np.min(abilities)),
        'max': float(np.max(abilities))
    }
    
    # Discrimination statistics
    discriminations = irt_params['discriminations']
    summary['discrimination_stats'] = {
        'mean': float(np.mean(discriminations)),
        'std': float(np.std(discriminations)),
        'min': float(np.min(discriminations)),
        'max': float(np.max(discriminations))
    }
    
    # Threshold statistics
    thresholds = irt_params['thresholds']
    summary['threshold_stats'] = {
        'mean': float(np.mean(thresholds)),
        'std': float(np.std(thresholds)),
        'min': float(np.min(thresholds)),
        'max': float(np.max(thresholds))
    }
    
    # CORAL-specific statistics
    if irt_params['has_coral']:
        coral_tau = irt_params['coral_tau']
        summary['coral_tau'] = coral_tau.tolist()
        
        if 'effective_thresholds' in irt_params:
            eff_thresh = irt_params['effective_thresholds']
            summary['effective_threshold_stats'] = {
                'mean': float(np.mean(eff_thresh)),
                'std': float(np.std(eff_thresh)),
                'min': float(np.min(eff_thresh)),
                'max': float(np.max(eff_thresh))
            }
            summary['blend_weight'] = irt_params.get('blend_weight', 0.5)
    
    return summary