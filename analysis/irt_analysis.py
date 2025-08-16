#!/usr/bin/env python3
"""
Unified IRT Analysis Tool for Deep-GPCM
Combines all IRT-related functionality into a single script.
"""

import os
import sys
import torch
import torch.nn.functional as F
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Set backend for consistent rendering
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import seaborn as sns
import argparse
from pathlib import Path
from glob import glob
from scipy import stats
from matplotlib.patches import Ellipse
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.implementations import DeepGPCM, AttentionGPCM, EnhancedAttentionGPCM
from models.implementations.coral_gpcm_proper import CORALGPCM
from models.implementations.coral_gpcm_fixed import CORALGPCMFixed
from train import load_simple_data, create_data_loaders
from utils.irt_utils import extract_effective_thresholds, extract_irt_parameters, summarize_irt_parameters
from utils.path_utils import get_path_manager, find_best_model


class UnifiedIRTAnalyzer:
    """Unified IRT analysis tool with all functionality."""
    
    def __init__(self, dataset='synthetic_OC', output_dir=None):
        from utils.path_utils import get_plot_path
        self.dataset = dataset
        # Create dataset-specific output directory using new structure
        if output_dir is None:
            self.output_dir = get_plot_path(dataset, 'irt')
        else:
            self.output_dir = Path(output_dir) / dataset
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Import model registry for current model names and colors
        from models.factory import get_all_model_types, get_model_color
        
        # Get current model types and colors from factory
        current_model_types = get_all_model_types()
        self.model_colors = {model_type: get_model_color(model_type) for model_type in current_model_types}
        
        # Load data - support both old and new naming formats
        data_dir = Path('data') / dataset
        if dataset.startswith('synthetic_') and '_' in dataset[10:]:
            # New format: synthetic_4000_200_2
            train_path = data_dir / f'{dataset}_train.txt'
            test_path = data_dir / f'{dataset}_test.txt'
        else:
            # Legacy format: synthetic_OC -> synthetic_oc_train.txt
            train_path = data_dir / f'{dataset.lower()}_train.txt'
            test_path = data_dir / f'{dataset.lower()}_test.txt'
        
        self.train_data, self.test_data, self.n_questions, self.n_cats = load_simple_data(
            str(train_path), str(test_path)
        )
        
        # Load true parameters if available
        self.true_params = self._load_true_parameters(data_dir)
        
    def _load_true_parameters(self, dataset_path):
        """Load true IRT parameters from synthetic dataset."""
        true_params_path = dataset_path / 'true_irt_parameters.json'
        
        if not true_params_path.exists():
            print(f"Warning: No true parameters found at {true_params_path}")
            return None
        
        with open(true_params_path, 'r') as f:
            return json.load(f)
    
    def get_model_color(self, model_name: str) -> str:
        """Get consistent color for a model using factory-defined colors."""
        # First try to get color from model factory
        try:
            from models.factory import get_model_color as factory_get_color
            return factory_get_color(model_name)
        except:
            pass
        
        # Fallback to local mapping
        if model_name in self.model_colors:
            return self.model_colors[model_name]
        else:
            # Use tab10 colors for unknown models
            colors = plt.cm.tab10.colors if hasattr(plt.cm.tab10, 'colors') else plt.cm.tab10(range(10))
            base_idx = len(self.model_colors) % 10
            return colors[base_idx]
    
    def find_models(self):
        """Automatically find all trained models for the dataset."""
        path_manager = get_path_manager()
        
        # Find models using both current factory registry and legacy models
        model_files = []
        
        # Use current model types from factory registry
        from models.factory import get_all_model_types
        current_models = get_all_model_types()
        
        # Add legacy model types that might exist in saved models
        legacy_models = ['coral_gpcm_proper', 'coral_gpcm_fixed', 'test_gpcm', 
                        'attn_gpcm_new', 'modular_attn_gpcm', 'attention', 'ordinal_attn_gpcm']
        
        all_model_names = current_models + legacy_models
        
        for model_name in all_model_names:
            model_path = find_best_model(model_name, self.dataset)
            if model_path:
                model_files.append(str(model_path))
        
        # Filter out fold-specific models
        model_files = [f for f in model_files if not any(f'fold_{i}' in f for i in range(1, 10))]
        
        models = {}
        for model_path in model_files:
            # Extract model name from path
            base_name = os.path.basename(model_path)
            model_name = base_name.replace(f'best_', '').replace(f'_{self.dataset}.pth', '').replace('.pth', '')
            models[model_name] = model_path
            
        print(f"Found {len(models)} models: {list(models.keys())}")
        return models
    
    def load_model(self, model_path):
        """Load a model with proper handling for different architectures."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Import factory functions at the top for consistent access
        from models.factory import get_all_model_types, create_model, get_model_default_params
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Check for different model types
        has_learnable_params = any('learnable_ability_scale' in key or 'learnable_embedding' in key 
                                  for key in checkpoint['model_state_dict'].keys())
        has_memory_fusion = any('memory_fusion' in key or 'refinement_gates' in key 
                               for key in checkpoint['model_state_dict'].keys())
        has_coral_projection = any('coral_projection' in key for key in checkpoint['model_state_dict'].keys())
        has_coral_layer = any('coral_layer' in key for key in checkpoint['model_state_dict'].keys())
        
        # Determine model type from config or state dict
        filename = os.path.basename(model_path)
        if 'modular_attn_gpcm' in filename:
            model_type = 'modular_attn_gpcm'
        elif 'config' in checkpoint:
            model_type = checkpoint['config'].get('model_type', 'baseline')
        elif 'model_config' in checkpoint:
            # Handle model_config key (used by coral_gpcm_proper)
            config = checkpoint['model_config']
            # Try to infer model type from filename
            filename = os.path.basename(model_path)
            if 'coral_gpcm_proper' in filename:
                model_type = 'coral_gpcm_proper'
            elif 'modular_attn_gpcm' in filename:
                model_type = 'modular_attn_gpcm'
            else:
                model_type = 'baseline'
        else:
            if has_coral_projection:
                model_type = 'coral_gpcm'  # Updated to match new naming
            elif has_coral_layer:
                model_type = 'coral'  # Skip pure CORAL models as they don't have IRT params
            elif 'attention_refinement' in str(checkpoint['model_state_dict'].keys()):
                model_type = 'attention'
            else:
                model_type = 'deep_gpcm'  # Updated to match new naming
        
        # Skip old coral models as they don't exist anymore
        if model_type == 'coral':
            raise ValueError(f"Deprecated coral model - skipping")
        
        # Create model using factory system for consistency
        if has_learnable_params or ('attn_gpcm' in str(model_path) and 'modular_attn_gpcm' not in str(model_path)):
            # Try to detect specific attn_gpcm model type from filename
            
            # Get available models and sort by length to match longest first
            available_models = get_all_model_types()
            detected_type = None
            filename = os.path.basename(model_path)
            
            # Try to detect model type using factory registry
            for model_type_candidate in sorted(available_models, key=len, reverse=True):
                if model_type_candidate in filename:
                    detected_type = model_type_candidate
                    break
            
            if detected_type is None:
                # Fallback to generic attn_gpcm for backward compatibility
                detected_type = 'attn_gpcm'
            
            # Use factory to create model with proper configuration
            try:
                # Get saved configuration if available
                saved_config = checkpoint.get('config', {})
                factory_params = get_model_default_params(detected_type)
                
                # Create model_kwargs from saved config and factory defaults
                model_kwargs = {
                    'memory_size': saved_config.get('memory_size', 50),
                    'key_dim': saved_config.get('key_dim', 50), 
                    'value_dim': saved_config.get('value_dim', 200),
                    'final_fc_dim': saved_config.get('final_fc_dim', 50),
                }
                
                # Apply factory defaults
                model_kwargs.update(factory_params)
                
                # Override with saved config (highest priority)
                for key, value in saved_config.items():
                    if key in factory_params:
                        model_kwargs[key] = value
                
                model = create_model(detected_type, self.n_questions, self.n_cats, **model_kwargs)
                model_type = detected_type
                print(f"📍 IRT Analysis using factory model: {detected_type}")
                
            except Exception as e:
                print(f"⚠️ Factory creation failed for {detected_type}, falling back to hardcoded config: {e}")
                # Fallback to original hardcoded approach
                use_learnable_embedding = 'attn_gpcm_fixed' not in str(model_path) and 'attn_gpcm_linear' not in str(model_path)
                model = EnhancedAttentionGPCM(
                    n_questions=self.n_questions, 
                    n_cats=self.n_cats,
                    memory_size=50,
                    key_dim=50,
                    value_dim=200,
                    final_fc_dim=50,
                    n_heads=4,
                    n_cycles=2,
                    embedding_strategy="linear_decay",
                    ability_scale=2.0,
                    use_learnable_embedding=use_learnable_embedding
                )
                model_type = 'attn_gpcm'
        elif model_type == 'coral_gpcm_proper':
            # Use factory system for coral_gpcm_proper
            try:
                # Get saved configuration if available
                saved_config = checkpoint.get('config', {})
                factory_params = get_model_default_params(model_type)
                
                # Create model_kwargs from saved config and factory defaults
                model_kwargs = {
                    'memory_size': saved_config.get('memory_size', 50),
                    'key_dim': saved_config.get('key_dim', 50), 
                    'value_dim': saved_config.get('value_dim', 200),
                    'final_fc_dim': saved_config.get('final_fc_dim', 50),
                }
                
                # Apply factory defaults
                model_kwargs.update(factory_params)
                
                # Override with saved config (highest priority)
                for key, value in saved_config.items():
                    if key in factory_params:
                        model_kwargs[key] = value
                
                model = create_model(model_type, self.n_questions, self.n_cats, **model_kwargs)
                print(f"📍 IRT Analysis using factory model: {model_type}")
                
            except Exception as e:
                print(f"⚠️ Factory creation failed for {model_type}, falling back to hardcoded config: {e}")
                # Fallback to original hardcoded approach
                model = CORALGPCM(
                    n_questions=self.n_questions,
                    n_cats=self.n_cats,
                    memory_size=50,
                    key_dim=50,
                    value_dim=200,
                    final_fc_dim=50,
                    use_adaptive_blending=True
                )
        elif model_type == 'coral_gpcm_fixed':
            # Use factory system for coral_gpcm_fixed
            try:
                # Get saved configuration if available
                saved_config = checkpoint.get('config', {})
                factory_params = get_model_default_params(model_type)
                
                # Create model_kwargs from saved config and factory defaults
                model_kwargs = {
                    'memory_size': saved_config.get('memory_size', 50),
                    'key_dim': saved_config.get('key_dim', 50), 
                    'value_dim': saved_config.get('value_dim', 200),
                    'final_fc_dim': saved_config.get('final_fc_dim', 50),
                }
                
                # Apply factory defaults
                model_kwargs.update(factory_params)
                
                # Override with saved config (highest priority)
                for key, value in saved_config.items():
                    if key in factory_params:
                        model_kwargs[key] = value
                
                model = create_model(model_type, self.n_questions, self.n_cats, **model_kwargs)
                print(f"📍 IRT Analysis using factory model: {model_type}")
                
            except Exception as e:
                print(f"⚠️ Factory creation failed for {model_type}, falling back to hardcoded config: {e}")
                # Fallback to original hardcoded approach
                model = CORALGPCMFixed(
                    n_questions=self.n_questions,
                    n_cats=self.n_cats,
                    memory_size=50,
                    key_dim=50,
                    value_dim=200,
                    final_fc_dim=50
                )
        elif model_type == 'test_gpcm' and False:  # Disabled - TestGPCM not imported
            # Use factory system for test_gpcm
            try:
                # Get saved configuration if available
                saved_config = checkpoint.get('config', {})
                factory_params = get_model_default_params(model_type)
                
                # Create model_kwargs from saved config and factory defaults
                model_kwargs = {
                    'memory_size': saved_config.get('memory_size', 50),
                    'key_dim': saved_config.get('key_dim', 50), 
                    'value_dim': saved_config.get('value_dim', 200),
                    'final_fc_dim': saved_config.get('final_fc_dim', 50),
                }
                
                # Apply factory defaults
                model_kwargs.update(factory_params)
                
                # Override with saved config (highest priority)
                for key, value in saved_config.items():
                    if key in factory_params:
                        model_kwargs[key] = value
                
                model = create_model(model_type, self.n_questions, self.n_cats, **model_kwargs)
                print(f"📍 IRT Analysis using factory model: {model_type}")
                
            except Exception as e:
                print(f"⚠️ Factory creation failed for {model_type}, falling back: {e}")
                # Skip if no fallback available
                raise ValueError(f"Cannot create model {model_type} - no factory implementation")
        elif model_type == 'attn_gpcm_new' and False:  # Disabled - AttentionGPCMNew not imported
            # Use factory system for attn_gpcm_new
            try:
                # Get saved configuration if available
                saved_config = checkpoint.get('config', {})
                factory_params = get_model_default_params(model_type)
                
                # Create model_kwargs from saved config and factory defaults
                model_kwargs = {
                    'memory_size': saved_config.get('memory_size', 50),
                    'key_dim': saved_config.get('key_dim', 50), 
                    'value_dim': saved_config.get('value_dim', 200),
                    'final_fc_dim': saved_config.get('final_fc_dim', 50),
                    'embed_dim': saved_config.get('embed_dim', 64),
                    'n_heads': saved_config.get('n_heads', 4),
                    'n_cycles': saved_config.get('n_cycles', 2),
                    'ability_scale': saved_config.get('ability_scale', 2.0),
                }
                
                # Apply factory defaults
                model_kwargs.update(factory_params)
                
                # Override with saved config (highest priority)
                for key, value in saved_config.items():
                    if key in factory_params:
                        model_kwargs[key] = value
                
                model = create_model(model_type, self.n_questions, self.n_cats, **model_kwargs)
                print(f"📍 IRT Analysis using factory model: {model_type}")
                
            except Exception as e:
                print(f"⚠️ Factory creation failed for {model_type}, falling back: {e}")
                # Skip if no fallback available
                raise ValueError(f"Cannot create model {model_type} - no factory implementation")
        elif model_type == 'modular_attn_gpcm' and False:  # Disabled - ModularAttentionGPCM not imported
            # Use factory system for modular_attn_gpcm
            try:
                # Get saved configuration if available
                saved_config = checkpoint.get('config', {})
                factory_params = get_model_default_params(model_type)
                
                # Create model_kwargs from saved config and factory defaults
                model_kwargs = {
                    'memory_size': saved_config.get('memory_size', 50),
                    'key_dim': saved_config.get('key_dim', 50), 
                    'value_dim': saved_config.get('value_dim', 200),
                    'final_fc_dim': saved_config.get('final_fc_dim', 50),
                    'embed_dim': saved_config.get('embed_dim', 64),
                }
                
                # Apply factory defaults
                model_kwargs.update(factory_params)
                
                # Override with saved config (highest priority)
                for key, value in saved_config.items():
                    if key in factory_params:
                        model_kwargs[key] = value
                
                model = create_model(model_type, self.n_questions, self.n_cats, **model_kwargs)
                print(f"📍 IRT Analysis using factory model: {model_type}")
                
            except Exception as e:
                print(f"⚠️ Factory creation failed for {model_type}, falling back to hardcoded config: {e}")
                # Fallback to original hardcoded approach
                from models.components.attention_mechanisms import AttentionConfig
                attention_config = AttentionConfig(
                    embed_dim=64,
                    n_heads=4,
                    n_cats=self.n_cats,
                    dropout=0.1
                )
                model = ModularAttentionGPCM(
                    n_questions=self.n_questions,
                    n_cats=self.n_cats,
                    embed_dim=64,
                    memory_size=50,
                    key_dim=50,
                    value_dim=200,
                    final_fc_dim=50,
                    attention_config=attention_config,
                    attention_mechanisms=['ordinal_aware', 'response_conditioned', 'qwk_aligned'],
                    fusion_method='concat'
                )
        elif model_type == 'attention' or 'attention' in str(checkpoint['model_state_dict'].keys()):
            # Use factory system for attention models
            try:
                # Get saved configuration if available
                saved_config = checkpoint.get('config', {})
                factory_params = get_model_default_params('attn_gpcm')  # Use attn_gpcm as default for attention
                
                # Create model_kwargs from saved config and factory defaults
                model_kwargs = {
                    'memory_size': saved_config.get('memory_size', 50),
                    'key_dim': saved_config.get('key_dim', 50), 
                    'value_dim': saved_config.get('value_dim', 200),
                    'final_fc_dim': saved_config.get('final_fc_dim', 50),
                }
                
                # Apply factory defaults
                model_kwargs.update(factory_params)
                
                # Override with saved config (highest priority)
                for key, value in saved_config.items():
                    if key in factory_params:
                        model_kwargs[key] = value
                
                model = create_model('attn_gpcm', self.n_questions, self.n_cats, **model_kwargs)
                model_type = 'attn_gpcm'
                print(f"📍 IRT Analysis using factory model: {model_type}")
                
            except Exception as e:
                print(f"⚠️ Factory creation failed for attention model, falling back to hardcoded config: {e}")
                # Fallback to original hardcoded approach
                model = AttentionGPCM(n_questions=self.n_questions, n_cats=self.n_cats)
        else:
            # Use factory system for all other model types
            try:
                # Get saved configuration if available
                saved_config = checkpoint.get('config', {})
                # Try to use detected model type or fallback to deep_gpcm
                fallback_type = model_type if model_type in get_all_model_types() else 'deep_gpcm'
                factory_params = get_model_default_params(fallback_type)
                
                # Create model_kwargs from saved config and factory defaults
                model_kwargs = {
                    'memory_size': saved_config.get('memory_size', 50),
                    'key_dim': saved_config.get('key_dim', 50), 
                    'value_dim': saved_config.get('value_dim', 200),
                    'final_fc_dim': saved_config.get('final_fc_dim', 50),
                }
                
                # Apply factory defaults
                model_kwargs.update(factory_params)
                
                # Override with saved config (highest priority)
                for key, value in saved_config.items():
                    if key in factory_params:
                        model_kwargs[key] = value
                
                model = create_model(fallback_type, self.n_questions, self.n_cats, **model_kwargs)
                model_type = fallback_type
                print(f"📍 IRT Analysis using factory model: {model_type}")
                
            except Exception as e:
                print(f"⚠️ Factory creation failed for {model_type}, falling back to DeepGPCM: {e}")
                # Fallback to original hardcoded approach
                model = DeepGPCM(n_questions=self.n_questions, n_cats=self.n_cats)
        
        # Load weights with proper handling for wrapped models and compatibility
        try:
            state_dict = checkpoint['model_state_dict']
            
            # Handle missing threshold coupling parameters for backward compatibility
            if model_type in ['coral_gpcm', 'ecoral_gpcm', 'adaptive_coral_gpcm', 'full_adaptive_coral_gpcm', 'coral_gpcm_proper']:
                model_state_keys = set(model.state_dict().keys())
                saved_state_keys = set(state_dict.keys())
                
                # Add missing threshold coupling parameters with defaults
                missing_keys = model_state_keys - saved_state_keys
                for key in missing_keys:
                    if 'threshold_gpcm_weight' in key:
                        state_dict[key] = model.state_dict()[key].clone()
                    elif 'threshold_coral_weight' in key:
                        state_dict[key] = model.state_dict()[key].clone()
                
                # Remove unexpected parameters
                unexpected_keys = saved_state_keys - model_state_keys
                for key in list(unexpected_keys):
                    if key in state_dict:
                        del state_dict[key]
            
            model.load_state_dict(state_dict, strict=False)
        except RuntimeError as e:
            # Handle models saved with wrapper
            if 'gpcm_model' in str(e):
                # Extract the wrapped model state dict
                wrapped_state_dict = checkpoint['model_state_dict']
                unwrapped_state_dict = {}
                for key, value in wrapped_state_dict.items():
                    if key.startswith('gpcm_model.'):
                        new_key = key.replace('gpcm_model.', '')
                        unwrapped_state_dict[new_key] = value
                    else:
                        unwrapped_state_dict[key] = value
                model.load_state_dict(unwrapped_state_dict, strict=False)
            else:
                raise e
        
        model.to(device)
        model.eval()
        
        return model, model_type, device
    
    def extract_temporal_parameters(self, model, model_type, device, split='test'):
        """Extract temporal IRT parameters keeping time dimension."""
        # Create data loader
        if split == 'train':
            data_loader, _ = create_data_loaders(self.train_data, self.test_data, batch_size=32)
        else:
            _, data_loader = create_data_loaders(self.train_data, self.test_data, batch_size=32)
        
        # Extract parameters with temporal structure
        temporal_data = {
            'student_abilities': [],  # List of sequences
            'item_discriminations': [],
            'item_thresholds': [],
            'gpcm_probabilities': [],  # GPCM probability predictions
            'question_ids': [],
            'responses': [],
            'masks': [],
            'student_ids': [],  # Track which student each sequence belongs to
            # CORAL-specific parameters
            'coral_taus': None,
            'coral_weights': None,
            'coral_thresholds': None,
            'has_coral': False
        }
        
        student_id = 0
        
        with torch.no_grad():
            for batch_idx, (questions, responses, mask) in enumerate(data_loader):
                questions = questions.to(device)
                responses = responses.to(device)
                batch_size, seq_len = questions.shape
                
                # Forward pass - using EXACT same computational pathway as GPCM probability computation
                # This matches the beta extraction method implemented in extract_beta_params.py
                student_abilities, item_thresholds, discrimination_params, gpcm_probs = model(questions, responses)
                
                # Extract CORAL parameters if available (only need to do this once)
                if batch_idx == 0:
                    # Handle coral_gpcm_proper with special IRT utilities
                    if model_type == 'coral_gpcm_proper':
                        effective_tau, raw_tau = extract_effective_thresholds(model, model_type)
                        if effective_tau is not None:
                            temporal_data['has_coral'] = True
                            temporal_data['coral_taus'] = effective_tau.cpu().numpy()
                            
                            # Get CORAL info for additional details
                            coral_info = model.get_coral_info()
                            if coral_info:
                                temporal_data['coral_integration_type'] = coral_info.get('integration_type', 'unknown')
                                
                            # Extract blend weight if available
                            if hasattr(model, 'blend_weight'):
                                temporal_data['blend_weight'] = model.blend_weight
                                
                    # Legacy CORAL models
                    elif hasattr(model, 'coral_layer'):
                        temporal_data['has_coral'] = True
                        # Extract CORAL τ parameters (biases)
                        temporal_data['coral_taus'] = model.coral_layer.rank_classifier.bias.data.cpu().numpy()
                        # Extract CORAL τ weights
                        temporal_data['coral_weights'] = model.coral_layer.rank_classifier.weight.data.cpu().numpy()
                        # Extract ordinal thresholds if present
                        if hasattr(model.coral_layer, 'ordinal_thresholds'):
                            temporal_data['coral_thresholds'] = model.coral_layer.ordinal_thresholds.data.cpu().numpy()
                        
                        # Extract adaptive blending parameters if present
                        if hasattr(model, 'threshold_blender'):
                            temporal_data['adaptive_params'] = {
                                'range_sensitivity': model.threshold_blender.range_sensitivity.data.cpu().numpy(),
                                'distance_sensitivity': model.threshold_blender.distance_sensitivity.data.cpu().numpy(),
                                'baseline_bias': model.threshold_blender.baseline_bias.data.cpu().numpy()
                            }
                
                # Store temporal sequences for each student
                for i in range(batch_size):
                    seq_mask = mask[i].cpu().numpy()
                    valid_len = int(seq_mask.sum())
                    
                    if valid_len > 0:
                        temporal_data['student_abilities'].append(
                            student_abilities[i, :valid_len].cpu().numpy()
                        )
                        temporal_data['item_discriminations'].append(
                            discrimination_params[i, :valid_len].cpu().numpy()
                        )
                        temporal_data['item_thresholds'].append(
                            item_thresholds[i, :valid_len].cpu().numpy()
                        )
                        temporal_data['gpcm_probabilities'].append(
                            gpcm_probs[i, :valid_len].cpu().numpy()
                        )
                        temporal_data['question_ids'].append(
                            questions[i, :valid_len].cpu().numpy()
                        )
                        temporal_data['responses'].append(
                            responses[i, :valid_len].cpu().numpy()
                        )
                        temporal_data['masks'].append(
                            seq_mask[:valid_len]
                        )
                        temporal_data['student_ids'].append(student_id)
                        student_id += 1
        
        temporal_data['model_type'] = model_type
        temporal_data['n_students'] = student_id
        
        return temporal_data
    
    def extract_aggregated_parameters(self, temporal_data, theta_method='last', item_method='average'):
        """Extract aggregated parameters from temporal data."""
        results = {}
        
        # Extract student abilities
        n_students = temporal_data['n_students']
        student_abilities = np.zeros(n_students)
        student_abilities_avg = np.zeros(n_students)  # Always store average for correlation
        
        for i in range(n_students):
            abilities = temporal_data['student_abilities'][i]
            
            if theta_method == 'last':
                student_abilities[i] = abilities[-1]
            elif theta_method == 'average':
                student_abilities[i] = abilities.mean()
            
            # Always calculate average for correlation purposes
            student_abilities_avg[i] = abilities.mean()
        
        results['student_abilities'] = student_abilities
        results['student_abilities_avg'] = student_abilities_avg  # Store average separately
        
        # Extract item parameters
        max_q_id = 0
        for q_ids in temporal_data['question_ids']:
            max_q_id = max(max_q_id, int(np.max(q_ids)))
        
        n_questions = max_q_id + 1
        n_cats = temporal_data['item_thresholds'][0].shape[-1] + 1
        
        if item_method == 'average':
            # Calculate average parameters for each question
            item_alphas = np.zeros(n_questions)
            item_betas = np.zeros((n_questions, n_cats - 1))
            item_counts = np.zeros(n_questions)
            
            for i in range(len(temporal_data['question_ids'])):
                q_ids = temporal_data['question_ids'][i]
                alphas = temporal_data['item_discriminations'][i]
                betas = temporal_data['item_thresholds'][i]
                
                for j, q_id in enumerate(q_ids):
                    if q_id > 0:  # Skip padding
                        item_alphas[q_id] += alphas[j]
                        item_betas[q_id] += betas[j]
                        item_counts[q_id] += 1
            
            # Average
            for q_id in range(n_questions):
                if item_counts[q_id] > 0:
                    item_alphas[q_id] /= item_counts[q_id]
                    item_betas[q_id] /= item_counts[q_id]
                    
        elif item_method == 'last':
            # Track last occurrence of each question
            item_alphas = np.zeros(n_questions)
            item_betas = np.zeros((n_questions, n_cats - 1))
            item_counts = np.zeros(n_questions)
            item_last_time = np.zeros(n_questions, dtype=int) - 1
            
            global_time = 0
            for i in range(len(temporal_data['question_ids'])):
                q_ids = temporal_data['question_ids'][i]
                alphas = temporal_data['item_discriminations'][i]
                betas = temporal_data['item_thresholds'][i]
                
                for j, q_id in enumerate(q_ids):
                    if q_id > 0:  # Skip padding
                        if global_time > item_last_time[q_id]:
                            item_alphas[q_id] = alphas[j]
                            item_betas[q_id] = betas[j]
                            item_last_time[q_id] = global_time
                    global_time += 1
            
            item_counts = (item_last_time >= 0).astype(float)
        
        results['item_discriminations'] = item_alphas
        results['item_thresholds'] = item_betas
        results['item_counts'] = item_counts
        
        # Handle CORAL-specific parameters
        if temporal_data['has_coral']:
            results['coral_taus'] = temporal_data['coral_taus']
            results['has_coral'] = True
            
            # For coral_gpcm_proper, compute effective thresholds
            if temporal_data['model_type'] == 'coral_gpcm_proper':
                # Compute average effective thresholds using blend weight
                blend_weight = temporal_data.get('blend_weight', 0.5)
                effective_betas = np.zeros_like(item_betas)
                
                # For each question, compute effective threshold
                for q_id in range(n_questions):
                    if item_counts[q_id] > 0:
                        from utils.irt_utils import compute_effective_beta_for_item
                        effective_betas[q_id] = compute_effective_beta_for_item(
                            torch.tensor(item_betas[q_id]),
                            torch.tensor(temporal_data['coral_taus']),
                            blend_weight
                        ).numpy()
                
                results['effective_thresholds'] = effective_betas
                results['blend_weight'] = blend_weight
        else:
            results['has_coral'] = False
        
        return results
    
    def normalize_parameters(self, alphas, betas, thetas=None):
        """Normalize parameters to standard IRT scale."""
        results = {}
        
        # Normalize discriminations (log-normal prior)
        if alphas is not None and len(alphas) > 0:
            results['alphas'] = np.exp((np.log(alphas + 1e-6) - np.mean(np.log(alphas + 1e-6))) / 
                                     np.std(np.log(alphas + 1e-6)) * 0.5 + np.log(1.0))
        
        # Normalize thresholds (normal prior)
        if betas is not None and len(betas) > 0:
            betas_norm = np.zeros_like(betas)
            for k in range(betas.shape[1]):
                betas_norm[:, k] = (betas[:, k] - np.mean(betas[:, k])) / np.std(betas[:, k])
            results['betas'] = betas_norm
        
        # Normalize abilities (normal prior)
        if thetas is not None and len(thetas) > 0:
            results['thetas'] = (thetas - np.mean(thetas)) / np.std(thetas)
        
        return results
    
    def calculate_correlations(self, true_params, learned_params):
        """Calculate correlations between true and learned parameters."""
        results = {}
        
        # Student abilities
        if 'student_abilities' in learned_params and self.true_params:
            true_thetas = np.array(true_params['student_abilities']['theta'])
            # Use final theta for correlation calculation
            learned_thetas = learned_params['student_abilities']
            
            # Only use students that exist in both
            n_students = min(len(learned_thetas), len(true_thetas))
            learned_thetas = learned_thetas[:n_students]
            true_thetas = true_thetas[:n_students]
            
            # Normalize
            norm_true = self.normalize_parameters(None, None, true_thetas)['thetas']
            norm_learned = self.normalize_parameters(None, None, learned_thetas)['thetas']
            
            results['theta_correlation'] = np.corrcoef(norm_true, norm_learned)[0, 1]
            results['n_students'] = n_students
            
            # Store normalized values for plotting
            results['true_thetas_norm'] = norm_true
            results['learned_thetas_norm'] = norm_learned
        
        # Item parameters
        if all(k in learned_params for k in ['item_discriminations', 'item_thresholds', 'item_counts']):
            true_alphas = np.array(true_params['question_params']['discrimination']['alpha'])
            true_betas = np.array(true_params['question_params']['difficulties']['beta'])
            
            # Only consider items seen in data
            valid_items = learned_params['item_counts'] > 0
            valid_indices = np.where(valid_items)[0]
            
            # Normalize parameters
            norm_true_alpha = self.normalize_parameters(true_alphas, None)['alphas']
            norm_true_beta = self.normalize_parameters(None, true_betas)['betas']
            
            norm_learned_alpha = self.normalize_parameters(
                learned_params['item_discriminations'][valid_items], None)['alphas']
            norm_learned_beta = self.normalize_parameters(
                None, learned_params['item_thresholds'][valid_items])['betas']
            
            # Calculate correlations
            results['alpha_correlation'] = np.corrcoef(
                norm_true_alpha[valid_indices], norm_learned_alpha)[0, 1]
            
            beta_corrs = []
            n_thresholds = min(norm_true_beta.shape[1], norm_learned_beta.shape[1])
            for k in range(n_thresholds):
                beta_corrs.append(np.corrcoef(
                    norm_true_beta[valid_indices, k], 
                    norm_learned_beta[:, k])[0, 1])
            
            results['beta_correlations'] = beta_corrs
            results['beta_avg_correlation'] = np.mean(beta_corrs)
            results['n_items'] = len(valid_indices)
            
            # Store normalized values for plotting
            results['true_alphas_norm'] = norm_true_alpha
            results['learned_alphas_norm'] = norm_learned_alpha
            results['true_betas_norm'] = norm_true_beta
            results['learned_betas_norm'] = norm_learned_beta
            results['valid_indices'] = valid_indices
        
        return results
    
    def plot_parameter_recovery(self, model_results, save_path, model_colors=None):
        """Create parameter recovery comparison plots."""
        # Count only models with correlations for plotting
        models_with_corr = [(name, results) for name, results in model_results.items() 
                           if 'correlations' in results]
        n_models = len(models_with_corr)
        
        if n_models == 0:
            print("No models with correlations to plot")
            return
        
        # Determine maximum number of thresholds across all models
        max_thresholds = 0
        for model_name, results in models_with_corr:
            corr = results['correlations']
            if 'beta_correlations' in corr:
                max_thresholds = max(max_thresholds, len(corr['beta_correlations']))
        
        # Calculate number of columns: theta + theta_dist + alpha + thresholds
        n_cols = 3 + max_thresholds  # theta, theta_dist, alpha, then thresholds
        fig, axes = plt.subplots(n_models, n_cols, figsize=(4 * n_cols, 4 * n_models))
        
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        # Determine best model for each parameter type separately
        best_models = {}
        overall_scores = {}
        
        # Calculate overall performance for each model
        for model_name, results in models_with_corr:
            corr = results['correlations']
            correlations = []
            
            # Collect all correlations
            if 'theta_correlation' in corr:
                correlations.append(corr['theta_correlation'])
            if 'alpha_correlation' in corr:
                correlations.append(corr['alpha_correlation'])
            if 'beta_correlations' in corr:
                correlations.extend(corr['beta_correlations'])
            
            # Calculate average correlation as overall score
            if correlations:
                overall_scores[model_name] = np.mean(correlations)
        
        # Find overall best model
        best_overall_model = max(overall_scores, key=overall_scores.get) if overall_scores else None
        
        if len(models_with_corr) > 1:
            # Find best model for each parameter type (dynamic based on actual thresholds)
            parameter_types = ['theta', 'alpha'] + [f'beta_{i}' for i in range(max_thresholds)]
            
            for param_type in parameter_types:
                best_corr = -1
                best_model_for_param = None
                
                for model_name, results in models_with_corr:
                    corr = results['correlations']
                    param_corr = None
                    
                    if param_type == 'theta' and 'theta_correlation' in corr:
                        param_corr = corr['theta_correlation']
                    elif param_type == 'alpha' and 'alpha_correlation' in corr:
                        param_corr = corr['alpha_correlation']
                    elif param_type.startswith('beta_') and 'beta_correlations' in corr:
                        beta_idx = int(param_type.split('_')[1])
                        if beta_idx < len(corr['beta_correlations']):
                            param_corr = corr['beta_correlations'][beta_idx]
                    
                    if param_corr is not None and param_corr > best_corr:
                        best_corr = param_corr
                        best_model_for_param = model_name
                
                best_models[param_type] = best_model_for_param

        for idx, (model_name, results) in enumerate(models_with_corr):
                
            params = results['aggregated_params']
            corr = results['correlations']
            # Use passed model colors if available, otherwise fetch
            if model_colors and model_name in model_colors:
                model_color = model_colors[model_name]
            else:
                model_color = self.get_model_color(model_name)
            
            # Student ability plot
            if 'theta_correlation' in corr:
                ax = axes[idx, 0]
                ax.scatter(results['true_thetas_norm'][:corr['n_students']], 
                          results['learned_thetas_norm'], alpha=0.6, color=model_color)
                ax.plot([-3, 3], [-3, 3], 'k--', linewidth=2, label='Perfect recovery')
                ax.set_xlabel('True θ')
                ax.set_ylabel('Learned θ')
                
                # Show only parameter info, no model name
                title_text = f'Final θ (r={corr["theta_correlation"]:.3f})'
                is_best_theta = best_models.get('theta') == model_name
                if is_best_theta:
                    title_text += ' *'
                    ax.set_title(title_text, color='green', fontweight='bold')
                else:
                    ax.set_title(title_text, color='black', fontweight='bold')
                # No individual legend - will be in main legend
                ax.grid(True, alpha=0.3)
                
                # Theta distribution plot
                ax = axes[idx, 1]
                
                # Import scipy for KDE
                from scipy.stats import gaussian_kde
                
                # Create KDE for both distributions
                true_kde = gaussian_kde(results['true_thetas_norm'][:corr['n_students']])
                learned_kde = gaussian_kde(results['learned_thetas_norm'])
                
                # Plot range
                x = np.linspace(-4, 4, 200)
                
                # Plot KDE curves
                ax.plot(x, true_kde(x), 'k--', linewidth=2, label='Perfect recovery')  # Black dashed for true
                ax.plot(x, learned_kde(x), color=model_color, linewidth=2.5, linestyle='-')  # Model color for learned
                
                ax.set_xlabel('Normalized θ')
                ax.set_ylabel('Density')
                ax.set_title('θ Distribution (KDE)', color='black', fontweight='bold')
                # No individual legend - will be in main legend
                ax.grid(True, alpha=0.3)
                ax.set_ylim(0, None)  # Start y-axis at 0
            
            # Discrimination plot
            if 'alpha_correlation' in corr:
                ax = axes[idx, 2]
                ax.scatter(results['true_alphas_norm'][results['valid_indices']], 
                          results['learned_alphas_norm'], alpha=0.6, color=model_color)
                ax.plot([0, 3], [0, 3], 'k--', linewidth=2, label='Perfect recovery')
                ax.set_xlabel('True α')
                ax.set_ylabel('Learned α')
                
                title_text = f'Discrimination (r={corr["alpha_correlation"]:.3f})'
                is_best_alpha = best_models.get('alpha') == model_name
                if is_best_alpha:
                    title_text += ' *'
                    ax.set_title(title_text, color='green', fontweight='bold')
                else:
                    ax.set_title(title_text, color='black', fontweight='bold')
                # No individual legend - will be in main legend
                ax.grid(True, alpha=0.3)
            
            # Threshold plots - handle variable number of thresholds
            if 'beta_correlations' in corr:
                n_thresholds = len(corr['beta_correlations'])
                for k in range(n_thresholds):  # Use all available thresholds
                    ax = axes[idx, k + 3]
                    ax.scatter(results['true_betas_norm'][results['valid_indices'], k], 
                              results['learned_betas_norm'][:, k], alpha=0.6, color=model_color)
                    ax.plot([-3, 3], [-3, 3], 'k--', linewidth=2, label='Perfect recovery')
                    ax.set_xlabel(f'True β_{k}')
                    ax.set_ylabel(f'Learned β_{k}')
                    
                    title_text = f'Threshold {k} (r={corr["beta_correlations"][k]:.3f})'
                    is_best_beta = best_models.get(f'beta_{k}') == model_name
                    if is_best_beta:
                        title_text += ' *'
                        ax.set_title(title_text, color='green', fontweight='bold')
                    else:
                        ax.set_title(title_text, color='black', fontweight='bold')
                    # No individual legend - will be in main legend
                    ax.grid(True, alpha=0.3)
        
        # Create suptitle - always black, never green, never model colors
        title = 'IRT Parameter Recovery Analysis\n' + \
                'All parameters normalized with standard IRT priors\n* = Best correlation per parameter type'
        
        if best_overall_model:
            avg_corr = overall_scores.get(best_overall_model, 0)
            title += f'\nBest Overall: {best_overall_model.upper()} (avg r={avg_corr:.3f})'
        
        # Adaptive spacing based on number of models
        n_models = len(models_with_corr)
        title_y = 1 - 0.01 / n_models  # Very close to top
        legend_y = title_y - 0.04  # Smaller gap between title and legend
        
        plt.suptitle(title, fontsize=14, fontweight='bold', y=title_y)  # Always black
        
        # Add model color legend under title
        from matplotlib.patches import Patch
        from matplotlib.lines import Line2D
        legend_elements = []
        
        # Add model colors
        for model_name, _ in models_with_corr:
            if model_colors and model_name in model_colors:
                color = model_colors[model_name]
            else:
                color = self.get_model_color(model_name)
            legend_elements.append(Patch(facecolor=color, label=model_name.upper()))
        
        # Add perfect recovery line
        legend_elements.append(Line2D([0], [0], color='k', linestyle='--', linewidth=2, 
                                     label='Perfect recovery'))
        
        # Create a figure-level legend with adaptive positioning
        fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, legend_y),
                  ncol=min(len(models_with_corr) + 1, 7), frameon=True, fancybox=True, shadow=True)
        
        # Adaptive top margin for tight_layout - negative margin to overlap
        top_margin = legend_y + 0.02  # Add to pull plots up (negative margin effect)
        plt.tight_layout(rect=[0, 0, 1, top_margin])  # Leave space for title and legend
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_rank_rank_heatmap(self, models_data, save_path, model_colors=None):
        """Create rank-rank correlation heatmap for all parameters."""
        if not self.true_params:
            print("No ground truth available for rank-rank analysis")
            return
        
        # Extract all models with valid data
        valid_models = []
        for model_name, data in models_data.items():
            if 'aggregated_params' in data and 'correlations' in data:
                valid_models.append(model_name)
        
        if len(valid_models) < 2:
            print("Need at least 2 models for rank-rank heatmap")
            return
        
        # Prepare data for heatmap
        n_models = len(valid_models)
        param_types = ['θ (Student)', 'α (Discrim)', 'β₀', 'β₁', 'β₂']
        n_params = len(param_types)
        
        # Create figure with subplots for each parameter type
        fig = plt.figure(figsize=(20, 5))
        
        # Calculate rank correlations for each parameter type
        for param_idx, param_type in enumerate(param_types):
            ax = plt.subplot(1, n_params, param_idx + 1)
            
            # Create correlation matrix for this parameter
            corr_matrix = np.zeros((n_models, n_models))
            
            for i, model_i in enumerate(valid_models):
                for j, model_j in enumerate(valid_models):
                    if i == j:
                        corr_matrix[i, j] = 1.0
                    else:
                        # Get rank correlation between models
                        data_i = models_data[model_i]['aggregated_params']
                        data_j = models_data[model_j]['aggregated_params']
                        
                        if param_idx == 0:  # Student ability
                            if 'student_abilities' in data_i and 'student_abilities' in data_j:
                                # Get common students
                                n_students = min(len(data_i['student_abilities']), 
                                               len(data_j['student_abilities']))
                                ranks_i = np.argsort(np.argsort(data_i['student_abilities'][:n_students]))
                                ranks_j = np.argsort(np.argsort(data_j['student_abilities'][:n_students]))
                                corr_matrix[i, j] = np.corrcoef(ranks_i, ranks_j)[0, 1]
                        
                        elif param_idx == 1:  # Discrimination
                            if 'item_discriminations' in data_i and 'item_discriminations' in data_j:
                                valid_i = data_i['item_counts'] > 0
                                valid_j = data_j['item_counts'] > 0
                                valid_both = valid_i & valid_j
                                if valid_both.sum() > 0:
                                    ranks_i = np.argsort(np.argsort(data_i['item_discriminations'][valid_both]))
                                    ranks_j = np.argsort(np.argsort(data_j['item_discriminations'][valid_both]))
                                    corr_matrix[i, j] = np.corrcoef(ranks_i, ranks_j)[0, 1]
                        
                        else:  # Thresholds
                            k = param_idx - 2
                            if 'item_thresholds' in data_i and 'item_thresholds' in data_j:
                                valid_i = data_i['item_counts'] > 0
                                valid_j = data_j['item_counts'] > 0
                                valid_both = valid_i & valid_j
                                if valid_both.sum() > 0 and k < data_i['item_thresholds'].shape[1]:
                                    ranks_i = np.argsort(np.argsort(data_i['item_thresholds'][valid_both, k]))
                                    ranks_j = np.argsort(np.argsort(data_j['item_thresholds'][valid_both, k]))
                                    corr_matrix[i, j] = np.corrcoef(ranks_i, ranks_j)[0, 1]
            
            # Plot heatmap
            im = ax.imshow(corr_matrix, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
            
            # Add text annotations
            for i in range(n_models):
                for j in range(n_models):
                    if not np.isnan(corr_matrix[i, j]):
                        text_color = 'white' if abs(corr_matrix[i, j]) > 0.5 else 'black'
                        ax.text(j, i, f'{corr_matrix[i, j]:.2f}', 
                               ha='center', va='center', color=text_color, fontsize=9)
            
            # Set labels
            ax.set_xticks(range(n_models))
            ax.set_yticks(range(n_models))
            ax.set_xticklabels([m.upper() for m in valid_models], rotation=45, ha='right')
            ax.set_yticklabels([m.upper() for m in valid_models])
            ax.set_title(f'{param_type} Rank Correlation')
            
            # Add colorbar for last subplot
            if param_idx == n_params - 1:
                cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
                cbar.set_label('Rank Correlation')
        
        plt.suptitle('Rank-Rank Correlation Analysis\nHow well models agree on relative ordering', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Rank-rank heatmap saved to: {save_path}")
    
    def plot_temporal_rank_rank_heatmap(self, temporal_data_dict, save_path, model_colors=None):
        """Create temporal rank-rank correlation heatmap for student abilities over time."""
        n_models = len(temporal_data_dict)
        
        # Create figure with subplots for each model
        fig, axes = plt.subplots(1, n_models, figsize=(6 * n_models, 6))
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, temporal_data) in enumerate(temporal_data_dict.items()):
            ax = axes[idx]
            
            # Get student abilities over time
            student_abilities = temporal_data['student_abilities']
            n_students = temporal_data['n_students']
            
            # Find max sequence length
            max_len = max(len(seq) for seq in student_abilities)
            
            # Create matrix for all students at all time steps
            ability_matrix = np.full((n_students, max_len), np.nan)
            for i, abilities in enumerate(student_abilities):
                ability_matrix[i, :len(abilities)] = abilities
            
            # Select time points for comparison (e.g., every 10 steps)
            time_points = list(range(0, max_len, max(1, max_len // 10)))
            if time_points[-1] != max_len - 1:
                time_points.append(max_len - 1)
            
            n_time_points = len(time_points)
            
            # Create correlation matrix
            rank_corr_matrix = np.ones((n_time_points, n_time_points))
            
            for i, t1 in enumerate(time_points):
                for j, t2 in enumerate(time_points):
                    if i != j:
                        # Get abilities at both time points
                        abilities_t1 = ability_matrix[:, t1]
                        abilities_t2 = ability_matrix[:, t2]
                        
                        # Remove NaN values (students who haven't reached that time)
                        valid_mask = ~(np.isnan(abilities_t1) | np.isnan(abilities_t2))
                        
                        if valid_mask.sum() > 10:  # Need at least 10 students
                            # Calculate rank correlation
                            ranks_t1 = np.argsort(np.argsort(abilities_t1[valid_mask]))
                            ranks_t2 = np.argsort(np.argsort(abilities_t2[valid_mask]))
                            rank_corr_matrix[i, j] = np.corrcoef(ranks_t1, ranks_t2)[0, 1]
                        else:
                            rank_corr_matrix[i, j] = np.nan
            
            # Plot heatmap
            model_color = model_colors.get(model_name, 'blue') if model_colors else 'blue'
            
            # Use diverging colormap
            im = ax.imshow(rank_corr_matrix, cmap='RdBu', vmin=-1, vmax=1, aspect='auto')
            
            # Add text annotations - sparse to avoid clutter
            # Only annotate diagonal and a few key points
            for i in range(n_time_points):
                for j in range(n_time_points):
                    if not np.isnan(rank_corr_matrix[i, j]):
                        value = rank_corr_matrix[i, j]
                        show_annotation = False
                        
                        # Always show diagonal
                        if i == j:
                            show_annotation = True
                            fontsize = 9
                            fontweight = 'bold'
                        # Show corners
                        elif (i == 0 and j == n_time_points - 1) or (i == n_time_points - 1 and j == 0):
                            show_annotation = True
                            fontsize = 8
                            fontweight = 'normal'
                        # Show a few intermediate points
                        elif i % 3 == 0 and j % 3 == 0 and i != j:
                            show_annotation = True
                            fontsize = 7
                            fontweight = 'normal'
                        
                        if show_annotation:
                            text_color = 'white' if abs(value) > 0.5 else 'black'
                            ax.text(j, i, f'{value:.2f}', 
                                   ha='center', va='center', color=text_color, 
                                   fontsize=fontsize, fontweight=fontweight)
            
            # Set labels - show fewer labels to avoid clutter
            label_skip = max(1, n_time_points // 6)  # Show ~6 labels
            ax.set_xticks(range(0, n_time_points, label_skip))
            ax.set_yticks(range(0, n_time_points, label_skip))
            time_labels = [f't={time_points[i]}' for i in range(0, n_time_points, label_skip)]
            ax.set_xticklabels(time_labels, rotation=45, ha='right', fontsize=9)
            ax.set_yticklabels(time_labels, fontsize=9)
            
            # Add minor ticks without labels for reference
            ax.set_xticks(range(n_time_points), minor=True)
            ax.set_yticks(range(n_time_points), minor=True)
            ax.grid(True, which='minor', alpha=0.2, linestyle=':')
            ax.set_title(f'{model_name.upper()}\nTemporal θ Rank Stability', 
                        color=model_color, fontweight='bold')
            
            # Add colorbar for each subplot
            cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            if idx == n_models - 1:
                cbar.set_label('Rank Correlation')
        
        plt.suptitle('Temporal Rank-Rank Correlation Analysis\nHow stable are student rankings over time?', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Temporal rank-rank heatmap saved to: {save_path}")
    
    def plot_temporal_analysis(self, temporal_data_dict, save_path, best_model=None, model_colors=None):
        """Create temporal analysis plots."""
        n_models = len(temporal_data_dict)
        fig, axes = plt.subplots(n_models, 3, figsize=(15, 4 * n_models))
        
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (model_name, temporal_data) in enumerate(temporal_data_dict.items()):
            # Use passed model colors if available, otherwise fetch
            if model_colors and model_name in model_colors:
                model_color = model_colors[model_name]
            else:
                model_color = self.get_model_color(model_name)
            
            # Plot 1: Student ability evolution
            ax = axes[idx, 0]
            n_samples = min(10, len(temporal_data['student_abilities']))
            for i in range(n_samples):
                abilities = temporal_data['student_abilities'][i]
                ax.plot(abilities, alpha=0.5, linewidth=1, color=model_color)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Student Ability (θ)')
            ax.set_title(f'{model_name.upper()}: Student Ability Evolution', color=model_color, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Average ability trajectory
            ax = axes[idx, 1]
            max_len = max(len(seq) for seq in temporal_data['student_abilities'])
            avg_trajectory = []
            std_trajectory = []
            
            for t in range(max_len):
                abilities_at_t = []
                for seq in temporal_data['student_abilities']:
                    if t < len(seq):
                        abilities_at_t.append(seq[t])
                if abilities_at_t:
                    avg_trajectory.append(np.mean(abilities_at_t))
                    std_trajectory.append(np.std(abilities_at_t))
            
            avg_trajectory = np.array(avg_trajectory)
            std_trajectory = np.array(std_trajectory)
            
            ax.plot(avg_trajectory, color=model_color, linewidth=2, label='Mean')
            ax.fill_between(range(len(avg_trajectory)), 
                           avg_trajectory - std_trajectory,
                           avg_trajectory + std_trajectory,
                           color=model_color, alpha=0.3, label='±1 SD')
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Average Student Ability')
            ax.set_title('Population Ability Trajectory', color=model_color, fontweight='bold')
            # No individual legend - will be in main legend
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Final vs average ability comparison
            ax = axes[idx, 2]
            final_abilities = [seq[-1] for seq in temporal_data['student_abilities']]
            avg_abilities = [seq.mean() for seq in temporal_data['student_abilities']]
            
            ax.scatter(avg_abilities, final_abilities, alpha=0.5, color=model_color)
            ax.plot([-5, 5], [-5, 5], 'k--', linewidth=2, label='Perfect recovery')
            ax.set_xlabel('Average Ability')
            ax.set_ylabel('Final Ability')
            ax.set_title('Final vs Average Student Ability', color=model_color, fontweight='bold')
            # No individual legend - will be in main legend
            ax.grid(True, alpha=0.3)
        
        # Suptitle remains neutral - no green color, no emoji
        if best_model and best_model in temporal_data_dict:
            plt.suptitle(f'Temporal Analysis of IRT Parameters\nBest Model: {best_model.upper()}', 
                        fontsize=14, fontweight='bold')
        else:
            plt.suptitle('Temporal Analysis of IRT Parameters', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def select_students_by_static_hit_rate(self, temporal_data, n_students=20):
        """Select students based on static hit rate (overall prediction accuracy)."""
        hit_rates = []
        
        for student_idx in range(len(temporal_data['gpcm_probabilities'])):
            gpcm_probs = temporal_data['gpcm_probabilities'][student_idx]  # (seq_len, n_cats)
            responses = temporal_data['responses'][student_idx]  # (seq_len,)
            seq_len = len(responses)
            
            if seq_len < 10:  # Need sufficient sequence length
                continue
                
            # Calculate overall hit rate (correct predictions)
            correct_predictions = 0
            total_predictions = 0
            
            for t in range(seq_len):
                if 0 <= responses[t] < gpcm_probs.shape[1]:
                    predicted_category = np.argmax(gpcm_probs[t])
                    actual_category = int(responses[t])
                    if predicted_category == actual_category:
                        correct_predictions += 1
                    total_predictions += 1
            
            hit_rate = correct_predictions / total_predictions if total_predictions > 0 else 0.0
            hit_rates.append((student_idx, hit_rate))
        
        # Sort by hit rate and select students with diverse performance levels
        hit_rates.sort(key=lambda x: x[1], reverse=True)
        
        # Select students across different performance levels
        if len(hit_rates) >= n_students:
            # Take students from different quartiles for diversity
            selected_indices = []
            scores_per_quartile = max(1, n_students // 4)
            
            # High performers (top quartile)
            selected_indices.extend([x[0] for x in hit_rates[:scores_per_quartile]])
            
            # Medium-high performers
            q2_start = len(hit_rates) // 4
            selected_indices.extend([x[0] for x in hit_rates[q2_start:q2_start + scores_per_quartile]])
            
            # Medium-low performers  
            q3_start = len(hit_rates) // 2
            selected_indices.extend([x[0] for x in hit_rates[q3_start:q3_start + scores_per_quartile]])
            
            # Low performers (bottom quartile)
            remaining = n_students - len(selected_indices)
            q4_start = 3 * len(hit_rates) // 4
            selected_indices.extend([x[0] for x in hit_rates[q4_start:q4_start + remaining]])
            
        else:
            selected_indices = [x[0] for x in hit_rates[:n_students]]
        
        return selected_indices, hit_rates
    
    def select_students_by_model_average_hit_rate(self, temporal_data_dict, n_students=20):
        """Select students based on model-averaged hit rates for consistent visualization."""
        # Collect hit rates for each student across all models
        student_hit_rates = {}  # student_idx -> list of hit rates across models
        
        for model_name, temporal_data in temporal_data_dict.items():
            for student_idx in range(len(temporal_data['gpcm_probabilities'])):
                gpcm_probs = temporal_data['gpcm_probabilities'][student_idx]
                responses = temporal_data['responses'][student_idx]
                seq_len = len(responses)
                
                if seq_len < 10:  # Need sufficient sequence length
                    continue
                
                # Calculate hit rate for this student in this model
                correct_predictions = 0
                total_predictions = 0
                
                for t in range(seq_len):
                    if 0 <= responses[t] < gpcm_probs.shape[1]:
                        predicted_category = np.argmax(gpcm_probs[t])
                        actual_category = int(responses[t])
                        if predicted_category == actual_category:
                            correct_predictions += 1
                        total_predictions += 1
                
                hit_rate = correct_predictions / total_predictions if total_predictions > 0 else 0.0
                
                if student_idx not in student_hit_rates:
                    student_hit_rates[student_idx] = []
                student_hit_rates[student_idx].append(hit_rate)
        
        # Calculate average hit rate for each student
        avg_hit_rates = []
        for student_idx, hit_rates in student_hit_rates.items():
            avg_rate = np.mean(hit_rates)
            avg_hit_rates.append((student_idx, avg_rate))
        
        # Sort by average hit rate
        avg_hit_rates.sort(key=lambda x: x[1], reverse=True)
        
        # Select students across different performance levels
        selected_indices = []
        student_scores = []
        
        if len(avg_hit_rates) >= n_students:
            # Take students from different quartiles for diversity
            scores_per_quartile = max(1, n_students // 4)
            
            # High performers (top quartile)
            q1_indices = [x for x in avg_hit_rates[:len(avg_hit_rates)//4]]
            selected = q1_indices[:scores_per_quartile]
            selected_indices.extend([x[0] for x in selected])
            student_scores.extend([x[1] for x in selected])
            
            # Medium-high performers
            q2_start = len(avg_hit_rates) // 4
            q2_end = len(avg_hit_rates) // 2
            q2_indices = [x for x in avg_hit_rates[q2_start:q2_end]]
            selected = q2_indices[:scores_per_quartile]
            selected_indices.extend([x[0] for x in selected])
            student_scores.extend([x[1] for x in selected])
            
            # Medium-low performers
            q3_start = len(avg_hit_rates) // 2
            q3_end = 3 * len(avg_hit_rates) // 4
            q3_indices = [x for x in avg_hit_rates[q3_start:q3_end]]
            selected = q3_indices[:scores_per_quartile]
            selected_indices.extend([x[0] for x in selected])
            student_scores.extend([x[1] for x in selected])
            
            # Low performers (bottom quartile)
            q4_indices = [x for x in avg_hit_rates[3*len(avg_hit_rates)//4:]]
            remaining = n_students - len(selected_indices)
            selected = q4_indices[:remaining]
            selected_indices.extend([x[0] for x in selected])
            student_scores.extend([x[1] for x in selected])
        else:
            # Take all available students
            selected_indices = [x[0] for x in avg_hit_rates]
            student_scores = [x[1] for x in avg_hit_rates]
        
        print(f"Selected {len(selected_indices)} students based on model-averaged hit rates")
        print(f"Average hit rate range: {min(student_scores):.3f} - {max(student_scores):.3f}")
        
        return selected_indices, student_scores
    
    def plot_temporal_theta_heatmap(self, temporal_data_dict, save_path, best_model=None, model_colors=None):
        """Create temporal theta heatmap visualization (students x questions)."""
        n_models = len(temporal_data_dict)
        
        # Adaptive figure height based on number of students (fixed height per model)
        height_per_model = 6  # Fixed height for each model's heatmap
        fig, axes = plt.subplots(n_models, 1, figsize=(12, height_per_model * n_models))
        
        if n_models == 1:
            axes = [axes]
        
        # Select students based on model-averaged hit rates (consistent across all plots)
        selected_students, avg_scores = self.select_students_by_model_average_hit_rate(temporal_data_dict, n_students=20)
        n_students = len(selected_students)
        
        for idx, (model_name, temporal_data) in enumerate(temporal_data_dict.items()):
            # Use passed model colors if available, otherwise fetch
            if model_colors and model_name in model_colors:
                model_color = model_colors[model_name]
            else:
                model_color = self.get_model_color(model_name)
            ax = axes[idx]
            
            # Use full number of questions
            max_questions = max(len(q_seq) for q_seq in temporal_data['question_ids'])
            
            # Create theta matrix - rows: selected students, cols: questions (time steps)
            theta_matrix = np.full((n_students, max_questions), np.nan)
            student_labels = []  # Track which students we're showing
            
            for display_idx, student_idx in enumerate(selected_students):
                abilities = temporal_data['student_abilities'][student_idx]
                seq_len = min(max_questions, len(abilities))
                theta_matrix[display_idx, :seq_len] = abilities[:seq_len]
                # Include average hit rate in label
                student_labels.append(f"S{student_idx+1} ({avg_scores[display_idx]:.2f})")  # Student labels with hit rate
            
            # Create heatmap
            im = ax.imshow(theta_matrix, cmap='RdYlBu_r', aspect='auto', 
                          interpolation='nearest', vmin=-3, vmax=3)
            
            # Set labels and title
            ax.set_xlabel('Question/Time Steps')
            ax.set_ylabel('Students')
            ax.set_title(f'{model_name.upper()}: Student Ability (θ) Evolution Over Questions', 
                        color=model_color, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Student Ability (θ)', rotation=270, labelpad=15)
            
            # Set reasonable tick spacing
            if max_questions > 20:
                x_ticks = np.arange(0, max_questions, max(1, max_questions // 10))
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_ticks + 1)  # Questions start from 1
            
            # Set y-ticks for students (show actual student IDs, every 4th)
            y_ticks = np.arange(0, n_students, 4)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([student_labels[i] for i in y_ticks])  # Show actual student IDs
        
        # Suptitle remains neutral - no green color, no emoji
        if best_model and best_model in temporal_data_dict:
            plt.suptitle(f'Temporal Theta Heatmaps\nBest Model: {best_model.upper()}', 
                        fontsize=14, fontweight='bold')
        else:
            plt.suptitle('Temporal Theta Heatmaps', fontsize=14, fontweight='bold')
        # Adaptive spacing based on number of models
        top_space = max(0.85, 1 - 0.05 / n_models)  # More space for title with fewer models
        plt.tight_layout(rect=[0, 0.03, 1, top_space])
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_temporal_gpcm_probs_heatmap(self, temporal_data_dict, save_path, best_model=None, model_colors=None):
        """Create temporal GPCM probabilities heatmap for predicted categories."""
        n_models = len(temporal_data_dict)
        
        # Adaptive figure height based on number of students (fixed height per model)
        height_per_model = 6  # Fixed height for each model's heatmap
        fig, axes = plt.subplots(n_models, 1, figsize=(12, height_per_model * n_models))
        
        if n_models == 1:
            axes = [axes]
        
        # Select students based on model-averaged hit rates (consistent across all plots)
        selected_students, avg_scores = self.select_students_by_model_average_hit_rate(temporal_data_dict, n_students=20)
        n_students = len(selected_students)
        
        for idx, (model_name, temporal_data) in enumerate(temporal_data_dict.items()):
            # Use passed model colors if available, otherwise fetch
            if model_colors and model_name in model_colors:
                model_color = model_colors[model_name]
            else:
                model_color = self.get_model_color(model_name)
            ax = axes[idx]
            
            # Use full number of questions
            max_questions = max(len(q_seq) for q_seq in temporal_data['question_ids'])
            
            # Create probability matrix - rows: selected students, cols: questions (time steps)
            prob_matrix = np.full((n_students, max_questions), np.nan)
            student_labels = []  # Track which students we're showing
            
            for display_idx, student_idx in enumerate(selected_students):
                gpcm_probs = temporal_data['gpcm_probabilities'][student_idx]  # (seq_len, n_cats)
                seq_len = min(max_questions, len(gpcm_probs))
                # Include average hit rate in label
                student_labels.append(f"S{student_idx+1} ({avg_scores[display_idx]:.2f})")  # Student labels with hit rate
                
                # Extract probabilities for the predicted response categories (argmax)
                for t in range(seq_len):
                    predicted_category = np.argmax(gpcm_probs[t])  # Get predicted category
                    prob_matrix[display_idx, t] = gpcm_probs[t, predicted_category]  # Probability of predicted category
            
            # Create heatmap
            im = ax.imshow(prob_matrix, cmap='RdYlGn', aspect='auto', 
                          interpolation='nearest', vmin=0, vmax=1)
            
            # Set labels and title
            ax.set_xlabel('Question/Time Steps')
            ax.set_ylabel('Students')
            ax.set_title(f'{model_name.upper()}: GPCM Probabilities (using temporal α,β)', 
                        color=model_color, fontweight='bold')
            
            # Add colorbar
            cbar = plt.colorbar(im, ax=ax)
            cbar.set_label('Probability of Predicted Category', rotation=270, labelpad=15)
            
            # Set reasonable tick spacing
            if max_questions > 20:
                x_ticks = np.arange(0, max_questions, max(1, max_questions // 10))
                ax.set_xticks(x_ticks)
                ax.set_xticklabels(x_ticks + 1)  # Questions start from 1
            
            # Set y-ticks for students (show actual student IDs, every 4th)
            y_ticks = np.arange(0, n_students, 4)
            ax.set_yticks(y_ticks)
            ax.set_yticklabels([student_labels[i] for i in y_ticks])  # Show actual student IDs
        
        # Suptitle remains neutral - no green color, no emoji
        title = 'Temporal GPCM Probability Heatmaps\n(Probabilities computed using temporal alpha and beta parameters)'
        if best_model and best_model in temporal_data_dict:
            title += f'\nBest Model: {best_model.upper()}'
        plt.suptitle(title, fontsize=14, fontweight='bold')
        # Adaptive spacing based on number of models
        top_space = max(0.85, 1 - 0.05 / n_models)  # More space for title with fewer models
        plt.tight_layout(rect=[0, 0.03, 1, top_space])
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_temporal_parameters_combined(self, temporal_data_dict, save_path, best_model=None, model_colors=None):
        """Create combined visualization showing temporal α, β, and resulting GPCM probabilities."""
        n_models = len(temporal_data_dict)
        
        # Create subplots: 3 columns per model (alpha, beta_0, probabilities) arranged horizontally
        fig, axes = plt.subplots(n_models, 3, figsize=(18, 5 * n_models))
        
        if n_models == 1:
            # Single model, 3 subplots in horizontal row
            axes = axes.reshape(1, -1)
        
        # Ensure axes is always 2D for consistent indexing
        if axes.ndim == 1:
            axes = axes.reshape(1, -1)
        
        # Select students based on model-averaged hit rates (consistent across all plots)
        selected_students, avg_scores = self.select_students_by_model_average_hit_rate(temporal_data_dict, n_students=20)
        n_students = len(selected_students)
        
        for model_idx, (model_name, temporal_data) in enumerate(temporal_data_dict.items()):
            # Use passed model colors if available, otherwise fetch
            if model_colors and model_name in model_colors:
                model_color = model_colors[model_name]
            else:
                model_color = self.get_model_color(model_name)
            
            # Use full number of questions
            max_questions = max(len(q_seq) for q_seq in temporal_data['question_ids'])
            
            # Create matrices for parameters and probabilities
            alpha_matrix = np.full((n_students, max_questions), np.nan)
            beta_matrix = np.full((n_students, max_questions), np.nan)  # First threshold
            prob_matrix = np.full((n_students, max_questions), np.nan)
            student_labels = []
            
            for display_idx, student_idx in enumerate(selected_students):
                # Get temporal parameters
                alphas = temporal_data['item_discriminations'][student_idx]
                betas = temporal_data['item_thresholds'][student_idx]  # Shape: (seq_len, n_cats-1)
                gpcm_probs = temporal_data['gpcm_probabilities'][student_idx]
                
                seq_len = min(max_questions, len(alphas))
                # Include average hit rate in label
                student_labels.append(f"S{student_idx+1} ({avg_scores[display_idx]:.2f})")
                
                # Fill matrices
                alpha_matrix[display_idx, :seq_len] = alphas[:seq_len]
                beta_matrix[display_idx, :seq_len] = betas[:seq_len, 0]  # First threshold
                
                # Extract probabilities for predicted categories
                for t in range(seq_len):
                    predicted_category = np.argmax(gpcm_probs[t])
                    prob_matrix[display_idx, t] = gpcm_probs[t, predicted_category]
            
            # Plot 1: Temporal Alpha (Discrimination)
            ax_alpha = axes[model_idx, 0]
            im_alpha = ax_alpha.imshow(alpha_matrix, cmap='viridis', aspect='auto', 
                                     interpolation='nearest')
            ax_alpha.set_ylabel('Students')
            ax_alpha.set_title(f'{model_name.upper()}: Temporal Discrimination (α)', 
                             color=model_color, fontweight='bold')
            plt.colorbar(im_alpha, ax=ax_alpha, label='α value')
            
            # Set y-ticks
            y_ticks = np.arange(0, n_students, 4)
            ax_alpha.set_yticks(y_ticks)
            ax_alpha.set_yticklabels([student_labels[i] for i in y_ticks])
            
            # Plot 2: Temporal Beta (First Threshold)  
            ax_beta = axes[model_idx, 1]
            im_beta = ax_beta.imshow(beta_matrix, cmap='plasma', aspect='auto',
                                   interpolation='nearest')
            ax_beta.set_ylabel('Students')
            ax_beta.set_title(f'{model_name.upper()}: Temporal Threshold β₀', 
                            color=model_color, fontweight='bold')
            plt.colorbar(im_beta, ax=ax_beta, label='β₀ value')
            
            # Set y-ticks
            ax_beta.set_yticks(y_ticks)
            ax_beta.set_yticklabels([student_labels[i] for i in y_ticks])
            
            # Plot 3: Resulting GPCM Probabilities
            ax_prob = axes[model_idx, 2]
            im_prob = ax_prob.imshow(prob_matrix, cmap='RdYlGn', aspect='auto',
                                   interpolation='nearest', vmin=0, vmax=1)
            ax_prob.set_xlabel('Question/Time Steps')
            ax_prob.set_ylabel('Students')
            ax_prob.set_title(f'{model_name.upper()}: Resulting GPCM Probabilities', 
                            color=model_color, fontweight='bold')
            plt.colorbar(im_prob, ax=ax_prob, label='Probability')
            
            # Set ticks for all plots
            if max_questions > 20:
                x_ticks = np.arange(0, max_questions, max(1, max_questions // 10))
                for ax in [ax_alpha, ax_beta, ax_prob]:
                    ax.set_xticks(x_ticks)
                    ax.set_xticklabels(x_ticks + 1)
            
            ax_prob.set_yticks(y_ticks)
            ax_prob.set_yticklabels([student_labels[i] for i in y_ticks])
        
        # Suptitle remains neutral - no green color, no emoji
        title = 'Temporal IRT Parameters -> GPCM Probabilities\n(alpha and beta evolve over time, determining probability calculations)'
        if best_model and best_model in temporal_data_dict:
            title += f'\nBest Model: {best_model.upper()}'
        plt.suptitle(title, fontsize=14, fontweight='bold')
        # Adaptive spacing based on number of models
        top_space = max(0.85, 1 - 0.05 / n_models)  # More space for title with fewer models
        plt.tight_layout(rect=[0, 0.03, 1, top_space])
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def print_student_summary(self, temporal_data_dict):
        """Print summary of selected students and their static hit rates."""
        print(f"\n" + "="*75)
        print("SELECTED STUDENTS SUMMARY (Static Hit Rate)")
        print("="*75)
        
        for model_name, temporal_data in temporal_data_dict.items():
            print(f"\n{model_name.upper()} Model:")
            
            # Get static hit rates for all students
            selected_students, hit_rates = self.select_students_by_static_hit_rate(temporal_data, n_students=20)
            
            # Extract hit rate statistics
            rates = [rate[1] for rate in hit_rates]
            
            print(f"  Total students analyzed: {len(hit_rates)}")
            print(f"  Static hit rate range: {min(rates):.3f} - {max(rates):.3f}")
            print(f"  Average static hit rate: {np.mean(rates):.3f}")
            
            print(f"  Selected students (showing 5 across performance spectrum):")
            for i in [0, 4, 9, 14, 19]:  # Show sample students across performance range
                student_idx = selected_students[i]
                # Find the hit rate for this student
                student_data = next(rate for rate in hit_rates if rate[0] == student_idx)
                hit_rate = student_data[1]
                
                print(f"    S{student_idx+1}: hit_rate={hit_rate:.3f}")
        
        print(f"\nStatic Hit Rate: Overall prediction accuracy (correct predictions / total predictions)")
        print("="*75)
    
    def plot_irt_functions(self, params, model_name, output_dir):
        """Generate standard IRT plots (ICC, IIF, TIF, Wright Map)."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get consistent model color
        model_color = self.get_model_color(model_name)
        
        # Extract parameters
        alphas = params['item_discriminations']
        betas = params['item_thresholds']
        thetas = params.get('student_abilities', np.linspace(-3, 3, 100))
        
        # Item Characteristic Curves (ICC)
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        axes = axes.ravel()
        
        # Sample 4 items to plot
        n_items = min(4, len(alphas))
        item_indices = np.linspace(0, len(alphas)-1, n_items, dtype=int)
        
        theta_range = np.linspace(-3, 3, 100)
        
        for i, item_idx in enumerate(item_indices):
            ax = axes[i]
            alpha = alphas[item_idx]
            beta = betas[item_idx]
            
            # Calculate probabilities for each category
            probs = []
            for k in range(self.n_cats):
                if k == 0:
                    prob = 1 / (1 + np.exp(alpha * (theta_range - beta[0])))
                elif k == self.n_cats - 1:
                    prob = 1 / (1 + np.exp(-alpha * (theta_range - beta[-1])))
                else:
                    prob = 1 / (1 + np.exp(-alpha * (theta_range - beta[k-1]))) - \
                           1 / (1 + np.exp(-alpha * (theta_range - beta[k])))
                probs.append(prob)
            
            # Plot
            for k, prob in enumerate(probs):
                ax.plot(theta_range, prob, label=f'Category {k}')
            
            ax.set_xlabel('Ability (θ)')
            ax.set_ylabel('Probability')
            ax.set_title(f'Item {item_idx + 1} ICC')
            # No individual legend - will be in main legend
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} - Item Characteristic Curves', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f'{model_name}_icc.png', dpi=150)
        plt.close()
        
        # Test Information Function (TIF)
        fig, ax = plt.subplots(figsize=(8, 6))
        
        total_info = np.zeros_like(theta_range)
        for item_idx in range(len(alphas)):
            alpha = alphas[item_idx]
            beta = betas[item_idx]
            
            # Calculate item information
            item_info = np.zeros_like(theta_range)
            for k in range(self.n_cats - 1):
                p_k = 1 / (1 + np.exp(-alpha * (theta_range - beta[k])))
                item_info += alpha**2 * p_k * (1 - p_k)
            
            total_info += item_info
        
        ax.plot(theta_range, total_info, color=model_color, linewidth=2)
        ax.set_xlabel('Ability (θ)')
        ax.set_ylabel('Information')
        ax.set_title(f'{model_name} - Test Information Function')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / f'{model_name}_tif.png', dpi=150)
        plt.close()
        
        # Wright Map
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6), sharey=True)
        
        # Student ability distribution
        if 'student_abilities' in params:
            ax1.hist(thetas, bins=30, orientation='horizontal', alpha=0.7, color=model_color)
            ax1.set_xlabel('Count')
            ax1.set_ylabel('Ability (θ)')
            ax1.set_title('Student Distribution')
            ax1.grid(True, alpha=0.3)
        
        # Item difficulty distribution
        avg_betas = np.mean(betas, axis=1)
        ax2.scatter(np.ones_like(avg_betas), avg_betas, alpha=0.6, s=50, color=model_color)
        ax2.set_xlim(0, 2)
        ax2.set_xlabel('Items')
        ax2.set_title('Item Difficulty')
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'{model_name} - Wright Map', fontsize=14)
        plt.tight_layout()
        plt.savefig(output_dir / f'{model_name}_wright.png', dpi=150)
        plt.close()
    
    def generate_summary_report(self, all_results, args):
        """Generate comprehensive summary report."""
        summary_path = self.output_dir / 'irt_summary.txt'  # Shortened from irt_analysis_summary.txt
        
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("UNIFIED IRT ANALYSIS REPORT\n")
            f.write("="*80 + "\n\n")
            f.write(f"Dataset: {self.dataset}\n")
            f.write(f"Models analyzed: {len(all_results)}\n")
            f.write(f"\nAnalysis settings:\n")
            f.write(f"- Student ability method: {args.theta_method}\n")
            f.write(f"- Item parameter method: {args.item_method}\n")
            f.write(f"- Analysis types: {', '.join(args.analysis_types)}\n")
            
            if self.true_params:
                f.write("\nParameter normalization:\n")
                f.write("- Student ability (θ): Normal prior (mean=0, std=1)\n")
                f.write("- Discrimination (α): Log-normal prior (mean=1, std=0.5)\n")
                f.write("- Thresholds (β): Normal prior (mean=0, std=1)\n")
            
            f.write("\n" + "-"*80 + "\n")
            
            # Model-specific results
            for model_name, results in all_results.items():
                f.write(f"\nMODEL: {model_name.upper()}\n")
                f.write(f"  Type: {results.get('model_type', 'unknown')}\n")
                
                if 'correlations' in results:
                    corr = results['correlations']
                    
                    if 'theta_correlation' in corr:
                        f.write(f"\n  Student Ability (θ) Recovery:\n")
                        f.write(f"    Correlation: {corr['theta_correlation']:.4f}\n")
                        f.write(f"    Students analyzed: {corr['n_students']}\n")
                    
                    if 'alpha_correlation' in corr:
                        f.write(f"\n  Item Parameter Recovery:\n")
                        f.write(f"    Discrimination (α): {corr['alpha_correlation']:.4f}\n")
                        f.write(f"    Items analyzed: {corr['n_items']}/{self.n_questions}\n")
                    
                    if 'beta_correlations' in corr:
                        f.write(f"    Threshold correlations:\n")
                        for k, beta_corr in enumerate(corr['beta_correlations']):
                            f.write(f"      β_{k}: {beta_corr:.4f}\n")
                        f.write(f"    Average β correlation: {corr['beta_avg_correlation']:.4f}\n")
                
                # Parameter statistics
                if 'aggregated_params' in results:
                    params = results['aggregated_params']
                    
                    if 'student_abilities' in params:
                        abilities = params['student_abilities']
                        f.write(f"\n  Student ability statistics:\n")
                        f.write(f"    Range: [{abilities.min():.3f}, {abilities.max():.3f}]\n")
                        f.write(f"    Mean: {abilities.mean():.3f}, Std: {abilities.std():.3f}\n")
                    
                    if 'item_discriminations' in params:
                        alphas = params['item_discriminations']
                        valid = params['item_counts'] > 0
                        f.write(f"\n  Discrimination statistics:\n")
                        f.write(f"    Range: [{alphas[valid].min():.3f}, {alphas[valid].max():.3f}]\n")
                        f.write(f"    Mean: {alphas[valid].mean():.3f}, Std: {alphas[valid].std():.3f}\n")
            
            f.write("\n" + "-"*80 + "\n")
            f.write("\nKEY INSIGHTS:\n")
            f.write("- Deep-GPCM learns temporal IRT parameters that evolve over time\n")
            f.write("- Item parameters (α, β) show moderate recovery when aggregated\n")
            f.write("- Student abilities (θ) use different representations than static IRT\n")
            f.write("- Temporal dynamics capture learning trajectories not present in traditional IRT\n")
            f.write("="*80 + "\n")
        
        print(f"\nSummary report saved to: {summary_path}")
    
    def run_analysis(self, args):
        """Run complete IRT analysis based on arguments."""
        print("\n" + "="*80)
        print("UNIFIED IRT ANALYSIS")
        print("="*80)
        
        # Find models
        models = self.find_models()
        if not models:
            print("No trained models found!")
            return
        
        all_results = {}
        temporal_data_dict = {}
        best_overall_model = None
        best_overall_score = -1
        
        # Process each model
        for model_name, model_path in models.items():
            print(f"\nProcessing {model_name}...")
            
            try:
                # Load model
                model, model_type, device = self.load_model(model_path)
                
                # Extract temporal parameters
                temporal_data = self.extract_temporal_parameters(
                    model, model_type, device, split=args.split
                )
                temporal_data_dict[model_name] = temporal_data
                
                # Extract aggregated parameters
                aggregated_params = self.extract_aggregated_parameters(
                    temporal_data, 
                    theta_method=args.theta_method,
                    item_method=args.item_method
                )
                
                results = {
                    'model_type': model_type,
                    'temporal_data': temporal_data,
                    'aggregated_params': aggregated_params
                }
                
                # Calculate correlations if true parameters available
                if self.true_params and 'recovery' in args.analysis_types:
                    correlations = self.calculate_correlations(
                        self.true_params, aggregated_params
                    )
                    results['correlations'] = correlations
                    
                    # Calculate overall score for this model
                    model_correlations = []
                    if 'theta_correlation' in correlations:
                        model_correlations.append(correlations['theta_correlation'])
                    if 'alpha_correlation' in correlations:
                        model_correlations.append(correlations['alpha_correlation'])
                    if 'beta_correlations' in correlations:
                        model_correlations.extend(correlations['beta_correlations'])
                    
                    if model_correlations:
                        overall_score = np.mean(model_correlations)
                        if overall_score > best_overall_score:
                            best_overall_score = overall_score
                            best_overall_model = model_name
                    
                    # Transfer normalized parameters from correlations for plotting
                    for key in ['true_thetas_norm', 'learned_thetas_norm', 
                                'true_alphas_norm', 'learned_alphas_norm',
                                'true_betas_norm', 'learned_betas_norm', 
                                'valid_indices']:
                        if key in correlations:
                            results[key] = correlations[key]
                    
                    print(f"  Correlations: θ={correlations.get('theta_correlation', 'N/A'):.3f}, " +
                          f"α={correlations.get('alpha_correlation', 'N/A'):.3f}, " +
                          f"β_avg={correlations.get('beta_avg_correlation', 'N/A'):.3f}")
                
                all_results[model_name] = results
                
                # Save extracted parameters if requested
                if args.save_params:
                    param_file = self.output_dir / f'params_{model_name}.json'
                    save_params = {
                        k: v.tolist() if isinstance(v, np.ndarray) else v 
                        for k, v in aggregated_params.items()
                    }
                    save_params['model_type'] = model_type
                    with open(param_file, 'w') as f:
                        json.dump(save_params, f, indent=2)
                    print(f"  Parameters saved to: {param_file}")
                
            except Exception as e:
                error_msg = str(e)
                print(f"  ❌ Error loading {model_name}:")
                
                # Detailed error classification for better user understanding
                if "size mismatch" in error_msg and "embedding_projection" in error_msg:
                    print(f"    Architecture mismatch - Embedding projection dimensions don't match")
                    if "torch.Size([64, 800])" in error_msg and "torch.Size([64, 64])" in error_msg:
                        print(f"    Saved: 800-dim → 64-dim (with learnable embedding)")
                        print(f"    Current: 64-dim → 64-dim (without learnable embedding)")
                        print(f"    Solution: Retrain model with consistent learnable_embedding setting")
                    else:
                        print(f"    Dimension details: {error_msg.split('copying a param with shape')[1] if 'copying a param with shape' in error_msg else 'Unknown mismatch'}")
                elif "size mismatch" in error_msg:
                    print(f"    Parameter size mismatch - Model architecture changed")
                    print(f"    Details: {error_msg}")
                elif "Missing key(s)" in error_msg:
                    print(f"    Missing parameters - Model structure incomplete")
                    print(f"    Missing: {error_msg.split('Missing key(s) in state_dict:')[1] if 'Missing key(s) in state_dict:' in error_msg else 'Unknown keys'}")
                elif "Unexpected key(s)" in error_msg:
                    print(f"    Extra parameters - Model has additional components")
                    print(f"    Extra: {error_msg.split('Unexpected key(s) in state_dict:')[1] if 'Unexpected key(s) in state_dict:' in error_msg else 'Unknown keys'}")
                else:
                    print(f"    General error: {e}")
                
                print(f"  📋 Skipping {model_name} - incompatible with current configuration")
        
        # Create model color dictionary for all plots
        model_colors = {}
        for model_name in all_results.keys():
            model_colors[model_name] = self.get_model_color(model_name)
        
        # Generate plots based on analysis types
        if 'recovery' in args.analysis_types and self.true_params and all_results:
            plot_path = self.output_dir / 'param_recovery.png'  # Shortened from parameter_recovery.png
            self.plot_parameter_recovery(all_results, plot_path, model_colors=model_colors)
            print(f"\nParameter recovery plot saved to: {plot_path}")
        
        if 'temporal' in args.analysis_types and temporal_data_dict:
            plot_path = self.output_dir / 'temporal.png'  # Shortened from temporal_analysis.png
            self.plot_temporal_analysis(temporal_data_dict, plot_path, best_model=best_overall_model, model_colors=model_colors)
            print(f"Temporal analysis plot saved to: {plot_path}")
            
            # Generate temporal heatmaps
            theta_heatmap_path = self.output_dir / 'theta_heatmap.png'  # Shortened from temporal_theta_heatmap.png
            self.plot_temporal_theta_heatmap(temporal_data_dict, theta_heatmap_path, best_model=best_overall_model, model_colors=model_colors)
            print(f"Temporal theta heatmap saved to: {theta_heatmap_path}")
            
            gpcm_heatmap_path = self.output_dir / 'gpcm_probs_heatmap.png'  # Shortened from temporal_gpcm_probs_heatmap.png
            self.plot_temporal_gpcm_probs_heatmap(temporal_data_dict, gpcm_heatmap_path, best_model=best_overall_model, model_colors=model_colors)
            print(f"Temporal GPCM probabilities heatmap saved to: {gpcm_heatmap_path}")
            
            # Generate combined temporal parameters visualization
            combined_params_path = self.output_dir / 'params_combined.png'  # Shortened from temporal_parameters_combined.png
            self.plot_temporal_parameters_combined(temporal_data_dict, combined_params_path, best_model=best_overall_model, model_colors=model_colors)
            print(f"Combined temporal parameters visualization saved to: {combined_params_path}")
            
            # Generate temporal rank-rank heatmap
            temporal_rank_path = self.output_dir / 'temporal_rank_rank.png'
            self.plot_temporal_rank_rank_heatmap(temporal_data_dict, temporal_rank_path, model_colors=model_colors)
            
            # Print summary of selected students
            self.print_student_summary(temporal_data_dict)
        
        if 'irt_plots' in args.analysis_types:
            for model_name, results in all_results.items():
                if 'aggregated_params' in results:
                    plot_dir = self.output_dir / 'model_specs' / model_name
                    self.plot_irt_functions(
                        results['aggregated_params'], 
                        model_name, 
                        plot_dir
                    )
                    print(f"IRT plots for {model_name} saved to: {plot_dir}")
        
        # Generate summary report
        self.generate_summary_report(all_results, args)
        
        print("\nAnalysis complete!")
        print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='Unified IRT Analysis Tool for Deep-GPCM',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic parameter recovery analysis
  python irt_analysis.py --dataset synthetic_OC
  
  # Temporal analysis with average theta
  python irt_analysis.py --dataset synthetic_OC --theta_method average --analysis_types temporal
  
  # Complete analysis with all plots
  python irt_analysis.py --dataset synthetic_OC --analysis_types recovery temporal irt_plots
  
  # Extract and save parameters only
  python irt_analysis.py --dataset synthetic_OC --save_params --analysis_types none
        """
    )
    
    parser.add_argument('--dataset', default='synthetic_OC', 
                        help='Dataset name')
    parser.add_argument('--output_dir', default=None, 
                        help='Output directory for results (default: auto-detect new structure)')
    parser.add_argument('--split', default='test', choices=['train', 'test'],
                        help='Which data split to analyze')
    
    # Parameter extraction methods
    parser.add_argument('--theta_method', default='last', choices=['last', 'average'],
                        help='Method for extracting student abilities')
    parser.add_argument('--item_method', default='average', choices=['average', 'last'],
                        help='Method for extracting item parameters')
    
    # Analysis types
    parser.add_argument('--analysis_types', nargs='+', 
                        default=['recovery'],
                        choices=['recovery', 'temporal', 'irt_plots', 'none'],
                        help='Types of analysis to perform')
    
    # Options
    parser.add_argument('--save_params', action='store_true',
                        help='Save extracted parameters to JSON files')
    
    args = parser.parse_args()
    
    # Create analyzer and run
    analyzer = UnifiedIRTAnalyzer(dataset=args.dataset, output_dir=args.output_dir)
    analyzer.run_analysis(args)


if __name__ == "__main__":
    main()