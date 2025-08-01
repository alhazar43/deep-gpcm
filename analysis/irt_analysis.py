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

from core.model import DeepGPCM, AttentionGPCM
from core.attention_enhanced import EnhancedAttentionGPCM
from core.coral_gpcm import HybridCORALGPCM
from train import load_simple_data, create_data_loaders


class UnifiedIRTAnalyzer:
    """Unified IRT analysis tool with all functionality."""
    
    def __init__(self, dataset='synthetic_OC', output_dir='results/irt'):
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize consistent model coloring system (matching utils/plot_metrics.py)
        self.model_colors = {
            'deep_gpcm': '#ff7f0e',     # Orange
            'attn_gpcm': '#1f77b4',     # Blue
            'coral_gpcm': '#d62728'     # Red
        }
        
        # Load data
        data_dir = Path('data') / dataset
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
        """Get consistent color for a model, matching the main plotting system."""
        if model_name in self.model_colors:
            return self.model_colors[model_name]
        else:
            # Use tab10 colors for unknown models
            colors = plt.cm.tab10.colors if hasattr(plt.cm.tab10, 'colors') else plt.cm.tab10(range(10))
            base_idx = len(self.model_colors) % 10
            return colors[base_idx]
    
    def find_models(self):
        """Automatically find all trained models for the dataset."""
        model_pattern = f"save_models/best_*_{self.dataset}.pth"
        model_files = glob(model_pattern)
        
        # Filter out fold-specific models
        model_files = [f for f in model_files if not any(f'fold_{i}' in f for i in range(1, 10))]
        
        models = {}
        for model_path in model_files:
            # Extract model name from path
            base_name = os.path.basename(model_path)
            model_name = base_name.replace(f'best_', '').replace(f'_{self.dataset}.pth', '')
            models[model_name] = model_path
            
        print(f"Found {len(models)} models: {list(models.keys())}")
        return models
    
    def load_model(self, model_path):
        """Load a model with proper handling for different architectures."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
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
        if 'config' in checkpoint:
            model_type = checkpoint['config'].get('model_type', 'baseline')
        else:
            if has_coral_projection:
                model_type = 'coral_gpcm'  # Updated to match new naming
            elif has_coral_layer:
                model_type = 'coral'  # Skip pure CORAL models as they don't have IRT params
            elif 'attention_refinement' in str(checkpoint['model_state_dict'].keys()):
                model_type = 'attention'
            else:
                model_type = 'deep_gpcm'  # Updated to match new naming
        
        # Skip pure CORAL models as they don't have IRT parameters
        if model_type == 'coral':
            raise ValueError(f"Pure CORAL model does not have IRT parameters - skipping")
        
        # Create model
        if has_learnable_params or ('attn_gpcm' in model_path):
            # Use EnhancedAttentionGPCM for models with learnable parameters
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
                ability_scale=2.0
            )
            model_type = 'attn_gpcm'
        elif model_type == 'coral_gpcm' or has_coral_projection:
            # Import the correct CORAL model
            from core.coral_gpcm import HybridCORALGPCM
            model = HybridCORALGPCM(
                n_questions=self.n_questions,
                n_cats=self.n_cats,
                memory_size=50,
                key_dim=50,
                value_dim=200,
                final_fc_dim=50
            )
        elif model_type == 'attention' or 'attention' in str(checkpoint['model_state_dict'].keys()):
            model = AttentionGPCM(n_questions=self.n_questions, n_cats=self.n_cats)
        else:
            model = DeepGPCM(n_questions=self.n_questions, n_cats=self.n_cats)
        
        # Load weights with proper handling for wrapped models
        try:
            model.load_state_dict(checkpoint['model_state_dict'])
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
                model.load_state_dict(unwrapped_state_dict)
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
            'student_ids': []  # Track which student each sequence belongs to
        }
        
        student_id = 0
        
        with torch.no_grad():
            for batch_idx, (questions, responses, mask) in enumerate(data_loader):
                questions = questions.to(device)
                responses = responses.to(device)
                batch_size, seq_len = questions.shape
                
                # Forward pass
                student_abilities, item_thresholds, discrimination_params, gpcm_probs = model(questions, responses)
                
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
        
        for i in range(n_students):
            abilities = temporal_data['student_abilities'][i]
            
            if theta_method == 'last':
                student_abilities[i] = abilities[-1]
            elif theta_method == 'average':
                student_abilities[i] = abilities.mean()
        
        results['student_abilities'] = student_abilities
        
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
            for k in range(3):
                beta_corrs.append(np.corrcoef(
                    norm_true_beta[valid_indices, k], 
                    norm_learned_beta[:, k])[0, 1])
            
            results['beta_correlations'] = beta_corrs
            results['beta_avg_correlation'] = np.mean(beta_corrs)
            results['n_items'] = len(valid_indices)
        
        return results
    
    def plot_parameter_recovery(self, model_results, save_path):
        """Create parameter recovery comparison plots."""
        # Count only models with correlations for plotting
        models_with_corr = [(name, results) for name, results in model_results.items() 
                           if 'correlations' in results]
        n_models = len(models_with_corr)
        
        if n_models == 0:
            print("No models with correlations to plot")
            return
            
        fig, axes = plt.subplots(n_models, 5, figsize=(20, 4 * n_models))
        
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        # Determine best model for each parameter type separately
        best_models = {}
        
        if len(models_with_corr) > 1:
            # Find best model for each parameter type
            parameter_types = ['theta', 'alpha'] + [f'beta_{i}' for i in range(3)]  # Assuming max 3 thresholds
            
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
            model_color = self.get_model_color(model_name)
            
            # Student ability plot
            if 'theta_correlation' in corr:
                ax = axes[idx, 0]
                ax.scatter(results['true_thetas_norm'][:corr['n_students']], 
                          results['learned_thetas_norm'], alpha=0.6, color=model_color)
                ax.plot([-3, 3], [-3, 3], 'r--', label='Perfect recovery')
                ax.set_xlabel('True θ')
                ax.set_ylabel('Learned θ')
                
                # Add star and green color for best model
                title_text = f'{model_name.upper()}\nStudent Ability (r={corr["theta_correlation"]:.3f})'
                is_best_theta = best_models.get('theta') == model_name
                if is_best_theta:
                    title_text += ' ★'
                    ax.set_title(title_text, color='green', fontweight='bold')
                else:
                    ax.set_title(title_text)
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Discrimination plot
            if 'alpha_correlation' in corr:
                ax = axes[idx, 1]
                ax.scatter(results['true_alphas_norm'][results['valid_indices']], 
                          results['learned_alphas_norm'], alpha=0.6, color=model_color)
                ax.plot([0, 3], [0, 3], 'r--', label='Perfect recovery')
                ax.set_xlabel('True α')
                ax.set_ylabel('Learned α')
                
                title_text = f'Discrimination (r={corr["alpha_correlation"]:.3f})'
                is_best_alpha = best_models.get('alpha') == model_name
                if is_best_alpha:
                    title_text += ' ★'
                    ax.set_title(title_text, color='green', fontweight='bold')
                else:
                    ax.set_title(title_text)
                ax.legend()
                ax.grid(True, alpha=0.3)
            
            # Threshold plots
            if 'beta_correlations' in corr:
                for k in range(3):
                    ax = axes[idx, k + 2]
                    ax.scatter(results['true_betas_norm'][results['valid_indices'], k], 
                              results['learned_betas_norm'][:, k], alpha=0.6, color=model_color)
                    ax.plot([-3, 3], [-3, 3], 'r--', label='Perfect recovery')
                    ax.set_xlabel(f'True β_{k}')
                    ax.set_ylabel(f'Learned β_{k}')
                    
                    title_text = f'Threshold {k} (r={corr["beta_correlations"][k]:.3f})'
                    is_best_beta = best_models.get(f'beta_{k}') == model_name
                    if is_best_beta:
                        title_text += ' ★'
                        ax.set_title(title_text, color='green', fontweight='bold')
                    else:
                        ax.set_title(title_text)
                    ax.legend()
                    ax.grid(True, alpha=0.3)
        
        plt.suptitle('IRT Parameter Recovery Analysis\n' +
                    'All parameters normalized with standard IRT priors\n★ = Best correlation per parameter type',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_temporal_analysis(self, temporal_data_dict, save_path):
        """Create temporal analysis plots."""
        n_models = len(temporal_data_dict)
        fig, axes = plt.subplots(n_models, 3, figsize=(15, 4 * n_models))
        
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (model_name, temporal_data) in enumerate(temporal_data_dict.items()):
            model_color = self.get_model_color(model_name)
            
            # Plot 1: Student ability evolution
            ax = axes[idx, 0]
            n_samples = min(10, len(temporal_data['student_abilities']))
            for i in range(n_samples):
                abilities = temporal_data['student_abilities'][i]
                ax.plot(abilities, alpha=0.5, linewidth=1, color=model_color)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Student Ability (θ)')
            ax.set_title(f'{model_name.upper()}: Student Ability Evolution')
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
            ax.set_title('Population Ability Trajectory')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Plot 3: Final vs average ability comparison
            ax = axes[idx, 2]
            final_abilities = [seq[-1] for seq in temporal_data['student_abilities']]
            avg_abilities = [seq.mean() for seq in temporal_data['student_abilities']]
            
            ax.scatter(avg_abilities, final_abilities, alpha=0.5)
            ax.plot([-5, 5], [-5, 5], 'r--', label='y=x')
            ax.set_xlabel('Average Ability')
            ax.set_ylabel('Final Ability')
            ax.set_title('Final vs Average Student Ability')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
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
    
    def plot_temporal_theta_heatmap(self, temporal_data_dict, save_path):
        """Create temporal theta heatmap visualization (students x questions)."""
        n_models = len(temporal_data_dict)
        
        # Adaptive figure height based on number of students (fixed height per model)
        height_per_model = 6  # Fixed height for each model's heatmap
        fig, axes = plt.subplots(n_models, 1, figsize=(12, height_per_model * n_models))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, temporal_data) in enumerate(temporal_data_dict.items()):
            model_color = self.get_model_color(model_name)
            ax = axes[idx]
            
            # Select students based on temporal hit rate (limit to first 50 questions)
            max_questions = min(50, max(len(q_seq) for q_seq in temporal_data['question_ids']))
            selected_students, _ = self.select_students_by_static_hit_rate(temporal_data, n_students=20)
            n_students = len(selected_students)
            
            # Create theta matrix - rows: selected students, cols: questions (time steps)
            theta_matrix = np.full((n_students, max_questions), np.nan)
            student_labels = []  # Track which students we're showing
            
            for display_idx, student_idx in enumerate(selected_students):
                abilities = temporal_data['student_abilities'][student_idx]
                seq_len = min(max_questions, len(abilities))
                theta_matrix[display_idx, :seq_len] = abilities[:seq_len]
                student_labels.append(f"S{student_idx+1}")  # Student labels starting from 1
            
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
        
        plt.suptitle('Temporal Theta Heatmaps', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_temporal_gpcm_probs_heatmap(self, temporal_data_dict, save_path):
        """Create temporal GPCM probabilities heatmap for predicted categories."""
        n_models = len(temporal_data_dict)
        
        # Adaptive figure height based on number of students (fixed height per model)
        height_per_model = 6  # Fixed height for each model's heatmap
        fig, axes = plt.subplots(n_models, 1, figsize=(12, height_per_model * n_models))
        
        if n_models == 1:
            axes = [axes]
        
        for idx, (model_name, temporal_data) in enumerate(temporal_data_dict.items()):
            model_color = self.get_model_color(model_name)
            ax = axes[idx]
            
            # Select students based on temporal hit rate (limit to first 50 questions)
            max_questions = min(50, max(len(q_seq) for q_seq in temporal_data['question_ids']))
            selected_students, _ = self.select_students_by_static_hit_rate(temporal_data, n_students=20)
            n_students = len(selected_students)
            
            # Create probability matrix - rows: selected students, cols: questions (time steps)
            prob_matrix = np.full((n_students, max_questions), np.nan)
            student_labels = []  # Track which students we're showing
            
            for display_idx, student_idx in enumerate(selected_students):
                gpcm_probs = temporal_data['gpcm_probabilities'][student_idx]  # (seq_len, n_cats)
                seq_len = min(max_questions, len(gpcm_probs))
                student_labels.append(f"S{student_idx+1}")  # Student labels starting from 1
                
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
        
        plt.suptitle('Temporal GPCM Probability Heatmaps\n(Probabilities computed using temporal α and β parameters)', fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def plot_temporal_parameters_combined(self, temporal_data_dict, save_path):
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
        
        for model_idx, (model_name, temporal_data) in enumerate(temporal_data_dict.items()):
            model_color = self.get_model_color(model_name)
            
            # Select students and limit questions
            max_questions = min(50, max(len(q_seq) for q_seq in temporal_data['question_ids']))
            selected_students, _ = self.select_students_by_static_hit_rate(temporal_data, n_students=20)
            n_students = len(selected_students)
            
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
                student_labels.append(f"S{student_idx+1}")
                
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
        
        plt.suptitle('Temporal IRT Parameters → GPCM Probabilities\n(α and β evolve over time, determining probability calculations)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
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
            ax.legend()
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
        plt.savefig(output_dir / f'{model_name}_wright_map.png', dpi=150)
        plt.close()
    
    def generate_summary_report(self, all_results, args):
        """Generate comprehensive summary report."""
        summary_path = self.output_dir / 'irt_analysis_summary.txt'
        
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
                    
                    # Store normalized parameters for plotting
                    if 'theta_correlation' in correlations:
                        true_thetas = np.array(self.true_params['student_abilities']['theta'])
                        n_students = correlations['n_students']
                        results['true_thetas_norm'] = self.normalize_parameters(
                            None, None, true_thetas[:n_students])['thetas']
                        results['learned_thetas_norm'] = self.normalize_parameters(
                            None, None, aggregated_params['student_abilities'][:n_students])['thetas']
                    
                    if 'alpha_correlation' in correlations:
                        true_alphas = np.array(self.true_params['question_params']['discrimination']['alpha'])
                        true_betas = np.array(self.true_params['question_params']['difficulties']['beta'])
                        valid_items = aggregated_params['item_counts'] > 0
                        valid_indices = np.where(valid_items)[0]
                        
                        results['true_alphas_norm'] = self.normalize_parameters(true_alphas, None)['alphas']
                        results['true_betas_norm'] = self.normalize_parameters(None, true_betas)['betas']
                        results['learned_alphas_norm'] = self.normalize_parameters(
                            aggregated_params['item_discriminations'][valid_items], None)['alphas']
                        results['learned_betas_norm'] = self.normalize_parameters(
                            None, aggregated_params['item_thresholds'][valid_items])['betas']
                        results['valid_indices'] = valid_indices
                    
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
                print(f"  Error processing {model_name}: {str(e)}")
                print(f"  Skipping this model...")
        
        # Generate plots based on analysis types
        if 'recovery' in args.analysis_types and self.true_params and all_results:
            plot_path = self.output_dir / 'parameter_recovery.png'
            self.plot_parameter_recovery(all_results, plot_path)
            print(f"\nParameter recovery plot saved to: {plot_path}")
        
        if 'temporal' in args.analysis_types and temporal_data_dict:
            plot_path = self.output_dir / 'temporal_analysis.png'
            self.plot_temporal_analysis(temporal_data_dict, plot_path)
            print(f"Temporal analysis plot saved to: {plot_path}")
            
            # Generate temporal heatmaps
            theta_heatmap_path = self.output_dir / 'temporal_theta_heatmap.png'
            self.plot_temporal_theta_heatmap(temporal_data_dict, theta_heatmap_path)
            print(f"Temporal theta heatmap saved to: {theta_heatmap_path}")
            
            gpcm_heatmap_path = self.output_dir / 'temporal_gpcm_probs_heatmap.png'
            self.plot_temporal_gpcm_probs_heatmap(temporal_data_dict, gpcm_heatmap_path)
            print(f"Temporal GPCM probabilities heatmap saved to: {gpcm_heatmap_path}")
            
            # Generate combined temporal parameters visualization
            combined_params_path = self.output_dir / 'temporal_parameters_combined.png'
            self.plot_temporal_parameters_combined(temporal_data_dict, combined_params_path)
            print(f"Combined temporal parameters visualization saved to: {combined_params_path}")
            
            # Print summary of selected students
            self.print_student_summary(temporal_data_dict)
        
        if 'irt_plots' in args.analysis_types:
            for model_name, results in all_results.items():
                if 'aggregated_params' in results:
                    plot_dir = self.output_dir / 'irt_plots' / model_name
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
    parser.add_argument('--output_dir', default='results/irt', 
                        help='Output directory for results')
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