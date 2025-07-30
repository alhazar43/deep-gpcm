#!/usr/bin/env python3
"""
Enhanced IRT analysis with proper temporal parameter handling.
Includes student ability (theta) comparison.
"""

import os
import sys
import torch
import torch.nn.functional as F
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from glob import glob

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.model import DeepGPCM, AttentionGPCM
from train import load_simple_data, create_data_loaders


class TemporalIRTAnalyzer:
    """Enhanced IRT analysis with temporal parameter tracking."""
    
    def __init__(self, dataset='synthetic_OC', output_dir='results/irt_temporal'):
        self.dataset = dataset
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
    
    def extract_temporal_parameters(self, model_path, split='test'):
        """Extract temporal IRT parameters keeping time dimension."""
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Determine model type
        model_type = 'attention' if 'attention_refinement' in str(checkpoint['model_state_dict'].keys()) else 'baseline'
        
        # Create model
        if model_type == 'attention':
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
        
        # Create data loader
        data = self.test_data if split == 'test' else self.train_data
        if split == 'train':
            data_loader, _ = create_data_loaders(self.train_data, self.test_data, batch_size=32)
        else:
            _, data_loader = create_data_loaders(self.train_data, self.test_data, batch_size=32)
        
        # Extract parameters with temporal structure
        temporal_data = {
            'student_abilities': [],  # List of sequences
            'item_discriminations': [],
            'item_thresholds': [],
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
                student_abilities, item_thresholds, discrimination_params, _ = model(questions, responses)
                
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
    
    def extract_student_abilities(self, temporal_data, method='last'):
        """Extract student abilities using specified method.
        
        Args:
            temporal_data: Temporal parameter data
            method: 'last' for final ability, 'average' for temporal average
        """
        n_students = temporal_data['n_students']
        student_abilities = np.zeros(n_students)
        
        for i in range(n_students):
            abilities = temporal_data['student_abilities'][i]
            
            if method == 'last':
                # Take the last ability value
                student_abilities[i] = abilities[-1]
            elif method == 'average':
                # Take temporal average
                student_abilities[i] = abilities.mean()
            else:
                raise ValueError(f"Unknown method: {method}")
        
        return student_abilities
    
    def extract_item_parameters_temporal(self, temporal_data, method='average'):
        """Extract item parameters with temporal awareness.
        
        Args:
            temporal_data: Temporal parameter data
            method: 'average' for temporal average, 'last' for last occurrence
        """
        # Find max question ID
        max_q_id = 0
        for q_ids in temporal_data['question_ids']:
            max_q_id = max(max_q_id, int(np.max(q_ids)))
        
        n_questions = max_q_id + 1
        n_cats = temporal_data['item_thresholds'][0].shape[-1] + 1
        
        if method == 'average':
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
                    
        elif method == 'last':
            # Track last occurrence of each question
            item_alphas = np.zeros(n_questions)
            item_betas = np.zeros((n_questions, n_cats - 1))
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
        
        return item_alphas, item_betas, item_counts
    
    def normalize_parameters(self, alphas, betas, prior_alpha_mean=1.0, prior_alpha_std=0.5,
                           prior_beta_mean=0.0, prior_beta_std=1.0):
        """Normalize parameters to standard IRT scale."""
        # Normalize discriminations (log-normal prior)
        alphas_norm = np.exp((np.log(alphas + 1e-6) - np.mean(np.log(alphas + 1e-6))) / 
                            np.std(np.log(alphas + 1e-6)) * prior_alpha_std + np.log(prior_alpha_mean))
        
        # Normalize thresholds (normal prior)
        betas_norm = np.zeros_like(betas)
        for k in range(betas.shape[1]):
            betas_norm[:, k] = (betas[:, k] - np.mean(betas[:, k])) / np.std(betas[:, k]) * prior_beta_std + prior_beta_mean
        
        return alphas_norm, betas_norm
    
    def normalize_abilities(self, abilities, prior_mean=0.0, prior_std=1.0):
        """Normalize student abilities to standard scale."""
        return (abilities - np.mean(abilities)) / np.std(abilities) * prior_std + prior_mean
    
    def calculate_all_correlations(self, true_params, temporal_data, theta_method='last', item_method='average'):
        """Calculate correlations for all parameters including theta."""
        results = {}
        
        # Extract student abilities
        learned_thetas = self.extract_student_abilities(temporal_data, method=theta_method)
        true_thetas = np.array(true_params['student_abilities']['theta'])
        
        # Only use students that exist in both
        n_students = min(len(learned_thetas), len(true_thetas))
        learned_thetas = learned_thetas[:n_students]
        true_thetas = true_thetas[:n_students]
        
        # Normalize abilities
        learned_thetas_norm = self.normalize_abilities(learned_thetas)
        true_thetas_norm = self.normalize_abilities(true_thetas)
        
        # Calculate theta correlation
        theta_corr = np.corrcoef(true_thetas_norm, learned_thetas_norm)[0, 1]
        
        # Extract item parameters
        learned_alphas, learned_betas, item_counts = self.extract_item_parameters_temporal(
            temporal_data, method=item_method
        )
        
        # Get true parameters
        true_alphas = np.array(true_params['question_params']['discrimination']['alpha'])
        true_betas = np.array(true_params['question_params']['difficulties']['beta'])
        
        # Only consider items seen in data
        valid_items = item_counts > 0
        valid_indices = np.where(valid_items)[0]
        
        # Normalize parameters
        true_alphas_norm, true_betas_norm = self.normalize_parameters(true_alphas, true_betas)
        learned_alphas_norm, learned_betas_norm = self.normalize_parameters(
            learned_alphas[valid_items], learned_betas[valid_items]
        )
        
        # Calculate correlations
        alpha_corr = np.corrcoef(true_alphas_norm[valid_indices], learned_alphas_norm)[0, 1]
        beta_corrs = [np.corrcoef(true_betas_norm[valid_indices, k], learned_betas_norm[:, k])[0, 1] 
                     for k in range(3)]
        
        results = {
            'theta': {
                'correlation': theta_corr,
                'n_students': n_students,
                'true_norm': true_thetas_norm,
                'learned_norm': learned_thetas_norm
            },
            'alpha': {
                'correlation': alpha_corr,
                'n_items': len(valid_indices),
                'true_norm': true_alphas_norm,
                'learned_norm': learned_alphas_norm,
                'valid_indices': valid_indices
            },
            'beta': {
                'correlations': beta_corrs,
                'avg_correlation': np.mean(beta_corrs),
                'true_norm': true_betas_norm,
                'learned_norm': learned_betas_norm,
                'valid_indices': valid_indices
            }
        }
        
        return results
    
    def plot_enhanced_comparison(self, model_results, save_path):
        """Create enhanced comparison plots including theta."""
        n_models = len(model_results)
        fig, axes = plt.subplots(n_models, 5, figsize=(20, 4 * n_models))
        
        if n_models == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (model_name, results) in enumerate(model_results.items()):
            corr_data = results['correlations']
            
            # Student ability (theta) plot
            ax = axes[idx, 0]
            ax.scatter(corr_data['theta']['true_norm'], 
                      corr_data['theta']['learned_norm'], alpha=0.6)
            ax.plot([-3, 3], [-3, 3], 'r--', label='Perfect recovery')
            ax.set_xlabel('True θ')
            ax.set_ylabel('Learned θ')
            ax.set_title(f'{model_name.upper()}\nStudent Ability (r={corr_data["theta"]["correlation"]:.3f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Discrimination plot
            ax = axes[idx, 1]
            valid_idx = corr_data['alpha']['valid_indices']
            ax.scatter(corr_data['alpha']['true_norm'][valid_idx], 
                      corr_data['alpha']['learned_norm'], alpha=0.6)
            ax.plot([0, 3], [0, 3], 'r--', label='Perfect recovery')
            ax.set_xlabel('True α')
            ax.set_ylabel('Learned α')
            ax.set_title(f'Discrimination (r={corr_data["alpha"]["correlation"]:.3f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Threshold plots
            for k in range(3):
                ax = axes[idx, k + 2]
                valid_idx = corr_data['beta']['valid_indices']
                ax.scatter(corr_data['beta']['true_norm'][valid_idx, k], 
                          corr_data['beta']['learned_norm'][:, k], alpha=0.6)
                ax.plot([-3, 3], [-3, 3], 'r--', label='Perfect recovery')
                ax.set_xlabel(f'True β_{k}')
                ax.set_ylabel(f'Learned β_{k}')
                ax.set_title(f'Threshold {k} (r={corr_data["beta"]["correlations"][k]:.3f})')
                ax.legend()
                ax.grid(True, alpha=0.3)
        
        plt.suptitle('Enhanced IRT Parameter Recovery Comparison\n' +
                    'All parameters normalized with standard IRT priors',
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    
    def generate_temporal_analysis(self, model_results, temporal_data_dict):
        """Generate temporal analysis showing parameter evolution."""
        fig, axes = plt.subplots(len(model_results), 3, figsize=(15, 4 * len(model_results)))
        
        if len(model_results) == 1:
            axes = axes.reshape(1, -1)
        
        for idx, (model_name, results) in enumerate(model_results.items()):
            temporal_data = temporal_data_dict[model_name]
            
            # Plot 1: Student ability evolution
            ax = axes[idx, 0]
            # Sample a few students to show trajectories
            n_samples = min(10, len(temporal_data['student_abilities']))
            for i in range(n_samples):
                abilities = temporal_data['student_abilities'][i]
                ax.plot(abilities, alpha=0.5, linewidth=1)
            ax.set_xlabel('Time Step')
            ax.set_ylabel('Student Ability (θ)')
            ax.set_title(f'{model_name.upper()}: Student Ability Evolution')
            ax.grid(True, alpha=0.3)
            
            # Plot 2: Average ability trajectory
            ax = axes[idx, 1]
            # Calculate average trajectory
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
            
            ax.plot(avg_trajectory, 'b-', linewidth=2, label='Mean')
            ax.fill_between(range(len(avg_trajectory)), 
                           avg_trajectory - std_trajectory,
                           avg_trajectory + std_trajectory,
                           alpha=0.3, label='±1 SD')
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
        plt.savefig(self.output_dir / 'temporal_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
    
    def run_analysis(self, theta_method='last', item_method='average'):
        """Run complete temporal IRT analysis pipeline."""
        print("\n" + "="*80)
        print("TEMPORAL IRT ANALYSIS PIPELINE")
        print("="*80)
        print(f"Student ability method: {theta_method}")
        print(f"Item parameter method: {item_method}")
        
        # Find models
        models = self.find_models()
        if not models:
            print("No trained models found!")
            return
        
        # Extract parameters from each model
        model_results = {}
        temporal_data_dict = {}
        
        for model_name, model_path in models.items():
            print(f"\nProcessing {model_name}...")
            
            try:
                # Extract temporal parameters
                temporal_data = self.extract_temporal_parameters(model_path)
                temporal_data_dict[model_name] = temporal_data
                
                # Calculate correlations if true parameters available
                if self.true_params:
                    correlations = self.calculate_all_correlations(
                        self.true_params, temporal_data, 
                        theta_method=theta_method, 
                        item_method=item_method
                    )
                    model_results[model_name] = {
                        'temporal_data': temporal_data,
                        'correlations': correlations
                    }
                    print(f"  Correlations calculated:")
                    print(f"    θ (student ability): {correlations['theta']['correlation']:.3f}")
                    print(f"    α (discrimination): {correlations['alpha']['correlation']:.3f}")
                    print(f"    β (thresholds) avg: {correlations['beta']['avg_correlation']:.3f}")
                else:
                    model_results[model_name] = {'temporal_data': temporal_data}
            except Exception as e:
                print(f"  Error processing {model_name}: {str(e)}")
                print(f"  Skipping this model...")
        
        # Create enhanced comparison plots
        if self.true_params and model_results:
            plot_path = self.output_dir / 'enhanced_parameter_recovery.png'
            self.plot_enhanced_comparison(model_results, plot_path)
            print(f"\nEnhanced comparison plot saved to: {plot_path}")
            
            # Generate temporal analysis
            self.generate_temporal_analysis(model_results, temporal_data_dict)
            print(f"Temporal analysis saved to: {self.output_dir / 'temporal_analysis.png'}")
        
        # Generate summary report
        self.generate_summary(model_results, theta_method, item_method)
        
        print("\nAnalysis complete!")
        print("="*80)
    
    def generate_summary(self, model_results, theta_method, item_method):
        """Generate enhanced summary report."""
        summary_path = self.output_dir / 'temporal_irt_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("="*80 + "\n")
            f.write("TEMPORAL IRT PARAMETER RECOVERY ANALYSIS\n")
            f.write("="*80 + "\n\n")
            f.write(f"Dataset: {self.dataset}\n")
            f.write(f"Models analyzed: {len(model_results)}\n")
            f.write(f"\nMethods used:\n")
            f.write(f"- Student ability (θ): {theta_method}\n")
            f.write(f"- Item parameters (α, β): {item_method}\n")
            f.write("\nNormalization applied:\n")
            f.write("- Student ability (θ): Normal prior with mean=0.0, std=1.0\n")
            f.write("- Discrimination (α): Log-normal prior with mean=1.0, std=0.5\n")
            f.write("- Thresholds (β): Normal prior with mean=0.0, std=1.0\n")
            f.write("\n" + "-"*80 + "\n")
            
            for model_name, results in model_results.items():
                if 'correlations' in results:
                    corr = results['correlations']
                    
                    f.write(f"\nMODEL: {model_name.upper()}\n")
                    f.write(f"\n  Student Ability (θ) Recovery:\n")
                    f.write(f"    Correlation: {corr['theta']['correlation']:.4f}\n")
                    f.write(f"    Students analyzed: {corr['theta']['n_students']}\n")
                    
                    f.write(f"\n  Item Parameter Recovery:\n")
                    f.write(f"    Discrimination (α): {corr['alpha']['correlation']:.4f}\n")
                    f.write(f"    Threshold β₀: {corr['beta']['correlations'][0]:.4f}\n")
                    f.write(f"    Threshold β₁: {corr['beta']['correlations'][1]:.4f}\n")
                    f.write(f"    Threshold β₂: {corr['beta']['correlations'][2]:.4f}\n")
                    f.write(f"    Average β correlation: {corr['beta']['avg_correlation']:.4f}\n")
                    f.write(f"    Items analyzed: {corr['alpha']['n_items']}/{self.n_questions}\n")
            
            f.write("\n" + "-"*80 + "\n")
            f.write("\nKEY INSIGHTS:\n")
            f.write("- Student abilities (θ) are TEMPORAL and evolve as students progress\n")
            f.write("- Item parameters (α, β) are learned as temporal but represent question properties\n")
            f.write("- The temporal nature allows capturing learning dynamics\n")
            f.write("- Different aggregation methods (last vs average) affect correlation results\n")
            f.write("="*80 + "\n")
        
        print(f"\nSummary saved to: {summary_path}")


def main():
    parser = argparse.ArgumentParser(description='Temporal IRT analysis tool')
    parser.add_argument('--dataset', default='synthetic_OC', help='Dataset name')
    parser.add_argument('--output_dir', default='results/irt_temporal', 
                        help='Output directory for results')
    parser.add_argument('--theta_method', default='last', choices=['last', 'average'],
                        help='Method for extracting student abilities')
    parser.add_argument('--item_method', default='average', choices=['average', 'last'],
                        help='Method for extracting item parameters')
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = TemporalIRTAnalyzer(dataset=args.dataset, output_dir=args.output_dir)
    
    # Run analysis
    analyzer.run_analysis(theta_method=args.theta_method, item_method=args.item_method)


if __name__ == "__main__":
    main()