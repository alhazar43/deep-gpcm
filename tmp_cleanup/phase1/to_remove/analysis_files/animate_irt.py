"""
Animated IRT Parameter Analysis for Deep-GPCM Models
Extracts and visualizes temporal evolution of IRT parameters during student learning.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.patches import FancyBboxPatch
import seaborn as sns
import argparse
import json
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set headless backend for server environments
import matplotlib
matplotlib.use('Agg')

# Import model classes
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.implementations import DeepGPCM, AttentionGPCM
from utils.data_utils import load_gpcm_data

class TemporalIRTExtractor:
    """Extract temporal IRT parameters from trained models."""
    
    def __init__(self, model_path, data_path="data/synthetic_OC/synthetic_oc_test.txt"):
        self.model_path = model_path
        self.data_path = data_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and data
        self.model, self.model_info = self._load_model()
        self.data_loader = self._load_data()
        
    def _load_model(self):
        """Load trained model with automatic type detection."""
        checkpoint = torch.load(self.model_path, map_location=self.device)
        
        # Extract model info
        model_info = {
            'name': checkpoint.get('model_name', 'unknown'),
            'type': checkpoint.get('model_type', 'unknown'),
            'n_questions': checkpoint.get('n_questions', 30),
            'n_cats': checkpoint.get('n_cats', 4)
        }
        
        # Create model instance based on state dict keys
        state_dict_keys = list(checkpoint['model_state_dict'].keys()) if 'model_state_dict' in checkpoint else list(checkpoint.keys())
        
        # Check for AKVMN-specific keys
        is_akvmn = any('attention_layers' in key or 'fusion_layers' in key or 'cycle_norms' in key 
                      for key in state_dict_keys)
        
        if is_akvmn:
            model = AttentionGPCM(
                n_questions=model_info['n_questions'],
                n_cats=model_info['n_cats']
            )
            model_info['type'] = 'akvmn_gpcm'
        else:
            model = DeepGPCM(
                n_questions=model_info['n_questions'],
                n_cats=model_info['n_cats']
            )
            model_info['type'] = 'baseline_gpcm'
        
        # Load state dict
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        else:
            model.load_state_dict(checkpoint)
            
        model.to(self.device)
        model.eval()
        
        print(f"Loaded {model_info['type']} model with {sum(p.numel() for p in model.parameters())} parameters")
        return model, model_info
    
    def _load_data(self):
        """Load test data."""
        try:
            # Load data using gpcm_utils function - returns tuple
            sequences, questions, responses, n_cats = load_gpcm_data(self.data_path)
            
            # Convert to DataLoader
            from torch.utils.data import TensorDataset, DataLoader
            from utils.gpcm_utils import create_gpcm_batch
            
            # Create batched tensors
            q_batch, r_batch, _ = create_gpcm_batch(questions, responses)
            
            dataset = TensorDataset(q_batch, r_batch)
            dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
            
            return dataloader
        except Exception as e:
            print(f"Warning: Could not load data from {self.data_path}: {e}")
            return None
    
    def extract_temporal_parameters(self, max_sequences=50):
        """
        Extract temporal IRT parameters for animation.
        
        IMPORTANT CLARIFICATION:
        - theta (student ability): TEMPORAL - changes as student progresses
        - alpha (discrimination): STATIC per question - same for each question across students
        - beta (thresholds): STATIC per question - same for each question across students
        
        Returns:
            dict: {
                'theta': List of (seq_len,) arrays for each sequence - TEMPORAL
                'alpha_by_question': Dict mapping question_id -> alpha value - STATIC
                'beta_by_question': Dict mapping question_id -> beta array - STATIC  
                'questions': List of question sequences,
                'responses': List of response sequences,
                'seq_lens': List of sequence lengths,
                'metadata': Model and extraction info
            }
        """
        temporal_data = {
            'theta': [],  # Temporal: changes per student per time step
            'alpha_by_question': {},  # Static: one value per unique question
            'beta_by_question': {},   # Static: one array per unique question
            'questions': [], 'responses': [], 'seq_lens': [],
            'metadata': self.model_info
        }
        
        sequences_processed = 0
        
        with torch.no_grad():
            for batch_idx, (questions, responses) in enumerate(self.data_loader):
                if sequences_processed >= max_sequences:
                    break
                    
                questions = questions.to(self.device)
                responses = responses.to(self.device)
                
                # Get temporal parameters from model forward pass
                student_abilities, item_thresholds, discrimination_params, _ = self.model(questions, responses)
                
                # Process each sequence in batch
                batch_size = questions.shape[0]
                for seq_idx in range(batch_size):
                    if sequences_processed >= max_sequences:
                        break
                    
                    # Get actual sequence length (before padding)
                    seq_len = (questions[seq_idx] != 0).sum().item()
                    if seq_len < 3:  # Skip very short sequences
                        continue
                    
                    # Extract TEMPORAL theta (student ability) - changes over time
                    theta_seq = student_abilities[seq_idx, :seq_len].cpu().numpy()
                    
                    # Extract STATIC question parameters (alpha, beta) - same for each question
                    questions_seq = questions[seq_idx, :seq_len].cpu().numpy()
                    responses_seq = responses[seq_idx, :seq_len].cpu().numpy()
                    
                    # Store question-specific parameters (only once per unique question)
                    for t in range(seq_len):
                        q_id = int(questions_seq[t])
                        if q_id not in temporal_data['alpha_by_question']:
                            # Store alpha and beta for this question (static across students/time)
                            temporal_data['alpha_by_question'][q_id] = discrimination_params[seq_idx, t].cpu().item()
                            temporal_data['beta_by_question'][q_id] = item_thresholds[seq_idx, t, :].cpu().numpy()
                    
                    # Store temporal data
                    temporal_data['theta'].append(theta_seq)
                    temporal_data['questions'].append(questions_seq)
                    temporal_data['responses'].append(responses_seq)
                    temporal_data['seq_lens'].append(seq_len)
                    
                    sequences_processed += 1
        
        print(f"Extracted temporal parameters from {sequences_processed} sequences")
        print(f"Found {len(temporal_data['alpha_by_question'])} unique questions with static alpha/beta parameters")
        return temporal_data

class TemporalIRTAnimator:
    """Create animated visualizations of temporal IRT parameter evolution."""
    
    def __init__(self, temporal_data, output_dir="irt_animations"):
        self.data = temporal_data
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True, parents=True)
        
        # Set up color schemes
        self.colors = {
            'theta': '#2E86AB',    # Blue for student ability
            'alpha': '#A23B72',    # Purple for discrimination  
            'beta': '#F18F01',     # Orange for thresholds
            'correct': '#43AA8B',  # Green for correct responses
            'incorrect': '#F77F00' # Red for incorrect responses
        }
    
    def animate_student_learning_journey(self, sequence_idx=0, save_gif=True):
        """
        Animate a single student's learning journey showing parameter evolution.
        Now correctly shows:
        - theta: temporal student ability evolution
        - alpha: static discrimination per question (shown as bars)
        - beta: static threshold per question (shown as mean values)
        """
        if sequence_idx >= len(self.data['theta']):
            print(f"Sequence {sequence_idx} not available. Max: {len(self.data['theta'])-1}")
            return
        
        theta_seq = self.data['theta'][sequence_idx]
        questions_seq = self.data['questions'][sequence_idx]
        responses_seq = self.data['responses'][sequence_idx]
        seq_len = len(theta_seq)
        
        # Get question-specific static parameters
        alpha_seq = [self.data['alpha_by_question'][q_id] for q_id in questions_seq]
        beta_seq = [np.mean(self.data['beta_by_question'][q_id]) for q_id in questions_seq]
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f"Student Learning Journey - {self.data['metadata']['type'].upper()}", 
                    fontsize=16, fontweight='bold')
        
        # Initialize plots
        time_steps = np.arange(seq_len)
        
        # Plot 1: Student Ability Evolution
        line_theta, = ax1.plot([], [], color=self.colors['theta'], linewidth=3, marker='o', markersize=6)
        ax1.set_xlim(0, seq_len-1)
        ax1.set_ylim(min(theta_seq)-0.5, max(theta_seq)+0.5)
        ax1.set_xlabel('Question Number')
        ax1.set_ylabel('Student Ability (θ)')
        ax1.set_title('Student Ability Evolution')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Discrimination Parameters (STATIC per question)
        ax2.set_xlim(0, seq_len-1)
        ax2.set_ylim(min(alpha_seq)-0.1, max(alpha_seq)+0.1)
        ax2.set_xlabel('Question Number')
        ax2.set_ylabel('Discrimination (α)')
        ax2.set_title('Item Discrimination (Static per Question)')
        ax2.grid(True, alpha=0.3)
        
        # Show static alpha values as bars
        bars_alpha = ax2.bar(range(seq_len), alpha_seq, color=self.colors['alpha'], alpha=0.7, width=0.8)
        
        # Plot 3: Threshold Parameters (STATIC per question, mean values)
        ax3.set_xlim(0, seq_len-1)
        ax3.set_ylim(min(beta_seq)-0.2, max(beta_seq)+0.2)
        ax3.set_xlabel('Question Number')
        ax3.set_ylabel('Mean Threshold (β)')
        ax3.set_title('Item Difficulty (Static per Question)')
        ax3.grid(True, alpha=0.3)
        
        # Show static beta values as bars
        bars_beta = ax3.bar(range(seq_len), beta_seq, color=self.colors['beta'], alpha=0.7, width=0.8)
        
        # Plot 4: Response Pattern
        ax4.set_xlim(0, seq_len-1)
        ax4.set_ylim(-0.5, self.data['metadata']['n_cats']-0.5)
        ax4.set_xlabel('Question Number')
        ax4.set_ylabel('Response Category')
        ax4.set_title('Student Response Pattern')
        ax4.grid(True, alpha=0.3)
        
        # Response scatter plot data (will be updated in animation)
        response_scatter = ax4.scatter([], [], c=[], s=100, alpha=0.7, cmap='RdYlGn')
        
        # Animation function
        def animate(frame):
            if frame == 0:
                # Clear theta line only (alpha/beta are static bars)
                line_theta.set_data([], [])
                return line_theta, response_scatter
            
            # Update data up to current frame
            current_steps = time_steps[:frame+1]
            
            # Update ability evolution (only theta is temporal)
            line_theta.set_data(current_steps, theta_seq[:frame+1])
            
            # Highlight current question's alpha and beta bars
            for i, bar in enumerate(bars_alpha):
                if i <= frame:
                    bar.set_alpha(0.9 if i == frame else 0.4)
                else:
                    bar.set_alpha(0.2)
            
            for i, bar in enumerate(bars_beta):
                if i <= frame:
                    bar.set_alpha(0.9 if i == frame else 0.4)
                else:
                    bar.set_alpha(0.2)
            
            # Update response pattern
            if frame > 0:
                colors = ['red' if r == 0 else 'yellow' if r == 1 else 'lightgreen' if r == 2 else 'green' 
                         for r in responses_seq[:frame+1]]
                response_scatter.set_offsets(np.column_stack((current_steps, responses_seq[:frame+1])))
                response_scatter.set_color(colors)
            
            # Add current statistics as text  
            if frame < seq_len:
                fig.suptitle(f"Student Learning Journey - {self.data['metadata']['type'].upper()}\\n"
                            f"Question {frame+1}/{seq_len} | θ={theta_seq[frame]:.3f} | "
                            f"Q{questions_seq[frame]} α={alpha_seq[frame]:.3f} β̄={beta_seq[frame]:.3f} | Response={responses_seq[frame]}", 
                            fontsize=13, fontweight='bold')
            else:
                fig.suptitle(f"Student Learning Journey - {self.data['metadata']['type'].upper()}\\n"
                            f"Complete - Final θ={theta_seq[-1]:.3f}", 
                            fontsize=14, fontweight='bold')
            
            return line_theta, response_scatter
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=seq_len, interval=800, blit=False, repeat=True
        )
        
        if save_gif:
            output_path = self.output_dir / f"student_journey_seq{sequence_idx}_{self.data['metadata']['type']}.gif"
            print(f"Saving animation to {output_path}")
            anim.save(str(output_path), writer='pillow', fps=1.5)
            print(f"Animation saved: {output_path}")
        
        plt.tight_layout()
        return anim, fig
    
    def animate_parameter_distributions(self, save_gif=True):
        """
        Animate the evolution of parameter distributions across all students.
        Only theta is temporal - alpha/beta are static per question.
        """
        # Prepare data for distribution evolution
        max_seq_len = max(self.data['seq_lens'])
        n_sequences = len(self.data['theta'])
        
        # Create matrices for each time step
        theta_by_time = []
        
        # Get unique questions and their static parameters
        unique_questions = set()
        for questions_seq in self.data['questions']:
            unique_questions.update(questions_seq)
        unique_questions = sorted(list(unique_questions))
        
        static_alphas = [self.data['alpha_by_question'][q_id] for q_id in unique_questions]
        static_betas = [np.mean(self.data['beta_by_question'][q_id]) for q_id in unique_questions]
        
        for t in range(max_seq_len):
            theta_t = []
            
            for seq_idx in range(n_sequences):
                if t < self.data['seq_lens'][seq_idx]:
                    theta_t.append(self.data['theta'][seq_idx][t])
            
            theta_by_time.append(theta_t)
        
        # Create figure - only theta is temporal, show alpha/beta as static
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f"Parameter Distribution Evolution - {self.data['metadata']['type'].upper()}", 
                    fontsize=16, fontweight='bold')
        
        # Get actual data ranges (don't cap theta at [-2,2]!)
        all_theta = [val for seq in theta_by_time for val in seq if seq]
        theta_range = [min(all_theta)-0.5, max(all_theta)+0.5] if all_theta else [-3, 3]
        alpha_range = [min(static_alphas)-0.1, max(static_alphas)+0.1] if static_alphas else [0, 3]
        beta_range = [min(static_betas)-0.2, max(static_betas)+0.2] if static_betas else [-2, 2]
        
        print(f"True theta range: [{theta_range[0]:.2f}, {theta_range[1]:.2f}]")
        print(f"Alpha range: [{alpha_range[0]:.2f}, {alpha_range[1]:.2f}]")
        print(f"Beta range: [{beta_range[0]:.2f}, {beta_range[1]:.2f}]")
        
        def animate(frame):
            # Clear previous plots
            ax1.clear()
            ax2.clear()
            ax3.clear()
            
            if frame < len(theta_by_time) and len(theta_by_time[frame]) > 0:
                # Student Ability Distribution (TEMPORAL)
                ax1.hist(theta_by_time[frame], bins=15, alpha=0.7, color=self.colors['theta'], 
                        density=True, range=theta_range)
                ax1.axvline(np.mean(theta_by_time[frame]), color='red', linestyle='--', linewidth=2,
                           label=f'Mean: {np.mean(theta_by_time[frame]):.3f}')
                ax1.set_xlim(theta_range)
                ax1.set_xlabel('Student Ability (θ)')
                ax1.set_ylabel('Density')
                ax1.set_title(f'TEMPORAL Ability Distribution\\nQuestion {frame+1}')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Discrimination Distribution (STATIC - shows all unique questions)
                ax2.hist(static_alphas, bins=15, alpha=0.7, color=self.colors['alpha'],
                        density=True, range=alpha_range)
                ax2.axvline(np.mean(static_alphas), color='red', linestyle='--', linewidth=2,
                           label=f'Mean: {np.mean(static_alphas):.3f}')
                ax2.set_xlim(alpha_range)
                ax2.set_xlabel('Discrimination (α)')
                ax2.set_ylabel('Density')
                ax2.set_title(f'STATIC Discrimination Distribution\\n{len(unique_questions)} Unique Questions')
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Threshold Distribution (STATIC - shows all unique questions)
                ax3.hist(static_betas, bins=15, alpha=0.7, color=self.colors['beta'],
                        density=True, range=beta_range)
                ax3.axvline(np.mean(static_betas), color='red', linestyle='--', linewidth=2,
                           label=f'Mean: {np.mean(static_betas):.3f}')
                ax3.set_xlim(beta_range)
                ax3.set_xlabel('Mean Threshold (β)')
                ax3.set_ylabel('Density')
                ax3.set_title(f'STATIC Threshold Distribution\\n{len(unique_questions)} Unique Questions')
                ax3.legend()
                ax3.grid(True, alpha=0.3)
            
            plt.tight_layout()
        
        # Create animation
        anim = animation.FuncAnimation(
            fig, animate, frames=max_seq_len, interval=1000, blit=False, repeat=True
        )
        
        if save_gif:
            output_path = self.output_dir / f"parameter_distributions_{self.data['metadata']['type']}.gif"
            print(f"Saving distribution animation to {output_path}")
            anim.save(str(output_path), writer='pillow', fps=1)
            print(f"Distribution animation saved: {output_path}")
        
        return anim, fig
    
    def create_ability_trajectory_heatmap(self, save_plot=True):
        """
        Create a heatmap showing ability trajectories for all students.
        """
        # Prepare data matrix
        max_seq_len = max(self.data['seq_lens'])
        n_sequences = len(self.data['theta'])
        
        # Create ability matrix (students x time)
        ability_matrix = np.full((n_sequences, max_seq_len), np.nan)
        
        for seq_idx in range(n_sequences):
            seq_len = self.data['seq_lens'][seq_idx]
            ability_matrix[seq_idx, :seq_len] = self.data['theta'][seq_idx]
        
        # Create heatmap
        fig, ax = plt.subplots(figsize=(15, 10))
        
        # Use a custom colormap that handles NaN values
        cmap = plt.cm.RdYlBu_r
        cmap.set_bad(color='lightgray')
        
        # Get TRUE data range (don't cap at [-2,2])
        valid_data = ability_matrix[~np.isnan(ability_matrix)]
        true_vmin, true_vmax = np.min(valid_data), np.max(valid_data)
        print(f"True theta heatmap range: [{true_vmin:.2f}, {true_vmax:.2f}]")
        
        im = ax.imshow(ability_matrix, cmap=cmap, aspect='auto', 
                      vmin=true_vmin, vmax=true_vmax, interpolation='nearest')
        
        # Customize plot
        ax.set_xlabel('Question Number', fontsize=12)
        ax.set_ylabel('Student Sequence', fontsize=12)
        ax.set_title(f'Student Ability Trajectories - {self.data["metadata"]["type"].upper()}\\n'
                    f'Temporal Evolution Across {n_sequences} Students', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Student Ability (θ)', fontsize=12)
        
        # Add grid
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mean_final_ability = np.nanmean([seq[-1] for seq in self.data['theta']])
        std_final_ability = np.nanstd([seq[-1] for seq in self.data['theta']])
        
        textstr = f'Final Ability: μ={mean_final_ability:.3f}, σ={std_final_ability:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_plot:
            output_path = self.output_dir / f"ability_trajectories_{self.data['metadata']['type']}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Ability trajectory heatmap saved: {output_path}")
        
        return fig
    
    def create_temporal_summary_stats(self, save_plot=True):
        """
        Create summary statistics plots showing parameter evolution over time.
        Only theta is temporal - alpha/beta are static per question.
        """
        max_seq_len = max(self.data['seq_lens'])
        
        # Calculate statistics for each time step (only theta is temporal)
        theta_stats = {'mean': [], 'std': [], 'count': []}
        
        for t in range(max_seq_len):
            theta_t = [self.data['theta'][i][t] for i in range(len(self.data['theta'])) 
                      if t < self.data['seq_lens'][i]]
            
            if theta_t:
                theta_stats['mean'].append(np.mean(theta_t))
                theta_stats['std'].append(np.std(theta_t))
                theta_stats['count'].append(len(theta_t))
        
        # Get static question statistics
        unique_questions = set()
        for questions_seq in self.data['questions']:
            unique_questions.update(questions_seq)
        unique_questions = sorted(list(unique_questions))
        
        static_alphas = [self.data['alpha_by_question'][q_id] for q_id in unique_questions]
        static_betas = [np.mean(self.data['beta_by_question'][q_id]) for q_id in unique_questions]
        
        # Create plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'Temporal Parameter Statistics - {self.data["metadata"]["type"].upper()}', 
                    fontsize=16, fontweight='bold')
        
        time_steps = np.arange(len(theta_stats['mean']))
        
        # Student Ability Evolution (TEMPORAL)
        ax1.plot(time_steps, theta_stats['mean'], color=self.colors['theta'], linewidth=2, marker='o')
        ax1.fill_between(time_steps, 
                        np.array(theta_stats['mean']) - np.array(theta_stats['std']),
                        np.array(theta_stats['mean']) + np.array(theta_stats['std']),
                        alpha=0.3, color=self.colors['theta'])
        ax1.set_xlabel('Question Number')
        ax1.set_ylabel('Student Ability (θ)')
        ax1.set_title('TEMPORAL: Mean Student Ability Evolution')
        ax1.grid(True, alpha=0.3)
        
        # Discrimination Distribution (STATIC)
        question_ids = np.arange(len(unique_questions))
        ax2.bar(question_ids, static_alphas, color=self.colors['alpha'], alpha=0.7, width=0.8)
        ax2.axhline(np.mean(static_alphas), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(static_alphas):.3f}')
        ax2.set_xlabel('Question ID')
        ax2.set_ylabel('Discrimination (α)')
        ax2.set_title(f'STATIC: Discrimination by Question\\n{len(unique_questions)} Unique Questions')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Threshold Distribution (STATIC)
        ax3.bar(question_ids, static_betas, color=self.colors['beta'], alpha=0.7, width=0.8)
        ax3.axhline(np.mean(static_betas), color='red', linestyle='--', linewidth=2,
                   label=f'Mean: {np.mean(static_betas):.3f}')
        ax3.set_xlabel('Question ID')
        ax3.set_ylabel('Mean Threshold (β)')
        ax3.set_title(f'STATIC: Thresholds by Question\\n{len(unique_questions)} Unique Questions')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Sample Size Evolution
        ax4.plot(time_steps, theta_stats['count'], color='black', linewidth=2, marker='d')
        ax4.set_xlabel('Question Number')
        ax4.set_ylabel('Number of Students')
        ax4.set_title('Sample Size by Question Position')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plot:
            output_path = self.output_dir / f"temporal_summary_stats_{self.data['metadata']['type']}.png"
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            print(f"Temporal summary statistics saved: {output_path}")
        
        return fig

def main():
    parser = argparse.ArgumentParser(description="Animate temporal IRT parameter evolution")
    parser.add_argument('--model_path', required=True, help='Path to trained model')
    parser.add_argument('--data_path', default='data/synthetic_OC/synthetic_oc_test.txt', 
                       help='Path to test data')
    parser.add_argument('--output_dir', default='irt_animations', 
                       help='Output directory for animations')
    parser.add_argument('--max_sequences', type=int, default=50,
                       help='Maximum number of sequences to analyze')
    parser.add_argument('--sequence_idx', type=int, default=0,
                       help='Specific sequence to animate for student journey')
    parser.add_argument('--animation_type', choices=['journey', 'distributions', 'heatmap', 'stats', 'all'], 
                       default='all', help='Type of animation to create')
    
    args = parser.parse_args()
    
    print(f"Extracting temporal IRT parameters from {args.model_path}")
    
    # Extract temporal parameters
    extractor = TemporalIRTExtractor(args.model_path, args.data_path)
    temporal_data = extractor.extract_temporal_parameters(args.max_sequences)
    
    # Create animator
    animator = TemporalIRTAnimator(temporal_data, args.output_dir)
    
    # Generate requested animations
    if args.animation_type in ['journey', 'all']:
        print("Creating student learning journey animation...")
        anim, fig = animator.animate_student_learning_journey(args.sequence_idx)
        plt.close(fig)
    
    if args.animation_type in ['distributions', 'all']:
        print("Creating parameter distributions animation...")
        anim, fig = animator.animate_parameter_distributions()
        plt.close(fig)
    
    if args.animation_type in ['heatmap', 'all']:
        print("Creating ability trajectory heatmap...")
        fig = animator.create_ability_trajectory_heatmap()
        plt.close(fig)
    
    if args.animation_type in ['stats', 'all']:
        print("Creating temporal summary statistics...")
        fig = animator.create_temporal_summary_stats()
        plt.close(fig)
    
    print(f"\\nAll animations completed! Check {args.output_dir}/ for outputs.")

if __name__ == "__main__":
    main()