#!/usr/bin/env python3
"""
Optimized IRT Analysis System for Deep-GPCM Pipeline
Maintains all original IRT analysis functionality with factory integration and configuration system.
"""

import os
import sys
import torch
import torch.nn.functional as F
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Set backend for consistent rendering
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
from glob import glob
from scipy import stats
from matplotlib.patches import Ellipse
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import original IRT analysis functionality
from analysis.irt_analysis import UnifiedIRTAnalyzer as OriginalIRTAnalyzer

# Import optimized configuration system
from config.evaluation import IRTAnalysisConfig
from config.parser import SmartArgumentParser
from models.implementations import DeepGPCM, AttentionGPCM, EnhancedAttentionGPCM
from models.implementations.coral_gpcm_proper import CORALGPCM
from models.implementations.coral_gpcm_fixed import CORALGPCMFixed
from utils.data_loading import load_simple_data, get_data_file_paths
from utils.irt_utils import extract_effective_thresholds, extract_irt_parameters, summarize_irt_parameters
from utils.path_utils import get_path_manager, find_best_model

# Set style
plt.style.use('default')
sns.set_palette("husl")

# Configure matplotlib for better font rendering
import matplotlib.font_manager as fm
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
plt.rcParams['font.weight'] = 'normal'

# Force matplotlib to use a font that supports bold
available_fonts = [f.name for f in fm.fontManager.ttflist]
if 'Arial' in available_fonts:
    plt.rcParams['font.sans-serif'].insert(0, 'Arial')
elif 'Liberation Sans' in available_fonts:
    plt.rcParams['font.sans-serif'].insert(0, 'Liberation Sans')


class OptimizedIRTAnalyzer(OriginalIRTAnalyzer):
    """
    Optimized IRT analyzer that extends the original with configuration system integration.
    Maintains all original IRT analysis functionality while adding intelligent defaults.
    """
    
    def __init__(self, config: IRTAnalysisConfig):
        """Initialize with optimized configuration."""
        self.config = config
        
        # Initialize parent with configuration values
        super().__init__(
            dataset=config.dataset,
            output_dir=str(config.output_dir)
        )
        
        # Override settings with config values
        self.analysis_types = config.analysis_types
        self.extract_parameters = config.extract_parameters
        self.n_students_temporal = config.n_students_temporal
        self.student_selection_method = config.student_selection_method
        self.correlation_metrics = config.correlation_metrics
        self.plot_recovery = config.plot_recovery
        self.plot_heatmaps = config.plot_heatmaps
        self.plot_trajectories = config.plot_trajectories
        self.plot_distributions = config.plot_distributions
        self.save_parameters = config.save_parameters
        self.save_summary = config.save_summary
        
        # Override parent's load_model method to use saved model dimensions
        # This ensures we use the same dimensions the model was trained with
    
    def load_model(self, model_path):
        """Load model using saved checkpoint configuration - no hardcoded parameters."""
        device = torch.device('cpu')  # Force CPU to avoid CUDA issues during analysis
        
        try:
            checkpoint = torch.load(model_path, map_location=device)
        except Exception as e:
            print(f"❌ Error loading checkpoint {model_path}: {e}")
            return None
        
        # Extract complete configuration from checkpoint (systematic approach)
        config = checkpoint.get('config', {})
        print(f"    🔍 Config found: {bool(config)}")
        if config:
            print(f"    📋 Config keys: {list(config.keys())}")
        
        # Always extract from root level and model_params as primary source
        n_questions = checkpoint.get('n_questions')
        n_cats = checkpoint.get('n_cats') 
        model_type = checkpoint.get('model_type', self._determine_model_type(model_path))
        
        print(f"    📊 From checkpoint root: n_questions={n_questions}, n_cats={n_cats}, model_type={model_type}")
        
        # Try to extract from model_params as backup
        model_params = checkpoint.get('model_params', {})
        if model_params:
            print(f"    📋 Model params: {model_params}")
            if n_questions is None:
                n_questions = model_params.get('n_questions')
            if n_cats is None:
                n_cats = model_params.get('n_cats')
        
        # Final validation
        if n_questions is None or n_cats is None:
            print(f"⚠️  Could not determine model dimensions from checkpoint {model_path}")
            print(f"    📊 Final: n_questions={n_questions}, n_cats={n_cats}")
            return None
        
        # Merge config with extracted dimensions for architecture parameters
        if config:
            # Extract model type from config if available
            config_model_type = config.get('model', config.get('model_type'))
            if config_model_type:
                model_type = config_model_type
        
        print(f"    📏 Model config: n_questions={n_questions}, n_cats={n_cats}, type={model_type}")
        
        # Extract architecture parameters from saved config or factory defaults
        from models.factory import MODEL_REGISTRY
        
        # Get factory defaults for this model type
        factory_config = MODEL_REGISTRY.get(model_type, {})
        factory_defaults = factory_config.get('default_params', {})
        
        architecture_params = {
            'memory_size': config.get('memory_size', factory_defaults.get('memory_size', 50)),
            'key_dim': config.get('key_dim', factory_defaults.get('key_dim', 50)), 
            'value_dim': config.get('value_dim', factory_defaults.get('value_dim', 200)),
            'final_fc_dim': config.get('final_fc_dim', factory_defaults.get('final_fc_dim', 50)),
            'embed_dim': config.get('embed_dim', factory_defaults.get('embed_dim', 64)),
            'n_heads': config.get('n_heads', factory_defaults.get('n_heads', 4)),
            'n_cycles': config.get('n_cycles', factory_defaults.get('n_cycles', 2)),
            'ability_scale': config.get('ability_scale', factory_defaults.get('ability_scale', 2.0))
        }
        
        print(f"    🏗️  Architecture params: {architecture_params}")
        print(f"    🏭  Using factory defaults: {list(factory_defaults.keys())}")
        
        # Use factory pattern for consistency (preferred approach)
        try:
            from models.factory import create_model as factory_create_model
            
            # Filter parameters based on model type to avoid unexpected keyword arguments
            if model_type == 'deep_gpcm':
                filtered_params = {}  # DeepGPCM only needs n_questions and n_cats
            elif model_type in ['attn_gpcm', 'coral_gpcm_proper', 'coral_gpcm_fixed']:
                # These models support additional architecture parameters
                filtered_params = {k: v for k, v in architecture_params.items() if v is not None}
            else:
                filtered_params = {}
            
            model = factory_create_model(
                model_type, 
                n_questions=n_questions, 
                n_cats=n_cats,
                **filtered_params
            )
            print(f"    ✅ Created model using factory pattern")
        except Exception as e:
            print(f"    ⚠️  Factory creation failed ({e}), using direct instantiation")
            
            # Fallback: direct model creation with saved parameters
            if model_type == 'deep_gpcm':
                model = DeepGPCM(n_questions=n_questions, n_cats=n_cats)
            elif model_type == 'attn_gpcm':
                model = EnhancedAttentionGPCM(
                    n_questions=n_questions,
                    n_cats=n_cats,
                    embed_dim=architecture_params['embed_dim'],
                    memory_size=architecture_params['memory_size'],
                    key_dim=architecture_params['key_dim'],
                    value_dim=architecture_params['value_dim'],
                    final_fc_dim=architecture_params['final_fc_dim'],
                    n_heads=architecture_params['n_heads'],
                    n_cycles=architecture_params['n_cycles'],
                    ability_scale=architecture_params['ability_scale']
                )
            elif model_type == 'coral_gpcm_proper':
                model = CORALGPCM(
                    n_questions=n_questions,
                    n_cats=n_cats,
                    memory_size=architecture_params['memory_size'],
                    key_dim=architecture_params['key_dim'],
                    value_dim=architecture_params['value_dim'],
                    final_fc_dim=architecture_params['final_fc_dim']
                )
            elif model_type == 'coral_gpcm_fixed':
                model = CORALGPCMFixed(
                    n_questions=n_questions,
                    n_cats=n_cats,
                    memory_size=architecture_params['memory_size'],
                    key_dim=architecture_params['key_dim'],
                    value_dim=architecture_params['value_dim'],
                    final_fc_dim=architecture_params['final_fc_dim']
                )
            else:
                # Default fallback
                model = DeepGPCM(n_questions=n_questions, n_cats=n_cats)
        
        # Load state dict
        try:
            state_dict = checkpoint['model_state_dict']
            model.load_state_dict(state_dict, strict=False)
        except Exception as e:
            print(f"    ⚠️  Error loading model state: {e}")
            return None
        
        model.to(device)
        model.eval()
        
        return model
    
    def load_data_for_model(self, n_questions, n_cats):
        """Load data matching the model's expected dimensions."""
        # Override parent's data to match model dimensions
        # The parent class loads data based on file naming, but we need 
        # data that matches the saved model dimensions
        
        dataset = self.config.dataset
        
        # First try to load data that matches the model dimensions
        # This handles cases where models were trained on different data sizes
        try:
            # Use unified data loading utility for consistent file path detection
            train_path, test_path = get_data_file_paths(dataset)
            
            if train_path.exists() and test_path.exists():
                print(f"    📂 Loading data from {train_path} and {test_path}")
                train_data, test_data, data_n_questions, data_n_cats = load_simple_data(
                    train_path, test_path
                )
                
                print(f"    📏 Data dimensions: n_questions={data_n_questions}, n_cats={data_n_cats}")
                print(f"    📏 Model dimensions: n_questions={n_questions}, n_cats={n_cats}")
                
                # If dimensions don't match, we need to handle this
                if data_n_questions != n_questions or data_n_cats != n_cats:
                    print(f"    ⚠️  Data/model dimension mismatch. Using model dimensions for analysis.")
                    print(f"    💡 This means the model was trained on different data than currently available.")
                    
                    # For IRT analysis, we still need some data structure for the analysis
                    # but we'll rely on the model's saved parameters rather than recomputing from data
                        
                    return train_data, test_data, n_questions, n_cats
            
            # If no data files found, fall back to parent's data but with model dimensions
            print(f"    ⚠️  No matching data files found, using parent data with model dimensions")
            return self.train_data, self.test_data, n_questions, n_cats
            
        except Exception as e:
            print(f"    ⚠️  Error loading data: {e}, using parent data with model dimensions")
            return self.train_data, self.test_data, n_questions, n_cats
        
    def run_analysis_optimized(self) -> dict:
        """
        Optimized version of run_analysis with configuration-driven execution.
        Maintains all original functionality while respecting configuration settings.
        """
        print("🧠 Starting optimized IRT analysis system...")
        print(f"📊 Dataset: {self.dataset}")
        print(f"📁 Output directory: {self.output_dir}")
        print(f"🎯 Analysis types: {', '.join(self.analysis_types)}")
        print(f"🔬 Extract parameters: {', '.join(self.extract_parameters)}")
        
        # Find available models
        available_models = self.find_models()
        print(f"🤖 Found {len(available_models)} trained models")
        
        all_results = {}
        
        try:
            # Core analysis based on configuration
            if 'recovery' in self.analysis_types:
                recovery_results = self._run_recovery_analysis(available_models)
                all_results['recovery'] = recovery_results
            
            if 'temporal' in self.analysis_types:
                temporal_results = self._run_temporal_analysis(available_models)
                all_results['temporal'] = temporal_results
            
            if 'difficulty' in self.analysis_types:
                difficulty_results = self._run_difficulty_analysis(available_models)
                all_results['difficulty'] = difficulty_results
            
            if 'discrimination' in self.analysis_types:
                discrimination_results = self._run_discrimination_analysis(available_models)
                all_results['discrimination'] = discrimination_results
            
            # Generate summary if requested
            if self.save_summary:
                summary_path = self._generate_summary_report(all_results)
                all_results['summary_path'] = summary_path
            
        except Exception as e:
            print(f"⚠️  Error during IRT analysis: {e}")
            import traceback
            traceback.print_exc()
        
        # Summary
        total_plots = sum(len(v.get('plots', [])) if isinstance(v, dict) else 0 
                         for v in all_results.values())
        print(f"\n✅ IRT analysis complete:")
        print(f"  📈 Analysis types completed: {len([k for k in all_results.keys() if k != 'summary_path'])}")
        print(f"  📊 Total plots generated: {total_plots}")
        
        return all_results
    
    def _run_recovery_analysis(self, available_models) -> dict:
        """Run parameter recovery analysis."""
        print("🔍 Running parameter recovery analysis...")
        
        model_results = {}
        recovery_plots = []
        
        # Extract parameters from all models
        for model_name, model_path in available_models.items():
            print(f"  📊 Analyzing model: {model_name}")
            
            try:
                # Load model and extract parameters
                model = self.load_model(model_path)
                model_type = self._determine_model_type(model_path)
                device = torch.device('cpu')  # Force CPU to match model loading device
                
                # Ensure model is on correct device
                if model is not None:
                    model = model.to(device)
                
                # Extract temporal parameters
                temporal_data = self.extract_temporal_parameters(model, model_type, device, 'test')
                
                # Extract aggregated parameters
                learned_params = self.extract_aggregated_parameters(
                    temporal_data, 
                    theta_method=getattr(self.config, 'theta_method', 'last'),
                    item_method=getattr(self.config, 'item_method', 'average')
                )
                
                # Normalize parameters
                learned_params = self.normalize_parameters(
                    learned_params['alphas'], 
                    learned_params['betas'], 
                    learned_params.get('thetas')
                )
                
                model_results[model_name] = {
                    'learned_params': learned_params,
                    'temporal_data': temporal_data
                }
                
                # Save parameters if requested
                if self.save_parameters:
                    param_file = self.output_dir / f'{model_name}_parameters.json'
                    self._save_parameters(learned_params, param_file)
                
            except Exception as e:
                print(f"    ⚠️  Error analyzing {model_name}: {e}")
                continue
        
        # Generate recovery plots if we have true parameters
        if self.plot_recovery and self.true_params and model_results:
            print("  📈 Generating parameter recovery plots...")
            try:
                recovery_plot_path = self.output_dir / 'param_recovery.png'
                self.plot_parameter_recovery(model_results, recovery_plot_path)
                recovery_plots.append(recovery_plot_path)
                
                # Additional recovery analysis plots
                if len(model_results) > 1:
                    rank_plot_path = self.output_dir / 'temporal_rank_rank.png'
                    temporal_data_dict = {k: v['temporal_data'] for k, v in model_results.items()}
                    self.plot_rank_rank_heatmap(temporal_data_dict, rank_plot_path)
                    recovery_plots.append(rank_plot_path)
                
            except Exception as e:
                print(f"    ⚠️  Error generating recovery plots: {e}")
        
        return {
            'model_results': model_results,
            'plots': recovery_plots,
            'correlations': self._calculate_recovery_correlations(model_results) if self.true_params else None
        }
    
    def _run_temporal_analysis(self, available_models) -> dict:
        """Run temporal analysis of student abilities and item parameters."""
        print("⏰ Running temporal analysis...")
        
        temporal_data_dict = {}
        temporal_plots = []
        
        # Extract temporal data from all models
        for model_name, model_path in available_models.items():
            print(f"  📊 Analyzing temporal data for: {model_name}")
            
            try:
                model = self.load_model(model_path)
                model_type = self._determine_model_type(model_path)
                device = torch.device('cpu')  # Force CPU to match model loading device
                
                # Ensure model is on correct device
                if model is not None:
                    model = model.to(device)
                
                temporal_data = self.extract_temporal_parameters(model, model_type, device, 'test')
                temporal_data_dict[model_name] = temporal_data
                
            except Exception as e:
                print(f"    ⚠️  Error extracting temporal data for {model_name}: {e}")
                continue
        
        if not temporal_data_dict:
            print("    ⚠️  No temporal data extracted")
            return {'plots': [], 'temporal_data': {}}
        
        # Generate temporal plots based on configuration
        try:
            if self.plot_trajectories:
                print("  📈 Generating temporal trajectory plots...")
                temporal_plot_path = self.output_dir / 'temporal.png'
                best_model = self._select_best_model(temporal_data_dict)
                self.plot_temporal_analysis(temporal_data_dict, temporal_plot_path, best_model)
                temporal_plots.append(temporal_plot_path)
            
            if self.plot_heatmaps:
                print("  🔥 Generating temporal heatmaps...")
                
                # Theta heatmap
                theta_heatmap_path = self.output_dir / 'theta_heatmap.png'
                best_model = self._select_best_model(temporal_data_dict)
                self.plot_temporal_theta_heatmap(temporal_data_dict, theta_heatmap_path, best_model)
                temporal_plots.append(theta_heatmap_path)
                
                # GPCM probabilities heatmap
                gpcm_heatmap_path = self.output_dir / 'gpcm_probs_heatmap.png'
                self.plot_temporal_gpcm_probs_heatmap(temporal_data_dict, gpcm_heatmap_path, best_model)
                temporal_plots.append(gpcm_heatmap_path)
            
            # Combined parameters plot
            params_plot_path = self.output_dir / 'params_combined.png'
            best_model = self._select_best_model(temporal_data_dict)
            self.plot_temporal_parameters_combined(temporal_data_dict, params_plot_path, best_model)
            temporal_plots.append(params_plot_path)
            
        except Exception as e:
            print(f"    ⚠️  Error generating temporal plots: {e}")
        
        return {
            'temporal_data': temporal_data_dict,
            'plots': temporal_plots,
            'best_model': self._select_best_model(temporal_data_dict)
        }
    
    def _run_difficulty_analysis(self, available_models) -> dict:
        """Run item difficulty analysis."""
        print("📏 Running difficulty analysis...")
        
        difficulty_results = {}
        difficulty_plots = []
        
        for model_name, model_path in available_models.items():
            
            try:
                model = self.load_model(model_path)
                model_type = self._determine_model_type(model_path)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Extract difficulty parameters (betas)
                temporal_data = self.extract_temporal_parameters(model, model_type, device, 'test')
                learned_params = self.extract_aggregated_parameters(temporal_data, item_method='average')
                
                difficulty_results[model_name] = {
                    'betas': learned_params['betas'],
                    'difficulty_order': np.argsort(learned_params['betas'])
                }
                
            except Exception as e:
                print(f"    ⚠️  Error analyzing difficulty for {model_name}: {e}")
                continue
        
        # Generate difficulty comparison plots if multiple models
        if len(difficulty_results) > 1 and self.plot_distributions:
            try:
                difficulty_plot_path = self.output_dir / 'difficulty_comparison.png'
                self._plot_difficulty_comparison(difficulty_results, difficulty_plot_path)
                difficulty_plots.append(difficulty_plot_path)
            except Exception as e:
                print(f"    ⚠️  Error generating difficulty plots: {e}")
        
        return {
            'difficulty_results': difficulty_results,
            'plots': difficulty_plots
        }
    
    def _run_discrimination_analysis(self, available_models) -> dict:
        """Run item discrimination analysis."""
        print("🎯 Running discrimination analysis...")
        
        discrimination_results = {}
        discrimination_plots = []
        
        for model_name, model_path in available_models.items():
            
            try:
                model = self.load_model(model_path)
                model_type = self._determine_model_type(model_path)
                device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
                
                # Extract discrimination parameters (alphas)
                temporal_data = self.extract_temporal_parameters(model, model_type, device, 'test')
                learned_params = self.extract_aggregated_parameters(temporal_data, item_method='average')
                
                discrimination_results[model_name] = {
                    'alphas': learned_params['alphas'],
                    'discrimination_order': np.argsort(learned_params['alphas'])[::-1]  # High to low
                }
                
            except Exception as e:
                print(f"    ⚠️  Error analyzing discrimination for {model_name}: {e}")
                continue
        
        # Generate discrimination comparison plots
        if len(discrimination_results) > 1 and self.plot_distributions:
            try:
                discrimination_plot_path = self.output_dir / 'discrimination_comparison.png'
                self._plot_discrimination_comparison(discrimination_results, discrimination_plot_path)
                discrimination_plots.append(discrimination_plot_path)
            except Exception as e:
                print(f"    ⚠️  Error generating discrimination plots: {e}")
        
        return {
            'discrimination_results': discrimination_results,
            'plots': discrimination_plots
        }
    
    def _extract_model_name(self, model_path):
        """Extract model name from path."""
        path = Path(model_path)
        # Extract from best_model_name.pth -> model_name
        name = path.stem
        if name.startswith('best_'):
            name = name[5:]  # Remove 'best_' prefix
        return name
    
    def _determine_model_type(self, model_path):
        """Determine model type from path."""
        model_name = self._extract_model_name(model_path)
        
        if 'deep_gpcm' in model_name.lower():
            return 'deep_gpcm'
        elif 'attn_gpcm' in model_name.lower():
            return 'attn_gpcm'
        elif 'coral_gpcm_proper' in model_name.lower():
            return 'coral_gpcm_proper'
        elif 'coral_gpcm_fixed' in model_name.lower():
            return 'coral_gpcm_fixed'
        else:
            return 'deep_gpcm'  # Default fallback
    
    def _select_best_model(self, temporal_data_dict):
        """Select best model based on available criteria."""
        if not temporal_data_dict:
            return None
        
        # Simple heuristic: prefer models with more complete temporal data
        best_model = None
        max_completeness = 0
        
        for model_name, data in temporal_data_dict.items():
            completeness = 0
            if 'thetas' in data and data['thetas'] is not None:
                completeness += len(data['thetas'])
            if 'alphas' in data and data['alphas'] is not None:
                completeness += len(data['alphas'])
            
            if completeness > max_completeness:
                max_completeness = completeness
                best_model = model_name
        
        return best_model
    
    def _calculate_recovery_correlations(self, model_results):
        """Calculate parameter recovery correlations."""
        if not self.true_params:
            return None
        
        correlations = {}
        
        for model_name, results in model_results.items():
            learned_params = results['learned_params']
            model_correlations = self.calculate_correlations(self.true_params, learned_params)
            correlations[model_name] = model_correlations
        
        return correlations
    
    def _plot_difficulty_comparison(self, difficulty_results, save_path):
        """Plot difficulty parameter comparison across models."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Difficulty distributions
        for model_name, results in difficulty_results.items():
            betas = results['betas']
            color = self.get_model_color(model_name)
            axes[0].hist(betas, alpha=0.6, label=model_name, color=color, bins=20)
        
        axes[0].set_xlabel('Difficulty (β)')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Item Difficulty Distributions')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Difficulty rank comparison
        if len(difficulty_results) >= 2:
            model_names = list(difficulty_results.keys())
            model1, model2 = model_names[0], model_names[1]
            
            order1 = difficulty_results[model1]['difficulty_order']
            order2 = difficulty_results[model2]['difficulty_order']
            
            # Create rank mapping
            rank1 = np.argsort(order1)
            rank2 = np.argsort(order2)
            
            axes[1].scatter(rank1, rank2, alpha=0.6)
            axes[1].plot([0, len(rank1)], [0, len(rank1)], 'r--', alpha=0.5)
            axes[1].set_xlabel(f'{model1} Difficulty Rank')
            axes[1].set_ylabel(f'{model2} Difficulty Rank')
            axes[1].set_title('Difficulty Rank Correlation')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    📊 Difficulty comparison plot saved: {save_path}")
    
    def _plot_discrimination_comparison(self, discrimination_results, save_path):
        """Plot discrimination parameter comparison across models."""
        fig, axes = plt.subplots(1, 2, figsize=(15, 6))
        
        # Plot 1: Discrimination distributions
        for model_name, results in discrimination_results.items():
            alphas = results['alphas']
            color = self.get_model_color(model_name)
            axes[0].hist(alphas, alpha=0.6, label=model_name, color=color, bins=20)
        
        axes[0].set_xlabel('Discrimination (α)')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Item Discrimination Distributions')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Discrimination rank comparison
        if len(discrimination_results) >= 2:
            model_names = list(discrimination_results.keys())
            model1, model2 = model_names[0], model_names[1]
            
            order1 = discrimination_results[model1]['discrimination_order']
            order2 = discrimination_results[model2]['discrimination_order']
            
            # Create rank mapping
            rank1 = np.argsort(order1)
            rank2 = np.argsort(order2)
            
            axes[1].scatter(rank1, rank2, alpha=0.6)
            axes[1].plot([0, len(rank1)], [0, len(rank1)], 'r--', alpha=0.5)
            axes[1].set_xlabel(f'{model1} Discrimination Rank')
            axes[1].set_ylabel(f'{model2} Discrimination Rank')
            axes[1].set_title('Discrimination Rank Correlation')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"    📊 Discrimination comparison plot saved: {save_path}")
    
    def _save_parameters(self, parameters, file_path):
        """Save extracted parameters to JSON file."""
        # Convert numpy arrays to lists for JSON serialization
        json_params = {}
        for key, value in parameters.items():
            if isinstance(value, np.ndarray):
                json_params[key] = value.tolist()
            else:
                json_params[key] = value
        
        with open(file_path, 'w') as f:
            json.dump(json_params, f, indent=2)
        
        print(f"    💾 Parameters saved: {file_path}")
    
    def _generate_summary_report(self, all_results):
        """Generate comprehensive summary report."""
        summary_path = self.output_dir / 'irt_summary.txt'
        
        with open(summary_path, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("DEEP-GPCM IRT ANALYSIS SUMMARY REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Dataset: {self.dataset}\n")
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Analysis Types: {', '.join(self.analysis_types)}\n\n")
            
            # Recovery analysis summary
            if 'recovery' in all_results:
                recovery = all_results['recovery']
                f.write("PARAMETER RECOVERY ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Models analyzed: {len(recovery.get('model_results', {}))}\n")
                f.write(f"Plots generated: {len(recovery.get('plots', []))}\n")
                
                if recovery.get('correlations'):
                    f.write("\nParameter Correlations:\n")
                    for model, corrs in recovery['correlations'].items():
                        f.write(f"  {model}:\n")
                        for param, corr in corrs.items():
                            f.write(f"    {param}: {corr:.3f}\n")
                f.write("\n")
            
            # Temporal analysis summary
            if 'temporal' in all_results:
                temporal = all_results['temporal']
                f.write("TEMPORAL ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Models analyzed: {len(temporal.get('temporal_data', {}))}\n")
                f.write(f"Plots generated: {len(temporal.get('plots', []))}\n")
                f.write(f"Best model: {temporal.get('best_model', 'N/A')}\n\n")
            
            # Difficulty analysis summary
            if 'difficulty' in all_results:
                difficulty = all_results['difficulty']
                f.write("DIFFICULTY ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Models analyzed: {len(difficulty.get('difficulty_results', {}))}\n")
                f.write(f"Plots generated: {len(difficulty.get('plots', []))}\n\n")
            
            # Discrimination analysis summary
            if 'discrimination' in all_results:
                discrimination = all_results['discrimination']
                f.write("DISCRIMINATION ANALYSIS\n")
                f.write("-" * 40 + "\n")
                f.write(f"Models analyzed: {len(discrimination.get('discrimination_results', {}))}\n")
                f.write(f"Plots generated: {len(discrimination.get('plots', []))}\n\n")
            
            f.write("=" * 80 + "\n")
            f.write("Analysis completed successfully!\n")
        
        print(f"📋 Summary report saved: {summary_path}")
        return summary_path


class IRTOrchestrator:
    """Orchestrates IRT analysis workflows with configuration management."""
    
    def __init__(self, config: IRTAnalysisConfig = None):
        self.config = config or IRTAnalysisConfig()
        self.analyzer = OptimizedIRTAnalyzer(self.config)
    
    def analyze_dataset(self, dataset: str) -> dict:
        """Analyze IRT parameters for a specific dataset."""
        print(f"🧠 Running IRT analysis for {dataset}")
        
        # Update config for specific dataset
        dataset_config = IRTAnalysisConfig(
            dataset=dataset,
            output_dir=self.config.output_dir,
            analysis_types=self.config.analysis_types
        )
        
        analyzer = OptimizedIRTAnalyzer(dataset_config)
        return analyzer.run_analysis_optimized()
    
    def batch_analysis(self, datasets: list, models: list = None) -> dict:
        """Run IRT analysis across multiple datasets."""
        print(f"📊 Running batch IRT analysis")
        print(f"   Datasets: {', '.join(datasets)}")
        
        all_results = {}
        
        for dataset in datasets:
            print(f"\n🔍 Analyzing dataset: {dataset}")
            try:
                results = self.analyze_dataset(dataset)
                all_results[dataset] = results
            except Exception as e:
                print(f"❌ Error analyzing {dataset}: {e}")
                all_results[dataset] = {'error': str(e)}
        
        return all_results
    
    def generate_comparative_report(self, datasets: list) -> Path:
        """Generate a comparative report across datasets."""
        print(f"📋 Generating comparative IRT report for {len(datasets)} datasets")
        
        report_path = Path("results/irt_plots/comparative_report.html")
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create comparative HTML report
        html_content = self._create_comparative_html_report(datasets)
        
        with open(report_path, 'w') as f:
            f.write(html_content)
        
        print(f"📄 Comparative report saved to: {report_path}")
        return report_path
    
    def _create_comparative_html_report(self, datasets: list) -> str:
        """Create HTML comparative report."""
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Deep-GPCM IRT Analysis Comparative Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 40px; }
                h1 { color: #2E86AB; }
                h2 { color: #43AA8B; }
                .dataset { margin: 20px 0; padding: 20px; border: 1px solid #ddd; }
                .analysis { margin: 10px; }
                .metric { background-color: #f8f9fa; padding: 10px; margin: 5px 0; }
                img { max-width: 600px; height: auto; margin: 10px; }
            </style>
        </head>
        <body>
            <h1>🧠 Deep-GPCM IRT Analysis Comparative Report</h1>
            <p>Generated with optimized IRT analysis pipeline</p>
        """
        
        for dataset in datasets:
            html += f"""
            <div class="dataset">
                <h2>📊 Dataset: {dataset}</h2>
                <div class="analysis">
                    <h3>Parameter Recovery Analysis</h3>
                    <p>Correlation analysis between learned and true parameters</p>
                    <!-- Recovery plots would be embedded here -->
                </div>
                <div class="analysis">
                    <h3>Temporal Analysis</h3>
                    <p>Evolution of student abilities and item parameters over time</p>
                    <!-- Temporal plots would be embedded here -->
                </div>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html


def run_irt_analysis_optimized(config: IRTAnalysisConfig = None) -> dict:
    """
    Optimized convenience function to run IRT analysis.
    Maintains backward compatibility while adding configuration support.
    """
    
    if config is None:
        config = IRTAnalysisConfig(
            analysis_types=['recovery', 'temporal'],
            save_parameters=True,
            save_summary=True
        )
    
    analyzer = OptimizedIRTAnalyzer(config)
    return analyzer.run_analysis_optimized()


def main():
    """Main entry point with optimized argument parsing."""
    
    try:
        # Try to parse as IRT analysis configuration
        parser = SmartArgumentParser.create_plotting_parser()  # Will create based on script name
        
        # For IRT analysis, we need custom arguments
        import argparse
        cli_parser = argparse.ArgumentParser(description='Generate optimized IRT analysis for Deep-GPCM results')
        cli_parser.add_argument('--dataset', type=str, default='synthetic_OC', help='Dataset name')
        cli_parser.add_argument('--output_dir', type=str, default='results/irt_plots', help='Output directory')
        cli_parser.add_argument('--analysis_types', nargs='+', 
                               default=['recovery', 'temporal'],
                               choices=['recovery', 'temporal', 'difficulty', 'discrimination'],
                               help='Types of analysis to perform')
        cli_parser.add_argument('--save_parameters', action='store_true',
                               help='Save extracted parameters to JSON files')
        cli_parser.add_argument('--save_summary', action='store_true', default=True,
                               help='Save analysis summary report')
        
        args = cli_parser.parse_args()
        
        # Create IRT analysis configuration
        config = IRTAnalysisConfig(
            dataset=args.dataset,
            output_dir=Path(args.output_dir),
            analysis_types=args.analysis_types,
            save_parameters=args.save_parameters,
            save_summary=args.save_summary
        )
        
        # Run optimized IRT analysis
        results = run_irt_analysis_optimized(config=config)
        
        total_plots = sum(len(v.get('plots', [])) if isinstance(v, dict) else 0 
                         for v in results.values())
        print(f"\n🎉 Successfully completed IRT analysis!")
        print(f"📊 Total plots generated: {total_plots}")
        
        # Optionally create comparative report
        if config.dataset:
            orchestrator = IRTOrchestrator(config)
            report = orchestrator.generate_comparative_report([config.dataset])
            print(f"📋 Comparative report: {report}")
        
    except Exception as e:
        print(f"❌ Error in optimized IRT analysis: {e}")
        
        # Fallback to original IRT analysis system
        print("🔄 Falling back to original IRT analysis system...")
        
        import argparse
        parser = argparse.ArgumentParser(description='IRT Analysis for Deep-GPCM')
        parser.add_argument('--dataset', default='synthetic_OC', help='Dataset name')
        parser.add_argument('--output_dir', default='results/irt_plots', help='Output directory')
        parser.add_argument('--analysis_types', nargs='+', default=['recovery'],
                           choices=['recovery', 'temporal', 'irt_plots', 'none'],
                           help='Types of analysis to perform')
        
        args = parser.parse_args()
        
        # Use original IRT analysis system as fallback
        from analysis.irt_analysis import UnifiedIRTAnalyzer
        analyzer = UnifiedIRTAnalyzer(dataset=args.dataset, output_dir=args.output_dir)
        analyzer.run_analysis(args)


if __name__ == "__main__":
    main()