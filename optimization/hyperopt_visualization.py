"""
Hyperparameter Optimization Visualization and Analysis

Provides comprehensive visualization and analysis tools for hyperparameter optimization results.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from scipy.stats import pearsonr, spearmanr
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

class HyperoptVisualizer:
    """Comprehensive hyperparameter optimization visualization and analysis."""
    
    def __init__(self, results_dir: str = "results/hyperopt"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Set up plotting style
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        
    def analyze_optimization_results(self, optimization_results: Dict[str, Any], 
                                   model_name: str, dataset: str) -> Dict[str, Any]:
        """Analyze hyperparameter optimization results and generate comprehensive report."""
        
        # Extract trial data
        trials_data = self._extract_trials_data(optimization_results)
        
        # Create analysis report
        analysis = {
            'summary': self._create_summary_stats(optimization_results, trials_data),
            'parameter_importance': self._analyze_parameter_importance(trials_data),
            'convergence_analysis': self._analyze_convergence(trials_data),
            'parameter_correlations': self._analyze_parameter_correlations(trials_data),
            'best_parameters': optimization_results.get('best_params', {}),
            'optimization_efficiency': self._analyze_optimization_efficiency(trials_data)
        }
        
        # Generate visualizations
        viz_paths = self._create_visualizations(trials_data, model_name, dataset, analysis)
        analysis['visualization_paths'] = viz_paths
        
        # Save detailed analysis
        analysis_path = self.results_dir / f"{model_name}_{dataset}_hyperopt_analysis.json"
        with open(analysis_path, 'w') as f:
            # Convert numpy types for JSON serialization
            serializable_analysis = self._make_json_serializable(analysis)
            json.dump(serializable_analysis, f, indent=2)
        
        # Generate summary report
        report_path = self._generate_summary_report(analysis, model_name, dataset)
        analysis['report_path'] = str(report_path)
        
        return analysis
    
    def _extract_trials_data(self, optimization_results: Dict[str, Any]) -> pd.DataFrame:
        """Extract trial data into pandas DataFrame for analysis."""
        trials = optimization_results.get('all_trials', [])
        
        if not trials:
            return pd.DataFrame()
        
        data = []
        for trial in trials:
            row = trial['params'].copy()
            row['trial_id'] = trial['trial_id']
            row['score'] = trial['score']
            row['cv_std'] = trial['cv_std']
            row['early_stopped'] = trial['early_stopped']
            row['training_time'] = trial['training_time']
            data.append(row)
        
        return pd.DataFrame(data)
    
    def _create_summary_stats(self, optimization_results: Dict[str, Any], 
                            trials_df: pd.DataFrame) -> Dict[str, Any]:
        """Create summary statistics for optimization results."""
        if trials_df.empty:
            return {}
        
        successful_trials = trials_df[~trials_df['early_stopped']]
        
        return {
            'total_trials': len(trials_df),
            'successful_trials': len(successful_trials),
            'success_rate': len(successful_trials) / len(trials_df),
            'best_score': optimization_results.get('best_score', 0.0),
            'mean_score': successful_trials['score'].mean() if not successful_trials.empty else 0.0,
            'std_score': successful_trials['score'].std() if not successful_trials.empty else 0.0,
            'total_time': optimization_results.get('total_time', 0.0),
            'avg_trial_time': trials_df['training_time'].mean(),
            'convergence_trial': optimization_results.get('convergence_trial', 0),
            'optimization_metric': optimization_results.get('metric', 'unknown')
        }
    
    def _analyze_parameter_importance(self, trials_df: pd.DataFrame) -> Dict[str, float]:
        """Analyze parameter importance using Random Forest."""
        if trials_df.empty or len(trials_df) < 5:
            return {}
        
        # Prepare features and target
        feature_cols = [col for col in trials_df.columns 
                       if col not in ['trial_id', 'score', 'cv_std', 'early_stopped', 'training_time']]
        
        if not feature_cols:
            return {}
        
        X = trials_df[feature_cols].copy()
        y = trials_df['score'].values
        
        # Handle categorical variables
        label_encoders = {}
        for col in X.columns:
            if X[col].dtype == 'object':
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                label_encoders[col] = le
        
        # Remove rows with NaN scores
        valid_mask = ~np.isnan(y)
        if valid_mask.sum() < 3:
            return {}
        
        X = X[valid_mask]
        y = y[valid_mask]
        
        try:
            # Train Random Forest
            rf = RandomForestRegressor(n_estimators=100, random_state=42)
            rf.fit(X, y)
            
            # Get feature importance
            importance = dict(zip(feature_cols, rf.feature_importances_))
            
            # Normalize to percentages
            total_importance = sum(importance.values())
            if total_importance > 0:
                importance = {k: (v / total_importance) * 100 for k, v in importance.items()}
            
            return importance
            
        except Exception as e:
            print(f"Warning: Parameter importance analysis failed: {e}")
            return {}
    
    def _analyze_convergence(self, trials_df: pd.DataFrame) -> Dict[str, Any]:
        """Analyze convergence characteristics."""
        if trials_df.empty:
            return {}
        
        # Sort by trial order
        trials_df = trials_df.sort_values('trial_id')
        scores = trials_df['score'].values
        
        # Calculate running best
        running_best = np.maximum.accumulate(scores)
        
        # Find convergence point (when improvement becomes minimal)
        improvements = np.diff(running_best)
        convergence_threshold = 0.01  # 1% improvement threshold
        
        convergence_point = len(improvements)
        for i, improvement in enumerate(improvements):
            if improvement < convergence_threshold:
                # Check if next few trials also have minimal improvement
                window = improvements[i:i+5]
                if len(window) > 0 and np.all(window < convergence_threshold):
                    convergence_point = i + 1
                    break
        
        return {
            'convergence_trial': convergence_point,
            'final_best_score': running_best[-1],
            'improvement_rate': np.mean(improvements),
            'convergence_efficiency': convergence_point / len(trials_df),
            'running_best': running_best.tolist(),
            'raw_scores': scores.tolist()
        }
    
    def _analyze_parameter_correlations(self, trials_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Analyze correlations between parameters and performance."""
        if trials_df.empty or len(trials_df) < 5:
            return {}
        
        feature_cols = [col for col in trials_df.columns 
                       if col not in ['trial_id', 'score', 'cv_std', 'early_stopped', 'training_time']]
        
        correlations = {}
        
        for param in feature_cols:
            try:
                param_values = trials_df[param].values
                scores = trials_df['score'].values
                
                # Remove NaN values
                valid_mask = ~(np.isnan(param_values) | np.isnan(scores))
                if valid_mask.sum() < 3:
                    continue
                
                param_clean = param_values[valid_mask]
                scores_clean = scores[valid_mask]
                
                # Handle categorical parameters
                if trials_df[param].dtype == 'object':
                    # For categorical, use variance between groups
                    unique_values = np.unique(param_clean)
                    if len(unique_values) > 1:
                        group_scores = [scores_clean[param_clean == val] for val in unique_values]
                        group_means = [np.mean(group) for group in group_scores if len(group) > 0]
                        correlation = np.std(group_means) / np.mean(scores_clean) if np.mean(scores_clean) > 0 else 0
                    else:
                        correlation = 0.0
                else:
                    # For numeric, use Pearson correlation
                    if len(np.unique(param_clean)) > 1:
                        correlation, _ = pearsonr(param_clean, scores_clean)
                        if np.isnan(correlation):
                            correlation = 0.0
                    else:
                        correlation = 0.0
                
                correlations[param] = {
                    'correlation': float(correlation),
                    'abs_correlation': float(abs(correlation)),
                    'sample_size': int(valid_mask.sum())
                }
                
            except Exception as e:
                print(f"Warning: Correlation analysis failed for {param}: {e}")
                correlations[param] = {'correlation': 0.0, 'abs_correlation': 0.0, 'sample_size': 0}
        
        return correlations
    
    def _analyze_optimization_efficiency(self, trials_df: pd.DataFrame) -> Dict[str, float]:
        """Analyze optimization efficiency metrics."""
        if trials_df.empty:
            return {}
        
        scores = trials_df['score'].values
        valid_scores = scores[~np.isnan(scores)]
        
        if len(valid_scores) < 2:
            return {}
        
        # Calculate efficiency metrics
        best_score = np.max(valid_scores)
        worst_score = np.min(valid_scores)
        score_range = best_score - worst_score
        
        # Exploration vs exploitation balance
        score_std = np.std(valid_scores)
        exploration_ratio = score_std / score_range if score_range > 0 else 0
        
        # Improvement rate
        running_best = np.maximum.accumulate(valid_scores)
        improvements = np.diff(running_best)
        improvement_rate = np.mean(improvements[improvements > 0]) if len(improvements[improvements > 0]) > 0 else 0
        
        return {
            'score_range': float(score_range),
            'exploration_ratio': float(exploration_ratio),
            'improvement_rate': float(improvement_rate),
            'final_score_percentile': float(np.percentile(valid_scores, 95))
        }
    
    def _create_visualizations(self, trials_df: pd.DataFrame, model_name: str, 
                             dataset: str, analysis: Dict[str, Any]) -> Dict[str, str]:
        """Create comprehensive visualizations."""
        if trials_df.empty:
            return {}
        
        viz_paths = {}
        
        # Create main figure with subplots
        fig = plt.figure(figsize=(20, 16))
        
        # 1. Convergence Plot
        ax1 = plt.subplot(3, 3, 1)
        self._plot_convergence(trials_df, analysis['convergence_analysis'], ax1)
        
        # 2. Parameter Importance
        ax2 = plt.subplot(3, 3, 2)
        self._plot_parameter_importance(analysis['parameter_importance'], ax2)
        
        # 3. Score Distribution
        ax3 = plt.subplot(3, 3, 3)
        self._plot_score_distribution(trials_df, ax3)
        
        # 4. Parameter Correlations
        ax4 = plt.subplot(3, 3, 4)
        self._plot_parameter_correlations(analysis['parameter_correlations'], ax4)
        
        # 5. Trial Timeline
        ax5 = plt.subplot(3, 3, 5)
        self._plot_trial_timeline(trials_df, ax5)
        
        # 6. Best Parameters Radar
        ax6 = plt.subplot(3, 3, 6, projection='polar')
        self._plot_best_parameters_radar(analysis['best_parameters'], trials_df, ax6)
        
        # 7. Parameter vs Score Scatter
        ax7 = plt.subplot(3, 3, 7)
        self._plot_parameter_vs_score(trials_df, ax7)
        
        # 8. Optimization Efficiency
        ax8 = plt.subplot(3, 3, 8)
        self._plot_optimization_efficiency(analysis['optimization_efficiency'], ax8)
        
        # 9. Summary Statistics
        ax9 = plt.subplot(3, 3, 9)
        self._plot_summary_stats(analysis['summary'], ax9)
        
        plt.tight_layout()
        
        # Save main visualization
        main_viz_path = self.results_dir / f"{model_name}_{dataset}_hyperopt_analysis.png"
        plt.savefig(main_viz_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        viz_paths['main_analysis'] = str(main_viz_path)
        
        # Create detailed parameter plots
        detailed_path = self._create_detailed_parameter_plots(trials_df, model_name, dataset)
        if detailed_path:
            viz_paths['detailed_parameters'] = detailed_path
        
        return viz_paths
    
    def _plot_convergence(self, trials_df: pd.DataFrame, convergence_analysis: Dict[str, Any], ax):
        """Plot convergence analysis."""
        if 'running_best' not in convergence_analysis:
            ax.text(0.5, 0.5, 'No convergence data', ha='center', va='center', transform=ax.transAxes)
            return
        
        running_best = convergence_analysis['running_best']
        raw_scores = convergence_analysis['raw_scores']
        
        trials = range(1, len(running_best) + 1)
        
        # Plot raw scores and running best
        ax.scatter(trials, raw_scores, alpha=0.6, s=30, label='Trial Scores')
        ax.plot(trials, running_best, 'r-', linewidth=2, label='Best Score')
        
        # Mark convergence point
        conv_point = convergence_analysis.get('convergence_trial', len(trials))
        if conv_point < len(trials):
            ax.axvline(x=conv_point, color='orange', linestyle='--', 
                      label=f'Convergence (Trial {conv_point})')
        
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Score')
        ax.set_title('Optimization Convergence')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_parameter_importance(self, importance: Dict[str, float], ax):
        """Plot parameter importance."""
        if not importance:
            ax.text(0.5, 0.5, 'No importance data', ha='center', va='center', transform=ax.transAxes)
            return
        
        params = list(importance.keys())
        values = list(importance.values())
        
        bars = ax.barh(params, values)
        ax.set_xlabel('Importance (%)')
        ax.set_title('Parameter Importance')
        
        # Color bars by importance
        for i, (bar, val) in enumerate(zip(bars, values)):
            bar.set_color(plt.cm.RdYlBu_r(val / max(values)))
        
        # Add value labels
        for i, v in enumerate(values):
            ax.text(v + 0.5, i, f'{v:.1f}%', va='center', fontsize=8)
    
    def _plot_score_distribution(self, trials_df: pd.DataFrame, ax):
        """Plot score distribution."""
        scores = trials_df['score'].dropna()
        if len(scores) == 0:
            ax.text(0.5, 0.5, 'No score data', ha='center', va='center', transform=ax.transAxes)
            return
        
        ax.hist(scores, bins=min(20, len(scores)//2), alpha=0.7, edgecolor='black')
        ax.axvline(scores.mean(), color='red', linestyle='--', label=f'Mean: {scores.mean():.3f}')
        ax.axvline(scores.max(), color='green', linestyle='--', label=f'Best: {scores.max():.3f}')
        
        ax.set_xlabel('Score')
        ax.set_ylabel('Frequency')
        ax.set_title('Score Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    def _plot_parameter_correlations(self, correlations: Dict[str, Dict[str, float]], ax):
        """Plot parameter correlations with performance."""
        if not correlations:
            ax.text(0.5, 0.5, 'No correlation data', ha='center', va='center', transform=ax.transAxes)
            return
        
        params = list(correlations.keys())
        corr_values = [correlations[p]['correlation'] for p in params]
        
        colors = ['red' if x < 0 else 'green' for x in corr_values]
        bars = ax.barh(params, corr_values, color=colors, alpha=0.7)
        
        ax.set_xlabel('Correlation with Score')
        ax.set_title('Parameter-Performance Correlations')
        ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for i, v in enumerate(corr_values):
            ax.text(v + 0.01 * np.sign(v), i, f'{v:.2f}', va='center', fontsize=8)
    
    def _plot_trial_timeline(self, trials_df: pd.DataFrame, ax):
        """Plot trial timeline with training time."""
        if trials_df.empty:
            ax.text(0.5, 0.5, 'No timeline data', ha='center', va='center', transform=ax.transAxes)
            return
        
        trials_sorted = trials_df.sort_values('trial_id')
        
        # Color by success/failure
        colors = ['red' if stopped else 'green' for stopped in trials_sorted['early_stopped']]
        
        scatter = ax.scatter(trials_sorted['trial_id'], trials_sorted['training_time'], 
                           c=colors, s=50, alpha=0.7)
        
        ax.set_xlabel('Trial Number')
        ax.set_ylabel('Training Time (s)')
        ax.set_title('Trial Timeline')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor='green', label='Successful'),
                          Patch(facecolor='red', label='Early Stopped')]
        ax.legend(handles=legend_elements)
    
    def _plot_best_parameters_radar(self, best_params: Dict[str, Any], trials_df: pd.DataFrame, ax):
        """Plot best parameters as radar chart."""
        if not best_params or trials_df.empty:
            ax.text(0.5, 0.5, 'No parameter data', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Get numeric parameters only
        numeric_params = {}
        for param, value in best_params.items():
            if isinstance(value, (int, float)) and not param.endswith('_logit'):
                numeric_params[param] = value
        
        if len(numeric_params) < 3:
            ax.text(0.5, 0.5, 'Need ≥3 numeric params\nfor radar chart', 
                   ha='center', va='center', transform=ax.transAxes)
            return
        
        params = list(numeric_params.keys())
        values = list(numeric_params.values())
        
        # Normalize values to 0-1 range based on observed ranges
        normalized_values = []
        for param, value in zip(params, values):
            param_col = trials_df[param] if param in trials_df.columns else [value]
            param_min = min(param_col)
            param_max = max(param_col)
            if param_max > param_min:
                norm_val = (value - param_min) / (param_max - param_min)
            else:
                norm_val = 0.5
            normalized_values.append(norm_val)
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(params), endpoint=False).tolist()
        normalized_values += [normalized_values[0]]  # Close the circle
        angles += [angles[0]]
        
        ax.plot(angles, normalized_values, 'o-', linewidth=2, color='blue')
        ax.fill(angles, normalized_values, alpha=0.25, color='blue')
        ax.set_thetagrids(np.degrees(angles[:-1]), params)
        ax.set_ylim(0, 1)
        ax.set_title('Best Parameters\n(Normalized)', y=1.1)
    
    def _plot_parameter_vs_score(self, trials_df: pd.DataFrame, ax):
        """Plot most important parameter vs score."""
        if trials_df.empty:
            ax.text(0.5, 0.5, 'No data for scatter plot', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Find most variable numeric parameter
        numeric_cols = trials_df.select_dtypes(include=[np.number]).columns
        param_cols = [col for col in numeric_cols 
                     if col not in ['trial_id', 'score', 'cv_std', 'training_time']]
        
        if not param_cols:
            ax.text(0.5, 0.5, 'No numeric parameters', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Choose parameter with highest variance
        variances = {col: trials_df[col].var() for col in param_cols}
        best_param = max(variances.keys(), key=lambda k: variances[k])
        
        x = trials_df[best_param]
        y = trials_df['score']
        
        # Remove NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        
        if len(x_clean) > 0:
            ax.scatter(x_clean, y_clean, alpha=0.6, s=50)
            
            # Add trend line if enough points
            if len(x_clean) > 2:
                z = np.polyfit(x_clean, y_clean, 1)
                p = np.poly1d(z)
                ax.plot(x_clean, p(x_clean), "r--", alpha=0.8)
        
        ax.set_xlabel(best_param)
        ax.set_ylabel('Score')
        ax.set_title(f'{best_param} vs Score')
        ax.grid(True, alpha=0.3)
    
    def _plot_optimization_efficiency(self, efficiency: Dict[str, float], ax):
        """Plot optimization efficiency metrics."""
        if not efficiency:
            ax.text(0.5, 0.5, 'No efficiency data', ha='center', va='center', transform=ax.transAxes)
            return
        
        metrics = ['Score Range', 'Exploration', 'Improvement\nRate', 'Top 5%\nPercentile']
        values = [
            efficiency.get('score_range', 0),
            efficiency.get('exploration_ratio', 0),
            efficiency.get('improvement_rate', 0),
            efficiency.get('final_score_percentile', 0)
        ]
        
        # Normalize values for comparison
        max_val = max(values) if max(values) > 0 else 1
        normalized_values = [v / max_val for v in values]
        
        bars = ax.bar(metrics, normalized_values, color=['blue', 'green', 'orange', 'red'], alpha=0.7)
        ax.set_ylabel('Normalized Value')
        ax.set_title('Optimization Efficiency')
        ax.set_ylim(0, 1.1)
        
        # Add value labels
        for bar, val in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=8)
    
    def _plot_summary_stats(self, summary: Dict[str, Any], ax):
        """Plot summary statistics."""
        ax.axis('off')
        
        stats_text = f"""
Hyperparameter Optimization Summary

Total Trials: {summary.get('total_trials', 0)}
Successful: {summary.get('successful_trials', 0)} ({summary.get('success_rate', 0)*100:.1f}%)

Best Score: {summary.get('best_score', 0):.4f}
Mean Score: {summary.get('mean_score', 0):.4f} ± {summary.get('std_score', 0):.4f}

Total Time: {summary.get('total_time', 0):.1f}s
Avg Trial: {summary.get('avg_trial_time', 0):.1f}s

Convergence: Trial {summary.get('convergence_trial', 0)}
Metric: {summary.get('optimization_metric', 'unknown')}
        """
        
        ax.text(0.1, 0.9, stats_text.strip(), transform=ax.transAxes, 
               fontsize=10, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7))
    
    def _create_detailed_parameter_plots(self, trials_df: pd.DataFrame, 
                                       model_name: str, dataset: str) -> Optional[str]:
        """Create detailed parameter analysis plots."""
        if trials_df.empty:
            return None
        
        # Get parameter columns
        param_cols = [col for col in trials_df.columns 
                     if col not in ['trial_id', 'score', 'cv_std', 'early_stopped', 'training_time']]
        
        if not param_cols:
            return None
        
        # Create subplot grid
        n_params = len(param_cols)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
        if n_params == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes if isinstance(axes, np.ndarray) else [axes]
        else:
            axes = axes.flatten()
        
        for i, param in enumerate(param_cols):
            ax = axes[i]
            
            if trials_df[param].dtype == 'object':
                # Categorical parameter - box plot
                self._plot_categorical_parameter(trials_df, param, ax)
            else:
                # Numeric parameter - scatter plot
                self._plot_numeric_parameter(trials_df, param, ax)
        
        # Hide unused subplots
        for i in range(n_params, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        
        detailed_path = self.results_dir / f"{model_name}_{dataset}_detailed_parameters.png"
        plt.savefig(detailed_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return str(detailed_path)
    
    def _plot_categorical_parameter(self, trials_df: pd.DataFrame, param: str, ax):
        """Plot categorical parameter analysis."""
        param_data = trials_df.groupby(param)['score'].agg(['mean', 'std', 'count'])
        param_data = param_data.sort_values('mean', ascending=False)
        
        x_pos = range(len(param_data))
        bars = ax.bar(x_pos, param_data['mean'], yerr=param_data['std'], 
                     capsize=5, alpha=0.7)
        
        ax.set_xlabel(param)
        ax.set_ylabel('Mean Score')
        ax.set_title(f'{param} vs Performance')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(param_data.index, rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add count labels
        for i, (bar, count) in enumerate(zip(bars, param_data['count'])):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'n={count}', ha='center', va='bottom', fontsize=8)
    
    def _plot_numeric_parameter(self, trials_df: pd.DataFrame, param: str, ax):
        """Plot numeric parameter analysis."""
        x = trials_df[param]
        y = trials_df['score']
        
        # Remove NaN values
        valid_mask = ~(np.isnan(x) | np.isnan(y))
        x_clean = x[valid_mask]
        y_clean = y[valid_mask]
        
        if len(x_clean) == 0:
            ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
            return
        
        # Color by score percentile
        percentiles = np.percentile(y_clean, [25, 50, 75])
        colors = []
        for score in y_clean:
            if score >= percentiles[2]:
                colors.append('green')  # Top quartile
            elif score >= percentiles[1]:
                colors.append('orange')  # Third quartile
            elif score >= percentiles[0]:
                colors.append('blue')  # Second quartile
            else:
                colors.append('red')  # Bottom quartile
        
        scatter = ax.scatter(x_clean, y_clean, c=colors, alpha=0.6, s=50)
        
        # Add trend line
        if len(x_clean) > 2:
            z = np.polyfit(x_clean, y_clean, 1)
            p = np.poly1d(z)
            ax.plot(x_clean, p(x_clean), "r--", alpha=0.8)
            
            # Add correlation
            corr, _ = pearsonr(x_clean, y_clean)
            ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                   bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
        
        ax.set_xlabel(param)
        ax.set_ylabel('Score')
        ax.set_title(f'{param} vs Performance')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='green', label='Top 25%'),
            Patch(facecolor='orange', label='50-75%'),
            Patch(facecolor='blue', label='25-50%'),
            Patch(facecolor='red', label='Bottom 25%')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
    
    def _generate_summary_report(self, analysis: Dict[str, Any], 
                               model_name: str, dataset: str) -> Path:
        """Generate comprehensive summary report with automated analysis."""
        report_path = self.results_dir / f"{model_name}_{dataset}_hyperopt_report.md"
        
        with open(report_path, 'w') as f:
            f.write(f"# Hyperparameter Optimization Report\n\n")
            f.write(f"**Model:** {model_name}  \n")
            f.write(f"**Dataset:** {dataset}  \n")
            f.write(f"**Generated:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}  \n\n")
            
            # Summary Statistics
            summary = analysis.get('summary', {})
            f.write("## Summary\n\n")
            f.write(f"- **Total Trials:** {summary.get('total_trials', 0)}\n")
            f.write(f"- **Successful Trials:** {summary.get('successful_trials', 0)} ({summary.get('success_rate', 0)*100:.1f}%)\n")
            f.write(f"- **Best Score:** {summary.get('best_score', 0):.4f}\n")
            f.write(f"- **Mean Score:** {summary.get('mean_score', 0):.4f} ± {summary.get('std_score', 0):.4f}\n")
            f.write(f"- **Total Time:** {summary.get('total_time', 0):.1f}s\n")
            f.write(f"- **Convergence Trial:** {summary.get('convergence_trial', 0)}\n\n")
            
            # Best Parameters
            best_params = analysis.get('best_parameters', {})
            f.write("## Best Parameters\n\n")
            for param, value in best_params.items():
                if isinstance(value, float):
                    f.write(f"- **{param}:** {value:.6f}\n")
                else:
                    f.write(f"- **{param}:** {value}\n")
            f.write("\n")
            
            # Parameter Importance
            importance = analysis.get('parameter_importance', {})
            if importance:
                f.write("## Parameter Importance\n\n")
                sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
                for param, imp in sorted_importance:
                    f.write(f"- **{param}:** {imp:.1f}%\n")
                f.write("\n")
            
            # Correlations
            correlations = analysis.get('parameter_correlations', {})
            if correlations:
                f.write("## Parameter-Performance Correlations\n\n")
                sorted_corr = sorted(correlations.items(), 
                                   key=lambda x: x[1].get('abs_correlation', 0), reverse=True)
                for param, corr_data in sorted_corr:
                    corr_val = corr_data.get('correlation', 0)
                    f.write(f"- **{param}:** {corr_val:.3f}\n")
                f.write("\n")
            
            # Convergence Analysis
            convergence = analysis.get('convergence_analysis', {})
            if convergence:
                f.write("## Convergence Analysis\n\n")
                f.write(f"- **Convergence Trial:** {convergence.get('convergence_trial', 0)}\n")
                f.write(f"- **Final Best Score:** {convergence.get('final_best_score', 0):.4f}\n")
                f.write(f"- **Improvement Rate:** {convergence.get('improvement_rate', 0):.4f}\n")
                f.write(f"- **Convergence Efficiency:** {convergence.get('convergence_efficiency', 0)*100:.1f}%\n\n")
            
            # Optimization Efficiency
            efficiency = analysis.get('optimization_efficiency', {})
            if efficiency:
                f.write("## Optimization Efficiency\n\n")
                f.write(f"- **Score Range:** {efficiency.get('score_range', 0):.4f}\n")
                f.write(f"- **Exploration Ratio:** {efficiency.get('exploration_ratio', 0):.3f}\n")
                f.write(f"- **Improvement Rate:** {efficiency.get('improvement_rate', 0):.4f}\n")
                f.write(f"- **95th Percentile Score:** {efficiency.get('final_score_percentile', 0):.4f}\n\n")
            
            # Add Automated Analysis Section
            f.write("## 🤖 Automated Analysis & Recommendations\n\n")
            f.write("*This section provides AI-generated insights and actionable recommendations based on the optimization results.*\n\n")
            
            # Generate automated analysis
            automated_analysis = self._generate_automated_analysis(analysis)
            
            for section_title, content_list in automated_analysis.items():
                f.write(f"### {section_title}\n\n")
                for item in content_list:
                    f.write(f"- {item}\n")
                f.write("\n")
            
            # Visualizations
            viz_paths = analysis.get('visualization_paths', {})
            if viz_paths:
                f.write("## Visualizations\n\n")
                for viz_name, viz_path in viz_paths.items():
                    f.write(f"- **{viz_name.replace('_', ' ').title()}:** `{viz_path}`\n")
                f.write("\n")
            
            f.write("---\n")
            f.write("*Generated by Deep-GPCM Hyperparameter Optimization System*\n")
        
        return report_path
    
    def _generate_automated_analysis(self, analysis: Dict[str, Any]) -> Dict[str, List[str]]:
        """Generate automated analysis and recommendations from optimization results."""
        
        # Extract data for analysis
        summary = analysis.get('summary', {})
        best_params = analysis.get('best_parameters', {})
        importance = analysis.get('parameter_importance', {})
        correlations = analysis.get('parameter_correlations', {})
        convergence = analysis.get('convergence_analysis', {})
        efficiency = analysis.get('optimization_efficiency', {})
        
        analysis_results = {}
        
        # 1. Performance Summary
        performance_analysis = []
        best_score = summary.get('best_score', 0)
        mean_score = summary.get('mean_score', 0)
        convergence_trial = summary.get('convergence_trial', 0)
        total_trials = summary.get('total_trials', 0)
        
        performance_analysis.append(f"Best performance: {best_score:.4f} QWK (Trial #{convergence_trial})")
        
        score_range = efficiency.get('score_range', 0)
        performance_analysis.append(f"Performance range: {score_range:.4f} spread across all trials")
        
        # Convergence efficiency analysis
        conv_efficiency = convergence.get('convergence_efficiency', 0)
        if conv_efficiency < 0.2:
            efficiency_desc = "Excellent"
        elif conv_efficiency < 0.4:
            efficiency_desc = "Good"
        else:
            efficiency_desc = "Slow"
        performance_analysis.append(f"Convergence efficiency: {efficiency_desc} ({conv_efficiency*100:.1f}% of trials needed)")
        
        analysis_results["📊 Performance Summary"] = performance_analysis
        
        # 2. Parameter Pattern Analysis
        pattern_analysis = []
        
        if importance:
            # Find most important parameter
            top_param = max(importance.items(), key=lambda x: x[1])
            if top_param[1] > 50:
                pattern_analysis.append(f"{top_param[0]} is critical ({top_param[1]:.1f}% importance) - focus optimization here")
            
            # Analyze parameter distribution
            sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)
            top_3 = sorted_importance[:3]
            top_3_total = sum(imp for _, imp in top_3)
            if top_3_total > 80:
                pattern_analysis.append(f"Top 3 parameters dominate ({top_3_total:.1f}%): {', '.join([p for p, _ in top_3])}")
        
        # Analyze correlations for insights
        if correlations:
            strong_negative_corr = [(p, data['correlation']) for p, data in correlations.items() 
                                  if data['correlation'] < -0.5]
            strong_positive_corr = [(p, data['correlation']) for p, data in correlations.items() 
                                  if data['correlation'] > 0.5]
            
            for param, corr in strong_negative_corr:
                pattern_analysis.append(f"Lower {param} values strongly improve performance (correlation: {corr:.3f})")
            
            for param, corr in strong_positive_corr:
                pattern_analysis.append(f"Higher {param} values strongly improve performance (correlation: {corr:.3f})")
        
        analysis_results["🔍 Parameter Patterns"] = pattern_analysis
        
        # 3. Loss Weight Analysis (if available)
        loss_analysis = []
        if 'ce_weight_logit' in best_params and 'focal_weight_logit' in best_params:
            # Convert logits to weights (simplified softmax approximation)
            ce_logit = best_params['ce_weight_logit']
            focal_logit = best_params['focal_weight_logit']
            
            # Approximate weight conversion
            ce_weight = np.exp(ce_logit) / (np.exp(ce_logit) + np.exp(focal_logit) + 1)
            focal_weight = np.exp(focal_logit) / (np.exp(ce_logit) + np.exp(focal_logit) + 1)
            qwk_weight = 1 / (np.exp(ce_logit) + np.exp(focal_logit) + 1)
            
            loss_analysis.append(f"Optimal loss combination: CE={ce_weight:.2f}, Focal={focal_weight:.2f}, QWK={qwk_weight:.2f}")
            
            if focal_weight > ce_weight:
                loss_analysis.append("Focal loss dominance suggests class imbalance handling is critical")
            if qwk_weight > 0.2:
                loss_analysis.append("High QWK weight indicates ordinal structure is important")
        
        if loss_analysis:
            analysis_results["⚖️ Loss Function Insights"] = loss_analysis
        
        # 4. Actionable Recommendations
        recommendations = []
        
        # Based on parameter importance
        if importance:
            top_param = max(importance.items(), key=lambda x: x[1])
            if top_param[1] > 50:
                recommendations.append(f"Focus future optimization on {top_param[0]} ({top_param[1]:.1f}% importance)")
        
        # Based on convergence
        if conv_efficiency > 0.7:
            recommendations.append("Early convergence detected - consider expanding search space or increasing exploration")
        elif conv_efficiency < 0.3:
            recommendations.append("Efficient convergence - current search space and strategy are effective")
        
        # Memory size specific recommendations
        if 'memory_size' in best_params:
            memory_val = best_params['memory_size']
            if isinstance(memory_val, (int, str)) and int(memory_val) <= 20:
                recommendations.append("Small memory networks work best - consider testing even smaller sizes (10-15)")
            elif isinstance(memory_val, (int, str)) and int(memory_val) >= 100:
                recommendations.append("Large memory networks optimal - explore 150-200 range")
        
        # Dropout specific recommendations
        if 'dropout_rate' in best_params:
            dropout_val = best_params['dropout_rate']
            if isinstance(dropout_val, (int, float)) and dropout_val < 0.05:
                recommendations.append("Very low dropout works best - try 0.01-0.03 range for fine-tuning")
            elif isinstance(dropout_val, (int, float)) and dropout_val > 0.1:
                recommendations.append("Higher regularization needed - current dropout level is appropriate")
        
        # Search space recommendations
        current_params = set(best_params.keys())
        missing_important = {'embed_dim', 'key_dim', 'value_dim', 'n_heads'} - current_params
        if missing_important:
            recommendations.append(f"Expand search space: add {', '.join(sorted(missing_important))} for architectural optimization")
        
        analysis_results["🚀 Actionable Recommendations"] = recommendations
        
        # 5. Next Steps
        next_steps = []
        
        if best_score < 0.65:
            next_steps.append("Performance below 65% QWK - expand architectural parameters and increase trial budget")
        elif best_score < 0.70:
            next_steps.append("Good performance - fine-tune with extended search space and adaptive epochs")
        else:
            next_steps.append("Excellent performance - focus on transfer learning to other datasets")
        
        next_steps.append("Implement adaptive epoch allocation (5→20→40 epochs based on performance)")
        next_steps.append("Add learning rate scheduling and optimizer parameters to search space")
        
        if total_trials < 50:
            next_steps.append("Consider increasing trial budget for more thorough exploration")
        
        analysis_results["📋 Next Steps"] = next_steps
        
        return analysis_results
    
    def _make_json_serializable(self, obj):
        """Convert numpy types and other non-serializable objects to JSON-compatible types."""
        if isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float64)):
            return float(obj)
        elif pd.isna(obj):
            return None
        else:
            return obj


def create_hyperopt_visualizer() -> HyperoptVisualizer:
    """Factory function to create hyperparameter optimization visualizer."""
    return HyperoptVisualizer()