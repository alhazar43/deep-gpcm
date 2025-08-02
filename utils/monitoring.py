#!/usr/bin/env python3
"""
Monitoring infrastructure for unified prediction system development.
Tracks performance, memory usage, and prediction statistics.
"""

import time
import torch
import numpy as np
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
import json
import os

@dataclass
class PredictionStats:
    """Statistics for prediction methods."""
    method: str
    count: int = 0
    total_time: float = 0.0
    min_time: float = float('inf')
    max_time: float = 0.0
    memory_usage: List[float] = field(default_factory=list)
    prediction_distribution: Dict[int, int] = field(default_factory=dict)
    confidence_stats: Dict[str, float] = field(default_factory=dict)
    edge_cases_encountered: Dict[str, int] = field(default_factory=dict)
    
    def update(self, elapsed_time: float, predictions: torch.Tensor, 
               probabilities: torch.Tensor, memory_mb: float):
        """Update statistics with new prediction batch."""
        self.count += 1
        self.total_time += elapsed_time
        self.min_time = min(self.min_time, elapsed_time)
        self.max_time = max(self.max_time, elapsed_time)
        self.memory_usage.append(memory_mb)
        
        # Update prediction distribution
        for pred in predictions.flatten().tolist():
            self.prediction_distribution[pred] = self.prediction_distribution.get(pred, 0) + 1
        
        # Update confidence statistics
        max_probs = probabilities.max(dim=-1)[0]
        self.confidence_stats['mean'] = float(max_probs.mean())
        self.confidence_stats['std'] = float(max_probs.std())
        self.confidence_stats['min'] = float(max_probs.min())
        self.confidence_stats['max'] = float(max_probs.max())
        
        # Detect edge cases
        # Uniform probabilities
        uniform_threshold = 1.0 / probabilities.shape[-1]
        near_uniform = (probabilities.std(dim=-1) < 0.05).sum().item()
        if near_uniform > 0:
            self.edge_cases_encountered['near_uniform'] = \
                self.edge_cases_encountered.get('near_uniform', 0) + near_uniform
        
        # Near ties
        sorted_probs = probabilities.sort(dim=-1, descending=True)[0]
        near_ties = ((sorted_probs[:, 0] - sorted_probs[:, 1]) < 0.05).sum().item()
        if near_ties > 0:
            self.edge_cases_encountered['near_ties'] = \
                self.edge_cases_encountered.get('near_ties', 0) + near_ties
        
        # Extreme confidence
        extreme_conf = (max_probs > 0.99).sum().item()
        if extreme_conf > 0:
            self.edge_cases_encountered['extreme_confidence'] = \
                self.edge_cases_encountered.get('extreme_confidence', 0) + extreme_conf
    
    @property
    def avg_time(self) -> float:
        return self.total_time / self.count if self.count > 0 else 0.0
    
    @property
    def avg_memory(self) -> float:
        return np.mean(self.memory_usage) if self.memory_usage else 0.0

class PredictionMonitor:
    """Monitor for tracking unified prediction system performance."""
    
    def __init__(self, log_dir: str = "logs/prediction_monitor"):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)
        
        self.stats = {
            'hard': PredictionStats('hard'),
            'soft': PredictionStats('soft'),
            'threshold': PredictionStats('threshold')
        }
        
        self.comparison_stats = {
            'hard_vs_soft': [],
            'hard_vs_threshold': [],
            'soft_vs_threshold': []
        }
        
        self.session_start = datetime.now()
        self.warnings = []
        
    def get_memory_usage(self) -> float:
        """Get current GPU memory usage in MB."""
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1024 / 1024
        return 0.0
    
    def time_prediction(self, method: str, func, *args, **kwargs):
        """Time a prediction function and update statistics."""
        start_mem = self.get_memory_usage()
        start_time = time.time()
        
        result = func(*args, **kwargs)
        
        elapsed = time.time() - start_time
        end_mem = self.get_memory_usage()
        
        # Extract predictions and probabilities for stats
        if isinstance(result, dict):
            predictions = result.get(f'{method}_predictions')
            probabilities = result.get('probabilities')
        else:
            predictions = result
            probabilities = args[0] if len(args) > 0 else kwargs.get('probabilities')
        
        if method in self.stats and predictions is not None and probabilities is not None:
            self.stats[method].update(elapsed, predictions, probabilities, end_mem - start_mem)
        
        return result
    
    def compare_predictions(self, predictions: Dict[str, torch.Tensor]):
        """Compare predictions across methods."""
        methods = list(predictions.keys())
        
        for i in range(len(methods)):
            for j in range(i + 1, len(methods)):
                method1, method2 = methods[i], methods[j]
                pred1 = predictions[method1]
                pred2 = predictions[method2]
                
                # Ensure tensors are on the same device (CPU)
                if pred1.is_cuda:
                    pred1 = pred1.cpu()
                if pred2.is_cuda:
                    pred2 = pred2.cpu()
                
                # Calculate agreement
                agreement = (pred1 == pred2).float().mean().item()
                
                # Calculate average difference for ordinal predictions
                avg_diff = torch.abs(pred1.float() - pred2.float()).mean().item()
                
                comparison_key = f'{method1}_vs_{method2}'
                if comparison_key in self.comparison_stats:
                    self.comparison_stats[comparison_key].append({
                        'agreement': agreement,
                        'avg_difference': avg_diff,
                        'timestamp': datetime.now().isoformat()
                    })
    
    def add_warning(self, warning: str, severity: str = "info"):
        """Add a warning or notable event."""
        self.warnings.append({
            'message': warning,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        })
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of monitoring statistics."""
        summary = {
            'session_duration': (datetime.now() - self.session_start).total_seconds(),
            'methods': {}
        }
        
        for method, stats in self.stats.items():
            if stats.count > 0:
                summary['methods'][method] = {
                    'count': stats.count,
                    'avg_time_ms': stats.avg_time * 1000,
                    'min_time_ms': stats.min_time * 1000,
                    'max_time_ms': stats.max_time * 1000,
                    'avg_memory_mb': stats.avg_memory,
                    'confidence': stats.confidence_stats,
                    'edge_cases': stats.edge_cases_encountered,
                    'prediction_distribution': stats.prediction_distribution
                }
        
        # Add comparison statistics
        summary['comparisons'] = {}
        for comp, data in self.comparison_stats.items():
            if data:
                recent = data[-10:]  # Last 10 comparisons
                summary['comparisons'][comp] = {
                    'avg_agreement': np.mean([d['agreement'] for d in recent]),
                    'avg_difference': np.mean([d['avg_difference'] for d in recent]),
                    'samples': len(recent)
                }
        
        summary['warnings'] = self.warnings[-10:]  # Last 10 warnings
        
        return summary
    
    def save_summary(self, filename: Optional[str] = None):
        """Save monitoring summary to file."""
        if filename is None:
            filename = f"prediction_monitor_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        filepath = os.path.join(self.log_dir, filename)
        summary = self.get_summary()
        
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
        
        return filepath
    
    def print_summary(self):
        """Print a formatted summary of monitoring statistics."""
        summary = self.get_summary()
        
        print("\n" + "="*60)
        print("PREDICTION MONITORING SUMMARY")
        print("="*60)
        print(f"Session Duration: {summary['session_duration']:.1f} seconds")
        
        # Method statistics
        print("\nMethod Performance:")
        print("-"*60)
        print(f"{'Method':<12} {'Count':<8} {'Avg(ms)':<10} {'Min(ms)':<10} {'Max(ms)':<10} {'Mem(MB)':<10}")
        print("-"*60)
        
        for method, stats in summary['methods'].items():
            print(f"{method:<12} {stats['count']:<8} "
                  f"{stats['avg_time_ms']:<10.3f} {stats['min_time_ms']:<10.3f} "
                  f"{stats['max_time_ms']:<10.3f} {stats['avg_memory_mb']:<10.2f}")
        
        # Comparison statistics
        if summary['comparisons']:
            print("\nMethod Comparisons:")
            print("-"*60)
            print(f"{'Comparison':<25} {'Agreement':<15} {'Avg Diff':<15}")
            print("-"*60)
            
            for comp, stats in summary['comparisons'].items():
                print(f"{comp:<25} {stats['avg_agreement']:<15.3%} {stats['avg_difference']:<15.3f}")
        
        # Edge cases
        print("\nEdge Cases Encountered:")
        print("-"*60)
        for method, stats in summary['methods'].items():
            if stats['edge_cases']:
                print(f"{method}: {stats['edge_cases']}")
        
        # Warnings
        if summary['warnings']:
            print("\nRecent Warnings:")
            print("-"*60)
            for warning in summary['warnings'][-5:]:
                print(f"[{warning['severity'].upper()}] {warning['message']}")
        
        print("="*60)

# Global monitor instance
_monitor = None

def get_monitor() -> PredictionMonitor:
    """Get or create global monitor instance."""
    global _monitor
    if _monitor is None:
        _monitor = PredictionMonitor()
    return _monitor

def reset_monitor():
    """Reset the global monitor."""
    global _monitor
    _monitor = PredictionMonitor()

# Decorator for automatic monitoring
def monitor_prediction(method: str):
    """Decorator to automatically monitor prediction functions."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            monitor = get_monitor()
            return monitor.time_prediction(method, func, *args, **kwargs)
        return wrapper
    return decorator