#!/usr/bin/env python3
"""
Beta Parameter Extraction Summary
Demonstrates the successful implementation of beta extraction using GPCM computation method.
"""

import os
import sys
import json
import numpy as np
import argparse
from pathlib import Path
from glob import glob

def analyze_extraction_results(results_dir: str = "results/beta_extraction"):
    """Analyze all beta extraction results and provide summary."""
    
    results_dir = Path(results_dir)
    if not results_dir.exists():
        print(f"❌ Results directory not found: {results_dir}")
        return
    
    # Find all extraction result files
    result_files = list(results_dir.glob("beta_extraction_*.json"))
    
    if not result_files:
        print(f"❌ No beta extraction results found in {results_dir}")
        return
    
    print("=" * 80)
    print("BETA PARAMETER EXTRACTION SUMMARY")
    print("=" * 80)
    print(f"📁 Results directory: {results_dir}")
    print(f"📊 Found {len(result_files)} extraction results")
    print()
    
    all_results = []
    
    for result_file in sorted(result_files):
        try:
            with open(result_file, 'r') as f:
                data = json.load(f)
            
            model_type = data['model_type']
            dataset = data['dataset']
            extraction_method = data['extraction_method']
            
            # Get statistics
            beta_stats = data['extraction_summary']['beta_statistics']
            theta_stats = data['extraction_summary']['theta_statistics']
            total_sequences = data['extraction_summary']['total_sequences']
            total_beta_values = data['extraction_summary']['total_beta_values']
            
            # Verification info
            verification = data['extraction_verification']
            matches_gpcm = verification['matches_gpcm_computation']
            computational_pathway = verification['computational_pathway']
            
            result_summary = {
                'model_type': model_type,
                'dataset': dataset,
                'extraction_method': extraction_method,
                'total_sequences': total_sequences,
                'total_beta_values': total_beta_values,
                'beta_range': [beta_stats['min'], beta_stats['max']],
                'beta_mean': beta_stats['mean'],
                'beta_std': beta_stats['std'],
                'theta_range': [theta_stats['min'], theta_stats['max']],
                'theta_mean': theta_stats['mean'],
                'theta_std': theta_stats['std'],
                'matches_gpcm_computation': matches_gpcm,
                'computational_pathway': computational_pathway,
                'extraction_location': verification['extraction_location'],
                'file_path': str(result_file)
            }
            
            all_results.append(result_summary)
            
        except Exception as e:
            print(f"⚠️  Could not process {result_file}: {e}")
    
    if not all_results:
        print("❌ No valid extraction results found")
        return
    
    # Display summary table
    print("📊 EXTRACTION RESULTS SUMMARY:")
    print("-" * 120)
    print(f"{'Model Type':<25} {'Dataset':<15} {'Sequences':<10} {'Beta Values':<12} {'Beta Range':<20} {'θ Range':<20} {'GPCM Match':<12}")
    print("-" * 120)
    
    for result in all_results:
        model_type = result['model_type']
        dataset = result['dataset']
        sequences = f"{result['total_sequences']:,}"
        beta_values = f"{result['total_beta_values']:,}"
        beta_range = f"[{result['beta_range'][0]:.2f}, {result['beta_range'][1]:.2f}]"
        theta_range = f"[{result['theta_range'][0]:.2f}, {result['theta_range'][1]:.2f}]"
        gpcm_match = "✅ YES" if result['matches_gpcm_computation'] else "❌ NO"
        
        print(f"{model_type:<25} {dataset:<15} {sequences:<10} {beta_values:<12} {beta_range:<20} {theta_range:<20} {gpcm_match:<12}")
    
    print("-" * 120)
    
    # Detailed analysis
    print("\n📈 DETAILED ANALYSIS:")
    
    # 1. Method verification
    all_match_gpcm = all(r['matches_gpcm_computation'] for r in all_results)
    print(f"  ✅ All extractions match GPCM computation: {all_match_gpcm}")
    
    # 2. Extraction method consistency
    extraction_methods = set(r['extraction_method'] for r in all_results)
    print(f"  📋 Extraction method(s): {', '.join(extraction_methods)}")
    
    # 3. Computational pathway verification
    pathways = set(r['computational_pathway'] for r in all_results)
    print(f"  🔄 Computational pathway(s): {', '.join(pathways)}")
    
    # 4. Model type coverage
    model_types = set(r['model_type'] for r in all_results)
    print(f"  🏗️  Model types tested: {', '.join(sorted(model_types))}")
    
    # 5. Parameter statistics
    all_beta_means = [r['beta_mean'] for r in all_results]
    all_beta_stds = [r['beta_std'] for r in all_results]
    all_theta_means = [r['theta_mean'] for r in all_results]
    all_theta_stds = [r['theta_std'] for r in all_results]
    
    print(f"\n📊 PARAMETER STATISTICS ACROSS ALL MODELS:")
    print(f"  Beta means: {np.min(all_beta_means):.3f} to {np.max(all_beta_means):.3f}")
    print(f"  Beta stds: {np.min(all_beta_stds):.3f} to {np.max(all_beta_stds):.3f}")
    print(f"  Theta means: {np.min(all_theta_means):.3f} to {np.max(all_theta_means):.3f}")
    print(f"  Theta stds: {np.min(all_theta_stds):.3f} to {np.max(all_theta_stds):.3f}")
    
    # 6. Success summary
    total_sequences = sum(r['total_sequences'] for r in all_results)
    total_beta_values = sum(r['total_beta_values'] for r in all_results)
    
    print(f"\n🎉 SUCCESS SUMMARY:")
    print(f"  Total models processed: {len(all_results)}")
    print(f"  Total sequences analyzed: {total_sequences:,}")
    print(f"  Total beta parameters extracted: {total_beta_values:,}")
    print(f"  Extraction method: GPCM computation pathway")
    print(f"  Verification status: {'✅ ALL PASSED' if all_match_gpcm else '❌ SOME FAILED'}")
    
    if all_match_gpcm:
        print(f"\n✨ CONCLUSION:")
        print(f"  The beta parameter extraction method successfully extracts the EXACT")
        print(f"  same beta parameters that are used in GPCM probability computation")
        print(f"  across all tested model types. This confirms that the extraction")
        print(f"  method perfectly mirrors the computational pathway as requested.")
    else:
        print(f"\n⚠️  WARNING:")
        print(f"  Some extractions did not match GPCM computation. Review failed cases.")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Analyze Beta Parameter Extraction Results')
    parser.add_argument('--results_dir', default='results/beta_extraction', 
                       help='Directory containing extraction results')
    
    args = parser.parse_args()
    
    results = analyze_extraction_results(args.results_dir)
    
    if results:
        return 0
    else:
        return 1


if __name__ == "__main__":
    exit(main())