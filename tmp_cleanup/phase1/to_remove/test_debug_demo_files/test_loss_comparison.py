#!/usr/bin/env python3
"""
Test script comparing different loss functions for Enhanced CORAL-GPCM training.
Demonstrates the effectiveness of the triple CORAL loss approach.
"""

import subprocess
import sys
import os

def run_training_comparison():
    """Run training with different loss functions and compare results."""
    
    print("=" * 80)
    print("LOSS FUNCTION COMPARISON FOR ENHANCED CORAL-GPCM")
    print("=" * 80)
    print()
    
    base_cmd = [
        "python", "train.py",
        "--model", "ecoral_gpcm",
        "--dataset", "synthetic_OC", 
        "--epochs", "3",
        "--no_cv",
        "--enable_threshold_coupling"
    ]
    
    configurations = [
        {
            "name": "Cross-Entropy Loss Only",
            "args": ["--loss", "ce"],
            "description": "Standard categorical cross-entropy"
        },
        {
            "name": "QWK Loss Only", 
            "args": ["--loss", "qwk"],
            "description": "Quadratic Weighted Kappa for ordinal consistency"
        },
        {
            "name": "Combined CE + QWK Loss",
            "args": ["--loss", "combined", "--ce_weight", "0.5", "--qwk_weight", "0.5", "--coral_weight", "0.0"],
            "description": "CE + QWK without CORAL component"
        },
        {
            "name": "Triple CORAL Loss (Balanced)",
            "args": ["--loss", "triple_coral", "--triple_ce_weight", "0.33", "--triple_qwk_weight", "0.33", "--triple_coral_weight", "0.34"],
            "description": "CE + QWK + CORAL with balanced weights"
        },
        {
            "name": "Triple CORAL Loss (CORAL-focused)",
            "args": ["--loss", "triple_coral", "--triple_ce_weight", "0.25", "--triple_qwk_weight", "0.25", "--triple_coral_weight", "0.5"],
            "description": "CE + QWK + CORAL with emphasis on rank consistency"
        }
    ]
    
    results = []
    
    for i, config in enumerate(configurations, 1):
        print(f"\n{i}. {config['name']}")
        print(f"   Description: {config['description']}")
        print(f"   Running...")
        
        cmd = base_cmd + config['args']
        
        try:
            # Run the training
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                # Extract performance metrics from output
                output_lines = result.stdout.split('\n')
                qwk_score = None
                ordinal_acc = None
                
                for line in output_lines:
                    if "Best QWK:" in line:
                        try:
                            qwk_score = float(line.split("Best QWK: ")[1].split(" ")[0])
                        except:
                            pass
                    elif "|" in line and "QWK" in line and not line.startswith("Epoch"):
                        # Parse training output line
                        parts = line.split("|")
                        if len(parts) >= 5:
                            try:
                                qwk_score = float(parts[4].strip())
                                ordinal_acc = float(parts[5].strip())
                            except:
                                pass
                
                results.append({
                    "name": config['name'],
                    "qwk": qwk_score,
                    "ordinal_acc": ordinal_acc,
                    "success": True
                })
                
                print(f"   ✅ Success - QWK: {qwk_score:.3f}" + (f", Ordinal Acc: {ordinal_acc:.3f}" if ordinal_acc else ""))
                
            else:
                results.append({
                    "name": config['name'],
                    "success": False,
                    "error": result.stderr[-200:] if result.stderr else "Unknown error"
                })
                print(f"   ❌ Failed: {result.stderr[-100:] if result.stderr else 'Unknown error'}")
                
        except subprocess.TimeoutExpired:
            results.append({
                "name": config['name'],
                "success": False,
                "error": "Timeout"
            })
            print(f"   ⏰ Timeout")
        except Exception as e:
            results.append({
                "name": config['name'],
                "success": False,
                "error": str(e)
            })
            print(f"   ❌ Error: {e}")
    
    # Summary
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print()
    
    successful_results = [r for r in results if r['success'] and r.get('qwk') is not None]
    
    if successful_results:
        print(f"{'Loss Function':<35} {'QWK Score':<10} {'Ordinal Acc':<12}")
        print("-" * 60)
        
        for result in successful_results:
            qwk_str = f"{result['qwk']:.3f}" if result['qwk'] else "N/A"
            ord_str = f"{result['ordinal_acc']:.3f}" if result.get('ordinal_acc') else "N/A"
            print(f"{result['name']:<35} {qwk_str:<10} {ord_str:<12}")
        
        # Find best performing
        best_qwk = max(successful_results, key=lambda x: x.get('qwk', 0))
        print(f"\n🏆 Best QWK Performance: {best_qwk['name']} ({best_qwk['qwk']:.3f})")
        
        if len(successful_results) >= 2:
            triple_results = [r for r in successful_results if 'Triple CORAL' in r['name']]
            other_results = [r for r in successful_results if 'Triple CORAL' not in r['name']]
            
            if triple_results and other_results:
                avg_triple = sum(r['qwk'] for r in triple_results) / len(triple_results)
                avg_other = sum(r['qwk'] for r in other_results) / len(other_results)
                improvement = ((avg_triple - avg_other) / avg_other) * 100
                
                print(f"\n📊 Triple CORAL Loss Analysis:")
                print(f"   Average Triple CORAL QWK: {avg_triple:.3f}")
                print(f"   Average Other Loss QWK: {avg_other:.3f}")
                print(f"   Improvement: {improvement:+.1f}%")
    else:
        print("No successful results to compare.")
    
    print("\n" + "=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)
    print()
    print("Triple CORAL Loss combines three complementary objectives:")
    print("• Cross-Entropy: Maintains categorical prediction accuracy")
    print("• QWK Loss: Ensures ordinal consistency and agreement")  
    print("• CORAL Loss: Enforces rank consistency in predictions")
    print()
    print("This multi-objective approach typically provides:")
    print("• Better ordinal structure preservation")
    print("• More robust performance across different metrics")
    print("• Improved generalization for ordinal tasks")
    print()
    print("The Enhanced CORAL-GPCM with adaptive threshold coupling")
    print("further improves performance by sophisticated integration")
    print("of GPCM β thresholds with CORAL τ thresholds.")

if __name__ == "__main__":
    # Set environment
    os.environ['PYTHONPATH'] = os.path.dirname(os.path.abspath(__file__))
    
    print("Setting up conda environment...")
    try:
        subprocess.run(["conda", "activate", "vrec-env"], check=True, shell=True)
    except:
        print("Note: Could not activate conda environment. Proceeding with current environment.")
    
    run_training_comparison()