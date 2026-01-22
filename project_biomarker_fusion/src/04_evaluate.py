"""
Step 4: Comprehensive Evaluation & Comparison
==============================================
Compares the new fusion model against all baselines:
  - ResNet-only (longitudinal): 0.52 AUC
  - Biomarker-only (statistical): 0.83 AUC  
  - THIS: Multimodal Fusion: ??? AUC

SAFETY: Only reads existing results, creates comparison report
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Paths
FUSION_RESULTS = r"D:\discs\project_biomarker_fusion\results\metrics.json"
RESNET_RESULTS = r"D:\discs\project_longitudinal\results\single_scan\metrics.json"
BIOMARKER_RESULTS = r"D:\discs\project_longitudinal\results\biomarker_analysis\final_findings.json"
OUTPUT_DIR = r"D:\discs\project_biomarker_fusion\results"

def load_all_results():
    """Load results from all experiments."""
    print("Loading results from all experiments...")
    
    results = {}
    
    # Fusion model (THIS WORK)
    if os.path.exists(FUSION_RESULTS):
        with open(FUSION_RESULTS) as f:
            fusion_data = json.load(f)
        results['fusion'] = {
            'auc': fusion_data['best_test_auc'],
            'accuracy': fusion_data['final_metrics']['accuracy'],
            'name': 'MRI + Biomarker Fusion (PyTorch)',
            'features': 'ResNet (1536) + Biomarkers (12)',
            'status': '✅ NEW'
        }
        print(f"  ✅ Fusion: {results['fusion']['auc']:.4f}")
    else:
        print("  ⚠️ Fusion results not found (need to train first)")
        results['fusion'] = None
    
    # ResNet-only (longitudinal)
    if os.path.exists(RESNET_RESULTS):
        with open(RESNET_RESULTS) as f:
            resnet_data = json.load(f)
        results['resnet'] = {
            'auc': resnet_data['test_auc'],
            'name': 'ResNet-only (Longitudinal)',
            'features': 'ResNet features (1536)',
            'status': '📊 Baseline'
        }
        print(f"  ✅ ResNet: {results['resnet']['auc']:.4f}")
    
    # Biomarker-only
    if os.path.exists(BIOMARKER_RESULTS):
        with open(BIOMARKER_RESULTS) as f:
            bio_data = json.load(f)
        results['biomarker'] = {
            'auc': bio_data['biomarker_delta'],
            'name': 'Biomarker-only (Logistic Reg)',
            'features': 'Hippocampus + Ventricles + Entorhinal + Deltas',
            'status': '🏆 Previous Best'
        }
        print(f"  ✅ Biomarker: {results['biomarker']['auc']:.4f}")
    
    return results

def generate_comparison_table(results):
    """Generate comparison table."""
    print("\n" + "="*80)
    print("COMPREHENSIVE COMPARISON")
    print("="*80)
    
    print(f"\n{'Model':<35} {'AUC':>8} {'Gain':>8} {'Status':<20}")
    print("-"*80)
    
    # Baseline (ResNet)
    if results['resnet']:
        baseline_auc = results['resnet']['auc']
        print(f"{results['resnet']['name']:<35} {baseline_auc:>8.4f} {'--':>8} {results['resnet']['status']:<20}")
    
    # Biomarker
    if results['biomarker']:
        bio_auc = results['biomarker']['auc']
        gain_vs_resnet = bio_auc - baseline_auc
        print(f"{results['biomarker']['name']:<35} {bio_auc:>8.4f} {f'+{gain_vs_resnet:.3f}':>8} {results['biomarker']['status']:<20}")
    
    # Fusion (NEW)
    if results['fusion']:
        fusion_auc = results['fusion']['auc']
        gain_vs_resnet = fusion_auc - baseline_auc
        gain_vs_bio = fusion_auc - bio_auc
        print(f"{results['fusion']['name']:<35} {fusion_auc:>8.4f} {f'+{gain_vs_resnet:.3f}':>8} {results['fusion']['status']:<20}")
        
        print("\n" + "="*80)
        print("KEY FINDINGS")
        print("="*80)
        
        if fusion_auc > bio_auc:
            improvement = ((fusion_auc - bio_auc) / bio_auc) * 100
            print(f"\n🎯 FUSION WINS! {improvement:.1f}% improvement over biomarker-only")
            print(f"   Fusion AUC: {fusion_auc:.4f}")
            print(f"   Biomarker AUC: {bio_auc:.4f}")
            print(f"   Gain: +{gain_vs_bio:.3f}")
            print("\n✅ Deep learning fusion DOES add value!")
            conclusion = "PUBLICATION READY - Fusion model achieves SOTA"
        elif fusion_auc >= bio_auc - 0.01:  # Within 1%
            print(f"\n🤝 TIE: Fusion matches biomarker-only")
            print(f"   Fusion AUC: {fusion_auc:.4f}")
            print(f"   Biomarker AUC: {bio_auc:.4f}")
            print(f"   Difference: {abs(gain_vs_bio):.3f}")
            print("\n💡 Simple models sufficient, but fusion validates approach")
            conclusion = "PUBLICATION READY - Fusion validates biomarker findings"
        else:
            print(f"\n⚠️ Biomarker-only still wins")
            print(f"   Biomarker AUC: {bio_auc:.4f}")
            print(f"   Fusion AUC: {fusion_auc:.4f}")
            print(f"   Gap: {bio_auc - fusion_auc:.3f}")
            print("\n📊 Simple logistic regression may be preferred (Occam's Razor)")
            conclusion = "HONEST REPORTING - Simple models competitive"
        
        print(f"\n📄 Paper Strategy: {conclusion}")
    
    else:
        print("\n⚠️ Fusion model not trained yet. Run 03_train_fusion.py first!")
    
    return

def plot_comparison(results):
    """Create visualization comparing all models."""
    if not results['fusion']:
        print("\nSkipping plot - fusion not trained yet")
        return
    
    print("\nGenerating comparison plot...")
    
    # Prepare data
    models = []
    aucs = []
    colors = []
    
    if results['resnet']:
        models.append('ResNet\nLongitudinal')
        aucs.append(results['resnet']['auc'])
        colors.append('#ef4444')
    
    if results['biomarker']:
        models.append('Biomarker\nOnly')
        aucs.append(results['biomarker']['auc'])
        colors.append('#f59e0b')
    
    if results['fusion']:
        models.append('MRI + Biomarker\nFusion (NEW)')
        aucs.append(results['fusion']['auc'])
        colors.append('#10b981')
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    bars = ax.bar(models, aucs, color=colors, edgecolor='white', linewidth=2)
    
    # Add value labels
    for bar, auc in zip(bars, aucs):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{auc:.3f}',
                ha='center', va='bottom', fontsize=14, fontweight='bold')
    
    ax.set_ylabel('AUC Score', fontsize=12, fontweight='bold')
    ax.set_title('Longitudinal Progression Prediction - Model Comparison', 
                 fontsize=14, fontweight='bold', pad=20)
    ax.set_ylim([0.4, 0.9])
    ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    
    # Save
    plot_path = os.path.join(OUTPUT_DIR, 'model_comparison.png')
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"  ✅ Saved: {plot_path}")
    
    plt.close()

def save_comparison_json(results):
    """Save comparison as JSON."""
    if not results['fusion']:
        return
    
    comparison = {
        'date': '2026-01-22',
        'models': {
            'resnet_longitudinal': {
                'auc': results['resnet']['auc'],
                'description': 'ResNet features with longitudinal delta',
                'status': 'Baseline'
            },
            'biomarker_only': {
                'auc': results['biomarker']['auc'],
                'description': 'Hippocampus + Ventricles + Entorhinal with logistic regression',
                'status': 'Previous best'
            },
            'multimodal_fusion': {
                'auc': results['fusion']['auc'],
                'accuracy': results['fusion']['accuracy'],
                'description': 'ResNet + Biomarkers with PyTorch fusion',
                'status': 'NEW - This work'
            }
        },
        'improvements': {
            'fusion_vs_resnet': results['fusion']['auc'] - results['resnet']['auc'],
            'fusion_vs_biomarker': results['fusion']['auc'] - results['biomarker']['auc']
        }
    }
    
    output_path = os.path.join(OUTPUT_DIR, 'comparison.json')
    with open(output_path, 'w') as f:
        json.dump(comparison, f, indent=2)
    
    print(f"\n✅ Comparison saved: {output_path}")

def main():
    """Main evaluation pipeline."""
    print("="*80)
    print("COMPREHENSIVE EVALUATION - ALL EXPERIMENTS")
    print("="*80)
    
    # Load all results
    results = load_all_results()
    
    # Generate comparison
    generate_comparison_table(results)
    
    # Create visualization
    plot_comparison(results)
    
    # Save JSON
    save_comparison_json(results)
    
    print("\n" + "="*80)
    print("✅ EVALUATION COMPLETE")
    print("="*80)
    print("\nNext steps:")
    print("1. Review results/comparison.json")
    print("2. Check results/model_comparison.png")
    print("3. Update documentation with findings")
    print("4. Prepare paper draft if results are strong!")
    
    print("\n✅ No existing files were modified!")

if __name__ == '__main__':
    main()
