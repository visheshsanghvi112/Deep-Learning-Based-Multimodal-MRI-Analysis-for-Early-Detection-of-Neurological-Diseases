"""
Generate key visualizations for the Longitudinal ADNI experiment.
These visualizations document the research journey from initial negative results
to breakthrough findings with biomarkers.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# Create output directory
output_dir = r"d:\discs\figures\longitudinal"
os.makedirs(output_dir, exist_ok=True)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

# =============================================================================
# Figure L1: Phase 1 Results - ResNet Features (Near-Chance)
# =============================================================================
def create_phase1_results():
    fig, ax = plt.subplots(figsize=(8, 5))
    
    models = ['Single-Scan\n(Baseline)', 'Delta Model\n(Baseline + Δ)', 'LSTM\n(Sequence)']
    aucs = [0.510, 0.517, 0.441]
    colors = ['#6b7280', '#6b7280', '#ef4444']  # Gray, Gray, Red for LSTM
    
    bars = ax.bar(models, aucs, color=colors, edgecolor='white', linewidth=2)
    
    # Add value labels
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02, 
                f'{auc:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Add chance line
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Chance (0.50)')
    
    ax.set_ylim(0, 0.7)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('Phase 1: ResNet Features - Near-Chance Performance', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right')
    
    # Add annotation
    ax.text(0.5, 0.05, '❌ All models near chance level → Prompted deep investigation', 
            transform=ax.transAxes, ha='center', fontsize=10, 
            bbox=dict(boxstyle='round', facecolor='#fef2f2', edgecolor='#ef4444'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'L1_phase1_resnet_results.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: L1_phase1_resnet_results.png")

# =============================================================================
# Figure L2: Individual Biomarker Predictive Power
# =============================================================================
def create_biomarker_power():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    biomarkers = ['Hippocampus', 'Entorhinal', 'MidTemp', 'Fusiform', 
                  'WholeBrain', 'Ventricles', 'ICV']
    aucs = [0.725, 0.691, 0.678, 0.670, 0.604, 0.581, 0.537]
    
    # Color gradient from green (best) to gray (worst)
    colors = plt.cm.RdYlGn(np.linspace(0.8, 0.3, len(biomarkers)))
    
    bars = ax.barh(biomarkers[::-1], aucs[::-1], color=colors[::-1], edgecolor='white', linewidth=2)
    
    # Add value labels
    for bar, auc in zip(bars, aucs[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{auc:.3f}', ha='left', va='center', fontsize=10, fontweight='bold')
    
    # Add chance line
    ax.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Chance')
    
    ax.set_xlim(0.4, 0.8)
    ax.set_xlabel('AUC', fontsize=12)
    ax.set_title('Individual Biomarker Predictive Power (MCI Cohort, N=737)', fontsize=14, fontweight='bold')
    
    # Highlight best predictor
    ax.text(0.725, 6.5, '← BEST: Hippocampus', fontsize=10, color='#059669', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'L2_biomarker_power.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: L2_biomarker_power.png")

# =============================================================================
# Figure L3: Feature Combination Results (The Breakthrough)
# =============================================================================
def create_feature_combinations():
    fig, ax = plt.subplots(figsize=(10, 6))
    
    approaches = ['ResNet\nfeatures', 'Baseline\nbiomarkers', 'Delta\nbiomarkers', 
                  'Baseline +\nDelta (RF)', '+ Age +\nAPOE4', '+ ADAS13']
    aucs = [0.52, 0.736, 0.759, 0.848, 0.813, 0.842]
    
    colors = ['#ef4444', '#3b82f6', '#3b82f6', '#10b981', '#10b981', '#8b5cf6']
    
    bars = ax.bar(approaches, aucs, color=colors, edgecolor='white', linewidth=2)
    
    # Add value labels
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{auc:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add improvement annotations
    ax.annotate('', xy=(3, 0.831), xytext=(0, 0.52),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(1.5, 0.68, '+31 pts!', fontsize=12, color='green', fontweight='bold')
    
    ax.set_ylim(0, 0.95)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_title('Phase 3: Feature Combination Results', fontsize=14, fontweight='bold')
    
    # Add legend
    legend_elements = [
        mpatches.Patch(color='#ef4444', label='ResNet (failed)'),
        mpatches.Patch(color='#3b82f6', label='Biomarkers'),
        mpatches.Patch(color='#10b981', label='Biomarkers + Longitudinal'),
        mpatches.Patch(color='#8b5cf6', label='+ Cognitive (semi-circular)')
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'L3_feature_combinations.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: L3_feature_combinations.png")

# =============================================================================
# Figure L4: APOE4 Genetic Risk Analysis
# =============================================================================
def create_apoe4_analysis():
    fig, ax = plt.subplots(figsize=(8, 5))
    
    alleles = ['0 alleles\n(Non-carrier)', '1 allele\n(Heterozygous)', '2 alleles\n(Homozygous)']
    conversion_rates = [23.5, 44.2, 49.1]
    n_subjects = [511, 394, 110]
    
    colors = ['#22c55e', '#f59e0b', '#ef4444']  # Green, Orange, Red
    
    bars = ax.bar(alleles, conversion_rates, color=colors, edgecolor='white', linewidth=2)
    
    # Add value labels
    for bar, rate, n in zip(bars, conversion_rates, n_subjects):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=14, fontweight='bold')
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                f'N={n}', ha='center', va='center', fontsize=10, color='white')
    
    ax.set_ylim(0, 60)
    ax.set_ylabel('MCI → Dementia Conversion Rate (%)', fontsize=12)
    ax.set_title('APOE4 Genetic Risk: Carriers Have 2x Conversion Risk', fontsize=14, fontweight='bold')
    
    # Add annotation
    ax.annotate('2x higher\nrisk!', xy=(1.5, 46), xytext=(2.3, 55),
                fontsize=12, fontweight='bold', color='#ef4444',
                arrowprops=dict(arrowstyle='->', color='#ef4444', lw=2))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'L4_apoe4_risk.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: L4_apoe4_risk.png")

# =============================================================================
# Figure L5: Longitudinal Improvement (+9.5%)
# =============================================================================
def create_longitudinal_improvement():
    fig, ax = plt.subplots(figsize=(8, 5))
    
    conditions = ['Baseline\nBiomarkers Only', 'Baseline +\nLongitudinal Δ (RF)']
    aucs = [0.736, 0.848]
    
    colors = ['#3b82f6', '#10b981']
    
    bars = ax.bar(conditions, aucs, color=colors, edgecolor='white', linewidth=3, width=0.5)
    
    # Add value labels
    for bar, auc in zip(bars, aucs):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{auc:.3f}', ha='center', va='bottom', fontsize=16, fontweight='bold')
    
    # Add improvement arrow
    ax.annotate('', xy=(1, 0.848), xytext=(0, 0.736),
                arrowprops=dict(arrowstyle='->', color='#059669', lw=3))
    ax.text(0.5, 0.79, '+11.2%', fontsize=20, color='#059669', fontweight='bold', ha='center')
    
    ax.set_ylim(0.5, 0.95)
    ax.set_ylabel('AUC', fontsize=14)
    ax.set_title('Longitudinal Data Improves Prediction by +9.5%', fontsize=14, fontweight='bold')
    
    # Add conclusion box
    ax.text(0.5, 0.55, '✅ Longitudinal change (atrophy rate) is predictive!', 
            transform=ax.transAxes, ha='center', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='#ecfdf5', edgecolor='#10b981'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'L5_longitudinal_improvement.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: L5_longitudinal_improvement.png")

# =============================================================================
# Figure L6: Complete Research Journey Summary
# =============================================================================
def create_journey_summary():
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Timeline data
    phases = ['Phase 1\nResNet', 'Phase 2\nInvestigation', 'Phase 3\nBiomarkers']
    aucs = [0.52, None, 0.848]  # None for investigation phase
    x_positions = [0, 1, 2]
    
    # Draw timeline
    ax.plot([0, 2], [0.52, 0.52], 'r--', linewidth=2, alpha=0.5)  # Baseline
    ax.plot([0, 2], [0.848, 0.848], 'g--', linewidth=2, alpha=0.5)  # Target
    
    # Phase 1 - Red circle
    ax.scatter([0], [0.52], s=500, c='#ef4444', zorder=5, edgecolors='white', linewidth=3)
    ax.text(0, 0.52, '0.52', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    ax.text(0, 0.40, 'ResNet:\nNear Chance', ha='center', va='top', fontsize=10)
    
    # Phase 2 - Yellow circle (no AUC, just investigation)
    ax.scatter([1], [0.65], s=500, c='#f59e0b', zorder=5, edgecolors='white', linewidth=3)
    ax.text(1, 0.65, '🔍', ha='center', va='center', fontsize=16)
    ax.text(1, 0.53, 'Investigation:\n3 Issues Found', ha='center', va='top', fontsize=10)
    
    # Phase 3 - Green circle
    ax.scatter([2], [0.848], s=500, c='#10b981', zorder=5, edgecolors='white', linewidth=3)
    ax.text(2, 0.848, '0.85', ha='center', va='center', fontsize=12, fontweight='bold', color='white')
    ax.text(2, 0.73, 'Biomarkers:\nBreakthrough!', ha='center', va='top', fontsize=10)
    
    # Draw arrows
    ax.annotate('', xy=(0.85, 0.65), xytext=(0.15, 0.52),
                arrowprops=dict(arrowstyle='->', color='gray', lw=2))
    ax.annotate('', xy=(1.85, 0.848), xytext=(1.15, 0.65),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    
    ax.set_xlim(-0.5, 2.5)
    ax.set_ylim(0.35, 0.95)
    ax.set_ylabel('AUC', fontsize=12)
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(phases, fontsize=11)
    ax.set_title('Complete Research Journey: From Failure to Breakthrough', fontsize=14, fontweight='bold')
    
    # Add key insight
    ax.text(0.5, 0.92, 'Key Insight: Right features (biomarkers) > Complex models (LSTM)', 
            transform=ax.transAxes, ha='center', fontsize=11, 
            bbox=dict(boxstyle='round', facecolor='#f0fdf4', edgecolor='#10b981'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'L6_research_journey.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("Created: L6_research_journey.png")

# =============================================================================
# Run all visualizations
# =============================================================================
if __name__ == "__main__":
    print("Generating Longitudinal Experiment Visualizations...")
    print("=" * 50)
    
    create_phase1_results()
    create_biomarker_power()
    create_feature_combinations()
    create_apoe4_analysis()
    create_longitudinal_improvement()
    create_journey_summary()
    
    print("=" * 50)
    print(f"All visualizations saved to: {output_dir}")
    print("\nGenerated figures:")
    for f in os.listdir(output_dir):
        print(f"  - {f}")
