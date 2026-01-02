"""
Publication-Quality Visualization Generator
Honest Evaluation of Multimodal Deep Learning for Early Dementia Detection

ALL RESULTS ARE FROZEN - This script only visualizes existing reported metrics.
NO new experiments, NO model changes, NO data processing.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Create output directory
os.makedirs('figures', exist_ok=True)

# Set publication-quality style
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 9
plt.rcParams['ytick.labelsize'] = 9
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.titlesize'] = 12

# Color scheme (consistent across all plots)
COLOR_MRI = '#2E86DE'      # Blue
COLOR_LATE = '#10AC84'     # Green  
COLOR_ATTN = '#EE5A6F'     # Orange/Red

# ============================================================================
# PART A: OASIS-ONLY VISUALS
# ============================================================================

def figure_a1_oasis_model_comparison():
    """OASIS In-Dataset Model Comparison"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    models = ['MRI-Only', 'Late Fusion', 'Attention\nFusion']
    aucs = [0.770, 0.794, 0.790]
    stds = [0.080, 0.083, 0.109]
    colors = [COLOR_MRI, COLOR_LATE, COLOR_ATTN]
    
    x = np.arange(len(models))
    bars = ax.bar(x, aucs, yerr=stds, capsize=5, color=colors, alpha=0.8, 
                   edgecolor='black', linewidth=1.2)
    
    ax.set_ylabel('ROC-AUC', fontweight='bold')
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_title('OASIS-1 In-Dataset Performance\n(Homogeneous Single-Site Data)', 
                 fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim([0.6, 1.0])
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Chance')
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    
    # Annotate values
    for i, (bar, auc, std) in enumerate(zip(bars, aucs, stds)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + std + 0.02,
                f'{auc:.3f}±{std:.3f}',
                ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('figures/A1_oasis_model_comparison.png', bbox_inches='tight')
    plt.savefig('figures/A1_oasis_model_comparison.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Generated A1: OASIS Model Comparison")


def figure_a2_oasis_class_distribution():
    """OASIS Class Distribution"""
    fig, ax = plt.subplots(figsize=(5, 4))
    
    labels = ['CDR 0\n(Normal)', 'CDR 0.5\n(Very Mild\nDementia)']
    sizes = [138, 67]
    colors = ['#58B19F', '#F4B740']
    explode = (0.05, 0.05)
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                        startangle=90, colors=colors, explode=explode,
                                        textprops={'fontsize': 10, 'fontweight': 'bold'},
                                        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize(11)
        autotext.set_fontweight('bold')
    
    ax.set_title('OASIS-1 Class Distribution\n(N=205 subjects)', fontweight='bold', fontsize=12)
    
    # Add legend with counts
    legend_labels = [f'{label.replace(chr(10), " ")}: n={count}' 
                     for label, count in zip(labels, sizes)]
    ax.legend(legend_labels, loc='upper right', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig('figures/A2_oasis_class_distribution.png', bbox_inches='tight')
    plt.savefig('figures/A2_oasis_class_distribution.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Generated A2: OASIS Class Distribution")


# ============================================================================
# PART B: ADNI-ONLY VISUALS
# ============================================================================

def figure_b1_adni_level1_comparison():
    """ADNI Level-1 (Honest) Model Comparison"""
    fig, ax = plt.subplots(figsize=(6, 4))
    
    models = ['MRI-Only', 'Late Fusion', 'Attention\nFusion']
    aucs = [0.583, 0.598, 0.571]
    colors = [COLOR_MRI, COLOR_LATE, COLOR_ATTN]
    
    x = np.arange(len(models))
    bars = ax.bar(x, aucs, color=colors, alpha=0.8, edgecolor='black', linewidth=1.2)
    
    ax.set_ylabel('ROC-AUC', fontweight='bold')
    ax.set_xlabel('Model', fontweight='bold')
    ax.set_title('ADNI-1 Level-1: Honest Baseline\n(MRI + Age + Sex ONLY - NO MMSE/CDR-SB)', 
                 fontweight='bold', color='#D63031')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim([0.4, 1.0])
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1.5, alpha=0.7, label='Chance')
    ax.axhline(y=0.7, color='green', linestyle=':', linewidth=1, alpha=0.5, label='Competitive Threshold')
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    
    # Annotate values
    for i, (bar, auc) in enumerate(zip(bars, aucs)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{auc:.3f}',
                ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Add text box
    textstr = 'Realistic early detection\nis HARD without\ncognitive scores'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    ax.text(0.98, 0.97, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right', bbox=props)
    
    ax.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig('figures/B1_adni_level1_honest.png', bbox_inches='tight')
    plt.savefig('figures/B1_adni_level1_honest.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Generated B1: ADNI Level-1 Honest Baseline")


def figure_b2_adni_level1_vs_level2():
    """ADNI Level-1 vs Level-2 Contrast (CRITICAL CIRCULARITY FIGURE)"""
    fig, ax = plt.subplots(figsize=(7, 3.5))
    
    levels = ['Level-1\n(Honest)\nMRI + Age + Sex', 
              'Level-2\n(Circular)\n+ MMSE']
    aucs = [0.598, 0.988]
    colors = ['#F39C12', '#E74C3C']
    
    y = np.arange(len(levels))
    bars = ax.barh(y, aucs, color=colors, alpha=0.85, edgecolor='black', linewidth=1.5, height=0.6)
    
    ax.set_xlabel('ROC-AUC', fontweight='bold', fontsize=12)
    ax.set_title('Level-1 vs Level-2: Exposing Circular Feature Dominance', 
                 fontweight='bold', fontsize=13)
    ax.set_yticks(y)
    ax.set_yticklabels(levels, fontsize=11)
    ax.set_xlim([0.0, 1.05])
    ax.axvline(x=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5)
    ax.grid(axis='x', alpha=0.3, linestyle=':')
    
    # Annotate values
    for i, (bar, auc) in enumerate(zip(bars, aucs)):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2.,
                f'  {auc:.3f}',
                va='center', fontsize=12, fontweight='bold')
    
    # Draw arrow showing jump
    ax.annotate('', xy=(0.988, 0.7), xytext=(0.598, 0.3),
                arrowprops=dict(arrowstyle='<->', lw=2, color='red'))
    ax.text(0.79, 0.5, f'+0.390\n(65% gain)', ha='center', fontsize=11, 
            fontweight='bold', color='red',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8))
    
    # Add interpretation box
    textstr = 'MMSE dominates prediction.\nLiterature AUC 0.90-0.95\nlikely due to MMSE inclusion.'
    props = dict(boxstyle='round', facecolor='lightcoral', alpha=0.6)
    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig('figures/B2_level1_vs_level2_circularity.png', bbox_inches='tight')
    plt.savefig('figures/B2_level1_vs_level2_circularity.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Generated B2: Level-1 vs Level-2 Circularity (CRITICAL)")


def figure_b3_adni_class_distribution():
    """ADNI Class Distribution"""
    fig, ax = plt.subplots(figsize=(5, 4))
    
    labels = ['CN\n(Cognitively\nNormal)', 'MCI\n(Mild Cognitive\nImpairment)', 
              'AD\n(Alzheimer\'s\nDisease)']
    sizes = [194, 302, 133]
    colors = ['#58B19F', '#F4B740', '#E74C3C']
    explode = (0.05, 0, 0)
    
    wedges, texts, autotexts = ax.pie(sizes, labels=labels, autopct='%1.1f%%',
                                        startangle=90, colors=colors, explode=explode,
                                        textprops={'fontsize': 9, 'fontweight': 'bold'},
                                        wedgeprops={'edgecolor': 'black', 'linewidth': 1.5})
    
    for autotext in autotexts:
        autotext.set_color('white')
        autotext.set_fontsize=10
        autotext.set_fontweight('bold')
    
    ax.set_title('ADNI-1 Diagnostic Distribution\n(N=629 baseline subjects)', 
                 fontweight='bold', fontsize=12)
    
    # Add legend with counts
    legend_labels = [f'{label.replace(chr(10), " ")}: n={count}' 
                     for label, count in zip(labels, sizes)]
    ax.legend(legend_labels, loc='upper right', fontsize=8, framealpha=0.9)
    
    # Add imbalance note
    textstr = f'Class Imbalance:\nPositive (MCI+AD): 69.2%\nNegative (CN): 30.8%'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.7)
    ax.text(0.02, 0.02, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom', bbox=props)
    
    plt.tight_layout()
    plt.savefig('figures/B3_adni_class_distribution.png', bbox_inches='tight')
    plt.savefig('figures/B3_adni_class_distribution.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Generated B3: ADNI Class Distribution")


# ============================================================================
# PART C: CROSS-DATASET VISUALS (MOST IMPORTANT)
# ============================================================================

def figure_c1_in_dataset_vs_cross_dataset():
    """In-Dataset vs Cross-Dataset Performance (FUSION COLLAPSE)"""
    fig, ax = plt.subplots(figsize=(8, 5))
    
    models = ['MRI-Only', 'Late Fusion', 'Attention Fusion']
    
    # Data
    oasis_in = [0.770, 0.794, 0.790]
    adni_in = [0.583, 0.598, 0.571]
    oasis_to_adni = [0.607, 0.575, 0.557]
    adni_to_oasis = [0.569, 0.624, 0.548]
    
    x = np.arange(len(models))
    width = 0.2
    
    bars1 = ax.bar(x - 1.5*width, oasis_in, width, label='OASIS In-Dataset', 
                   color='#27AE60', alpha=0.9, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x - 0.5*width, adni_in, width, label='ADNI Level-1 In-Dataset', 
                   color='#3498DB', alpha=0.9, edgecolor='black', linewidth=1)
    bars3 = ax.bar(x + 0.5*width, oasis_to_adni, width, label='OASIS→ADNI Transfer', 
                   color='#E67E22', alpha=0.9, edgecolor='black', linewidth=1)
    bars4 = ax.bar(x + 1.5*width, adni_to_oasis, width, label='ADNI→OASIS Transfer', 
                   color='#9B59B6', alpha=0.9, edgecolor='black', linewidth=1)
    
    ax.set_ylabel('ROC-AUC', fontweight='bold', fontsize=12)
    ax.set_xlabel('Model', fontweight='bold', fontsize=12)
    ax.set_title('In-Dataset vs Cross-Dataset Robustness:\nFusion Gains Collapse Under Dataset Shift', 
                 fontweight='bold', fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim([0.4, 1.0])
    ax.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.7, 
               label='Chance Level')
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    ax.legend(loc='upper right', ncol=2, fontsize=8, framealpha=0.95)
    
    # Highlight key findings
    # Best OASIS→ADNI (MRI-Only)
    ax.plot(0 + 0.5*width, 0.607, marker='*', markersize=15, color='gold', 
            markeredgecolor='black', markeredgewidth=1.5, zorder=10)
    
    # Best ADNI→OASIS (Late Fusion)
    ax.plot(1 + 1.5*width, 0.624, marker='*', markersize=15, color='gold', 
            markeredgecolor='black', markeredgewidth=1.5, zorder=10)
    
    plt.tight_layout()
    plt.savefig('figures/C1_in_vs_cross_dataset_collapse.png', bbox_inches='tight')
    plt.savefig('figures/C1_in_vs_cross_dataset_collapse.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Generated C1: In-Dataset vs Cross-Dataset (FUSION COLLAPSE)")


def figure_c2_transfer_robustness_heatmap():
    """Transfer Robustness Heatmap (Asymmetry)"""
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.5))
    
    models = ['MRI-Only', 'Late Fusion', 'Attention Fusion']
    
    # Data: rows=source, cols=target
    data_mri = np.array([[0.770, 0.607],   # OASIS row
                         [0.569, 0.583]])  # ADNI row
    
    data_late = np.array([[0.794, 0.575],
                          [0.624, 0.598]])
    
    data_attn = np.array([[0.790, 0.557],
                          [0.548, 0.571]])
    
    data_all = [data_mri, data_late, data_attn]
    
    for idx, (ax, data, model) in enumerate(zip(axes, data_all, models)):
        sns.heatmap(data, annot=True, fmt='.3f', cmap='RdYlGn', center=0.65,
                    vmin=0.5, vmax=0.85, cbar=(idx==2), 
                    xticklabels=['OASIS', 'ADNI'],
                    yticklabels=['OASIS', 'ADNI'],
                    linewidths=2, linecolor='black',
                    annot_kws={'fontsize': 11, 'fontweight': 'bold'},
                    ax=ax)
        
        ax.set_title(f'{model}', fontweight='bold', fontsize=11)
        ax.set_xlabel('Target Dataset', fontweight='bold', fontsize=10)
        if idx == 0:
            ax.set_ylabel('Source Dataset', fontweight='bold', fontsize=10)
        else:
            ax.set_ylabel('')
        
        # Highlight best per transfer direction
        if model == 'MRI-Only':
            # Best OASIS→ADNI
            ax.add_patch(plt.Rectangle((1, 0), 1, 1, fill=False, edgecolor='gold', 
                                       lw=4, linestyle='--'))
        elif model == 'Late Fusion':
            # Best ADNI→OASIS
            ax.add_patch(plt.Rectangle((0, 1), 1, 1, fill=False, edgecolor='gold', 
                                       lw=4, linestyle='--'))
    
    fig.suptitle('Transfer Robustness Matrix: Asymmetric Best-Model Selection\n(Gold borders = Best per direction)', 
                 fontweight='bold', fontsize=12)
    plt.tight_layout()
    plt.savefig('figures/C2_transfer_robustness_heatmap.png', bbox_inches='tight')
    plt.savefig('figures/C2_transfer_robustness_heatmap.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Generated C2: Transfer Robustness Heatmap (ASYMMETRY)")


def figure_c3_auc_drop_visualization():
    """AUC Drop Visualization (Complexity = Fragility)"""
    fig, ax = plt.subplots(figsize=(7, 4.5))
    
    models = ['MRI-Only', 'Late Fusion', 'Attention\nFusion']
    
    # OASIS→ADNI drops
    oasis_to_adni_drops = [-0.207, -0.289, -0.269]
    
    # ADNI→OASIS drops
    adni_to_oasis_drops = [-0.117, -0.110, -0.165]
    
    x = np.arange(len(models))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, oasis_to_adni_drops, width, 
                   label='OASIS→ADNI Drop', color='#E67E22', alpha=0.85, 
                   edgecolor='black', linewidth=1.2)
    bars2 = ax.bar(x + width/2, adni_to_oasis_drops, width, 
                   label='ADNI→OASIS Drop', color='#9B59B6', alpha=0.85, 
                   edgecolor='black', linewidth=1.2)
    
    ax.set_ylabel('ΔAUC (Transfer - In-Dataset)', fontweight='bold', fontsize=11)
    ax.set_xlabel('Model', fontweight='bold', fontsize=11)
    ax.set_title('Performance Drop Under Cross-Dataset Transfer\n(Lower drop = More robust)', 
                 fontweight='bold', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.axhline(y=0, color='green', linestyle='--', linewidth=1.5, alpha=0.7, 
               label='No Degradation')
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    ax.legend(loc='lower right', fontsize=9)
    ax.set_ylim([-0.35, 0.05])
    
    # Annotate values
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height - 0.015,
                    f'{height:.3f}',
                    ha='center', va='top', fontsize=8, fontweight='bold', color='white')
    
    # Highlight smallest drops (most robust)
    # OASIS→ADNI: MRI-Only best
    ax.plot(0 - width/2, -0.207, marker='o', markersize=12, color='lime', 
            markeredgecolor='black', markeredgewidth=2, zorder=10)
    
    # ADNI→OASIS: Late Fusion best  
    ax.plot(1 + width/2, -0.110, marker='o', markersize=12, color='lime', 
            markeredgecolor='black', markeredgewidth=2, zorder=10)
    
    # Add interpretation
    textstr = 'Key: Smaller drop = Better generalization\nMRI-Only robust OASIS→ADNI\nLate Fusion robust ADNI→OASIS'
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    ax.text(0.98, 0.02, textstr, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom', horizontalalignment='right', bbox=props)
    
    plt.tight_layout()
    plt.savefig('figures/C3_auc_drop_robustness.png', bbox_inches='tight')
    plt.savefig('figures/C3_auc_drop_robustness.pdf', bbox_inches='tight')
    plt.close()
    print("✓ Generated C3: AUC Drop Visualization (COMPLEXITY=FRAGILITY)")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("Generating Publication-Quality Visualizations")
    print("All results frozen - NO experiments, only plotting")
    print("="*60 + "\n")
    
    print("PART A: OASIS-ONLY VISUALS")
    print("-" * 40)
    figure_a1_oasis_model_comparison()
    figure_a2_oasis_class_distribution()
    
    print("\nPART B: ADNI-ONLY VISUALS")
    print("-" * 40)
    figure_b1_adni_level1_comparison()
    figure_b2_adni_level1_vs_level2()
    figure_b3_adni_class_distribution()
    
    print("\nPART C: CROSS-DATASET VISUALS (MOST IMPORTANT)")
    print("-" * 40)
    figure_c1_in_dataset_vs_cross_dataset()
    figure_c2_transfer_robustness_heatmap()
    figure_c3_auc_drop_visualization()
    
    print("\n" + "="*60)
    print("✅ ALL VISUALIZATIONS GENERATED SUCCESSFULLY")
    print("Output directory: figures/")
    print("Formats: .png (high-res) + .pdf (vector)")
    print("="*60 + "\n")
    
    print("Next steps:")
    print("1. Review all figures in figures/ directory")
    print("2. Select 4 essential figures for main paper")
    print("3. Move remaining figures to supplementary material")
    print("\nNo further coding needed - visualizations complete!")
