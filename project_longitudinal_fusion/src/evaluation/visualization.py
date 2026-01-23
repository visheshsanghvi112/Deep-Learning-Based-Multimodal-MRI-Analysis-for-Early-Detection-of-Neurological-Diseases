"""
Publication-Quality Visualizations
==================================
Figures for ROC curves, confusion matrices, attention weights, etc.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from sklearn.metrics import roc_curve, confusion_matrix, precision_recall_curve
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from config import FIGURE_DPI, FIGURE_FORMAT, COLORS, FIGURES_DIR

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.dpi': FIGURE_DPI,
    'savefig.dpi': FIGURE_DPI,
    'font.family': 'sans-serif'
})


def plot_roc_curves(
    y_true: np.ndarray,
    predictions: Dict[str, np.ndarray],
    title: str = "ROC Curves Comparison",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    Plot ROC curves for multiple models.
    
    Args:
        y_true: True labels
        predictions: Dict of {model_name: y_prob}
        title: Figure title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = [COLORS['resnet'], COLORS['biomarker'], COLORS['fusion'], 
              COLORS['warning'], COLORS['secondary']]
    
    for idx, (name, y_prob) in enumerate(predictions.items()):
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(y_true, y_prob)
        
        color = colors[idx % len(colors)]
        ax.plot(fpr, tpr, color=color, linewidth=2.5,
                label=f'{name} (AUC = {auc:.3f})')
    
    # Diagonal reference
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Random')
    
    ax.set_xlabel('False Positive Rate (1 - Specificity)')
    ax.set_ylabel('True Positive Rate (Sensitivity)')
    ax.set_title(title, fontweight='bold')
    ax.legend(loc='lower right', framealpha=0.9)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, format=FIGURE_FORMAT, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    class_names: List[str] = ['Stable', 'Converter'],
    title: str = "Confusion Matrix",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (8, 6)
) -> plt.Figure:
    """
    Plot confusion matrix with counts and percentages.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: Names for classes
        title: Figure title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    cm = confusion_matrix(y_true, y_pred)
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create custom colormap
    cmap = LinearSegmentedColormap.from_list(
        'custom', ['#F0F4F8', COLORS['primary']], N=256
    )
    
    sns.heatmap(cm, annot=False, cmap=cmap, ax=ax,
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'})
    
    # Add annotations with counts and percentages
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            text_color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
            ax.text(j + 0.5, i + 0.5, 
                   f'{cm[i, j]}\n({cm_percent[i, j]:.1f}%)',
                   ha='center', va='center', fontsize=14,
                   color=text_color, fontweight='bold')
    
    ax.set_xlabel('Predicted Label', fontweight='bold')
    ax.set_ylabel('True Label', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, format=FIGURE_FORMAT, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_training_history(
    history: Dict[str, List[float]],
    title: str = "Training History",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot training history (loss and AUC).
    
    Args:
        history: Dictionary with 'train_loss', 'val_loss', 'train_auc', 'val_auc'
        title: Figure title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1 = axes[0]
    ax1.plot(epochs, history['train_loss'], color=COLORS['primary'], 
            linewidth=2, label='Train Loss')
    ax1.plot(epochs, history['val_loss'], color=COLORS['danger'], 
            linewidth=2, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training & Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # AUC plot
    ax2 = axes[1]
    ax2.plot(epochs, history['train_auc'], color=COLORS['primary'], 
            linewidth=2, label='Train AUC')
    ax2.plot(epochs, history['val_auc'], color=COLORS['success'], 
            linewidth=2, label='Val AUC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC')
    ax2.set_title('Training & Validation AUC')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim([0.4, 1.0])
    
    # Mark best epoch
    best_epoch = np.argmax(history['val_auc']) + 1
    best_auc = max(history['val_auc'])
    ax2.axvline(x=best_epoch, color=COLORS['warning'], linestyle='--', 
               linewidth=1.5, alpha=0.7, label=f'Best: {best_auc:.4f}')
    ax2.scatter([best_epoch], [best_auc], color=COLORS['warning'], 
               s=100, zorder=5, marker='*')
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, format=FIGURE_FORMAT, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_attention_weights(
    temporal_weights: np.ndarray,
    fusion_gates: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    title: str = "Attention Weights Analysis",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 5)
) -> plt.Figure:
    """
    Plot attention weights for interpretability.
    
    Args:
        temporal_weights: [N, 3] temporal attention weights (baseline, followup, delta)
        fusion_gates: Tuple of (mri_gate, bio_gate) arrays
        title: Figure title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    n_plots = 2 if fusion_gates is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    # Temporal attention
    ax1 = axes[0]
    timepoints = ['Baseline', 'Follow-up', 'Delta']
    mean_weights = np.mean(temporal_weights, axis=0)
    std_weights = np.std(temporal_weights, axis=0)
    
    bars = ax1.bar(timepoints, mean_weights, yerr=std_weights, 
                  color=[COLORS['primary'], COLORS['secondary'], COLORS['success']],
                  capsize=5, alpha=0.8)
    ax1.set_ylabel('Attention Weight')
    ax1.set_title('Temporal Attention Weights')
    ax1.set_ylim([0, max(mean_weights) * 1.3])
    
    # Add value labels
    for bar, val in zip(bars, mean_weights):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    if fusion_gates is not None:
        ax2 = axes[1]
        mri_gate, bio_gate = fusion_gates
        
        mean_mri = np.mean(mri_gate)
        mean_bio = np.mean(bio_gate)
        
        bars = ax2.bar(['MRI', 'Biomarker'], [mean_mri, mean_bio],
                      color=[COLORS['resnet'], COLORS['biomarker']], alpha=0.8)
        ax2.set_ylabel('Gate Value')
        ax2.set_title('Modality Fusion Gates')
        ax2.set_ylim([0, 1])
        
        for bar, val in zip(bars, [mean_mri, mean_bio]):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.3f}', ha='center', va='bottom', fontweight='bold')
    
    fig.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, format=FIGURE_FORMAT, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def plot_feature_importance(
    importance_dict: Dict[str, float],
    top_k: int = 15,
    title: str = "Feature Importance",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Plot feature importance (e.g., from Random Forest).
    
    Args:
        importance_dict: Dictionary of {feature_name: importance}
        top_k: Number of top features to show
        title: Figure title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    # Sort by importance
    sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    top_items = sorted_items[:top_k]
    
    features = [item[0] for item in top_items]
    importances = [item[1] for item in top_items]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Color by feature type
    colors = []
    for feat in features:
        if 'resnet' in feat.lower() or 'mri' in feat.lower():
            colors.append(COLORS['resnet'])
        elif 'delta' in feat.lower():
            colors.append(COLORS['success'])
        else:
            colors.append(COLORS['biomarker'])
    
    bars = ax.barh(range(len(features)), importances, color=colors, alpha=0.8)
    ax.set_yticks(range(len(features)))
    ax.set_yticklabels(features)
    ax.set_xlabel('Importance')
    ax.set_title(title, fontweight='bold')
    ax.invert_yaxis()
    
    # Add legend
    legend_patches = [
        mpatches.Patch(color=COLORS['resnet'], label='MRI Features'),
        mpatches.Patch(color=COLORS['biomarker'], label='Biomarkers'),
        mpatches.Patch(color=COLORS['success'], label='Temporal Delta')
    ]
    ax.legend(handles=legend_patches, loc='lower right')
    
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, format=FIGURE_FORMAT, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig


def create_results_summary_figure(
    results: Dict[str, Dict],
    title: str = "Model Comparison Summary",
    save_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8)
) -> plt.Figure:
    """
    Create comprehensive results summary figure.
    
    Args:
        results: Dictionary of {model_name: {'auc': ..., 'ci_lower': ..., 'ci_upper': ...}}
        title: Figure title
        save_path: Path to save figure
        figsize: Figure size
        
    Returns:
        matplotlib Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Filter out non-model keys
    model_results = {k: v for k, v in results.items() 
                    if isinstance(v, dict) and 'auc' in v}
    
    model_names = list(model_results.keys())
    aucs = [model_results[name]['auc'] for name in model_names]
    ci_lowers = [model_results[name].get('ci_lower', aucs[i] - 0.05) 
                 for i, name in enumerate(model_names)]
    ci_uppers = [model_results[name].get('ci_upper', aucs[i] + 0.05) 
                 for i, name in enumerate(model_names)]
    
    # Error bars
    errors = [[auc - low for auc, low in zip(aucs, ci_lowers)],
              [high - auc for auc, high in zip(aucs, ci_uppers)]]
    
    # AUC comparison (bar chart with CI)
    ax1 = axes[0]
    colors = [COLORS['resnet'] if 'resnet' in name.lower() else
              COLORS['biomarker'] if 'bio' in name.lower() else
              COLORS['fusion'] for name in model_names]
    
    bars = ax1.bar(range(len(model_names)), aucs, yerr=errors,
                  color=colors, capsize=5, alpha=0.8)
    ax1.set_xticks(range(len(model_names)))
    ax1.set_xticklabels(model_names, rotation=45, ha='right')
    ax1.set_ylabel('AUC')
    ax1.set_title('Model Comparison (with 95% CI)', fontweight='bold')
    ax1.set_ylim([0.4, 1.0])
    ax1.axhline(y=0.83, color=COLORS['danger'], linestyle='--', 
               linewidth=1.5, label='Baseline (0.83)')
    ax1.legend()
    
    # Add value labels
    for bar, auc in zip(bars, aucs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{auc:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')
    
    # Metrics table
    ax2 = axes[1]
    ax2.axis('off')
    
    # Create table data
    metrics = ['AUC', 'Accuracy', 'Sensitivity', 'Specificity', 'F1']
    table_data = []
    
    for name in model_names[:4]:  # Limit to 4 models for readability
        row = [name]
        for metric in ['auc', 'accuracy', 'recall', 'specificity', 'f1']:
            val = model_results[name].get(metric, 0)
            row.append(f'{val:.3f}')
        table_data.append(row)
    
    table = ax2.table(
        cellText=table_data,
        colLabels=['Model'] + metrics,
        loc='center',
        cellLoc='center'
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.8)
    
    # Style header
    for i in range(len(metrics) + 1):
        table[(0, i)].set_facecolor(COLORS['primary'])
        table[(0, i)].set_text_props(color='white', fontweight='bold')
    
    ax2.set_title('Detailed Metrics', fontweight='bold', y=0.95)
    
    fig.suptitle(title, fontsize=18, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, format=FIGURE_FORMAT, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    return fig
