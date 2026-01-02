import os
import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
# Add source path to allow imports if needed, though we primarily visualization here
import sys
sys.path.append(r"d:\discs\project_adni\src")

# We need the model definitions to load weights. 
# Importing directly from train_level_max.py to avoid duplication
from train_level_max import MRIOnlyModel, LateFusionModel, AttentionFusionModel, Config, load_adni_level_max, ADNIDataset
from torch.utils.data import DataLoader
from sklearn.metrics import roc_curve, auc

def plot_roc_curves():
    print("Generating ROC Plots...")
    
    # 1. Load Data
    _, X_clin_test, y_test, scaler = load_adni_level_max(Config.TRAIN_CSV, is_train=True) # Get scaler from train
    X_mri_test, X_clin_test, y_test, _ = load_adni_level_max(Config.TEST_CSV, scaler=scaler, is_train=False) # Use scaler on test
    
    test_ds = ADNIDataset(X_mri_test, X_clin_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)
    
    # 2. Load Models
    models = {
        "MRI-Only": MRIOnlyModel(),
        "Late Fusion (Level-MAX)": LateFusionModel(),
        # We can skip attention fusion for plot clarity as it overlaps Late Fusion
    }
    
    plt.figure(figsize=(10, 8))
    
    # Colors suitable for publication
    colors = {"MRI-Only": "#1f77b4", "Late Fusion (Level-MAX)": "#d62728"}
    
    for name, model in models.items():
        print(f"Evaluating {name}...")
        path = os.path.join(Config.RESULTS_DIR, f"best_{name.split(' ')[0].replace('-','_')}.pth")
        if "Late" in name: path = os.path.join(Config.RESULTS_DIR, "best_Late_Fusion.pth")
        
        if not os.path.exists(path):
            print(f"Warning: {path} not found.")
            continue
            
        model.load_state_dict(torch.load(path))
        model.to(Config.DEVICE)
        model.eval()
        
        all_probs = []
        all_labels = []
        
        with torch.no_grad():
            for batch in test_loader:
                mri = batch['mri'].to(Config.DEVICE)
                clin = batch['clinical'].to(Config.DEVICE)
                labels = batch['label'].to(Config.DEVICE)
                
                outputs = model(mri, clin)
                if isinstance(outputs, dict): outputs = outputs['logits']
                probs = torch.softmax(outputs, dim=1)[:, 1]
                
                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        fpr, tpr, _ = roc_curve(all_labels, all_probs)
        roc_auc = auc(fpr, tpr)
        
        plt.plot(fpr, tpr, color=colors[name], lw=2, label=f'{name} (AUC = {roc_auc:.2f})')
        
    # Plot Level-1 Baseline (Approximate line for reference if valid file not found, or could load it)
    # Since we can't easily load the Level-1 weights (different script), let's just note comparison in text or add a dummy line.
    # Actually, we can just plot the ones we have.
    
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curves: Level-MAX Honest Fusion', fontsize=16)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(alpha=0.3)
    
    output_path = os.path.join(Config.RESULTS_DIR, "roc_comparison.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved ROC plot to {output_path}")

def plot_performance_bar():
    print("Generating Bar Chart...")
    
    # Hardcoded values for comparison across levels
    levels = ['Level-1 (Age/Sex)', 'Level-MAX (Bio-Fusion)', 'Level-2 (Circular)']
    aucs = [0.60, 0.81, 0.99]
    colors = ['gray', '#d62728', 'green']
    
    plt.figure(figsize=(8, 6))
    bars = plt.bar(levels, aucs, color=colors, width=0.6)
    
    plt.ylim(0, 1.1)
    plt.ylabel('AUC Score', fontsize=12)
    plt.title('Performance Ceiling Analysis', fontsize=16)
    
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{height:.2f}',
                 ha='center', va='bottom', fontsize=12, fontweight='bold')
                 
    output_path = os.path.join(Config.RESULTS_DIR, "level_comparison.png")
    plt.savefig(output_path, dpi=300)
    print(f"Saved Bar chart to {output_path}")

if __name__ == "__main__":
    plot_roc_curves()
    plot_performance_bar()
