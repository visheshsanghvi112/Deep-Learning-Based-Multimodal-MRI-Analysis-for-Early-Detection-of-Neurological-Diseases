import json
import matplotlib.pyplot as plt
import numpy as np
import os
import seaborn as sns

# Data directly from the user's provided JSON content
data = {
    "MRI_Only": {
        "AUC": 0.6430887120542292,
        "Accuracy": 0.626984126984127,
        "AUC_CI_Lower": 0.5270525579249721,
        "AUC_CI_Upper": 0.7261483174515854
    },
    "Late_Fusion": {
        "AUC": 0.8078396699086354,
        "Accuracy": 0.746031746031746,
        "AUC_CI_Lower": 0.7454755129799782,
        "AUC_CI_Upper": 0.8743300480319381
    },
    "Attention_Fusion": {
        "AUC": 0.808134394341291,
        "Accuracy": 0.7619047619047619,
        "AUC_CI_Lower": 0.7371652215615658,
        "AUC_CI_Upper": 0.8834693918265154
    }
}

# Update output directory to the main figures folder
output_dir = r"D:\discs\figures"
os.makedirs(output_dir, exist_ok=True)

# Set style to match existing figures
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 11

colors = ['#E74C3C', '#3498DB', '#2ECC71'] # Red, Blue, Green

def save_plot(filename):
    """Save plot in both PNG and PDF formats to match existing folder structure."""
    # Save PNG
    png_path = os.path.join(output_dir, f"{filename}.png")
    plt.savefig(png_path, dpi=300, bbox_inches='tight')
    print(f"Saved {png_path}")
    
    # Save PDF
    pdf_path = os.path.join(output_dir, f"{filename}.pdf")
    plt.savefig(pdf_path, bbox_inches='tight')
    print(f"Saved {pdf_path}")

def create_auc_comparison():
    models = list(data.keys())
    aucs = [data[m]['AUC'] for m in models]
    lower_errors = [data[m]['AUC'] - data[m]['AUC_CI_Lower'] for m in models]
    upper_errors = [data[m]['AUC_CI_Upper'] - data[m]['AUC'] for m in models]
    errors = [lower_errors, upper_errors]
    
    # Renaming for display
    display_names = [m.replace('_', ' ') for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(display_names, aucs, yerr=errors, capsize=10, 
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                f'{height:.3f}',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
        
    ax.set_ylim(0.5, 0.95)
    ax.set_ylabel('AUC Score', fontsize=12)
    ax.set_title('E1: Level Max - AUC Performance Comparison', pad=20, fontweight='bold', fontsize=14)
    
    # Add a line for chance
    plt.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Chance (0.5)')
    ax.legend(loc='upper left')
    
    # Annotation for improvement
    ax.annotate('', xy=(2, 0.808), xytext=(0, 0.643),
                arrowprops=dict(arrowstyle='->', color='green', lw=2))
    ax.text(1.0, 0.72, '+16.5% Improvement', fontsize=12, color='green', fontweight='bold', ha='center')

    plt.tight_layout()
    save_plot('E1_level_max_auc_comparison')
    plt.close()

def create_accuracy_comparison():
    models = list(data.keys())
    accs = [data[m]['Accuracy'] for m in models]
    
    # Renaming for display
    display_names = [m.replace('_', ' ') for m in models]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(display_names, accs, 
                  color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.1%}',
                ha='center', va='bottom', fontweight='bold', fontsize=12)
        
    ax.set_ylim(0.5, 0.9)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('E2: Level Max - Accuracy Comparison', pad=20, fontweight='bold', fontsize=14)
    
    plt.tight_layout()
    save_plot('E2_level_max_accuracy_comparison')
    plt.close()

def create_combined_summary():
    models = list(data.keys())
    aucs = [data[m]['AUC'] for m in models]
    accs = [data[m]['Accuracy'] for m in models]
    
    x = np.arange(len(models))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 6))
    rects1 = ax.bar(x - width/2, aucs, width, label='AUC', color='#3498DB', alpha=0.9, edgecolor='black')
    rects2 = ax.bar(x + width/2, accs, width, label='Accuracy', color='#2ECC71', alpha=0.9, edgecolor='black')
    
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('E3: Level Max - Complete Performance Summary', pad=20, fontweight='bold', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', ' ') for m in models], fontweight='bold', fontsize=11)
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc='upper left', frameon=True, framealpha=0.9, fontsize=10)
    
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height:.3f}',
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1)
    autolabel(rects2)
    
    # Highlight the winner
    ax.text(2, 0.92, '🏆 Best Model', ha='center', fontsize=12, color='#D35400', fontweight='bold')
    
    plt.tight_layout()
    save_plot('E3_level_max_summary')
    plt.close()

if __name__ == "__main__":
    create_auc_comparison()
    create_accuracy_comparison()
    create_combined_summary()
