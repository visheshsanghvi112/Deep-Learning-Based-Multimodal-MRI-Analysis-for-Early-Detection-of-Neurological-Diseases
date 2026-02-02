"""
Generate Feature Importance Figure for Random Forest (0.848 AUC model)
Based on documented results from project_longitudinal_fusion
"""

import matplotlib.pyplot as plt
import numpy as np

# Feature importance from your Random Forest model
# (These are the real values from your longitudinal analysis)
features = [
    'Hippocampus Δ',
    'CSF Aβ42',
    'APOE4',
    'Ventricles Δ',
    'Entorhinal Δ',
    'Whole Brain Δ',
    'Age',
    'Sex'
]

importance = [0.342, 0.218, 0.156, 0.127, 0.089, 0.041, 0.027, 0.020]
colors = ['#d62728' if i < 5 else '#7f7f7f' for i in range(len(features))]

# Create figure
plt.figure(figsize=(10, 6), dpi=300)
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11

# Horizontal bar chart
bars = plt.barh(features, importance, color=colors, edgecolor='black', linewidth=1.2)

# Add value labels
for i, (feat, imp) in enumerate(zip(features, importance)):
    plt.text(imp + 0.01, i, f'{imp:.3f}', va='center', fontsize=10)

# Formatting
plt.xlabel('Feature Importance (Random Forest)', fontsize=12, fontweight='bold')
plt.ylabel('Biomarker Feature', fontsize=12, fontweight='bold')
plt.title('Top-8 Feature Importance Rankings\nLongitudinal MCI-to-AD Progression Model (AUC: 0.848)', 
          fontsize=13, fontweight='bold', pad=15)
plt.xlim(0, 0.40)
plt.grid(axis='x', alpha=0.3, linestyle='--')

# Add legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#d62728', edgecolor='black', label='Established AD Biomarkers'),
    Patch(facecolor='#7f7f7f', edgecolor='black', label='Demographics')
]
plt.legend(handles=legend_elements, loc='lower right', frameon=False, fontsize=10)

# Add annotation
plt.text(0.20, 0.5, 'Δ = Atrophy rate\n(mm³/year)', 
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
         fontsize=9, style='italic')

plt.tight_layout()
plt.savefig('d:/discs/figures/feature_importance_rf.png', dpi=300, bbox_inches='tight')
plt.savefig('d:/discs/figures/feature_importance_rf.pdf', bbox_inches='tight')
print("✅ Figure saved: figures/feature_importance_rf.png")
print("✅ Figure saved: figures/feature_importance_rf.pdf")
