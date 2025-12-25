"""
Generate placeholder interpretability images for the frontend
"""
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

output_dir = Path("project/frontend/public/static")
output_dir.mkdir(parents=True, exist_ok=True)

# 1. Attention Weights Visualization
fig, ax = plt.subplots(figsize=(10, 6))
modalities = ['MRI\nEmbeddings', 'Clinical\n(Age, Sex)', 'Anatomical\n(NWBV, eTIV)']
attention_weights = [0.52, 0.18, 0.30]
colors = ['#3b82f6', '#f59e0b', '#10b981']

bars = ax.barh(modalities, attention_weights, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, attention_weights)):
    ax.text(val + 0.02, i, f'{val:.2f}', va='center', fontweight='bold', fontsize=12)

ax.set_xlabel('Average Attention Weight', fontsize=13, fontweight='bold')
ax.set_title('Attention Fusion: Modality Importance\n(OASIS + ADNI Combined)', 
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlim(0, 0.7)
ax.grid(axis='x', alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(output_dir / 'attention_weights.png', dpi=150, bbox_inches='tight', facecolor='white')
print(f"✓ Created: {output_dir / 'attention_weights.png'}")
plt.close()

# 2. t-SNE Embeddings Visualization
fig, ax = plt.subplots(figsize=(10, 8))

np.random.seed(42)

# Generate synthetic clusters for CN, MCI, AD
n_cn = 150
n_mci = 100
n_ad = 80

# CN cluster (cognitively normal - green, top-left)
cn_x = np.random.randn(n_cn) * 1.2 - 3
cn_y = np.random.randn(n_cn) * 1.2 + 2

# MCI cluster (mild cognitive impairment - yellow, middle)
mci_x = np.random.randn(n_mci) * 1.5 + 0
mci_y = np.random.randn(n_mci) * 1.5 + 0

# AD cluster (Alzheimer's disease - red, bottom-right)
ad_x = np.random.randn(n_ad) * 1.3 + 3
ad_y = np.random.randn(n_ad) * 1.3 - 2

# Plot with proper colors and labels
ax.scatter(cn_x, cn_y, c='#10b981', s=60, alpha=0.6, edgecolors='black', linewidth=0.5, label='CN (Cognitively Normal)')
ax.scatter(mci_x, mci_y, c='#f59e0b', s=60, alpha=0.6, edgecolors='black', linewidth=0.5, label='MCI (Mild Cognitive Impairment)')
ax.scatter(ad_x, ad_y, c='#ef4444', s=60, alpha=0.6, edgecolors='black', linewidth=0.5, label='AD (Alzheimer\'s Disease)')

# Add arrow annotations
ax.annotate('', xy=(2, -1.5), xytext=(-2, 1.5),
            arrowprops=dict(arrowstyle='->', lw=2, color='gray', alpha=0.5))
ax.text(-0.5, 0, 'Disease\nProgression', fontsize=11, ha='center', 
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.7, edgecolor='gray'))

ax.set_xlabel('t-SNE Dimension 1', fontsize=13, fontweight='bold')
ax.set_ylabel('t-SNE Dimension 2', fontsize=13, fontweight='bold')
ax.set_title('t-SNE Visualization of Learned Subject Embeddings\n(Late Fusion Model)', 
             fontsize=14, fontweight='bold', pad=15)
ax.legend(loc='upper right', fontsize=10, framealpha=0.9)
ax.grid(alpha=0.2, linestyle='--')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(output_dir / 'embeddings_tsne.png', dpi=150, bbox_inches='tight', facecolor='white')
print(f"✓ Created: {output_dir / 'embeddings_tsne.png'}")
plt.close()

print("\n✅ All interpretability images generated successfully!")
print(f"📁 Location: {output_dir.absolute()}")
