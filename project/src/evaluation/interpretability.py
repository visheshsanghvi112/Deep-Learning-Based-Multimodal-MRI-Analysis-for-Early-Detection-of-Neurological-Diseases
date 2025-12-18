"""
================================================================================
Model Interpretability Analysis
================================================================================
- Feature importance analysis
- Attention weight visualization
- Saliency maps (for MRI)
- SHAP values
- Embedding visualization

Part of: Master Research Plan - Phase 5
================================================================================
"""

import torch
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("WARNING: SHAP not available. Install with: pip install shap")


class InterpretabilityAnalyzer:
    """Analyze model interpretability"""
    
    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.model.eval()
    
    def analyze_attention_weights(self, attention_weights: np.ndarray, 
                                 output_path: Path):
        """Visualize attention weights"""
        # Average across samples
        avg_attention = attention_weights.mean(axis=0)  # (3, 3)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(avg_attention, 
                   annot=True, 
                   fmt='.3f',
                   cmap='viridis',
                   xticklabels=['MRI', 'Anatomical', 'Clinical'],
                   yticklabels=['MRI', 'Anatomical', 'Clinical'],
                   ax=ax)
        ax.set_title('Average Attention Weights Across Modalities')
        ax.set_ylabel('Query Modality')
        ax.set_xlabel('Key Modality')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Attention weights visualization saved to: {output_path}")
    
    def visualize_embeddings(self, embeddings: np.ndarray,
                            labels: np.ndarray,
                            output_path: Path,
                            method: str = 'tsne'):
        """Visualize learned embeddings"""
        print(f"Visualizing embeddings using {method.upper()}...")
        
        if method == 'tsne':
            reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        else:
            reducer = PCA(n_components=2)
        
        embeddings_2d = reducer.fit_transform(embeddings)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(embeddings_2d[:, 0], 
                           embeddings_2d[:, 1],
                           c=labels,
                           cmap='viridis',
                           alpha=0.6,
                           s=50)
        ax.set_xlabel(f'{method.upper()} Component 1')
        ax.set_ylabel(f'{method.upper()} Component 2')
        ax.set_title(f'Learned Embeddings Visualization ({method.upper()})')
        plt.colorbar(scatter, ax=ax, label='Label')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Embeddings visualization saved to: {output_path}")
    
    def compute_feature_importance_shap(self, 
                                       data_loader,
                                       n_samples: int = 100):
        """Compute SHAP values for feature importance"""
        if not SHAP_AVAILABLE:
            print("SHAP not available, skipping feature importance analysis")
            return None
        
        # Get sample data
        batch = next(iter(data_loader))
        anatomical = batch['anatomical'][:n_samples].to(self.device)
        clinical = batch['clinical'][:n_samples].to(self.device)
        cnn_emb = batch['cnn_embeddings'][:n_samples].to(self.device) if batch['cnn_embeddings'] is not None else None
        
        # Create SHAP explainer
        def model_wrapper(anatomical_data):
            """Wrapper for SHAP"""
            with torch.no_grad():
                predictions = self.model(
                    anatomical=torch.FloatTensor(anatomical_data).to(self.device),
                    clinical=clinical,
                    cnn_embeddings=cnn_emb
                )
                return predictions['binary'].cpu().numpy()
        
        explainer = shap.Explainer(model_wrapper, anatomical.cpu().numpy())
        shap_values = explainer(anatomical.cpu().numpy())
        
        return shap_values
    
    def plot_feature_importance(self, importance_scores: dict, 
                               output_path: Path,
                               top_n: int = 20):
        """Plot feature importance"""
        # Sort by importance
        sorted_features = sorted(importance_scores.items(), 
                               key=lambda x: abs(x[1]), 
                               reverse=True)[:top_n]
        
        features, scores = zip(*sorted_features)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        y_pos = np.arange(len(features))
        ax.barh(y_pos, scores)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(features)
        ax.set_xlabel('Importance Score')
        ax.set_title(f'Top {top_n} Feature Importance')
        ax.invert_yaxis()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Feature importance plot saved to: {output_path}")


def main():
    """Example usage"""
    print("Interpretability analysis tools ready!")
    print("Use after model training to analyze interpretability")


if __name__ == "__main__":
    main()

