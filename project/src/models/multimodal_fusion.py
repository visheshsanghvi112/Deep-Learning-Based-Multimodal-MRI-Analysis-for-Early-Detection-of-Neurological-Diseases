"""
================================================================================
Hybrid Multimodal Fusion Model Architecture
================================================================================
Phase 3: Deep Learning Model Development

Architecture:
- MRI Branch: 3D CNN encoder
- Anatomical Branch: MLP encoder
- Clinical Branch: MLP encoder
- Fusion: Attention-based multimodal fusion
- Output: Multi-task heads (CDR, MMSE, Diagnosis, Binary)

Part of: Master Research Plan - Phase 3
================================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import numpy as np


class MRIEncoder(nn.Module):
    """
    3D CNN Encoder for MRI volumes
    Input: 3D MRI volume (176×208×176 or similar)
    Output: 512-dim embedding
    """
    
    def __init__(self, input_shape: Tuple[int, int, int] = (176, 208, 176), 
                 embedding_dim: int = 512):
        super(MRIEncoder, self).__init__()
        
        self.input_shape = input_shape
        self.embedding_dim = embedding_dim
        
        # 3D Convolutional layers
        self.conv1 = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2, stride=2)
        )
        
        self.conv4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool3d((4, 4, 4))
        )
        
        # Calculate flattened size
        # After 3 max pools: 176/8 × 208/8 × 176/8 = 22 × 26 × 22
        # After adaptive pool: 4 × 4 × 4 = 64
        # 256 channels × 64 = 16384
        self.fc = nn.Sequential(
            nn.Linear(256 * 4 * 4 * 4, 1024),
            nn.LayerNorm(1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, 1, H, W, D) - 3D MRI volume
        Returns:
            embedding: (batch_size, embedding_dim)
        """
        # Add channel dimension if needed
        if x.dim() == 4:
            x = x.unsqueeze(1)  # (B, 1, H, W, D)
        
        # 3D CNN
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        
        # Flatten
        x = x.view(x.size(0), -1)
        
        # Fully connected
        x = self.fc(x)
        
        return x


class AnatomicalEncoder(nn.Module):
    """
    MLP Encoder for Anatomical Features
    Input: Selected anatomical features (50-200 features)
    Output: 128-dim embedding
    """
    
    def __init__(self, input_dim: int, embedding_dim: int = 128):
        super(AnatomicalEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim) - anatomical features
        Returns:
            embedding: (batch_size, embedding_dim)
        """
        return self.encoder(x)


class ClinicalEncoder(nn.Module):
    """
    MLP Encoder for Clinical Features
    Input: Clinical features (Age, Gender, Education, etc.)
    Output: 64-dim embedding
    """
    
    def __init__(self, input_dim: int, embedding_dim: int = 64):
        super(ClinicalEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(64, embedding_dim),
            nn.LayerNorm(embedding_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, input_dim) - clinical features
        Returns:
            embedding: (batch_size, embedding_dim)
        """
        return self.encoder(x)


class AttentionFusion(nn.Module):
    """
    Attention-based Multimodal Fusion
    Uses multi-head self-attention to fuse MRI, Anatomical, and Clinical embeddings
    """
    
    def __init__(self, 
                 mri_dim: int = 512,
                 anatomical_dim: int = 128,
                 clinical_dim: int = 64,
                 hidden_dim: int = 256,
                 num_heads: int = 8):
        super(AttentionFusion, self).__init__()
        
        self.mri_dim = mri_dim
        self.anatomical_dim = anatomical_dim
        self.clinical_dim = clinical_dim
        self.hidden_dim = hidden_dim
        
        # Project all modalities to same dimension
        self.mri_proj = nn.Linear(mri_dim, hidden_dim)
        self.anatomical_proj = nn.Linear(anatomical_dim, hidden_dim)
        self.clinical_proj = nn.Linear(clinical_dim, hidden_dim)
        
        # Multi-head self-attention
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feed-forward network
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Dropout(0.1)
        )
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, hidden_dim)
    
    def forward(self, mri_emb: torch.Tensor, 
                anatomical_emb: torch.Tensor,
                clinical_emb: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            mri_emb: (batch_size, mri_dim)
            anatomical_emb: (batch_size, anatomical_dim)
            clinical_emb: (batch_size, clinical_dim)
        Returns:
            fused_emb: (batch_size, hidden_dim)
            attention_weights: (batch_size, 3, 3) - attention weights for interpretability
        """
        batch_size = mri_emb.size(0)
        
        # Project to same dimension
        mri_proj = self.mri_proj(mri_emb).unsqueeze(1)  # (B, 1, hidden_dim)
        anatomical_proj = self.anatomical_proj(anatomical_emb).unsqueeze(1)  # (B, 1, hidden_dim)
        clinical_proj = self.clinical_proj(clinical_emb).unsqueeze(1)  # (B, 1, hidden_dim)
        
        # Concatenate modalities
        modalities = torch.cat([mri_proj, anatomical_proj, clinical_proj], dim=1)  # (B, 3, hidden_dim)
        
        # Self-attention
        attn_out, attn_weights = self.attention(modalities, modalities, modalities)
        
        # Residual connection and normalization
        out = self.norm1(modalities + attn_out)
        
        # Feed-forward
        ffn_out = self.ffn(out)
        out = self.norm2(out + ffn_out)
        
        # Global average pooling across modalities
        fused_emb = out.mean(dim=1)  # (B, hidden_dim)
        
        # Final projection
        fused_emb = self.output_proj(fused_emb)
        
        return fused_emb, attn_weights


class MultiTaskHead(nn.Module):
    """
    Multi-task prediction heads
    - CDR regression
    - MMSE regression
    - Diagnosis classification (3-class: CN/MCI/AD)
    - Binary classification (Normal vs Impaired)
    """
    
    def __init__(self, input_dim: int = 256):
        super(MultiTaskHead, self).__init__()
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.LayerNorm(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(inplace=True)
        )
        
        # Task-specific heads
        self.cdr_head = nn.Linear(64, 1)  # Regression
        self.mmse_head = nn.Linear(64, 1)  # Regression
        self.diagnosis_head = nn.Linear(64, 3)  # Classification (CN/MCI/AD)
        self.binary_head = nn.Linear(64, 2)  # Binary classification
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (batch_size, input_dim) - fused embedding
        Returns:
            Dictionary with predictions for each task
        """
        shared = self.shared(x)
        
        return {
            'cdr': self.cdr_head(shared),
            'mmse': self.mmse_head(shared),
            'diagnosis': self.diagnosis_head(shared),
            'binary': self.binary_head(shared)
        }


class HybridMultimodalModel(nn.Module):
    """
    Complete Hybrid Multimodal Fusion Model
    Combines MRI, Anatomical, and Clinical features
    """
    
    def __init__(self,
                 mri_shape: Tuple[int, int, int] = (176, 208, 176),
                 anatomical_dim: int = 50,
                 clinical_dim: int = 6,
                 mri_embedding_dim: int = 512,
                 anatomical_embedding_dim: int = 128,
                 clinical_embedding_dim: int = 64,
                 fusion_dim: int = 256,
                 num_attention_heads: int = 8):
        super(HybridMultimodalModel, self).__init__()
        
        # Encoders
        self.mri_encoder = MRIEncoder(mri_shape, mri_embedding_dim)
        self.anatomical_encoder = AnatomicalEncoder(anatomical_dim, anatomical_embedding_dim)
        self.clinical_encoder = ClinicalEncoder(clinical_dim, clinical_embedding_dim)
        
        # Fusion
        self.fusion = AttentionFusion(
            mri_dim=mri_embedding_dim,
            anatomical_dim=anatomical_embedding_dim,
            clinical_dim=clinical_embedding_dim,
            hidden_dim=fusion_dim,
            num_heads=num_attention_heads
        )
        
        # Multi-task heads
        self.task_heads = MultiTaskHead(fusion_dim)
    
    def forward(self, 
                mri: Optional[torch.Tensor] = None,
                anatomical: Optional[torch.Tensor] = None,
                clinical: Optional[torch.Tensor] = None,
                cnn_embeddings: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass
        
        Args:
            mri: (batch_size, H, W, D) - 3D MRI volume (optional)
            anatomical: (batch_size, anatomical_dim) - anatomical features
            clinical: (batch_size, clinical_dim) - clinical features
            cnn_embeddings: (batch_size, 512) - pre-extracted CNN embeddings (optional, replaces MRI)
        Returns:
            Dictionary with predictions for all tasks
        """
        # Encode modalities
        if cnn_embeddings is not None:
            # Use pre-extracted CNN embeddings
            mri_emb = cnn_embeddings
        elif mri is not None:
            # Extract from 3D volume
            mri_emb = self.mri_encoder(mri)
        else:
            # Use zero embedding if MRI not available
            batch_size = anatomical.size(0) if anatomical is not None else clinical.size(0)
            mri_emb = torch.zeros(batch_size, 512, device=anatomical.device if anatomical is not None else clinical.device)
        
        if anatomical is not None:
            anatomical_emb = self.anatomical_encoder(anatomical)
        else:
            batch_size = mri_emb.size(0)
            anatomical_emb = torch.zeros(batch_size, 128, device=mri_emb.device)
        
        if clinical is not None:
            clinical_emb = self.clinical_encoder(clinical)
        else:
            batch_size = mri_emb.size(0)
            clinical_emb = torch.zeros(batch_size, 64, device=mri_emb.device)
        
        # Fuse modalities
        fused_emb, attention_weights = self.fusion(mri_emb, anatomical_emb, clinical_emb)
        
        # Multi-task predictions
        predictions = self.task_heads(fused_emb)
        
        # Add attention weights for interpretability
        predictions['attention_weights'] = attention_weights
        
        return predictions
    
    def get_embeddings(self, 
                      mri: Optional[torch.Tensor] = None,
                      anatomical: Optional[torch.Tensor] = None,
                      clinical: Optional[torch.Tensor] = None,
                      cnn_embeddings: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Get intermediate embeddings for analysis/visualization
        """
        # Encode modalities
        if cnn_embeddings is not None:
            mri_emb = cnn_embeddings
        elif mri is not None:
            mri_emb = self.mri_encoder(mri)
        else:
            batch_size = anatomical.size(0) if anatomical is not None else clinical.size(0)
            mri_emb = torch.zeros(batch_size, 512, device=anatomical.device if anatomical is not None else clinical.device)
        
        if anatomical is not None:
            anatomical_emb = self.anatomical_encoder(anatomical)
        else:
            batch_size = mri_emb.size(0)
            anatomical_emb = torch.zeros(batch_size, 128, device=mri_emb.device)
        
        if clinical is not None:
            clinical_emb = self.clinical_encoder(clinical)
        else:
            batch_size = mri_emb.size(0)
            clinical_emb = torch.zeros(batch_size, 64, device=mri_emb.device)
        
        # Fuse
        fused_emb, attention_weights = self.fusion(mri_emb, anatomical_emb, clinical_emb)
        
        return {
            'mri_embedding': mri_emb,
            'anatomical_embedding': anatomical_emb,
            'clinical_embedding': clinical_emb,
            'fused_embedding': fused_emb,
            'attention_weights': attention_weights
        }


def create_model(config: Dict) -> HybridMultimodalModel:
    """
    Factory function to create model from config
    """
    return HybridMultimodalModel(
        mri_shape=config.get('mri_shape', (176, 208, 176)),
        anatomical_dim=config.get('anatomical_dim', 50),
        clinical_dim=config.get('clinical_dim', 6),
        mri_embedding_dim=config.get('mri_embedding_dim', 512),
        anatomical_embedding_dim=config.get('anatomical_embedding_dim', 128),
        clinical_embedding_dim=config.get('clinical_embedding_dim', 64),
        fusion_dim=config.get('fusion_dim', 256),
        num_attention_heads=config.get('num_attention_heads', 8)
    )


if __name__ == "__main__":
    # Test model
    print("Testing Hybrid Multimodal Model...")
    
    # Create model
    model = HybridMultimodalModel(
        anatomical_dim=50,
        clinical_dim=6
    )
    
    # Test forward pass
    batch_size = 4
    anatomical = torch.randn(batch_size, 50)
    clinical = torch.randn(batch_size, 6)
    cnn_emb = torch.randn(batch_size, 512)  # Pre-extracted CNN embeddings
    
    predictions = model(anatomical=anatomical, clinical=clinical, cnn_embeddings=cnn_emb)
    
    print(f"\nModel created successfully!")
    print(f"Input shapes:")
    print(f"  Anatomical: {anatomical.shape}")
    print(f"  Clinical: {clinical.shape}")
    print(f"  CNN embeddings: {cnn_emb.shape}")
    print(f"\nOutput shapes:")
    for task, pred in predictions.items():
        if task != 'attention_weights':
            print(f"  {task}: {pred.shape}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

