"""
Multimodal Transformer Fusion Model
====================================
State-of-the-art architecture for MRI + Biomarker fusion.

Features:
- Temporal attention for longitudinal features
- Cross-modal attention for MRI-biomarker interaction
- Gated fusion for dynamic modality weighting
- Regularization: Dropout, LayerNorm, weight decay
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple

from .attention import (
    MultiHeadSelfAttention,
    CrossModalAttention,
    GatedFusion,
    TemporalAttention
)


class MRIEncoder(nn.Module):
    """
    Encoder for longitudinal MRI (ResNet) features.
    
    Processes baseline, followup, and delta CNN features
    with temporal attention.
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.3
    ):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Project each timepoint
        self.baseline_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.followup_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.delta_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(
            feature_dim=hidden_dim,
            num_timepoints=3,
            dropout=dropout
        )
        
        # Final projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        
    def forward(
        self,
        baseline: torch.Tensor,
        followup: torch.Tensor,
        delta: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            baseline: [batch, 512] baseline ResNet features
            followup: [batch, 512] followup ResNet features
            delta: [batch, 512] delta ResNet features
            
        Returns:
            output: [batch, output_dim]
            temporal_weights: Optional [batch, 3] attention weights
        """
        # Project each timepoint
        h_baseline = self.baseline_proj(baseline)
        h_followup = self.followup_proj(followup)
        h_delta = self.delta_proj(delta)
        
        # Stack for temporal attention
        temporal = torch.stack([h_baseline, h_followup, h_delta], dim=1)
        
        # Apply temporal attention
        aggregated, weights = self.temporal_attention(
            temporal, return_weights=return_attention
        )
        
        # Final projection
        output = self.output_proj(aggregated)
        
        return output, weights


class BiomarkerEncoder(nn.Module):
    """
    Encoder for longitudinal biomarker features.
    
    Processes baseline (9-dim), followup (6-dim), and delta (6-dim)
    clinical biomarkers.
    """
    
    def __init__(
        self,
        baseline_dim: int = 9,
        followup_dim: int = 6,
        delta_dim: int = 6,
        hidden_dim: int = 64,
        output_dim: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        
        # Shared hidden dim for all biomarker types
        self.baseline_proj = nn.Sequential(
            nn.Linear(baseline_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.followup_proj = nn.Sequential(
            nn.Linear(followup_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        self.delta_proj = nn.Sequential(
            nn.Linear(delta_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Temporal attention
        self.temporal_attention = TemporalAttention(
            feature_dim=hidden_dim,
            num_timepoints=3,
            dropout=dropout
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU()
        )
        
    def forward(
        self,
        baseline: torch.Tensor,
        followup: torch.Tensor,
        delta: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            baseline: [batch, 9] baseline biomarkers
            followup: [batch, 6] followup biomarkers
            delta: [batch, 6] delta biomarkers
            
        Returns:
            output: [batch, output_dim]
            temporal_weights: Optional attention weights
        """
        # Project each timepoint
        h_baseline = self.baseline_proj(baseline)
        h_followup = self.followup_proj(followup)
        h_delta = self.delta_proj(delta)
        
        # Stack for temporal attention
        temporal = torch.stack([h_baseline, h_followup, h_delta], dim=1)
        
        # Apply temporal attention
        aggregated, weights = self.temporal_attention(
            temporal, return_weights=return_attention
        )
        
        # Final projection
        output = self.output_proj(aggregated)
        
        return output, weights


class MultimodalTransformerFusion(nn.Module):
    """
    Main Multimodal Fusion Model.
    
    Architecture:
    1. MRI Encoder (processes longitudinal CNN features)
    2. Biomarker Encoder (processes longitudinal clinical features)
    3. Cross-Modal Attention (learns MRI-biomarker interactions)
    4. Gated Fusion (dynamically weights modalities)
    5. Classifier (binary: Stable vs Converter)
    
    Total parameters: ~300K (efficient for ~300 samples)
    """
    
    def __init__(
        self,
        resnet_dim: int = 512,
        baseline_bio_dim: int = 9,
        followup_bio_dim: int = 6,
        delta_bio_dim: int = 6,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.4,
        num_classes: int = 2
    ):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Encoders
        self.mri_encoder = MRIEncoder(
            input_dim=resnet_dim,
            hidden_dim=hidden_dim * 2,
            output_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.bio_encoder = BiomarkerEncoder(
            baseline_dim=baseline_bio_dim,
            followup_dim=followup_bio_dim,
            delta_dim=delta_bio_dim,
            hidden_dim=hidden_dim // 2,
            output_dim=hidden_dim // 2,
            dropout=dropout
        )
        
        # Cross-modal attention
        self.cross_attention = CrossModalAttention(
            query_dim=hidden_dim // 2,  # biomarker as query
            key_dim=hidden_dim,         # MRI as key/value
            hidden_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Gated fusion
        self.gated_fusion = GatedFusion(
            mri_dim=hidden_dim,
            bio_dim=hidden_dim,
            output_dim=hidden_dim,
            dropout=dropout
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes)
        )
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
                
    def forward(
        self,
        baseline_resnet: torch.Tensor,
        followup_resnet: torch.Tensor,
        delta_resnet: torch.Tensor,
        baseline_bio: torch.Tensor,
        followup_bio: torch.Tensor,
        delta_bio: torch.Tensor,
        return_attention: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            baseline_resnet: [batch, 512]
            followup_resnet: [batch, 512]
            delta_resnet: [batch, 512]
            baseline_bio: [batch, 9]
            followup_bio: [batch, 6]
            delta_bio: [batch, 6]
            
        Returns:
            Dictionary with:
            - logits: [batch, 2]
            - probabilities: [batch, 2]
            - mri_features: [batch, hidden_dim]
            - bio_features: [batch, hidden_dim//2]
            - attention weights (if requested)
        """
        # Encode MRI features
        mri_features, mri_temporal_attn = self.mri_encoder(
            baseline_resnet, followup_resnet, delta_resnet,
            return_attention=return_attention
        )
        
        # Encode biomarker features
        bio_features, bio_temporal_attn = self.bio_encoder(
            baseline_bio, followup_bio, delta_bio,
            return_attention=return_attention
        )
        
        # Cross-modal attention (biomarkers attend to MRI)
        cross_features, cross_attn = self.cross_attention(
            bio_features, mri_features,
            return_attention=return_attention
        )
        
        # Gated fusion
        fused, gates = self.gated_fusion(
            mri_features, cross_features,
            return_gates=return_attention
        )
        
        # Classification
        logits = self.classifier(fused)
        probs = F.softmax(logits, dim=1)
        
        output = {
            'logits': logits,
            'probabilities': probs,
            'mri_features': mri_features,
            'bio_features': bio_features,
            'fused_features': fused
        }
        
        if return_attention:
            output['mri_temporal_attention'] = mri_temporal_attn
            output['bio_temporal_attention'] = bio_temporal_attn
            output['cross_attention'] = cross_attn
            output['fusion_gates'] = gates
            
        return output
    
    def predict(self, **kwargs) -> torch.Tensor:
        """Get class predictions."""
        output = self.forward(**kwargs)
        return output['probabilities'][:, 1]  # Probability of conversion
    
    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


class SimpleMLPFusion(nn.Module):
    """
    Simple MLP baseline for comparison.
    Concatenates all features and uses an MLP for classification.
    """
    
    def __init__(
        self,
        resnet_dim: int = 512,
        bio_dim: int = 21,
        hidden_dims: Tuple[int, ...] = (256, 128, 64),
        dropout: float = 0.4,
        num_classes: int = 2
    ):
        super().__init__()
        
        input_dim = resnet_dim * 3 + bio_dim  # All ResNet + all biomarkers
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, num_classes))
        
        self.network = nn.Sequential(*layers)
        
    def forward(
        self,
        baseline_resnet: torch.Tensor,
        followup_resnet: torch.Tensor,
        delta_resnet: torch.Tensor,
        baseline_bio: torch.Tensor,
        followup_bio: torch.Tensor,
        delta_bio: torch.Tensor,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """Forward pass."""
        # Concatenate all features
        all_resnet = torch.cat([baseline_resnet, followup_resnet, delta_resnet], dim=1)
        all_bio = torch.cat([baseline_bio, followup_bio, delta_bio], dim=1)
        combined = torch.cat([all_resnet, all_bio], dim=1)
        
        logits = self.network(combined)
        probs = F.softmax(logits, dim=1)
        
        return {
            'logits': logits,
            'probabilities': probs
        }


def get_model(
    model_type: str = 'transformer',
    **kwargs
) -> nn.Module:
    """
    Factory function to create models.
    
    Args:
        model_type: 'transformer' or 'mlp'
        **kwargs: Model-specific arguments
        
    Returns:
        PyTorch model
    """
    if model_type == 'transformer':
        return MultimodalTransformerFusion(**kwargs)
    elif model_type == 'mlp':
        return SimpleMLPFusion(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
