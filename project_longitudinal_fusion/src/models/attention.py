"""
Attention Mechanisms for Multimodal Fusion
==========================================
Publication-quality attention implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class MultiHeadSelfAttention(nn.Module):
    """
    Multi-Head Self-Attention mechanism.
    
    Allows the model to attend to different positions in the sequence
    and learn different representation subspaces.
    """
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        dropout: float = 0.1,
        bias: bool = True
    ):
        super().__init__()
        
        assert embed_dim % num_heads == 0, \
            f"embed_dim ({embed_dim}) must be divisible by num_heads ({num_heads})"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        
        self.dropout = nn.Dropout(dropout)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize with Xavier uniform."""
        for module in [self.q_proj, self.k_proj, self.v_proj, self.out_proj]:
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: [batch, seq_len, embed_dim]
            mask: Optional attention mask
            return_attention: Whether to return attention weights
            
        Returns:
            output: [batch, seq_len, embed_dim]
            attention_weights: Optional [batch, num_heads, seq_len, seq_len]
        """
        batch_size, seq_len, _ = x.shape
        
        # Project to Q, K, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Scaled dot-product attention
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, float('-inf'))
        
        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention to values
        out = torch.matmul(attn_weights, v)
        
        # Reshape back
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        out = self.out_proj(out)
        
        if return_attention:
            return out, attn_weights
        return out, None


class CrossModalAttention(nn.Module):
    """
    Cross-Modal Attention for fusing MRI and biomarker features.
    
    Allows one modality (e.g., biomarkers) to attend to another
    modality (e.g., MRI features) to learn cross-modal relationships.
    """
    
    def __init__(
        self,
        query_dim: int,
        key_dim: int,
        hidden_dim: int = 128,
        num_heads: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.query_dim = query_dim
        self.key_dim = key_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        
        # Project both modalities to common space
        self.query_proj = nn.Linear(query_dim, hidden_dim)
        self.key_proj = nn.Linear(key_dim, hidden_dim)
        self.value_proj = nn.Linear(key_dim, hidden_dim)
        
        # Multi-head attention in common space
        self.attention = MultiHeadSelfAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Output projection
        self.out_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            query: [batch, query_dim] - e.g., biomarker features
            key_value: [batch, key_dim] - e.g., MRI features
            
        Returns:
            output: [batch, hidden_dim]
            attention_weights: Optional attention weights
        """
        # Project to common space
        q = self.query_proj(query).unsqueeze(1)  # [batch, 1, hidden]
        kv = self.key_proj(key_value).unsqueeze(1)  # [batch, 1, hidden]
        
        # Concatenate for self-attention
        combined = torch.cat([q, kv], dim=1)  # [batch, 2, hidden]
        
        # Self-attention
        attended, attn_weights = self.attention(combined, return_attention=return_attention)
        
        # Take the query position output
        output = attended[:, 0, :]  # [batch, hidden]
        
        output = self.out_proj(output)
        output = self.dropout(output)
        
        return output, attn_weights


class GatedFusion(nn.Module):
    """
    Gated Fusion mechanism for combining multiple modalities.
    
    Uses learned gates to dynamically weight the contribution
    of each modality based on the input.
    """
    
    def __init__(
        self,
        mri_dim: int,
        bio_dim: int,
        output_dim: int,
        dropout: float = 0.1
    ):
        super().__init__()
        
        combined_dim = mri_dim + bio_dim
        
        # Gate networks (one for each modality)
        self.mri_gate = nn.Sequential(
            nn.Linear(combined_dim, output_dim),
            nn.Sigmoid()
        )
        
        self.bio_gate = nn.Sequential(
            nn.Linear(combined_dim, output_dim),
            nn.Sigmoid()
        )
        
        # Transform networks
        self.mri_transform = nn.Sequential(
            nn.Linear(mri_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        self.bio_transform = nn.Sequential(
            nn.Linear(bio_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU()
        )
        
    def forward(
        self,
        mri_features: torch.Tensor,
        bio_features: torch.Tensor,
        return_gates: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor, torch.Tensor]]]:
        """
        Forward pass.
        
        Args:
            mri_features: [batch, mri_dim]
            bio_features: [batch, bio_dim]
            return_gates: Whether to return gate values
            
        Returns:
            fused: [batch, output_dim]
            gates: Optional tuple of (mri_gate, bio_gate)
        """
        # Concatenate for gate computation
        combined = torch.cat([mri_features, bio_features], dim=1)
        
        # Compute gates
        g_mri = self.mri_gate(combined)
        g_bio = self.bio_gate(combined)
        
        # Transform features
        h_mri = self.mri_transform(mri_features)
        h_bio = self.bio_transform(bio_features)
        
        # Gated combination
        fused = g_mri * h_mri + g_bio * h_bio
        fused = self.fusion(fused)
        
        if return_gates:
            return fused, (g_mri, g_bio)
        return fused, None


class TemporalAttention(nn.Module):
    """
    Temporal Attention for longitudinal features.
    
    Learns to weight baseline, followup, and delta features
    based on their importance for the prediction task.
    """
    
    def __init__(
        self,
        feature_dim: int,
        num_timepoints: int = 3,  # baseline, followup, delta
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_timepoints = num_timepoints
        
        # Temporal position encoding
        self.pos_encoding = nn.Parameter(
            torch.randn(1, num_timepoints, feature_dim) * 0.02
        )
        
        # Attention
        self.attention = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.Tanh(),
            nn.Linear(feature_dim // 2, 1)
        )
        
        self.layer_norm = nn.LayerNorm(feature_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        features: torch.Tensor,
        return_weights: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            features: [batch, num_timepoints, feature_dim]
            
        Returns:
            aggregated: [batch, feature_dim]
            attention_weights: Optional [batch, num_timepoints]
        """
        batch_size = features.shape[0]
        
        # Add positional encoding
        features = features + self.pos_encoding
        
        # Compute attention scores
        scores = self.attention(features).squeeze(-1)  # [batch, num_timepoints]
        weights = F.softmax(scores, dim=1)
        
        # Weighted sum
        aggregated = torch.bmm(
            weights.unsqueeze(1),
            features
        ).squeeze(1)  # [batch, feature_dim]
        
        aggregated = self.layer_norm(aggregated)
        aggregated = self.dropout(aggregated)
        
        if return_weights:
            return aggregated, weights
        return aggregated, None
