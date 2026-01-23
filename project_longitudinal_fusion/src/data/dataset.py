"""
PyTorch Dataset Classes for Multimodal Fusion
==============================================
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional


class MultimodalFusionDataset(Dataset):
    """
    PyTorch Dataset for multimodal MRI + Biomarker fusion.
    
    Each sample contains:
    - baseline_resnet: 512-dim CNN features from baseline scan
    - followup_resnet: 512-dim CNN features from follow-up scan
    - delta_resnet: 512-dim temporal change in CNN features
    - baseline_bio: 9-dim baseline biomarkers (volumes + demographics)
    - followup_bio: 6-dim follow-up volumetric biomarkers
    - delta_bio: 6-dim biomarker changes (atrophy rates)
    - label: 0 = Stable, 1 = Converter
    """
    
    def __init__(
        self,
        baseline_resnet: np.ndarray,
        followup_resnet: np.ndarray,
        delta_resnet: np.ndarray,
        baseline_bio: np.ndarray,
        followup_bio: np.ndarray,
        delta_bio: np.ndarray,
        labels: np.ndarray,
        subject_ids: Optional[np.ndarray] = None
    ):
        """
        Initialize dataset.
        
        Args:
            baseline_resnet: [N, 512] baseline CNN features
            followup_resnet: [N, 512] follow-up CNN features
            delta_resnet: [N, 512] temporal change in CNN
            baseline_bio: [N, 9] baseline biomarkers
            followup_bio: [N, 6] follow-up biomarkers
            delta_bio: [N, 6] biomarker changes
            labels: [N] binary labels
            subject_ids: [N] subject IDs (optional)
        """
        self.baseline_resnet = torch.FloatTensor(baseline_resnet)
        self.followup_resnet = torch.FloatTensor(followup_resnet)
        self.delta_resnet = torch.FloatTensor(delta_resnet)
        
        self.baseline_bio = torch.FloatTensor(baseline_bio)
        self.followup_bio = torch.FloatTensor(followup_bio)
        self.delta_bio = torch.FloatTensor(delta_bio)
        
        self.labels = torch.LongTensor(labels)
        self.subject_ids = subject_ids
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a single sample."""
        sample = {
            'baseline_resnet': self.baseline_resnet[idx],
            'followup_resnet': self.followup_resnet[idx],
            'delta_resnet': self.delta_resnet[idx],
            'baseline_bio': self.baseline_bio[idx],
            'followup_bio': self.followup_bio[idx],
            'delta_bio': self.delta_bio[idx],
            'label': self.labels[idx]
        }
        
        if self.subject_ids is not None:
            sample['subject_id'] = self.subject_ids[idx]
            
        return sample
    
    @classmethod
    def from_dict(cls, data: Dict[str, np.ndarray]) -> 'MultimodalFusionDataset':
        """Create dataset from dictionary."""
        return cls(
            baseline_resnet=data['baseline_resnet'],
            followup_resnet=data['followup_resnet'],
            delta_resnet=data['delta_resnet'],
            baseline_bio=data['baseline_bio'],
            followup_bio=data['followup_bio'],
            delta_bio=data['delta_bio'],
            labels=data['labels'],
            subject_ids=data.get('subject_ids')
        )


class BiomarkerOnlyDataset(Dataset):
    """Dataset using only biomarker features (for baseline comparison)."""
    
    def __init__(
        self,
        baseline_bio: np.ndarray,
        followup_bio: np.ndarray,
        delta_bio: np.ndarray,
        labels: np.ndarray
    ):
        # Concatenate all biomarkers into single vector
        all_bio = np.concatenate([baseline_bio, followup_bio, delta_bio], axis=1)
        self.features = torch.FloatTensor(all_bio)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


class ResNetOnlyDataset(Dataset):
    """Dataset using only ResNet features (for baseline comparison)."""
    
    def __init__(
        self,
        baseline_resnet: np.ndarray,
        followup_resnet: np.ndarray,
        delta_resnet: np.ndarray,
        labels: np.ndarray
    ):
        # Concatenate all ResNet features
        all_resnet = np.concatenate([baseline_resnet, followup_resnet, delta_resnet], axis=1)
        self.features = torch.FloatTensor(all_resnet)
        self.labels = torch.LongTensor(labels)
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.labels[idx]


def create_dataloaders(
    train_data: Dict[str, np.ndarray],
    test_data: Dict[str, np.ndarray],
    batch_size: int = 32,
    num_workers: int = 0
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train and test DataLoaders.
    
    Args:
        train_data: Dictionary with train features
        test_data: Dictionary with test features
        batch_size: Batch size
        num_workers: Number of data loading workers
        
    Returns:
        train_loader, test_loader
    """
    train_dataset = MultimodalFusionDataset.from_dict(train_data)
    test_dataset = MultimodalFusionDataset.from_dict(test_data)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader
