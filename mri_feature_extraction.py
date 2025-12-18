"""
================================================================================
Deep Learning-Based Multimodal MRI Feature Extraction Pipeline
================================================================================
Master's Research Project: Early Detection of Neurological Diseases
Dataset: OASIS-1 Cross-sectional MRI

Author: [Your Name]
Date: December 2025

This pipeline extracts features from:
1. Preprocessed T1-weighted MRI volumes (T88_111 atlas-registered)
2. Clinical/demographic variables (Age, MMSE, CDR, eTIV, nWBV, etc.)

Target Task: Binary classification of CDR=0 (normal) vs CDR=0.5 (very mild dementia)

================================================================================
DESIGN DECISIONS AND RATIONALE:
================================================================================

1. WHY 2.5D (MULTI-SLICE) APPROACH INSTEAD OF FULL 3D:
   - Full 3D CNNs require massive GPU memory (176×208×176 = 6.4M voxels)
   - OASIS-1 has only ~200 subjects → high risk of overfitting with 3D networks
   - 2.5D approach: Extract features from multiple orthogonal slices (axial, coronal, sagittal)
   - This captures 3D spatial information while using efficient 2D pretrained models
   - Pretrained ImageNet models (ResNet) transfer well to medical imaging slices

2. SLICE SELECTION STRATEGY:
   - Select center + offset slices from each axis (axial, coronal, sagittal)
   - Center slices capture hippocampus, ventricles - key regions for dementia
   - Multiple slices provide redundancy and capture more brain regions
   - Total: 3 axes × 3 slices = 9 slices → balanced representation

3. PRETRAINED MODEL CHOICE (ResNet18):
   - Proven transfer learning success in medical imaging
   - ResNet18 is lightweight (11M params) - good for small datasets
   - Feature vector: 512-dimensional (before final FC layer)
   - Alternative: MedicalNet or RadImageNet for medical-specific pretraining

4. CLINICAL FEATURE SELECTION:
   - Age: Strong predictor of brain atrophy
   - MMSE: Direct cognitive assessment (30-point scale)
   - nWBV: Normalized whole brain volume - atrophy marker
   - eTIV: Estimated total intracranial volume - normalization factor
   - ASF: Atlas scaling factor - head size correction
   - Educ: Education level - cognitive reserve proxy
   These features are clinically validated biomarkers for dementia detection.

5. FEATURE FUSION STRATEGY:
   - Early fusion: Concatenate MRI features + clinical features
   - MRI features: 512-dim (from ResNet18 pooling layer)
   - Clinical features: 6-dim (normalized)
   - Final vector: 518-dim per subject
   - Later: Can experiment with attention-based fusion

================================================================================
"""

import os
import glob
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import warnings

import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.models as models
import torchvision.transforms as transforms

# Try importing nibabel for MRI loading
try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    print("WARNING: nibabel not installed. Run: pip install nibabel")

# Try importing pandas for CSV handling
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("WARNING: pandas not installed. Run: pip install pandas")


# ==============================================================================
# CONFIGURATION
# ==============================================================================

class Config:
    """
    Central configuration for the feature extraction pipeline.
    Modify these parameters to adjust the pipeline behavior.
    """
    # Path configuration
    BASE_DIR = "D:/"  # Parent directory containing disc1, disc2, etc.
    DISC_PATTERN = "disc*"  # Pattern to match disc folders
    CSV_PATH = None  # Will be auto-detected or set manually
    
    # MRI file configuration
    MRI_SUBFOLDER = "PROCESSED/MPRAGE/T88_111"
    # Support both n3 and n4 variants (61 subjects have n3, 375 have n4)
    MRI_FILENAME_PATTERNS = [
        "*_mpr_n4_anon_111_t88_masked_gfc.hdr",  # Most subjects (375)
        "*_mpr_n3_anon_111_t88_gfc.hdr",          # Some subjects (61)
        "*_mpr_n*_anon_111_t88*.hdr",             # Fallback pattern
    ]
    
    # Slice extraction configuration
    # T88_111 volumes are typically 176 x 208 x 176 (x, y, z)
    # Slice indices are relative to center (0 = center slice)
    SLICE_OFFSETS = [-20, 0, 20]  # Center ± 20 slices
    
    # Image preprocessing
    TARGET_SIZE = (224, 224)  # ResNet input size
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]  # ImageNet normalization
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # Model configuration
    FEATURE_DIM = 512  # ResNet18 feature dimension
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Clinical features to extract
    CLINICAL_FEATURES = ['AGE', 'MMSE', 'nWBV', 'eTIV', 'ASF', 'EDUC']
    
    # Output configuration
    OUTPUT_DIR = "extracted_features"
    SAVE_FORMAT = "npz"  # 'npz' or 'pt'


# ==============================================================================
# MRI DATA LOADING AND PREPROCESSING
# ==============================================================================

class MRILoader:
    """
    Handles loading and preprocessing of OASIS MRI volumes.
    
    The T88_111 preprocessed volumes are:
    - Registered to Talairach atlas space
    - Bias-field corrected (N4)
    - Brain-masked (skull stripped)
    - Isotropic 1mm³ voxels
    - Dimensions: typically 176 x 208 x 176
    """
    
    def __init__(self, base_dir: str = Config.BASE_DIR):
        """
        Initialize the MRI loader.
        
        Args:
            base_dir: Parent directory containing disc folders
        """
        if not NIBABEL_AVAILABLE:
            raise ImportError("nibabel is required for MRI loading. Install with: pip install nibabel")
        
        self.base_dir = Path(base_dir)
        self.disc_folders = self._find_disc_folders()
        print(f"[MRILoader] Found {len(self.disc_folders)} disc folders: {[d.name for d in self.disc_folders]}")
    
    def _find_disc_folders(self) -> List[Path]:
        """Find all disc folders in the base directory."""
        pattern = self.base_dir / Config.DISC_PATTERN
        folders = sorted(glob.glob(str(pattern)))
        return [Path(f) for f in folders if os.path.isdir(f)]
    
    def get_all_subject_ids(self) -> List[str]:
        """
        Discover all subject IDs from disc folders.
        
        Returns:
            List of subject IDs (e.g., ['OAS1_0001_MR1', 'OAS1_0002_MR1', ...])
        """
        all_subjects = []
        for disc in self.disc_folders:
            # Find folders matching OAS1_XXXX_MR* pattern
            for subject_folder in disc.iterdir():
                if subject_folder.is_dir() and subject_folder.name.startswith('OAS1_'):
                    all_subjects.append(subject_folder.name)
        
        # Sort for consistent ordering
        all_subjects = sorted(list(set(all_subjects)))
        print(f"[MRILoader] Found {len(all_subjects)} subjects across all discs")
        return all_subjects
    
    def find_subject_folder(self, subject_id: str) -> Optional[Path]:
        """
        Find the folder for a given subject ID across all disc folders.
        
        Args:
            subject_id: e.g., "OAS1_0001_MR1"
            
        Returns:
            Path to subject folder, or None if not found
        """
        for disc in self.disc_folders:
            subject_path = disc / subject_id
            if subject_path.exists():
                return subject_path
        
        print(f"[WARNING] Subject {subject_id} not found in any disc folder")
        return None
    
    def get_mri_path(self, subject_id: str) -> Optional[Path]:
        """
        Get the path to the preprocessed T88_111 MRI volume.
        
        Args:
            subject_id: e.g., "OAS1_0001_MR1"
            
        Returns:
            Path to .hdr file, or None if not found
        """
        subject_folder = self.find_subject_folder(subject_id)
        if subject_folder is None:
            return None
        
        mri_folder = subject_folder / Config.MRI_SUBFOLDER
        if not mri_folder.exists():
            print(f"[WARNING] MRI folder not found: {mri_folder}")
            return None
        
        # Try multiple filename patterns (n3, n4 variants)
        for pattern_str in Config.MRI_FILENAME_PATTERNS:
            pattern = str(mri_folder / pattern_str)
            matches = glob.glob(pattern)
            if matches:
                return Path(matches[0])
        
        print(f"[WARNING] No MRI files matching any pattern in: {mri_folder}")
        return None
    
    def load_volume(self, subject_id: str) -> Optional[np.ndarray]:
        """
        Load the MRI volume for a subject.
        
        Args:
            subject_id: e.g., "OAS1_0001_MR1"
            
        Returns:
            3D numpy array (x, y, z) of MRI intensities, or None if failed
        """
        mri_path = self.get_mri_path(subject_id)
        if mri_path is None:
            return None
        
        try:
            # nibabel handles ANALYZE .hdr/.img pairs automatically
            img = nib.load(str(mri_path))
            data = img.get_fdata().astype(np.float32)
            
            # Handle 4D volumes (squeeze singleton dimensions)
            if data.ndim == 4 and data.shape[-1] == 1:
                data = data.squeeze(-1)
            elif data.ndim == 4:
                # Take first time point/channel if multiple
                data = data[..., 0]
            
            print(f"[MRILoader] Loaded {subject_id}: shape={data.shape}, "
                  f"range=[{data.min():.1f}, {data.max():.1f}]")
            
            return data
        
        except Exception as e:
            print(f"[ERROR] Failed to load MRI for {subject_id}: {e}")
            return None


class MRIPreprocessor:
    """
    Preprocesses MRI volumes for deep learning input.
    
    Steps:
    1. Intensity normalization (0-1 range, then z-score)
    2. Slice extraction (multi-planar: axial, coronal, sagittal)
    3. Resize to CNN input size
    4. Convert to RGB (replicate grayscale to 3 channels)
    5. Apply ImageNet normalization
    """
    
    def __init__(self, 
                 slice_offsets: List[int] = Config.SLICE_OFFSETS,
                 target_size: Tuple[int, int] = Config.TARGET_SIZE):
        """
        Initialize the preprocessor.
        
        Args:
            slice_offsets: Offsets from center slice to extract
            target_size: (H, W) for CNN input
        """
        self.slice_offsets = slice_offsets
        self.target_size = target_size
        
        # PyTorch transforms for final preprocessing
        self.transform = transforms.Compose([
            transforms.Normalize(
                mean=Config.NORMALIZE_MEAN,
                std=Config.NORMALIZE_STD
            )
        ])
    
    def normalize_volume(self, volume: np.ndarray) -> np.ndarray:
        """
        Normalize MRI intensities.
        
        Brain MRI typically has:
        - Background: 0 (masked out)
        - Brain tissue: variable intensity range
        
        We normalize only the non-zero (brain) voxels.
        """
        # Create brain mask (non-zero voxels)
        mask = volume > 0
        
        if mask.sum() == 0:
            print("[WARNING] Empty volume (all zeros)")
            return volume
        
        # Normalize brain voxels to [0, 1]
        brain_min = volume[mask].min()
        brain_max = volume[mask].max()
        
        normalized = np.zeros_like(volume)
        normalized[mask] = (volume[mask] - brain_min) / (brain_max - brain_min + 1e-8)
        
        return normalized
    
    def extract_slices(self, volume: np.ndarray) -> Dict[str, List[np.ndarray]]:
        """
        Extract multiple slices from each anatomical plane.
        
        Planes:
        - Axial (transverse): horizontal slices, top-down view
        - Coronal: front-back slices
        - Sagittal: left-right slices
        
        Args:
            volume: 3D array of shape (X, Y, Z)
            
        Returns:
            Dictionary with keys 'axial', 'coronal', 'sagittal',
            each containing a list of 2D slice arrays
        """
        x_dim, y_dim, z_dim = volume.shape
        
        # Calculate center indices for each axis
        centers = {
            'sagittal': x_dim // 2,  # X-axis: left-right
            'coronal': y_dim // 2,   # Y-axis: front-back  
            'axial': z_dim // 2      # Z-axis: top-bottom
        }
        
        slices = {'axial': [], 'coronal': [], 'sagittal': []}
        
        for offset in self.slice_offsets:
            # Axial slices (Z-axis)
            z_idx = np.clip(centers['axial'] + offset, 0, z_dim - 1)
            slices['axial'].append(volume[:, :, z_idx])
            
            # Coronal slices (Y-axis)
            y_idx = np.clip(centers['coronal'] + offset, 0, y_dim - 1)
            slices['coronal'].append(volume[:, y_idx, :])
            
            # Sagittal slices (X-axis)
            x_idx = np.clip(centers['sagittal'] + offset, 0, x_dim - 1)
            slices['sagittal'].append(volume[x_idx, :, :])
        
        return slices
    
    def preprocess_slice(self, slice_2d: np.ndarray) -> torch.Tensor:
        """
        Preprocess a single 2D slice for CNN input.
        
        Steps:
        1. Resize to target size
        2. Convert to 3-channel (RGB) by replication
        3. Convert to PyTorch tensor
        4. Apply ImageNet normalization
        
        Args:
            slice_2d: 2D numpy array
            
        Returns:
            Tensor of shape (3, H, W), normalized
        """
        # Resize using bilinear interpolation
        # Convert to tensor first for F.interpolate
        tensor = torch.from_numpy(slice_2d).float().unsqueeze(0).unsqueeze(0)
        resized = F.interpolate(tensor, size=self.target_size, mode='bilinear', align_corners=False)
        resized = resized.squeeze()  # Back to (H, W)
        
        # Replicate to 3 channels (grayscale → RGB)
        rgb = resized.unsqueeze(0).repeat(3, 1, 1)  # (3, H, W)
        
        # Apply ImageNet normalization
        normalized = self.transform(rgb)
        
        return normalized
    
    def process_volume(self, volume: np.ndarray) -> torch.Tensor:
        """
        Full preprocessing pipeline for a single MRI volume.
        
        Args:
            volume: Raw 3D MRI array
            
        Returns:
            Tensor of shape (N_slices, 3, H, W) where N_slices = 9
            (3 planes × 3 slices per plane)
        """
        # Step 1: Intensity normalization
        normalized = self.normalize_volume(volume)
        
        # Step 2: Extract multi-planar slices
        slices_dict = self.extract_slices(normalized)
        
        # Step 3: Preprocess each slice
        all_slices = []
        for plane in ['axial', 'coronal', 'sagittal']:
            for slice_2d in slices_dict[plane]:
                processed = self.preprocess_slice(slice_2d)
                all_slices.append(processed)
        
        # Stack all slices: (N_slices, 3, H, W)
        batch = torch.stack(all_slices, dim=0)
        
        return batch


# ==============================================================================
# DEEP LEARNING FEATURE EXTRACTOR
# ==============================================================================

class MRIFeatureExtractor(nn.Module):
    """
    Extracts deep features from MRI slices using a pretrained CNN.
    
    Architecture:
    - Base: ResNet18 pretrained on ImageNet
    - Remove final FC layer
    - Use global average pooling output (512-dim)
    - Aggregate features from all slices (mean pooling)
    
    Why ResNet18:
    - Proven transfer learning success in medical imaging
    - Lightweight (11M params) - reduces overfitting on small datasets
    - Good depth/performance tradeoff
    - Well-studied feature representations
    """
    
    def __init__(self, 
                 model_name: str = 'resnet18',
                 pretrained: bool = True,
                 freeze_backbone: bool = True):
        """
        Initialize the feature extractor.
        
        Args:
            model_name: Currently supports 'resnet18', 'resnet34', 'resnet50'
            pretrained: Whether to use ImageNet pretrained weights
            freeze_backbone: Whether to freeze CNN weights (recommended for feature extraction)
        """
        super().__init__()
        
        self.model_name = model_name
        self.feature_dim = Config.FEATURE_DIM
        
        # Load pretrained model
        if model_name == 'resnet18':
            weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.resnet18(weights=weights)
            self.feature_dim = 512
        elif model_name == 'resnet34':
            weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.resnet34(weights=weights)
            self.feature_dim = 512
        elif model_name == 'resnet50':
            weights = models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            base_model = models.resnet50(weights=weights)
            self.feature_dim = 2048
        else:
            raise ValueError(f"Unsupported model: {model_name}")
        
        # Remove final FC layer - keep everything up to avgpool
        self.backbone = nn.Sequential(*list(base_model.children())[:-1])
        
        # Freeze backbone weights if specified
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        print(f"[FeatureExtractor] Initialized {model_name} "
              f"(pretrained={pretrained}, frozen={freeze_backbone}, "
              f"feature_dim={self.feature_dim})")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Extract features from input slices.
        
        Args:
            x: Tensor of shape (N_slices, 3, H, W)
            
        Returns:
            Tensor of shape (feature_dim,) - aggregated features
        """
        # Get features for each slice: (N_slices, feature_dim, 1, 1)
        features = self.backbone(x)
        
        # Flatten: (N_slices, feature_dim)
        features = features.view(features.size(0), -1)
        
        # Aggregate across slices: mean pooling
        # This gives a single feature vector regardless of number of slices
        aggregated = features.mean(dim=0)  # (feature_dim,)
        
        return aggregated
    
    def extract_features(self, slice_batch: torch.Tensor) -> np.ndarray:
        """
        Extract features from preprocessed slices (convenience method).
        
        Args:
            slice_batch: Preprocessed slices tensor (N, 3, H, W)
            
        Returns:
            Feature vector as numpy array (feature_dim,)
        """
        self.eval()
        device = next(self.parameters()).device
        
        with torch.no_grad():
            x = slice_batch.to(device)
            features = self.forward(x)
            return features.cpu().numpy()


# ==============================================================================
# CLINICAL FEATURE EXTRACTION
# ==============================================================================

class ClinicalFeatureExtractor:
    """
    Extracts and normalizes clinical/demographic features.
    
    Sources:
    1. Subject .txt files (always available in each folder)
    2. CSV file (oasis_cross-sectional.csv) if available
    
    Features:
    - AGE: Subject age in years
    - MMSE: Mini-Mental State Examination score (0-30)
    - nWBV: Normalized whole brain volume (0-1)
    - eTIV: Estimated total intracranial volume (ml)
    - ASF: Atlas scaling factor
    - EDUC: Education level (1-5)
    - CDR: Clinical Dementia Rating (target variable for classification)
    """
    
    def __init__(self, 
                 csv_path: Optional[str] = None,
                 feature_names: List[str] = Config.CLINICAL_FEATURES):
        """
        Initialize the clinical feature extractor.
        
        Args:
            csv_path: Path to oasis_cross-sectional.csv (optional)
            feature_names: List of clinical features to extract
        """
        self.csv_path = csv_path
        self.feature_names = feature_names
        self.csv_data = None
        
        # Load CSV if provided
        if csv_path and PANDAS_AVAILABLE and os.path.exists(csv_path):
            self.csv_data = pd.read_csv(csv_path)
            print(f"[ClinicalExtractor] Loaded CSV with {len(self.csv_data)} subjects")
        
        # Normalization statistics (approximate ranges for OASIS)
        # These will be used for z-score normalization
        self.normalization_stats = {
            'AGE': {'mean': 50.0, 'std': 20.0},       # Typical age range
            'MMSE': {'mean': 27.0, 'std': 3.0},       # MMSE range ~20-30
            'nWBV': {'mean': 0.75, 'std': 0.05},      # Brain volume fraction
            'eTIV': {'mean': 1500.0, 'std': 200.0},   # Intracranial volume (ml)
            'ASF': {'mean': 1.2, 'std': 0.2},         # Scaling factor
            'EDUC': {'mean': 3.0, 'std': 1.5},        # Education (1-5)
            'SES': {'mean': 2.5, 'std': 1.0},         # Socioeconomic status
        }
    
    def _parse_subject_txt(self, txt_path: str) -> Dict[str, any]:
        """
        Parse clinical features from subject .txt file.
        
        The .txt file format:
        SESSION ID:   OAS1_0001_MR1
        AGE:          74
        M/F:          Female
        ...
        """
        metadata = {}
        
        try:
            with open(txt_path, 'r') as f:
                for line in f:
                    if ':' in line:
                        # Split on first colon only
                        parts = line.strip().split(':', 1)
                        if len(parts) == 2:
                            key = parts[0].strip().upper()
                            value = parts[1].strip()
                            
                            # Handle special mappings
                            if key == 'M/F':
                                key = 'GENDER'
                            
                            # Try to convert to numeric
                            try:
                                value = float(value)
                            except ValueError:
                                pass  # Keep as string
                            
                            metadata[key] = value
        
        except Exception as e:
            print(f"[WARNING] Error parsing {txt_path}: {e}")
        
        return metadata
    
    def get_features_from_txt(self, subject_folder: Path) -> Dict[str, any]:
        """
        Extract clinical features from subject's .txt file.
        
        Args:
            subject_folder: Path to subject folder (e.g., .../OAS1_0001_MR1/)
            
        Returns:
            Dictionary of clinical features
        """
        subject_id = subject_folder.name
        txt_path = subject_folder / f"{subject_id}.txt"
        
        if not txt_path.exists():
            print(f"[WARNING] Subject txt file not found: {txt_path}")
            return {}
        
        return self._parse_subject_txt(str(txt_path))
    
    def get_features_from_csv(self, subject_id: str) -> Dict[str, any]:
        """
        Extract clinical features from CSV file.
        
        Args:
            subject_id: e.g., "OAS1_0001_MR1"
            
        Returns:
            Dictionary of clinical features
        """
        if self.csv_data is None:
            return {}
        
        # Find row for this subject
        # CSV column might be 'ID' or 'Subject' or similar
        id_columns = ['ID', 'Subject', 'Session ID', 'SESSION ID']
        
        for col in id_columns:
            if col in self.csv_data.columns:
                mask = self.csv_data[col] == subject_id
                if mask.any():
                    row = self.csv_data[mask].iloc[0]
                    return row.to_dict()
        
        return {}
    
    def extract_features(self, 
                        subject_id: str, 
                        mri_loader: MRILoader) -> Tuple[np.ndarray, Dict[str, any]]:
        """
        Extract normalized clinical feature vector for a subject.
        
        Args:
            subject_id: e.g., "OAS1_0001_MR1"
            mri_loader: MRILoader instance to find subject folder
            
        Returns:
            Tuple of:
            - Feature vector (numpy array of shape (n_features,))
            - Raw metadata dictionary (for reference)
        """
        # Get subject folder
        subject_folder = mri_loader.find_subject_folder(subject_id)
        if subject_folder is None:
            print(f"[WARNING] Cannot find folder for {subject_id}")
            return np.zeros(len(self.feature_names)), {}
        
        # Get features from .txt file (primary source)
        metadata = self.get_features_from_txt(subject_folder)
        
        # Supplement with CSV data if available
        csv_metadata = self.get_features_from_csv(subject_id)
        for key, value in csv_metadata.items():
            if key.upper() not in metadata:
                metadata[key.upper()] = value
        
        # Build feature vector
        features = []
        for feat_name in self.feature_names:
            value = metadata.get(feat_name.upper(), None)
            
            if value is None or value == '' or (isinstance(value, float) and np.isnan(value)):
                # Use mean as default for missing values
                value = self.normalization_stats.get(feat_name, {}).get('mean', 0.0)
                print(f"[INFO] Using default value for {subject_id}.{feat_name}: {value}")
            else:
                value = float(value)
            
            # Z-score normalization
            stats = self.normalization_stats.get(feat_name, {'mean': 0, 'std': 1})
            normalized = (value - stats['mean']) / (stats['std'] + 1e-8)
            features.append(normalized)
        
        feature_vector = np.array(features, dtype=np.float32)
        
        # Store CDR separately (for labels)
        cdr = metadata.get('CDR', None)
        metadata['CDR_VALUE'] = cdr
        
        return feature_vector, metadata


# ==============================================================================
# MAIN PIPELINE
# ==============================================================================

class MultimodalFeatureExtractor:
    """
    Complete pipeline for extracting multimodal features from OASIS subjects.
    
    Combines:
    1. MRI features (deep learning from T1-weighted scans)
    2. Clinical features (demographics, cognitive scores, brain volumes)
    
    Output per subject:
    - MRI features: 512-dimensional vector (ResNet18)
    - Clinical features: 6-dimensional vector
    - Combined: 518-dimensional vector
    """
    
    def __init__(self,
                 base_dir: str = Config.BASE_DIR,
                 csv_path: Optional[str] = Config.CSV_PATH,
                 model_name: str = 'resnet18',
                 device: str = Config.DEVICE):
        """
        Initialize the multimodal feature extraction pipeline.
        
        Args:
            base_dir: Parent directory containing disc folders
            csv_path: Optional path to clinical CSV file
            model_name: CNN model for MRI feature extraction
            device: 'cuda' or 'cpu'
        """
        self.device = device
        print(f"\n{'='*60}")
        print("Initializing Multimodal Feature Extraction Pipeline")
        print(f"{'='*60}")
        print(f"Device: {device}")
        print(f"Base directory: {base_dir}")
        
        # Initialize components
        self.mri_loader = MRILoader(base_dir)
        self.preprocessor = MRIPreprocessor()
        self.mri_extractor = MRIFeatureExtractor(model_name=model_name).to(device)
        self.clinical_extractor = ClinicalFeatureExtractor(csv_path=csv_path)
        
        print(f"\nMRI feature dimension: {self.mri_extractor.feature_dim}")
        print(f"Clinical feature dimension: {len(Config.CLINICAL_FEATURES)}")
        print(f"Combined feature dimension: {self.mri_extractor.feature_dim + len(Config.CLINICAL_FEATURES)}")
        print(f"{'='*60}\n")
    
    def extract_subject_features(self, subject_id: str) -> Optional[Dict]:
        """
        Extract all features for a single subject.
        
        Args:
            subject_id: e.g., "OAS1_0001_MR1"
            
        Returns:
            Dictionary with:
            - 'subject_id': str
            - 'mri_features': np.ndarray (512,)
            - 'clinical_features': np.ndarray (6,)
            - 'combined_features': np.ndarray (518,)
            - 'metadata': dict with raw clinical values
            - 'cdr': CDR value (for labels)
        """
        print(f"\n[Pipeline] Processing subject: {subject_id}")
        print("-" * 40)
        
        # Step 1: Load MRI volume
        volume = self.mri_loader.load_volume(subject_id)
        if volume is None:
            print(f"[ERROR] Failed to load MRI for {subject_id}")
            return None
        
        # Step 2: Preprocess MRI (normalize, extract slices)
        print(f"[Pipeline] Preprocessing MRI...")
        slice_batch = self.preprocessor.process_volume(volume)
        print(f"[Pipeline] Extracted {slice_batch.shape[0]} slices, "
              f"shape per slice: {slice_batch.shape[1:]}")
        
        # Step 3: Extract MRI features
        print(f"[Pipeline] Extracting deep features...")
        mri_features = self.mri_extractor.extract_features(slice_batch)
        print(f"[Pipeline] MRI feature vector shape: {mri_features.shape}")
        
        # Step 4: Extract clinical features
        print(f"[Pipeline] Extracting clinical features...")
        clinical_features, metadata = self.clinical_extractor.extract_features(
            subject_id, self.mri_loader
        )
        print(f"[Pipeline] Clinical feature vector shape: {clinical_features.shape}")
        print(f"[Pipeline] Clinical values: Age={metadata.get('AGE')}, "
              f"MMSE={metadata.get('MMSE')}, CDR={metadata.get('CDR')}")
        
        # Step 5: Combine features
        combined_features = np.concatenate([mri_features, clinical_features])
        print(f"[Pipeline] Combined feature vector shape: {combined_features.shape}")
        
        return {
            'subject_id': subject_id,
            'mri_features': mri_features,
            'clinical_features': clinical_features,
            'combined_features': combined_features,
            'metadata': metadata,
            'cdr': metadata.get('CDR_VALUE', metadata.get('CDR'))
        }
    
    def extract_batch(self, subject_ids: List[str]) -> Dict[str, any]:
        """
        Extract features for multiple subjects.
        
        Args:
            subject_ids: List of subject IDs
            
        Returns:
            Dictionary with:
            - 'subject_ids': list of successfully processed IDs
            - 'mri_features': np.ndarray (N, 512)
            - 'clinical_features': np.ndarray (N, 6)
            - 'combined_features': np.ndarray (N, 518)
            - 'labels': np.ndarray of CDR values
            - 'metadata': list of metadata dicts
        """
        results = {
            'subject_ids': [],
            'mri_features': [],
            'clinical_features': [],
            'combined_features': [],
            'labels': [],
            'metadata': []
        }
        
        for subject_id in subject_ids:
            try:
                features = self.extract_subject_features(subject_id)
                
                if features is not None:
                    results['subject_ids'].append(features['subject_id'])
                    results['mri_features'].append(features['mri_features'])
                    results['clinical_features'].append(features['clinical_features'])
                    results['combined_features'].append(features['combined_features'])
                    results['labels'].append(features['cdr'])
                    results['metadata'].append(features['metadata'])
                    
            except Exception as e:
                print(f"[ERROR] Failed to process {subject_id}: {e}")
                import traceback
                traceback.print_exc()
        
        # Convert lists to arrays
        if results['mri_features']:
            results['mri_features'] = np.stack(results['mri_features'])
            results['clinical_features'] = np.stack(results['clinical_features'])
            results['combined_features'] = np.stack(results['combined_features'])
            results['labels'] = np.array(results['labels'])
        
        return results
    
    def save_features(self, 
                     results: Dict, 
                     output_path: str,
                     format: str = Config.SAVE_FORMAT):
        """
        Save extracted features to file.
        
        Args:
            results: Output from extract_batch()
            output_path: Path to save file (without extension)
            format: 'npz' or 'pt'
        """
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        
        if format == 'npz':
            save_path = f"{output_path}.npz"
            np.savez(
                save_path,
                subject_ids=np.array(results['subject_ids']),
                mri_features=results['mri_features'],
                clinical_features=results['clinical_features'],
                combined_features=results['combined_features'],
                labels=results['labels']
            )
            print(f"\n[Pipeline] Saved features to: {save_path}")
            
        elif format == 'pt':
            save_path = f"{output_path}.pt"
            # Handle labels with empty strings/None values - convert to NaN
            labels_array = []
            for label in results['labels']:
                if label is None or label == '' or (isinstance(label, str) and label.strip() == ''):
                    labels_array.append(float('nan'))
                else:
                    labels_array.append(float(label))
            labels_tensor = torch.tensor(labels_array, dtype=torch.float32)
            
            torch.save({
                'subject_ids': results['subject_ids'],
                'mri_features': torch.from_numpy(results['mri_features']),
                'clinical_features': torch.from_numpy(results['clinical_features']),
                'combined_features': torch.from_numpy(results['combined_features']),
                'labels': labels_tensor
            }, save_path)
            print(f"\n[Pipeline] Saved features to: {save_path}")
        
        else:
            raise ValueError(f"Unsupported format: {format}")


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """
    Main function for FULL feature extraction from ALL OASIS-1 subjects.
    
    Features:
    - Auto-discovers all subjects from disc1-disc12
    - Checkpoint saving every 50 subjects
    - Resume from checkpoint if exists
    - Progress tracking with ETA
    """
    import time
    import json
    
    print("\n" + "="*70)
    print("MULTIMODAL MRI FEATURE EXTRACTION FOR EARLY DEMENTIA DETECTION")
    print("="*70)
    print("\nProject: Deep Learning-Based Multimodal MRI Analysis")
    print("Dataset: OASIS-1 Cross-sectional (ALL SUBJECTS)")
    print("Target: CDR=0 (normal) vs CDR=0.5 (very mild dementia)")
    print("="*70 + "\n")
    
    # ==========================================
    # CONFIGURATION
    # ==========================================
    
    BASE_DIR = "D:/discs"
    CSV_PATH = None
    OUTPUT_PATH = "D:/discs/extracted_features/oasis_all_features"
    CHECKPOINT_PATH = "D:/discs/extracted_features/checkpoint.json"
    CHECKPOINT_EVERY = 50  # Save progress every N subjects
    
    # ==========================================
    # INITIALIZE PIPELINE
    # ==========================================
    
    pipeline = MultimodalFeatureExtractor(
        base_dir=BASE_DIR,
        csv_path=CSV_PATH,
        model_name='resnet18',
        device=Config.DEVICE
    )
    
    # ==========================================
    # AUTO-DISCOVER ALL SUBJECTS
    # ==========================================
    
    print("\n" + "="*70)
    print("DISCOVERING SUBJECTS")
    print("="*70)
    
    all_subjects = pipeline.mri_loader.get_all_subject_ids()
    print(f"\nTotal subjects found: {len(all_subjects)}")
    
    # ==========================================
    # CHECK FOR EXISTING CHECKPOINT
    # ==========================================
    
    processed_subjects = set()
    results = {
        'subject_ids': [],
        'mri_features': [],
        'clinical_features': [],
        'combined_features': [],
        'labels': [],
        'metadata': []
    }
    
    if os.path.exists(CHECKPOINT_PATH):
        print(f"\n[CHECKPOINT] Found existing checkpoint, loading...")
        try:
            with open(CHECKPOINT_PATH, 'r') as f:
                checkpoint = json.load(f)
            processed_subjects = set(checkpoint.get('processed', []))
            print(f"[CHECKPOINT] Resuming from {len(processed_subjects)} already processed subjects")
            
            # Load existing features if they exist
            npz_path = f"{OUTPUT_PATH}_partial.npz"
            if os.path.exists(npz_path):
                existing = np.load(npz_path, allow_pickle=True)
                results['subject_ids'] = list(existing['subject_ids'])
                results['mri_features'] = list(existing['mri_features'])
                results['clinical_features'] = list(existing['clinical_features'])
                results['combined_features'] = list(existing['combined_features'])
                results['labels'] = list(existing['labels'])
                print(f"[CHECKPOINT] Loaded {len(results['subject_ids'])} existing feature vectors")
        except Exception as e:
            print(f"[WARNING] Failed to load checkpoint: {e}")
            processed_subjects = set()
    
    # Filter to only unprocessed subjects
    subjects_to_process = [s for s in all_subjects if s not in processed_subjects]
    print(f"\nSubjects remaining to process: {len(subjects_to_process)}")
    
    # ==========================================
    # EXTRACT FEATURES WITH PROGRESS TRACKING
    # ==========================================
    
    print("\n" + "="*70)
    print("FEATURE EXTRACTION")
    print("="*70)
    
    start_time = time.time()
    success_count = 0
    fail_count = 0
    
    for i, subject_id in enumerate(subjects_to_process):
        try:
            # Progress indicator
            elapsed = time.time() - start_time
            if success_count > 0:
                avg_time = elapsed / (i + 1)
                remaining = avg_time * (len(subjects_to_process) - i - 1)
                eta_min = remaining / 60
                print(f"\n[{i+1}/{len(subjects_to_process)}] Processing {subject_id} "
                      f"(ETA: {eta_min:.1f} min)")
            else:
                print(f"\n[{i+1}/{len(subjects_to_process)}] Processing {subject_id}")
            
            # Extract features
            features = pipeline.extract_subject_features(subject_id)
            
            if features is not None:
                results['subject_ids'].append(features['subject_id'])
                results['mri_features'].append(features['mri_features'])
                results['clinical_features'].append(features['clinical_features'])
                results['combined_features'].append(features['combined_features'])
                results['labels'].append(features['cdr'])
                results['metadata'].append(features['metadata'])
                processed_subjects.add(subject_id)
                success_count += 1
                
                # Show CDR info
                cdr = features.get('cdr', 'N/A')
                print(f"    ✓ Success! CDR={cdr}, Features shape: {features['combined_features'].shape}")
            else:
                fail_count += 1
                print(f"    ✗ Failed to extract features")
            
            # Checkpoint save
            if (i + 1) % CHECKPOINT_EVERY == 0:
                print(f"\n[CHECKPOINT] Saving progress at {i+1} subjects...")
                
                # Save checkpoint file
                with open(CHECKPOINT_PATH, 'w') as f:
                    json.dump({'processed': list(processed_subjects)}, f)
                
                # Save partial results
                if results['mri_features']:
                    np.savez(
                        f"{OUTPUT_PATH}_partial.npz",
                        subject_ids=np.array(results['subject_ids']),
                        mri_features=np.stack(results['mri_features']),
                        clinical_features=np.stack(results['clinical_features']),
                        combined_features=np.stack(results['combined_features']),
                        labels=np.array(results['labels'], dtype=object)
                    )
                print(f"[CHECKPOINT] Saved {len(results['subject_ids'])} subjects")
                
        except Exception as e:
            print(f"[ERROR] Failed to process {subject_id}: {e}")
            fail_count += 1
            import traceback
            traceback.print_exc()
    
    # Convert lists to arrays
    if results['mri_features']:
        results['mri_features'] = np.stack(results['mri_features'])
        results['clinical_features'] = np.stack(results['clinical_features'])
        results['combined_features'] = np.stack(results['combined_features'])
        results['labels'] = np.array(results['labels'], dtype=object)
    
    # ==========================================
    # SUMMARY
    # ==========================================
    
    total_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("EXTRACTION SUMMARY")
    print("="*70)
    
    print(f"\nTotal time: {total_time/60:.1f} minutes")
    print(f"Successful: {success_count}")
    print(f"Failed: {fail_count}")
    
    if results['subject_ids']:
        print(f"\nTotal subjects with features: {len(results['subject_ids'])}")
        
        print(f"\nFeature shapes:")
        print(f"  MRI features:      {results['mri_features'].shape}")
        print(f"  Clinical features: {results['clinical_features'].shape}")
        print(f"  Combined features: {results['combined_features'].shape}")
        print(f"  Labels (CDR):      {results['labels'].shape}")
        
        # CDR distribution
        cdr_values = [c for c in results['labels'] if c is not None]
        cdr_0 = sum(1 for c in cdr_values if c == 0)
        cdr_05 = sum(1 for c in cdr_values if c == 0.5)
        cdr_1 = sum(1 for c in cdr_values if c == 1)
        cdr_2 = sum(1 for c in cdr_values if c == 2)
        cdr_none = len(results['labels']) - len(cdr_values)
        
        print(f"\nCDR Distribution:")
        print(f"  CDR=0 (Normal):        {cdr_0}")
        print(f"  CDR=0.5 (Very Mild):   {cdr_05}")
        print(f"  CDR=1 (Mild):          {cdr_1}")
        print(f"  CDR=2 (Moderate):      {cdr_2}")
        print(f"  CDR=None (Young ctrl): {cdr_none}")
        
        # Print feature statistics
        print(f"\nMRI feature statistics:")
        print(f"  Mean: {results['mri_features'].mean():.4f}")
        print(f"  Std:  {results['mri_features'].std():.4f}")
        print(f"  Min:  {results['mri_features'].min():.4f}")
        print(f"  Max:  {results['mri_features'].max():.4f}")
        
        # Save final features
        print(f"\nSaving final features...")
        pipeline.save_features(results, OUTPUT_PATH, format='npz')
        pipeline.save_features(results, OUTPUT_PATH, format='pt')
        
        # Clean up checkpoint
        if os.path.exists(CHECKPOINT_PATH):
            os.remove(CHECKPOINT_PATH)
            print("[CHECKPOINT] Cleaned up checkpoint file")
        partial_path = f"{OUTPUT_PATH}_partial.npz"
        if os.path.exists(partial_path):
            os.remove(partial_path)
            print("[CHECKPOINT] Cleaned up partial file")
        
        print(f"\n" + "="*70)
        print("EXTRACTION COMPLETE!")
        print("="*70)
        print(f"\nFiles saved:")
        print(f"  {OUTPUT_PATH}.npz")
        print(f"  {OUTPUT_PATH}.pt")
        print(f"\nUsable for classification:")
        print(f"  CDR=0 vs CDR=0.5: {cdr_0 + cdr_05} subjects")
        
    else:
        print("\n[ERROR] No subjects were successfully processed!")
        print("Please check:")
        print("  1. BASE_DIR is correct")
        print("  2. Subject folders exist in disc folders")
        print("  3. MRI files are present in PROCESSED/MPRAGE/T88_111/")
    
    return results


if __name__ == "__main__":
    # Check dependencies
    print("\nChecking dependencies...")
    print(f"  nibabel: {'✓ installed' if NIBABEL_AVAILABLE else '✗ NOT INSTALLED'}")
    print(f"  pandas:  {'✓ installed' if PANDAS_AVAILABLE else '✗ NOT INSTALLED (optional)'}")
    print(f"  PyTorch: ✓ installed (version {torch.__version__})")
    print(f"  CUDA:    {'✓ available' if torch.cuda.is_available() else '✗ not available (using CPU)'}")
    
    if not NIBABEL_AVAILABLE:
        print("\n[ERROR] nibabel is required! Install with: pip install nibabel")
    else:
        results = main()

