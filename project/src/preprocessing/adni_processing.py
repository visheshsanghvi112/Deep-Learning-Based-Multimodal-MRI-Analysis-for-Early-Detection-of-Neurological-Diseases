"""
================================================================================
ADNI Dataset Processing Pipeline
================================================================================
Processes ADNI dataset for feature extraction:
1. Clinical data linkage
2. Spatial normalization
3. Feature extraction (same pipeline as OASIS-1)

Part of: Master Research Plan - Phase 1.2
================================================================================
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

try:
    import nibabel as nib
    from nilearn.image import resample_to_img, load_img
    from nilearn.datasets import load_mni152_template
    NIBABEL_AVAILABLE = True
    NILEARN_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    NILEARN_AVAILABLE = False
    print("WARNING: nibabel/nilearn not available")

# Configuration
BASE_DIR = Path("D:/discs")
ADNI_DIR = BASE_DIR / "ADNI"
OUTPUT_DIR = BASE_DIR / "project" / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
NORMALIZED_DIR = BASE_DIR / "project" / "data" / "adni" / "normalized"
NORMALIZED_DIR.mkdir(parents=True, exist_ok=True)

# Target space (MNI152 or match OASIS Talairach)
TARGET_SPACE = "MNI152"  # or "Talairach"
TARGET_VOXEL_SIZE = (1.0, 1.0, 1.0)  # mm
TARGET_SHAPE = (176, 208, 176)  # Match OASIS if using Talairach


class ADNIProcessor:
    """Process ADNI dataset for feature extraction"""
    
    def __init__(self, adni_dir: Path = ADNI_DIR):
        self.adni_dir = adni_dir
        self.nifti_files = []
        self.subject_ids = []
        self.clinical_data = None
        self.processed_files = []
        
    def find_all_nifti_files(self) -> List[Path]:
        """Find all NIfTI files in ADNI directory"""
        print("=" * 80)
        print("FINDING ALL ADNI NIFTI FILES")
        print("=" * 80)
        
        nifti_files = []
        for nii_file in self.adni_dir.rglob("*.nii"):
            nifti_files.append(nii_file)
        
        self.nifti_files = sorted(nifti_files)
        print(f"\nFound {len(self.nifti_files)} NIfTI files")
        
        # Extract subject IDs
        import re
        for nii_file in self.nifti_files:
            match = re.search(r'(\d{3}_S_\d{4})', nii_file.name)
            if match:
                self.subject_ids.append(match.group(1))
        
        self.subject_ids = sorted(list(set(self.subject_ids)))
        print(f"Found {len(self.subject_ids)} unique subjects")
        print(f"{'='*80}\n")
        
        return self.nifti_files
    
    def load_clinical_data(self, clinical_file: Optional[Path] = None) -> pd.DataFrame:
        """
        Load ADNI clinical data
        
        Note: This requires ADNI data access.
        Clinical data should be downloaded from ADNI website.
        Expected format: CSV with columns including Subject ID, Diagnosis, MMSE, Age, Gender, etc.
        """
        print("=" * 80)
        print("LOADING ADNI CLINICAL DATA")
        print("=" * 80)
        
        if clinical_file and clinical_file.exists():
            try:
                self.clinical_data = pd.read_csv(clinical_file)
                print(f"Loaded clinical data: {clinical_file}")
                print(f"Shape: {self.clinical_data.shape}")
                print(f"Columns: {list(self.clinical_data.columns)}")
            except Exception as e:
                print(f"ERROR loading clinical data: {e}")
                print("Creating placeholder clinical data structure...")
                self._create_placeholder_clinical_data()
        else:
            print("WARNING: Clinical data file not provided")
            print("Creating placeholder structure...")
            self._create_placeholder_clinical_data()
        
        print(f"{'='*80}\n")
        return self.clinical_data
    
    def _create_placeholder_clinical_data(self):
        """Create placeholder clinical data structure"""
        # Extract subject IDs from filenames
        import re
        subject_data = []
        for nii_file in self.nifti_files:
            match = re.search(r'(\d{3}_S_\d{4})', nii_file.name)
            if match:
                subject_id = match.group(1)
                subject_data.append({
                    'Subject_ID': subject_id,
                    'Diagnosis': 'Unknown',  # Needs to be filled from ADNI database
                    'MMSE': np.nan,
                    'Age': np.nan,
                    'Gender': 'Unknown',
                    'Education': np.nan
                })
        
        self.clinical_data = pd.DataFrame(subject_data)
        self.clinical_data = self.clinical_data.drop_duplicates(subset='Subject_ID')
        print("Placeholder clinical data created")
        print("NOTE: This needs to be replaced with actual ADNI clinical data!")
    
    def normalize_spatial(self, nii_file: Path, subject_id: str) -> Optional[Path]:
        """
        Normalize ADNI image to target space
        
        Options:
        1. MNI152 (standard, recommended)
        2. Talairach (to match OASIS)
        """
        try:
            # Load image
            img = load_img(str(nii_file))
            
            if TARGET_SPACE == "MNI152":
                # Load MNI152 template
                template = load_mni152_template(resolution=1)
                
                # Resample to template
                normalized_img = resample_to_img(
                    img, template, 
                    interpolation='continuous',
                    copy=True
                )
            else:
                # For Talairach, would need Talairach template
                # For now, just resample to target shape
                print(f"  [WARNING] Talairach normalization not fully implemented")
                print(f"  Using basic resampling to target shape")
                # This is a placeholder - would need proper Talairach template
                normalized_img = img
            
            # Save normalized image
            output_file = NORMALIZED_DIR / f"{subject_id}_{nii_file.stem}_normalized.nii.gz"
            normalized_img.to_filename(str(output_file))
            
            return output_file
            
        except Exception as e:
            print(f"  [ERROR] Normalization failed for {nii_file.name}: {e}")
            return None
    
    def normalize_all_images(self, max_files: Optional[int] = None):
        """Normalize all ADNI images"""
        print("=" * 80)
        print("SPATIAL NORMALIZATION")
        print("=" * 80)
        print(f"Target space: {TARGET_SPACE}")
        print(f"Target voxel size: {TARGET_VOXEL_SIZE} mm")
        print(f"Target shape: {TARGET_SHAPE}")
        print(f"{'='*80}\n")
        
        if not self.nifti_files:
            self.find_all_nifti_files()
        
        files_to_process = self.nifti_files[:max_files] if max_files else self.nifti_files
        
        print(f"Processing {len(files_to_process)} files...\n")
        
        for i, nii_file in enumerate(files_to_process, 1):
            # Extract subject ID
            import re
            match = re.search(r'(\d{3}_S_\d{4})', nii_file.name)
            subject_id = match.group(1) if match else "UNKNOWN"
            
            print(f"[{i}/{len(files_to_process)}] {nii_file.name}...")
            
            normalized_file = self.normalize_spatial(nii_file, subject_id)
            
            if normalized_file:
                self.processed_files.append({
                    'original_file': str(nii_file),
                    'normalized_file': str(normalized_file),
                    'subject_id': subject_id
                })
                print(f"  [OK] Saved to {normalized_file.name}")
            else:
                print(f"  [FAILED]")
        
        print(f"\n{'='*80}")
        print(f"Normalized: {len(self.processed_files)} files")
        print(f"{'='*80}\n")
        
        return self.processed_files
    
    def extract_features(self, normalized_file: Path) -> Dict:
        """Extract features from normalized ADNI image (same pipeline as OASIS)"""
        features = {}
        
        try:
            img = nib.load(str(normalized_file))
            data = img.get_fdata()
            
            # Basic MRI features (same as OASIS)
            features['mri_shape_x'] = data.shape[0]
            features['mri_shape_y'] = data.shape[1]
            features['mri_shape_z'] = data.shape[2]
            features['mri_total_voxels'] = int(data.size)
            features['mri_non_zero_voxels'] = int(np.count_nonzero(data))
            features['mri_brain_percentage'] = float(np.count_nonzero(data) / data.size * 100) if data.size > 0 else 0
            features['mri_mean_intensity'] = float(np.nanmean(data))
            features['mri_std_intensity'] = float(np.nanstd(data))
            features['mri_min_intensity'] = float(np.nanmin(data))
            features['mri_max_intensity'] = float(np.nanmax(data))
            
            # Voxel dimensions
            if hasattr(img.header, 'get_zooms'):
                zooms = img.header.get_zooms()[:3]
                features['mri_voxel_x'] = float(zooms[0])
                features['mri_voxel_y'] = float(zooms[1])
                features['mri_voxel_z'] = float(zooms[2])
            
            # TODO: Add FSL segmentation features
            # TODO: Add regional volume features
            # TODO: Add CNN embeddings
            
        except Exception as e:
            print(f"  [ERROR] Feature extraction failed: {e}")
        
        return features
    
    def process_all(self, clinical_file: Optional[Path] = None, max_files: Optional[int] = None):
        """Complete processing pipeline"""
        print("=" * 80)
        print("ADNI COMPLETE PROCESSING PIPELINE")
        print("=" * 80)
        
        # Step 1: Find all files
        self.find_all_nifti_files()
        
        # Step 2: Load clinical data
        self.load_clinical_data(clinical_file)
        
        # Step 3: Normalize images
        self.normalize_all_images(max_files)
        
        # Step 4: Extract features (to be implemented)
        print("\nFeature extraction from normalized images...")
        print("(This will be expanded to match OASIS-1 feature extraction pipeline)")
        
        # Save processing summary
        self.save_summary()
    
    def save_summary(self):
        """Save processing summary"""
        summary_file = OUTPUT_DIR / "adni_processing_summary.txt"
        
        with open(summary_file, 'w') as f:
            f.write("ADNI PROCESSING SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            f.write(f"Total NIfTI files: {len(self.nifti_files)}\n")
            f.write(f"Unique subjects: {len(self.subject_ids)}\n")
            f.write(f"Normalized files: {len(self.processed_files)}\n")
            f.write(f"Clinical data: {'Loaded' if self.clinical_data is not None else 'Not loaded'}\n")
            if self.clinical_data is not None:
                f.write(f"Clinical data shape: {self.clinical_data.shape}\n")
            f.write("\n" + "=" * 80 + "\n")
        
        print(f"Summary saved to: {summary_file}")


def main():
    """Main execution"""
    print("=" * 80)
    print("ADNI DATASET PROCESSING")
    print("=" * 80)
    print("\nThis script processes ADNI dataset for feature extraction")
    print("Part of Master Research Plan - Phase 1.2\n")
    
    processor = ADNIProcessor()
    
    # Process all
    # Note: Provide clinical data file path if available
    # clinical_file = Path("path/to/adni_clinical_data.csv")
    processor.process_all()
    
    print("\n" + "=" * 80)
    print("ADNI PROCESSING COMPLETE!")
    print("=" * 80)
    print("\nNEXT STEPS:")
    print("1. Download ADNI clinical data from ADNI website")
    print("2. Link clinical data using Subject IDs")
    print("3. Expand feature extraction to match OASIS-1 pipeline")
    print("4. Extract CNN embeddings using same encoder as OASIS-1")


if __name__ == "__main__":
    main()

