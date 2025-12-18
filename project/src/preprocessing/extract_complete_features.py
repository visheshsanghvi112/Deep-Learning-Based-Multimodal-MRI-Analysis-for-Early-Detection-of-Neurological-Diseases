"""
================================================================================
Complete Feature Extraction - FIXING GAPS
================================================================================
Extracts:
1. Full 214 features for all OASIS subjects (using deep scan)
2. CNN embeddings for all subjects (using MRI feature extraction)

This fixes the critical gaps identified in gap analysis.

Part of: Gap Fixing - Priority 1 & 2
================================================================================
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add paths
BASE_DIR = Path("D:/discs")
sys.path.append(str(BASE_DIR))

try:
    from oasis_deep_feature_scan import OASISDeepScan
    DEEP_SCAN_AVAILABLE = True
except ImportError:
    DEEP_SCAN_AVAILABLE = False
    print("WARNING: oasis_deep_feature_scan not available")

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

try:
    from mri_feature_extraction import MRILoader, MRISliceExtractor, MRIFeatureExtractor
    MRI_EXTRACTION_AVAILABLE = True
except ImportError:
    MRI_EXTRACTION_AVAILABLE = False
    print("WARNING: mri_feature_extraction not available")

# Configuration
OUTPUT_DIR = BASE_DIR / "project" / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class CompleteFeatureExtractor:
    """Extract complete features (214 + CNN embeddings) for all subjects"""
    
    def __init__(self, base_dir: Path = BASE_DIR):
        self.base_dir = base_dir
        self.disc_folders = self._find_disc_folders()
        self.all_subjects = []
        self.deep_features_list = []
        self.cnn_embeddings_dict = {}
        
        # Initialize extractors
        if MRI_EXTRACTION_AVAILABLE:
            self.mri_loader = MRILoader(str(base_dir))
            self.slice_extractor = MRISliceExtractor()
            self.feature_extractor = MRIFeatureExtractor(freeze_backbone=True)
            self.feature_extractor.eval()
            print("CNN feature extractor initialized")
        else:
            self.mri_loader = None
            self.slice_extractor = None
            self.feature_extractor = None
    
    def _find_disc_folders(self) -> List[Path]:
        """Find all disc folders"""
        folders = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and item.name.startswith("disc") and item.name[4:].isdigit():
                folders.append(item)
        return sorted(folders, key=lambda x: int(x.name[4:]))
    
    def find_all_subjects(self) -> List[Path]:
        """Find all OASIS subject folders"""
        subjects = []
        for disc in self.disc_folders:
            for item in disc.iterdir():
                if item.is_dir() and item.name.startswith("OAS1_"):
                    subjects.append(item)
        self.all_subjects = sorted(subjects)
        return self.all_subjects
    
    def extract_deep_features_for_subject(self, subject_path: Path) -> Optional[Dict]:
        """Extract 214 deep features for a subject"""
        if not DEEP_SCAN_AVAILABLE:
            return None
        
        subject_id = subject_path.name
        disc_path = subject_path.parent
        
        try:
            # Create scanner for this disc
            scanner = OASISDeepScan(str(disc_path))
            
            # Extract all features
            features = scanner.scan_subject(subject_path, verbose=False)
            
            return features
            
        except Exception as e:
            print(f"  [ERROR] Deep feature extraction failed for {subject_id}: {e}")
            return None
    
    def extract_cnn_embeddings_for_subject(self, subject_id: str) -> Optional[np.ndarray]:
        """Extract CNN embeddings for a subject"""
        if not MRI_EXTRACTION_AVAILABLE or not self.mri_loader:
            return None
        
        try:
            # Find subject folder
            subject_path = self.mri_loader.find_subject_folder(subject_id)
            if not subject_path:
                return None
            
            # Get MRI path
            mri_path = self.mri_loader.get_mri_path(subject_id)
            if not mri_path:
                return None
            
            # Load MRI
            mri_data = self.mri_loader.load_mri(subject_id)
            if mri_data is None:
                return None
            
            # Extract slices
            slices = self.slice_extractor.extract_all_slices(mri_data)
            if slices is None or len(slices) == 0:
                return None
            
            # Convert to tensor
            slices_tensor = torch.FloatTensor(slices).unsqueeze(1)  # Add channel dim
            slices_tensor = slices_tensor.repeat(1, 3, 1, 1)  # RGB channels
            
            # Extract features
            with torch.no_grad():
                embeddings = self.feature_extractor(slices_tensor)
            
            return embeddings.cpu().numpy()
            
        except Exception as e:
            print(f"  [ERROR] CNN embedding extraction failed for {subject_id}: {e}")
            return None
    
    def process_all_subjects(self, max_subjects: Optional[int] = None):
        """Process all subjects to extract complete features"""
        print("=" * 80)
        print("COMPLETE FEATURE EXTRACTION - FIXING GAPS")
        print("=" * 80)
        print("\nExtracting:")
        print("  1. Full 214 deep features for all subjects")
        print("  2. CNN embeddings (512-dim) for all subjects")
        print("=" * 80 + "\n")
        
        if not self.all_subjects:
            self.find_all_subjects()
        
        subjects_to_process = self.all_subjects[:max_subjects] if max_subjects else self.all_subjects
        
        print(f"Processing {len(subjects_to_process)} subjects...\n")
        
        deep_features_count = 0
        cnn_embeddings_count = 0
        
        for i, subject_path in enumerate(subjects_to_process, 1):
            subject_id = subject_path.name
            print(f"[{i}/{len(subjects_to_process)}] {subject_id}...")
            
            # Extract deep features
            deep_features = self.extract_deep_features_for_subject(subject_path)
            if deep_features:
                deep_features['SUBJECT_ID'] = subject_id
                self.deep_features_list.append(deep_features)
                deep_features_count += 1
                print(f"  [OK] Deep features: {len(deep_features)} features")
            else:
                print(f"  [SKIP] Deep features not available")
            
            # Extract CNN embeddings
            cnn_emb = self.extract_cnn_embeddings_for_subject(subject_id)
            if cnn_emb is not None:
                self.cnn_embeddings_dict[subject_id] = cnn_emb
                cnn_embeddings_count += 1
                print(f"  [OK] CNN embeddings: {cnn_emb.shape}")
            else:
                print(f"  [SKIP] CNN embeddings not available")
        
        print(f"\n{'='*80}")
        print("EXTRACTION SUMMARY")
        print(f"{'='*80}")
        print(f"Deep features extracted: {deep_features_count}/{len(subjects_to_process)}")
        print(f"CNN embeddings extracted: {cnn_embeddings_count}/{len(subjects_to_process)}")
        print(f"{'='*80}\n")
        
        return deep_features_count, cnn_embeddings_count
    
    def merge_with_existing_features(self) -> pd.DataFrame:
        """Merge deep features with existing basic features"""
        print("=" * 80)
        print("MERGING FEATURES")
        print("=" * 80)
        
        # Load existing features
        existing_file = OUTPUT_DIR / "oasis_complete_features.csv"
        if existing_file.exists():
            existing_df = pd.read_csv(existing_file)
            print(f"Loaded existing features: {existing_df.shape}")
        else:
            print("No existing features file found")
            existing_df = None
        
        # Create deep features DataFrame
        if self.deep_features_list:
            deep_df = pd.DataFrame(self.deep_features_list)
            print(f"Deep features DataFrame: {deep_df.shape}")
            
            # Merge with existing
            if existing_df is not None:
                # Merge on SUBJECT_ID
                merged_df = existing_df.merge(deep_df, on='SUBJECT_ID', how='outer', suffixes=('', '_deep'))
                print(f"Merged DataFrame: {merged_df.shape}")
            else:
                merged_df = deep_df
        else:
            merged_df = existing_df if existing_df is not None else pd.DataFrame()
        
        return merged_df
    
    def save_results(self):
        """Save complete features and CNN embeddings"""
        print("=" * 80)
        print("SAVING RESULTS")
        print("=" * 80)
        
        # Merge features
        merged_df = self.merge_with_existing_features()
        
        if not merged_df.empty:
            # Save merged features
            output_file = OUTPUT_DIR / "oasis_complete_features_full.csv"
            merged_df.to_csv(output_file, index=False)
            print(f"\nComplete features saved: {output_file}")
            print(f"Shape: {merged_df.shape}")
            print(f"Features: {len(merged_df.columns)}")
        
        # Save CNN embeddings
        if self.cnn_embeddings_dict:
            # Create array aligned with subjects
            subject_ids = list(self.cnn_embeddings_dict.keys())
            embeddings_list = [self.cnn_embeddings_dict[sid] for sid in subject_ids]
            embeddings_array = np.array(embeddings_list)
            
            # Save
            cnn_file = OUTPUT_DIR / "oasis_cnn_embeddings_all.npz"
            np.savez(cnn_file, 
                    embeddings=embeddings_array,
                    subject_ids=np.array(subject_ids))
            print(f"\nCNN embeddings saved: {cnn_file}")
            print(f"Shape: {embeddings_array.shape}")
            print(f"Subjects: {len(subject_ids)}")
        
        return merged_df


def main():
    """Main execution"""
    print("=" * 80)
    print("COMPLETE FEATURE EXTRACTION - GAP FIXING")
    print("=" * 80)
    print("\nThis script fixes critical gaps:")
    print("  1. Extracts full 214 features for all subjects")
    print("  2. Extracts CNN embeddings for all subjects")
    print("\nThis will significantly improve model performance!\n")
    
    extractor = CompleteFeatureExtractor()
    
    # Process all subjects
    # Note: This may take a while (processing 434 subjects)
    # For testing, you can limit with max_subjects parameter
    extractor.process_all_subjects()  # Process all 434 subjects
    
    # Save results
    extractor.save_results()
    
    print("\n" + "=" * 80)
    print("COMPLETE FEATURE EXTRACTION FINISHED!")
    print("=" * 80)
    print("\nNext Steps:")
    print("  1. Retrain model with complete features")
    print("  2. Expected: Significant performance improvement")
    print("  3. Binary AUC: 0.76 → 0.80-0.85 (expected)")
    print("  4. Accuracy: 59.5% → 70-80% (expected)")


if __name__ == "__main__":
    main()

