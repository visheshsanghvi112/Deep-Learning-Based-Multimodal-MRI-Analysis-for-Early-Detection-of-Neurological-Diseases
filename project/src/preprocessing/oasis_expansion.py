"""
================================================================================
OASIS-1 Dataset Expansion Pipeline
================================================================================
Extends feature extraction to all disc folders (disc1-disc12)
Processes all 436 subjects in OASIS-1 dataset

Part of: Master Research Plan - Phase 1.1
================================================================================
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional
import pandas as pd
import numpy as np
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path to import existing modules
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

try:
    from oasis_deep_feature_scan import OASISDeepScan, XMLMetadataExtractor
    from oasis_data_exploration import find_all_subjects, parse_subject_txt, parse_fseg_txt
    DEEP_SCAN_AVAILABLE = True
except ImportError:
    DEEP_SCAN_AVAILABLE = False
    print("WARNING: oasis_deep_feature_scan not available")

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    print("WARNING: nibabel not available")

# Configuration
BASE_DIR = Path("D:/discs")
DISC_PATTERN = "disc*"
OUTPUT_DIR = BASE_DIR / "project" / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


class OASISExpansion:
    """Extend OASIS-1 feature extraction to all disc folders"""
    
    def __init__(self, base_dir: Path = BASE_DIR):
        self.base_dir = base_dir
        self.disc_folders = self._find_disc_folders()
        self.all_subjects = []
        self.processed_subjects = []
        self.failed_subjects = []
        self.features_list = []
        
    def _find_disc_folders(self) -> List[Path]:
        """Find all disc folders"""
        folders = []
        for item in self.base_dir.iterdir():
            if item.is_dir() and item.name.startswith("disc") and item.name[4:].isdigit():
                folders.append(item)
        return sorted(folders, key=lambda x: int(x.name[4:]))
    
    def find_all_subjects(self) -> List[Path]:
        """Find all OASIS subject folders across all discs"""
        print("=" * 80)
        print("FINDING ALL OASIS SUBJECTS")
        print("=" * 80)
        
        subjects = []
        for disc in self.disc_folders:
            print(f"\nScanning {disc.name}...")
            for item in disc.iterdir():
                if item.is_dir() and item.name.startswith("OAS1_"):
                    subjects.append(item)
                    print(f"  Found: {item.name}")
        
        self.all_subjects = sorted(subjects)
        print(f"\n{'='*80}")
        print(f"TOTAL SUBJECTS FOUND: {len(self.all_subjects)}")
        print(f"Across {len(self.disc_folders)} disc folders")
        print(f"{'='*80}\n")
        
        return self.all_subjects
    
    def process_subject(self, subject_path: Path) -> Optional[Dict]:
        """Process a single subject and extract features"""
        subject_id = subject_path.name
        
        try:
            # Check if subject has required files
            txt_file = subject_path / f"{subject_id}.txt"
            xml_file = subject_path / f"{subject_id}.xml"
            
            if not txt_file.exists():
                print(f"  [SKIP] {subject_id}: Missing .txt file")
                return None
            
            # Parse clinical data
            clinical_data = parse_subject_txt(txt_file)
            if not clinical_data:
                print(f"  [SKIP] {subject_id}: Could not parse clinical data")
                return None
            
            # Find MRI file (T88_111 space)
            mri_path = self._find_mri_file(subject_path, subject_id)
            if not mri_path:
                print(f"  [SKIP] {subject_id}: MRI file not found")
                return None
            
            # Find FSL segmentation
            fseg_path = self._find_fseg_file(subject_path, subject_id)
            
            # Extract features
            features = {
                'SUBJECT_ID': subject_id,
                'dataset': 'OASIS-1',
                'disc_folder': subject_path.parent.name
            }
            
            # Add clinical data
            features.update(clinical_data)
            
            # Add MRI-based features if available
            if NIBABEL_AVAILABLE and mri_path:
                mri_features = self._extract_mri_features(mri_path)
                features.update(mri_features)
            
            # Add FSL segmentation features if available
            if fseg_path:
                fseg_features = parse_fseg_txt(fseg_path)
                if fseg_features:
                    features.update(fseg_features)
            
            # Add XML metadata if available
            if xml_file.exists() and DEEP_SCAN_AVAILABLE:
                try:
                    xml_extractor = XMLMetadataExtractor()
                    xml_data = xml_extractor.parse_xml(xml_file)
                    # Add relevant XML features
                    if 'acquisition_params' in xml_data:
                        features.update(xml_data['acquisition_params'])
                except Exception as e:
                    pass  # XML parsing is optional
            
            return features
            
        except Exception as e:
            print(f"  [ERROR] {subject_id}: {str(e)}")
            self.failed_subjects.append((subject_id, str(e)))
            return None
    
    def _find_mri_file(self, subject_path: Path, subject_id: str) -> Optional[Path]:
        """Find T88_111 MRI file"""
        # Try T88_111 path
        t88_path = subject_path / "PROCESSED" / "MPRAGE" / "T88_111"
        if t88_path.exists():
            # Look for masked file (preferred)
            patterns = [
                f"{subject_id}_mpr_n4_anon_111_t88_masked_gfc.hdr",
                f"{subject_id}_mpr_n3_anon_111_t88_masked_gfc.hdr",
                f"{subject_id}_mpr_n4_anon_111_t88_masked.hdr",
                f"{subject_id}_mpr_n3_anon_111_t88_masked.hdr"
            ]
            for pattern in patterns:
                mri_file = t88_path / pattern
                if mri_file.exists():
                    return mri_file
        
        return None
    
    def _find_fseg_file(self, subject_path: Path, subject_id: str) -> Optional[Path]:
        """Find FSL segmentation file"""
        fseg_path = subject_path / "FSL_SEG"
        if fseg_path.exists():
            patterns = [
                f"{subject_id}_fseg.hdr",
                f"{subject_id}_fseg.txt"
            ]
            for pattern in patterns:
                fseg_file = fseg_path / pattern
                if fseg_file.exists():
                    return fseg_file
        return None
    
    def _extract_mri_features(self, mri_path: Path) -> Dict:
        """Extract basic MRI features"""
        features = {}
        try:
            img = nib.load(str(mri_path))
            data = img.get_fdata()
            
            # Basic statistics
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
            
        except Exception as e:
            print(f"    [WARNING] Could not extract MRI features: {e}")
        
        return features
    
    def process_all_subjects(self, max_subjects: Optional[int] = None):
        """Process all subjects across all disc folders"""
        print("=" * 80)
        print("PROCESSING ALL OASIS SUBJECTS")
        print("=" * 80)
        
        if not self.all_subjects:
            self.find_all_subjects()
        
        subjects_to_process = self.all_subjects[:max_subjects] if max_subjects else self.all_subjects
        
        print(f"\nProcessing {len(subjects_to_process)} subjects...")
        print(f"{'='*80}\n")
        
        for i, subject_path in enumerate(subjects_to_process, 1):
            subject_id = subject_path.name
            print(f"[{i}/{len(subjects_to_process)}] Processing {subject_id}...")
            
            features = self.process_subject(subject_path)
            
            if features:
                self.features_list.append(features)
                self.processed_subjects.append(subject_id)
                print(f"  [OK] {subject_id}: Features extracted")
            else:
                print(f"  [FAILED] {subject_id}")
        
        print(f"\n{'='*80}")
        print("PROCESSING COMPLETE")
        print(f"{'='*80}")
        print(f"Processed: {len(self.processed_subjects)}")
        print(f"Failed: {len(self.failed_subjects)}")
        print(f"Total: {len(subjects_to_process)}")
        
        return self.features_list
    
    def save_results(self):
        """Save extracted features to CSV"""
        if not self.features_list:
            print("No features to save!")
            return
        
        # Convert to DataFrame
        df = pd.DataFrame(self.features_list)
        
        # Save
        output_file = OUTPUT_DIR / "oasis_complete_features.csv"
        df.to_csv(output_file, index=False)
        print(f"\nFeatures saved to: {output_file}")
        print(f"Shape: {df.shape}")
        
        # Save summary
        summary = {
            'total_subjects': len(self.all_subjects),
            'processed_subjects': len(self.processed_subjects),
            'failed_subjects': len(self.failed_subjects),
            'disc_folders': len(self.disc_folders),
            'features_extracted': len(df.columns) if not df.empty else 0
        }
        
        summary_file = OUTPUT_DIR / "oasis_expansion_summary.txt"
        with open(summary_file, 'w') as f:
            f.write("OASIS-1 EXPANSION SUMMARY\n")
            f.write("=" * 80 + "\n\n")
            for key, value in summary.items():
                f.write(f"{key}: {value}\n")
            f.write("\n" + "=" * 80 + "\n")
            f.write("FAILED SUBJECTS:\n")
            for subject_id, error in self.failed_subjects:
                f.write(f"  {subject_id}: {error}\n")
        
        print(f"Summary saved to: {summary_file}")
        
        return df


def main():
    """Main execution"""
    print("=" * 80)
    print("OASIS-1 DATASET EXPANSION")
    print("=" * 80)
    print("\nThis script extends feature extraction to all disc folders")
    print("Part of Master Research Plan - Phase 1.1\n")
    
    expander = OASISExpansion()
    
    # Find all subjects
    expander.find_all_subjects()
    
    # Process all subjects
    # For testing, you can limit with max_subjects parameter
    expander.process_all_subjects()
    
    # Save results
    df = expander.save_results()
    
    print("\n" + "=" * 80)
    print("EXPANSION COMPLETE!")
    print("=" * 80)


if __name__ == "__main__":
    main()

