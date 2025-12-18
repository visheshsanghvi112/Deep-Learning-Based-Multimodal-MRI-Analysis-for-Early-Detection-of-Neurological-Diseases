"""
================================================================================
OASIS Dataset Deep Exploration
================================================================================
Purpose: Discover ALL potentially useful features before finalizing the feature set

This script will:
1. Analyze clinical metadata completeness and missingness patterns
2. Extract FSL segmentation volumes (GM, WM, CSF, brain percentage)
3. Compute MRI intensity statistics (mean, std, entropy)
4. Check subject-level scan counts
5. Analyze demographic stratification vs CDR classes
6. Generate a comprehensive summary CSV

Output: oasis_comprehensive_features.csv
================================================================================
"""

import os
import re
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import warnings

import numpy as np

# Optional imports
try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("WARNING: pandas not installed. Run: uv pip install pandas")

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    print("WARNING: nibabel not installed. Run: uv pip install nibabel")

try:
    from scipy.stats import entropy as scipy_entropy
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("WARNING: scipy not installed. Run: uv pip install scipy")


# ==============================================================================
# CONFIGURATION
# ==============================================================================

BASE_DIR = "D:/"
DISC_PATTERN = "disc*"
OUTPUT_CSV = "oasis_comprehensive_features.csv"
OUTPUT_REPORT = "oasis_exploration_report.txt"


# ==============================================================================
# HELPER FUNCTIONS
# ==============================================================================

def find_all_subjects(base_dir: str) -> List[Path]:
    """Find all subject folders across all disc directories."""
    subjects = []
    disc_pattern = os.path.join(base_dir, DISC_PATTERN)
    
    for disc_folder in sorted(glob.glob(disc_pattern)):
        if os.path.isdir(disc_folder):
            for item in os.listdir(disc_folder):
                if item.startswith("OAS1_") and os.path.isdir(os.path.join(disc_folder, item)):
                    subjects.append(Path(disc_folder) / item)
    
    return sorted(subjects)


def parse_subject_txt(txt_path: Path) -> Dict[str, Any]:
    """
    Parse the subject .txt file for clinical and demographic data.
    
    Returns dict with keys: SESSION_ID, AGE, M/F, HAND, EDUC, SES, CDR, MMSE, eTIV, ASF, nWBV
    """
    metadata = {}
    scan_list = []
    
    if not txt_path.exists():
        return metadata
    
    try:
        with open(txt_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                    
                if ':' in line:
                    parts = line.split(':', 1)
                    if len(parts) == 2:
                        key = parts[0].strip().upper()
                        value = parts[1].strip()
                        
                        # Normalize key names
                        if key == 'M/F':
                            key = 'GENDER'
                        elif key == 'SESSION ID':
                            key = 'SESSION_ID'
                        
                        # Try to convert to numeric
                        if value == '':
                            metadata[key] = None
                        else:
                            try:
                                # Handle special float cases
                                metadata[key] = float(value)
                            except ValueError:
                                metadata[key] = value
                
                # Detect MPRAGE scan lines (e.g., "mpr-1      MPRAGE")
                if line.startswith('mpr-') and 'MPRAGE' in line:
                    scan_match = re.match(r'(mpr-\d+)', line)
                    if scan_match:
                        scan_list.append(scan_match.group(1))
        
        metadata['SCAN_COUNT'] = len(scan_list)
        metadata['SCANS'] = scan_list
        
    except Exception as e:
        print(f"[WARNING] Error parsing {txt_path}: {e}")
    
    return metadata


def parse_fseg_txt(fseg_txt_path: Path) -> Dict[str, Any]:
    """
    Parse FSL FAST segmentation log file for tissue volumes.
    
    The file contains lines like:
    Class:		CSF		tissue 1	tissue 2	brain percentage
    Volumes:	428136.9  	740840.2  	498315.1  	0.743214
    
    tissue 1 = Gray Matter
    tissue 2 = White Matter
    """
    seg_data = {
        'CSF_VOLUME': None,
        'GM_VOLUME': None,  # Gray Matter (tissue 1)
        'WM_VOLUME': None,  # White Matter (tissue 2)
        'BRAIN_PERCENTAGE': None,
        'TOTAL_BRAIN_VOLUME': None,
    }
    
    if not fseg_txt_path.exists():
        return seg_data
    
    try:
        with open(fseg_txt_path, 'r') as f:
            content = f.read()
            
            # Find the Volumes line
            # Pattern: Volumes: <CSF> <GM> <WM> <brain_pct>
            vol_match = re.search(
                r'Volumes:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',
                content
            )
            
            if vol_match:
                seg_data['CSF_VOLUME'] = float(vol_match.group(1))
                seg_data['GM_VOLUME'] = float(vol_match.group(2))
                seg_data['WM_VOLUME'] = float(vol_match.group(3))
                seg_data['BRAIN_PERCENTAGE'] = float(vol_match.group(4))
                
                # Total brain = GM + WM (excluding CSF)
                seg_data['TOTAL_BRAIN_VOLUME'] = seg_data['GM_VOLUME'] + seg_data['WM_VOLUME']
            
            # Also try to extract image dimensions
            dim_match = re.search(r'Imagesize\s*:\s*(\d+)\s*x\s*(\d+)\s*x\s*(\d+)', content)
            if dim_match:
                seg_data['SEG_DIMS'] = f"{dim_match.group(1)}x{dim_match.group(2)}x{dim_match.group(3)}"
                
    except Exception as e:
        print(f"[WARNING] Error parsing {fseg_txt_path}: {e}")
    
    return seg_data


def compute_image_statistics(img_path: Path) -> Dict[str, Any]:
    """
    Compute intensity statistics from MRI volume.
    
    Returns: mean, std, min, max, entropy, non-zero voxel count
    """
    stats = {
        'IMG_MEAN': None,
        'IMG_STD': None,
        'IMG_MIN': None,
        'IMG_MAX': None,
        'IMG_ENTROPY': None,
        'BRAIN_VOXELS': None,
        'TOTAL_VOXELS': None,
        'IMG_DIMS': None,
    }
    
    if not NIBABEL_AVAILABLE:
        return stats
    
    if not img_path.exists():
        return stats
    
    try:
        img = nib.load(str(img_path))
        data = img.get_fdata().astype(np.float32)
        
        # Squeeze singleton dimensions
        if data.ndim == 4 and data.shape[-1] == 1:
            data = data.squeeze(-1)
        
        stats['IMG_DIMS'] = f"{data.shape[0]}x{data.shape[1]}x{data.shape[2]}"
        stats['TOTAL_VOXELS'] = int(data.size)
        
        # Create brain mask (non-zero voxels)
        brain_mask = data > 0
        brain_voxels = data[brain_mask]
        
        stats['BRAIN_VOXELS'] = int(brain_mask.sum())
        
        if len(brain_voxels) > 0:
            stats['IMG_MEAN'] = float(brain_voxels.mean())
            stats['IMG_STD'] = float(brain_voxels.std())
            stats['IMG_MIN'] = float(brain_voxels.min())
            stats['IMG_MAX'] = float(brain_voxels.max())
            
            # Compute entropy (measure of intensity distribution complexity)
            if SCIPY_AVAILABLE:
                # Create histogram for entropy calculation
                hist, _ = np.histogram(brain_voxels, bins=256, density=True)
                # Filter out zero bins to avoid log(0)
                hist = hist[hist > 0]
                stats['IMG_ENTROPY'] = float(scipy_entropy(hist, base=2))
        
    except Exception as e:
        print(f"[WARNING] Error computing stats for {img_path}: {e}")
    
    return stats


def compute_scan_variability(subject_folder: Path) -> Dict[str, Any]:
    """
    Analyze variability across multiple MPRAGE scans for a subject.
    
    This can indicate scan quality and subject movement.
    """
    variability = {
        'RAW_SCAN_COUNT': 0,
        'SCAN_MEAN_VAR': None,  # Variance of mean intensities across scans
        'SCAN_STD_VAR': None,   # Variance of stds across scans
    }
    
    if not NIBABEL_AVAILABLE:
        return variability
    
    raw_folder = subject_folder / "RAW"
    if not raw_folder.exists():
        return variability
    
    # Find all raw MPRAGE scans
    scan_files = list(raw_folder.glob("*_mpr-*_anon.hdr"))
    variability['RAW_SCAN_COUNT'] = len(scan_files)
    
    if len(scan_files) < 2:
        return variability
    
    means = []
    stds = []
    
    for scan_path in scan_files:
        try:
            img = nib.load(str(scan_path))
            data = img.get_fdata().astype(np.float32)
            
            if data.ndim == 4:
                data = data.squeeze()
            
            brain_voxels = data[data > 0]
            if len(brain_voxels) > 0:
                means.append(brain_voxels.mean())
                stds.append(brain_voxels.std())
        except:
            continue
    
    if len(means) >= 2:
        variability['SCAN_MEAN_VAR'] = float(np.var(means))
        variability['SCAN_STD_VAR'] = float(np.var(stds))
    
    return variability


# ==============================================================================
# MAIN EXPLORATION CLASS
# ==============================================================================

class OASISDataExplorer:
    """
    Comprehensive data explorer for OASIS-1 dataset.
    """
    
    def __init__(self, base_dir: str = BASE_DIR):
        self.base_dir = base_dir
        self.subjects = find_all_subjects(base_dir)
        print(f"[Explorer] Found {len(self.subjects)} subjects")
        
        # Storage for all extracted data
        self.subject_data = []
        
    def explore_all_subjects(self, compute_image_stats: bool = True):
        """
        Extract all available features for all subjects.
        """
        print("\n" + "="*70)
        print("EXPLORING ALL SUBJECTS")
        print("="*70 + "\n")
        
        for i, subject_folder in enumerate(self.subjects):
            subject_id = subject_folder.name
            print(f"[{i+1}/{len(self.subjects)}] Processing {subject_id}...", end=" ")
            
            record = {'SUBJECT_ID': subject_id}
            
            # 1. Parse clinical metadata from .txt file
            txt_path = subject_folder / f"{subject_id}.txt"
            clinical_data = parse_subject_txt(txt_path)
            record.update(clinical_data)
            
            # 2. Parse segmentation volumes from FSL_SEG
            fseg_folder = subject_folder / "FSL_SEG"
            fseg_txt_files = list(fseg_folder.glob("*_fseg.txt")) if fseg_folder.exists() else []
            if fseg_txt_files:
                seg_data = parse_fseg_txt(fseg_txt_files[0])
                record.update(seg_data)
            
            # 3. Compute MRI intensity statistics
            if compute_image_stats:
                t88_folder = subject_folder / "PROCESSED" / "MPRAGE" / "T88_111"
                t88_files = list(t88_folder.glob("*_t88_masked_gfc.hdr")) if t88_folder.exists() else []
                if t88_files:
                    img_stats = compute_image_statistics(t88_files[0])
                    record.update(img_stats)
            
            # 4. Compute scan variability
            variability = compute_scan_variability(subject_folder)
            record.update(variability)
            
            self.subject_data.append(record)
            print("✓")
        
        print(f"\n[Explorer] Completed exploration of {len(self.subject_data)} subjects")
    
    def generate_dataframe(self) -> 'pd.DataFrame':
        """Convert subject data to pandas DataFrame."""
        if not PANDAS_AVAILABLE:
            raise ImportError("pandas required for DataFrame generation")
        
        df = pd.DataFrame(self.subject_data)
        
        # Reorder columns for clarity
        priority_cols = [
            'SUBJECT_ID', 'AGE', 'GENDER', 'HAND', 'EDUC', 'SES', 
            'CDR', 'MMSE', 'eTIV', 'ASF', 'nWBV',
            'SCAN_COUNT', 'RAW_SCAN_COUNT',
            'CSF_VOLUME', 'GM_VOLUME', 'WM_VOLUME', 'TOTAL_BRAIN_VOLUME', 'BRAIN_PERCENTAGE',
            'IMG_MEAN', 'IMG_STD', 'IMG_MIN', 'IMG_MAX', 'IMG_ENTROPY',
            'BRAIN_VOXELS', 'TOTAL_VOXELS',
            'SCAN_MEAN_VAR', 'SCAN_STD_VAR'
        ]
        
        # Get columns that exist
        existing_cols = [c for c in priority_cols if c in df.columns]
        other_cols = [c for c in df.columns if c not in priority_cols]
        
        df = df[existing_cols + other_cols]
        
        return df
    
    def analyze_missingness(self, df: 'pd.DataFrame') -> str:
        """Analyze missing data patterns."""
        report = []
        report.append("\n" + "="*70)
        report.append("CLINICAL METADATA COMPLETENESS ANALYSIS")
        report.append("="*70)
        
        key_fields = ['AGE', 'GENDER', 'MMSE', 'CDR', 'EDUC', 'SES', 'HAND', 'eTIV', 'ASF', 'nWBV']
        
        report.append(f"\nTotal subjects: {len(df)}")
        report.append("\nMissingness by field:")
        report.append("-" * 50)
        
        for field in key_fields:
            if field in df.columns:
                missing = df[field].isna().sum()
                valid = len(df) - missing
                pct = (valid / len(df)) * 100
                report.append(f"  {field:15s}: {valid:3d}/{len(df)} valid ({pct:5.1f}%)")
            else:
                report.append(f"  {field:15s}: NOT FOUND in data")
        
        # CDR distribution
        report.append("\n" + "-"*50)
        report.append("CDR DISTRIBUTION (Critical for classification):")
        report.append("-"*50)
        
        if 'CDR' in df.columns:
            cdr_counts = df['CDR'].value_counts(dropna=False).sort_index()
            for cdr_val, count in cdr_counts.items():
                label = "Missing" if pd.isna(cdr_val) else f"CDR={cdr_val}"
                report.append(f"  {label:15s}: {count:3d} subjects")
            
            # Count CDR=0 and CDR=0.5 specifically
            cdr_0 = (df['CDR'] == 0).sum()
            cdr_05 = (df['CDR'] == 0.5).sum()
            cdr_valid = df['CDR'].notna().sum()
            
            report.append(f"\n  CDR=0 (Normal):     {cdr_0} subjects")
            report.append(f"  CDR=0.5 (Very Mild): {cdr_05} subjects")
            report.append(f"  CDR valid total:    {cdr_valid} subjects")
            report.append(f"\n  → Usable for CDR=0 vs CDR=0.5 classification: {cdr_0 + cdr_05} subjects")
        
        return "\n".join(report)
    
    def analyze_segmentation_features(self, df: 'pd.DataFrame') -> str:
        """Analyze FSL segmentation derived features."""
        report = []
        report.append("\n" + "="*70)
        report.append("SEGMENTATION FEATURES ANALYSIS (FSL FAST)")
        report.append("="*70)
        
        seg_fields = ['CSF_VOLUME', 'GM_VOLUME', 'WM_VOLUME', 'TOTAL_BRAIN_VOLUME', 'BRAIN_PERCENTAGE']
        
        report.append("\nAvailability:")
        for field in seg_fields:
            if field in df.columns:
                valid = df[field].notna().sum()
                report.append(f"  {field:20s}: {valid}/{len(df)} subjects")
        
        report.append("\nStatistics (if available):")
        report.append("-"*50)
        
        for field in seg_fields:
            if field in df.columns and df[field].notna().any():
                vals = df[field].dropna()
                report.append(f"\n  {field}:")
                report.append(f"    Mean: {vals.mean():12.2f}")
                report.append(f"    Std:  {vals.std():12.2f}")
                report.append(f"    Min:  {vals.min():12.2f}")
                report.append(f"    Max:  {vals.max():12.2f}")
        
        # Compute derived ratios
        report.append("\n" + "-"*50)
        report.append("DERIVED NEUROANATOMICAL BIOMARKERS:")
        report.append("-"*50)
        
        if all(f in df.columns for f in ['GM_VOLUME', 'WM_VOLUME', 'CSF_VOLUME']):
            df_valid = df[['GM_VOLUME', 'WM_VOLUME', 'CSF_VOLUME']].dropna()
            if len(df_valid) > 0:
                # GM/WM ratio (indicator of atrophy patterns)
                gm_wm_ratio = df_valid['GM_VOLUME'] / df_valid['WM_VOLUME']
                report.append(f"\n  GM/WM Ratio (atrophy indicator):")
                report.append(f"    Mean: {gm_wm_ratio.mean():.3f}")
                report.append(f"    Std:  {gm_wm_ratio.std():.3f}")
                
                # CSF fraction (ventricular enlargement indicator)
                total_vol = df_valid['GM_VOLUME'] + df_valid['WM_VOLUME'] + df_valid['CSF_VOLUME']
                csf_fraction = df_valid['CSF_VOLUME'] / total_vol
                report.append(f"\n  CSF Fraction (ventricle enlargement):")
                report.append(f"    Mean: {csf_fraction.mean():.3f}")
                report.append(f"    Std:  {csf_fraction.std():.3f}")
        
        return "\n".join(report)
    
    def analyze_mri_quality(self, df: 'pd.DataFrame') -> str:
        """Analyze MRI intensity statistics as quality indicators."""
        report = []
        report.append("\n" + "="*70)
        report.append("MRI INTENSITY STATISTICS (Quality Indicators)")
        report.append("="*70)
        
        img_fields = ['IMG_MEAN', 'IMG_STD', 'IMG_MIN', 'IMG_MAX', 'IMG_ENTROPY']
        
        for field in img_fields:
            if field in df.columns and df[field].notna().any():
                vals = df[field].dropna()
                report.append(f"\n  {field}:")
                report.append(f"    Mean:   {vals.mean():10.2f}")
                report.append(f"    Std:    {vals.std():10.2f}")
                report.append(f"    Range:  [{vals.min():.2f}, {vals.max():.2f}]")
        
        # Scan variability analysis
        report.append("\n" + "-"*50)
        report.append("SCAN VARIABILITY (Multi-scan quality indicator):")
        report.append("-"*50)
        
        if 'RAW_SCAN_COUNT' in df.columns:
            scan_counts = df['RAW_SCAN_COUNT'].value_counts().sort_index()
            report.append("\n  Number of raw MPRAGE scans per subject:")
            for count, n_subj in scan_counts.items():
                report.append(f"    {count} scans: {n_subj} subjects")
        
        if 'SCAN_MEAN_VAR' in df.columns and df['SCAN_MEAN_VAR'].notna().any():
            report.append("\n  Inter-scan intensity variance (lower = more consistent):")
            vals = df['SCAN_MEAN_VAR'].dropna()
            report.append(f"    Mean variance: {vals.mean():.2f}")
            report.append(f"    Max variance:  {vals.max():.2f} (potential quality issue)")
        
        return "\n".join(report)
    
    def analyze_demographics_vs_cdr(self, df: 'pd.DataFrame') -> str:
        """Analyze demographic stratification vs CDR classes."""
        report = []
        report.append("\n" + "="*70)
        report.append("DEMOGRAPHIC STRATIFICATION VS CDR (Confounder Analysis)")
        report.append("="*70)
        
        if 'CDR' not in df.columns:
            report.append("\n[WARNING] CDR column not found!")
            return "\n".join(report)
        
        # Filter to CDR=0 and CDR=0.5 for comparison
        df_cdr0 = df[df['CDR'] == 0]
        df_cdr05 = df[df['CDR'] == 0.5]
        
        report.append(f"\nComparing CDR=0 (n={len(df_cdr0)}) vs CDR=0.5 (n={len(df_cdr05)})")
        report.append("-"*50)
        
        # Age analysis
        if 'AGE' in df.columns:
            age_0 = df_cdr0['AGE'].dropna()
            age_05 = df_cdr05['AGE'].dropna()
            
            report.append("\n  AGE Distribution:")
            if len(age_0) > 0:
                report.append(f"    CDR=0:   Mean={age_0.mean():.1f}, Std={age_0.std():.1f}, Range=[{age_0.min():.0f}-{age_0.max():.0f}]")
            if len(age_05) > 0:
                report.append(f"    CDR=0.5: Mean={age_05.mean():.1f}, Std={age_05.std():.1f}, Range=[{age_05.min():.0f}-{age_05.max():.0f}]")
            
            if len(age_0) > 0 and len(age_05) > 0:
                age_diff = age_05.mean() - age_0.mean()
                if abs(age_diff) > 5:
                    report.append(f"    ⚠️  WARNING: Age difference of {age_diff:.1f} years - potential confounder!")
                else:
                    report.append(f"    ✓ Age difference acceptable ({age_diff:.1f} years)")
        
        # Gender analysis
        if 'GENDER' in df.columns:
            report.append("\n  GENDER Distribution:")
            for cdr_val, cdr_df in [("CDR=0", df_cdr0), ("CDR=0.5", df_cdr05)]:
                gender_counts = cdr_df['GENDER'].value_counts()
                report.append(f"    {cdr_val}: {dict(gender_counts)}")
        
        # Education analysis
        if 'EDUC' in df.columns:
            educ_0 = df_cdr0['EDUC'].dropna()
            educ_05 = df_cdr05['EDUC'].dropna()
            
            report.append("\n  EDUCATION Distribution:")
            if len(educ_0) > 0:
                report.append(f"    CDR=0:   Mean={educ_0.mean():.2f}, n={len(educ_0)}")
            if len(educ_05) > 0:
                report.append(f"    CDR=0.5: Mean={educ_05.mean():.2f}, n={len(educ_05)}")
        
        # SES analysis
        if 'SES' in df.columns:
            ses_0 = df_cdr0['SES'].dropna()
            ses_05 = df_cdr05['SES'].dropna()
            
            report.append("\n  SOCIOECONOMIC STATUS Distribution:")
            if len(ses_0) > 0:
                report.append(f"    CDR=0:   Mean={ses_0.mean():.2f}, n={len(ses_0)}")
            if len(ses_05) > 0:
                report.append(f"    CDR=0.5: Mean={ses_05.mean():.2f}, n={len(ses_05)}")
        
        # Brain volume analysis
        if 'nWBV' in df.columns:
            nwbv_0 = df_cdr0['nWBV'].dropna()
            nwbv_05 = df_cdr05['nWBV'].dropna()
            
            report.append("\n  NORMALIZED BRAIN VOLUME (nWBV) - Key biomarker:")
            if len(nwbv_0) > 0:
                report.append(f"    CDR=0:   Mean={nwbv_0.mean():.4f}, Std={nwbv_0.std():.4f}")
            if len(nwbv_05) > 0:
                report.append(f"    CDR=0.5: Mean={nwbv_05.mean():.4f}, Std={nwbv_05.std():.4f}")
            
            if len(nwbv_0) > 0 and len(nwbv_05) > 0:
                nwbv_diff = nwbv_05.mean() - nwbv_0.mean()
                report.append(f"    → Difference: {nwbv_diff:.4f} ({'lower' if nwbv_diff < 0 else 'higher'} in CDR=0.5)")
                if nwbv_diff < 0:
                    report.append(f"    ✓ Expected pattern: CDR=0.5 subjects show brain atrophy")
        
        return "\n".join(report)
    
    def generate_recommendations(self, df: 'pd.DataFrame') -> str:
        """Generate feature selection recommendations."""
        report = []
        report.append("\n" + "="*70)
        report.append("FEATURE SELECTION RECOMMENDATIONS")
        report.append("="*70)
        
        report.append("\n### RECOMMENDED FEATURES FOR EARLY DETECTION MODEL ###\n")
        
        # Category 1: Clinical/Demographic
        report.append("1. CLINICAL/DEMOGRAPHIC FEATURES:")
        clinical_features = []
        for feat in ['AGE', 'GENDER', 'EDUC', 'MMSE']:
            if feat in df.columns:
                valid_pct = (df[feat].notna().sum() / len(df)) * 100
                if valid_pct > 50:
                    clinical_features.append(feat)
                    report.append(f"   ✓ {feat} ({valid_pct:.0f}% complete)")
                else:
                    report.append(f"   ? {feat} ({valid_pct:.0f}% complete - may need imputation)")
        
        # Category 2: Volumetric
        report.append("\n2. VOLUMETRIC BIOMARKERS:")
        volumetric_features = []
        for feat in ['eTIV', 'nWBV', 'ASF', 'GM_VOLUME', 'WM_VOLUME', 'CSF_VOLUME', 'BRAIN_PERCENTAGE']:
            if feat in df.columns:
                valid_pct = (df[feat].notna().sum() / len(df)) * 100
                if valid_pct > 50:
                    volumetric_features.append(feat)
                    report.append(f"   ✓ {feat} ({valid_pct:.0f}% complete)")
        
        # Category 3: Derived ratios
        report.append("\n3. DERIVED RATIOS (Compute from above):")
        report.append("   ✓ GM/WM_RATIO = GM_VOLUME / WM_VOLUME")
        report.append("   ✓ CSF_FRACTION = CSF_VOLUME / (GM + WM + CSF)")
        report.append("   ✓ GM_FRACTION = GM_VOLUME / TOTAL_BRAIN_VOLUME")
        
        # Category 4: Image statistics
        report.append("\n4. IMAGE INTENSITY FEATURES:")
        img_features = []
        for feat in ['IMG_MEAN', 'IMG_STD', 'IMG_ENTROPY']:
            if feat in df.columns:
                valid_pct = (df[feat].notna().sum() / len(df)) * 100
                if valid_pct > 50:
                    img_features.append(feat)
                    report.append(f"   ✓ {feat} ({valid_pct:.0f}% complete)")
        
        # Category 5: Deep features
        report.append("\n5. DEEP LEARNING FEATURES (from your existing pipeline):")
        report.append("   ✓ 512-dim ResNet18 features from T88_111 scans")
        
        # Summary
        report.append("\n" + "-"*50)
        report.append("TOTAL FEATURE SET SUMMARY:")
        report.append("-"*50)
        total_features = len(clinical_features) + len(volumetric_features) + 3 + len(img_features) + 512
        report.append(f"  Clinical:     {len(clinical_features)} features")
        report.append(f"  Volumetric:   {len(volumetric_features)} features")
        report.append(f"  Derived:      3 features")
        report.append(f"  Image stats:  {len(img_features)} features")
        report.append(f"  Deep (CNN):   512 features")
        report.append(f"  ─────────────────────────")
        report.append(f"  TOTAL:        {total_features} potential features")
        
        # Subjects for training
        report.append("\n" + "-"*50)
        report.append("SUBJECTS FOR CDR=0 vs CDR=0.5 CLASSIFICATION:")
        report.append("-"*50)
        
        if 'CDR' in df.columns:
            cdr_0 = (df['CDR'] == 0).sum()
            cdr_05 = (df['CDR'] == 0.5).sum()
            report.append(f"  CDR=0 (Normal):      {cdr_0} subjects")
            report.append(f"  CDR=0.5 (Very Mild): {cdr_05} subjects")
            report.append(f"  Total usable:        {cdr_0 + cdr_05} subjects")
            
            if cdr_05 < cdr_0:
                ratio = cdr_0 / max(cdr_05, 1)
                report.append(f"\n  ⚠️  Class imbalance detected (ratio {ratio:.1f}:1)")
                report.append("     Recommendations:")
                report.append("     - Use class weights in loss function")
                report.append("     - Consider SMOTE or oversampling")
                report.append("     - Use stratified cross-validation")
        
        return "\n".join(report)
    
    def save_results(self, df: 'pd.DataFrame', report: str):
        """Save CSV and report files."""
        # Save CSV
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n[Explorer] Saved comprehensive features to: {OUTPUT_CSV}")
        
        # Save report (with UTF-8 encoding for special characters)
        with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
            f.write(report)
        print(f"[Explorer] Saved exploration report to: {OUTPUT_REPORT}")
        
        return OUTPUT_CSV, OUTPUT_REPORT


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Run comprehensive OASIS data exploration."""
    print("\n" + "="*70)
    print("OASIS DATASET DEEP EXPLORATION")
    print("="*70)
    print("\nDiscovering all available features for multimodal analysis...")
    
    # Initialize explorer
    explorer = OASISDataExplorer(BASE_DIR)
    
    # Explore all subjects
    explorer.explore_all_subjects(compute_image_stats=True)
    
    # Generate DataFrame
    if not PANDAS_AVAILABLE:
        print("[ERROR] pandas required for analysis!")
        return None
    
    df = explorer.generate_dataframe()
    
    # Generate comprehensive report
    report_sections = []
    
    # Header
    report_sections.append("="*70)
    report_sections.append("OASIS-1 DATASET COMPREHENSIVE EXPLORATION REPORT")
    report_sections.append("="*70)
    report_sections.append(f"\nGenerated for: Deep Learning-Based Multimodal MRI Analysis")
    report_sections.append(f"Total subjects explored: {len(df)}")
    report_sections.append(f"Total features extracted: {len(df.columns)}")
    
    # Analysis sections
    report_sections.append(explorer.analyze_missingness(df))
    report_sections.append(explorer.analyze_segmentation_features(df))
    report_sections.append(explorer.analyze_mri_quality(df))
    report_sections.append(explorer.analyze_demographics_vs_cdr(df))
    report_sections.append(explorer.generate_recommendations(df))
    
    full_report = "\n".join(report_sections)
    
    # Print report
    print(full_report)
    
    # Save results
    explorer.save_results(df, full_report)
    
    # Print final summary
    print("\n" + "="*70)
    print("EXPLORATION COMPLETE")
    print("="*70)
    print(f"\nFiles generated:")
    print(f"  1. {OUTPUT_CSV} - All extracted features")
    print(f"  2. {OUTPUT_REPORT} - Detailed analysis report")
    print(f"\nNext steps:")
    print("  1. Review the CSV to see all available features")
    print("  2. Decide which features to include in your model")
    print("  3. Handle missing data (imputation or exclusion)")
    print("  4. Consider derived ratios as additional biomarkers")
    
    return df


if __name__ == "__main__":
    df = main()

