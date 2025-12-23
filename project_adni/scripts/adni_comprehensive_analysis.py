"""
================================================================================
ADNI DATASET COMPREHENSIVE ANALYSIS
================================================================================
Professional Research-Level Deep Analysis of ADNI Folder Structure and Contents

This script performs exhaustive analysis of:
1. Folder hierarchy and organization
2. File types and extensions
3. NIfTI file metadata (dimensions, voxel sizes, data types)
4. Naming conventions and patterns
5. Subject distribution
6. Scan types and acquisition parameters
7. Image characteristics (if applicable)
8. Data quality assessment

Output: ADNI_COMPREHENSIVE_REPORT.md
================================================================================
"""

import os
import re
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Any, Optional
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

try:
    import nibabel as nib
    import numpy as np
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False
    print("WARNING: nibabel/numpy not available. Install with: pip install nibabel numpy")

try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Configuration
_base_dir = Path(os.environ.get("DISCS_BASE_DIR", "D:/discs"))
_adni_dirs_env = os.environ.get("ADNI_DATA_DIRS") or os.environ.get("ADNI_DATA_DIR")
if _adni_dirs_env:
    _adni_dirs = [p.strip() for p in _adni_dirs_env.split(";") if p.strip()]
    ADNI_BASE = Path(_adni_dirs[0])
else:
    ADNI_BASE = _base_dir / "ADNI"
OUTPUT_REPORT = "ADNI_COMPREHENSIVE_REPORT.md"

class ADNIAnalyzer:
    """Comprehensive ADNI dataset analyzer"""
    
    def __init__(self, base_path: Path):
        self.base_path = Path(base_path)
        self.all_files = []
        self.subject_folders = []
        self.nifti_files = []
        self.file_stats = defaultdict(int)
        self.naming_patterns = defaultdict(list)
        self.scan_types = defaultdict(int)
        self.subject_ids = set()
        self.nifti_metadata = []
        self.errors = []
        
    def scan_directory(self):
        """Recursively scan all files and folders"""
        print("Scanning directory structure...")
        
        if not self.base_path.exists():
            raise FileNotFoundError(f"ADNI folder not found: {self.base_path}")
        
        # Get all files
        for root, dirs, files in os.walk(self.base_path):
            root_path = Path(root)
            
            # Track subject folders (top-level folders)
            if root_path.parent == self.base_path:
                self.subject_folders.append(root_path.name)
            
            for file in files:
                file_path = root_path / file
                self.all_files.append(file_path)
                
                # Track file extensions
                ext = file_path.suffix.lower()
                self.file_stats[ext] += 1
                
                # Track NIfTI files
                if ext == '.nii':
                    self.nifti_files.append(file_path)
        
        print(f"Found {len(self.all_files)} total files")
        print(f"Found {len(self.nifti_files)} NIfTI files")
        print(f"Found {len(self.subject_folders)} subject folders")
    
    def analyze_naming_patterns(self):
        """Analyze file and folder naming conventions"""
        print("\nAnalyzing naming patterns...")
        
        for nii_file in self.nifti_files:
            filename = nii_file.name
            rel_path = nii_file.relative_to(self.base_path)
            
            # Extract subject ID (format: XXX_S_XXXX)
            subject_match = re.search(r'(\d{3}_S_\d{4})', filename)
            if subject_match:
                subject_id = subject_match.group(1)
                self.subject_ids.add(subject_id)
            
            # Extract scan type
            if 'MPR' in filename:
                if 'MPR-R' in filename:
                    scan_type = 'MPR-R'
                elif 'MPR____N3__Scaled' in filename:
                    scan_type = 'MPR_N3_Scaled'
                elif 'MPR__GradWarp__B1_Correction__N3__Scaled' in filename:
                    scan_type = 'MPR_GradWarp_B1_N3_Scaled'
                else:
                    scan_type = 'MPR_Other'
            else:
                scan_type = 'Unknown'
            
            self.scan_types[scan_type] += 1
            
            # Store naming pattern
            pattern_info = {
                'filename': filename,
                'subject_id': subject_id if subject_match else 'Unknown',
                'scan_type': scan_type,
                'path_depth': len(rel_path.parts) - 1,
                'folder_structure': '/'.join(rel_path.parts[:-1])
            }
            self.naming_patterns[scan_type].append(pattern_info)
    
    def analyze_nifti_metadata(self, sample_size: int = 50):
        """Analyze NIfTI file headers and metadata"""
        print(f"\nAnalyzing NIfTI metadata (sampling {min(sample_size, len(self.nifti_files))} files)...")
        
        if not NIBABEL_AVAILABLE:
            print("SKIPPED: nibabel not available")
            return
        
        # Sample files for analysis (stratified by scan type if possible)
        files_to_analyze = []
        if len(self.nifti_files) <= sample_size:
            files_to_analyze = self.nifti_files
        else:
            # Sample from each scan type
            files_by_type = defaultdict(list)
            for nii_file in self.nifti_files:
                filename = nii_file.name
                if 'MPR-R' in filename:
                    files_by_type['MPR-R'].append(nii_file)
                elif 'MPR__GradWarp' in filename:
                    files_by_type['MPR_GradWarp'].append(nii_file)
                elif 'MPR____N3' in filename:
                    files_by_type['MPR_N3'].append(nii_file)
                else:
                    files_by_type['Other'].append(nii_file)
            
            samples_per_type = sample_size // len(files_by_type)
            for scan_type, files in files_by_type.items():
                files_to_analyze.extend(files[:samples_per_type])
            
            # Fill remaining slots
            remaining = sample_size - len(files_to_analyze)
            if remaining > 0:
                remaining_files = [f for f in self.nifti_files if f not in files_to_analyze]
                files_to_analyze.extend(remaining_files[:remaining])
        
        for nii_file in files_to_analyze:
            try:
                img = nib.load(str(nii_file))
                data = img.get_fdata()
                hdr = img.header
                
                metadata = {
                    'file': nii_file.name,
                    'path': str(nii_file.relative_to(self.base_path)),
                    'shape': data.shape,
                    'dtype': str(data.dtype),
                    'voxel_dims': hdr.get_zooms()[:3] if len(hdr.get_zooms()) >= 3 else None,
                    'affine_shape': img.affine.shape if img.affine is not None else None,
                    'data_min': float(np.nanmin(data)),
                    'data_max': float(np.nanmax(data)),
                    'data_mean': float(np.nanmean(data)),
                    'data_std': float(np.nanstd(data)),
                    'non_zero_voxels': int(np.count_nonzero(data)),
                    'total_voxels': int(data.size),
                    'brain_percentage': float(np.count_nonzero(data) / data.size * 100) if data.size > 0 else 0,
                    'header_class': str(type(hdr).__name__),
                }
                
                # Try to extract additional header info
                try:
                    if hasattr(hdr, 'get_sform'):
                        sform = hdr.get_sform()
                        metadata['sform_code'] = int(hdr.get_sform()[1]) if sform is not None else None
                except:
                    pass
                
                self.nifti_metadata.append(metadata)
                
            except Exception as e:
                self.errors.append(f"Error analyzing {nii_file}: {str(e)}")
    
    def analyze_folder_structure(self):
        """Analyze the hierarchical folder structure"""
        print("\nAnalyzing folder structure...")
        
        structure_stats = {
            'max_depth': 0,
            'depth_distribution': defaultdict(int),
            'folder_patterns': defaultdict(int),
            'scan_type_folders': defaultdict(set),
            'acquisition_dates': []
        }
        
        for nii_file in self.nifti_files:
            rel_path = nii_file.relative_to(self.base_path)
            depth = len(rel_path.parts) - 1
            structure_stats['max_depth'] = max(structure_stats['max_depth'], depth)
            structure_stats['depth_distribution'][depth] += 1
            
            # Analyze folder names
            for part in rel_path.parts[:-1]:
                if 'MPR' in part:
                    structure_stats['scan_type_folders']['MPR'].add(part)
                # Extract acquisition dates (format: YYYY-MM-DD_HH_MM_SS.0)
                date_match = re.match(r'(\d{4}-\d{2}-\d{2})', part)
                if date_match:
                    structure_stats['folder_patterns']['Date_Folder'] += 1
                    try:
                        date_str = date_match.group(1)
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                        structure_stats['acquisition_dates'].append(date_obj)
                    except:
                        pass
                if re.match(r'^I\d+$', part):
                    structure_stats['folder_patterns']['ImageID_Folder'] += 1
        
        return structure_stats
    
    def check_for_images(self):
        """Check for any image files (PNG, JPG, etc.)"""
        print("\nChecking for image files...")
        
        image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.tif']
        image_files = []
        
        for file_path in self.all_files:
            if file_path.suffix.lower() in image_extensions:
                image_files.append(file_path)
        
        image_analysis = {
            'count': len(image_files),
            'files': image_files,
            'by_type': defaultdict(int)
        }
        
        for img_file in image_files:
            ext = img_file.suffix.lower()
            image_analysis['by_type'][ext] += 1
            
            # If PIL available, analyze image properties
            if PIL_AVAILABLE:
                try:
                    img = Image.open(img_file)
                    # Store basic info (we'll add this to metadata if needed)
                except:
                    pass
        
        return image_analysis
    
    def generate_report(self):
        """Generate comprehensive markdown report"""
        print("\nGenerating comprehensive report...")
        
        structure_stats = self.analyze_folder_structure()
        image_analysis = self.check_for_images()
        
        # Calculate statistics
        total_size = sum(f.stat().st_size for f in self.all_files if f.exists())
        total_size_gb = total_size / (1024**3)
        
        # NIfTI statistics
        nifti_shapes = [m['shape'] for m in self.nifti_metadata if 'shape' in m]
        unique_shapes = Counter(nifti_shapes)
        
        voxel_dims_list = [m['voxel_dims'] for m in self.nifti_metadata if m.get('voxel_dims')]
        unique_voxel_dims = Counter([tuple(v) if v else None for v in voxel_dims_list])
        
        # Generate report
        report_lines = []
        report_lines.append("# ADNI DATASET COMPREHENSIVE ANALYSIS REPORT")
        report_lines.append("=" * 80)
        report_lines.append(f"\n**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"**Analysis Base Path:** `{self.base_path}`")
        report_lines.append(f"**Analysis Type:** Professional Research-Level Deep Analysis")
        report_lines.append("\n---\n")
        
        # Executive Summary
        report_lines.append("## 1. EXECUTIVE SUMMARY")
        report_lines.append("-" * 80)
        report_lines.append(f"\n- **Total Files:** {len(self.all_files)}")
        report_lines.append(f"- **Total Size:** {total_size_gb:.2f} GB")
        report_lines.append(f"- **NIfTI Files:** {len(self.nifti_files)}")
        report_lines.append(f"- **Subject Folders:** {len(self.subject_folders)}")
        report_lines.append(f"- **Unique Subjects:** {len(self.subject_ids)}")
        report_lines.append(f"- **Scan Types Identified:** {len(self.scan_types)}")
        report_lines.append(f"- **Files Analyzed (Metadata):** {len(self.nifti_metadata)}")
        report_lines.append(f"- **Errors Encountered:** {len(self.errors)}")
        report_lines.append("\n---\n")
        
        # File Types
        report_lines.append("## 2. FILE TYPES AND EXTENSIONS")
        report_lines.append("-" * 80)
        report_lines.append("\n| Extension | Count | Percentage |")
        report_lines.append("|-----------|-------|------------|")
        for ext, count in sorted(self.file_stats.items(), key=lambda x: x[1], reverse=True):
            pct = (count / len(self.all_files)) * 100
            ext_display = ext if ext else "(no extension)"
            report_lines.append(f"| `{ext_display}` | {count} | {pct:.2f}% |")
        report_lines.append("\n---\n")
        
        # Folder Structure
        report_lines.append("## 3. FOLDER STRUCTURE ANALYSIS")
        report_lines.append("-" * 80)
        report_lines.append(f"\n- **Maximum Depth:** {structure_stats['max_depth']} levels")
        report_lines.append("\n### Depth Distribution:")
        report_lines.append("| Depth | File Count |")
        report_lines.append("|-------|------------|")
        for depth in sorted(structure_stats['depth_distribution'].keys()):
            count = structure_stats['depth_distribution'][depth]
            report_lines.append(f"| {depth} | {count} |")
        
        report_lines.append("\n### Folder Pattern Analysis:")
        for pattern, count in structure_stats['folder_patterns'].items():
            report_lines.append(f"- **{pattern}:** {count} occurrences")
        
        # Date analysis
        if structure_stats['acquisition_dates']:
            dates = structure_stats['acquisition_dates']
            min_date = min(dates)
            max_date = max(dates)
            date_span = (max_date - min_date).days
            
            report_lines.append("\n### Acquisition Date Range:")
            report_lines.append(f"- **Earliest Scan:** {min_date.strftime('%Y-%m-%d')}")
            report_lines.append(f"- **Latest Scan:** {max_date.strftime('%Y-%m-%d')}")
            report_lines.append(f"- **Time Span:** {date_span} days ({date_span/365.25:.1f} years)")
            
            # Count by year
            years = Counter([d.year for d in dates])
            report_lines.append("\n### Scans by Year:")
            report_lines.append("| Year | Count |")
            report_lines.append("|------|-------|")
            for year in sorted(years.keys()):
                report_lines.append(f"| {year} | {years[year]} |")
        
        report_lines.append("\n### Typical Folder Structure:")
        if self.nifti_files:
            example_path = self.nifti_files[0].relative_to(self.base_path)
            report_lines.append("```")
            report_lines.append(str(example_path))
            report_lines.append("```")
            report_lines.append("\n**Structure Breakdown:**")
            report_lines.append("1. **Subject ID Folder** (e.g., `002_S_0295`)")
            report_lines.append("2. **Scan Type Folder** (e.g., `MPR__GradWarp__B1_Correction__N3__Scaled`)")
            report_lines.append("3. **Date-Time Folder** (e.g., `2006-11-02_08_16_44.0`)")
            report_lines.append("4. **Image ID Folder** (e.g., `I40966`)")
            report_lines.append("5. **NIfTI File** (e.g., `ADNI_002_S_0295_MR_...nii`)")
        report_lines.append("\n---\n")
        
        # Subject Analysis
        report_lines.append("## 4. SUBJECT DISTRIBUTION")
        report_lines.append("-" * 80)
        report_lines.append(f"\n- **Total Subject Folders:** {len(self.subject_folders)}")
        report_lines.append(f"- **Unique Subject IDs (from filenames):** {len(self.subject_ids)}")
        
        # Count scans per subject
        scans_per_subject = defaultdict(int)
        for nii_file in self.nifti_files:
            subject_match = re.search(r'(\d{3}_S_\d{4})', nii_file.name)
            if subject_match:
                scans_per_subject[subject_match.group(1)] += 1
        
        report_lines.append("\n### Scans per Subject Distribution:")
        scan_counts = Counter(scans_per_subject.values())
        report_lines.append("| Scans per Subject | Number of Subjects |")
        report_lines.append("|-------------------|---------------------|")
        for scan_count in sorted(scan_counts.keys()):
            subject_count = scan_counts[scan_count]
            report_lines.append(f"| {scan_count} | {subject_count} |")
        
        report_lines.append("\n---\n")
        
        # Scan Types
        report_lines.append("## 5. SCAN TYPE ANALYSIS")
        report_lines.append("-" * 80)
        report_lines.append("\n| Scan Type | Count | Percentage |")
        report_lines.append("|-----------|-------|------------|")
        total_scans = sum(self.scan_types.values())
        for scan_type, count in sorted(self.scan_types.items(), key=lambda x: x[1], reverse=True):
            pct = (count / total_scans) * 100 if total_scans > 0 else 0
            report_lines.append(f"| `{scan_type}` | {count} | {pct:.2f}% |")
        
        report_lines.append("\n### Scan Type Descriptions:")
        report_lines.append("- **MPR-R**: Magnetization Prepared Rapid Gradient Echo - Reverse")
        report_lines.append("- **MPR_N3_Scaled**: MPRAGE with N3 bias field correction and scaling")
        report_lines.append("- **MPR_GradWarp_B1_N3_Scaled**: MPRAGE with gradient distortion correction, B1 field correction, N3 bias correction, and scaling")
        report_lines.append("\n---\n")
        
        # NIfTI Metadata Analysis
        report_lines.append("## 6. NIFTI FILE METADATA ANALYSIS")
        report_lines.append("-" * 80)
        
        if self.nifti_metadata:
            report_lines.append(f"\n**Files Analyzed:** {len(self.nifti_metadata)}")
            
            # Shape analysis
            report_lines.append("\n### Image Dimensions (Shape):")
            report_lines.append("| Shape | Count |")
            report_lines.append("|-------|-------|")
            for shape, count in unique_shapes.most_common(10):
                report_lines.append(f"| `{shape}` | {count} |")
            
            # Voxel dimensions
            report_lines.append("\n### Voxel Dimensions:")
            report_lines.append("| Voxel Size (mm) | Count |")
            report_lines.append("|-----------------|-------|")
            for vox_dims, count in unique_voxel_dims.most_common(10):
                if vox_dims:
                    dims_str = f"{vox_dims[0]:.2f} × {vox_dims[1]:.2f} × {vox_dims[2]:.2f}"
                    report_lines.append(f"| `{dims_str}` | {count} |")
            
            # Data type analysis
            dtypes = Counter([m.get('dtype', 'Unknown') for m in self.nifti_metadata])
            report_lines.append("\n### Data Types:")
            report_lines.append("| Data Type | Count |")
            report_lines.append("|-----------|-------|")
            for dtype, count in dtypes.items():
                report_lines.append(f"| `{dtype}` | {count} |")
            
            # Intensity statistics
            if self.nifti_metadata:
                all_means = [m.get('data_mean', 0) for m in self.nifti_metadata if m.get('data_mean') is not None]
                all_stds = [m.get('data_std', 0) for m in self.nifti_metadata if m.get('data_std') is not None]
                all_brain_pcts = [m.get('brain_percentage', 0) for m in self.nifti_metadata if m.get('brain_percentage') is not None]
                
                if all_means:
                    report_lines.append("\n### Intensity Statistics:")
                    report_lines.append(f"- **Mean Intensity (across files):** {np.mean(all_means):.2f} ± {np.std(all_means):.2f}")
                    report_lines.append(f"- **Std Intensity (across files):** {np.mean(all_stds):.2f} ± {np.std(all_stds):.2f}")
                
                if all_brain_pcts:
                    report_lines.append(f"- **Brain Percentage (non-zero voxels):** {np.mean(all_brain_pcts):.2f}% ± {np.std(all_brain_pcts):.2f}%")
            
            # Sample metadata
            report_lines.append("\n### Sample File Metadata:")
            report_lines.append("\n<details>")
            report_lines.append("<summary>Click to expand sample metadata (first 5 files)</summary>")
            report_lines.append("\n```")
            import json
            for i, meta in enumerate(self.nifti_metadata[:5]):
                report_lines.append(f"\n// File {i+1}: {meta['file']}")
                # Convert numpy types to native Python types for JSON
                json_meta = {}
                for k, v in meta.items():
                    if isinstance(v, (np.integer, np.floating)):
                        json_meta[k] = float(v) if isinstance(v, (np.floating, np.float32, np.float64)) else int(v)
                    elif isinstance(v, (np.float32, np.float64)):
                        json_meta[k] = float(v)
                    elif isinstance(v, (np.int32, np.int64)):
                        json_meta[k] = int(v)
                    elif isinstance(v, tuple):
                        json_meta[k] = [float(x) if isinstance(x, (np.floating, np.float32, np.float64)) else int(x) if isinstance(x, (np.integer, np.int32, np.int64)) else x for x in v]
                    elif isinstance(v, list):
                        json_meta[k] = [float(x) if isinstance(x, (np.floating, np.float32, np.float64)) else int(x) if isinstance(x, (np.integer, np.int32, np.int64)) else x for x in v]
                    else:
                        json_meta[k] = v
                report_lines.append(json.dumps(json_meta, indent=2, default=str))
            report_lines.append("```")
            report_lines.append("</details>")
        else:
            report_lines.append("\n**No NIfTI metadata available** (nibabel may not be installed)")
        
        report_lines.append("\n---\n")
        
        # Image Files Analysis
        report_lines.append("## 7. IMAGE FILES ANALYSIS")
        report_lines.append("-" * 80)
        report_lines.append(f"\n- **Total Image Files Found:** {image_analysis['count']}")
        
        if image_analysis['count'] > 0:
            report_lines.append("\n### Image Files by Type:")
            report_lines.append("| Extension | Count |")
            report_lines.append("|-----------|-------|")
            for ext, count in sorted(image_analysis['by_type'].items(), key=lambda x: x[1], reverse=True):
                report_lines.append(f"| `{ext}` | {count} |")
            
            if image_analysis['files']:
                report_lines.append("\n### Sample Image Files:")
                for img_file in image_analysis['files'][:10]:
                    report_lines.append(f"- `{img_file.relative_to(self.base_path)}`")
        else:
            report_lines.append("\n**No standard image files (PNG, JPG, etc.) found in ADNI folder.**")
            report_lines.append("All imaging data is stored in NIfTI format (.nii files).")
        
        report_lines.append("\n---\n")
        
        # Naming Convention Analysis
        report_lines.append("## 8. NAMING CONVENTION ANALYSIS")
        report_lines.append("-" * 80)
        
        if self.nifti_files:
            example_file = self.nifti_files[0].name
            report_lines.append(f"\n### Example Filename:")
            report_lines.append(f"```")
            report_lines.append(example_file)
            report_lines.append("```")
            
            report_lines.append("\n### Filename Structure Breakdown:")
            report_lines.append("The ADNI filename follows this pattern:")
            report_lines.append("```")
            report_lines.append("ADNI_[SubjectID]_MR_[ScanType]_Br_[Date]_S[SeriesID]_I[ImageID].nii")
            report_lines.append("```")
            
            report_lines.append("\n**Components:**")
            report_lines.append("- `ADNI`: Dataset identifier")
            report_lines.append("- `[SubjectID]`: Format `XXX_S_XXXX` (e.g., `002_S_0295`)")
            report_lines.append("- `MR`: Modality (Magnetic Resonance)")
            report_lines.append("- `[ScanType]`: Processing pipeline identifier")
            report_lines.append("- `Br`: Brain")
            report_lines.append("- `[Date]`: Processing date (YYYYMMDDHHMMSS)")
            report_lines.append("- `S[SeriesID]`: Series identifier")
            report_lines.append("- `I[ImageID]`: Image identifier")
        
        report_lines.append("\n---\n")
        
        # Data Quality Assessment
        report_lines.append("## 9. DATA QUALITY ASSESSMENT")
        report_lines.append("-" * 80)
        
        if self.nifti_metadata:
            # Check for consistency
            shape_consistency = len(unique_shapes) == 1
            voxel_consistency = len(unique_voxel_dims) == 1
            
            report_lines.append("\n### Consistency Checks:")
            report_lines.append(f"- **Shape Consistency:** {'[OK] All files have same shape' if shape_consistency else f'[WARN] {len(unique_shapes)} different shapes found'}")
            report_lines.append(f"- **Voxel Size Consistency:** {'[OK] All files have same voxel size' if voxel_consistency else f'[WARN] {len(unique_voxel_dims)} different voxel sizes found'}")
            
            # Check for potential issues
            issues = []
            if len(unique_shapes) > 5:
                issues.append("High shape variability - may need resampling for analysis")
            if len(unique_voxel_dims) > 5:
                issues.append("High voxel size variability - may need spatial normalization")
            
            if self.errors:
                issues.append(f"{len(self.errors)} files had errors during analysis")
            
            if issues:
                report_lines.append("\n### Potential Issues:")
                for issue in issues:
                    report_lines.append(f"- [WARN] {issue}")
            else:
                report_lines.append("\n### Quality Status:")
                report_lines.append("[OK] No major quality issues detected")
        
        report_lines.append("\n---\n")
        
        # Research Implications
        report_lines.append("## 10. RESEARCH IMPLICATIONS AND RECOMMENDATIONS")
        report_lines.append("-" * 80)
        
        report_lines.append("\n### Dataset Characteristics:")
        report_lines.append(f"- This ADNI dataset contains **{len(self.nifti_files)} structural MRI scans** from **{len(self.subject_ids)} unique subjects**")
        report_lines.append("- All data is in **NIfTI format** (.nii), which is standard for neuroimaging research")
        report_lines.append("- Multiple scan types indicate different preprocessing pipelines applied")
        
        report_lines.append("\n### Recommended Preprocessing Steps:")
        report_lines.append("1. **Spatial Normalization**: Register all images to a common space (e.g., MNI152)")
        report_lines.append("2. **Resampling**: Ensure consistent voxel sizes across all images")
        report_lines.append("3. **Skull Stripping**: Verify brain extraction (may already be done)")
        report_lines.append("4. **Bias Field Correction**: Some scans have N3 correction, ensure consistency")
        
        report_lines.append("\n### Analysis Considerations:")
        report_lines.append("- **Multi-site data**: ADNI data comes from multiple sites - account for site effects")
        report_lines.append("- **Longitudinal potential**: Check if multiple timepoints exist per subject")
        report_lines.append("- **Clinical metadata**: ADNI provides extensive clinical data - link using Subject IDs")
        report_lines.append("- **Quality control**: Review images for artifacts, motion, or acquisition issues")
        
        report_lines.append("\n### Compatible Tools:")
        report_lines.append("- **Python**: nibabel, nilearn, scipy, scikit-learn")
        report_lines.append("- **FSL**: fslmaths, flirt, fnirt for preprocessing")
        report_lines.append("- **SPM**: Statistical Parametric Mapping")
        report_lines.append("- **ANTs**: Advanced Normalization Tools")
        report_lines.append("- **FreeSurfer**: Cortical surface analysis")
        
        report_lines.append("\n---\n")
        
        # Errors
        if self.errors:
            report_lines.append("## 11. ERRORS AND WARNINGS")
            report_lines.append("-" * 80)
            report_lines.append(f"\n**Total Errors:** {len(self.errors)}")
            report_lines.append("\n### Error Summary:")
            error_summary = Counter([e.split(':')[0] if ':' in e else e for e in self.errors])
            for error_type, count in error_summary.items():
                report_lines.append(f"- `{error_type}`: {count} occurrences")
            
            report_lines.append("\n<details>")
            report_lines.append("<summary>Click to expand full error list</summary>")
            report_lines.append("\n```")
            for error in self.errors[:20]:  # Limit to first 20
                report_lines.append(error)
            if len(self.errors) > 20:
                report_lines.append(f"\n... and {len(self.errors) - 20} more errors")
            report_lines.append("```")
            report_lines.append("</details>")
            report_lines.append("\n---\n")
        
        # Appendix
        report_lines.append("## APPENDIX: TECHNICAL DETAILS")
        report_lines.append("-" * 80)
        report_lines.append("\n### Analysis Parameters:")
        report_lines.append(f"- Base Path: `{self.base_path}`")
        report_lines.append(f"- Files Scanned: {len(self.all_files)}")
        report_lines.append(f"- NIfTI Files Found: {len(self.nifti_files)}")
        report_lines.append(f"- Metadata Samples Analyzed: {len(self.nifti_metadata)}")
        report_lines.append(f"- Analysis Date: {datetime.now().isoformat()}")
        
        report_lines.append("\n### Software Versions:")
        if NIBABEL_AVAILABLE:
            report_lines.append(f"- nibabel: {nib.__version__}")
            report_lines.append(f"- numpy: {np.__version__}")
        else:
            report_lines.append("- nibabel: Not available")
        
        report_lines.append("\n---\n")
        report_lines.append("\n**END OF REPORT**")
        report_lines.append(f"\n*Report generated by ADNI Comprehensive Analysis Script*")
        report_lines.append(f"*For questions or issues, review the analysis script: `adni_comprehensive_analysis.py`*")
        
        # Write report
        report_content = '\n'.join(report_lines)
        with open(OUTPUT_REPORT, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        print(f"\n[OK] Report generated: {OUTPUT_REPORT}")
        return report_content


def main():
    """Main analysis function"""
    print("=" * 80)
    print("ADNI COMPREHENSIVE ANALYSIS")
    print("=" * 80)
    print()
    
    analyzer = ADNIAnalyzer(ADNI_BASE)
    
    try:
        # Run analysis
        analyzer.scan_directory()
        analyzer.analyze_naming_patterns()
        analyzer.analyze_nifti_metadata(sample_size=100)  # Analyze up to 100 files
        report = analyzer.generate_report()
        
        print("\n" + "=" * 80)
        print("ANALYSIS COMPLETE")
        print("=" * 80)
        print(f"\nReport saved to: {OUTPUT_REPORT}")
        print(f"\nSummary:")
        print(f"  - Total files: {len(analyzer.all_files)}")
        print(f"  - NIfTI files: {len(analyzer.nifti_files)}")
        print(f"  - Subjects: {len(analyzer.subject_ids)}")
        print(f"  - Scan types: {len(analyzer.scan_types)}")
        print(f"  - Files analyzed: {len(analyzer.nifti_metadata)}")
        
    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

