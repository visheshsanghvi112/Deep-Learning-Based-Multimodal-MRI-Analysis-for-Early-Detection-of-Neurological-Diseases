"""
================================================================================
OASIS DEEP FEATURE SCAN - EXHAUSTIVE DATA MINING
================================================================================
Mission: Discover EVERY possible feature, variable, map, segmentation, score,
measurement, label, mask, annotation, or metadata inside OASIS.

"We have left no clinically relevant information unused."
================================================================================
"""

import os
import re
import glob
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from collections import defaultdict
import warnings
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

try:
    import nibabel as nib
    NIBABEL_AVAILABLE = True
except ImportError:
    NIBABEL_AVAILABLE = False

try:
    from scipy import stats as scipy_stats
    from scipy.ndimage import center_of_mass, label as ndimage_label
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from nilearn import datasets as nilearn_datasets
    from nilearn.image import resample_to_img
    NILEARN_AVAILABLE = True
except ImportError:
    NILEARN_AVAILABLE = False


BASE_DIR = "D:/disc1"


# ==============================================================================
# 1. XML METADATA DEEP PARSER
# ==============================================================================

class XMLMetadataExtractor:
    """
    Exhaustively parse ALL XML metadata fields.
    """
    
    def __init__(self):
        self.namespaces = {
            'xnat': 'http://nrg.wustl.edu/xnat',
            'oasis': 'http://nmr.mgh.harvard.edu/oasis',
            'xsi': 'http://www.w3.org/2001/XMLSchema-instance'
        }
    
    def parse_xml(self, xml_path: Path) -> Dict[str, Any]:
        """Extract ALL fields from XML file."""
        result = {
            'xml_parsed': True,
            'scan_quality_flags': [],
            'scan_count': 0,
            'all_scans_usable': True,
            'acquisition_params': {},
            'assessor_data': {},
            'file_metadata': {}
        }
        
        try:
            tree = ET.parse(str(xml_path))
            root = tree.getroot()
            
            # Extract session ID
            result['session_id'] = root.attrib.get('ID', '')
            
            # Extract subject ID
            for elem in root.iter():
                if 'subject_ID' in elem.tag:
                    result['subject_id'] = elem.text
                    break
            
            # Extract stabilization method
            for elem in root.iter():
                if 'stabilization' in elem.tag:
                    result['stabilization_method'] = elem.text
            
            # Parse all scans
            scans = []
            for scan in root.iter():
                if 'scan' in scan.tag and scan.attrib.get('ID'):
                    scan_info = {
                        'scan_id': scan.attrib.get('ID'),
                        'scan_type': scan.attrib.get('type'),
                    }
                    
                    # Get quality flag
                    for child in scan:
                        if 'quality' in child.tag:
                            scan_info['quality'] = child.text
                            result['scan_quality_flags'].append(child.text)
                            if child.text != 'usable':
                                result['all_scans_usable'] = False
                        
                        # Get parameters
                        if 'parameters' in child.tag:
                            for param in child:
                                tag_name = param.tag.split('}')[-1]
                                if param.text:
                                    scan_info[f'param_{tag_name}'] = param.text
                                for attr, val in param.attrib.items():
                                    scan_info[f'param_{tag_name}_{attr}'] = val
                    
                    scans.append(scan_info)
            
            result['scan_count'] = len(scans)
            result['scans'] = scans
            
            # Get first scan params as primary
            if scans:
                for key, val in scans[0].items():
                    if key.startswith('param_'):
                        result['acquisition_params'][key] = val
            
            # Parse assessors (ASF, segmentation data)
            for assessor in root.iter():
                if 'assessor' in assessor.tag:
                    assessor_id = assessor.attrib.get('ID', '')
                    assessor_type = assessor.attrib.get('{http://www.w3.org/2001/XMLSchema-instance}type', '')
                    
                    if 'ASF' in assessor_id or 'ScalingFactor' in assessor_type:
                        for child in assessor.iter():
                            if 'scalingFactor' in child.tag:
                                result['assessor_data']['xml_scalingFactor'] = float(child.text)
                            if 'eICV' in child.tag:
                                result['assessor_data']['xml_eICV'] = float(child.text)
                    
                    if 'FSEG' in assessor_id or 'segmentation' in assessor_type.lower():
                        brain_pct = assessor.attrib.get('brainPercent')
                        if brain_pct:
                            result['assessor_data']['xml_brainPercent'] = float(brain_pct)
            
            # Parse reconstructions for file dimensions
            for recon in root.iter():
                if 'reconstructedImage' in recon.tag:
                    for file_elem in recon.iter():
                        if 'file' in file_elem.tag:
                            content_type = file_elem.attrib.get('content', '')
                            for dim in file_elem.iter():
                                if 'dimensions' in dim.tag:
                                    result['file_metadata'][f'{content_type}_dims'] = {
                                        'x': dim.attrib.get('x'),
                                        'y': dim.attrib.get('y'),
                                        'z': dim.attrib.get('z')
                                    }
                                if 'voxelRes' in dim.tag:
                                    result['file_metadata'][f'{content_type}_voxel'] = {
                                        'x': dim.attrib.get('x'),
                                        'y': dim.attrib.get('y'),
                                        'z': dim.attrib.get('z'),
                                        'units': dim.attrib.get('units')
                                    }
        
        except Exception as e:
            result['xml_parsed'] = False
            result['xml_error'] = str(e)
        
        return result


# ==============================================================================
# 2. TRANSFORMATION MATRIX FEATURE EXTRACTOR
# ==============================================================================

class TransformationMatrixExtractor:
    """
    Extract registration quality metrics from t4 transformation files.
    
    These can indicate:
    - Head size (scaling factor)
    - Head orientation/tilt
    - Registration quality
    - Inter-scan motion
    """
    
    def parse_t4_file(self, t4_path: Path) -> Dict[str, Any]:
        """Parse a single t4 transformation file."""
        result = {}
        
        try:
            with open(t4_path, 'r') as f:
                content = f.read()
            
            # Extract 4x4 matrix
            matrix_match = re.search(
                r't4\s*\n\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\n'
                r'\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\n'
                r'\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s*\n'
                r'\s*([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)\s+([-\d.]+)',
                content
            )
            
            if matrix_match:
                matrix = np.array([float(x) for x in matrix_match.groups()]).reshape(4, 4)
                result['matrix'] = matrix
                
                # Extract rotation matrix (upper-left 3x3)
                rotation = matrix[:3, :3]
                
                # Extract translation (rightmost column)
                translation = matrix[:3, 3]
                result['translation_x'] = translation[0]
                result['translation_y'] = translation[1]
                result['translation_z'] = translation[2]
                result['translation_magnitude'] = np.linalg.norm(translation)
                
                # Compute rotation angles (Euler angles)
                # This indicates head tilt/rotation during scan
                sy = np.sqrt(rotation[0, 0]**2 + rotation[1, 0]**2)
                singular = sy < 1e-6
                if not singular:
                    result['rotation_x'] = np.arctan2(rotation[2, 1], rotation[2, 2]) * 180 / np.pi
                    result['rotation_y'] = np.arctan2(-rotation[2, 0], sy) * 180 / np.pi
                    result['rotation_z'] = np.arctan2(rotation[1, 0], rotation[0, 0]) * 180 / np.pi
                
                # Compute determinant (should be close to 1 for rigid transform)
                det = np.linalg.det(rotation)
                result['rotation_determinant'] = det
            
            # Extract scale factor
            scale_match = re.search(r'scale:\s*([-\d.]+)', content)
            if scale_match:
                result['scale_factor'] = float(scale_match.group(1))
        
        except Exception as e:
            result['error'] = str(e)
        
        return result
    
    def extract_all_transforms(self, subject_folder: Path) -> Dict[str, Any]:
        """Extract all transformation features for a subject."""
        t4_folder = subject_folder / "PROCESSED" / "MPRAGE" / "T88_111" / "t4_files"
        
        result = {
            'n_transforms': 0,
            'atlas_scale_factor': None,
            'atlas_scale_mean': None,
            'interscan_translation_var': None,
            'interscan_rotation_var': None,
            'max_rotation_angle': None,
        }
        
        if not t4_folder.exists():
            return result
        
        t4_files = list(t4_folder.glob("*_t4"))
        result['n_transforms'] = len(t4_files)
        
        atlas_scales = []
        interscan_translations = []
        interscan_rotations = []
        
        for t4_file in t4_files:
            t4_data = self.parse_t4_file(t4_file)
            
            # Atlas registration transforms (to 711-2C)
            if '711-2C' in t4_file.name:
                if 'scale_factor' in t4_data:
                    atlas_scales.append(t4_data['scale_factor'])
                    if 'mpr-1' in t4_file.name:
                        result['atlas_scale_factor'] = t4_data['scale_factor']
            
            # Inter-scan registration transforms
            elif 'to_OAS1' in t4_file.name:
                if 'translation_magnitude' in t4_data:
                    interscan_translations.append(t4_data['translation_magnitude'])
                if 'rotation_x' in t4_data:
                    max_rot = max(abs(t4_data.get('rotation_x', 0)),
                                  abs(t4_data.get('rotation_y', 0)),
                                  abs(t4_data.get('rotation_z', 0)))
                    interscan_rotations.append(max_rot)
        
        if atlas_scales:
            result['atlas_scale_mean'] = np.mean(atlas_scales)
            result['atlas_scale_std'] = np.std(atlas_scales)
        
        if interscan_translations:
            result['interscan_translation_mean'] = np.mean(interscan_translations)
            result['interscan_translation_var'] = np.var(interscan_translations)
        
        if interscan_rotations:
            result['interscan_rotation_mean'] = np.mean(interscan_rotations)
            result['interscan_rotation_var'] = np.var(interscan_rotations)
            result['max_rotation_angle'] = max(interscan_rotations)
        
        return result


# ==============================================================================
# 3. ADVANCED MRI INTENSITY FEATURES
# ==============================================================================

class AdvancedMRIFeatures:
    """
    Extract advanced intensity and morphological features from MRI.
    """
    
    def compute_histogram_features(self, data: np.ndarray, n_bins: int = 64) -> Dict[str, float]:
        """Compute histogram-based features."""
        brain_voxels = data[data > 0].flatten()
        
        if len(brain_voxels) == 0:
            return {}
        
        hist, bin_edges = np.histogram(brain_voxels, bins=n_bins, density=True)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Percentiles
        percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
        result = {}
        for p in percentiles:
            result[f'intensity_p{p}'] = float(np.percentile(brain_voxels, p))
        
        # Interquartile range
        result['intensity_iqr'] = result['intensity_p75'] - result['intensity_p25']
        
        # Histogram shape features
        result['hist_peak_intensity'] = float(bin_centers[np.argmax(hist)])
        result['hist_peak_height'] = float(hist.max())
        
        # Skewness and kurtosis
        if SCIPY_AVAILABLE:
            result['intensity_skewness'] = float(scipy_stats.skew(brain_voxels))
            result['intensity_kurtosis'] = float(scipy_stats.kurtosis(brain_voxels))
        
        # Coefficient of variation
        result['intensity_cv'] = float(brain_voxels.std() / brain_voxels.mean())
        
        return result
    
    def compute_regional_features(self, data: np.ndarray, seg_data: np.ndarray = None) -> Dict[str, float]:
        """Compute regional/spatial intensity features."""
        result = {}
        
        # Assume data is in atlas space (176 x 208 x 176)
        x_dim, y_dim, z_dim = data.shape[:3]
        
        # Hemisphere analysis (left vs right asymmetry)
        mid_x = x_dim // 2
        left_hemi = data[:mid_x, :, :]
        right_hemi = data[mid_x:, :, :]
        
        left_mean = left_hemi[left_hemi > 0].mean() if (left_hemi > 0).any() else 0
        right_mean = right_hemi[right_hemi > 0].mean() if (right_hemi > 0).any() else 0
        
        result['hemisphere_asymmetry'] = abs(left_mean - right_mean) / max(left_mean, right_mean, 1)
        result['left_hemisphere_mean'] = float(left_mean)
        result['right_hemisphere_mean'] = float(right_mean)
        
        # Anterior-posterior gradient
        mid_y = y_dim // 2
        anterior = data[:, :mid_y, :]
        posterior = data[:, mid_y:, :]
        
        ant_mean = anterior[anterior > 0].mean() if (anterior > 0).any() else 0
        post_mean = posterior[posterior > 0].mean() if (posterior > 0).any() else 0
        result['anterior_posterior_ratio'] = float(ant_mean / max(post_mean, 1))
        
        # Superior-inferior gradient
        mid_z = z_dim // 2
        superior = data[:, :, mid_z:]
        inferior = data[:, :, :mid_z]
        
        sup_mean = superior[superior > 0].mean() if (superior > 0).any() else 0
        inf_mean = inferior[inferior > 0].mean() if (inferior > 0).any() else 0
        result['superior_inferior_ratio'] = float(sup_mean / max(inf_mean, 1))
        
        # Slice-wise statistics
        slice_means = []
        slice_stds = []
        for z in range(z_dim):
            slice_data = data[:, :, z]
            brain_slice = slice_data[slice_data > 0]
            if len(brain_slice) > 100:
                slice_means.append(brain_slice.mean())
                slice_stds.append(brain_slice.std())
        
        if slice_means:
            result['slice_mean_variability'] = float(np.std(slice_means))
            result['slice_std_variability'] = float(np.std(slice_stds))
        
        return result
    
    def compute_tissue_boundary_features(self, seg_data: np.ndarray) -> Dict[str, float]:
        """Compute features from tissue segmentation boundaries."""
        result = {}
        
        if seg_data is None:
            return result
        
        seg_data = seg_data.squeeze()
        
        # Tissue label mapping: 0=background, 1=CSF, 2=GM, 3=WM
        csf_mask = (seg_data == 1)
        gm_mask = (seg_data == 2)
        wm_mask = (seg_data == 3)
        
        # Compute surface areas (boundary voxels)
        from scipy.ndimage import binary_erosion
        
        # GM surface area (boundary with CSF and WM)
        gm_eroded = binary_erosion(gm_mask)
        gm_surface = gm_mask.astype(int) - gm_eroded.astype(int)
        result['gm_surface_voxels'] = int(gm_surface.sum())
        
        # WM surface area
        wm_eroded = binary_erosion(wm_mask)
        wm_surface = wm_mask.astype(int) - wm_eroded.astype(int)
        result['wm_surface_voxels'] = int(wm_surface.sum())
        
        # CSF surface area (ventricular boundaries)
        csf_eroded = binary_erosion(csf_mask)
        csf_surface = csf_mask.astype(int) - csf_eroded.astype(int)
        result['csf_surface_voxels'] = int(csf_surface.sum())
        
        # Surface to volume ratios (compactness measures)
        gm_vol = gm_mask.sum()
        wm_vol = wm_mask.sum()
        csf_vol = csf_mask.sum()
        
        if gm_vol > 0:
            result['gm_surface_to_volume'] = result['gm_surface_voxels'] / gm_vol
        if wm_vol > 0:
            result['wm_surface_to_volume'] = result['wm_surface_voxels'] / wm_vol
        if csf_vol > 0:
            result['csf_surface_to_volume'] = result['csf_surface_voxels'] / csf_vol
        
        return result
    
    def compute_ventricle_features(self, seg_data: np.ndarray) -> Dict[str, float]:
        """Compute ventricle-specific features from CSF segmentation."""
        result = {}
        
        if seg_data is None:
            return result
        
        seg_data = seg_data.squeeze()
        csf_mask = (seg_data == 1).astype(np.uint8)
        
        if csf_mask.sum() == 0:
            return result
        
        # Label connected components in CSF
        labeled_csf, n_components = ndimage_label(csf_mask)
        
        # Find largest CSF component (likely ventricles)
        component_sizes = []
        for i in range(1, n_components + 1):
            component_sizes.append((labeled_csf == i).sum())
        
        if component_sizes:
            result['csf_n_components'] = n_components
            result['largest_csf_component'] = max(component_sizes)
            result['csf_fragmentation'] = n_components / (csf_mask.sum() / 1000)  # normalized
            
            # Ventricle location (center of mass of largest component)
            largest_idx = np.argmax(component_sizes) + 1
            largest_mask = (labeled_csf == largest_idx)
            com = center_of_mass(largest_mask)
            result['ventricle_center_x'] = float(com[0])
            result['ventricle_center_y'] = float(com[1])
            result['ventricle_center_z'] = float(com[2])
        
        return result


# ==============================================================================
# 4. TALAIRACH ATLAS REGIONAL EXTRACTION
# ==============================================================================

class TalairachRegionalExtractor:
    """
    Extract regional brain volumes using Talairach atlas coordinates.
    
    Since data is already in T88 (Talairach) space, we can define ROIs
    based on standard Talairach coordinates.
    """
    
    # Approximate Talairach coordinates for key dementia-relevant regions
    # Format: (x_min, x_max, y_min, y_max, z_min, z_max) in voxel coordinates
    # for a 176x208x176 volume with 1mm isotropic voxels
    # Origin approximately at AC (center of volume)
    
    TALAIRACH_ROIS = {
        # Hippocampus (bilateral) - critical for Alzheimer's
        'hippocampus_left': (58, 72, 85, 105, 55, 75),
        'hippocampus_right': (104, 118, 85, 105, 55, 75),
        
        # Medial temporal lobe (entorhinal cortex area)
        'medial_temporal_left': (50, 75, 80, 120, 45, 80),
        'medial_temporal_right': (100, 125, 80, 120, 45, 80),
        
        # Lateral ventricles (CSF expansion indicator)
        'lateral_ventricle_left': (65, 88, 90, 130, 65, 110),
        'lateral_ventricle_right': (88, 110, 90, 130, 65, 110),
        
        # Frontal lobe
        'frontal_left': (40, 88, 120, 190, 60, 150),
        'frontal_right': (88, 135, 120, 190, 60, 150),
        
        # Parietal lobe
        'parietal_left': (40, 88, 60, 120, 100, 160),
        'parietal_right': (88, 135, 60, 120, 100, 160),
        
        # Temporal lobe
        'temporal_left': (25, 70, 50, 130, 30, 80),
        'temporal_right': (105, 150, 50, 130, 30, 80),
        
        # Occipital lobe
        'occipital_left': (45, 88, 18, 60, 50, 120),
        'occipital_right': (88, 130, 18, 60, 50, 120),
        
        # Central structures
        'thalamus': (70, 105, 90, 115, 70, 95),
        'basal_ganglia': (60, 115, 100, 130, 60, 90),
        
        # Posterior cingulate (early AD affected)
        'posterior_cingulate': (75, 100, 55, 85, 85, 115),
        
        # Precuneus (early AD affected)
        'precuneus': (75, 100, 45, 75, 100, 140),
    }
    
    def extract_regional_volumes(self, mri_data: np.ndarray, seg_data: np.ndarray = None) -> Dict[str, float]:
        """Extract volumes and intensities for each ROI."""
        result = {}
        
        mri_data = mri_data.squeeze()
        if seg_data is not None:
            seg_data = seg_data.squeeze()
        
        for roi_name, (x1, x2, y1, y2, z1, z2) in self.TALAIRACH_ROIS.items():
            # Clip to valid range
            x1, x2 = max(0, x1), min(mri_data.shape[0], x2)
            y1, y2 = max(0, y1), min(mri_data.shape[1], y2)
            z1, z2 = max(0, z1), min(mri_data.shape[2], z2)
            
            # Extract MRI ROI
            roi_mri = mri_data[x1:x2, y1:y2, z1:z2]
            brain_voxels = roi_mri[roi_mri > 0]
            
            result[f'{roi_name}_volume'] = int(len(brain_voxels))
            if len(brain_voxels) > 0:
                result[f'{roi_name}_mean_intensity'] = float(brain_voxels.mean())
                result[f'{roi_name}_std_intensity'] = float(brain_voxels.std())
            
            # If segmentation available, get tissue-specific volumes
            if seg_data is not None:
                roi_seg = seg_data[x1:x2, y1:y2, z1:z2]
                result[f'{roi_name}_csf_volume'] = int((roi_seg == 1).sum())
                result[f'{roi_name}_gm_volume'] = int((roi_seg == 2).sum())
                result[f'{roi_name}_wm_volume'] = int((roi_seg == 3).sum())
        
        # Compute important derived biomarkers
        # Hippocampal volume ratio (left+right hippocampus / total brain)
        total_brain = (mri_data > 0).sum()
        hippo_vol = result.get('hippocampus_left_volume', 0) + result.get('hippocampus_right_volume', 0)
        result['hippocampal_fraction'] = hippo_vol / max(total_brain, 1)
        
        # Ventricular volume ratio
        vent_vol = result.get('lateral_ventricle_left_volume', 0) + result.get('lateral_ventricle_right_volume', 0)
        result['ventricular_fraction'] = vent_vol / max(total_brain, 1)
        
        # Medial temporal lobe ratio
        mtl_vol = result.get('medial_temporal_left_volume', 0) + result.get('medial_temporal_right_volume', 0)
        result['medial_temporal_fraction'] = mtl_vol / max(total_brain, 1)
        
        # Hemisphere asymmetry for each bilateral region
        bilateral_regions = ['hippocampus', 'medial_temporal', 'lateral_ventricle', 
                            'frontal', 'parietal', 'temporal', 'occipital']
        for region in bilateral_regions:
            left_vol = result.get(f'{region}_left_volume', 0)
            right_vol = result.get(f'{region}_right_volume', 0)
            if left_vol + right_vol > 0:
                result[f'{region}_asymmetry'] = abs(left_vol - right_vol) / (left_vol + right_vol)
        
        return result


# ==============================================================================
# 5. FSL FAST DETAILED PARSER
# ==============================================================================

class FSLFastDetailedParser:
    """
    Parse ALL information from FSL FAST segmentation logs.
    """
    
    def parse_fseg_log(self, log_path: Path) -> Dict[str, Any]:
        """Extract detailed segmentation parameters and QC metrics."""
        result = {}
        
        if not log_path.exists():
            return result
        
        try:
            with open(log_path, 'r') as f:
                content = f.read()
            
            # Tool version
            version_match = re.search(r'Version\s+([\d.]+)', content)
            if version_match:
                result['fsl_fast_version'] = version_match.group(1)
            
            # Image dimensions
            dim_match = re.search(r'Imagesize\s*:\s*(\d+)\s*x\s*(\d+)\s*x\s*(\d+)', content)
            if dim_match:
                result['seg_dim_x'] = int(dim_match.group(1))
                result['seg_dim_y'] = int(dim_match.group(2))
                result['seg_dim_z'] = int(dim_match.group(3))
            
            # Pixel size
            pix_match = re.search(r'Pixelsize\s*:\s*([\d.]+)\s*x\s*([\d.]+)\s*x\s*([\d.]+)', content)
            if pix_match:
                result['seg_voxel_x'] = float(pix_match.group(1))
                result['seg_voxel_y'] = float(pix_match.group(2))
                result['seg_voxel_z'] = float(pix_match.group(3))
            
            # Slice range used
            slice_match = re.search(r'from slice (\d+) to (\d+)', content)
            if slice_match:
                result['seg_slice_start'] = int(slice_match.group(1))
                result['seg_slice_end'] = int(slice_match.group(2))
                result['seg_slice_range'] = result['seg_slice_end'] - result['seg_slice_start']
            
            # Initial K-means statistics
            kmeans_match = re.search(r'starting with 1 class -- mean:\s*([\d.]+)\s+stddev:\s*([\d.]+)', content)
            if kmeans_match:
                result['initial_mean'] = float(kmeans_match.group(1))
                result['initial_stddev'] = float(kmeans_match.group(2))
            
            # Final tissue statistics (mean and stddev for each class)
            final_match = re.search(
                r'The final statistics are:\s*\n'
                r'\s*Tissue 0:\s*([\d.]+)\s+([\d.]+)\s*\n'
                r'\s*Tissue 1:\s*([\d.]+)\s+([\d.]+)\s*\n'
                r'\s*Tissue 2:\s*([\d.]+)\s+([\d.]+)',
                content
            )
            if final_match:
                result['csf_final_mean'] = float(final_match.group(1))
                result['csf_final_std'] = float(final_match.group(2))
                result['gm_final_mean'] = float(final_match.group(3))
                result['gm_final_std'] = float(final_match.group(4))
                result['wm_final_mean'] = float(final_match.group(5))
                result['wm_final_std'] = float(final_match.group(6))
                
                # Tissue contrast (WM-GM difference / GM-CSF difference)
                gm_csf_diff = result['gm_final_mean'] - result['csf_final_mean']
                wm_gm_diff = result['wm_final_mean'] - result['gm_final_mean']
                if gm_csf_diff > 0:
                    result['tissue_contrast_ratio'] = wm_gm_diff / gm_csf_diff
            
            # Number of iterations
            iter_matches = re.findall(r'Main iteration, No\. (\d+)', content)
            if iter_matches:
                result['n_iterations'] = int(iter_matches[-1]) + 1
            
            # Processing time
            time_match = re.search(r'Calculation time (\d+) seconds', content)
            if time_match:
                result['processing_time_sec'] = int(time_match.group(1))
            
            # Final volumes
            vol_match = re.search(
                r'Volumes:\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)',
                content
            )
            if vol_match:
                result['csf_volume_mm3'] = float(vol_match.group(1))
                result['gm_volume_mm3'] = float(vol_match.group(2))
                result['wm_volume_mm3'] = float(vol_match.group(3))
                result['brain_percentage'] = float(vol_match.group(4))
                
                # Derived features
                total = result['csf_volume_mm3'] + result['gm_volume_mm3'] + result['wm_volume_mm3']
                result['total_intracranial_mm3'] = total
                result['gm_fraction'] = result['gm_volume_mm3'] / total
                result['wm_fraction'] = result['wm_volume_mm3'] / total
                result['csf_fraction'] = result['csf_volume_mm3'] / total
                result['gm_wm_ratio'] = result['gm_volume_mm3'] / result['wm_volume_mm3']
        
        except Exception as e:
            result['parse_error'] = str(e)
        
        return result


# ==============================================================================
# 6. MAIN DEEP SCAN EXECUTOR
# ==============================================================================

class OASISDeepScan:
    """
    Execute exhaustive feature extraction across all subjects.
    """
    
    def __init__(self, base_dir: str = BASE_DIR):
        self.base_dir = Path(base_dir)
        self.xml_extractor = XMLMetadataExtractor()
        self.transform_extractor = TransformationMatrixExtractor()
        self.advanced_mri = AdvancedMRIFeatures()
        self.regional_extractor = TalairachRegionalExtractor()
        self.fsl_parser = FSLFastDetailedParser()
        
        # Find all subjects
        self.subjects = sorted([
            d for d in self.base_dir.iterdir()
            if d.is_dir() and d.name.startswith('OAS1_')
        ])
        print(f"[DeepScan] Found {len(self.subjects)} subjects")
    
    def scan_subject(self, subject_folder: Path, verbose: bool = False) -> Dict[str, Any]:
        """Extract ALL features for a single subject."""
        subject_id = subject_folder.name
        result = {'SUBJECT_ID': subject_id}
        
        if verbose:
            print(f"  Scanning {subject_id}...")
        
        # 1. XML Metadata
        xml_path = subject_folder / f"{subject_id}.xml"
        if xml_path.exists():
            xml_data = self.xml_extractor.parse_xml(xml_path)
            for key, value in xml_data.items():
                if not isinstance(value, (dict, list)):
                    result[f'xml_{key}'] = value
                elif key == 'acquisition_params':
                    for pk, pv in value.items():
                        result[pk] = pv
                elif key == 'assessor_data':
                    for pk, pv in value.items():
                        result[pk] = pv
        
        # 2. Transformation matrices
        transform_data = self.transform_extractor.extract_all_transforms(subject_folder)
        result.update(transform_data)
        
        # 3. FSL FAST detailed parsing
        fseg_folder = subject_folder / "FSL_SEG"
        fseg_logs = list(fseg_folder.glob("*_fseg.txt")) if fseg_folder.exists() else []
        if fseg_logs:
            fsl_data = self.fsl_parser.parse_fseg_log(fseg_logs[0])
            for key, value in fsl_data.items():
                result[f'fsl_{key}'] = value
        
        # 4. Load MRI and segmentation for advanced features
        t88_folder = subject_folder / "PROCESSED" / "MPRAGE" / "T88_111"
        t88_files = list(t88_folder.glob("*_t88_masked_gfc.hdr")) if t88_folder.exists() else []
        fseg_files = list(fseg_folder.glob("*_fseg.hdr")) if fseg_folder.exists() else []
        
        mri_data = None
        seg_data = None
        
        if NIBABEL_AVAILABLE and t88_files:
            try:
                mri_img = nib.load(str(t88_files[0]))
                mri_data = mri_img.get_fdata().squeeze()
            except:
                pass
        
        if NIBABEL_AVAILABLE and fseg_files:
            try:
                seg_img = nib.load(str(fseg_files[0]))
                seg_data = seg_img.get_fdata().squeeze()
            except:
                pass
        
        # 5. Advanced MRI features
        if mri_data is not None:
            # Histogram features
            hist_features = self.advanced_mri.compute_histogram_features(mri_data)
            result.update(hist_features)
            
            # Regional features
            regional_features = self.advanced_mri.compute_regional_features(mri_data, seg_data)
            result.update(regional_features)
        
        # 6. Tissue boundary features
        if seg_data is not None:
            boundary_features = self.advanced_mri.compute_tissue_boundary_features(seg_data)
            result.update(boundary_features)
            
            ventricle_features = self.advanced_mri.compute_ventricle_features(seg_data)
            result.update(ventricle_features)
        
        # 7. Talairach regional volumes
        if mri_data is not None:
            talairach_features = self.regional_extractor.extract_regional_volumes(mri_data, seg_data)
            result.update(talairach_features)
        
        return result
    
    def scan_all_subjects(self, verbose: bool = True) -> pd.DataFrame:
        """Scan all subjects and return comprehensive DataFrame."""
        all_results = []
        
        print("\n" + "="*70)
        print("DEEP FEATURE SCAN - EXHAUSTIVE DATA MINING")
        print("="*70 + "\n")
        
        for i, subject_folder in enumerate(self.subjects):
            if verbose:
                print(f"[{i+1}/{len(self.subjects)}] ", end="")
            
            result = self.scan_subject(subject_folder, verbose=verbose)
            all_results.append(result)
            
            if verbose:
                print(f" -> {len(result)} features extracted")
        
        df = pd.DataFrame(all_results)
        
        print(f"\n[DeepScan] Total features discovered: {len(df.columns)}")
        print(f"[DeepScan] Total subjects processed: {len(df)}")
        
        return df
    
    def generate_feature_report(self, df: pd.DataFrame) -> str:
        """Generate detailed feature discovery report."""
        report = []
        
        report.append("="*80)
        report.append("OASIS DEEP FEATURE SCAN - COMPREHENSIVE DISCOVERY REPORT")
        report.append("="*80)
        report.append(f"\nTotal subjects: {len(df)}")
        report.append(f"Total features discovered: {len(df.columns)}")
        
        # Categorize features
        categories = {
            'XML Metadata': [c for c in df.columns if c.startswith('xml_') or c.startswith('param_')],
            'Transformation/Registration': [c for c in df.columns if any(x in c.lower() for x in ['transform', 'scale', 'rotation', 'translation'])],
            'FSL Segmentation': [c for c in df.columns if c.startswith('fsl_')],
            'Intensity Statistics': [c for c in df.columns if 'intensity' in c.lower() or c.startswith('hist_')],
            'Regional Volumes': [c for c in df.columns if any(x in c for x in ['hippocampus', 'temporal', 'frontal', 'parietal', 'occipital', 'ventricle', 'thalamus', 'cingulate', 'precuneus', 'basal'])],
            'Tissue Boundaries': [c for c in df.columns if 'surface' in c.lower() or 'boundary' in c.lower()],
            'Hemisphere Asymmetry': [c for c in df.columns if 'asymmetry' in c.lower() or 'hemisphere' in c.lower()],
            'Derived Biomarkers': [c for c in df.columns if 'fraction' in c.lower() or 'ratio' in c.lower()],
        }
        
        for cat_name, cat_features in categories.items():
            if cat_features:
                report.append(f"\n{'='*60}")
                report.append(f"{cat_name.upper()} ({len(cat_features)} features)")
                report.append("="*60)
                for feat in sorted(cat_features):
                    valid_pct = df[feat].notna().mean() * 100
                    report.append(f"  - {feat}: {valid_pct:.0f}% valid")
        
        # Uncategorized
        all_categorized = set()
        for feats in categories.values():
            all_categorized.update(feats)
        uncategorized = [c for c in df.columns if c not in all_categorized and c != 'SUBJECT_ID']
        
        if uncategorized:
            report.append(f"\n{'='*60}")
            report.append(f"OTHER FEATURES ({len(uncategorized)} features)")
            report.append("="*60)
            for feat in sorted(uncategorized):
                valid_pct = df[feat].notna().mean() * 100
                report.append(f"  - {feat}: {valid_pct:.0f}% valid")
        
        return "\n".join(report)


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

def main():
    """Run exhaustive deep scan."""
    scanner = OASISDeepScan(BASE_DIR)
    
    # Scan all subjects
    df = scanner.scan_all_subjects(verbose=True)
    
    # Generate report
    report = scanner.generate_feature_report(df)
    print(report)
    
    # Save results
    output_csv = "oasis_deep_features_ALL.csv"
    df.to_csv(output_csv, index=False)
    print(f"\n[DeepScan] Saved to: {output_csv}")
    
    output_report = "oasis_deep_scan_report.txt"
    with open(output_report, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"[DeepScan] Report saved to: {output_report}")
    
    return df


if __name__ == "__main__":
    df = main()

