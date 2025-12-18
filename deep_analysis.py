"""
OASIS Dataset Deep Intelligence Scan
=====================================
Exhaustive exploration of all possible features
"""

import os
import re
import glob
import xml.etree.ElementTree as ET
from pathlib import Path
from collections import defaultdict
import numpy as np
import nibabel as nib
import pandas as pd

BASE_DIR = "D:/disc1"

def analyze_t88_volume():
    """Analyze T88 atlas-registered volume for coordinate system and regional analysis."""
    print("=" * 70)
    print("T88 ATLAS SPACE ANALYSIS")
    print("=" * 70)
    
    t88_path = f'{BASE_DIR}/OAS1_0001_MR1/PROCESSED/MPRAGE/T88_111/OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc.hdr'
    img = nib.load(t88_path)
    data = img.get_fdata().squeeze()
    hdr = img.header
    
    print(f'Shape: {data.shape}')
    print(f'Voxel dimensions: {hdr.get_zooms()[:3]}')
    
    affine = img.affine
    print('\nAffine transformation matrix:')
    print(affine)
    
    # Compute brain centroid
    brain_mask = data > 0
    brain_coords = np.where(brain_mask)
    centroid_vox = [np.mean(c) for c in brain_coords]
    print(f'\nBrain centroid (voxels): {[f"{c:.1f}" for c in centroid_vox]}')
    
    # Regional analysis possibility
    print('\n=== REGIONAL ANALYSIS POSSIBILITY ===')
    print('T88 (Talairach) space allows approximate regional ROI definition:')
    print('  - Coordinates are standardized across subjects')
    print('  - Can define ROIs based on Talairach coordinates')
    print('  - Example ROIs:')
    print('    * Hippocampus: ~X=±25, Y=-20, Z=-15')
    print('    * Temporal lobe: X=±45, Y=-15, Z=-10')
    print('    * Frontal cortex: X=0, Y=40, Z=20')
    print('    * Ventricles: Central region near origin')
    
    return data, affine

def analyze_fseg_segmentation():
    """Deep analysis of FSL FAST segmentation."""
    print("\n" + "=" * 70)
    print("FSL FAST SEGMENTATION DEEP ANALYSIS")
    print("=" * 70)
    
    fseg_path = f'{BASE_DIR}/OAS1_0001_MR1/FSL_SEG/OAS1_0001_MR1_mpr_n4_anon_111_t88_masked_gfc_fseg.hdr'
    img = nib.load(fseg_path)
    data = img.get_fdata().squeeze()
    
    print(f'Shape: {data.shape}')
    print(f'Label mapping: 0=Background, 1=CSF, 2=GM, 3=WM')
    
    # Voxel counts
    labels = {0: 'Background', 1: 'CSF', 2: 'Gray Matter', 3: 'White Matter'}
    print('\nTissue volumes:')
    for label, name in labels.items():
        count = (data == label).sum()
        vol_mm3 = count  # 1mm isotropic
        print(f'  {name}: {count:,} voxels = {vol_mm3/1000:.2f} cm³')
    
    # Hemisphere analysis
    print('\n=== HEMISPHERE ANALYSIS ===')
    mid_x = data.shape[0] // 2
    left_brain = data[:mid_x, :, :]
    right_brain = data[mid_x:, :, :]
    
    print('Left vs Right hemisphere tissue distribution:')
    for label, name in [(2, 'GM'), (3, 'WM')]:
        left_count = (left_brain == label).sum()
        right_count = (right_brain == label).sum()
        asymmetry = (right_count - left_count) / (right_count + left_count) * 100
        print(f'  {name}: Left={left_count:,}, Right={right_count:,}, Asymmetry={asymmetry:+.2f}%')
    
    # Slice-wise analysis
    print('\n=== SLICE-WISE TISSUE DISTRIBUTION ===')
    n_slices = data.shape[2]
    gm_per_slice = [(data[:, :, z] == 2).sum() for z in range(n_slices)]
    wm_per_slice = [(data[:, :, z] == 3).sum() for z in range(n_slices)]
    
    peak_gm_slice = np.argmax(gm_per_slice)
    peak_wm_slice = np.argmax(wm_per_slice)
    print(f'Peak GM slice (axial): {peak_gm_slice} with {gm_per_slice[peak_gm_slice]:,} voxels')
    print(f'Peak WM slice (axial): {peak_wm_slice} with {wm_per_slice[peak_wm_slice]:,} voxels')
    
    return data

def analyze_transformation_matrices():
    """Extract QC metrics from transformation matrices."""
    print("\n" + "=" * 70)
    print("TRANSFORMATION MATRIX ANALYSIS (QC METRICS)")
    print("=" * 70)
    
    t4_folder = f'{BASE_DIR}/OAS1_0001_MR1/PROCESSED/MPRAGE/T88_111/t4_files'
    t4_files = glob.glob(f'{t4_folder}/*_t4')
    
    print(f'Found {len(t4_files)} transformation files')
    
    for t4_file in t4_files:
        filename = os.path.basename(t4_file)
        print(f'\n--- {filename} ---')
        
        with open(t4_file, 'r') as f:
            content = f.read()
        
        # Parse transformation matrix
        lines = content.strip().split('\n')
        matrix_lines = [l for l in lines if l.strip().startswith(('-', ' ')) and len(l.split()) >= 4]
        
        if len(matrix_lines) >= 3:
            # Extract 3x3 rotation matrix and translation
            matrix = []
            for line in matrix_lines[:3]:
                vals = [float(x) for x in line.split()[:4]]
                matrix.append(vals)
            matrix = np.array(matrix)
            
            # Extract rotation angles
            rotation_3x3 = matrix[:, :3]
            
            # Compute rotation angles (Euler angles approximation)
            # For small rotations: angle ≈ arccos((trace-1)/2)
            trace = np.trace(rotation_3x3)
            rotation_angle = np.arccos(np.clip((trace - 1) / 2, -1, 1)) * 180 / np.pi
            
            # Translation
            translation = matrix[:, 3]
            translation_magnitude = np.linalg.norm(translation)
            
            print(f'  Rotation angle: {rotation_angle:.2f}°')
            print(f'  Translation: [{translation[0]:.2f}, {translation[1]:.2f}, {translation[2]:.2f}] mm')
            print(f'  Translation magnitude: {translation_magnitude:.2f} mm')
        
        # Extract scale factor
        scale_match = re.search(r'scale:\s+([\d.]+)', content)
        if scale_match:
            scale = float(scale_match.group(1))
            print(f'  Scale factor: {scale:.6f}')
            scale_deviation = abs(1 - scale) * 100
            print(f'  Scale deviation from 1.0: {scale_deviation:.2f}%')

def parse_xml_exhaustively(xml_path):
    """Extract ALL data from XML file."""
    print("\n" + "=" * 70)
    print("XML METADATA EXHAUSTIVE PARSING")
    print("=" * 70)
    
    tree = ET.parse(xml_path)
    root = tree.getroot()
    
    # Remove namespaces for easier parsing
    for elem in root.iter():
        if '}' in elem.tag:
            elem.tag = elem.tag.split('}', 1)[1]
    
    extracted_data = {}
    
    def extract_all(element, prefix=''):
        for child in element:
            tag = child.tag
            full_path = f'{prefix}/{tag}' if prefix else tag
            
            # Get attributes
            if child.attrib:
                for attr_name, attr_val in child.attrib.items():
                    if '}' in attr_name:
                        attr_name = attr_name.split('}', 1)[1]
                    extracted_data[f'{full_path}@{attr_name}'] = attr_val
            
            # Get text content
            if child.text and child.text.strip():
                extracted_data[full_path] = child.text.strip()
            
            # Recurse
            extract_all(child, full_path)
    
    extract_all(root)
    
    print(f'\nExtracted {len(extracted_data)} data points from XML:')
    for key, value in sorted(extracted_data.items()):
        print(f'  {key}: {value}')
    
    return extracted_data

def compute_advanced_mri_features(subject_id='OAS1_0001_MR1'):
    """Compute advanced MRI-derived features."""
    print("\n" + "=" * 70)
    print("ADVANCED MRI FEATURE EXTRACTION")
    print("=" * 70)
    
    t88_path = f'{BASE_DIR}/{subject_id}/PROCESSED/MPRAGE/T88_111/{subject_id}_mpr_n4_anon_111_t88_masked_gfc.hdr'
    fseg_path = f'{BASE_DIR}/{subject_id}/FSL_SEG/{subject_id}_mpr_n4_anon_111_t88_masked_gfc_fseg.hdr'
    
    # Load data
    mri = nib.load(t88_path).get_fdata().squeeze()
    seg = nib.load(fseg_path).get_fdata().squeeze()
    
    brain_mask = mri > 0
    brain_voxels = mri[brain_mask]
    
    features = {}
    
    # 1. Basic intensity stats (already extracted)
    features['intensity_mean'] = brain_voxels.mean()
    features['intensity_std'] = brain_voxels.std()
    features['intensity_median'] = np.median(brain_voxels)
    
    # 2. Higher-order intensity statistics
    from scipy import stats
    features['intensity_skewness'] = stats.skew(brain_voxels)
    features['intensity_kurtosis'] = stats.kurtosis(brain_voxels)
    features['intensity_iqr'] = np.percentile(brain_voxels, 75) - np.percentile(brain_voxels, 25)
    
    # 3. Histogram features
    hist, bins = np.histogram(brain_voxels, bins=256, density=True)
    features['hist_peak_intensity'] = bins[np.argmax(hist)]
    features['hist_num_peaks'] = len([i for i in range(1, len(hist)-1) if hist[i] > hist[i-1] and hist[i] > hist[i+1]])
    
    # 4. Tissue-specific intensity statistics
    for label, name in [(1, 'csf'), (2, 'gm'), (3, 'wm')]:
        tissue_voxels = mri[seg == label]
        if len(tissue_voxels) > 0:
            features[f'{name}_intensity_mean'] = tissue_voxels.mean()
            features[f'{name}_intensity_std'] = tissue_voxels.std()
    
    # 5. GM/WM contrast ratio
    gm_mean = features.get('gm_intensity_mean', 0)
    wm_mean = features.get('wm_intensity_mean', 1)
    features['gm_wm_contrast'] = abs(gm_mean - wm_mean) / ((gm_mean + wm_mean) / 2)
    
    # 6. Hemisphere asymmetry (NEW!)
    mid_x = mri.shape[0] // 2
    left_gm = mri[:mid_x][seg[:mid_x] == 2]
    right_gm = mri[mid_x:][seg[mid_x:] == 2]
    features['gm_volume_asymmetry'] = (len(right_gm) - len(left_gm)) / (len(right_gm) + len(left_gm) + 1e-8)
    features['gm_intensity_asymmetry'] = (right_gm.mean() - left_gm.mean()) / ((right_gm.mean() + left_gm.mean()) / 2 + 1e-8)
    
    # 7. Spatial distribution features
    gm_coords = np.where(seg == 2)
    wm_coords = np.where(seg == 3)
    
    features['gm_centroid_z'] = np.mean(gm_coords[2])  # Superior-inferior position
    features['wm_centroid_z'] = np.mean(wm_coords[2])
    features['gm_spread_z'] = np.std(gm_coords[2])
    
    # 8. Surface/volume ratio approximation (shape complexity)
    from scipy import ndimage
    brain_dilated = ndimage.binary_dilation(brain_mask)
    surface_voxels = brain_dilated.sum() - brain_mask.sum()
    features['brain_surface_voxels'] = surface_voxels
    features['brain_sphericity'] = (brain_mask.sum() ** (2/3)) / (surface_voxels + 1)
    
    # 9. Ventricular features (CSF in central region)
    center_x, center_y, center_z = [s//2 for s in mri.shape]
    central_region = seg[center_x-30:center_x+30, center_y-30:center_y+30, center_z-30:center_z+30]
    features['central_csf_volume'] = (central_region == 1).sum()
    
    # 10. Slice-wise entropy
    from scipy.stats import entropy as sp_entropy
    slice_entropies = []
    for z in range(mri.shape[2]):
        slice_data = mri[:, :, z][mri[:, :, z] > 0]
        if len(slice_data) > 100:
            hist, _ = np.histogram(slice_data, bins=50, density=True)
            hist = hist[hist > 0]
            slice_entropies.append(sp_entropy(hist))
    features['mean_slice_entropy'] = np.mean(slice_entropies)
    features['std_slice_entropy'] = np.std(slice_entropies)
    
    print('\n=== ADVANCED MRI FEATURES ===')
    for name, value in sorted(features.items()):
        print(f'  {name}: {value:.4f}')
    
    return features

def define_talairach_rois():
    """Define approximate Talairach-based ROIs for regional analysis."""
    print("\n" + "=" * 70)
    print("TALAIRACH-BASED REGIONAL ROI DEFINITIONS")
    print("=" * 70)
    
    # T88 volume is 176 x 208 x 176, centered at AC
    # Voxels are 1mm isotropic
    # Center is approximately at voxel (88, 104, 88)
    center = np.array([88, 104, 88])
    
    rois = {
        'hippocampus_left': {
            'center': center + np.array([-25, -20, -15]),
            'radius': 10,
            'clinical': 'Early atrophy in Alzheimer\'s'
        },
        'hippocampus_right': {
            'center': center + np.array([25, -20, -15]),
            'radius': 10,
            'clinical': 'Early atrophy in Alzheimer\'s'
        },
        'lateral_ventricle_left': {
            'center': center + np.array([-15, 0, 5]),
            'radius': 15,
            'clinical': 'Enlargement indicates atrophy'
        },
        'lateral_ventricle_right': {
            'center': center + np.array([15, 0, 5]),
            'radius': 15,
            'clinical': 'Enlargement indicates atrophy'
        },
        'frontal_lobe': {
            'center': center + np.array([0, 45, 15]),
            'radius': 25,
            'clinical': 'Executive function, personality'
        },
        'temporal_lobe_left': {
            'center': center + np.array([-45, -15, -10]),
            'radius': 20,
            'clinical': 'Memory, language (dominant)'
        },
        'temporal_lobe_right': {
            'center': center + np.array([45, -15, -10]),
            'radius': 20,
            'clinical': 'Memory, spatial processing'
        },
        'parietal_lobe': {
            'center': center + np.array([0, -35, 35]),
            'radius': 25,
            'clinical': 'Spatial awareness'
        },
        'occipital_lobe': {
            'center': center + np.array([0, -70, 5]),
            'radius': 20,
            'clinical': 'Visual processing'
        },
        'thalamus': {
            'center': center + np.array([0, -10, 5]),
            'radius': 12,
            'clinical': 'Relay station, consciousness'
        },
        'caudate': {
            'center': center + np.array([12, 10, 10]),
            'radius': 8,
            'clinical': 'Motor control, learning'
        },
        'putamen': {
            'center': center + np.array([25, 0, 5]),
            'radius': 10,
            'clinical': 'Motor control'
        },
        'entorhinal_cortex': {
            'center': center + np.array([20, -5, -25]),
            'radius': 8,
            'clinical': 'VERY EARLY Alzheimer\'s marker!'
        }
    }
    
    print(f'Defined {len(rois)} approximate ROIs:')
    for name, roi in rois.items():
        print(f'\n  {name.upper()}:')
        print(f'    Center (Talairach): {roi["center"] - center}')
        print(f'    Radius: {roi["radius"]}mm')
        print(f'    Clinical significance: {roi["clinical"]}')
    
    return rois, center

def extract_regional_features(subject_id='OAS1_0001_MR1'):
    """Extract regional volumes and intensities using Talairach ROIs."""
    print("\n" + "=" * 70)
    print(f"REGIONAL FEATURE EXTRACTION: {subject_id}")
    print("=" * 70)
    
    # Load data
    t88_path = f'{BASE_DIR}/{subject_id}/PROCESSED/MPRAGE/T88_111/{subject_id}_mpr_n4_anon_111_t88_masked_gfc.hdr'
    fseg_path = f'{BASE_DIR}/{subject_id}/FSL_SEG/{subject_id}_mpr_n4_anon_111_t88_masked_gfc_fseg.hdr'
    
    mri = nib.load(t88_path).get_fdata().squeeze()
    seg = nib.load(fseg_path).get_fdata().squeeze()
    
    rois, center = define_talairach_rois()
    
    regional_features = {}
    
    for roi_name, roi_def in rois.items():
        roi_center = roi_def['center']
        radius = roi_def['radius']
        
        # Create spherical mask
        x, y, z = np.ogrid[:mri.shape[0], :mri.shape[1], :mri.shape[2]]
        dist = np.sqrt((x - roi_center[0])**2 + (y - roi_center[1])**2 + (z - roi_center[2])**2)
        roi_mask = dist <= radius
        
        # Extract features within ROI
        roi_mri = mri[roi_mask]
        roi_seg = seg[roi_mask]
        
        # Volume features
        regional_features[f'{roi_name}_gm_volume'] = (roi_seg == 2).sum()
        regional_features[f'{roi_name}_wm_volume'] = (roi_seg == 3).sum()
        regional_features[f'{roi_name}_csf_volume'] = (roi_seg == 1).sum()
        regional_features[f'{roi_name}_total_brain'] = ((roi_seg == 2) | (roi_seg == 3)).sum()
        
        # Intensity features
        brain_in_roi = roi_mri[roi_seg > 1]
        if len(brain_in_roi) > 0:
            regional_features[f'{roi_name}_intensity_mean'] = brain_in_roi.mean()
            regional_features[f'{roi_name}_intensity_std'] = brain_in_roi.std()
    
    # Compute key clinical ratios
    print('\n=== CLINICALLY RELEVANT REGIONAL FEATURES ===')
    
    # Hippocampal asymmetry (important for early AD)
    left_hipp = regional_features.get('hippocampus_left_gm_volume', 0)
    right_hipp = regional_features.get('hippocampus_right_gm_volume', 0)
    regional_features['hippocampal_asymmetry'] = (right_hipp - left_hipp) / (right_hipp + left_hipp + 1e-8)
    print(f'  Hippocampal asymmetry: {regional_features["hippocampal_asymmetry"]:.4f}')
    
    # Ventricular expansion (inverse correlate of brain volume)
    vent_csf = regional_features.get('lateral_ventricle_left_csf_volume', 0) + regional_features.get('lateral_ventricle_right_csf_volume', 0)
    regional_features['total_ventricular_csf'] = vent_csf
    print(f'  Total ventricular CSF: {vent_csf}')
    
    # Temporal lobe volumes (early dementia marker)
    temp_gm = regional_features.get('temporal_lobe_left_gm_volume', 0) + regional_features.get('temporal_lobe_right_gm_volume', 0)
    regional_features['total_temporal_gm'] = temp_gm
    print(f'  Total temporal GM: {temp_gm}')
    
    # Entorhinal cortex (EARLIEST AD marker)
    ent_gm = regional_features.get('entorhinal_cortex_gm_volume', 0)
    print(f'  Entorhinal cortex GM: {ent_gm} (CRITICAL EARLY MARKER)')
    
    # Frontal/Parietal ratio
    frontal = regional_features.get('frontal_lobe_gm_volume', 0)
    parietal = regional_features.get('parietal_lobe_gm_volume', 0)
    regional_features['frontal_parietal_ratio'] = frontal / (parietal + 1e-8)
    
    print(f'\nTotal regional features extracted: {len(regional_features)}')
    
    return regional_features

def check_for_atlas_masks():
    """Search for any atlas or parcellation masks in the dataset."""
    print("\n" + "=" * 70)
    print("ATLAS/PARCELLATION MASK SEARCH")
    print("=" * 70)
    
    # Search for potential atlas files
    atlas_patterns = [
        '*atlas*', '*parcell*', '*label*', '*roi*', '*mask*',
        '*brodmann*', '*aal*', '*harvard*', '*talairach*',
        '*freesurfer*', '*aparc*', '*aseg*'
    ]
    
    found_files = []
    for pattern in atlas_patterns:
        matches = glob.glob(f'{BASE_DIR}/**/{pattern}', recursive=True)
        matches += glob.glob(f'{BASE_DIR}/**/{pattern.upper()}', recursive=True)
        found_files.extend(matches)
    
    if found_files:
        print(f'Found {len(found_files)} potential atlas files:')
        for f in found_files:
            print(f'  {f}')
    else:
        print('No explicit atlas/parcellation files found in dataset.')
        print('\nHowever, since data is in T88 (Talairach) space:')
        print('  - We can apply standard Talairach atlas labels')
        print('  - We can download MNI→Talairach transformed atlases')
        print('  - Recommended atlases for OASIS T88 space:')
        print('    * Talairach Daemon labels')
        print('    * Brodmann areas (can be applied)')
        print('    * Custom coordinate-based ROIs (implemented above)')

def generate_complete_feature_inventory():
    """Generate complete inventory of all discoverable features."""
    print("\n" + "=" * 70)
    print("COMPLETE FEATURE INVENTORY")
    print("=" * 70)
    
    inventory = {
        'ALREADY_EXTRACTED': {
            'clinical': [
                'AGE', 'GENDER', 'HAND', 'EDUC', 'SES', 'CDR', 'MMSE'
            ],
            'volumetric': [
                'eTIV', 'nWBV', 'ASF', 'GM_VOLUME', 'WM_VOLUME', 'CSF_VOLUME', 
                'TOTAL_BRAIN_VOLUME', 'BRAIN_PERCENTAGE'
            ],
            'derived': [
                'GM_WM_RATIO', 'CSF_FRACTION', 'GM_FRACTION'
            ],
            'image_qc': [
                'IMG_MEAN', 'IMG_STD', 'IMG_MIN', 'IMG_MAX', 'IMG_ENTROPY',
                'SCAN_COUNT', 'SCAN_MEAN_VAR'
            ],
            'deep_learning': [
                '512-dim ResNet18 features'
            ]
        },
        'NEWLY_DISCOVERED_EXTRACTABLE': {
            'xml_metadata': [
                'scan_quality (usable/unusable per scan)',
                'stabilization_method (mask type)',
                'sequence_name (MP-RAGE)',
                'partitions (128)',
                'TR, TE, TI, flip_angle',
                'voxel_resolution_raw',
                'voxel_resolution_processed',
                'reconstruction_content_type',
                'scaling_factor (from assessor)',
                'eICV (from assessor)',
                'brainPercent (from assessor)'
            ],
            'transformation_qc': [
                'atlas_scale_factor',
                'atlas_scale_deviation_pct',
                'rotation_angle_to_atlas',
                'translation_magnitude_to_atlas',
                'inter_scan_rotation_angle',
                'inter_scan_translation',
                'inter_scan_scale_factor'
            ],
            'fseg_detailed': [
                'tissue_intensity_mean_per_class (CSF, GM, WM)',
                'tissue_intensity_std_per_class',
                'segmentation_convergence_iterations',
                'initial_kmeans_parameters',
                'final_hmrf_parameters',
                'segmentation_processing_time'
            ],
            'advanced_intensity': [
                'intensity_skewness',
                'intensity_kurtosis',
                'intensity_iqr',
                'histogram_peak_intensity',
                'histogram_num_peaks',
                'gm_wm_contrast_ratio',
                'mean_slice_entropy',
                'std_slice_entropy'
            ],
            'hemisphere_features': [
                'gm_volume_left',
                'gm_volume_right', 
                'gm_volume_asymmetry',
                'wm_volume_left',
                'wm_volume_right',
                'wm_volume_asymmetry',
                'gm_intensity_asymmetry'
            ],
            'spatial_features': [
                'gm_centroid_z (superior-inferior position)',
                'wm_centroid_z',
                'gm_spread_z (spatial extent)',
                'brain_surface_voxels',
                'brain_sphericity',
                'central_csf_volume (ventricular)'
            ],
            'regional_talairach': [
                'hippocampus_left_gm_volume (CRITICAL!)',
                'hippocampus_right_gm_volume (CRITICAL!)',
                'hippocampal_asymmetry (CRITICAL!)',
                'entorhinal_cortex_gm_volume (EARLIEST AD MARKER!)',
                'temporal_lobe_gm_volume (early marker)',
                'lateral_ventricle_csf_volume (atrophy indicator)',
                'frontal_lobe_gm_volume',
                'parietal_lobe_gm_volume',
                'occipital_lobe_gm_volume',
                'thalamus_volume',
                'caudate_volume',
                'putamen_volume',
                'frontal_parietal_ratio',
                'Per-region intensity_mean',
                'Per-region intensity_std'
            ]
        },
        'POTENTIALLY_EXTRACTABLE': {
            'with_external_atlas': [
                'Brodmann area volumes (need atlas)',
                'Harvard-Oxford subcortical volumes (need atlas)',
                'AAL region volumes (need atlas)',
                'Cortical thickness by region (need surface reconstruction)',
                'Sulcal depth measures (need surface reconstruction)'
            ],
            'with_additional_processing': [
                'Hippocampal subfield volumes (need specialized segmentation)',
                'White matter hyperintensities (need FLAIR sequence - not available)',
                'Cortical surface area (need FreeSurfer)',
                'Gyrification index (need FreeSurfer)',
                'Fractional anisotropy (need DTI - not available)'
            ],
            'from_raw_scans': [
                'Signal-to-noise ratio per scan',
                'Contrast-to-noise ratio',
                'Motion artifact score',
                'Ghosting artifact score',
                'Intensity non-uniformity score'
            ]
        },
        'NOT_AVAILABLE': [
            'Functional MRI (no fMRI data)',
            'Diffusion tensor imaging (no DTI data)',
            'Perfusion imaging (no ASL/DSC data)',
            'PET imaging (not in this dataset)',
            'Longitudinal change measures (cross-sectional only)',
            'Genetic markers (APOE status not in disc1 metadata)',
            'Biofluid markers (CSF Abeta, tau not available)'
        ]
    }
    
    # Print inventory
    for category, items in inventory.items():
        print(f'\n### {category} ###')
        if isinstance(items, dict):
            for subcategory, features in items.items():
                print(f'\n  {subcategory.upper()}:')
                for feat in features:
                    print(f'    - {feat}')
        else:
            for item in items:
                print(f'  - {item}')
    
    # Count totals
    total_extracted = sum(len(v) for v in inventory['ALREADY_EXTRACTED'].values())
    total_new = sum(len(v) for v in inventory['NEWLY_DISCOVERED_EXTRACTABLE'].values())
    total_potential = sum(len(v) for v in inventory['POTENTIALLY_EXTRACTABLE'].values())
    
    print(f'\n' + '=' * 70)
    print('FEATURE COUNT SUMMARY')
    print('=' * 70)
    print(f'Already extracted:           {total_extracted} features')
    print(f'Newly discovered extractable: {total_new} features')
    print(f'Potentially extractable:     {total_potential} features')
    print(f'NOT available:               {len(inventory["NOT_AVAILABLE"])} categories')
    print(f'─' * 70)
    print(f'TOTAL EXTRACTABLE NOW:       {total_extracted + total_new} features')
    
    return inventory

if __name__ == '__main__':
    print('=' * 70)
    print('OASIS DATASET DEEP INTELLIGENCE SCAN')
    print('=' * 70)
    
    # Run all analyses
    analyze_t88_volume()
    analyze_fseg_segmentation()
    analyze_transformation_matrices()
    parse_xml_exhaustively(f'{BASE_DIR}/OAS1_0001_MR1/OAS1_0001_MR1.xml')
    compute_advanced_mri_features()
    extract_regional_features()
    check_for_atlas_masks()
    generate_complete_feature_inventory()

