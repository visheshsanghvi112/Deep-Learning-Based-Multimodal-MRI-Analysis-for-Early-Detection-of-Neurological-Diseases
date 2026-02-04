"""
Create clean grids of MRI scans WITHOUT any labels.
1. Grid of 3 scans (1 from each category)
2. Grid of 9 scans (3 from each category)
"""

import os
import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import ndimage

ADNI_SOURCE_DIR = r"C:\Users\gener\Downloads\ADNI1_Complete 1Yr 1.5T\ADNI"
OUTPUT_DIR = r"d:\discs\figures"

def compute_brain_metrics(nii_path):
    """Compute brain volume metrics."""
    try:
        img = nib.load(nii_path)
        data = img.get_fdata()
        data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)
        
        brain_mask = data_norm > 0.3
        csf_mask = (data_norm > 0.05) & (data_norm < 0.15)
        
        brain_volume = np.sum(brain_mask)
        csf_volume = np.sum(csf_mask)
        total = brain_volume + csf_volume
        brain_fraction = brain_volume / (total + 1e-8)
        
        return {'brain_fraction': brain_fraction}
    except:
        return None

def load_sagittal_slice(nii_path):
    """Load sagittal slice for display."""
    try:
        img = nib.load(nii_path)
        data = img.get_fdata()
        
        slice_idx = int(data.shape[0] * 0.45)
        sagittal = data[slice_idx, :, :]
        sagittal = np.rot90(sagittal, k=1)
        sagittal = np.clip(sagittal, 0, np.percentile(sagittal, 99))
        
        return sagittal
    except:
        return None

def find_and_validate_scans():
    """Find and validate scans."""
    csv_path = r"d:\discs\data\ADNI\ADNIMERGE_23Dec2025.csv"
    df = pd.read_csv(csv_path, low_memory=False)
    baseline = df[df['VISCODE'] == 'bl'] if 'VISCODE' in df.columns else df
    
    print("Analyzing scans...")
    all_scans = {'CN': [], 'MCI': [], 'AD': []}
    
    subject_dirs = glob.glob(os.path.join(ADNI_SOURCE_DIR, "*_S_*"))
    
    for subj_dir in subject_dirs[:100]:
        if not os.path.isdir(subj_dir):
            continue
        
        subj_id = os.path.basename(subj_dir)
        nii_files = glob.glob(os.path.join(subj_dir, "**", "*.nii"), recursive=True)
        
        if not nii_files:
            continue
        
        variants = [subj_id, subj_id.replace('_S_', '_')]
        
        for variant in variants:
            matches = baseline[baseline['PTID'] == variant]
            
            if len(matches) > 0:
                row = matches.iloc[0]
                dx = row['DX']
                
                metrics = compute_brain_metrics(nii_files[0])
                
                if metrics:
                    scan_info = {
                        'subject_id': subj_id,
                        'nii_path': nii_files[0],
                        'brain_fraction': metrics['brain_fraction']
                    }
                    
                    if dx == 'CN':
                        all_scans['CN'].append(scan_info)
                    elif dx == 'MCI':
                        all_scans['MCI'].append(scan_info)
                    elif dx == 'Dementia':
                        all_scans['AD'].append(scan_info)
                
                break
    
    print(f"Found: {len(all_scans['CN'])} CN, {len(all_scans['MCI'])} MCI, {len(all_scans['AD'])} AD")
    return all_scans

def select_scans(all_scans):
    """Select best representative scans."""
    # CN: highest brain fraction
    cn_sorted = sorted(all_scans['CN'], key=lambda x: x['brain_fraction'], reverse=True)
    
    # AD: lowest brain fraction
    ad_sorted = sorted(all_scans['AD'], key=lambda x: x['brain_fraction'])
    
    # MCI: middle range
    mci_sorted = sorted(all_scans['MCI'], key=lambda x: x['brain_fraction'])
    mid_idx = len(mci_sorted) // 2
    
    selected = {
        'CN': cn_sorted[:3],
        'MCI': mci_sorted[mid_idx:mid_idx+3] if mid_idx+3 <= len(mci_sorted) else mci_sorted[:3],
        'AD': ad_sorted[:3]
    }
    
    return selected

def create_3_scan_grid(selected):
    """Create 1x3 grid - one scan from each category."""
    print("\nCreating 3-scan grid...")
    
    # Select one from each
    scans = [
        selected['CN'][0],
        selected['MCI'][0],
        selected['AD'][0]
    ]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, scan in enumerate(scans):
        ax = axes[idx]
        slice_data = load_sagittal_slice(scan['nii_path'])
        
        if slice_data is not None:
            ax.imshow(slice_data, cmap='gray', aspect='auto')
        
        ax.axis('off')
    
    # Remove all padding and spacing
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0)
    
    output_png = os.path.join(OUTPUT_DIR, "mri_grid_3_scans_clean.png")
    plt.savefig(output_png, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='black')
    
    print(f"✅ 3-scan grid saved: {output_png}")
    plt.close()

def create_9_scan_grid(selected):
    """Create 3x3 grid - three scans from each category."""
    print("\nCreating 9-scan grid...")
    
    # Arrange: CN top row, MCI middle row, AD bottom row
    scans = selected['CN'] + selected['MCI'] + selected['AD']
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    
    for idx in range(9):
        row = idx // 3
        col = idx % 3
        ax = axes[row, col]
        
        if idx < len(scans):
            slice_data = load_sagittal_slice(scans[idx]['nii_path'])
            
            if slice_data is not None:
                ax.imshow(slice_data, cmap='gray', aspect='auto')
        
        ax.axis('off')
    
    # Remove all padding and spacing
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0.02)
    
    output_png = os.path.join(OUTPUT_DIR, "mri_grid_9_scans_clean.png")
    plt.savefig(output_png, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='black')
    
    print(f"✅ 9-scan grid saved: {output_png}")
    plt.close()

if __name__ == "__main__":
    all_scans = find_and_validate_scans()
    selected = select_scans(all_scans)
    
    create_3_scan_grid(selected)
    create_9_scan_grid(selected)
    
    print("\n" + "=" * 60)
    print("COMPLETED: 2 clean grids created without any labels")
    print("=" * 60)
    print("\n1. mri_grid_3_scans_clean.png  (1x3 grid)")
    print("2. mri_grid_9_scans_clean.png  (3x3 grid)")
    print("\nBoth images have:")
    print("  - NO titles")
    print("  - NO labels")
    print("  - NO annotations")
    print("  - Just clean brain scans")
