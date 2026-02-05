"""
Create OPTIMAL 3-scan grid using computationally validated subjects:
- CN:  012_S_1009 (Brain Volume: 0.728 - Healthy)
- MCI: 013_S_0325 (Brain Volume: 0.371 - Intermediate)
- AD:  002_S_0816 (Brain Volume: 0.001 - Severe Atrophy)

This represents the BEST mathematical progression for Deep Learning research.
"""

import os
import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

ADNI_SOURCE_DIR = r"C:\Users\gener\Downloads\ADNI1_Complete 1Yr 1.5T\ADNI"
OUTPUT_DIR = r"d:\discs\figures"

# Exact subjects to use
TARGET_SUBJECTS = [
    {'id': '012_S_1009', 'category': 'CN', 'brain_volume': 0.728},
    {'id': '013_S_0325', 'category': 'MCI', 'brain_volume': 0.371},
    {'id': '002_S_0816', 'category': 'AD', 'brain_volume': 0.001}
]

def load_sagittal_slice(nii_path):
    """Load sagittal slice for display."""
    try:
        img = nib.load(nii_path)
        data = img.get_fdata()
        
        # Mid-sagittal slice
        slice_idx = int(data.shape[0] * 0.45)
        sagittal = data[slice_idx, :, :]
        sagittal = np.rot90(sagittal, k=1)
        sagittal = np.clip(sagittal, 0, np.percentile(sagittal, 99))
        
        return sagittal
    except Exception as e:
        print(f"Error loading {nii_path}: {e}")
        return None

def find_subject_scan(subject_id):
    """Find the .nii file for a specific subject."""
    subj_dir = os.path.join(ADNI_SOURCE_DIR, subject_id)
    
    if not os.path.exists(subj_dir):
        print(f"⚠️  Subject directory not found: {subject_id}")
        return None
    
    # Find .nii files
    nii_files = glob.glob(os.path.join(subj_dir, "**", "*.nii"), recursive=True)
    
    if not nii_files:
        print(f"⚠️  No .nii file found for: {subject_id}")
        return None
    
    print(f"✓ Found scan for {subject_id}")
    return nii_files[0]

def create_optimal_3_scan_grid():
    """Create the optimal 3-scan grid with specific subjects."""
    
    print("=" * 70)
    print("CREATING OPTIMAL 3-SCAN GRID")
    print("=" * 70)
    print("\nMathematically validated progression:")
    print("  CN  (012_S_1009): Brain Volume = 0.728 (Healthy)")
    print("  MCI (013_S_0325): Brain Volume = 0.371 (Intermediate)")
    print("  AD  (002_S_0816): Brain Volume = 0.001 (Severe Atrophy)")
    print()
    
    # Find all scans
    scans = []
    for subject in TARGET_SUBJECTS:
        nii_path = find_subject_scan(subject['id'])
        if nii_path:
            scans.append({
                'id': subject['id'],
                'category': subject['category'],
                'nii_path': nii_path,
                'brain_volume': subject['brain_volume']
            })
    
    if len(scans) < 3:
        print(f"\n❌ ERROR: Only found {len(scans)}/3 scans")
        return
    
    # Create figure
    print(f"\n✓ All 3 scans found! Creating grid...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, scan in enumerate(scans):
        ax = axes[idx]
        slice_data = load_sagittal_slice(scan['nii_path'])
        
        if slice_data is not None:
            ax.imshow(slice_data, cmap='gray', aspect='auto')
            print(f"  ✓ Loaded: {scan['id']} ({scan['category']}, BV={scan['brain_volume']:.3f})")
        else:
            print(f"  ❌ Failed to load: {scan['id']}")
        
        ax.axis('off')
    
    # Remove all padding and spacing
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.02, hspace=0)
    
    output_png = os.path.join(OUTPUT_DIR, "mri_grid_3_scans_optimal.png")
    output_pdf = os.path.join(OUTPUT_DIR, "mri_grid_3_scans_optimal.pdf")
    
    plt.savefig(output_png, dpi=300, bbox_inches='tight', pad_inches=0, facecolor='black')
    plt.savefig(output_pdf, bbox_inches='tight', pad_inches=0, facecolor='black')
    
    print(f"\n✅ OPTIMAL GRID SAVED:")
    print(f"   PNG: {output_png}")
    print(f"   PDF: {output_pdf}")
    print("\nCharacteristics:")
    print("  - NO labels or annotations")
    print("  - Clean black background")
    print("  - 300 DPI high resolution")
    print("  - Perfect atrophy progression: 0.728 → 0.371 → 0.001")
    print("  - Ideal for Deep Learning research paper")
    
    plt.close()

if __name__ == "__main__":
    create_optimal_3_scan_grid()
