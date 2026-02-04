"""
Create IMPROVED ADNI figure with SAGITTAL views to show clearer atrophy differences.
Select scans with more advanced dementia for visibility.
"""

import os
import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Configuration
ADNI_SOURCE_DIR = r"C:\Users\gener\Downloads\ADNI1_Complete 1Yr 1.5T\ADNI"
OUTPUT_DIR = r"d:\discs\figures"
OUTPUT_FILE = "adni_mri_diagnostic_samples_grid_v2"

def load_and_slice_mri_sagittal(nii_path):
    """Load scan and extract SAGITTAL (side view) slice showing hippocampus."""
    try:
        img = nib.load(nii_path)
        data = img.get_fdata()
        
        # Get sagittal slice (side view) - best for seeing hippocampus
        # Use slice slightly off midline to see hippocampus structure
        slice_idx = int(data.shape[0] * 0.45)  # Slightly left of midline
        sagittal_slice = data[slice_idx, :, :]
        
        # Rotate for proper orientation
        sagittal_slice = np.rot90(sagittal_slice, k=1)
        
        # Normalize
        sagittal_slice = np.clip(sagittal_slice, 0, np.percentile(sagittal_slice, 99))
        
        return sagittal_slice
    except:
        return None

def find_and_categorize_scans():
    """Find scans and categorize with CSV metadata."""
    csv_path = r"d:\discs\data\ADNI\ADNIMERGE_23Dec2025.csv"
    df = pd.read_csv(csv_path, low_memory=False)
    
    # Get baseline visits only
    baseline = df[df['VISCODE'] == 'bl'] if 'VISCODE' in df.columns else df
    
    # Find all subject directories
    subject_dirs = glob.glob(os.path.join(ADNI_SOURCE_DIR, "*_S_*"))
    
    print(f"Scanning {len(subject_dirs)} subjects...")
    
    categorized = {'CN': [], 'MCI': [], 'AD': []}
    
    for subj_dir in subject_dirs:
        if not os.path.isdir(subj_dir):
            continue
            
        subj_id = os.path.basename(subj_dir)
        
        # Find .nii files
        nii_files = glob.glob(os.path.join(subj_dir, "**", "*.nii"), recursive=True)
        if not nii_files:
            continue
        
        # Match with CSV
        subj_variants = [subj_id, subj_id.replace('_S_', '_')]
        
        for variant in subj_variants:
            matches = baseline[baseline['PTID'] == variant]
            
            if len(matches) > 0:
                row = matches.iloc[0]
                dx = row['DX']
                mmse = row['MMSE'] if 'MMSE' in row and pd.notna(row['MMSE']) else None
                age = row['AGE'] if 'AGE' in row else None
                
                # Categorize
                category = None
                if dx == 'CN':
                    category = 'CN'
                elif dx == 'MCI':
                    category = 'MCI'
                elif dx == 'Dementia':
                    category = 'AD'
                
                if category:
                    categorized[category].append({
                        'subject_id': subj_id,
                        'nii_path': nii_files[0],  # Use first scan
                        'dx': dx,
                        'mmse': mmse,
                        'age': age,
                        'diagnosis': category
                    })
                break
    
    print(f"Found: {len(categorized['CN'])} CN, {len(categorized['MCI'])} MCI, {len(categorized['AD'])} AD")
    return categorized

def select_representative_scans(categorized):
    """Select scans that show clear atrophy for AD."""
    selected = []
    
    # For AD, prefer subjects with lower MMSE (more severe dementia)
    ad_subjects = categorized['AD']
    ad_with_mmse = [s for s in ad_subjects if s['mmse'] is not None]
    ad_with_mmse.sort(key=lambda x: x['mmse'])  # Lower MMSE = more severe
    
    print("\nSelecting scans...")
    
    # Select 3 from each category
    for category, candidates in [('CN', categorized['CN']), 
                                   ('MCI', categorized['MCI']),
                                   ('AD', ad_with_mmse if ad_with_mmse else categorized['AD'])]:
        
        np.random.seed(42 if category == 'CN' else (43 if category == 'MCI' else 44))
        
        valid_count = 0
        tried_indices = set()
        
        while valid_count < 3 and len(tried_indices) < min(20, len(candidates)):
            idx = np.random.randint(0, len(candidates))
            
            if idx in tried_indices:
                continue
            tried_indices.add(idx)
            
            scan = candidates[idx]
            
            # Test load
            slice_data = load_and_slice_mri_sagittal(scan['nii_path'])
            if slice_data is not None:
                selected.append(scan)
                mmse_str = f"MMSE={scan['mmse']:.0f}" if scan['mmse'] is not None else "MMSE=N/A"
                print(f"✓ {category}: {scan['subject_id']} ({mmse_str})")
                valid_count += 1
    
    return selected

def create_improved_figure():
    """Create figure with sagittal views."""
    
    categorized = find_and_categorize_scans()
    selected = select_representative_scans(categorized)
    
    if len(selected) < 9:
        print(f"Warning: Only {len(selected)} scans found")
        return
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 16))
    fig.suptitle('ADNI MRI Scans: Sagittal View Showing Hippocampal Atrophy\nT1-weighted MPRAGE, 1.5T', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    row_labels = [
        'Cognitively Normal (CN)',
        'Mild Cognitive Impairment (MCI)',
        'Alzheimer\'s Disease (Dementia)'
    ]
    
    for row_idx, label in enumerate(row_labels):
        fig.text(0.08, 0.825 - row_idx * 0.285, label, 
                fontsize=13, fontweight='bold', rotation=90, 
                va='center', ha='center')
    
    # Plot
    for idx in range(9):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        
        if idx < len(selected):
            scan = selected[idx]
            slice_data = load_and_slice_mri_sagittal(scan['nii_path'])
            
            if slice_data is not None:
                ax.imshow(slice_data, cmap='gray', aspect='auto')
                
                mmse_str = f"MMSE: {scan['mmse']:.0f}" if scan['mmse'] is not None else ""
                info_text = f"{scan['subject_id']}\n{scan['diagnosis']}\n{mmse_str}"
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        ax.axis('off')
    
    plt.tight_layout(rect=[0.10, 0.02, 1, 0.96])
    
    output_png = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILE}.png")
    output_pdf = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILE}.pdf")
    
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    
    print(f"\n✅ Improved figure saved:")
    print(f"   {output_png}")
    print(f"   {output_pdf}")
    
    plt.close()

if __name__ == "__main__":
    create_improved_figure()
