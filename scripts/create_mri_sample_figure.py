"""
Create a professional 3x3 grid figure showing MRI brain scans from different diagnostic categories.
The figure will contain:
- Top row: 3 Cognitively Normal (CN) scans
- Middle row: 3 Mild Cognitive Impairment (MCI) scans
- Bottom row: 3 Alzheimer's Disease (AD) scans
"""

import os
import pandas as pd
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
DATA_DIR = r"d:\discs\data"
FEATURES_CSV = os.path.join(DATA_DIR, "extracted_features", "adni_features.csv")
OUTPUT_DIR = r"d:\discs\figures"


OUTPUT_FILE = "mri_diagnostic_samples_grid"

def load_and_slice_mri(nii_path, slice_idx=None):
    """Load MRI scan and extract a representative sagittal slice."""
    try:
        # Load the NIfTI file
        img = nib.load(nii_path)
        data = img.get_fdata()
        
        # Get midline sagittal slice (shows hippocampus well)
        if slice_idx is None:
            # Use middle of first dimension for sagittal view
            slice_idx = int(data.shape[0] * 0.5)
        
        # Extract sagittal slice (YZ plane at X=slice_idx)  
        sagittal_slice = data[slice_idx, :, :]
        
        # Rotate 90 degrees counterclockwise for proper orientation
        sagittal_slice = np.rot90(sagittal_slice, k=1)
        
        # Normalize intensity for better visualization
        sagittal_slice = np.clip(sagittal_slice, 0, np.percentile(sagittal_slice, 99))
        
        return sagittal_slice
    except Exception as e:
        print(f"Error loading {nii_path}: {e}")
        return None

def create_sample_figure():
    """Create a 3x3 grid of MRI samples across diagnostic categories."""
    
    # Load the features CSV to get subject information
    df = pd.read_csv(FEATURES_CSV)
    
    # Select baseline scans only (sc = screening)
    baseline_df = df[df['Visit'] == 'sc'].copy()
    
    # Group by diagnosis
    cn_subjects = baseline_df[baseline_df['Group'] == 'CN']
    mci_subjects = baseline_df[baseline_df['Group'] == 'MCI']
    ad_subjects = baseline_df[baseline_df['Group'] == 'AD']
    
    print(f"Found {len(cn_subjects)} CN, {len(mci_subjects)} MCI, {len(ad_subjects)} AD baseline scans")
    
    # Helper function to get 3 valid samples
    def get_valid_samples(group_df, n_samples=3, max_attempts=15):
        """Try to get n valid samples (with existing files) from a group."""
        samples = []
        attempted = set()
        
        while len(samples) < n_samples and len(attempted) < min(max_attempts, len(group_df)):
            # Sample from remaining subjects
            remaining = group_df[~group_df.index.isin(attempted)]
            if len(remaining) == 0:
                break
                
            sample = remaining.sample(n=1).iloc[0]
            attempted.add(sample.name)
            
            # Check if file exists
            subject_dir = os.path.join(DATA_DIR, 'ADNI', sample['Subject'])
            if os.path.exists(subject_dir):
                # Quick check for .nii file
                has_nii = False
                for root, dirs, files in os.walk(subject_dir):
                    if any(f.endswith('.nii') and sample['ImageID'] in f for f in files):
                        has_nii = True
                        break
                
                if has_nii:
                    samples.append(sample)
        
        return pd.DataFrame(samples)
    
    # Get valid samples from each group
    np.random.seed(42)  # For reproducibility
    cn_sample = get_valid_samples(cn_subjects)
    mci_sample = get_valid_samples(mci_subjects)
    ad_sample = get_valid_samples(ad_subjects)
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 16))
    fig.suptitle('Representative MRI Brain Scans Across Diagnostic Categories\nMid-Sagittal View, T1-weighted MPRAGE, 1.5T', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Define row labels
    row_labels = [
        'Cognitively Normal (CN)',
        'Mild Cognitive Impairment (MCI)',
        'Alzheimer\'s Disease (AD)'
    ]
    
    samples = [cn_sample, mci_sample, ad_sample]
    
    for row_idx, (sample_df, label) in enumerate(zip(samples, row_labels)):
        # Add row label
        fig.text(0.08, 0.825 - row_idx * 0.285, label, 
                fontsize=13, fontweight='bold', rotation=90, 
                va='center', ha='center')
        
        for col_idx, (_, row) in enumerate(sample_df.iterrows()):
            ax = axes[row_idx, col_idx]
            
             # Extract subject ID and ImageID
            subject_id = row['Subject']
            image_id = row['ImageID']
            
            # Construct path in ADNI directory
            subject_dir = os.path.join(DATA_DIR, 'ADNI', subject_id)
            
            # Try to find the NII file
            slice_data = None
            if os.path.exists(subject_dir):
                # Search for .nii files in subdirectories
                for root, dirs, files in os.walk(subject_dir):
                    for file in files:
                        if file.endswith('.nii') and image_id in file:
                            nii_path = os.path.join(root, file)
                            slice_data = load_and_slice_mri(nii_path)
                            if slice_data is not None:
                                print(f"Loaded: {subject_id} ({row['Group']}) - {file}")
                                break
                    if slice_data is not None:
                        break
            
            if slice_data is not None:
                # Display the slice
                ax.imshow(slice_data, cmap='gray', aspect='auto')
                
                # Add subject info
                info_text = f"Subject: {row['Subject']}\nAge: {int(row['Age'])}, Sex: {row['Sex']}"
                ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            else:
                # Show placeholder if scan not found
                ax.text(0.5, 0.5, f"Scan not found\n{row['Subject']}", 
                       ha='center', va='center', fontsize=10)
                ax.set_facecolor('#f0f0f0')
            
            ax.axis('off')
    
    # Adjust layout
    plt.tight_layout(rect=[0.10, 0.02, 1, 0.96])
    
    # Save figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path_png = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILE}.png")
    output_path_pdf = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILE}.pdf")
    
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    
    print(f"\n✅ Figure saved successfully:")
    print(f"   PNG: {output_path_png}")
    print(f"   PDF: {output_path_pdf}")
    
    plt.show()

if __name__ == "__main__":
    create_sample_figure()
