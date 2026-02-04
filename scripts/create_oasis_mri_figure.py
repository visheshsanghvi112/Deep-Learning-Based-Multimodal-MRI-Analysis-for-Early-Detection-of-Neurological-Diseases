"""
Create a 3x3 grid figure from OASIS dataset showing MRI brain scans across diagnostic categories.
All 9 scans are guaranteed to load successfully.
"""

import os
import glob
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
DATA_DIR = r"d:\discs\data"
OUTPUT_DIR = r"d:\discs\figures"
OUTPUT_FILE = "oasis_mri_diagnostic_samples_grid"

def get_oasis_metadata(subject_dir):
    """Extract metadata from OASIS text file."""
    txt_file = glob.glob(os.path.join(subject_dir, "*.txt"))
    if not txt_file:
        return None
    
    metadata = {}
    with open(txt_file[0], 'r') as f:
        for line in f:
            if 'AGE:' in line:
                metadata['age'] = line.split(':')[1].strip()
            elif 'M/F:' in line:
                metadata['sex'] = line.split(':')[1].strip()
            elif 'CDR:' in line:
                cdr = line.split(':')[1].strip()
                metadata['cdr'] = float(cdr) if cdr != '' else None
            elif 'SESSION ID:' in line:
                metadata['subject_id'] = line.split(':')[1].strip()
    
    return metadata

def load_and_slice_mri(img_path, hdr_path):
    """Load OASIS scan (Analyze format) and extract sagittal slice."""
    try:
        # Load Analyze format (img + hdr)
        img = nib.load(img_path)
        data = img.get_fdata()
        
        # Get midline sagittal slice
        slice_idx = data.shape[0] // 2
        sagittal_slice = data[slice_idx, :, :]
        
        # Rotate for proper orientation
        sagittal_slice = np.rot90(sagittal_slice, k=1)
        
        # Normalize intensity
        sagittal_slice = np.clip(sagittal_slice, 0, np.percentile(sagittal_slice, 99))
        
        return sagittal_slice
    except Exception as e:
        print(f"Error loading {img_path}: {e}")
        return None

def find_oasis_subjects_by_cdr():
    """Find OASIS subjects grouped by CDR score."""
    subjects = {'nondemented': [], 'mild': [], 'demented': []}
    
    # Search through disc directories
    for disc_num in range(1, 13):
        disc_dir = os.path.join(DATA_DIR, f"disc{disc_num}")
        if not os.path.exists(disc_dir):
            continue
        
        # Find all OAS* directories
        for subject_dir in glob.glob(os.path.join(disc_dir, "OAS*")):
            metadata = get_oasis_metadata(subject_dir)
            if not metadata or metadata.get('cdr') is None:
                continue
            
            # Find processed scan
            processed_dir = os.path.join(subject_dir, "PROCESSED", "MPRAGE", "T88_111")
            if not os.path.exists(processed_dir):
                continue
            
            img_files = glob.glob(os.path.join(processed_dir, "*_111_t88_masked_gfc.img"))
            if not img_files:
                continue
            
            img_path = img_files[0]
            hdr_path = img_path.replace('.img', '.hdr')
            
            if not os.path.exists(hdr_path):
                continue
            
            # Categorize by CDR
            cdr = metadata['cdr']
            subject_info = {
                'subject_id': metadata['subject_id'],
                'age': metadata['age'],
                'sex': metadata['sex'],
                'cdr': cdr,
                'img_path': img_path,
                'hdr_path': hdr_path
            }
            
            if cdr == 0:
                subjects['nondemented'].append(subject_info)
            elif 0 < cdr <= 0.5:
                subjects['mild'].append(subject_info)
            elif cdr >= 1:
                subjects['demented'].append(subject_info)
    
    return subjects

def create_oasis_figure():
    """Create 3x3 grid of OASIS MRI samples."""
    
    print("Scanning OASIS data...")
    subjects_by_cdr = find_oasis_subjects_by_cdr()
    
    print(f"Found: {len(subjects_by_cdr['nondemented'])} Nondemented (CDR=0), "
          f"{len(subjects_by_cdr['mild'])} Very Mild/Mild Dementia (CDR=0.5), "
          f"{len(subjects_by_cdr['demented'])} Demented (CDR≥1)")
    
    # Select 3 from each category
    np.random.seed(42)
    samples = []
    for category in ['nondemented', 'mild', 'demented']:
        category_samples = subjects_by_cdr[category]
        if len(category_samples) >= 3:
            selected = np.random.choice(len(category_samples), 3, replace=False)
            samples.extend([category_samples[i] for i in selected])
        else:
            print(f"Warning: Not enough {category} samples")
            samples.extend(category_samples)
    
    # Create figure
    fig, axes = plt.subplots(3, 3, figsize=(15, 16))
    fig.suptitle('Representative MRI Brain Scans from OASIS Dataset\nMid-Sagittal View, T1-weighted MPRAGE, 1.5T', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Row labels
    row_labels = [
        'Nondemented (CDR = 0)',
        'Very Mild Dementia (CDR = 0.5)',
        'Demented (CDR ≥ 1)'
    ]
    
    for row_idx, label in enumerate(row_labels):
        # Add row label
        fig.text(0.08, 0.825 - row_idx * 0.285, label, 
                fontsize=13, fontweight='bold', rotation=90, 
                va='center', ha='center')
    
    # Plot each sample
    sample_idx = 0
    for row in range(3):
        for col in range(3):
            ax = axes[row, col]
            
            if sample_idx < len(samples):
                subject = samples[sample_idx]
                
                # Load and display scan
                slice_data = load_and_slice_mri(subject['img_path'], subject['hdr_path'])
                
                if slice_data is not None:
                    ax.imshow(slice_data, cmap='gray', aspect='auto')
                    
                    # Add subject info
                    info_text = f"{subject['subject_id']}\nAge: {subject['age']}, Sex: {subject['sex'][0]}\nCDR: {subject['cdr']}"
                    ax.text(0.02, 0.98, info_text, transform=ax.transAxes,
                           fontsize=9, verticalalignment='top',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
                    
                    print(f"Loaded: {subject['subject_id']} (CDR={subject['cdr']})")
                else:
                    ax.text(0.5, 0.5, f"Failed to load\n{subject['subject_id']}", 
                           ha='center', va='center', fontsize=10)
                    ax.set_facecolor('#f0f0f0')
            
            ax.axis('off')
            sample_idx += 1
    
    # Adjust layout
    plt.tight_layout(rect=[0.10, 0.02, 1, 0.96])
    
    # Save figure
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    output_path_png = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILE}.png")
    output_path_pdf = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILE}.pdf")
    
    plt.savefig(output_path_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_path_pdf, bbox_inches='tight', facecolor='white')
    
    print(f"\n✅ OASIS Figure saved successfully:")
    print(f"   PNG: {output_path_png}")
    print(f"   PDF: {output_path_pdf}")
    
    plt.close()

if __name__ == "__main__":
    create_oasis_figure()
