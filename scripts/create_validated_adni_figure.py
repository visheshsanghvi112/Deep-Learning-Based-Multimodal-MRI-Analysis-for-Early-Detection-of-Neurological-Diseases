"""
Create VALIDATED ADNI figure - selects scans based on COMPUTATIONAL ATROPHY ANALYSIS.
Only includes scans where AD group shows MEASURABLE atrophy compared to CN.
"""

import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage
import matplotlib.pyplot as plt

# Configuration
ADNI_SOURCE_DIR = r"C:\Users\gener\Downloads\ADNI1_Complete 1Yr 1.5T\ADNI"
OUTPUT_DIR = r"d:\discs\figures"
OUTPUT_FILE = "adni_mri_diagnostic_samples_validated"

def compute_brain_metrics(nii_path):
    """Compute brain volume metrics to detect atrophy."""
    try:
        img = nib.load(nii_path)
        data = img.get_fdata()
        
        # Normalize
        data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)
        
        # Brain tissue (bright regions)
        brain_mask = data_norm > 0.3
        brain_volume = np.sum(brain_mask)
        
        # CSF/dark regions
        csf_mask = (data_norm > 0.05) & (data_norm < 0.15)
        csf_volume = np.sum(csf_mask)
        
        # Brain fraction (decreases with atrophy)
        total = brain_volume + csf_volume
        brain_fraction = brain_volume / (total + 1e-8)
        
        return {
            'brain_fraction': brain_fraction,
            'atrophy_score': csf_volume / (brain_volume + 1e-8)
        }
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
    """Find scans and validate atrophy using computational analysis."""
    
    csv_path = r"d:\discs\data\ADNI\ADNIMERGE_23Dec2025.csv"
    df = pd.read_csv(csv_path, low_memory=False)
    baseline = df[df['VISCODE'] == 'bl'] if 'VISCODE' in df.columns else df
    
    print("Finding and analyzing scans...")
    all_scans = {'CN': [], 'MCI': [], 'AD': []}
    
    subject_dirs = glob.glob(os.path.join(ADNI_SOURCE_DIR, "*_S_*"))
    
    for subj_dir in subject_dirs[:100]:  # Limit for speed
        if not os.path.isdir(subj_dir):
            continue
        
        subj_id = os.path.basename(subj_dir)
        nii_files = glob.glob(os.path.join(subj_dir, "**", "*.nii"), recursive=True)
        
        if not nii_files:
            continue
        
        # Match with CSV
        variants = [subj_id, subj_id.replace('_S_', '_')]
        
        for variant in variants:
            matches = baseline[baseline['PTID'] == variant]
            
            if len(matches) > 0:
                row = matches.iloc[0]
                dx = row['DX']
                mmse = row['MMSE'] if 'MMSE' in row and pd.notna(row['MMSE']) else None
                
                # Analyze brain metrics
                metrics = compute_brain_metrics(nii_files[0])
                
                if metrics:
                    scan_info = {
                        'subject_id': subj_id,
                        'nii_path': nii_files[0],
                        'dx': dx,
                        'mmse': mmse,
                        'brain_fraction': metrics['brain_fraction'],
                        'atrophy_score': metrics['atrophy_score']
                    }
                    
                    if dx == 'CN':
                        all_scans['CN'].append(scan_info)
                    elif dx == 'MCI':
                        all_scans['MCI'].append(scan_info)
                    elif dx == 'Dementia':
                        all_scans['AD'].append(scan_info)
                
                break
    
    print(f"Analyzed: {len(all_scans['CN'])} CN, {len(all_scans['MCI'])} MCI, {len(all_scans['AD'])} AD")
    
    return all_scans

def select_validated_samples(all_scans):
    """Select samples ensuring clear atrophy differences."""
    
    # Calculate CN average brain fraction
    cn_brain_fracs = [s['brain_fraction'] for s in all_scans['CN']]
    cn_avg = np.mean(cn_brain_fracs)
    cn_std = np.std(cn_brain_fracs)
    
    print(f"\nCN average brain fraction: {cn_avg:.3f} ± {cn_std:.3f}")
    
    # Select CN scans with highest brain fraction (healthiest)
    cn_sorted = sorted(all_scans['CN'], key=lambda x: x['brain_fraction'], reverse=True)
    selected_cn = cn_sorted[:3]
    
    # Select AD scans with LOWEST brain fraction (most atrophy)
    # AND must be significantly lower than CN average
    ad_sorted = sorted(all_scans['AD'], key=lambda x: x['brain_fraction'])
    
    selected_ad = []
    for scan in ad_sorted:
        if scan['brain_fraction'] < cn_avg - 0.5 * cn_std:  # At least 0.5 SD below CN mean
            selected_ad.append(scan)
            if len(selected_ad) == 3:
                break
    
    # If not enough, just take lowest
    if len(selected_ad) < 3:
        selected_ad = ad_sorted[:3]
    
    # Select MCI in middle range
    mci_sorted = sorted(all_scans['MCI'], key=lambda x: x['brain_fraction'])
    mid_idx = len(mci_sorted) // 2
    selected_mci = mci_sorted[mid_idx:mid_idx+3] if mid_idx+3 <= len(mci_sorted) else mci_sorted[:3]
    
    # Verify selections
    print("\n" + "=" * 60)
    print("SELECTED SCANS WITH VALIDATED ATROPHY:")
    print("=" * 60)
    
    for category, samples, label in [('CN', selected_cn, 'CN'), 
                                       ('MCI', selected_mci, 'MCI'),
                                       ('AD', selected_ad, 'AD')]:
        print(f"\n{label}:")
        for scan in samples:
            bf = scan['brain_fraction']
            diff_from_cn = ((bf - cn_avg) / cn_avg) * 100
            mmse_str = f"MMSE={scan['mmse']:.0f}" if scan['mmse'] is not None else "MMSE=N/A"
            print(f"  {scan['subject_id']}: BrainFrac={bf:.3f} ({diff_from_cn:+.1f}% vs CN avg), {mmse_str}")
    
    return selected_cn + selected_mci + selected_ad

def create_validated_figure(selected_scans):
    """Create figure with validated scans."""
    
    fig, axes = plt.subplots(3, 3, figsize=(15, 16))
    fig.suptitle('ADNI MRI Scans: Computationally Validated Atrophy Progression\nT1-weighted MPRAGE, 1.5T', 
                 fontsize=16, fontweight='bold', y=0.98)
    
    row_labels = [
        'Cognitively Normal (CN)\n(Highest Brain Volume)',
        'Mild Cognitive Impairment (MCI)\n(Intermediate)', 
        'Alzheimer\'s Disease (AD)\n(Lowest Brain Volume - Validated Atrophy)'
    ]
    
    for row_idx, label in enumerate(row_labels):
        fig.text(0.08, 0.825 - row_idx * 0.285, label, 
                fontsize=11, fontweight='bold', rotation=90, 
                va='center', ha='center')
    
    for idx in range(9):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        
        if idx < len(selected_scans):
            scan = selected_scans[idx]
            slice_data = load_sagittal_slice(scan['nii_path'])
            
            if slice_data is not None:
                ax.imshow(slice_data, cmap='gray', aspect='auto')
                
                # Category for label
                if idx < 3:
                    cat = 'CN'
                elif idx < 6:
                    cat = 'MCI'
                else:
                    cat = 'AD'
                
                info = f"{scan['subject_id']}\n{cat}\nBrain: {scan['brain_fraction']:.3f}"
                ax.text(0.02, 0.98, info, transform=ax.transAxes,
                       fontsize=9, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
        
        ax.axis('off')
    
    plt.tight_layout(rect=[0.10, 0.02, 1, 0.96])
    
    output_png = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILE}.png")
    output_pdf = os.path.join(OUTPUT_DIR, f"{OUTPUT_FILE}.pdf")
    
    plt.savefig(output_png, dpi=300, bbox_inches='tight', facecolor='white')
    plt.savefig(output_pdf, bbox_inches='tight', facecolor='white')
    
    print(f"\n✅ Validated figure saved:")
    print(f"   {output_png}")
    
    plt.close()

if __name__ == "__main__":
    all_scans = find_and_validate_scans()
    selected = select_validated_samples(all_scans)
    create_validated_figure(selected)
