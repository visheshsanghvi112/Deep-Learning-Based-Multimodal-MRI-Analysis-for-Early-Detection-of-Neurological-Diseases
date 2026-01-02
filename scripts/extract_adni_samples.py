"""
Extract sample MRI slices from ADNI NIfTI files for frontend display
"""
import os
from pathlib import Path
import numpy as np
import nibabel as nib
from PIL import Image
import imageio

def extract_middle_slices(nii_path: Path, num_slices: int = 20):
    """Extract middle sagittal slices from a NIfTI file."""
    img = nib.load(str(nii_path))
    data = img.get_fdata()
    
    # Get dimensions
    x, y, z = data.shape[:3]
    mid_x = x // 2
    
    # Extract slices around the middle
    start = mid_x - num_slices // 2
    slices = []
    for i in range(num_slices):
        slice_idx = start + i
        if 0 <= slice_idx < x:
            slice_2d = data[slice_idx, :, :]
            # Normalize to 0-255
            slice_2d = slice_2d - slice_2d.min()
            if slice_2d.max() > 0:
                slice_2d = (slice_2d / slice_2d.max() * 255).astype(np.uint8)
            else:
                slice_2d = slice_2d.astype(np.uint8)
            # Rotate for proper orientation
            slice_2d = np.rot90(slice_2d)
            slices.append(slice_2d)
    
    return slices

def create_gif(slices: list, output_path: Path, duration: float = 0.1):
    """Create an animated GIF from slices."""
    # Convert to PIL images
    images = [Image.fromarray(s) for s in slices]
    # Resize to consistent size
    target_size = (160, 160)
    images = [img.resize(target_size, Image.Resampling.LANCZOS) for img in images]
    
    # Save as GIF
    imageio.mimsave(str(output_path), images, duration=duration, loop=0)
    print(f"  Created: {output_path}")

def main():
    # ADNI data directory
    adni_dir = Path("D:/discs/ADNI")
    output_dir = Path("D:/discs/project/frontend/public/adni-samples")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find NIfTI files
    nii_files = list(adni_dir.rglob("*.nii"))
    print(f"Found {len(nii_files)} NIfTI files")
    
    if not nii_files:
        print("No NIfTI files found!")
        return
    
    # Sample subjects from each group (we'll need to read the CSV to know groups)
    import pandas as pd
    features_file = Path("D:/discs/project_adni/data/features/subject_features.csv")
    
    if features_file.exists():
        df = pd.read_csv(features_file)
        cn_subjects = df[df['Group'] == 'CN']['Subject'].tolist()[:3]
        mci_subjects = df[df['Group'] == 'MCI']['Subject'].tolist()[:3]
        ad_subjects = df[df['Group'] == 'AD']['Subject'].tolist()[:3]
        
        target_subjects = cn_subjects + mci_subjects + ad_subjects
        print(f"Target subjects: {target_subjects}")
    else:
        # Just take first 9 files
        target_subjects = None
    
    count = 0
    for nii_path in nii_files:
        if count >= 9:
            break
            
        # Extract subject ID from path
        subject_id = None
        for part in nii_path.parts:
            if '_S_' in part:
                subject_id = part
                break
        
        if target_subjects and subject_id not in target_subjects:
            continue
        
        try:
            print(f"Processing: {nii_path.name}")
            slices = extract_middle_slices(nii_path)
            
            if slices:
                # Determine group
                group = "unknown"
                if target_subjects:
                    if subject_id in cn_subjects:
                        group = "cn"
                    elif subject_id in mci_subjects:
                        group = "mci"
                    elif subject_id in ad_subjects:
                        group = "ad"
                
                output_file = output_dir / f"{group}_{count+1}.gif"
                create_gif(slices, output_file)
                count += 1
        except Exception as e:
            print(f"  Error: {e}")
    
    print(f"\nCreated {count} sample GIFs in {output_dir}")

if __name__ == "__main__":
    main()
