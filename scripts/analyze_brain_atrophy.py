"""
Computational analysis of brain scans to verify atrophy in AD vs CN scans.
Measures brain volume, CSF volume, and ventricular size.
"""

import os
import glob
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage

def analyze_brain_volume(nii_path):
    """Compute brain tissue and CSF volumes from MRI scan."""
    try:
        img = nib.load(nii_path)
        data = img.get_fdata()
        
        # Normalize intensity
        data_normalized = (data - data.min()) / (data.max() - data.min() + 1e-8)
        
        # Brain tissue threshold (gray and white matter are bright)
        brain_threshold = 0.3
        brain_mask = data_normalized > brain_threshold
        
        # CSF threshold (dark areas within skull)
        csf_threshold = 0.15
        csf_mask = (data_normalized > 0.05) & (data_normalized < csf_threshold)
        
        # Calculate volumes (in voxels)
        brain_volume = np.sum(brain_mask)
        csf_volume = np.sum(csf_mask)
        total_volume = brain_volume + csf_volume
        
        # Brain fraction (lower in AD due to atrophy)
        brain_fraction = brain_volume / (total_volume + 1e-8)
        
        # Ventricular estimation (central CSF)
        # Find center of image
        center_z = data.shape[2] // 2
        center_slices = data[:, :, center_z-10:center_z+10]
        center_norm = (center_slices - center_slices.min()) / (center_slices.max() - center_slices.min() + 1e-8)
        
        # Ventricles are dark areas in center
        ventricle_mask = center_norm < 0.1
        
        # Find connected components in center
        labeled, num_features = ndimage.label(ventricle_mask)
        
        # Get center coordinates
        center_y, center_x = data.shape[0] // 2, data.shape[1] // 2
        
        # Find largest dark region near center
        max_ventricle_size = 0
        for i in range(1, num_features + 1):
            component = (labeled == i)
            # Check if near center
            ys, xs, _ = np.where(component)
            if len(ys) > 0:
                center_dist = np.sqrt((ys.mean() - center_y)**2 + (xs.mean() - center_x)**2)
                if center_dist < 100:  # Within central region
                    max_ventricle_size = max(max_ventricle_size, np.sum(component))
        
        ventricular_volume = max_ventricle_size
        
        return {
            'brain_volume': brain_volume,
            'csf_volume': csf_volume,
            'brain_fraction': brain_fraction,
            'ventricular_volume': ventricular_volume,
            'atrophy_score': csf_volume / (brain_volume + 1e-8)  # Higher = more atrophy
        }
    except Exception as e:
        print(f"Error analyzing {nii_path}: {e}")
        return None

def analyze_current_scans():
    """Analyze the scans currently in the figures."""
    
    # Scans from V2 figure
    scans = {
        'CN': ['051_S_1123', '136_S_0184', '036_S_0813'],
        'MCI': ['127_S_1140', '141_S_1245', '133_S_0912'],
        'AD': ['094_S_0347', '135_S_0098', '136_S_0429']
    }
    
    ADNI_SOURCE_DIR = r"C:\Users\gener\Downloads\ADNI1_Complete 1Yr 1.5T\ADNI"
    
    results = []
    
    print("=" * 80)
    print("COMPUTATIONAL ANALYSIS OF BRAIN SCANS")
    print("=" * 80)
    print("\nAnalyzing scans from the figure...")
    print("Measuring: Brain Volume | CSF Volume | Brain Fraction | Atrophy Score")
    print("-" * 80)
    
    for category, subjects in scans.items():
        print(f"\n{category} GROUP:")
        
        for subj in subjects:
            # Find the scan
            subj_dir = os.path.join(ADNI_SOURCE_DIR, subj)
            
            if os.path.exists(subj_dir):
                nii_files = glob.glob(os.path.join(subj_dir, "**", "*.nii"), recursive=True)
                
                if nii_files:
                    analysis = analyze_brain_volume(nii_files[0])
                    
                    if analysis:
                        results.append({
                            'subject': subj,
                            'category': category,
                            **analysis
                        })
                        
                        print(f"  {subj}:")
                        print(f"    Brain Fraction: {analysis['brain_fraction']:.3f} (higher = less atrophy)")
                        print(f"    Atrophy Score:  {analysis['atrophy_score']:.3f} (higher = more atrophy)")
                        print(f"    Ventricle Vol:  {analysis['ventricular_volume']:,} voxels")
                else:
                    print(f"  {subj}: No .nii file found")
            else:
                print(f"  {subj}: Directory not found")
    
    # Summary statistics
    if results:
        df = pd.DataFrame(results)
        
        print("\n" + "=" * 80)
        print("SUMMARY STATISTICS BY GROUP")
        print("=" * 80)
        
        summary = df.groupby('category').agg({
            'brain_fraction': ['mean', 'std'],
            'atrophy_score': ['mean', 'std'],
            'ventricular_volume': ['mean', 'std']
        })
        
        print("\nBrain Fraction (higher = healthier, expect: CN > MCI > AD):")
        for cat in ['CN', 'MCI', 'AD']:
            if cat in summary.index:
                mean = summary.loc[cat, ('brain_fraction', 'mean')]
                std = summary.loc[cat, ('brain_fraction', 'std')]
                print(f"  {cat:4s}: {mean:.3f} ± {std:.3f}")
        
        print("\nAtrophy Score (higher = more atrophy, expect: AD > MCI > CN):")
        for cat in ['CN', 'MCI', 'AD']:
            if cat in summary.index:
                mean = summary.loc[cat, ('atrophy_score', 'mean')]
                std = summary.loc[cat, ('atrophy_score', 'std')]
                print(f"  {cat:4s}: {mean:.3f} ± {std:.3f}")
        
        print("\nVentricular Volume (higher = more atrophy, expect: AD > MCI > CN):")
        for cat in ['CN', 'MCI', 'AD']:
            if cat in summary.index:
                mean = summary.loc[cat, ('ventricular_volume', 'mean')]
                std = summary.loc[cat, ('ventricular_volume', 'std')]
                print(f"  {cat:4s}: {mean:,.0f} ± {std:,.0f} voxels")
        
        # Check for misclassifications
        print("\n" + "=" * 80)
        print("VERIFICATION: Are AD scans showing atrophy?")
        print("=" * 80)
        
        cn_mean_brain_frac = df[df['category'] == 'CN']['brain_fraction'].mean()
        
        print("\nChecking each AD scan against CN average...")
        ad_scans = df[df['category'] == 'AD']
        
        for _, scan in ad_scans.iterrows():
            brain_frac = scan['brain_fraction']
            atrophy_score = scan['atrophy_score']
            
            if brain_frac > cn_mean_brain_frac:
                status = "⚠️  SUSPICIOUS - Brain fraction HIGHER than CN average!"
            elif brain_frac > cn_mean_brain_frac * 0.95:
                status = "⚠️  BORDERLINE - Only slightly lower than CN average"
            else:
                status = "✅ VERIFIED - Clear atrophy compared to CN"
            
            print(f"  {scan['subject']}: Brain Frac={brain_frac:.3f} - {status}")
        
        # Save results
        df.to_csv(r"d:\discs\figures\scan_analysis_results.csv", index=False)
        print(f"\n📊 Detailed results saved to: d:\\discs\\figures\\scan_analysis_results.csv")

if __name__ == "__main__":
    analyze_current_scans()
