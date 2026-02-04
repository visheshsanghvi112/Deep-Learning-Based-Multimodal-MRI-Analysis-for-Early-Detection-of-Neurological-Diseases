"""
Verify the diagnoses of subjects shown in the ADNI figure and check for visible atrophy.
"""
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import numpy as np

# Load the figure to extract subject IDs
img_path = r"d:\discs\figures\adni_mri_diagnostic_samples_grid.png"
img = Image.open(img_path)

# Subject IDs from the image (reading from the labels)
subjects_in_figure = {
    'CN': ['036_S_0672', '137_S_0841', '116_S_0811'],
    'MCI': ['002_S_0619', '002_S_0954', '027_S_0307'],
    'AD': ['126_S_0606', '007_S_0101', '094_S_0347']
}

# Load ADNI CSV
csv_path = r"d:\discs\data\ADNI\ADNIMERGE_23Dec2025.csv"
df = pd.read_csv(csv_path, low_memory=False)

print("=" * 80)
print("VERIFICATION OF SUBJECT DIAGNOSES")
print("=" * 80)

for category, subjects in subjects_in_figure.items():
    print(f"\n{category} Group:")
    print("-" * 40)
    
    for subj in subjects:
        # Try different formats
        subj_variants = [subj, subj.replace('_S_', '_')]
        
        found = False
        for variant in subj_variants:
            matches = df[df['PTID'] == variant]
            if len(matches) > 0:
                # Get baseline visit
                baseline = matches[matches['VISCODE'] == 'bl'] if 'VISCODE' in matches.columns else matches.iloc[[0]]
                
                if len(baseline) > 0:
                    dx = baseline.iloc[0]['DX']
                    age = baseline.iloc[0]['AGE'] if 'AGE' in baseline.columns else 'N/A'
                    mmse = baseline.iloc[0]['MMSE'] if 'MMSE' in baseline.columns else 'N/A'
                    
                    # Check if diagnosis matches category
                    expected_dx = category if category != 'AD' else 'Dementia'
                    match_status = "✅ CORRECT" if dx == expected_dx else f"❌ WRONG (actual: {dx})"
                    
                    print(f"  {subj}: {dx} {match_status}")
                    print(f"    Age: {age}, MMSE: {mmse}")
                    
                    found = True
                    break
        
        if not found:
            print(f"  {subj}: ❌ NOT FOUND IN CSV")

print("\n" + "=" * 80)
print("CHECKING FOR VISIBLE ATROPHY MARKERS")
print("=" * 80)
print("""
Key differences to look for in AD vs CN scans:

1. **Hippocampal Atrophy** (medial temporal lobe):
   - AD: Smaller, more shrunken hippocampus
   - CN: Larger, more prominent hippocampus

2. **Ventricular Enlargement** (dark central spaces):
   - AD: Larger ventricles (more black space in center)
   - CN: Smaller ventricles

3. **Cortical Thinning** (outer brain layer):
   - AD: Thinner cortex, more prominent sulci (grooves)
   - CN: Thicker cortex, less prominent sulci

4. **Overall Brain Volume**:
   - AD: More CSF (dark) spaces, less brain tissue
   - CN: More brain tissue, less dark space

⚠️ NOTE: These differences can be subtle in 2D slices and vary by:
   - Disease stage (early vs late)
   - Slice position (which part of brain shown)
   - Individual variation

For research papers, it's common to show scans that are LABELED correctly
even if visual differences are subtle, as the diagnosis is based on clinical
and comprehensive imaging criteria, not just one slice.
""")

print("\n" + "=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print("""
If the visual differences are too subtle:
1. We can select scans from more advanced AD cases (higher CDR scores)
2. We can add anatomical annotations pointing out key structures
3. We can use coronal views which show hippocampus better
4. We can create a composite showing multiple orientations per subject

Should I create an improved version with clearer examples?
""")
